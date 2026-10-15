"""Fit Mont Terri material, neural storage and matrix evolution on training specimens."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import csv
import json
from itertools import repeat
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute/src"))

from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.neural_storage import MonotoneStorageNetwork
from tfenn_invcalib.neural_evolution import MatrixEvolutionNetwork
from tfenn_invcalib.winhausen_data import load_specimens


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def elastic_start(parameters, network, channels, training_specimens=None, storage_form="neural"):
    """Fit the recorded 0--0.08% loading interval for an initial material.

    This is an initialization only. The full-path fit can adjust all of these
    constants; early inelasticity is not permanently excluded from the model.
    """
    paths = []
    for p in load_specimens("train", training_specimens=training_specimens):
        h = p["history"]
        control = wt.recording_control(h["axial_strain"])
        index = np.flatnonzero((control > 0) & (control <= .0008))
        index = index[np.linspace(0, len(index)-1, min(20, len(index))).astype(int)]
        paths.append((p, index, control))

    def objective(coordinates):
        p = wt.decode(coordinates, parameters, wt.ELASTIC_NAMES)
        residual = []
        for specimen, index, control in paths:
            model = wt.material(p, specimen, network, storage_form=storage_form)
            m = model.energy
            stiffness = m.C0+m.M*np.outer(m.biot, m.biot)
            strain = np.r_[1., np.linalg.solve(stiffness[1:, 1:], -stiffness[1:, 0])]
            slope_q, slope_u = (stiffness@strain)[0], m.M*(m.biot@strain)
            h = specimen["history"]
            for name in channels:
                slope = slope_q if name == "q_MPa" else slope_u
                values = h[name][index]-h[name][0]
                residual.extend((slope*control[index]-values)/np.sqrt(len(index)))
        return residual

    bounds = np.array(wt.coordinate_bounds(wt.ELASTIC_NAMES)).T
    result = least_squares(objective, wt.encode(parameters, wt.ELASTIC_NAMES), bounds=bounds)
    fitted = wt.decode(result.x, parameters, wt.ELASTIC_NAMES)
    return fitted, dict(status=result.message, cost=float(result.cost),
                        role="training-only small-strain initialization",
                        material_parameters=fitted.copy())


def solve_training_path(*arguments, storage_form="neural", fabric_source="plastic_stress"):
    """Keep a constitutive trial failure attached to its candidate in a batch."""
    try:
        return wt.simulate(*arguments, storage_form=storage_form, fabric_source=fabric_source)
    except (wt.ConstitutiveSolveError, wt.PrestressInitializationError) as exc:
        return exc


class TrainingLoss:
    def __init__(self, record, output, mapper):
        self.record, self.output, self.mapper = record, output, mapper
        self.paths = wt.training_paths(record["samples"], record.get("training_specimens"))
        self.evaluations, self.best_loss = 0, np.inf
        self.names = tuple(record["active_material_parameters"])
        self.storage_form = record.get("storage_form", "neural")
        self.train_network = record["train_network"] and self.storage_form == "neural"
        self.train_evolution = record.get("train_evolution", False)
        self.evolution_inputs = record.get("evolution_network", {}).get("input_size", 2)
        self.fabric_source = record.get("evolution_network", {}).get("fabric_source", "plastic_stress")
        self.evolution_local_width = record.get("evolution_network", {}).get("local_width", 0)
        self.evolution_fabric_knots = record.get("evolution_network", {}).get("fabric_knots", ())
        self.resistance_uses_pressure = record.get("evolution_network", {}).get("resistance_uses_pressure", True)
        self.evolution_indices = record.get("active_evolution_parameters",
            list(range(len(record.get("evolution_network", {}).get("raw_parameters", [])))))
        self.initial_loss = None
        self.unsolved_evaluations = 0
        self.last_unsolved_candidate = None
        self.residual_size = None
        self.last_coordinates = None
        self.last_residual = None

    def optimizer_state(self):
        return dict(status="running", evaluations=self.evaluations,
                    initial_loss=self.initial_loss,
                    unsolved_evaluations=self.unsolved_evaluations,
                    last_unsolved_candidate=self.last_unsolved_candidate)

    def unpack(self, coordinates):
        n = len(self.names)
        parameters = wt.decode(coordinates[:n], self.record["material_parameters"], self.names)
        size = len(self.record["network"]["raw_parameters"]) if self.train_network else 0
        network = (coordinates[n:n+size] if self.train_network
                   else np.array(self.record["network"]["raw_parameters"]))
        evolution = self.record.get("evolution_network")
        evolution = np.array(evolution["raw_parameters"]) if evolution else None
        if self.train_evolution:
            evolution[self.evolution_indices] = coordinates[n+size:]
        return parameters, network, evolution

    def __call__(self, coordinates):
        residual = self.residual_vector(coordinates)
        return float(residual @ residual)

    def residual_vector(self, coordinates):
        """Return observed-history residuals, retaining accepted complete paths.

        An unresolved trial has infinite residuals. The reflective
        least-squares solver then shrinks its trust region, just as COBYQA
        rejects the infinite scalar objective. Neither case supplies a path.
        """
        if self.last_coordinates is None or not np.array_equal(coordinates, self.last_coordinates):
            residual = self.residual_vectors([coordinates])[0]
            if not np.all(np.isfinite(residual)):
                self.last_coordinates = self.last_residual = None
                return residual
            self.last_coordinates = np.array(coordinates, copy=True)
            self.last_residual = residual.copy()
        return self.last_residual.copy()

    def residual_vectors(self, coordinates):
        """Solve independent candidates together, then save them in input order.

        Only path solves run in the process pool. Checkpoint writes and the
        accepted candidate remain owned by this process.
        """
        candidates = [self.unpack(x) for x in coordinates]
        tasks = [(p, network, path, self.record["max_increment"], evolution, self.evolution_inputs,
                  self.resistance_uses_pressure, self.record.get("max_log_density_increment", 0.),
                  self.record.get("min_increment", 1e-9), self.evolution_local_width,
                  self.evolution_fabric_knots)
                 for p, network, evolution in candidates for path in self.paths]
        outcomes = list(self.mapper(partial(solve_training_path, storage_form=self.storage_form,
                                            fabric_source=self.fabric_source),
                                    *zip(*tasks)))
        n = len(self.paths)
        return [self.record_candidate(*candidate, outcomes[i*n:(i+1)*n])
                for i, candidate in enumerate(candidates)]

    def record_candidate(self, parameters, network, evolution, results):
        self.evaluations += 1
        exc = next((r for r in results if isinstance(r, Exception)), None)
        if exc is not None:
            # Optimization may propose a material that cannot be integrated.
            # Expose that candidate to the optimizer's infeasible-trial handling;
            # never supply fabricated paths or replace the last accepted fit.
            # An unresolved starting material still stops the experiment.
            if self.initial_loss is None:
                raise exc
            self.unsolved_evaluations += 1
            self.last_unsolved_candidate = dict(evaluation=self.evaluations,
                message=str(exc), material_parameters=parameters,
                raw_network_parameters=np.asarray(network).tolist(),
                raw_evolution_parameters=evolution.tolist() if evolution is not None else None)
            self.record["optimizer"] = self.optimizer_state()
            save_json(self.output / "checkpoint.json", self.record)
            print(f"evaluation={self.evaluations} unsolved_candidate: {exc}", flush=True)
            return np.full(self.residual_size, np.inf)
        residual = np.concatenate([wt.residuals(path, result, self.record["channels"],
                                               self.record.get("increment_weight", 0.),
                                               self.record.get("response_measure", "point"),
                                               self.record.get("volume_stage_weight", 0.),
                                               self.record.get("stage_increment_weight", 0.))
                                   for path, result in zip(self.paths, results)])
        self.residual_size = len(residual)
        loss = float(residual@residual)
        if self.initial_loss is None:
            self.initial_loss = loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.record.update(material_parameters=parameters,
                network=dict(raw_parameters=network.tolist(), hidden_width=len(network)//3),
                loss=loss, metrics={p["specimen_id"]: wt.path_metrics(p, r)
                                    for p, r in zip(self.paths, results)},
                optimizer=self.optimizer_state())
            if evolution is not None:
                self.record["evolution_network"]["raw_parameters"] = evolution.tolist()
            save_json(self.output / "checkpoint.json", self.record)
            save_paths(self.output / "train_paths.csv", results)
            print(f"evaluation={self.evaluations} loss={loss:.7g}", flush=True)
        elif self.evaluations % 25 == 0:
            print(f"evaluation={self.evaluations} best_loss={self.best_loss:.7g}", flush=True)
        return residual


def history_jacobian(objective, coordinates, bounds, step=1e-4, batch_size=1,
                     linearization_path=None):
    """Differentiate complete histories in material/network coordinates.

    A fixed coordinate increment avoids differencing the implicit solves at
    machine-scale parameter perturbations. At a search bound use the inward
    side. If one side cannot be integrated, use the other feasible side;
    if neither side supplies a history, expose the missing derivative.
    """
    base = objective.residual_vector(coordinates)
    columns = []
    box = [(lo if lo is not None else -np.inf,
            hi if hi is not None else np.inf) for lo, hi in bounds]
    if linearization_path is not None and linearization_path.exists():
        with np.load(linearization_path) as saved:
            if ("difference_step" in saved and saved["difference_step"] == step
                    and np.array_equal(saved["coordinates"], coordinates)
                    and np.array_equal(saved["bounds"], box)
                    and np.array_equal(saved["residual"], base)):
                return saved["jacobian"].copy()
    for start in range(0, len(bounds), batch_size):
        trials, directions = [], []
        for j in range(start, min(start+batch_size, len(bounds))):
            lower, upper = box[j]
            delta = step if coordinates[j]+step <= upper else -step
            if not lower <= coordinates[j]+delta <= upper:
                raise wt.ConstitutiveSolveError(
                    f"No resolved history derivative for training coordinate {j}.")
            trial = coordinates.copy()
            trial[j] += delta
            trials.append(trial)
            directions.append(delta)
        residuals = (objective.residual_vectors(trials) if batch_size > 1 else
                     [objective.residual_vector(trials[0])])
        for j, delta, residual in zip(range(start, start+len(trials)), directions, residuals):
            if not np.all(np.isfinite(residual)):
                lower, upper = box[j]
                delta = -delta
                if lower <= coordinates[j]+delta <= upper:
                    trial = coordinates.copy()
                    trial[j] += delta
                    residual = objective.residual_vector(trial)
                if not np.all(np.isfinite(residual)):
                    raise wt.ConstitutiveSolveError(
                        f"No resolved history derivative for training coordinate {j}.")
            columns.append((residual-base)/delta)
    jacobian = np.column_stack(columns)
    if linearization_path is not None:
        temporary = linearization_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, coordinates=coordinates, residual=base,
                                jacobian=jacobian, bounds=np.asarray(box), difference_step=step)
        temporary.replace(linearization_path)
    return jacobian


def fit_history_least_squares(objective, start, bounds, max_evaluations, batch_size=1,
                             trust_solver="exact", linearization_path=None,
                             coordinate_scaling="jac", coordinate_origin="absolute",
                             difference_step=1e-4):
    """Use residual directions to resolve stage-dependent training errors.

    Bounds and finite differences remain in the existing material and network
    coordinates. Unit scaling avoids expanding nearly insensitive neural
    coordinates by their inverse Jacobian norms. The
    evaluation limit counts trust-region trials; the checkpoint also records
    all history calls used for their Jacobians.
    """
    box = np.array([(lo if lo is not None else -np.inf,
                     hi if hi is not None else np.inf) for lo, hi in bounds]).T
    # TRF initializes its radius from the absolute starting coordinates and
    # their distances to active bounds. An all-zero translated start can be
    # nudged inward by a tiny amount, giving TRF a tiny nonzero initial radius.
    # Place one free coordinate at one so the checkpoint translation retains
    # a finite radius without changing any physical parameter or bound.
    origin = start.copy() if coordinate_origin == "checkpoint" else np.zeros_like(start)
    if coordinate_origin == "checkpoint":
        free = np.flatnonzero(np.isneginf(box[0]) & np.isposinf(box[1]))
        anchor = free[0] if len(free) else np.argmax(np.minimum(start-box[0], box[1]-start))
        origin[anchor] -= 1.
    result = least_squares(lambda x: objective.residual_vector(x+origin), start-origin,
                         jac=lambda x: history_jacobian(objective, x+origin, bounds,
                                                       step=difference_step,
                                                       batch_size=batch_size,
                                                       linearization_path=linearization_path),
                         bounds=box-origin, x_scale=1. if coordinate_scaling == "unit" else "jac",
                         max_nfev=max_evaluations,
                         tr_solver=trust_solver)
    result.x += origin
    return result


def save_paths(path, results):
    rows = [row for result in results for row in result["rows"]]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "train", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/neural-constitutive/winhausen-joint")
    parser.add_argument("--parameters", type=Path,
                        help="Evaluate a supplied material/network input without creating a training checkpoint.")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-increment", type=float)
    parser.add_argument("--max-log-density-increment", type=float,
                        help="Limit log-density growth and relative matrix energy-norm strain change; zero retains fixed axial substeps.")
    parser.add_argument("--min-increment", type=float,
                        help="Stop bisecting a finite damage jump at or below this axial interval.")
    parser.add_argument("--workers", type=int, default=7,
                        help="Path processes; extra workers also evaluate independent Jacobian columns.")
    parser.add_argument("--channels", choices=("joint", "stress"), default="joint")
    parser.add_argument("--increment-weight", type=float,
                        help="Initialize the weight of squared adjacent observed-response increment errors; inherited on warm starts, otherwise zero.")
    parser.add_argument("--volume-stage-weight", type=float,
                        help="Initialize the weight of coupled volume-stage errors for joint cell fitting; inherited on warm starts, otherwise zero.")
    parser.add_argument("--stage-increment-weight", type=float,
                        help="Initialize the weight of response increments over common stress-defined loading stages; inherited on warm starts, otherwise zero.")
    parser.add_argument("--response-measure", choices=("point", "cell"),
                        help="Initialize point errors or integrated observation-cell curve errors; inherited on warm starts.")
    parser.add_argument("--role", choices=("train", "test"), default="train")
    parser.add_argument("--training-specimens", nargs="+",
                        help="Initialize an experiment with these complete histories; otherwise inherit the warm-start split or use the source split.")
    parser.add_argument("--initial-network", action="store_true",
                        help="Evaluate initial storage and evolution weights at the saved fitted material.")
    parser.add_argument("--initial-evolution", action="store_true",
                        help="Reset only the matrix network for a learning diagnostic.")
    parser.add_argument("--matrix-evolution", action="store_true",
                        help="Initialize a new experiment with the pressure/history matrix network.")
    parser.add_argument("--storage-form", choices=("neural", "identity"),
                        help="Initialize plane storage; inherit a warm start's form, otherwise use neural storage.")
    parser.add_argument("--fabric-evolution", action="store_true",
                        help="Add the normalized objective bedding projection; new input weights start at zero.")
    parser.add_argument("--fabric-source", choices=("plastic_stress", "matrix_force"),
                        help="Initialize the third input from effective stress or matrix force after energetic backstress subtraction.")
    parser.add_argument("--separable-features", type=int, default=0,
                        help="Convert this many appended fabric features to a separable local map.")
    parser.add_argument("--interpolate-fabric", action="store_true",
                        help="Connect the matrix maps at training fabric projections and F=0 with C1 convex interpolation.")
    parser.add_argument("--gate-center", type=float, default=.8125)
    parser.add_argument("--gate-slope", type=float, default=24.)
    parser.add_argument("--resistance-without-pressure", action="store_true",
                        help="Use density and optional fabric for resistance; retain pressure in the flow law.")
    parser.add_argument("--initialize-from", type=Path,
                        help="Retain material and learned networks; stress-only training may also initialize a joint experiment.")
    parser.add_argument("--reset-storage", action="store_true",
                        help="Warm-start a new experiment with canonical storage weights, retaining material and matrix evolution.")
    parser.add_argument("--set-material", nargs=2, action="append", default=[],
                        metavar=("NAME", "VALUE"),
                        help="Set a named material value at initialization; it remains trainable.")
    parser.add_argument("--set-evolution-weight", nargs=2, action="append", default=[],
                        metavar=("INDEX", "VALUE"),
                        help="Set a zero-based matrix-network weight at initialization; it remains trainable.")
    parser.add_argument("--material", nargs="+", choices=wt.NAMES)
    parser.add_argument("--block", choices=("joint", "material", "network", "storage", "evolution", "fabric", "flow", "response"), default="joint",
                        help="Coordinates to optimize in this continuation stage.")
    parser.add_argument("--initial-radius", type=float, default=.1,
                        help="Initial COBYQA trust-region radius.")
    parser.add_argument("--final-radius", type=float, default=.005,
                        help="Final COBYQA trust-region radius.")
    parser.add_argument("--optimizer", choices=("cobyqa", "least-squares"), default="cobyqa",
                        help="Scalar interpolation or complete-history residual Jacobians.")
    parser.add_argument("--trust-solver", choices=("exact", "lsmr"), default="exact",
                        help="Least-squares trust-region solve; lsmr regularizes poorly determined directions.")
    parser.add_argument("--coordinate-scaling", choices=("jac", "unit"), default="jac",
                        help="Unit scaling keeps nearly insensitive raw neural coordinates from taking huge steps.")
    parser.add_argument("--coordinate-origin", choices=("absolute", "checkpoint"), default="absolute",
                        help="A checkpoint origin initializes the trust region in parameter increments.")
    parser.add_argument("--difference-step", type=float, default=1e-4,
                        help="Complete-history finite-difference increment in parameter coordinates.")
    parser.add_argument("--max-evaluations", type=int,
                        help="Resumable batch: COBYQA calls or least-squares trials (excluding Jacobian calls).")
    args = parser.parse_args()
    if not np.isfinite(args.difference_step) or args.difference_step <= 0:
        parser.error("The finite-difference increment must be positive and finite.")
    if args.increment_weight is not None:
        if args.action != "initialize":
            parser.error("Changing the objective requires initialize with a separate --output.")
        if not np.isfinite(args.increment_weight) or args.increment_weight < 0:
            parser.error("The response increment weight must be nonnegative and finite.")
    if args.volume_stage_weight is not None:
        if args.action != "initialize":
            parser.error("Changing the objective requires initialize with a separate --output.")
        if not np.isfinite(args.volume_stage_weight) or args.volume_stage_weight < 0:
            parser.error("The volume-stage weight must be nonnegative and finite.")
    if args.stage_increment_weight is not None:
        if args.action != "initialize":
            parser.error("Changing the objective requires initialize with a separate --output.")
        if not np.isfinite(args.stage_increment_weight) or args.stage_increment_weight < 0:
            parser.error("The stage-increment weight must be nonnegative and finite.")
    if args.response_measure is not None and args.action != "initialize":
        parser.error("Changing the response measure requires initialize with a separate --output.")
    if args.training_specimens is not None and args.action != "initialize":
        parser.error("Changing training specimens requires initialize with a separate --output.")
    if args.parameters and args.action != "evaluate":
        parser.error("--parameters applies only to evaluate; use --initialize-from for training.")
    if (args.initial_network or args.initial_evolution) and args.action != "evaluate":
        parser.error("Initial-weight diagnostics require evaluate.")
    if (args.initialize_from or args.reset_storage or args.matrix_evolution or args.fabric_evolution or args.fabric_source
            or args.separable_features or args.interpolate_fabric
            or args.resistance_without_pressure
            or args.set_material
            or args.set_evolution_weight or args.storage_form) and args.action != "initialize":
        parser.error("Architecture and warm-start options apply only to initialize.")
    if args.reset_storage and not args.initialize_from:
        parser.error("Resetting storage requires --initialize-from.")
    overrides = {}
    for name, value in args.set_material:
        if name not in wt.PARAMETERS:
            parser.error(f"Unknown material parameter: {name}.")
        try:
            value = float(value)
        except ValueError:
            parser.error(f"Material value must be numeric: {name}={value}.")
        lower, upper = wt.PARAMETERS[name][1:3]
        if not lower <= value <= upper:
            parser.error(f"Initial {name} must lie in [{lower}, {upper}].")
        overrides[name] = value
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.json"
    channels = ["q_MPa"] if args.channels == "stress" else ["q_MPa", "pressure_MPa"]
    source_evolution = None
    if args.parameters or checkpoint.exists():
        parameter_source = args.parameters or checkpoint
        record = json.loads(parameter_source.read_text(encoding="utf-8"))
        if args.action == "initialize":
            parser.error("An initialized experiment already exists; choose a new --output.")
        if channels != record["channels"]:
            parser.error("Channel changes require a separate experiment.")
    else:
        if args.action != "initialize":
            parser.error("Initialize this experiment first.")
        if args.initialize_from:
            source = json.loads(args.initialize_from.read_text(encoding="utf-8"))
            add_pressure = source["channels"] == ["q_MPa"] and channels == ["q_MPa", "pressure_MPa"]
            if source["channels"] != channels and not add_pressure:
                parser.error("An initialization cannot remove a previously trained channel.")
            if source["dataset"] != "Winhausen2023 Mont Terri":
                parser.error("Warm start requires the same dataset.")
            training_specimens = (args.training_specimens if args.training_specimens is not None
                                  else source["training_specimens"])
            if not set(source["training_specimens"]) <= set(training_specimens):
                parser.error("Warm start cannot exclude previously fitted training specimens.")
            parameters = source["material_parameters"]
            network = np.array(source["network"]["raw_parameters"])
            storage_form = args.storage_form or source.get("storage_form", "neural")
            if storage_form == "identity":
                network = np.array([])
            elif source.get("storage_form", "neural") == "identity":
                network = MonotoneStorageNetwork.initialize(
                    parameters["initial_density"], parameters["damage_peak_density"]).raw_parameters
            source_evolution = source.get("evolution_network")
            initialization = dict(role=("stress-to-joint training checkpoint warm start" if add_pressure
                                        else "same-channel training checkpoint warm start"),
                source_channels=source["channels"],
                source_storage_form=source.get("storage_form", "neural"),
                source_training_specimens=source["training_specimens"],
                checkpoint=str(args.initialize_from.resolve()), material_parameters=parameters.copy(),
                storage_raw_parameters=network.tolist(),
                evolution_raw_parameters=(source_evolution["raw_parameters"]
                                          if source_evolution else None),
                evolution_input_size=source_evolution.get("input_size", 2) if source_evolution else None,
                source_directional_evolution=source.get("directional_evolution"))
        else:
            training_specimens = args.training_specimens
            parameters = wt.initial_parameters()
            storage_form = args.storage_form or "neural"
            network = (MonotoneStorageNetwork.initialize(
                parameters["initial_density"], parameters["damage_peak_density"]).raw_parameters
                if storage_form == "neural" else np.array([]))
            parameters, initialization = elastic_start(parameters, network, channels,
                                                        training_specimens, storage_form)
        try:
            specimens = load_specimens(None, training_specimens=training_specimens)
        except ValueError as exc:
            parser.error(str(exc))
        training = [p for p in specimens if p["role"] == "train"]
        training_specimens = [p["specimen_id"] for p in training]
        if overrides:
            initialization["material_overrides"] = overrides
            parameters = {**parameters, **overrides}
        if args.reset_storage:
            if storage_form == "identity":
                parser.error("Analytical storage has no network weights to reset.")
            network = MonotoneStorageNetwork.initialize(
                parameters["initial_density"], parameters["damage_peak_density"],
                width=len(network)//3).raw_parameters
            initialization["storage_reset"] = dict(
                rule="Canonical monotone storage initialization at the retained material; weights remain trainable.",
                raw_parameters=network.tolist())
        record = dict(method="physics-constrained neural constitutive storage", dataset="Winhausen2023 Mont Terri",
            validation_role=(f"{len(training)}-specimen fitting; {len(specimens)-len(training)} "
                             "nonfitted specimens also used in structural diagnosis"),
            channels=channels, material_parameters=parameters, storage_form=storage_form,
            network=dict(raw_parameters=network.tolist(), hidden_width=len(network)//3),
            fixed_parameters=wt.FIXED, reference_confinement_MPa=wt.REFERENCE_CONFINEMENT,
            material_bounds={n: list(v[1:3]) for n, v in wt.PARAMETERS.items()},
            initialization=initialization,
            preparation="First recorded q is a drained elastic prestress at zero excess pressure; fluid content is then sealed.",
            objective=wt.OBJECTIVE,
            increment_weight=(args.increment_weight if args.increment_weight is not None else
                              source.get("increment_weight", 0.) if args.initialize_from else 0.),
            volume_stage_weight=(args.volume_stage_weight if args.volume_stage_weight is not None else
                                 source.get("volume_stage_weight", 0.) if args.initialize_from else 0.),
            stage_increment_weight=(args.stage_increment_weight if args.stage_increment_weight is not None else
                                    source.get("stage_increment_weight", 0.) if args.initialize_from else 0.),
            response_measure=(args.response_measure if args.response_measure is not None else
                              source.get("response_measure", "point") if args.initialize_from else "point"),
            samples=source.get("samples", 100) if args.initialize_from else 100,
            max_increment=source.get("max_increment", .0005) if args.initialize_from else .0005,
            training_specimens=training_specimens)
        if args.matrix_evolution or args.fabric_evolution or source_evolution is not None:
            evolution = (MatrixEvolutionNetwork(
                np.array(source_evolution["raw_parameters"]), parameters["damage_peak_density"],
                source_evolution["pressure_scale_MPa"], source_evolution.get("input_size", 2),
                local_width=source_evolution.get("local_width", 0),
                fabric_knots=source_evolution.get("fabric_knots", ())) if source_evolution else
                MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"]))
            if args.fabric_evolution:
                evolution = evolution.with_fabric_input()
                initialization["fabric_input"] = "Zero new weights preserve the supplied constitutive response."
            fabric_source = args.fabric_source or (source_evolution or {}).get("fabric_source", "plastic_stress")
            if args.fabric_source and evolution.input_size != 3:
                parser.error("Selecting a fabric source requires a three-input network; use --fabric-evolution.")
            if args.separable_features:
                evolution = evolution.with_separable_features(args.separable_features,
                    args.gate_center, args.gate_slope)
                initialization["separable_features"] = dict(local_width=args.separable_features,
                    gate_center=args.gate_center, gate_slope=args.gate_slope)
            if args.interpolate_fabric or (evolution.fabric_knots and args.training_specimens is not None):
                if fabric_source != "plastic_stress":
                    parser.error("Angle-derived interpolation supports require the effective-stress direction.")
                knots = sorted({0., *[float((3*np.sin(np.deg2rad(p["angle"]))**2-1)/2)
                                     for p in training]})
                evolution = replace(evolution, fabric_knots=knots)
                initialization["fabric_interpolation"] = dict(
                    knots=knots, rule="C1 cubic smoothstep between training projections and F=0; constant extension.")
            if args.set_evolution_weight:
                raw = evolution.raw_parameters.copy()
                weight_overrides = {}
                for index, value in args.set_evolution_weight:
                    try:
                        index, value = int(index), float(value)
                    except ValueError:
                        parser.error("Evolution weight requires an integer index and numeric value.")
                    if not 0 <= index < len(raw) or not np.isfinite(value):
                        parser.error(f"Evolution weight requires index in [0, {len(raw)-1}] and a finite value.")
                    raw[index] = value
                    weight_overrides[str(index)] = value
                initialization["evolution_weight_overrides"] = weight_overrides
                evolution = replace(evolution, raw_parameters=raw)
            record.update(method="physics-constrained neural storage and matrix evolution",
                evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                    hidden_width=evolution.hidden_width, local_width=evolution.local_width,
                    input_size=evolution.input_size,
                    fabric_source=fabric_source,
                    resistance_uses_pressure=(not args.resistance_without_pressure
                        and (source_evolution or {}).get("resistance_uses_pressure", True)),
                    pressure_scale_MPa=evolution.pressure_scale),
                evolution_closure="Inputs log(d/d_peak) and tr(sigma-uI)/(3 p_ref)-1; "
                    "anchored tanh log factors; R=R_base exp(g_R), beta=mu-(mu-beta_base)exp(g_beta).")
            if evolution.fabric_knots:
                record["evolution_network"]["fabric_knots"] = list(evolution.fabric_knots)
                record["evolution_closure"] += (
                    " Complete log-factor maps at the saved fabric knots are blended by "
                    "s(t)=t^2(3-2t), t=(F-F_left)/(F_right-F_left), clipped to [0,1]. "
                    "Density and pressure derivatives use the same blend; the fabric derivative is "
                    "s'(t)*(g_right-g_left)/(F_right-F_left).")
            if evolution.input_size == 3:
                if fabric_source == "matrix_force":
                    record["evolution_closure"] += (
                        " Third input is M_N:X_c/sqrt(X_c:Kdev:X_c), X_c=sigma-uI-Hc*c/d; "
                        "value and gradient are zero at zero deviatoric matrix force. "
                        "Reference: d=d_peak, p_eff=p_ref, deviatoric X_c=0.")
                else:
                    record["evolution_closure"] += (
                        " Third input is 3/2 (N-I/3):s_eff/sqrt(3/2 s_eff:s_eff), "
                        "set to zero at hydrostatic stress. Reference: d=d_peak, sigma_eff=p_ref I.")
            if evolution.local_width:
                record["evolution_closure"] += (
                    " Separable local density/pressure map multiplied by normalized tanh G(F), "
                    "with G(0)=0 and G(1)=1; gate center uses a logit coordinate and slope a log coordinate.")
            if not record["evolution_network"]["resistance_uses_pressure"]:
                record["evolution_closure"] += (
                    " Resistance evaluates the network at p_ref, so R depends only on density and fabric; "
                    "flow retains current effective mean stress. Reference anchors are unchanged.")
        elif args.resistance_without_pressure:
            parser.error("Removing resistance pressure dependence requires a matrix evolution network.")
        elif args.separable_features:
            parser.error("Separating fabric features requires a matrix evolution network.")
        elif args.interpolate_fabric:
            parser.error("Fabric interpolation requires a three-input matrix evolution network.")
        elif args.set_evolution_weight:
            parser.error("Evolution initialization options require a matrix evolution network.")
        elif args.fabric_source:
            parser.error("Selecting a fabric source requires a matrix evolution network.")
    storage_form = record.get("storage_form", "neural")
    if storage_form == "identity":
        record["method"] = ("analytical plane storage and neural matrix evolution"
                            if "evolution_network" in record else "analytical plane storage")
    if args.block == "storage" and storage_form == "identity":
        parser.error("Analytical storage has no trainable storage coordinates.")
    args.samples = args.samples if args.samples is not None else record.get("samples", 100)
    args.max_increment = (args.max_increment if args.max_increment is not None
                          else record.get("max_increment", .0005))
    for name, default in (("max_log_density_increment", 0.), ("min_increment", 1e-9)):
        requested = getattr(args, name)
        # Warm starts inherit integration choices just like samples and axial steps.
        inherited = source.get(name, default) if args.action == "initialize" and args.initialize_from else record.get(name, default)
        setattr(args, name, inherited if requested is None else requested)
    if args.samples < 3 or args.max_increment <= 0:
        parser.error("At least three samples and a positive integration increment are required.")
    if (not np.isfinite(args.max_log_density_increment) or args.max_log_density_increment < 0
            or not np.isfinite(args.min_increment) or args.min_increment <= 0):
        parser.error("Density substepping requires a nonnegative limit and positive minimum increment.")
    if args.block in ("network", "storage", "evolution", "fabric") and args.material:
        parser.error("A network block cannot also select material coordinates.")
    if (args.block in ("evolution", "fabric", "flow", "response") or args.initial_evolution) and "evolution_network" not in record:
        parser.error("This checkpoint has no matrix evolution network.")
    if args.block == "fabric" and record["evolution_network"].get("input_size", 2) != 3:
        parser.error("The fabric block requires a fabric-conditioned matrix network.")
    flow_material = ["dilation_fraction", "dilation_residual_fraction",
                     "dilation_density_ratio", "matrix_contraction"]
    if args.block in ("flow", "response") and args.material and not set(args.material) <= set(flow_material):
        parser.error("Flow and response blocks select only dilatancy material parameters.")
    record.update(samples=args.samples, max_increment=args.max_increment, objective=wt.OBJECTIVE,
                  max_log_density_increment=args.max_log_density_increment, min_increment=args.min_increment,
                  increment_weight=record.get("increment_weight", 0.),
                  volume_stage_weight=record.get("volume_stage_weight", 0.),
                  stage_increment_weight=record.get("stage_increment_weight", 0.),
                  response_measure=record.get("response_measure", "point"),
                  directional_evolution=wt.DIRECTIONAL_EVOLUTION,
                  sampling="Uniform acquisition times, early loading and observed q peak; pressure-independent stations. Least-squares monotone strain in acquisition order with at most 0.02 percentage-point recording correction; original strain and response targets retained.",
                  active_material_parameters=(args.material or flow_material if args.block in ("flow", "response") else
                                              args.material or list(wt.NAMES)
                                              if args.block in ("joint", "material") else []),
                  train_network=storage_form == "neural" and args.block in ("joint", "network", "storage"),
                  train_evolution=("evolution_network" in record
                                   and args.block in ("joint", "network", "evolution", "fabric", "flow", "response")))
    if record["response_measure"] == "cell":
        record["objective"] = wt.CELL_OBJECTIVE
    if record["increment_weight"]:
        error_description = ("cell-mean errors" if record["response_measure"] == "cell"
                             else "response errors")
        record["objective"] += (
            f" Plus {record['increment_weight']:g} times the sum of squared adjacent "
            f"differences of {error_description} (MPa^2) in each fitted channel, at the same "
            "stress-defined observation stations; finite increments, not measured tangents.")
    if record["stage_increment_weight"]:
        record["objective"] += (
            f" Plus {record['stage_increment_weight']:g} times squared response-error changes "
            "over common stress-defined mid-rise-to-peak and peak-to-mid-fall intervals, "
            "using point errors or signed cell-mean errors in every fitted channel.")
    if record["volume_stage_weight"]:
        if record["response_measure"] != "cell" or set(channels) != {"q_MPa", "pressure_MPa"}:
            parser.error("Volume-stage residuals require joint channels and the cell response measure.")
        record["objective"] += (
            f" Plus {record['volume_stage_weight']:g} times squared changes of e_u-B_q*e_q "
            "over the union of q/u mid-rise-to-peak and peak-to-mid-fall cell intervals. "
            "B_q=M_Sigma*(b*S0)[0] follows the candidate material; both errors use common "
            "endpoints. Pressure-equivalent irreversible-volume errors from the same observations.")
    if record["train_evolution"]:
        e = record["evolution_network"]
        evolution = MatrixEvolutionNetwork(e["raw_parameters"], record["material_parameters"]["damage_peak_density"],
            e.get("pressure_scale_MPa", wt.REFERENCE_CONFINEMENT), e.get("input_size", 2),
            local_width=e.get("local_width", 0), fabric_knots=e.get("fabric_knots", ()))
        record["active_evolution_parameters"] = evolution.parameter_indices(args.block)
    else:
        record["active_evolution_parameters"] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        mapper = pool.map if args.workers > 1 else map
        if args.action == "evaluate":
            paths = [wt.prepare_path(p, args.samples) for p in
                     load_specimens(args.role, training_specimens=record.get("training_specimens"))]
            network = record["network"]
            evolution = record.get("evolution_network")
            if args.initial_network and storage_form == "neural":
                p = record["material_parameters"]
                raw = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"],
                                                       network["hidden_width"]).raw_parameters
                network = dict(raw_parameters=raw.tolist(), hidden_width=len(raw)//3)
            if evolution and (args.initial_network or args.initial_evolution):
                input_size = evolution.get("input_size", 2)
                raw = MatrixEvolutionNetwork.initialize(record["material_parameters"]["damage_peak_density"],
                    input_size=input_size).raw_parameters
                evolution = dict(raw_parameters=raw.tolist(), hidden_width=len(raw)//(input_size+3),
                                 input_size=input_size,
                                 resistance_uses_pressure=evolution.get("resistance_uses_pressure", True),
                                 fabric_source=evolution.get("fabric_source", "plastic_stress"),
                                 pressure_scale_MPa=wt.REFERENCE_CONFINEMENT)
                if record["evolution_network"].get("fabric_knots"):
                    evolution["fabric_knots"] = record["evolution_network"]["fabric_knots"]
            raw_evolution = np.array(evolution["raw_parameters"]) if evolution else None
            results = list(mapper(partial(wt.simulate, storage_form=storage_form,
                fabric_source=(evolution or {}).get("fabric_source", "plastic_stress")),
                repeat(record["material_parameters"]),
                repeat(np.array(network["raw_parameters"])), paths, repeat(args.max_increment), repeat(raw_evolution),
                repeat(evolution.get("input_size", 2) if evolution else 2),
                repeat(evolution.get("resistance_uses_pressure", True) if evolution else True),
                repeat(args.max_log_density_increment), repeat(args.min_increment),
                repeat(evolution.get("local_width", 0) if evolution else 0),
                repeat(evolution.get("fabric_knots", ()) if evolution else ())))
            prefix = args.role+("_initial_network" if args.initial_network else
                               "_initial_evolution" if args.initial_evolution else "")
            save_paths(args.output / f"{prefix}_evaluation_paths.csv", results)
            evaluation = dict(role=args.role, samples=args.samples, max_increment=args.max_increment,
                training_specimens=record.get("training_specimens"),
                max_log_density_increment=args.max_log_density_increment, min_increment=args.min_increment,
                directional_evolution=wt.DIRECTIONAL_EVOLUTION,
                material_parameters=record["material_parameters"], network=network, storage_form=storage_form,
                evolution_network=evolution,
                channels=record["channels"], objective=record["objective"],
                increment_weight=record["increment_weight"], response_measure=record["response_measure"],
                volume_stage_weight=record["volume_stage_weight"],
                stage_increment_weight=record["stage_increment_weight"],
                loss=sum(float(np.linalg.norm(wt.residuals(p, r, record["channels"],
                    record["increment_weight"], record["response_measure"],
                    record["volume_stage_weight"], record["stage_increment_weight"]))**2)
                    for p, r in zip(paths, results)),
                network_state=("initial weights" if args.initial_network else
                               "initial matrix weights" if args.initial_evolution else "trained weights"),
                metrics={p["specimen_id"]: wt.path_metrics(p, r) for p, r in zip(paths, results)})
            save_json(args.output / f"{prefix}_evaluation.json", evaluation)
            for key, metrics in evaluation["metrics"].items():
                print(key, "RMSE q/u", metrics["q_MPa"]["rmse"], metrics["pressure_MPa"]["rmse"],
                      "q peak", metrics["q_peak"]["predicted_MPa"],
                      "u end", metrics["pressure_end"]["predicted_MPa"])
            return
        objective = TrainingLoss(record, args.output, mapper)
        start = wt.encode(record["material_parameters"], objective.names)
        bounds = wt.coordinate_bounds(objective.names)
        if objective.train_network:
            start = np.r_[start, record["network"]["raw_parameters"]]
            bounds += [(None, None)]*len(record["network"]["raw_parameters"])
        if objective.train_evolution:
            start = np.r_[start, np.array(record["evolution_network"]["raw_parameters"])[objective.evolution_indices]]
            bounds += [(None, None)]*len(objective.evolution_indices)
        try:
            objective(start)
            if args.action == "train":
                if args.optimizer == "least-squares":
                    # Queue the independent coordinate histories together so
                    # short specimens do not leave workers idle behind the
                    # longest specimen at every small candidate-batch barrier.
                    batch_size = len(bounds) if args.workers > 1 else 1
                    options = dict(x_scale=1. if args.coordinate_scaling == "unit" else "jac",
                                   coordinate_origin=args.coordinate_origin,
                                   difference_step=args.difference_step,
                                   max_nfev=args.max_evaluations,
                                   history_batch_size=batch_size, workers=args.workers,
                                   trust_solver=args.trust_solver)
                    result = fit_history_least_squares(objective, start, bounds,
                                                      args.max_evaluations, batch_size,
                                                      args.trust_solver,
                                                      args.output / "linearization.npz"
                                                      if args.trust_solver == "lsmr" else None,
                                                      args.coordinate_scaling, args.coordinate_origin,
                                                      args.difference_step)
                else:
                    options = dict(initial_tr_radius=args.initial_radius, final_tr_radius=args.final_radius)
                    if args.max_evaluations is not None:
                        options["maxfev"] = args.max_evaluations
                    result = minimize(objective, start, method="COBYQA", bounds=bounds, options=options)
                record["optimizer"].update(status="stopped", success=bool(result.success),
                    message=str(result.message), method=args.optimizer, options=options,
                    evaluations=objective.evaluations, optimizer_evaluations=int(result.nfev),
                    unsolved_evaluations=objective.unsolved_evaluations,
                    last_unsolved_candidate=objective.last_unsolved_candidate,
                    initial_loss=objective.initial_loss)
                if args.optimizer == "least-squares":
                    # These describe the optimizer's terminal iterate; the
                    # selected checkpoint can instead be a better finite-
                    # difference candidate. A batch limit or small step alone
                    # does not establish stationarity of the history fit.
                    record["optimizer"].update(
                        terminal_loss=float(2*result.cost),
                        optimality=float(result.optimality),
                        terminal_coordinates=result.x.tolist(),
                        active_mask=result.active_mask.tolist())
            else:
                record["optimizer"]["status"] = "initialized"
        except Exception as exc:
            record["optimizer"] = objective.optimizer_state()
            record["optimizer"].update(status="failed", message=str(exc))
            save_json(checkpoint, record)
            raise
        save_json(checkpoint, record)


if __name__ == "__main__":
    main()
