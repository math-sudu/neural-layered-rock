"""Fit Mont Terri material, neural storage and matrix evolution on training specimens."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from itertools import repeat
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


def elastic_start(parameters, network, channels):
    """Fit the recorded 0--0.08% loading interval for an initial material.

    This is an initialization only. The full-path fit can adjust all of these
    constants; early inelasticity is not permanently excluded from the model.
    """
    paths = []
    for p in load_specimens("train"):
        h = p["history"]
        control = wt.recording_control(h["axial_strain"])
        index = np.flatnonzero((control > 0) & (control <= .0008))
        index = index[np.linspace(0, len(index)-1, min(20, len(index))).astype(int)]
        paths.append((p, index, control))

    def objective(coordinates):
        p = wt.decode(coordinates, parameters, wt.ELASTIC_NAMES)
        residual = []
        for specimen, index, control in paths:
            model = wt.material(p, specimen, network)
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


def solve_training_path(*arguments):
    """Keep a constitutive trial failure attached to its candidate in a batch."""
    try:
        return wt.simulate(*arguments)
    except (wt.ConstitutiveSolveError, wt.PrestressInitializationError) as exc:
        return exc


class TrainingLoss:
    def __init__(self, record, output, mapper):
        self.record, self.output, self.mapper = record, output, mapper
        self.paths = wt.training_paths(record["samples"])
        self.evaluations, self.best_loss = 0, np.inf
        self.names = tuple(record["active_material_parameters"])
        self.train_network = record["train_network"]
        self.train_evolution = record.get("train_evolution", False)
        self.initial_loss = None
        self.unsolved_evaluations = 0
        self.last_unsolved_candidate = None
        self.residual_size = None

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
        evolution = (coordinates[n+size:] if self.train_evolution else
                     np.array(evolution["raw_parameters"]) if evolution else None)
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
        return self.residual_vectors([coordinates])[0]

    def residual_vectors(self, coordinates):
        """Solve independent candidates together, then save them in input order.

        Only path solves run in the process pool. Checkpoint writes and the
        accepted candidate remain owned by this process.
        """
        candidates = [self.unpack(x) for x in coordinates]
        tasks = [(p, network, path, self.record["max_increment"], evolution)
                 for p, network, evolution in candidates for path in self.paths]
        outcomes = list(self.mapper(solve_training_path, *zip(*tasks)))
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
        residual = np.concatenate([wt.residuals(path, result, self.record["channels"])
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
                self.record["evolution_network"] = dict(raw_parameters=evolution.tolist(),
                    hidden_width=len(evolution)//5, pressure_scale_MPa=wt.REFERENCE_CONFINEMENT)
            save_json(self.output / "checkpoint.json", self.record)
            save_paths(self.output / "train_paths.csv", results)
            print(f"evaluation={self.evaluations} loss={loss:.7g}", flush=True)
        elif self.evaluations % 25 == 0:
            print(f"evaluation={self.evaluations} best_loss={self.best_loss:.7g}", flush=True)
        return residual


def history_jacobian(objective, coordinates, bounds, step=1e-4, batch_size=1):
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
    return np.column_stack(columns)


def fit_history_least_squares(objective, start, bounds, max_evaluations, batch_size=1):
    """Use residual directions to resolve stage-dependent training errors.

    Bounds and finite differences remain in the existing material and network
    coordinates. Jacobian column norms scale the trust-region steps. The
    evaluation limit counts trust-region trials; the checkpoint also records
    all history calls used for their Jacobians.
    """
    box = np.array([(lo if lo is not None else -np.inf,
                     hi if hi is not None else np.inf) for lo, hi in bounds]).T
    return least_squares(objective.residual_vector, start,
                         jac=lambda x: history_jacobian(objective, x, bounds,
                                                       batch_size=batch_size),
                         bounds=box, x_scale="jac", max_nfev=max_evaluations)


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
    parser.add_argument("--workers", type=int, default=7,
                        help="Path processes; extra workers also evaluate independent Jacobian columns.")
    parser.add_argument("--channels", choices=("joint", "stress"), default="joint")
    parser.add_argument("--role", choices=("train", "test"), default="train")
    parser.add_argument("--initial-network", action="store_true",
                        help="Evaluate initial storage and evolution weights at the saved fitted material.")
    parser.add_argument("--initial-evolution", action="store_true",
                        help="Reset only the matrix network for a learning diagnostic.")
    parser.add_argument("--matrix-evolution", action="store_true",
                        help="Initialize a new experiment with the pressure/history matrix network.")
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
    parser.add_argument("--block", choices=("joint", "material", "network", "storage", "evolution"), default="joint",
                        help="Coordinates to optimize in this continuation stage.")
    parser.add_argument("--initial-radius", type=float, default=.1,
                        help="Initial COBYQA trust-region radius.")
    parser.add_argument("--final-radius", type=float, default=.005,
                        help="Final COBYQA trust-region radius.")
    parser.add_argument("--optimizer", choices=("cobyqa", "least-squares"), default="cobyqa",
                        help="Scalar interpolation or complete-history residual Jacobians.")
    parser.add_argument("--max-evaluations", type=int,
                        help="Resumable batch: COBYQA calls or least-squares trials (excluding Jacobian calls).")
    args = parser.parse_args()
    if args.parameters and args.action != "evaluate":
        parser.error("--parameters applies only to evaluate; use --initialize-from for training.")
    if (args.initial_network or args.initial_evolution) and args.action != "evaluate":
        parser.error("Initial-weight diagnostics require evaluate.")
    if (args.initialize_from or args.reset_storage or args.matrix_evolution or args.set_material
            or args.set_evolution_weight) and args.action != "initialize":
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
            if (source["dataset"] != "Winhausen2023 Mont Terri"
                    or source["training_specimens"] != [p["specimen_id"] for p in load_specimens("train")]):
                parser.error("Warm start requires the same dataset and training specimens.")
            parameters = source["material_parameters"]
            network = np.array(source["network"]["raw_parameters"])
            source_evolution = source.get("evolution_network")
            initialization = dict(role=("stress-to-joint training checkpoint warm start" if add_pressure
                                        else "same-channel training checkpoint warm start"),
                source_channels=source["channels"],
                checkpoint=str(args.initialize_from.resolve()), material_parameters=parameters.copy(),
                storage_raw_parameters=network.tolist(),
                evolution_raw_parameters=(source_evolution["raw_parameters"]
                                          if source_evolution else None),
                source_directional_evolution=source.get("directional_evolution"))
        else:
            parameters = wt.initial_parameters()
            network = MonotoneStorageNetwork.initialize(
                parameters["initial_density"], parameters["damage_peak_density"]).raw_parameters
            parameters, initialization = elastic_start(parameters, network, channels)
        if overrides:
            initialization["material_overrides"] = overrides
            parameters = {**parameters, **overrides}
        if args.reset_storage:
            network = MonotoneStorageNetwork.initialize(
                parameters["initial_density"], parameters["damage_peak_density"],
                width=len(network)//3).raw_parameters
            initialization["storage_reset"] = dict(
                rule="Canonical monotone storage initialization at the retained material; weights remain trainable.",
                raw_parameters=network.tolist())
        record = dict(method="physics-constrained neural constitutive storage", dataset="Winhausen2023 Mont Terri",
            validation_role="seven-specimen training; five complete orientations withheld",
            channels=channels, material_parameters=parameters,
            network=dict(raw_parameters=network.tolist(), hidden_width=len(network)//3),
            fixed_parameters=wt.FIXED, reference_confinement_MPa=wt.REFERENCE_CONFINEMENT,
            material_bounds={n: list(v[1:3]) for n, v in wt.PARAMETERS.items()},
            initialization=initialization,
            preparation="First recorded q is a drained elastic prestress at zero excess pressure; fluid content is then sealed.",
            objective=wt.OBJECTIVE,
            training_specimens=[p["specimen_id"] for p in load_specimens("train")])
        if args.matrix_evolution or source_evolution is not None:
            evolution = (MatrixEvolutionNetwork(
                np.array(source_evolution["raw_parameters"]), parameters["damage_peak_density"],
                source_evolution["pressure_scale_MPa"]) if source_evolution else
                MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"]))
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
                evolution = MatrixEvolutionNetwork(raw, parameters["damage_peak_density"],
                                                   evolution.pressure_scale)
            record.update(method="physics-constrained neural storage and matrix evolution",
                evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                    hidden_width=len(evolution.raw_parameters)//5,
                    pressure_scale_MPa=evolution.pressure_scale),
                evolution_closure="Inputs log(d/d_peak) and tr(sigma-uI)/(3 p_ref)-1; "
                    "anchored tanh log factors; R=R_base exp(g_R), beta=mu-(mu-beta_base)exp(g_beta).")
        elif args.set_evolution_weight:
            parser.error("Evolution weight initialization requires a matrix evolution network.")
    args.samples = args.samples if args.samples is not None else record.get("samples", 100)
    args.max_increment = (args.max_increment if args.max_increment is not None
                          else record.get("max_increment", .0005))
    if args.samples < 3 or args.max_increment <= 0:
        parser.error("At least three samples and a positive integration increment are required.")
    if args.block in ("network", "storage", "evolution") and args.material:
        parser.error("A network block cannot also select material coordinates.")
    if (args.block == "evolution" or args.initial_evolution) and "evolution_network" not in record:
        parser.error("This checkpoint has no matrix evolution network.")
    record.update(samples=args.samples, max_increment=args.max_increment, objective=wt.OBJECTIVE,
                  directional_evolution=wt.DIRECTIONAL_EVOLUTION,
                  sampling="Uniform acquisition times, early loading and observed q peak; pressure-independent stations. Least-squares monotone strain in acquisition order with at most 0.02 percentage-point recording correction; original strain and response targets retained.",
                  active_material_parameters=(args.material or list(wt.NAMES)
                                              if args.block in ("joint", "material") else []),
                  train_network=args.block in ("joint", "network", "storage"),
                  train_evolution=("evolution_network" in record
                                   and args.block in ("joint", "network", "evolution")))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        mapper = pool.map if args.workers > 1 else map
        if args.action == "evaluate":
            paths = [wt.prepare_path(p, args.samples) for p in load_specimens(args.role)]
            network = record["network"]
            evolution = record.get("evolution_network")
            if args.initial_network:
                p = record["material_parameters"]
                raw = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"],
                                                       network["hidden_width"]).raw_parameters
                network = dict(raw_parameters=raw.tolist(), hidden_width=len(raw)//3)
            if evolution and (args.initial_network or args.initial_evolution):
                raw = MatrixEvolutionNetwork.initialize(record["material_parameters"]["damage_peak_density"]).raw_parameters
                evolution = dict(raw_parameters=raw.tolist(), hidden_width=len(raw)//5,
                                 pressure_scale_MPa=wt.REFERENCE_CONFINEMENT)
            raw_evolution = np.array(evolution["raw_parameters"]) if evolution else None
            results = list(mapper(wt.simulate, repeat(record["material_parameters"]),
                repeat(np.array(network["raw_parameters"])), paths, repeat(args.max_increment), repeat(raw_evolution)))
            prefix = args.role+("_initial_network" if args.initial_network else
                               "_initial_evolution" if args.initial_evolution else "")
            save_paths(args.output / f"{prefix}_evaluation_paths.csv", results)
            evaluation = dict(role=args.role, samples=args.samples, max_increment=args.max_increment,
                directional_evolution=wt.DIRECTIONAL_EVOLUTION,
                material_parameters=record["material_parameters"], network=network,
                evolution_network=evolution,
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
            start = np.r_[start, record["evolution_network"]["raw_parameters"]]
            bounds += [(None, None)]*len(record["evolution_network"]["raw_parameters"])
        try:
            objective(start)
            if args.action == "train":
                if args.optimizer == "least-squares":
                    # Queue the independent coordinate histories together so
                    # short specimens do not leave workers idle behind the
                    # longest specimen at every small candidate-batch barrier.
                    batch_size = len(bounds) if args.workers > 1 else 1
                    options = dict(x_scale="jac", difference_step=1e-4,
                                   max_nfev=args.max_evaluations,
                                   history_batch_size=batch_size, workers=args.workers)
                    result = fit_history_least_squares(objective, start, bounds,
                                                      args.max_evaluations, batch_size)
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
