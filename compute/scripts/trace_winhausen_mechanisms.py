"""Replay selected parameters and retain every accepted constitutive step."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute/src"))

from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.winhausen_data import load_specimens


def trace_one(arguments):
    source, specimen, samples, max_increment, output, density_limit, min_increment, storage_form = arguments
    record = json.loads(source.read_text(encoding="utf-8"))
    storage_form = storage_form or record.get("storage_form", "neural")
    density_limit = record.get("max_log_density_increment", 0.) if density_limit is None else density_limit
    min_increment = record.get("min_increment", 1e-9) if min_increment is None else min_increment
    parameters = record["material_parameters"]
    storage = np.asarray(record["network"]["raw_parameters"])
    evolution = record.get("evolution_network")
    evolution_inputs = evolution.get("input_size", 2) if evolution else 2
    fabric_source = evolution.get("fabric_source", "plastic_stress") if evolution else "plastic_stress"
    resistance_uses_pressure = evolution.get("resistance_uses_pressure", True) if evolution else True
    local_width = evolution.get("local_width", 0) if evolution else 0
    fabric_knots = evolution.get("fabric_knots", ()) if evolution else ()
    evolution = np.asarray(evolution["raw_parameters"]) if evolution else None
    path = wt.prepare_path(specimen, samples)
    model = wt.material(parameters, path, storage, evolution, evolution_inputs,
                        resistance_uses_pressure, local_width, fabric_knots, storage_form=storage_form,
                        fabric_source=fabric_source)
    m = model.energy
    initial = wt.initial_state(model, parameters, path)
    pressure_modulus = 1/(1/m.M + m.biot @ m.S0 @ m.biot)
    pressure_coefficient = pressure_modulus*(m.biot @ m.S0)[0]
    routes = Counter()
    writer = None
    last_axial = 0.
    largest_damage_step = None
    largest_stress_drop = None
    steps = 0
    destination = output / specimen["specimen_id"]
    # A failed replay leaves its partial trace available for diagnosis, but
    # must not retain completion metadata from a different parameter state.
    destination.with_suffix(".json").unlink(missing_ok=True)
    with gzip.open(destination.with_suffix(".csv.gz"), "wt", encoding="utf-8", newline="") as stream:
        def consume(axial, state, previous):
            nonlocal writer, last_axial, largest_damage_step, largest_stress_drop, steps
            f = state["response"]
            cm, z, d = state["matrix_strain"], state["plane_modes"], state["d"]
            dcm, dz = cm-previous["matrix_strain"], z-previous["plane_modes"]
            q = f["total_stress"][0]-path["confinement"]
            dq = f["total_stress"][0]-previous["response"]["total_stress"][0]
            (resistance, resistance_d, resistance_p, _, _, _,
             resistance_fabric, _) = model.law.evolution(
                 d, state["effective_mean_stress"], state["fabric_projection"])
            h, hp, _ = m.network.value_derivatives(d)
            backstress = m.plane_storage(d)[0] @ z/model.D
            traction = m.B.T @ f["plastic_stress"]
            row = dict(step=steps, axial_strain=axial, axial_increment=axial-last_axial,
                solver_route=state["solver_route"], q_MPa=q, delta_q_MPa=dq,
                pressure_MPa=f["pressure"], density=d, delta_density=d-previous["d"],
                log_density_increment=np.log(d/previous["d"]),
                matrix_multiplier=state["matrix_multiplier"], plane_multiplier=state["plane_multiplier"],
                matrix_volume=VOLUME @ cm, delta_matrix_volume=VOLUME @ dcm,
                matrix_axial=cm[0], delta_matrix_axial=dcm[0],
                plane_axial=m.B[0] @ z, delta_plane_axial=m.B[0] @ dz,
                compaction=state["compaction"],
                matrix_mean_force_MPa=state["matrix_mean_force"],
                matrix_equivalent_force_MPa=state["matrix_equivalent_force"],
                matrix_yield_MPa=state["matrix_yield"],
                plane_yield_MPa=state["plane_yield"]/np.sqrt(2),
                plane_normal_force_MPa=f["plane_force"][0],
                effective_mean_MPa=state["effective_mean_stress"],
                fabric_projection=state["fabric_projection"],
                damage_resistance_MPa=resistance,
                resistance_density_derivative_MPa=resistance_d,
                resistance_pressure_derivative=resistance_p,
                resistance_fabric_derivative_MPa=resistance_fabric,
                matrix_dilatancy=state["matrix_dilatancy"],
                matrix_flow_fraction=state["matrix_flow_fraction"],
                matrix_friction=model.law.friction(d)[0],
                matrix_release_MPa=f["matrix_self_release"],
                plane_release_MPa=f["plane_to_matrix_release"],
                storage_coordinate=h, storage_derivative=hp,
                pressure_stress_MPa=pressure_coefficient*(q-path["observations"]["q_MPa"][0]),
                pressure_matrix_MPa=pressure_modulus*(VOLUME @ (cm-initial["matrix_strain"])),
                pressure_plane_MPa=pressure_modulus*(z[0]-initial["plane_modes"][0]),
                pressure_compaction_MPa=pressure_modulus*(state["compaction"]-initial["compaction"]))
            for name, values in (("strain", state["strain"]), ("matrix_strain", cm),
                                 ("plane_mode", z), ("plane_traction_MPa", traction),
                                 ("plane_backstress_MPa", backstress)):
                row.update({f"{name}_{i}": value for i, value in enumerate(values)})
            row.update({f"work_{name}_MPa": value
                        for name, value in state["dissipation_increments"].items()})
            row = {name: value if isinstance(value, str) else float(value)
                   for name, value in row.items()}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            routes[state["solver_route"]] += 1
            event = {name: row[name] for name in ("step", "axial_strain", "axial_increment",
                "solver_route", "density", "delta_density", "log_density_increment", "q_MPa", "delta_q_MPa")}
            if largest_damage_step is None or event["log_density_increment"] > largest_damage_step["log_density_increment"]:
                largest_damage_step = event
            if largest_stress_drop is None or dq < largest_stress_drop["delta_q_MPa"]:
                largest_stress_drop = event
            last_axial = axial
            steps += 1

        result = wt.simulate(parameters, storage, path, max_increment, evolution,
                             evolution_inputs, resistance_uses_pressure, density_limit, min_increment, local_width, fabric_knots,
                             on_step=consume, storage_form=storage_form, fabric_source=fabric_source)
    metadata = dict(specimen_id=path["specimen_id"], role=path["role"],
        angle_deg=path["angle"], confinement_MPa=path["confinement"],
        parameters=str(source.resolve()), samples=samples, max_increment=max_increment,
        storage_form=storage_form,
        fabric_source=fabric_source,
        max_log_density_increment=density_limit, min_increment=min_increment,
        pressure_modulus_MPa=float(pressure_modulus), pressure_stress_coefficient=float(pressure_coefficient),
        initial_state={key: value.tolist() if isinstance(value, np.ndarray) else float(value)
                       for key, value in initial.items()
                       if key in ("strain", "matrix_strain", "plane_modes", "compaction", "d", "fluid_content")},
        steps=steps, solver_routes=dict(routes), largest_damage_step=largest_damage_step,
        largest_stress_drop=largest_stress_drop, metrics=wt.path_metrics(path, result))
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return {name: metadata[name] for name in ("specimen_id", "steps", "solver_routes", "largest_damage_step")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--storage-form", choices=("neural", "identity"),
                        help="Override the saved storage form for a fixed-parameter substitution.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-increment", type=float, required=True)
    parser.add_argument("--max-log-density-increment", type=float)
    parser.add_argument("--min-increment", type=float)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--role", choices=("train", "test", "all"), default="all")
    parser.add_argument("--specimens", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    record = json.loads(args.parameters.read_text(encoding="utf-8"))
    specimens = load_specimens(None if args.role == "all" else args.role,
                              training_specimens=record.get("training_specimens"))
    if args.specimens:
        available = {path["specimen_id"] for path in specimens}
        missing = set(args.specimens)-available
        if missing:
            parser.error(f"Specimens unavailable in this role: {sorted(missing)}")
        specimens = [path for path in specimens if path["specimen_id"] in args.specimens]
    if args.samples < 3 or args.max_increment <= 0 or args.workers < 1:
        parser.error("Use at least three samples, a positive increment and at least one worker.")
    args.output.mkdir(parents=True, exist_ok=True)
    arguments = [(args.parameters, path, args.samples, args.max_increment, args.output,
                  args.max_log_density_increment, args.min_increment, args.storage_form) for path in specimens]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(trace_one, argument): argument[1]["specimen_id"]
                   for argument in arguments}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(specimen_id=futures[future], error=str(exc))
                failures.append(result)
            print(json.dumps(result), flush=True)
    if failures:
        parser.exit(1, f"Unresolved paths: {', '.join(item['specimen_id'] for item in failures)}\n")


if __name__ == "__main__":
    main()
