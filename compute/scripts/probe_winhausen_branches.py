"""Probe branch continuation and matrix-force input feedback on saved states."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute/src"))

from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.coupled_friction_damage import ConstitutiveSolveError
from tfenn_invcalib.winhausen_data import load_specimens


def restore_state(row, model, fluid_content):
    state = dict(strain=np.array([float(row[f"strain_{i}"]) for i in range(6)]),
        matrix_strain=np.array([float(row[f"matrix_strain_{i}"]) for i in range(6)]),
        plane_modes=np.array([float(row[f"plane_mode_{i}"]) for i in range(3)]),
        compaction=float(row["compaction"]), d=float(row["density"]), fluid_content=fluid_content,
        matrix_flow_fraction=float(row["matrix_flow_fraction"]),
        matrix_multiplier=float(row["matrix_multiplier"]))
    state["response"] = model.energy.response(state["strain"], state["matrix_strain"],
        state["plane_modes"], state["compaction"], state["d"], model.D, fluid_content)
    return state


def probe(trace, output):
    metadata = json.loads(trace.with_suffix(".json").read_text(encoding="utf-8"))
    record = json.loads(Path(metadata["parameters"]).read_text(encoding="utf-8"))
    path = dict(angle=metadata["angle_deg"], confinement=metadata["confinement_MPa"])
    evolution = record.get("evolution_network")
    model = wt.material(record["material_parameters"], path,
        np.asarray(record["network"]["raw_parameters"]),
        np.asarray(evolution["raw_parameters"]) if evolution else None,
        evolution.get("input_size", 2) if evolution else 2,
        evolution.get("resistance_uses_pressure", True) if evolution else True,
        evolution.get("local_width", 0) if evolution else 0,
        evolution.get("fabric_knots", ()) if evolution else (),
        storage_form=metadata.get("storage_form", "neural"),
        fabric_source=(evolution or {}).get("fabric_source", "plastic_stress"))
    with gzip.open(trace.with_suffix(".csv.gz"), "rt", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    index = int(metadata["largest_damage_step"]["step"])
    fluid = metadata["initial_state"]["fluid_content"]
    previous = restore_state(rows[index-1], model, fluid)
    recorded = restore_state(rows[index], model, fluid)
    p0 = path["confinement"]
    target = recorded["strain"][0]
    common = dict(transverse_stress=[p0, p0, 0., 0., 0.], fluid_content=fluid)
    outcomes = []

    def outcome(kind, value, state, max_damage, min_dq):
        return dict(specimen_id=metadata["specimen_id"], kind=kind, value=value,
            axial_strain=float(rows[index]["axial_strain"]),
            q_MPa=float(state["response"]["total_stress"][0]-p0),
            pressure_MPa=float(state["response"]["pressure"]), density=float(state["d"]),
            max_damage_increment=float(max_damage), min_stress_increment_MPa=float(min_dq),
            solver_route=state["solver_route"], error="")

    for subdivisions in (1, 2, 4, 8):
        state = previous
        max_damage, min_dq = 0., 0.
        for axial in np.linspace(previous["strain"][0], target, subdivisions+1)[1:]:
            new = model.step(axial, previous=state, **common)
            max_damage = max(max_damage, new["d"]-state["d"])
            min_dq = min(min_dq, new["response"]["total_stress"][0]-state["response"]["total_stress"][0])
            state = new
        outcomes.append(outcome("subdivisions", subdivisions, state, max_damage, min_dq))

    for factor in (0., .25, .5, .75, 1., 1.5, 2.):
        # Extrapolations also explore beyond the recorded endpoint. History
        # remains fixed at previous; only the nonlinear solver seed changes.
        guess = {key: previous[key]+factor*(recorded[key]-previous[key])
                 for key in ("matrix_strain", "plane_modes", "compaction", "d")}
        guess["matrix_multiplier"] = factor*recorded["matrix_multiplier"]
        guess["matrix_flow_fraction"] = np.clip(previous["matrix_flow_fraction"]
            + factor*(recorded["matrix_flow_fraction"]-previous["matrix_flow_fraction"]), 0., 1.)
        try:
            state = model.step(target, previous=previous, initial_guess=guess, **common)
        except ConstitutiveSolveError as exc:
            outcomes.append(dict(specimen_id=metadata["specimen_id"], kind="seed_factor",
                                 value=factor, error=str(exc)))
        else:
            outcomes.append(outcome("seed_factor", factor, state, state["d"]-previous["d"],
                state["response"]["total_stress"][0]-previous["response"]["total_stress"][0]))
    if rows[index]["solver_route"] in {"fixed_density", "released_fixed_density"}:
        for subdivisions in (1, 2, 4):
            state = model.step(target, previous=previous,
                               density_search_subdivisions=subdivisions, **common)
            outcomes.append(outcome("density_bracket_subdivisions", subdivisions, state,
                state["d"]-previous["d"],
                state["response"]["total_stress"][0]-previous["response"]["total_stress"][0]))
    with (output / f"{metadata['specimen_id']}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
        writer.writeheader()
        writer.writerows(outcomes)
    seeds = [row for row in outcomes if row["kind"] == "seed_factor" and not row["error"]]
    print(json.dumps(dict(specimen_id=metadata["specimen_id"],
        seed_q_range_MPa=[min(row["q_MPa"] for row in seeds), max(row["q_MPa"] for row in seeds)],
        local_subdivisions=[{key: row[key] for key in ("value", "max_damage_increment", "min_stress_increment_MPa")}
                            for row in outcomes if row["kind"] == "subdivisions"],
        bracket_subdivisions=[{key: row[key] for key in ("value", "q_MPa", "density")}
                             for row in outcomes if row["kind"] == "density_bracket_subdivisions"],
        errors=[row["error"] for row in outcomes if row["error"]])), flush=True)


def probe_feedback(trace, parameters, output, increment):
    """Hold the preceding matrix-force direction only for a one-step control."""
    record = json.loads(parameters.read_text(encoding="utf-8"))
    evolution = record["evolution_network"]
    if evolution.get("fabric_source", "plastic_stress") != "matrix_force":
        raise ValueError("The feedback control requires a matrix-force input source.")
    specimen = next(item for item in load_specimens(None)
                    if item["specimen_id"] == trace.name)
    path = wt.prepare_path(specimen, record["samples"])
    model = wt.material(record["material_parameters"], path,
        np.asarray(record["network"]["raw_parameters"]),
        np.asarray(evolution["raw_parameters"]), evolution.get("input_size", 2),
        evolution.get("resistance_uses_pressure", True), evolution.get("local_width", 0),
        evolution.get("fabric_knots", ()), storage_form=record.get("storage_form", "neural"),
        fabric_source="matrix_force")
    initial = wt.initial_state(model, record["material_parameters"], path)
    with gzip.open(trace.with_suffix(".csv.gz"), "rt", encoding="utf-8") as stream:
        row = list(csv.DictReader(stream))[-1]
    previous = restore_state(row, model, initial["fluid_content"])
    axial = float(row["axial_strain"])+increment
    direction = model.fabric_direction
    projection = direction(previous["response"]["matrix_force"])[0]
    outcomes = []
    for kind in ("current_matrix_force", "frozen_previous_direction"):
        if kind == "frozen_previous_direction":
            model.fabric_direction = lambda stress: (projection, np.zeros(6))
        try:
            state = model.step(initial["strain"][0]+axial,
                [path["confinement"], path["confinement"], 0., 0., 0.], previous,
                fluid_content=initial["fluid_content"])
        except ConstitutiveSolveError as exc:
            outcomes.append(dict(kind=kind, resolved=False, error=str(exc)))
        else:
            updated_projection = direction(state["response"]["matrix_force"])[0]
            actual_resistance = model.law.evolution(state["d"], state["effective_mean_stress"],
                                                   updated_projection)[0]
            outcomes.append(dict(kind=kind, resolved=True, density=float(state["d"]),
                matrix_multiplier=float(state["matrix_multiplier"]),
                matrix_flow_fraction=float(state["matrix_flow_fraction"]),
                matrix_mean_force_MPa=float(state["matrix_mean_force"]),
                matrix_yield_MPa=float(state["matrix_yield"]),
                updated_force_projection=float(updated_projection),
                unfrozen_damage_consistency=float(state["response"]["matrix_release"]/actual_resistance-1),
                solver_route=state["solver_route"]))
    model.fabric_direction = direction
    result = dict(specimen_id=trace.name, parameters=str(parameters.resolve()),
        partial_trace=str(trace.with_suffix(".csv.gz").resolve()),
        previous_control_strain=float(row["axial_strain"]), target_control_strain=axial,
        axial_increment=increment, previous_density=previous["d"],
        previous_force_projection=float(projection),
        purpose="Local feedback control; freezing the input changes this step's closure and is not a repaired model.",
        outcomes=outcomes)
    destination = output/f"{trace.name}_input_feedback.json"
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specimens", nargs="+", required=True)
    parser.add_argument("--parameters", type=Path,
                        help="Parameter input for a feedback probe at an incomplete trace's last state.")
    parser.add_argument("--feedback-increment", type=float,
                        help="Next control-strain increment for the one-step feedback probe.")
    args = parser.parse_args()
    if (args.parameters is None) != (args.feedback_increment is None):
        parser.error("Supply both --parameters and --feedback-increment for a feedback probe.")
    if args.feedback_increment is not None and (not np.isfinite(args.feedback_increment) or args.feedback_increment <= 0):
        parser.error("The feedback increment must be positive and finite.")
    args.output.mkdir(parents=True, exist_ok=True)
    for specimen in args.specimens:
        if args.parameters:
            probe_feedback(args.traces/specimen, args.parameters, args.output, args.feedback_increment)
        else:
            probe(args.traces / specimen, args.output)


if __name__ == "__main__":
    main()
