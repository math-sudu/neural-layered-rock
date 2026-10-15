"""Compare loading processes and diagnose the fitted normal-bedding history."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_winhausen_mechanisms import read_trace, write_csv
from compare_winhausen_refits import compare_stages
from analyze_winhausen_strength import stage_projections
from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.winhausen_data import load_specimens


def normal_matrix_state(model, initial, path, axial, q, pressure):
    """Infer c, d and R conditional on a 90-degree closed, active matrix branch.

    Fluid balance supplies total volume and transverse isotropy supplies the
    two equal radial strains. Saturated elasticity then determines c. On the
    positive axial matrix-force branch, yield consistency is linear in d below
    d_peak and quadratic above it because mu=mu_res+mu_transient/d.
    These are constitutive diagnostics conditional on the supplied coefficients.
    """
    if path["angle"] != 90.:
        raise ValueError("The conditional reconstruction requires normal bedding.")
    energy, law = model.energy, model.law
    dq = q-path["observations"]["q_MPa"][0]
    volume = (dq/(3*wt.FIXED["grain_bulk"])+wt.FIXED["porosity"]
              *(1/wt.FIXED["fluid_bulk"]-1/wt.FIXED["grain_bulk"])*pressure)
    strain = initial["strain"]+np.array([axial, (volume-axial)/2, (volume-axial)/2, 0., 0., 0.])
    total = path["confinement"]*VOLUME+np.array([q, 0., 0., 0., 0., 0.])
    matrix = strain-energy.S0 @ (total-energy.biot*pressure)
    backstress_numerator = energy.Hc @ matrix
    a = backstress_numerator[0]-np.mean(backstress_numerator[1:3])
    b = VOLUME @ backstress_numerator/3
    mean = VOLUME @ total/3-pressure
    peak, mu = law.damage_peak_density, law.matrix_friction
    branches = [(mu, 0., 0., peak),
                (mu*law.friction_residual_fraction,
                 mu*(1-law.friction_residual_fraction)*peak, peak, np.inf)]
    densities = []
    for residual, transient, lower, upper in branches:
        polynomial = [q-law.matrix_cohesion-residual*mean,
                      -a-transient*mean+residual*b, transient*b]
        for root in np.roots(polynomial):
            if abs(root.imag) > 1e-10 or not lower < root.real <= upper:
                continue
            d = float(root.real)
            if q-a/d > 0 and mean-b/d > 0:
                densities.append(d)
    if len(densities) != 1:
        raise ValueError(f"Expected one positive compressive matrix branch, found {densities}.")
    density = densities[0]
    resistance = .5*matrix @ energy.Hc @ matrix/density**2
    return matrix, density, float(resistance), float(mean)


def normal_diagnosis(record, specimen, rows):
    path = wt.prepare_path(specimen, record["samples"])
    evolution = record["evolution_network"]
    model = wt.material(record["material_parameters"], path,
                        record["network"]["raw_parameters"], evolution["raw_parameters"],
                        evolution.get("input_size", 2), evolution.get("resistance_uses_pressure", True),
                        evolution.get("local_width", 0), evolution.get("fabric_knots", ()),
                        storage_form=record.get("storage_form", "neural"),
                        fabric_source=evolution.get("fabric_source", "plastic_stress"))
    initial = wt.initial_state(model, record["material_parameters"], path)
    history = specimen["history"]
    control = wt.recording_control(history["axial_strain"])
    peak = int(np.argmax(history["q_MPa"]))
    pressure_peak = int(np.argmax(history["pressure_MPa"]))
    stations = [.5*control[peak], .75*control[peak], .5*(control[peak]+control[-1])]
    indices = sorted({peak, pressure_peak,
                      *(int(np.argmin(abs(control-point))) for point in stations)})
    predicted_strain = np.array([row["axial_strain"] for row in rows])
    output, previous = [], None
    for i in indices:
        axial, q, pressure = control[i], history["q_MPa"][i], history["pressure_MPa"][i]
        matrix, density, resistance, mean = normal_matrix_state(model, initial, path, axial, q, pressure)
        force = path["confinement"]*VOLUME+np.array([q, 0., 0., 0., 0., 0.])-pressure*VOLUME
        if model.fabric_source == "matrix_force":
            force -= model.energy.Hc @ matrix/density
        learned = model.law.evolution(density, mean, model.fabric_direction(force)[0])
        beta = float("nan")
        if previous is not None:
            change = matrix-previous
            multiplier = 2*(change[0]-np.mean(change[1:3]))/3
            beta = float(-(VOLUME @ change)/multiplier)
        output.append(dict(source_row=int(history["source_row"][i]), axial_strain=float(axial),
            observed_q_MPa=float(q), observed_pressure_MPa=float(pressure),
            effective_mean_MPa=mean, conditioned_matrix_volume=float(VOLUME @ matrix),
            conditioned_density=density, conditioned_resistance_MPa=resistance,
            conditioned_interval_dilatancy=beta, closure_resistance_MPa=learned[0],
            closure_dilatancy=learned[3],
            **{f"forward_{key}": float(np.interp(axial, predicted_strain, [row[key] for row in rows]))
               for key in ("density", "damage_resistance_MPa", "matrix_dilatancy")}))
        previous = matrix
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specimens", nargs="+", help="Explicit subset of completed histories to diagnose.")
    args = parser.parse_args()
    record = json.loads(args.parameters.read_text(encoding="utf-8"))
    specimens = load_specimens(None, training_specimens=record["training_specimens"])
    if args.specimens:
        missing = set(args.specimens)-{specimen["specimen_id"] for specimen in specimens}
        if missing:
            parser.error(f"Unknown specimens: {sorted(missing)}")
        specimens = [specimen for specimen in specimens if specimen["specimen_id"] in args.specimens]
    stages, normal = [], None
    process_fields = ("density", "damage_resistance_MPa", "matrix_dilatancy",
                      "matrix_release_MPa", "plane_release_MPa", "effective_mean_MPa")
    for specimen in specimens:
        sid = specimen["specimen_id"]
        metadata = json.loads((args.traces/f"{sid}.json").read_text(encoding="utf-8"))
        rows = read_trace(args.traces/f"{sid}.csv.gz")
        strain = [row["axial_strain"] for row in rows]
        for interval in compare_stages(specimen, rows, metadata, metadata):
            result = dict(specimen_id=sid, role=specimen["role"], **interval,
                          **stage_projections(interval, rows, metadata))
            for key in process_fields:
                values = np.interp([interval["start_strain"], interval["end_strain"]],
                                   strain, [row[key] for row in rows])
                result.update({f"{endpoint}_{key}": float(value)
                               for endpoint, value in zip(("start", "end"), values)})
            stages.append(result)
        if sid == "OPA22":
            normal = normal_diagnosis(record, specimen, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/"loading_stages.csv", stages)
    normal_path = args.output/"OPA22_conditioned_evolution.csv"
    if normal is not None:
        write_csv(normal_path, normal)
    else:
        normal_path.unlink(missing_ok=True)
    print(f"Saved {len(stages)} observed intervals from {len(specimens)} histories and "
          f"{len(normal) if normal is not None else 0} conditional OPA22 states.")


if __name__ == "__main__":
    main()
