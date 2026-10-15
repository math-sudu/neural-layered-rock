"""Compare stored constitutive fits on the same observed loading intervals."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_winhausen_mechanisms import read_trace, write_csv
from analyze_winhausen_strength import observed_intervals
from tfenn_invcalib.winhausen_data import load_specimens
from tfenn_invcalib.winhausen_training import recording_control


def volume_coordinate(q, pressure, coefficient, modulus):
    """Pressure-equivalent irreversible volume relative to the sealed state."""
    q, pressure = np.asarray(q), np.asarray(pressure)
    return (pressure-pressure[0]-coefficient*(q-q[0]))/modulus


def volume_history(specimen, rows, metadata, reference):
    """Align conditional volume and internal volume at original acquisition rows.

    Each coordinate is cumulative from sealing. Consecutive coordinate changes
    are finite acquisition-interval increments, not instantaneous volume rates.
    The common reference is the joint fit of the supplied experiment.
    """
    history = specimen["history"]
    control = recording_control(history["axial_strain"])
    strain = np.array([row["axial_strain"] for row in rows])
    if control[0] < strain[0] or control[-1] > strain[-1]:
        raise ValueError(f"{specimen['specimen_id']}: observations exceed the saved path")
    coefficient, modulus = metadata["pressure_stress_coefficient"], metadata["pressure_modulus_MPa"]
    ref_coefficient, ref_modulus = reference["pressure_stress_coefficient"], reference["pressure_modulus_MPa"]
    predicted = {key: np.interp(control, strain, [row[key] for row in rows])
                 for key in ("q_MPa", "pressure_MPa", "matrix_volume", "plane_mode_0", "compaction")}
    initial = metadata["initial_state"]
    matrix = predicted["matrix_volume"]-sum(initial["matrix_strain"][:3])
    plane = predicted["plane_mode_0"]-initial["plane_modes"][0]
    cap = predicted["compaction"]-initial["compaction"]
    coordinates = dict(
        observed_volume_own_coefficients=volume_coordinate(
            history["q_MPa"], history["pressure_MPa"], coefficient, modulus),
        observed_volume_common_coefficients=volume_coordinate(
            history["q_MPa"], history["pressure_MPa"], ref_coefficient, ref_modulus),
        predicted_volume_own_coefficients=volume_coordinate(
            predicted["q_MPa"], predicted["pressure_MPa"], coefficient, modulus),
        predicted_volume_common_coefficients=volume_coordinate(
            predicted["q_MPa"], predicted["pressure_MPa"], ref_coefficient, ref_modulus),
        predicted_matrix_volume=matrix, predicted_plane_volume=plane,
        predicted_compaction=cap, predicted_total_inelastic_volume=matrix+plane+cap)
    increments = {f"{key}_increment": np.r_[0., np.diff(value)] for key, value in coordinates.items()
                  if key in ("observed_volume_common_coefficients", "predicted_matrix_volume")}
    return [dict(source_row=int(history["source_row"][i]),
        recorded_axial_strain=float(history["axial_strain"][i]), axial_strain=float(control[i]),
        observed_q_MPa=float(history["q_MPa"][i]), observed_pressure_MPa=float(history["pressure_MPa"][i]),
        predicted_q_MPa=float(predicted["q_MPa"][i]), predicted_pressure_MPa=float(predicted["pressure_MPa"][i]),
        pressure_stress_coefficient=coefficient, pressure_modulus_MPa=modulus,
        reference_pressure_stress_coefficient=ref_coefficient, reference_pressure_modulus_MPa=ref_modulus,
        **{key: float(value[i]) for key, value in {**coordinates, **increments}.items()})
        for i in range(len(control))]


def compare_stages(specimen, rows, metadata, reference):
    """Sample the accepted-step curve at observed, acquisition-ordered events.

    Endpoint brackets retain the interpolation scale. Repeated imposed strains
    use the last accepted state, as the full-path observation output does.
    Reversed or overlapping maximum ranges are retained with their status.
    """
    strain = np.array([row["axial_strain"] for row in rows])
    fields = ("q_MPa", "pressure_MPa", "matrix_volume", "matrix_axial", "plane_axial",
              "matrix_mean_force_MPa", "plane_normal_force_MPa")
    coefficient, modulus = (metadata["pressure_stress_coefficient"],
                            metadata["pressure_modulus_MPa"])
    ref_coefficient, ref_modulus = (reference["pressure_stress_coefficient"],
                                    reference["pressure_modulus_MPa"])
    history = specimen["history"]
    own_volume = volume_coordinate(history["q_MPa"], history["pressure_MPa"], coefficient, modulus)
    common_volume = volume_coordinate(history["q_MPa"], history["pressure_MPa"], ref_coefficient, ref_modulus)
    result = []
    for interval in observed_intervals(specimen):
        points = [interval["start_strain"], interval["end_strain"]]
        if min(points) < strain[0] or max(points) > strain[-1]:
            raise ValueError(f"{specimen['specimen_id']}: observed stage exceeds the saved path")
        values = {key: np.interp(points, strain, [row[key] for row in rows]) for key in fields}
        row = dict(interval)
        for label, point in zip(("start", "end"), points):
            left = max(0, np.searchsorted(strain, point, side="right")-1)
            right = min(len(strain)-1, np.searchsorted(strain, point, side="left"))
            row[f"{label}_bracket_width"] = float(strain[right]-strain[left])
        for key, pair in values.items():
            row[f"predicted_delta_{key}"] = float(pair[1]-pair[0])
        first, last = np.searchsorted(history["source_row"],
                                     [interval["start_source_row"], interval["end_source_row"]])
        row.update(pressure_stress_coefficient=coefficient, pressure_modulus_MPa=modulus,
            observed_volume_change_own_coefficients=float(own_volume[last]-own_volume[first]),
            reference_pressure_stress_coefficient=ref_coefficient,
            reference_pressure_modulus_MPa=ref_modulus,
            observed_volume_change_common_coefficients=float(common_volume[last]-common_volume[first]))
        result.append(row)
    return result


def compare(traces, output):
    specimens = {p["specimen_id"]: p for p in load_specimens(None)}
    summaries, stages, volumes = [], [], []
    for label, stress_directory, joint_directory in traces:
        for fit, directory in (("stress", stress_directory), ("joint", joint_directory)):
            for sid, specimen in specimens.items():
                source = directory/f"{sid}.json"
                metadata = json.loads(source.read_text(encoding="utf-8"))
                reference = json.loads((joint_directory/f"{sid}.json").read_text(encoding="utf-8"))
                identity = dict(experiment=label, fit=fit, specimen_id=sid,
                    role=metadata["role"], angle_deg=metadata["angle_deg"],
                    confinement_MPa=metadata["confinement_MPa"])
                rows = read_trace(source.with_suffix(".csv.gz"))
                stages.extend(dict(**identity, **row)
                              for row in compare_stages(specimen, rows, metadata, reference))
                volumes.extend(dict(**identity, **row)
                               for row in volume_history(specimen, rows, metadata, reference))
                metrics = metadata["metrics"]
                initial = metadata["initial_state"]
                summaries.append(dict(**identity,
                    q_rmse_MPa=metrics["q_MPa"]["rmse"],
                    pressure_rmse_MPa=metrics["pressure_MPa"]["rmse"],
                    predicted_q_peak_strain=metrics["q_peak"]["predicted_strain"],
                    observed_q_peak_strain=metrics["q_peak"]["observed_strain"],
                    predicted_pressure_peak_strain=metrics["pressure_peak"]["predicted_strain"],
                    observed_pressure_peak_strain=metrics["pressure_peak"]["observed_strain"],
                    q_at_observed_strength_peak_MPa=metrics["at_observed_q_peak"]["q_MPa"]["predicted_MPa"],
                    pressure_at_observed_strength_peak_MPa=metrics["at_observed_q_peak"]["pressure_MPa"]["predicted_MPa"],
                    net_matrix_volume=rows[-1]["matrix_volume"]-sum(initial["matrix_strain"][:3]),
                    end_density=rows[-1]["density"],
                    max_increment=metadata["max_increment"],
                    samples=metadata["samples"]))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"response_comparison.csv", summaries)
    write_csv(output/"observed_stage_comparison.csv", stages)
    with gzip.open(output/"volume_coordinates.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(volumes[0]))
        writer.writeheader()
        writer.writerows(volumes)
    return summaries, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs=2, action="append", default=[],
                        metavar=("LABEL", "DIRECTORY"))
    parser.add_argument("--pair", nargs=3, action="append", default=[],
                        metavar=("LABEL", "STRESS_DIRECTORY", "JOINT_DIRECTORY"),
                        help="Compare existing trace directories without copying their histories.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.traces and not args.pair:
        parser.error("Supply --traces or --pair.")
    pairs = [(label, Path(path)/"stress", Path(path)/"joint") for label, path in args.traces]
    pairs.extend((label, Path(stress), Path(joint)) for label, stress, joint in args.pair)
    summaries, stages = compare(pairs, args.output)
    print(f"Compared {len(summaries)} histories and {len(stages)} observed intervals")


if __name__ == "__main__":
    main()
