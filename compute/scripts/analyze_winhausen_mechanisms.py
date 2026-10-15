"""Extract activity intervals and axial partitions from accepted-step traces."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute/src"))
from tfenn_invcalib.energetic_coupling import plane_basis
from tfenn_invcalib.winhausen_data import load_specimens
from tfenn_invcalib.winhausen_training import prepare_path

# Suppress solver roundoff when naming activity; these are not fit criteria.
MULTIPLIER_ZERO = 1e-10
VOLUME_ZERO = 1e-12


def read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return [{key: value if key == "solver_route" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def loading_groups(rows):
    """Combine repeated imposed strains without discarding their state changes."""
    groups = []
    summed = ("axial_increment", "matrix_multiplier", "plane_multiplier",
              "delta_matrix_volume", "delta_matrix_axial", "delta_plane_axial",
              "delta_density", "delta_q_MPa")
    for i, row in enumerate(rows):
        if groups and abs(row["axial_strain"] - groups[-1]["axial_strain"]) <= 1e-14:
            previous = groups.pop()
            group = {**row, **{key: previous[key] + row[key] for key in summed},
                     "first_row": previous["first_row"], "last_row": i}
        else:
            group = {**row, "first_row": i, "last_row": i}
        groups.append(group)
    return [group for group in groups if group["axial_increment"] > 1e-14]


def activity(group):
    """Actual multipliers and signed volume increments determine activity."""
    volume = group["delta_matrix_volume"]
    return dict(matrix_flow="active" if group["matrix_multiplier"] > MULTIPLIER_ZERO else "inactive",
                plane_slip="active" if group["plane_multiplier"] > MULTIPLIER_ZERO else "inactive",
                matrix_volume="contraction" if volume > VOLUME_ZERO else
                              "dilation" if volume < -VOLUME_ZERO else "neutral")


def activity_intervals(rows):
    groups = loading_groups(rows)
    states = [activity(group) for group in groups]
    records = []
    for mechanism in ("matrix_flow", "plane_slip", "matrix_volume"):
        first = 0
        for stop in range(1, len(groups) + 1):
            if stop < len(groups) and states[stop][mechanism] == states[first][mechanism]:
                continue
            segment = groups[first:stop]
            start = segment[0]["axial_strain"] - segment[0]["axial_increment"]
            end = segment[-1]["axial_strain"]
            volume = sum(row["delta_matrix_volume"] for row in segment)
            records.append(dict(mechanism=mechanism, activity=states[first][mechanism],
                start_strain=start, first_step_end_strain=segment[0]["axial_strain"],
                end_strain=end, first_step=rows[segment[0]["first_row"]]["step"],
                last_step=rows[segment[-1]["last_row"]]["step"],
                delta_matrix_axial=sum(row["delta_matrix_axial"] for row in segment),
                delta_plane_axial=sum(row["delta_plane_axial"] for row in segment),
                delta_matrix_volume=volume, mean_matrix_volume_rate=volume/(end-start)))
            first = stop
    return records


def partition(row, metadata):
    """Axial increments since sealing, with the elastic preload subtracted."""
    initial = metadata["initial_state"]
    theta = np.deg2rad(metadata["angle_deg"])
    basis = plane_basis([np.sin(theta), 0., np.cos(theta)])
    matrix = row["matrix_axial"] - initial["matrix_strain"][0]
    plane = row["plane_axial"] - basis[0] @ initial["plane_modes"]
    cap = (row["compaction"] - initial["compaction"])/3
    total = row["strain_0"] - initial["strain"][0]
    return dict(axial_strain=row["axial_strain"], total_axial=total,
                elastic_axial=total-matrix-plane-cap, matrix_axial=matrix,
                plane_axial=plane, cap_axial=cap,
                matrix_volume=row["matrix_volume"] - sum(initial["matrix_strain"][:3]))


def stage_partitions(rows, metadata, path):
    strain = np.array([row["axial_strain"] for row in rows])
    observed_peak = int(np.argmax(path["observations"]["q_MPa"]))
    target = path["control_strain"][observed_peak]
    matches = np.flatnonzero(abs(strain-target) < 1e-12)
    if not len(matches):
        raise ValueError(f"{metadata['specimen_id']}: observed peak station absent from trace")
    stages = dict(observed_q_peak=int(matches[-1]),
                  predicted_q_peak=int(np.argmax([row["q_MPa"] for row in rows])),
                  predicted_pressure_peak=int(np.argmax([row["pressure_MPa"] for row in rows])),
                  end=len(rows)-1)
    records = []
    groups = loading_groups(rows)
    for stage, index in stages.items():
        row = rows[index]
        group = next(group for group in groups if group["last_row"] >= index)
        records.append(dict(stage=stage, step=row["step"],
            observed_q_peak_strain=path["observations"]["axial_strain"][observed_peak],
            **partition(row, metadata), q_MPa=row["q_MPa"], pressure_MPa=row["pressure_MPa"],
            density=row["density"],
            **{f"{key}_activity": value for key, value in activity(group).items()},
            matrix_volume_rate=group["delta_matrix_volume"]/group["axial_increment"],
            matrix_axial_rate=group["delta_matrix_axial"]/group["axial_increment"],
            plane_axial_rate=group["delta_plane_axial"]/group["axial_increment"],
            matrix_dilatancy=row["matrix_dilatancy"],
            matrix_flow_fraction=row["matrix_flow_fraction"],
            **{key: row[key] for key in ("pressure_stress_MPa", "pressure_matrix_MPa",
                                        "pressure_plane_MPa", "pressure_compaction_MPa")}))
    return records


def branch_event_partitions(rows, metadata, branch_file):
    """Keep the probed finite increments separate from continuous-branch rates."""
    with branch_file.open(encoding="utf-8", newline="") as stream:
        probes = list(csv.DictReader(stream))
    subdivisions = [row for row in probes if row["kind"] == "subdivisions"]
    first = next(row for row in subdivisions if float(row["value"]) == 1)
    refined = max(subdivisions, key=lambda row: float(row["value"]))
    index = int(metadata["largest_damage_step"]["step"])
    row, previous = rows[index], rows[index-1]
    if abs(row["axial_strain"] - float(first["axial_strain"])) > 1e-12:
        raise ValueError(f"{branch_file}: probe and trace refer to different steps")
    before, after = partition(previous, metadata), partition(row, metadata)
    return dict(step=row["step"], strain_before=previous["axial_strain"],
        strain_after=row["axial_strain"], solver_route=row["solver_route"],
        delta_q_MPa=row["delta_q_MPa"], delta_pressure_MPa=row["pressure_MPa"]-previous["pressure_MPa"],
        delta_density=row["delta_density"],
        refined_subdivisions=float(refined["value"]),
        refined_max_damage_increment=float(refined["max_damage_increment"]),
        **{f"delta_{key}": after[key]-before[key] for key in
           ("elastic_axial", "matrix_axial", "plane_axial", "cap_axial", "matrix_volume")})


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specimens = {p["specimen_id"]: p for p in load_specimens(None)}
    intervals, stages, events = [], [], []
    for fit, folder in (("joint", args.joint), ("stress", args.stress)):
        for sid, specimen in specimens.items():
            metadata = json.loads((folder / f"{sid}.json").read_text(encoding="utf-8"))
            rows = read_trace(folder / f"{sid}.csv.gz")
            identity = dict(fit=fit, specimen_id=sid, role=metadata["role"],
                            angle_deg=metadata["angle_deg"], confinement_MPa=metadata["confinement_MPa"])
            path = prepare_path(specimen, metadata["samples"])
            intervals.extend({**identity, **record} for record in activity_intervals(rows))
            stages.extend({**identity, **record} for record in stage_partitions(rows, metadata, path))
            branch = folder.with_name(folder.name + "-branches") / f"{sid}.csv"
            if branch.exists():
                events.append({**identity, **branch_event_partitions(rows, metadata, branch)})
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("activity_intervals.csv", intervals),
                              ("stage_partitions.csv", stages), ("branch_event_partitions.csv", events)):
        if records:
            write_csv(args.output / filename, records)
            print(f"{filename}: {len(records)} records")


if __name__ == "__main__":
    main()
