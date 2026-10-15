"""Replace learned plane storage by h(d)=d at fixed fitted parameters."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from trace_winhausen_mechanisms import trace_one
from analyze_winhausen_mechanisms import branch_event_partitions, read_trace, write_csv
from probe_winhausen_branches import probe
from tfenn_invcalib.winhausen_data import load_specimens
from tfenn_invcalib.winhausen_training import FIXED


def compare_history(baseline, candidate):
    original = json.loads(baseline.read_text(encoding="utf-8"))
    analytical = json.loads(candidate.read_text(encoding="utf-8"))
    old = read_trace(baseline.with_suffix(".csv.gz"))
    new = read_trace(candidate.with_suffix(".csv.gz"))
    x = np.array([row["axial_strain"] for row in old])
    y = np.array([row["axial_strain"] for row in new])
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal([step["storage_coordinate"] for step in new],
                                  [step["density"] for step in new])
    np.testing.assert_array_equal([step["storage_derivative"] for step in new], np.ones(len(new)))
    row = dict(fit=baseline.parent.name, specimen_id=original["specimen_id"],
               role=original["role"], angle_deg=original["angle_deg"],
               max_increment=original["max_increment"])
    for channel, prefix in (("q_MPa", "q"), ("pressure_MPa", "u")):
        change = np.array([b[channel]-a[channel] for a, b in zip(old, new)])
        row[f"{prefix}_max_abs_change_MPa"] = float(np.max(abs(change)))
        for label, metadata in (("neural", original), ("identity", analytical)):
            metrics = metadata["metrics"]
            row[f"{label}_{prefix}_rmse_MPa"] = metrics[channel]["rmse"]
            peak = metrics["q_peak" if prefix == "q" else "pressure_peak"]
            row[f"{label}_{prefix}_peak_MPa"] = peak["predicted_MPa"]
            row[f"{label}_{prefix}_peak_strain"] = peak["predicted_strain"]
        row[f"{prefix}_peak_shift_strain"] = (row[f"identity_{prefix}_peak_strain"]
                                              -row[f"neural_{prefix}_peak_strain"])
    for label, history in (("neural", old), ("identity", new)):
        row[f"{label}_end_density"] = history[-1]["density"]
        row[f"{label}_end_matrix_volume"] = history[-1]["matrix_volume"]
        row[f"{label}_peak_gap_strain"] = (row[f"{label}_q_peak_strain"]
                                           -row[f"{label}_u_peak_strain"])
    slope = np.array([step["storage_derivative"] for step in old])
    row["neural_min_storage_derivative"] = float(slope.min())
    row["neural_max_storage_derivative"] = float(slope.max())
    alpha = FIXED["matrix_slope"]
    row["max_fixed_state_stiffness_relative_change"] = float(max(
        abs((1+alpha*step["storage_coordinate"])/(1+alpha*step["density"])-1) for step in old))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    specimens = {p["specimen_id"]: p for p in load_specimens(None)}
    pairs, arguments = [], []
    for fit in ("joint", "stress"):
        output = args.output/fit
        output.mkdir(parents=True, exist_ok=True)
        for source in sorted((args.baseline/fit).glob("*.json")):
            metadata = json.loads(source.read_text(encoding="utf-8"))
            pairs.append((source, output/source.name))
            arguments.append((Path(metadata["parameters"]), specimens[metadata["specimen_id"]],
                metadata["samples"], metadata["max_increment"], output,
                metadata.get("max_log_density_increment", 0.),
                metadata.get("min_increment", 1e-9), "identity"))
    if not args.analyze_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(trace_one, arguments):
                print(json.dumps(result), flush=True)
    rows = [compare_history(*pair) for pair in pairs]
    write_csv(args.output/"comparison.csv", rows)
    # These are the baseline's already identified finite transitions. Check
    # their changed endpoints before excluding them from candidate rate budgets.
    with (args.baseline/"branch_event_partitions.csv").open(encoding="utf-8", newline="") as stream:
        baseline_events = list(csv.DictReader(stream))
    events = []
    for event in baseline_events:
        fit, sid = event["fit"], event["specimen_id"]
        trace = args.output/fit/sid
        branch_folder = args.output/(fit+"-branches")
        branch_folder.mkdir(exist_ok=True)
        if not args.analyze_only:
            probe(trace, branch_folder)
        metadata = json.loads(trace.with_suffix(".json").read_text(encoding="utf-8"))
        events.append(dict(fit=fit, specimen_id=sid, role=metadata["role"],
            angle_deg=metadata["angle_deg"], confinement_MPa=metadata["confinement_MPa"],
            **branch_event_partitions(read_trace(trace.with_suffix(".csv.gz")), metadata,
                                      branch_folder/(sid+".csv"))))
    write_csv(args.output/"branch_event_partitions.csv", events)
    subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_winhausen_strength.py")),
                    "--traces", str(args.output)], check=True)
    for fit in ("joint", "stress"):
        selected = [row for row in rows if row["fit"] == fit]
        print(json.dumps(dict(fit=fit, histories=len(selected),
            max_q_change_MPa=max(row["q_max_abs_change_MPa"] for row in selected),
            max_u_change_MPa=max(row["u_max_abs_change_MPa"] for row in selected),
            max_q_peak_shift=max(abs(row["q_peak_shift_strain"]) for row in selected),
            max_u_peak_shift=max(abs(row["u_peak_shift_strain"]) for row in selected))), flush=True)


if __name__ == "__main__":
    main()
