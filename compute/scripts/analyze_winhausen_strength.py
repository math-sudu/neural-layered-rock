"""Resolve continuous sliding strength and finite jumps from saved mechanisms.

The symmetric product increment k_bar*dz + z_bar*dk equals d(k*z).
It approximates the two path integrals on continuous accepted branches;
finite jumps retain endpoint changes without assigning an interpolated path.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_winhausen_mechanisms import (loading_groups, read_trace, write_csv,
                                         MULTIPLIER_ZERO)
from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.winhausen_data import load_specimens
from tfenn_invcalib.winhausen_training import material, recording_control


def product_increments(k_before, k_after, z_before, z_after):
    return ((k_before+k_after)/2*(z_after-z_before),
            (z_before+z_after)/2*(k_after-k_before))


def normal_matrix_increment(before, after, stiffness, pressure_modulus, pressure_coefficient):
    """Resolve two compressive matrix-sliding states normal to bedding.

    Here q_A=q-Q_H/d and p_A=p'-P_H/d, with Q_H=(Hc c)_1-(Hc c)_3
    and P_H=tr(Hc c)/3. Difference q_A=c+mu*p_A, use the symmetric
    product identity, and eliminate du=B_q*dq+M_Sigma*dv. The returned
    terms are matrix-strain growth, damage, volume and friction evolution.
    Path-integral use requires a continuous branch, as for plane sliding.
    """
    ca, cb = [np.array([row[f"matrix_strain_{i}"] for i in range(6)])
              for row in (before, after)]
    ha, hb = stiffness @ ca, stiffness @ cb
    qa, qb = ha[0]-ha[2], hb[0]-hb[2]
    pa, pb = VOLUME @ ha/3, VOLUME @ hb/3
    ka, kb = 1/before["density"], 1/after["density"]
    mua, mub = before["matrix_friction"], after["matrix_friction"]
    mu = (mua+mub)/2
    denominator = 1-mu/3+mu*pressure_coefficient
    return np.array([
        (ka+kb)/2*((qb-qa)-mu*(pb-pa)),
        ((qa+qb)/2-mu*(pa+pb)/2)*(kb-ka),
        -mu*pressure_modulus*(after["matrix_volume"]-before["matrix_volume"]),
        (before["matrix_mean_force_MPa"]+after["matrix_mean_force_MPa"])/2*(mub-mua),
    ])/denominator


def classify_step(current, previous, coefficient, jump):
    if jump:
        return "finite_jump"
    if current["open_contact"]:
        return "open_contact"
    if coefficient <= 0:
        return "no_plane_strength"
    if current["plane_multiplier"] <= MULTIPLIER_ZERO:
        return "non_sliding"
    if not current["sliding_balance"]:
        raise ValueError("Active plane step does not satisfy closed forward sliding")
    return "continuous_sliding" if previous["sliding_balance"] else "sliding_entry"


def strength_steps(rows, metadata, model, jump_steps):
    groups = loading_groups(rows)
    energy = model.energy
    angle = np.deg2rad(metadata["angle_deg"])
    normal_loading = abs(metadata["angle_deg"]-90.) < 1e-10
    mu = model.law.plane_friction
    ctheta = np.sin(angle)*np.cos(angle)-mu*np.sin(angle)**2
    bq, modulus = metadata["pressure_stress_coefficient"], metadata["pressure_modulus_MPa"]
    model_modulus = 1/(1/energy.M + energy.biot @ energy.S0 @ energy.biot)
    np.testing.assert_allclose([modulus, bq],
        [model_modulus, model_modulus*(energy.biot @ energy.S0)[0]], rtol=1e-12)
    coordinates = energy.network.value_derivatives(np.array([row["density"] for row in rows]))[0]
    np.testing.assert_allclose(coordinates, [row["storage_coordinate"] for row in rows], rtol=1e-12)
    denominator = ctheta+mu*bq
    initial = metadata["initial_state"]
    # In a TI host both tangential storage eigenvalues coincide.
    h0 = energy.plane_H0
    np.testing.assert_allclose(h0, np.diag([h0[0, 0], h0[1, 1], h0[1, 1]]), atol=1e-9)
    k0 = h0[1, 1]/(np.sqrt(2)*model.D)
    previous = dict(axial_strain=0., slip=np.linalg.norm(initial["plane_modes"][1:]),
        stiffness=k0/(1+energy.matrix_slope*initial["d"]),
        q_MPa=rows[0]["q_MPa"]-rows[0]["delta_q_MPa"],
        pressure_MPa=rows[0]["pressure_MPa"]-sum(rows[0][key] for key in
            ("pressure_stress_MPa", "pressure_matrix_MPa", "pressure_plane_MPa",
             "pressure_compaction_MPa")), pressure_plane_MPa=0.,
        pressure_compaction_MPa=0., sliding_balance=False, matrix_sliding_balance=False)
    records, segment = [], -1
    previous_kind = None
    for group in groups:
        slip = np.hypot(group["plane_mode_1"], group["plane_mode_2"])
        stiffness = k0/(1+energy.matrix_slope*group["storage_coordinate"])
        backstress = np.hypot(group["plane_backstress_MPa_1"],
                              group["plane_backstress_MPa_2"])/np.sqrt(2)
        np.testing.assert_allclose(stiffness*slip, backstress, atol=1e-9)
        closed = abs(group["plane_mode_0"]) < 1e-10 and group["plane_normal_force_MPa"] > 0
        forward = group["plane_traction_MPa_1"]*group["plane_mode_1"] >= -1e-12
        sliding = closed and forward and abs(group["plane_yield_MPa"]) < 1e-7
        matrix_sliding = (normal_loading and abs(group["matrix_yield_MPa"]) < 1e-7
                          and group["matrix_mean_force_MPa"] > 0)
        current = dict(group, slip=slip, stiffness=stiffness, sliding_balance=sliding,
                       matrix_sliding_balance=matrix_sliding,
                       open_contact=not closed and abs(group["plane_normal_force_MPa"]) < 1e-7)
        jump = any(group["first_row"] <= step <= group["last_row"] for step in jump_steps)
        kind = classify_step(current, previous, ctheta, jump)
        dq = group["q_MPa"]-previous["q_MPa"]
        du = group["pressure_MPa"]-previous["pressure_MPa"]
        volume_pressure = modulus*group["delta_matrix_volume"]
        other_volume = sum(group[key]-previous[key] for key in
                           ("pressure_plane_MPa", "pressure_compaction_MPa"))
        np.testing.assert_allclose(du, bq*dq+volume_pressure+other_volume, atol=1e-9)
        if abs(other_volume/modulus) > 1e-12 and kind != "finite_jump":
            kind = "other_volume_active"
        slip_term = damage_term = volume_term = backstress_term = float("nan")
        matrix_terms = np.full(4, np.nan)
        if ctheta > 0 and sliding and previous["sliding_balance"] and abs(other_volume/modulus) <= 1e-12:
            backstress_term = (backstress-previous["stiffness"]*previous["slip"])/denominator
            volume_term = -mu*volume_pressure/denominator
            np.testing.assert_allclose(backstress_term+volume_term, dq, atol=1e-7)
        if kind == "continuous_sliding":
            slip_term, damage_term = np.array(product_increments(
                previous["stiffness"], stiffness, previous["slip"], slip))/denominator
            np.testing.assert_allclose(slip_term+damage_term+volume_term, dq, atol=1e-7)
        if (matrix_sliding and previous["matrix_sliding_balance"] and not jump
                and group["matrix_multiplier"] > MULTIPLIER_ZERO
                and abs(other_volume/modulus) <= 1e-12):
            matrix_terms = normal_matrix_increment(previous, current, energy.Hc, modulus, bq)
            np.testing.assert_allclose(matrix_terms.sum(), dq, atol=1e-7)
            kind = "continuous_matrix_sliding"
        continuous = kind in ("continuous_sliding", "continuous_matrix_sliding")
        if continuous and previous_kind != kind:
            segment += 1
        records.append(dict(step=int(group["step"]), strain_before=previous["axial_strain"],
            axial_strain=group["axial_strain"], kind=kind,
            segment=segment if continuous else -1,
            q_MPa=group["q_MPa"], pressure_MPa=group["pressure_MPa"],
            density=group["density"], matrix_volume=group["matrix_volume"],
            delta_q_MPa=dq, delta_pressure_MPa=du,
            pressure_stress_increment_MPa=bq*dq,
            pressure_matrix_increment_MPa=volume_pressure,
            pressure_other_volume_increment_MPa=other_volume,
            slip_contribution_MPa=slip_term, damage_contribution_MPa=damage_term,
            volume_contribution_MPa=volume_term, backstress_contribution_MPa=backstress_term,
            matrix_strain_contribution_MPa=matrix_terms[0],
            matrix_damage_contribution_MPa=matrix_terms[1],
            matrix_volume_contribution_MPa=matrix_terms[2],
            friction_evolution_contribution_MPa=matrix_terms[3]))
        previous, previous_kind = current, kind
    return records


def peak_event(strain, values, kinds=None):
    """Locate an attained maximum, retaining repeated readings and boundaries.

    No smoothing or tolerance band is imposed on the measured channel.
    Repeated, separated maximum readings are not treated as a flat plateau.
    The immediate neighbors bracket a continuous model turning point; a
    finite jump touching the maximum is identified instead of differentiated.
    """
    strain, values = np.asarray(strain), np.asarray(values)
    indices = np.flatnonzero(values == np.max(values))
    first, last = int(indices[0]), int(indices[-1])
    if first == 0 or last == len(values)-1:
        kind = "boundary_maximum"
    elif kinds is not None and "finite_jump" in (kinds[first], kinds[last+1]):
        kind = "jump_maximum"
    elif len(indices) == 1:
        kind = "interior_turn"
    elif np.all(np.diff(indices) == 1):
        kind = "flat_maximum"
    else:
        kind = "repeated_maximum"
    return dict(kind=kind, first_index=first, last_index=last,
                maximum_count=len(indices), maximum_MPa=float(values[first]),
                strain_first=float(strain[first]), strain_last=float(strain[last]),
                strain_min=float(np.min(strain[indices])),
                strain_max=float(np.max(strain[indices])),
                bracket_start=float(strain[max(0, first-1)]),
                bracket_end=float(strain[min(len(values)-1, last+1)]))


def turning_rates(backstress_rate, volume_pressure_reduction_rate, bq, ctheta, mu):
    """Return (q_a, u_a) on a closed, continuous forward-sliding branch."""
    denominator = ctheta+mu*bq
    h, v = backstress_rate, volume_pressure_reduction_rate
    return ((h+mu*v)/denominator, (bq*h-ctheta*v)/denominator)


def orientation_coefficients(model):
    """TI sign changes of stress contributions at the actual prepared stiffness."""
    m = model.energy
    modulus = 1/(1/m.M+m.biot @ m.S0 @ m.biot)
    projection = modulus*(m.biot @ m.S0)
    normal = float(projection @ m.B[:, 0])
    parallel = float((projection @ VOLUME-normal)/2)
    difference = normal-parallel
    roundoff = 8*np.finfo(float).eps*max(1., abs(normal), abs(parallel))
    result = dict(pressure_coefficient_parallel=parallel, pressure_coefficient_normal=normal)
    for name, offset, slope in (("effective_mean", 1/3-parallel, -difference),
                                ("plane_reaction", -parallel, 1-difference)):
        angle = float("nan")
        if abs(slope) <= roundoff:
            status = "all_orientations" if abs(offset) <= roundoff else "no_root"
        else:
            square = -offset/slope
            status = "root" if -roundoff <= square <= 1.+roundoff else "no_root"
            if status == "root":
                angle = float(np.rad2deg(np.arcsin(np.sqrt(np.clip(square, 0., 1.)))))
        result[f"{name}_stress_switch_angle_deg"] = angle
        result[f"{name}_stress_switch_status"] = status
    return result


def stage_projections(interval, rows, metadata):
    """Resolve mean stress and plane reaction over the same observed window.

    The plane reaction uses its actual normal projection and energetic
    backstress. Matrix mean force is read independently from the saved state.
    All contributions are dimensional; no small net increment is a divisor.
    """
    points = [interval["start_strain"], interval["end_strain"]]
    strain = [row["axial_strain"] for row in rows]
    fields = ("effective_mean_MPa", "matrix_mean_force_MPa", "plane_normal_force_MPa",
              "plane_backstress_MPa_0", "pressure_matrix_MPa", "pressure_plane_MPa",
              "pressure_compaction_MPa")
    values = {key: np.interp(points, strain, [row[key] for row in rows]) for key in fields}
    change = {key: float(pair[1]-pair[0]) for key, pair in values.items()}
    dq = interval["predicted_delta_q_MPa"]
    bq = metadata["pressure_stress_coefficient"]
    normal_projection = np.sin(np.deg2rad(metadata["angle_deg"]))**2
    result = {f"{endpoint}_{key}": float(value)
              for key in fields[:3] for endpoint, value in zip(("start", "end"), values[key])}
    result.update(
        delta_effective_mean_MPa=change["effective_mean_MPa"],
        effective_mean_stress_contribution_MPa=(1/3-bq)*dq,
        delta_plane_reaction_MPa=change["plane_normal_force_MPa"],
        plane_reaction_stress_contribution_MPa=(normal_projection-bq)*dq,
        plane_reaction_backstress_contribution_MPa=-change["plane_backstress_MPa_0"],
        delta_matrix_mean_force_MPa=change["matrix_mean_force_MPa"])
    for name in ("matrix", "plane", "compaction"):
        for force in ("effective_mean", "plane_reaction"):
            result[f"{force}_{name}_volume_contribution_MPa"] = -change[f"pressure_{name}_MPa"]
    for force in ("effective_mean", "plane_reaction"):
        total = sum(value for key, value in result.items()
                    if key.startswith(force+"_") and key.endswith("_contribution_MPa"))
        np.testing.assert_allclose(total, result[f"delta_{force}_MPa"], atol=1e-9, rtol=0.)
    return result


def observed_intervals(specimen):
    history = specimen["history"]
    control = recording_control(history["axial_strain"])
    peaks = {channel: peak_event(control, history[channel])
             for channel in ("q_MPa", "pressure_MPa")}
    pressure, strength = peaks["pressure_MPa"], peaks["q_MPa"]
    overlap = (max(pressure["strain_min"], strength["strain_min"])
               <= min(pressure["strain_max"], strength["strain_max"]))
    intervals = []
    for name, first in (("first_pressure_to_strength", pressure["first_index"]),
                        ("last_pressure_to_strength", pressure["last_index"])):
        last = strength["first_index"]
        start, end = float(control[first]), float(control[last])
        intervals.append(dict(window=name,
            status=("overlapping_maximum_ranges" if overlap else
                    "pressure_before_strength" if end > start else "strength_before_pressure"),
            start_source_row=int(history["source_row"][first]),
            end_source_row=int(history["source_row"][last]),
            start_strain=start, end_strain=end,
            pressure_peak_first_strain=pressure["strain_first"],
            pressure_peak_last_strain=pressure["strain_last"],
            strength_peak_first_strain=strength["strain_first"],
            strength_peak_last_strain=strength["strain_last"],
            observed_delta_q_MPa=float(history["q_MPa"][last]-history["q_MPa"][first]),
            observed_delta_pressure_MPa=float(history["pressure_MPa"][last]-history["pressure_MPa"][first])))
    return intervals


def interval_budget(records, first, last):
    """Integrate accepted increments in loading order between two states.

    Slip/damage integrals use only continuous plane sliding. Jumps and all
    remaining branches retain their actual stress increments separately.
    Pressure balance is valid on every branch, including symmetry directions.
    """
    if last < first:
        raise ValueError("An event interval must follow loading order")
    interval = records[first+1:last+1]
    plane = [row for row in interval if row["kind"] == "continuous_sliding"]
    jumps = [row for row in interval if row["kind"] == "finite_jump"]
    other = [row for row in interval
             if row["kind"] not in ("continuous_sliding", "finite_jump")]
    result = dict(start_strain=records[first]["axial_strain"],
                  end_strain=records[last]["axial_strain"],
                  first_step=records[first]["step"], last_step=records[last]["step"],
                  continuous_plane_steps=len(plane), jump_steps=len(jumps),
                  other_steps=len(other))
    for key in ("delta_q_MPa", "delta_pressure_MPa", "pressure_stress_increment_MPa",
                "pressure_matrix_increment_MPa", "pressure_other_volume_increment_MPa"):
        result[key] = sum(row[key] for row in interval)
    for key in ("slip_contribution_MPa", "damage_contribution_MPa",
                "volume_contribution_MPa"):
        result[key] = sum(row[key] for row in plane) if plane else float("nan")
    for name, subset in (("continuous_plane", plane), ("jump", jumps), ("other", other)):
        result[f"{name}_delta_q_MPa"] = sum(row["delta_q_MPa"] for row in subset)
    np.testing.assert_allclose(result["delta_q_MPa"],
                              records[last]["q_MPa"]-records[first]["q_MPa"], atol=1e-9)
    np.testing.assert_allclose(result["delta_pressure_MPa"],
        [records[last]["pressure_MPa"]-records[first]["pressure_MPa"],
         sum(result[key] for key in ("pressure_stress_increment_MPa",
             "pressure_matrix_increment_MPa", "pressure_other_volume_increment_MPa"))], atol=1e-9)
    if plane:
        np.testing.assert_allclose(sum(result[key] for key in
            ("slip_contribution_MPa", "damage_contribution_MPa", "volume_contribution_MPa")),
            result["continuous_plane_delta_q_MPa"], atol=1e-7)
    return result


def window_records(records, start, end):
    """Integrate at observed strains using accepted-step secants at the edges.

    A complete finite jump remains a separate endpoint change. A boundary
    inside a jump cannot define a fractional physical jump and is rejected.
    Reversed windows retain signed changes; their observed event status is
    carried by the caller, including overlapping maximum ranges.
    """
    left, right = sorted((start, end))
    strain = [row["axial_strain"] for row in records]
    if left < strain[0] or right > strain[-1]:
        raise ValueError("The observed window exceeds the saved strength history")
    first = dict(step=-1, axial_strain=left,
                 **{key: float(np.interp(left, strain, [row[key] for row in records]))
                    for key in ("q_MPa", "pressure_MPa")})
    clipped = [first]
    for before, after in zip(records[:-1], records[1:]):
        a, b = before["axial_strain"], after["axial_strain"]
        lo, hi = max(left, a), min(right, b)
        if hi <= lo:
            continue
        fraction = (hi-lo)/(b-a)
        if after["kind"] == "finite_jump" and fraction < 1.:
            raise ValueError("An observed-window boundary intersects a finite jump")
        row = {key: value*fraction if key.endswith("_MPa") and key not in
               ("q_MPa", "pressure_MPa") else value for key, value in after.items()}
        row.update(axial_strain=hi, strain_before=lo,
                   **{key: before[key]+(hi-a)/(b-a)*(after[key]-before[key])
                      for key in ("q_MPa", "pressure_MPa")})
        clipped.append(row)
    return clipped


def window_budget(records, start, end):
    clipped = window_records(records, start, end)
    result = interval_budget(clipped, 0, len(clipped)-1)
    if end < start:
        result = {key: -value if key.endswith("_MPa") else value for key, value in result.items()}
    result.update(start_strain=start, end_strain=end)
    return result


def turning_analysis(records, metadata, model):
    strain = [row["axial_strain"] for row in records]
    kinds = [row["kind"] for row in records]
    peaks = {channel: peak_event(strain, [row[channel] for row in records], kinds)
             for channel in ("pressure_MPa", "q_MPa")}
    ui, qi = (peaks[channel]["first_index"] for channel in ("pressure_MPa", "q_MPa"))
    first, last = sorted((ui, qi))
    mu = model.law.plane_friction
    theta = np.deg2rad(metadata["angle_deg"])
    ctheta = np.sin(theta)*np.cos(theta)-mu*np.sin(theta)**2
    bq = metadata["pressure_stress_coefficient"]
    denominator = ctheta+mu*bq
    budget = dict(event_order="pressure_then_strength" if ui < qi else
                  "strength_then_pressure" if qi < ui else "coincident",
                  pressure_peak_strain=strain[ui], strength_peak_strain=strain[qi],
                  pressure_stress_coefficient=bq, plane_coefficient=ctheta,
                  plane_friction=mu, strength_denominator=denominator,
                  **interval_budget(records, first, last))
    # Use actual secants on each continuous accepted interval. No rate crosses
    # a jump, entry or inactive interval, and no extra history export is needed.
    supported, active = [], []
    for index, row in enumerate(records):
        applies = row["kind"] == "continuous_sliding"
        if applies:
            da = row["axial_strain"]-row["strain_before"]
            h = denominator*(row["slip_contribution_MPa"]+
                             row["damage_contribution_MPa"])/da
            v = -row["pressure_matrix_increment_MPa"]/da
            qr, ur = turning_rates(h, v, bq, ctheta, mu)
            np.testing.assert_allclose([qr*da, ur*da],
                [row["delta_q_MPa"], row["delta_pressure_MPa"]], atol=1e-7)
            applies = h < 0 and v > 0 and qr > 0 and ur < 0
        if applies:
            active.append(index)
        if active and (not applies or index == len(records)-1):
            start, end = active[0], active[-1]
            supported.append(interval_budget(records, start-1, end))
            active = []
    return [{"channel": key, **value} for key, value in peaks.items()], budget, supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, type=Path, metavar=("STRESS", "JOINT"),
                        help="Read an existing adaptive trace pair without copying histories.")
    args = parser.parse_args()
    if args.pair:
        directories = dict(zip(("stress", "joint"), args.pair))
        events = []
    else:
        directories = {fit: args.traces/fit for fit in ("stress", "joint")}
        with (args.traces/"branch_event_partitions.csv").open(encoding="utf-8", newline="") as stream:
            events = list(csv.DictReader(stream))
    records, peak_records, budgets, supported_intervals, observed_budgets = [], [], [], [], []
    specimens = {p["specimen_id"]: p for p in load_specimens(None)}
    for specimen in specimens.values():
        history = specimen["history"]
        identity = dict(fit="observed", specimen_id=specimen["specimen_id"],
                        angle_deg=specimen["angle"])
        for channel in ("pressure_MPa", "q_MPa"):
            peak_records.append(dict(**identity, channel=channel,
                **peak_event(history["axial_strain"], history[channel])))
    for fit in ("joint", "stress"):
        for source in sorted(directories[fit].glob("OPA*.json")):
            metadata = json.loads(source.read_text(encoding="utf-8"))
            sid = metadata["specimen_id"]
            checkpoint = json.loads(Path(metadata["parameters"]).read_text(encoding="utf-8"))
            model = material(checkpoint["material_parameters"],
                dict(angle=metadata["angle_deg"], confinement=metadata["confinement_MPa"]),
                np.array(checkpoint["network"]["raw_parameters"]),
                np.array(checkpoint["evolution_network"]["raw_parameters"]),
                checkpoint["evolution_network"].get("input_size", 2),
                checkpoint["evolution_network"].get("resistance_uses_pressure", True),
                checkpoint["evolution_network"].get("local_width", 0),
                checkpoint["evolution_network"].get("fabric_knots", ()),
                storage_form=metadata.get("storage_form", "neural"),
                fabric_source=checkpoint["evolution_network"].get("fabric_source", "plastic_stress"))
            rows = read_trace(source.with_suffix(".csv.gz"))
            jumps = {int(float(event["step"])) for event in events
                     if event["fit"] == fit and event["specimen_id"] == sid}
            if args.pair:
                # The adaptive replay resolves smooth damage increments down to
                # min_increment. A retained density step beyond its refinement
                # limit at that floor is an unresolved finite transition; do
                # not assign continuous rates to that accepted endpoint change.
                limit = metadata["max_log_density_increment"]
                if limit <= 0:
                    raise ValueError("Paired strength analysis needs adaptive trace metadata")
                jumps = {int(row["step"]) for row in rows
                         if row["log_density_increment"] > limit
                         and row["axial_increment"] <= metadata["min_increment"]}
            result = strength_steps(rows, metadata, model, jumps)
            identity = dict(fit=fit, specimen_id=sid, angle_deg=metadata["angle_deg"])
            records.extend({**identity, **row} for row in result)
            peaks, budget, supported = turning_analysis(result, metadata, model)
            peak_records.extend({**identity, **row} for row in peaks)
            budgets.append({**identity, **budget})
            supported_intervals.extend({**identity, **row} for row in supported)
            strain = [row["axial_strain"] for row in rows]
            coefficients = orientation_coefficients(model)
            theta = np.deg2rad(metadata["angle_deg"])
            mu = model.law.plane_friction
            ctheta = np.sin(theta)*np.cos(theta)-mu*np.sin(theta)**2
            bq = metadata["pressure_stress_coefficient"]
            backstress = [np.hypot(row["plane_backstress_MPa_1"],
                                  row["plane_backstress_MPa_2"])/np.sqrt(2) for row in rows]
            for interval in observed_intervals(specimens[sid]):
                start, end = interval["start_strain"], interval["end_strain"]
                common = window_budget(result, start, end)
                projection = stage_projections(
                    {**interval, "predicted_delta_q_MPa": common["delta_q_MPa"]}, rows, metadata)
                axial = {f"delta_{key}": float(np.diff(np.interp([start, end], strain,
                         [row[key] for row in rows]))[0])
                         for key in ("matrix_axial", "plane_axial")}
                conditioned = dict(
                    conditioned_observed_volume_pressure_reduction_MPa=(
                        bq*interval["observed_delta_q_MPa"]-interval["observed_delta_pressure_MPa"]),
                    conditioned_observed_backstress_change_MPa=(
                        ctheta*interval["observed_delta_q_MPa"]+mu*interval["observed_delta_pressure_MPa"]
                        if ctheta > 0 else float("nan")),
                    predicted_backstress_change_MPa=float(np.diff(np.interp([start, end], strain, backstress))[0]),
                    plane_coefficient=ctheta, plane_friction=mu,
                    strength_denominator=ctheta+mu*bq)
                observed_budgets.append({**identity, "confinement_MPa": metadata["confinement_MPa"],
                                        **interval, **common, **projection, **axial,
                                        **conditioned, **coefficients})
            count = sum(row["kind"] == "continuous_sliding" for row in result)
            matrix_count = sum(row["kind"] == "continuous_matrix_sliding" for row in result)
            print(f"{fit} {sid}: {count} plane and {matrix_count} normal-matrix continuous sliding increments; {len(jumps)} finite jumps")
    output = args.traces/"strength_competition.csv.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} loading increments to {output}")
    for name, result in (("turning_events.csv", peak_records),
                         ("turning_intervals.csv", budgets),
                         ("dilation_supported_intervals.csv", supported_intervals),
                         ("observed_stage_budgets.csv", observed_budgets)):
        write_csv(args.traces/name, result)
        print(f"{name}: {len(result)} records")


if __name__ == "__main__":
    main()
