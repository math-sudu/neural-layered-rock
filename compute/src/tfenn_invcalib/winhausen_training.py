"""Train neural constitutive energy and evolution on complete Mont Terri specimens.

The material parameters below belong to this experiment. No Nordas fit is
read. Acquisition order, initial prestress and source rows accompany every
consumed observation. Monotone loading is reconstructed within the observed
recording jitter, without sorting the records or altering response targets.
"""

import numpy as np
from scipy.optimize import isotonic_regression

from .coupled_friction_damage import (ConstitutiveSolveError, CoupledFrictionDamage,
                                      FrictionDamageLaw)
from .energetic_coupling import VOLUME
from .neural_storage import IdentityStorage, MonotoneStorageNetwork, NeuralPorousEnergy
from .neural_evolution import MatrixEvolutionNetwork, NeuralFrictionDamageLaw
from .poromechanical_coupling import TIElasticity
from .winhausen_data import load_specimens


# Value, lower bound, upper bound, logarithmic coordinate. These are search
# ranges, not confidence intervals. cross_fraction parameterizes positive
# definiteness instead of allowing invalid combinations of TI Poisson ratios.
PARAMETERS = {
    "E_parallel": (4500., 800., 18000., True),
    "E_normal": (2300., 500., 12000., True),
    "nu_parallel": (.25, .01, .48, False),
    "cross_fraction": (.32, .02, .9, False),
    "G_normal": (1600., 200., 10000., True),
    "stiffness_pressure_fraction": (.45, 0., .95, False),
    "matrix_cohesion": (2., .05, 10., True),
    "matrix_friction": (.85, .1, 2., False),
    "dilation_fraction": (.45, 0., 1., False),
    "parallel_weight": (.4, .05, 2., True),
    "plane_friction": (.3, .02, .8, False),
    "plane_compliance": (1., .02, 20., True),
    "initial_density": (.05, .005, 1., True),
    "damage_resistance": (.005, .0001, .2, True),
    "damage_peak_density": (.6, .04, 10., True),
    "damage_residual_fraction": (.03, 0., .7, False),
    "damage_shape": (1., .4, 2., False),
    "dilation_residual_fraction": (.1, 0., 1., False),
    "parallel_residual_weight": (1., .1, 3., True),
    "dilation_density_ratio": (1., .05, 20., True),
    "matrix_contraction": (.2, 0., 3., False),
    "friction_residual_fraction": (.7, 0., 1., False),
}
NAMES = tuple(PARAMETERS)
ELASTIC_NAMES = NAMES[:6]
REFERENCE_CONFINEMENT = 10.
DIRECTIONAL_EVOLUTION = (
    "C1 transition: w=w_peak for d<=d_peak; otherwise "
    "w=w_res+(w_peak-w_res)*r*(2-r), r=d_peak/d.")
# Constituents are explicit reference assumptions. The cap is outside the
# observed pressure range; its parameters are not estimated as if activated.
FIXED = dict(grain_bulk=50000., fluid_bulk=2200., porosity=.15,
             matrix_slope=1., plane_normal_factor=1., plane_shear_factor=1.,
             compaction_onset=100., compaction_hardening=5000.)
OBJECTIVE = (
    "Equal path/channel mean square in MPa, with equal pre/post stress-peak "
    "stage weights; additional channel maximum and response at the observed "
    "stress peak, plus terminal pressure. Not a likelihood.")
CELL_OBJECTIVE = (
    "Equal path/channel mean squared curve error in MPa, with equal pre/post "
    "stress-peak stage weights. Each observation cell extends halfway to the "
    "neighboring distinct accumulated-strain stations. Integrate the squared "
    "difference between the accepted-substep response and each unchanged "
    "recorded target over its cell; repeated strain records share a cell but "
    "retain their individual targets. "
    "additional channel maximum, stress-peak cell and terminal pressure cell. "
    "Not a likelihood.")


class PrestressInitializationError(ValueError):
    """A candidate cannot represent the specified elastic preparation."""


def initial_parameters():
    return {name: specification[0] for name, specification in PARAMETERS.items()}


def encode(parameters, names=NAMES):
    return np.array([np.log(parameters[name]) if PARAMETERS[name][3]
                     else parameters[name] for name in names])


def decode(coordinates, base=None, names=NAMES):
    parameters = dict(base) if base is not None else initial_parameters()
    parameters.update({name: float(np.exp(value) if PARAMETERS[name][3] else value)
                       for name, value in zip(names, coordinates)})
    return parameters


def coordinate_bounds(names=NAMES):
    return [(np.log(lo), np.log(hi)) if logarithmic else (lo, hi)
            for name in names for _, lo, hi, logarithmic in [PARAMETERS[name]]]


def elasticity(parameters, confinement):
    p = parameters
    nu_pn = p["cross_fraction"] * np.sqrt(
        (1-p["nu_parallel"])*p["E_parallel"]/(2*p["E_normal"]))
    return TIElasticity(p["E_parallel"], p["E_normal"], p["nu_parallel"],
                        nu_pn, p["G_normal"]).at_confinement(
        confinement, REFERENCE_CONFINEMENT, p["stiffness_pressure_fraction"])


def material(parameters, path, raw_network, raw_evolution=None, evolution_inputs=2,
             resistance_uses_pressure=True, evolution_local_width=0, evolution_fabric_knots=(),
             *, storage_form="neural", fabric_source="plastic_stress"):
    p = parameters
    if storage_form == "identity":
        network = IdentityStorage()
    elif storage_form == "neural":
        network = MonotoneStorageNetwork(raw_network, p["initial_density"],
                                        p["damage_peak_density"])
    else:
        raise ValueError(f"Unknown plane-storage form: {storage_form}")
    angle = np.deg2rad(path["angle"])
    energy = NeuralPorousEnergy(
        elasticity(p, path["confinement"]), [np.sin(angle), 0., np.cos(angle)],
        network=network, **FIXED)
    law_parameters = dict(
        matrix_cohesion=p["matrix_cohesion"], matrix_friction=p["matrix_friction"],
        matrix_dilatancy=p["matrix_friction"]*p["dilation_fraction"],
        parallel_weight=p["parallel_weight"], shear_weight=1.,
        plane_friction=p["plane_friction"], damage_resistance=p["damage_resistance"],
        damage_peak_density=p["damage_peak_density"],
        damage_residual_fraction=p["damage_residual_fraction"],
        damage_shape=p["damage_shape"],
        dilation_residual_fraction=p["dilation_residual_fraction"],
        parallel_residual_weight=p["parallel_residual_weight"],
        dilation_density_ratio=p["dilation_density_ratio"],
        dilation_initial_density=p["initial_density"],
        matrix_contraction=p["matrix_contraction"],
        friction_residual_fraction=p["friction_residual_fraction"])
    law = (FrictionDamageLaw(**law_parameters) if raw_evolution is None else
           NeuralFrictionDamageLaw(**law_parameters, network=MatrixEvolutionNetwork(
               raw_evolution, p["damage_peak_density"], REFERENCE_CONFINEMENT,
               evolution_inputs, local_width=evolution_local_width, fabric_knots=evolution_fabric_knots),
               resistance_uses_pressure=resistance_uses_pressure))
    return CoupledFrictionDamage(energy, law, p["plane_compliance"], fabric_source=fabric_source)


def recording_control(strain, maximum_adjustment=2e-4):
    """Least-squares monotone strain for these monotonic loading segments.

    OPA15 alternates between strain readings two 0.01-percentage-point units
    apart while stress keeps rising. Its monotone adjustment is at most 1.33
    units; all other specimens need less than one. Two units therefore bound
    this recording treatment. A larger correction needs a different history
    interpretation and raises instead of erasing a resolved unloading path.
    The general simulator itself accepts and retains unloading.
    """
    measured = np.asarray(strain, dtype=float)
    control = isotonic_regression(measured).x
    if np.max(abs(control-measured)) > maximum_adjustment+1e-12:
        raise ValueError("Recorded reversal exceeds the monotonic loading jitter allowance.")
    return control


def prepare_path(specimen, samples=100):
    """Select time-distributed source rows and observed stage landmarks.

    Extra points in the first 0.2% strain retain the initial stiffness. The
    stress peak is retained. Pressure does not determine the sampling, so
    stress-only and joint fits consume the same loading stations.
    No stress or pressure is interpolated between different source records.
    """
    h = specimen["history"]
    time = h["time_h"]
    uniform = np.searchsorted(time, np.linspace(time[0], time[-1], samples))
    early = np.flatnonzero(h["axial_strain"] <= .002)
    early = early[np.linspace(0, len(early)-1, min(16, len(early))).astype(int)]
    rows = np.unique(np.r_[0, uniform, early, np.argmax(h["q_MPa"]), len(time)-1])
    control = recording_control(h["axial_strain"])
    path = {key: value for key, value in specimen.items() if key != "history"}
    path.update(observations={name: values[rows] for name, values in h.items()},
                control_strain=control[rows],
                maximum_strain_adjustment=float(np.max(abs(control-h["axial_strain"]))),
                full_observed_peaks=dict(
                    q_MPa=float(np.max(h["q_MPa"])),
                    pressure_MPa=float(np.max(h["pressure_MPa"]))),
                full_observed_peak_strains={name: float(h["axial_strain"][np.argmax(h[name])])
                                           for name in ("q_MPa", "pressure_MPa")})
    return path


def training_paths(samples=100, training_specimens=None):
    return [prepare_path(p, samples) for p in
            load_specimens("train", training_specimens=training_specimens)]


def initial_state(model, parameters, path):
    """Prepare the recorded first q at zero excess pressure before sealing.

    These sub-MPa starting stresses are elastic for the fitted material. The
    explicit check exposes a candidate that would already activate inelastic
    deformation; such a candidate cannot use this preparation assumption.
    """
    p0 = path["confinement"]
    state = model.consolidated_state(p0, parameters["initial_density"])
    q0 = float(path["observations"]["q_MPa"][0])
    axial = state["strain"][0]+model.energy.S0[0, 0]*q0
    state = model.step(axial, [p0, p0, 0., 0., 0.], state, pressure=0.)
    if abs(state["response"]["total_stress"][0]-p0-q0) > 1e-6:
        raise PrestressInitializationError(
            f"{path['specimen_id']}: recorded prestress activates inelasticity.")
    return state


def observation_cell_errors(stations, trace_coordinate, trace_values, observations):
    """Exact first/second error moments on fixed observation-strain cells.

    Stations and trace coordinates are accumulated absolute imposed strain,
    preserving acquisition order through unloading. Repeated stations share
    the half-distance cell between distinct neighbors; each original target
    is compared separately. The substep curve remains piecewise linear, with
    zero-length segments contributing zero measure. Thus a hold does not give
    a jump an atom of fitting weight when its strain crosses the hold level.
    """
    stations, inverse = np.unique(stations, return_inverse=True)
    if len(stations) < 2:
        raise ValueError("Curve error requires a nonzero imposed-strain interval.")
    boundaries = np.r_[stations[0], .5*(stations[:-1]+stations[1:]), stations[-1]]
    coordinate, values = np.asarray(trace_coordinate), np.asarray(trace_values)
    positive = np.diff(coordinate) > 0
    x0, x1 = coordinate[:-1][positive], coordinate[1:][positive]
    y0, y1 = values[:-1][positive], values[1:][positive]
    mean, square = np.zeros_like(observations, dtype=float), np.zeros_like(observations, dtype=float)
    for cell, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        active = (x1 > a) & (x0 < b)
        left, right = np.maximum(x0[active], a), np.minimum(x1[active], b)
        slope = (y1[active]-y0[active])/(x1[active]-x0[active])[:, None]
        start = y0[active]+(left-x0[active])[:, None]*slope
        end = y0[active]+(right-x0[active])[:, None]*slope
        weight = (right-left)[:, None]/(b-a)
        for i in np.flatnonzero(inverse == cell):
            e0, e1 = start-observations[i], end-observations[i]
            mean[i] = np.sum(weight*(e0+e1)/2, axis=0)
            square[i] = np.sum(weight*(e0*e0+e0*e1+e1*e1)/3, axis=0)
    return mean, square


def simulate(parameters, raw_network, path, max_increment=.0005, raw_evolution=None, evolution_inputs=2,
             resistance_uses_pressure=True, max_log_density_increment=0., min_increment=1e-9,
             evolution_local_width=0, evolution_fabric_knots=(), *, on_step=None, storage_form="neural",
             fabric_source="plastic_stress"):
    """Integrate a history and optionally consume every accepted substep.

    A positive increment limit bounds both log-density growth and matrix
    strain change in the matrix-storage energy norm. The latter resolves
    rapid contraction before damage starts. At min_increment, a finite jump
    is accepted from the original coupled equations. Failed solves still
    raise there. Zero retains the fixed grid.
    """
    if (not np.isfinite(max_log_density_increment) or max_log_density_increment < 0
            or not np.isfinite(min_increment) or min_increment <= 0):
        raise ValueError("Density substepping requires a nonnegative limit and positive minimum increment.")
    model = material(parameters, path, raw_network, raw_evolution, evolution_inputs,
                     resistance_uses_pressure, evolution_local_width, evolution_fabric_knots,
                     storage_form=storage_form, fabric_source=fabric_source)
    initial = previous = initial_state(model, parameters, path)
    p0, m = path["confinement"], model.energy
    # Matrix self release is ||cm||_Hc^2/(2*d^2). Its initial resistance
    # supplies a nonzero strain reference before any matrix flow develops.
    matrix_reference = initial["d"]*np.sqrt(2*initial["damage_resistance"])

    def steps(left, right, history):
        subdivisions = max(1, int(np.ceil(abs(right-left)/max_increment)))
        targets = list(np.linspace(left, right, subdivisions+1)[1:][::-1])
        while targets:
            axial = targets.pop()
            can_split = max_log_density_increment > 0 and abs(axial-left) > min_increment
            try:
                state = model.step(initial["strain"][0]+axial, [p0, p0, 0., 0., 0.],
                                   history, fluid_content=initial["fluid_content"])
            except ConstitutiveSolveError as exc:
                if not can_split:
                    raise ConstitutiveSolveError(f"{path['specimen_id']}: {exc}") from exc
            else:
                change = 0.
                if can_split:
                    cm = history["matrix_strain"]
                    delta_cm = state["matrix_strain"]-cm
                    matrix_scale = max(np.sqrt(cm @ m.Hc @ cm), matrix_reference)
                    change = max(np.log(state["d"]/history["d"]),
                                 np.sqrt(delta_cm @ m.Hc @ delta_cm)/matrix_scale)
                if change <= max_log_density_increment:
                    yield axial, state
                    left, history = axial, state
                    continue
            # Trial states never enter the history or the accepted-step trace.
            targets.extend((axial, .5*(left+axial)))

    pressure_modulus = 1/(1/m.M+m.biot @ m.S0 @ m.biot)
    pressure_stress_coefficient = pressure_modulus*(m.biot @ m.S0)[0]
    control, observed = path["control_strain"], path["observations"]
    stations = np.r_[0., np.cumsum(abs(np.diff(control)))]
    trace_coordinate = [0.]
    trace_values = [[initial["response"]["total_stress"][0]-p0, initial["response"]["pressure"]]]
    rows, diagnostics = [], dict(min_dissipation_MPa=0., min_density_increment=0.,
                                max_fluid_balance=0., max_pressure_partition_MPa=0.,
                                min_cap_slack_MPa=np.inf)
    q_peak, u_peak = (-np.inf, 0.), (-np.inf, 0.)
    last_axial = 0.
    for i, target in enumerate(control):
        for axial, state in steps(last_axial, target, previous):
            f = state["response"]
            q = f["total_stress"][0]-p0
            if i:
                trace_coordinate.append(stations[i-1]+abs(axial-last_axial))
                trace_values.append([q, f["pressure"]])
            if q > q_peak[0]:
                q_peak = (float(q), float(axial))
            if f["pressure"] > u_peak[0]:
                u_peak = (float(f["pressure"]), float(axial))
            volume = VOLUME@(state["strain"]-initial["strain"])
            pressure_volume = ((q-observed["q_MPa"][0])/(3*FIXED["grain_bulk"])
                               + FIXED["porosity"]*(1/FIXED["fluid_bulk"]
                                 -1/FIXED["grain_bulk"])*f["pressure"])
            diagnostics["max_fluid_balance"] = max(diagnostics["max_fluid_balance"],
                                                   abs(float(volume-pressure_volume)))
            pressure_parts = dict(
                pressure_stress_MPa=pressure_stress_coefficient*(q-observed["q_MPa"][0]),
                pressure_matrix_MPa=pressure_modulus*(VOLUME @ (
                    state["matrix_strain"]-initial["matrix_strain"])),
                pressure_plane_MPa=pressure_modulus*(state["plane_modes"][0]-initial["plane_modes"][0]),
                pressure_compaction_MPa=pressure_modulus*(state["compaction"]-initial["compaction"]))
            diagnostics["max_pressure_partition_MPa"] = max(
                diagnostics["max_pressure_partition_MPa"],
                abs(float(sum(pressure_parts.values())-f["pressure"]+initial["response"]["pressure"])))
            diagnostics["min_dissipation_MPa"] = min(diagnostics["min_dissipation_MPa"],
                                                     *state["dissipation_increments"].values())
            diagnostics["min_density_increment"] = min(diagnostics["min_density_increment"],
                                                       state["d"]-previous["d"])
            diagnostics["min_cap_slack_MPa"] = min(diagnostics["min_cap_slack_MPa"],
                m.compaction_onset-f["compaction_force"])
            if on_step is not None:
                on_step(float(axial), state, previous)
            previous = state
        plane_backstress = m.plane_storage(state["d"])[0] @ state["plane_modes"] / model.D
        plane_traction = m.B.T @ f["plastic_stress"]
        resistance, resistance_d, resistance_p, *_ = model.law.evolution(
            state["d"], state["effective_mean_stress"], state["fabric_projection"])
        rows.append(dict(
            specimen_id=path["specimen_id"], role=path["role"],
            source_row=int(observed["source_row"][i]), time_h=float(observed["time_h"][i]),
            axial_strain=float(target), recorded_axial_strain=float(observed["axial_strain"][i]),
            q_MPa=float(q), observed_q_MPa=float(observed["q_MPa"][i]),
            pressure_MPa=float(f["pressure"]), observed_pressure_MPa=float(observed["pressure_MPa"][i]),
            density=float(state["d"]), matrix_volume=float(VOLUME@state["matrix_strain"]),
            matrix_axial=float(state["matrix_strain"][0]),
            matrix_backstress_axial_MPa=float((m.Hc@state["matrix_strain"])[0]/state["d"]),
            matrix_equivalent_force_MPa=float(state["matrix_equivalent_force"]),
            matrix_mean_force_MPa=float(state["matrix_mean_force"]),
            matrix_yield_MPa=float(state["matrix_yield"]),
            matrix_friction=float(model.law.friction(state["d"])[0]),
            plane_axial=float(m.B[0]@state["plane_modes"]),
            plane_slip=float(np.linalg.norm(state["plane_modes"][1:])),
            plane_shear_traction_MPa=float(np.linalg.norm(plane_traction[1:])/np.sqrt(2)),
            plane_shear_backstress_MPa=float(np.linalg.norm(plane_backstress[1:])/np.sqrt(2)),
            plane_normal_force_MPa=float(f["plane_force"][0]),
            plane_yield_MPa=float(state["plane_yield"]/np.sqrt(2)),
            volume=float(volume), matrix_release_MPa=float(f["matrix_self_release"]),
            plane_release_MPa=float(f["plane_to_matrix_release"]),
            effective_mean_MPa=float(state["effective_mean_stress"]),
            fabric_projection=float(state["fabric_projection"]),
            damage_resistance_MPa=float(state["damage_resistance"]),
            resistance_log_density_slope=float(state["d"]*resistance_d/resistance),
            resistance_log_pressure_slope_per_MPa=float(resistance_p/resistance),
            matrix_dilatancy=float(state["matrix_dilatancy"]),
            matrix_flow_fraction=float(state["matrix_flow_fraction"]),
            storage_coordinate=float(m.network.value_derivatives(state["d"])[0]),
            **{name: float(value) for name, value in pressure_parts.items()}))
        last_axial = target
    if stations[-1] > 0:
        cell_mean, cell_square = observation_cell_errors(stations, trace_coordinate, trace_values,
            np.column_stack([observed[name] for name in ("q_MPa", "pressure_MPa")]))
        for i, row in enumerate(rows):
            for j, name in enumerate(("q_MPa", "pressure_MPa")):
                row[name+"_cell_error_mean"] = float(cell_mean[i, j])
                row[name+"_cell_error_square"] = float(cell_square[i, j])
    return dict(rows=rows, q_peak=q_peak, pressure_peak=u_peak,
                pressure_stress_coefficient=float((m.biot @ m.S0)[0]
                    / (1/m.M + m.biot @ m.S0 @ m.biot)),
                diagnostics={k: float(v) for k, v in diagnostics.items()})


def path_metrics(path, result):
    rows, h = result["rows"], path["observations"]
    metrics = {name: dict(rmse=float(np.sqrt(np.mean(
        (np.array([row[name] for row in rows])-h[name])**2))), count=len(rows))
        for name in ("q_MPa", "pressure_MPa")}
    peaks, strains = path["full_observed_peaks"], path["full_observed_peak_strains"]
    peak_index = int(np.argmax(h["q_MPa"]))
    metrics.update(
        q_peak=dict(predicted_MPa=result["q_peak"][0], observed_MPa=peaks["q_MPa"],
                    predicted_strain=result["q_peak"][1], observed_strain=strains["q_MPa"]),
        pressure_peak=dict(predicted_MPa=result["pressure_peak"][0],
                           observed_MPa=peaks["pressure_MPa"],
                           predicted_strain=result["pressure_peak"][1],
                           observed_strain=strains["pressure_MPa"]),
        pressure_end=dict(predicted_MPa=rows[-1]["pressure_MPa"], observed_MPa=float(h["pressure_MPa"][-1])),
        at_observed_q_peak={name: dict(predicted_MPa=rows[peak_index][name],
                                     observed_MPa=float(h[name][peak_index]))
                            for name in ("q_MPa", "pressure_MPa")},
        physical_checks=result["diagnostics"], maximum_strain_adjustment=path["maximum_strain_adjustment"])
    return metrics


def stage_pairs(path, channel):
    """Observation cells at mid-rise, the channel peak and mid-fall."""
    strain = path["control_strain"]
    peak = int(np.argmin(abs(strain-path["full_observed_peak_strains"][channel])))
    before = int(np.argmin(abs(strain-(strain[0]+strain[peak])/2)))
    after = int(np.argmin(abs(strain-(strain[peak]+strain[-1])/2)))
    return [(a, b) for a, b in ((before, peak), (peak, after)) if strain[b] > strain[a]]


def volume_stage_residuals(path, rows, pressure_stress_coefficient):
    """Pressure-equivalent irreversible-volume error changes on common cells.

    The pressure partition gives M_Sigma*(v_pred-v_cond)=e_u-B_q*e_q.
    These are coupled errors from the same q/u observations, in MPa.
    """
    pairs = sorted({pair for name in ("q_MPa", "pressure_MPa") for pair in stage_pairs(path, name)})
    errors = np.array([row["pressure_MPa_cell_error_mean"]
                       -pressure_stress_coefficient*row["q_MPa_cell_error_mean"] for row in rows])
    return np.array([errors[b]-errors[a] for a, b in pairs])


def residuals(path, result, channels, increment_weight=0., response_measure="point",
              volume_stage_weight=0., stage_increment_weight=0.):
    """Keep the measured failure stage visible on paths with long tails.

    The two curve stages share one mean-square term. Both channels are also
    compared at the same observed stress-peak row. Optional increment matching
    retains local response changes that these value terms can underweight.
    Increments use acquisition order, without dividing by strain intervals or
    interpreting the experimental differences as measured tangents.
    Stress-only training consumes exactly the same stress-defined stations.
    Stage increments compare mid-rise to the observed stress peak and that
    peak to mid-fall, using those same endpoints for every fitted channel.

    The cell measure integrates squared curve errors on strain cells.
    Its landmark terms use the same cells, and its increment terms compare
    cell-mean errors. A moving finite jump therefore contributes continuously
    when it passes an observation station, without clipping the model jump.
    """
    if response_measure not in ("point", "cell"):
        raise ValueError(f"Unknown response measure: {response_measure}")
    if volume_stage_weight and (response_measure != "cell" or set(channels) != {"q_MPa", "pressure_MPa"}):
        raise ValueError("Volume-stage residuals require joint channels and the cell response measure.")
    residual = []
    peak_index = int(np.argmax(path["observations"]["q_MPa"]))
    stages = [slice(0, peak_index+1), slice(peak_index+1, None)]
    stages = [stage for stage in stages if len(path["observations"]["q_MPa"][stage])]
    for name in channels:
        predicted = np.array([row[name] for row in result["rows"]])
        observed = path["observations"][name]
        error = predicted-observed
        if response_measure == "cell":
            error = np.sqrt([row[name+"_cell_error_square"] for row in result["rows"]])
        for stage in stages:
            residual.extend(error[stage]/np.sqrt(len(stages)*len(observed[stage])))
        peak = result["q_peak" if name == "q_MPa" else "pressure_peak"][0]
        residual.append(peak-path["full_observed_peaks"][name])
        residual.append(error[peak_index])
        if name == "pressure_MPa":
            residual.append(error[-1])
        if increment_weight:
            increment_error = (np.array([row[name+"_cell_error_mean"] for row in result["rows"]])
                               if response_measure == "cell" else error)
            residual.extend(np.sqrt(increment_weight)*np.diff(increment_error))
        if stage_increment_weight:
            stage_error = (np.array([row[name+"_cell_error_mean"] for row in result["rows"]])
                           if response_measure == "cell" else error)
            residual.extend(np.sqrt(stage_increment_weight)*(stage_error[b]-stage_error[a])
                            for a, b in stage_pairs(path, "q_MPa"))
    if volume_stage_weight:
        residual.extend(np.sqrt(volume_stage_weight)*volume_stage_residuals(
            path, result["rows"], result["pressure_stress_coefficient"]))
    return np.asarray(residual)
