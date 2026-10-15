"""History handling, preparation and training isolation for Mont Terri fits."""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.neural_storage import MonotoneStorageNetwork
from tfenn_invcalib.neural_evolution import MatrixEvolutionNetwork
from tfenn_invcalib.winhausen_data import load_specimens


def test_recording_control_removes_jitter_without_sorting_or_rewriting_source():
    strain = np.array([0., .0001, .00005, .00015, .0004, .0003, .0005])
    control = wt.recording_control(strain)
    np.testing.assert_allclose(control, [0., .000075, .000075, .00015, .00035, .00035, .0005])
    np.testing.assert_array_equal(strain, [0., .0001, .00005, .00015, .0004, .0003, .0005])
    with pytest.raises(ValueError, match="reversal"):
        wt.recording_control([0., .004, .008, .004, .009])


def test_training_paths_only_consume_training_source_rows_and_preserve_peaks():
    specimens = {p["specimen_id"]: p for p in load_specimens("train")}
    paths = wt.training_paths(40)
    assert len(paths) == 7
    for p in paths:
        assert p["role"] == "train"
        h = specimens[p["specimen_id"]]["history"]
        rows = p["observations"]["source_row"]
        assert np.all(np.diff(rows) > 0)
        assert rows[0] == 0 and rows[-1] == len(h["q_MPa"])-1
        for name in ("q_MPa", "pressure_MPa"):
            np.testing.assert_array_equal(p["observations"][name], h[name][rows])
            assert p["full_observed_peaks"][name] == max(h[name])
        assert max(p["observations"]["q_MPa"]) == max(h["q_MPa"])


def test_pressure_values_do_not_determine_sampling_stations():
    specimen = load_specimens("train")[0]
    before = wt.prepare_path(specimen, 40)
    specimen["history"]["pressure_MPa"] = np.arange(len(specimen["history"]["time_h"]))
    after = wt.prepare_path(specimen, 40)
    np.testing.assert_array_equal(before["observations"]["source_row"], after["observations"]["source_row"])


def test_elastic_coordinates_remain_positive_definite_at_search_extremes():
    p = wt.initial_parameters()
    for ep in (800., 18000.):
        for en in (500., 12000.):
            for nu in (.01, .48):
                p.update(E_parallel=ep, E_normal=en, nu_parallel=nu, cross_fraction=.9)
                assert np.linalg.eigvalsh(wt.elasticity(p, 4.).local_compliance()).min() > 0


def test_recorded_nonzero_prestress_is_an_initial_load():
    path = wt.training_paths(25)[0]
    path["observations"]["q_MPa"][0] = .32
    p = wt.initial_parameters()
    net = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    model = wt.material(p, path, net.raw_parameters)
    state = wt.initial_state(model, p, path)
    assert state["response"]["total_stress"][0]-path["confinement"] == pytest.approx(.32, abs=1e-8)
    assert state["response"]["pressure"] == pytest.approx(0., abs=1e-8)


@pytest.mark.parametrize("density_limit", [0., .05])
@pytest.mark.parametrize("storage_form", ["neural", "identity"])
def test_full_history_preserves_state_through_unloading_and_fluid_constraint(density_limit, storage_form):
    path = wt.training_paths(25)[1]
    indices = np.array([0, 1, 2, 3, 4])
    path["observations"] = {name: value[indices] for name, value in path["observations"].items()}
    path["control_strain"] = np.array([0., .004, .009, .004, .011])
    p = wt.initial_parameters()
    net = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    steps = []
    result = wt.simulate(p, net.raw_parameters, path,
                         max_log_density_increment=density_limit,
                         storage_form=storage_form,
                         on_step=lambda axial, state, previous: steps.append((axial, state, previous)))
    rows = result["rows"]
    assert [row["axial_strain"] for row in rows] == list(path["control_strain"])
    assert rows[3]["density"] >= rows[2]["density"]-1e-8
    assert result["diagnostics"]["min_dissipation_MPa"] > -1e-7
    assert result["diagnostics"]["max_fluid_balance"] < 1e-10
    assert result["diagnostics"]["max_pressure_partition_MPa"] < 1e-9
    for row in rows:
        if storage_form == "identity":
            assert row["storage_coordinate"] == row["density"]
        parts = [row[f"pressure_{part}_MPa"] for part in ("stress", "matrix", "plane", "compaction")]
        assert sum(parts) == pytest.approx(row["pressure_MPa"], abs=1e-9)
        angle = np.deg2rad(path["angle"])
        physical_shear = abs(row["q_MPa"]*np.sin(angle)*np.cos(angle))
        assert row["plane_shear_traction_MPa"] == pytest.approx(physical_shear, abs=1e-9)
        assert row["plane_yield_MPa"] <= 1e-7
    assert result["diagnostics"]["min_cap_slack_MPa"] > 0.
    assert len(steps) > len(rows)
    for (axial, state, previous), following in zip(steps, steps[1:]):
        assert following[2] is state
        assert state["solver_route"] in {"previous_state", "active_damage", "matrix_corner",
                                         "released_corner", "fixed_density"}
    accepted = iter(steps)
    for row in rows:
        state = next(state for axial, state, previous in accepted if axial == row["axial_strain"])
        assert state["response"]["pressure"] == row["pressure_MPa"]
    # Forward sliding states separate physical backstress and pressure effects
    # on q, including the sqrt(2) conversion from the Mandel shear mode.
    denominator = np.sin(angle)*np.cos(angle)-p["plane_friction"]*np.sin(angle)**2
    sliding_rows = [row for row in rows[:3]+rows[-1:]
                    if row["plane_slip"] > 1e-8 and abs(row["plane_yield_MPa"]) < 1e-7]
    assert sliding_rows
    for row in sliding_rows:
        reconstructed = (row["plane_shear_backstress_MPa"]
                         +p["plane_friction"]*(path["confinement"]-row["pressure_MPa"])) / denominator
        assert reconstructed == pytest.approx(row["q_MPa"], abs=1e-7)


def test_density_substeps_localize_real_fold_without_clipping_the_jump():
    record = json.loads((Path(__file__).parent / "data/winhausen_near_fold_root.json").read_text())
    path = next(p for p in wt.training_paths(200) if p["specimen_id"] == "OPA11")
    keep = path["control_strain"] <= .0072
    path["control_strain"] = path["control_strain"][keep]
    path["observations"] = {name: values[keep] for name, values in path["observations"].items()}
    accepted = []
    result = wt.simulate(record["material_parameters"], record["storage_parameters"], path,
        1.5625e-5, record["evolution_parameters"], record["evolution_inputs"],
        record["resistance_uses_pressure"], .02, 1e-9,
        on_step=lambda axial, state, previous: accepted.append((axial, state, previous)))
    assert [row["axial_strain"] for row in result["rows"]] == list(path["control_strain"])
    jumps = []
    left = 0.
    for axial, state, previous in accepted:
        if np.log(state["d"]/previous["d"]) > .02:
            assert axial-left <= 1e-9
            jumps.append((axial, state, previous))
        left = axial
    assert jumps
    axial, state, previous = max(jumps, key=lambda step: step[1]["d"]-step[2]["d"])
    assert .0070 < axial < .0072
    assert state["d"]/previous["d"] > 1.2
    assert previous["response"]["total_stress"][0]-state["response"]["total_stress"][0] > .5
    assert result["diagnostics"]["min_density_increment"] >= -1e-10
    assert result["diagnostics"]["min_dissipation_MPa"] >= -1e-10
    assert result["diagnostics"]["max_fluid_balance"] < 1e-10
    assert result["diagnostics"]["max_pressure_partition_MPa"] < 1e-9
    for current, following in zip(accepted, accepted[1:]):
        assert following[2] is current[1]


@pytest.mark.parametrize("increment_weight", [0., 1.])
@pytest.mark.parametrize("stage_increment_weight", [0., 1.])
def test_stress_objective_does_not_consume_pressure_targets(increment_weight, stage_increment_weight):
    path = wt.training_paths(25)[0]
    rows = [{"q_MPa": float(x), "pressure_MPa": 0.} for x in path["observations"]["q_MPa"]]
    result = dict(rows=rows, q_peak=(path["full_observed_peaks"]["q_MPa"], 0.), pressure_peak=(0., 0.))
    before = wt.residuals(path, result, ["q_MPa"], increment_weight,
                          stage_increment_weight=stage_increment_weight)
    path["observations"]["pressure_MPa"][:] = 1234.
    path["full_observed_peaks"]["pressure_MPa"] = 1234.
    path["full_observed_peak_strains"]["pressure_MPa"] = 1234.
    np.testing.assert_array_equal(wt.residuals(path, result, ["q_MPa"], increment_weight,
                                  stage_increment_weight=stage_increment_weight), before)


def test_matrix_flow_refinement_stabilizes_saved_activation_cell():
    record = json.loads((Path(__file__).parent / "data/winhausen_matrix_flow_event.json").read_text())
    path = next(p for p in wt.training_paths(record["samples"], record["training_specimens"])
                if p["specimen_id"] == "OPA15")
    keep = path["control_strain"] <= .0021
    path["control_strain"] = path["control_strain"][keep]
    path["observations"] = {name: values[keep] for name, values in path["observations"].items()}
    evolution = record["evolution_network"]
    centered = []
    step = 1e-7
    for factor in (1., .5):
        values = []
        for delta in (-step, 0., step):
            parameters = dict(record["material_parameters"])
            parameters["initial_density"] *= np.exp(delta)
            result = wt.simulate(parameters, record["network"]["raw_parameters"], path,
                factor*record["max_increment"], evolution["raw_parameters"], evolution["input_size"],
                evolution["resistance_uses_pressure"], factor*record["max_log_density_increment"],
                factor*record["min_increment"], evolution["local_width"], evolution["fabric_knots"])
            values.append([result["rows"][31][channel+"_cell_error_mean"]
                           -result["rows"][30][channel+"_cell_error_mean"]
                           for channel in ("q_MPa", "pressure_MPa")])
        lower, base, upper = np.asarray(values)
        forward, backward = (upper-base)/step, (base-lower)/step
        # Previously these were -307/-21 for q and +307/+20 for pressure:
        # a small parameter change chose a different pre-damage substep tree.
        np.testing.assert_allclose(forward, backward, atol=.03, rtol=.1)
        centered.append((upper-lower)/(2*step))
    np.testing.assert_allclose(centered[0], centered[1], atol=.04, rtol=.1)


def test_increment_objective_penalizes_unobserved_drop_but_not_constant_offset():
    observed = np.array([0., 2., 4., 3., 2., 1.])
    path = dict(observations={"q_MPa": observed}, full_observed_peaks={"q_MPa": 4.})
    for error, extra_loss in [(np.ones(6), 0.), (np.array([0., 0., 0., -2., -2., -2.]), 1.)]:
        predicted = observed+error
        result = dict(rows=[dict(q_MPa=q) for q in predicted], q_peak=(max(predicted), .1))
        original = wt.residuals(path, result, ["q_MPa"])
        np.testing.assert_array_equal(wt.residuals(path, result, ["q_MPa"], 0.), original)
        weighted = wt.residuals(path, result, ["q_MPa"], .25)
        assert len(weighted) == len(original)+len(observed)-1
        assert weighted@weighted-original@original == pytest.approx(extra_loss)


def test_stage_increments_resolve_late_loading_and_ignore_constant_offsets():
    strain = np.arange(9, dtype=float)
    q = np.array([0., 2., 4., 6., 8., 6., 4., 2., 0.])
    u = np.array([0., 3., 5., 4., 3., 2., 1., 0., -1.])
    path = dict(control_strain=strain, observations={"q_MPa": q, "pressure_MPa": u},
                full_observed_peaks={"q_MPa": 8., "pressure_MPa": 5.},
                full_observed_peak_strains={"q_MPa": 4., "pressure_MPa": 2.})
    for channel, observed in (("q_MPa", q), ("pressure_MPa", u)):
        for error in (np.full(9, 2.), np.array([0., 0., 0., -1., -2., -2., -2., -2., -2.])):
            predicted = observed+error
            result = dict(rows=[{channel: value} for value in predicted],
                          q_peak=(max(predicted), 0.), pressure_peak=(max(predicted), 0.))
            base = wt.residuals(path, result, [channel])
            weighted = wt.residuals(path, result, [channel], stage_increment_weight=.25)
            # Both channels use strains 2 -> 4 -> 6, regardless of the u peak.
            np.testing.assert_array_equal(weighted[:-2], base)
            np.testing.assert_array_equal(weighted[-2:], .5*np.diff(error[[2, 4, 6]]))


@pytest.mark.parametrize("response_measure,volume_weight", [("point", 0.), ("cell", 0.), ("cell", 1.)])
def test_training_loss_uses_recorded_increment_weight(tmp_path, response_measure, volume_weight):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=storage.tolist()), active_material_parameters=[],
                  train_network=False, channels=["q_MPa", "pressure_MPa"], increment_weight=1.,
                  max_log_density_increment=.05, min_increment=1e-8,
                  response_measure=response_measure, volume_stage_weight=volume_weight,
                  stage_increment_weight=1.)
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    result = wt.simulate(p, storage, objective.paths[0], record["max_increment"],
                         max_log_density_increment=.05, min_increment=1e-8)
    expected = wt.residuals(objective.paths[0], result, record["channels"], 1., response_measure, volume_weight, 1.)
    np.testing.assert_array_equal(objective.residual_vector(np.array([])), expected)
    repeated = objective.residual_vector(np.array([]))
    np.testing.assert_array_equal(repeated, expected)
    repeated[:] = 0.
    np.testing.assert_array_equal(objective.residual_vector(np.array([])), expected)
    assert objective.evaluations == 1
    saved = json.loads((tmp_path/"checkpoint.json").read_text())
    assert saved["increment_weight"] == 1.
    assert saved["loss"] == pytest.approx(expected@expected)


def test_volume_stages_use_common_endpoints_and_current_elastic_coefficient():
    path = dict(control_strain=np.arange(9.),
                full_observed_peak_strains=dict(q_MPa=4., pressure_MPa=2.))
    rows = [dict(q_MPa_cell_error_mean=float(i), pressure_MPa_cell_error_mean=.4*i)
            for i in range(9)]
    # Nonzero q/u errors cancel in the pressure-equivalent volume when
    # they satisfy the candidate material's pressure partition.
    np.testing.assert_array_equal(wt.volume_stage_residuals(path, rows, .4), np.zeros(4))
    # The union has (1,2), (2,4), (2,5), (4,6); q/u peaks differ.
    np.testing.assert_allclose(wt.volume_stage_residuals(path, rows, .2), [.2, .4, .6, .4])
    # A time-independent offset cannot change a finite-stage volume error.
    for row in rows:
        row["pressure_MPa_cell_error_mean"] += 3.
    np.testing.assert_allclose(wt.volume_stage_residuals(path, rows, .2), [.2, .4, .6, .4])


@pytest.mark.parametrize("channels,measure", [(["q_MPa"], "cell"),
                                              (["q_MPa", "pressure_MPa"], "point")])
def test_volume_stages_reject_a_stress_only_or_point_objective(channels, measure):
    with pytest.raises(ValueError, match="joint channels and the cell"):
        wt.residuals({}, {}, channels, response_measure=measure, volume_stage_weight=1.)


def test_cell_integrals_match_linear_errors_and_do_not_cancel_opposite_errors():
    coordinate = np.array([0., .3, 1., 2.])
    predicted = np.column_stack([coordinate-1., np.full(4, 2.)])
    mean, square = wt.observation_cell_errors([0., 1., 2.], coordinate, predicted, np.zeros((3, 2)))
    np.testing.assert_allclose(mean, [[-.75, 2.], [0., 2.], [.75, 2.]], atol=1e-14)
    np.testing.assert_allclose(square, [[7/12, 4.], [1/12, 4.], [7/12, 4.]])


def test_cell_integral_tracks_a_finite_jump_across_an_observation():
    def cell_at_one(event):
        knots = [0., event, event, 2.]
        first, second = wt.observation_cell_errors([0., 1., 1., 2.], knots,
            np.array([[2.], [2.], [0.], [0.]]), np.array([[0.], [0.], [1.], [0.]]))
        # Repeated strain retains its distinct recorded response, without
        # adding an atom at that strain to the integral.
        assert first[2, 0] == pytest.approx(first[1, 0]-1.)
        assert second[2, 0] == pytest.approx(second[1, 0]-2*first[1, 0]+1.)
        return first[1, 0], second[1, 0]

    below, above = np.array(cell_at_one(1.-1e-5)), np.array(cell_at_one(1.+1e-5))
    np.testing.assert_allclose(below, [1.-2e-5, 2.-4e-5], atol=1e-9)
    np.testing.assert_allclose(above, [1.+2e-5, 2.+4e-5], atol=1e-9)
    np.testing.assert_allclose((above-below)/2e-5, [2., 4.], rtol=1e-8)


def test_cell_measure_uses_squared_curve_error_at_landmarks_and_mean_changes():
    observed = np.array([0., 2., 1.])
    path = dict(observations={"q_MPa": observed, "pressure_MPa": np.zeros(3)},
                full_observed_peaks={"q_MPa": 2., "pressure_MPa": 0.})
    result = dict(rows=[dict(q_MPa=float(q), q_MPa_cell_error_mean=float(m),
                            q_MPa_cell_error_square=float(s))
                       for q, m, s in zip(observed, [0., 1., 0.], [0., 4., 0.])],
                  q_peak=(2., 1.))
    residual = wt.residuals(path, result, ["q_MPa"], .5, "cell")
    # Stage error 1, peak-cell error 4, adjacent cell-mean changes 1.
    assert residual@residual == pytest.approx(6.)
    path["observations"]["pressure_MPa"][:] = 1000.
    np.testing.assert_array_equal(wt.residuals(path, result, ["q_MPa"], .5, "cell"), residual)


def test_equal_maxima_do_not_hide_response_error_at_observed_failure_stage():
    path = dict(observations={"q_MPa": np.array([0., 2., 4., 3., 2.]),
                              "pressure_MPa": np.array([0., 1., 2., 1., 0.])},
                full_observed_peaks={"q_MPa": 4., "pressure_MPa": 2.})
    result = dict(rows=[dict(q_MPa=q, pressure_MPa=u)
                        for q, u in zip([0., 4., 2., 2., 2.], [0., 2., 1., 0., 0.])],
                  q_peak=(4., .1), pressure_peak=(2., .1))
    residual = wt.residuals(path, result, ["q_MPa", "pressure_MPa"])
    assert residual[5] == 0.  # Matching maximum q.
    assert residual[6] == -2.  # q at the observed failure stage.
    assert residual[12] == 0.  # Matching maximum u.
    assert residual[13] == -1.  # u at that same stage.


def test_tail_length_does_not_dilute_prepeak_error():
    def loss(tail_length):
        observed = np.r_[0., 2., 4., np.ones(tail_length)]
        predicted = observed.copy()
        predicted[:2] += 1.
        path = dict(observations={"q_MPa": observed}, full_observed_peaks={"q_MPa": 4.})
        result = dict(rows=[dict(q_MPa=q) for q in predicted], q_peak=(4., .1))
        residual = wt.residuals(path, result, ["q_MPa"])
        return residual@residual
    assert loss(2) == pytest.approx(loss(200))


def test_unresolved_candidate_preserves_accepted_fit_and_exposes_failure(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/train_winhausen_neural.py"
    specification = importlib.util.spec_from_file_location("winhausen_trainer", script)
    trainer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(trainer)
    p = wt.initial_parameters()
    raw = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=raw.tolist()), active_material_parameters=list(wt.NAMES),
                  train_network=False, channels=["q_MPa", "pressure_MPa"])
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    assert np.isfinite(objective(wt.encode(p)))
    accepted_parameters = json.loads((tmp_path / "checkpoint.json").read_text())["material_parameters"]
    accepted_paths = (tmp_path / "train_paths.csv").read_bytes()
    candidate = wt.encode(p)
    candidate[0] += .1

    def unsolved(*args, **kwargs):
        raise wt.ConstitutiveSolveError("OPA07: unresolved material state")

    monkeypatch.setattr(wt, "simulate", unsolved)
    assert np.isinf(objective(candidate))
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["material_parameters"] == accepted_parameters
    assert (tmp_path / "train_paths.csv").read_bytes() == accepted_paths
    assert saved["optimizer"]["unsolved_evaluations"] == 1
    assert saved["optimizer"]["last_unsolved_candidate"]["material_parameters"]["E_parallel"] > p["E_parallel"]
    fresh = trainer.TrainingLoss(record, tmp_path, map)
    with pytest.raises(wt.ConstitutiveSolveError, match="unresolved material"):
        fresh(wt.encode(p))

    def programming_error(*args, **kwargs):
        raise ValueError("unexpected data shape")

    monkeypatch.setattr(wt, "simulate", programming_error)
    with pytest.raises(ValueError, match="unexpected data shape"):
        objective(candidate)


def load_trainer():
    script = Path(__file__).resolve().parents[1] / "scripts/train_winhausen_neural.py"
    specification = importlib.util.spec_from_file_location("winhausen_trainer", script)
    trainer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(trainer)
    return trainer


@pytest.mark.parametrize("resistance_uses_pressure", [False, True])
def test_joint_coordinates_reach_both_networks_and_failed_candidate_record(tmp_path, monkeypatch,
                                                                          resistance_uses_pressure):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=evolution.tolist(),
                                         resistance_uses_pressure=resistance_uses_pressure),
                  active_material_parameters=["matrix_friction"], train_network=True,
                  train_evolution=True, channels=["q_MPa", "pressure_MPa"])
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    start = np.r_[p["matrix_friction"], storage, evolution]
    assert np.isfinite(objective(start))
    candidate = start.copy()
    candidate[0] += .01
    candidate[1] += .03
    candidate[-1] += .05

    def unsolved(parameters, raw_storage, path, increment, raw_evolution, evolution_inputs,
                 pressure_in_resistance, density_limit, min_increment, local_width, fabric_knots,
                 *, storage_form, fabric_source):
        assert storage_form == "neural"
        assert fabric_source == "plastic_stress"
        assert evolution_inputs == 2
        assert local_width == 0
        assert fabric_knots == ()
        assert pressure_in_resistance is resistance_uses_pressure
        assert density_limit == 0.
        assert min_increment == 1e-9
        assert parameters["matrix_friction"] == candidate[0]
        np.testing.assert_array_equal(raw_storage, candidate[1:13])
        np.testing.assert_array_equal(raw_evolution, candidate[13:])
        raise wt.ConstitutiveSolveError("test unresolved coupled candidate")

    monkeypatch.setattr(wt, "simulate", unsolved)
    assert np.isinf(objective(candidate))
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    np.testing.assert_array_equal(saved["evolution_network"]["raw_parameters"], evolution)
    assert saved["evolution_network"]["resistance_uses_pressure"] is resistance_uses_pressure
    np.testing.assert_array_equal(saved["optimizer"]["last_unsolved_candidate"]["raw_evolution_parameters"],
                                  candidate[13:])


def test_stress_experiment_rejects_warm_start_trained_with_pressure(tmp_path, monkeypatch, capsys):
    source = tmp_path / "joint.json"
    source.write_text(json.dumps(dict(channels=["q_MPa", "pressure_MPa"])), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize", "--channels", "stress",
        "--output", str(tmp_path / "stress"), "--matrix-evolution", "--initialize-from", str(source)])
    with pytest.raises(SystemExit) as exc:
        load_trainer().main()
    assert exc.value.code == 2
    assert "cannot remove a previously trained channel" in capsys.readouterr().err
    assert not (tmp_path / "stress/checkpoint.json").exists()


@pytest.mark.parametrize("input_size", [2, 3])
@pytest.mark.parametrize("block", ["flow", "response"])
def test_response_blocks_change_selected_laws_and_preserve_storage(tmp_path, monkeypatch, input_size, block):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"], input_size=input_size).raw_parameters
    evolution[-8:] = [.12, -.15, .1, -.08, -.1, .15, .12, -.09]
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa", "pressure_MPa"],
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")],
                  material_parameters=p, network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=evolution.tolist(), input_size=input_size,
                                         pressure_scale_MPa=10., resistance_uses_pressure=False))
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / block
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize", "--block", block,
        "--output", str(output), "--initialize-from", str(source_file), "--workers", "1"])

    def check_change(self, coordinates):
        assert self.names == ("dilation_fraction", "dilation_residual_fraction",
                              "dilation_density_ratio", "matrix_contraction")
        output_count = 4 if block == "flow" else 8
        assert self.evolution_indices == list(range(len(evolution)-output_count, len(evolution)))
        assert not self.train_network
        changed = coordinates + .03
        fitted, network, evolved = self.unpack(changed)
        path = self.paths[-1]
        before = wt.material(p, path, storage, evolution, input_size, False)
        after = wt.material(fitted, path, network, evolved, input_size, False)
        np.testing.assert_array_equal(network, storage)
        np.testing.assert_array_equal(evolved[:-output_count], evolution[:-output_count])
        for d, pressure, fabric in [(.15, 4., -.5), (.9, 10., .25), (3., 14., 1.)]:
            old = before.law.evolution(d, pressure, fabric)
            new = after.law.evolution(d, pressure, fabric)
            if block == "flow":
                np.testing.assert_array_equal(np.array(new)[[0, 1, 2, 6]], np.array(old)[[0, 1, 2, 6]])
            else:
                assert new[0] != old[0]
                assert new[0] > 0
            assert new[3] != old[3]
            assert new[3] <= after.law.friction(d)[0]
            assert before.law.friction(d) == after.law.friction(d)
            np.testing.assert_array_equal(before.energy.plane_storage(d), after.energy.plane_storage(d))
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", check_change)
    trainer.main()
    assert json.loads(source_file.read_text()) == source


@pytest.mark.parametrize("field,value,message", [
    ("dataset", "different material", "same dataset"),
    ("training_specimens", ["OPA23"], "cannot exclude previously fitted")])
def test_warm_start_rejects_dataset_change_or_relabelling_fitted_specimens(
        tmp_path, monkeypatch, capsys, field, value, message):
    source = dict(channels=["q_MPa"], dataset="Winhausen2023 Mont Terri",
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")])
    source[field] = value
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "joint"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
        "--channels", "joint", "--output", str(output), "--initialize-from", str(source_file),
        "--training-specimens", *[p["specimen_id"] for p in load_specimens("train")]])
    with pytest.raises(SystemExit) as exc:
        load_trainer().main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (output / "checkpoint.json").exists()


def test_expanded_split_round_trip_selects_full_histories_and_angular_supports(tmp_path, monkeypatch):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"], input_size=3)
    original = [s["specimen_id"] for s in load_specimens("train")]
    selected = original + ["OPA15", "OPA24"]
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa", "pressure_MPa"],
                  training_specimens=original, material_parameters=p,
                  network=dict(raw_parameters=storage.raw_parameters.tolist()),
                  evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                                         input_size=3, pressure_scale_MPa=10.,
                                         fabric_knots=[-.5, 0., 1.]))
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    original_bytes = source_file.read_bytes()

    def capture(self, coordinates):
        assert {p["specimen_id"] for p in self.paths} == set(selected)
        assert all(p["role"] == "train" for p in self.paths)
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", capture)
    for name, parent, options in (
            ("expanded", source_file, ["--training-specimens", *selected]),
            ("inherited", tmp_path / "expanded/checkpoint.json", [])):
        monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
            "--output", str(tmp_path / name), "--initialize-from", str(parent), "--workers", "1", *options])
        trainer.main()
        saved = json.loads((tmp_path / name / "checkpoint.json").read_text())
        assert set(saved["training_specimens"]) == set(selected)
        training = load_specimens("train", training_specimens=saved["training_specimens"])
        knots = sorted({0., *[(3*np.sin(np.deg2rad(s["angle"]))**2-1)/2 for s in training]})
        np.testing.assert_array_equal(saved["evolution_network"]["fabric_knots"], knots)
        assert {s["specimen_id"] for s in load_specimens("test", training_specimens=selected)} == {
            "OPA17", "OPA18", "OPA23"}
    assert source_file.read_bytes() == original_bytes
    assert [s["specimen_id"] for s in load_specimens("train")] == original
    source_specimens = {s["specimen_id"]: s for s in load_specimens(None)}
    for path in wt.training_paths(40, selected):
        h = source_specimens[path["specimen_id"]]["history"]
        rows = path["observations"]["source_row"]
        for channel in ("q_MPa", "pressure_MPa"):
            np.testing.assert_array_equal(path["observations"][channel], h[channel][rows])


@pytest.mark.parametrize("selected,message", [([], "nonempty"), (["OPA15", "OPA15"], "unique"),
                                             (["OPA99"], "Unknown")])
def test_experiment_split_rejects_ambiguous_history_selection(selected, message):
    with pytest.raises(ValueError, match=message):
        load_specimens(None, training_specimens=selected)


def test_existing_experiment_cannot_change_fitted_specimens(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "train", "--output", str(tmp_path),
                                     "--training-specimens", "OPA15"])
    with pytest.raises(SystemExit):
        load_trainer().main()
    assert "separate --output" in capsys.readouterr().err


@pytest.mark.parametrize("has_evolution", [False, True])
@pytest.mark.parametrize("override", [None, .85])
@pytest.mark.parametrize("channels", ["stress", "joint"])
def test_warm_start_retains_learned_evolution_or_initializes_a_new_network(tmp_path, monkeypatch, has_evolution, override, channels):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    raw = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters.copy()
    raw[-1] = .37
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa"],
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")],
                  material_parameters=p, network=dict(raw_parameters=storage.tolist()))
    if has_evolution:
        source["evolution_network"] = dict(raw_parameters=raw.tolist(), pressure_scale_MPa=10.)
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "resumed"
    arguments = ["train_winhausen_neural.py", "initialize", "--channels", channels,
        "--output", str(output), "--initialize-from", str(source_file), "--workers", "1"]
    if not has_evolution:
        arguments.append("--matrix-evolution")
    if override is not None:
        arguments.extend(["--set-material", "damage_shape", str(override)])
    monkeypatch.setattr(sys, "argv", arguments)

    def capture_start(self, coordinates):
        parameters, network, evolution = self.unpack(coordinates)
        expected_material = p if override is None else {**p, "damage_shape": override}
        assert parameters == pytest.approx(expected_material)
        np.testing.assert_array_equal(network, storage)
        expected = raw if has_evolution else MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters
        np.testing.assert_array_equal(evolution, expected)
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", capture_start)
    trainer.main()
    saved = json.loads((output / "checkpoint.json").read_text())
    assert saved["channels"] == (["q_MPa"] if channels == "stress" else ["q_MPa", "pressure_MPa"])
    assert saved["initialization"]["source_channels"] == ["q_MPa"]
    assert saved["initialization"]["role"] == (
        "same-channel training checkpoint warm start" if channels == "stress" else
        "stress-to-joint training checkpoint warm start")
    assert json.loads(source_file.read_text()) == source
    assert saved["directional_evolution"] == wt.DIRECTIONAL_EVOLUTION
    assert saved["initialization"]["evolution_raw_parameters"] == (raw.tolist() if has_evolution else None)
    if override is not None:
        assert saved["initialization"]["material_parameters"] == p
        assert saved["initialization"]["material_overrides"] == {"damage_shape": override}
        assert saved["material_parameters"]["damage_shape"] == override
        assert "damage_shape" in saved["active_material_parameters"]
        assert json.loads(source_file.read_text())["material_parameters"] == p


@pytest.mark.parametrize("requested_weight,expected_weight", [(None, 1.), ("0", 0.), ("0.5", .5)])
@pytest.mark.parametrize("weight_name", ["increment_weight", "volume_stage_weight", "stage_increment_weight"])
def test_increment_weight_persists_on_resume_and_can_change_in_new_experiment(
        tmp_path, monkeypatch, requested_weight, expected_weight, weight_name):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa", "pressure_MPa"],
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")],
                  material_parameters=p, network=dict(raw_parameters=storage.tolist()),
                  samples=200, max_increment=1.5625e-5, max_log_density_increment=.02, min_increment=1e-9,
                  response_measure="cell")
    source[weight_name] = 1.
    source_file = tmp_path/"source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")

    def capture_start(self, coordinates):
        assert self.record[weight_name] == expected_weight
        assert self.record["samples"] == 200
        assert self.record["max_increment"] == 1.5625e-5
        assert self.record["max_log_density_increment"] == .02
        assert self.record["min_increment"] == 1e-9
        assert self.record["response_measure"] == "cell"
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", capture_start)
    monkeypatch.setattr(trainer, "minimize", lambda *a, **k:
        type("Result", (), dict(success=False, message="batch complete", nfev=1))())
    output = tmp_path/"new_objective"
    arguments = ["train_winhausen_neural.py", "initialize", "--output", str(output),
                 "--initialize-from", str(source_file), "--workers", "1"]
    if requested_weight is not None:
        arguments += ["--"+weight_name.replace("_", "-"), requested_weight]
    monkeypatch.setattr(sys, "argv", arguments)
    trainer.main()
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "train", "--output", str(output),
                                     "--workers", "1", "--max-evaluations", "1"])
    trainer.main()
    saved = json.loads((output/"checkpoint.json").read_text())
    assert saved[weight_name] == expected_weight
    assert saved["samples"] == source["samples"]
    assert saved["max_increment"] == source["max_increment"]
    assert saved["max_log_density_increment"] == source["max_log_density_increment"]
    assert saved["min_increment"] == source["min_increment"]
    assert saved["response_measure"] == "cell"
    description = {"increment_weight": "finite increments", "volume_stage_weight": "e_u-B_q*e_q",
                   "stage_increment_weight": "common stress-defined"}[weight_name]
    assert (description in saved["objective"]) is bool(expected_weight)
    assert json.loads(source_file.read_text()) == source


@pytest.mark.parametrize("action,weight", [("initialize", "-1"), ("initialize", "nan"),
                                           ("initialize", "inf"), ("train", "1")])
def test_increment_weight_rejects_invalid_or_in_place_changes(tmp_path, monkeypatch, action, weight):
    output = tmp_path/"invalid_objective"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", action, "--output", str(output),
                                     "--increment-weight", weight])
    with pytest.raises(SystemExit) as exc:
        load_trainer().main()
    assert exc.value.code == 2
    assert not output.exists()


def test_storage_restart_recovers_curvature_and_retains_matrix_evolution(tmp_path, monkeypatch):
    trainer = load_trainer()
    p = wt.initial_parameters()
    # A saturated storage closure has no usable local training direction.
    storage = np.r_[np.full(4, -300.), np.full(4, -1000.), np.full(4, -100.)]
    raw = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters.copy()
    raw[15], raw[19] = -.23, .004
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa", "pressure_MPa"],
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")],
                  material_parameters=p, network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=raw.tolist(), pressure_scale_MPa=10.))
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "storage_restart"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
        "--output", str(output), "--initialize-from", str(source_file), "--workers", "1",
        "--reset-storage"])

    def capture_start(self, coordinates):
        parameters, network, evolution = self.unpack(coordinates)
        assert parameters == pytest.approx(p)
        np.testing.assert_array_equal(evolution, raw)
        net = MonotoneStorageNetwork(network, p["initial_density"], p["damage_peak_density"])
        density = np.linspace(p["initial_density"], 10*p["damage_peak_density"], 20)
        h, hp, hpp = net.value_derivatives(density)
        assert h[0] == pytest.approx(density[0])
        assert hp[0] == pytest.approx(1.)
        assert np.min(hp) > 0
        assert np.max(abs(hpp)) > .01
        perturbed = network.copy()
        perturbed[0] += 1e-4
        new_h = MonotoneStorageNetwork(perturbed, net.initial_density,
                                      net.density_scale).value_derivatives(density)[0]
        assert np.max(abs(new_h-h)) > 1e-7
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", capture_start)
    trainer.main()
    saved = json.loads((output / "checkpoint.json").read_text())
    assert saved["initialization"]["storage_raw_parameters"] == storage.tolist()
    assert saved["initialization"]["storage_reset"]["raw_parameters"] == saved["network"]["raw_parameters"]
    assert saved["initialization"]["evolution_raw_parameters"] == raw.tolist()
    assert saved["train_network"] and saved["train_evolution"]
    assert json.loads(source_file.read_text()) == source


def test_evolution_weight_intervention_preserves_source_and_other_weights(tmp_path, monkeypatch):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    raw = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters.copy()
    raw[15], raw[19] = -.23, .004
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa", "pressure_MPa"],
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")],
                  material_parameters=p, network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=raw.tolist(), pressure_scale_MPa=10.))
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "intervention"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
        "--output", str(output), "--initialize-from", str(source_file), "--workers", "1",
        "--set-evolution-weight", "15", "0"])

    def capture_start(self, coordinates):
        parameters, network, evolution = self.unpack(coordinates)
        assert parameters == pytest.approx(p)
        np.testing.assert_array_equal(network, storage)
        expected = raw.copy()
        expected[15] = 0.
        np.testing.assert_array_equal(evolution, expected)
        self.record["optimizer"] = dict(status="running")
        return 0.

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", capture_start)
    trainer.main()
    saved = json.loads((output / "checkpoint.json").read_text())
    assert saved["initialization"]["evolution_weight_overrides"] == {"15": 0.}
    assert saved["initialization"]["evolution_raw_parameters"] == raw.tolist()
    assert saved["train_evolution"]
    assert json.loads(source_file.read_text()) == source


@pytest.mark.parametrize("name,value", [("unknown", "1"), ("damage_shape", "bad"),
                                        ("damage_shape", "nan"), ("damage_shape", "0.1")])
def test_material_start_rejects_invalid_values_before_creating_experiment(tmp_path, monkeypatch, name, value):
    output = tmp_path / "invalid"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
        "--output", str(output), "--set-material", name, value])
    with pytest.raises(SystemExit) as exc:
        load_trainer().main()
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("difference_step", [1e-4, 1e-6])
@pytest.mark.parametrize("resistance_uses_pressure", [False, True])
def test_history_jacobian_predicts_coupled_loading_residual_changes(tmp_path, difference_step,
                                                                  resistance_uses_pressure):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters
    evolution[12:] = [.12, -.15, .1, -.08, -.1, .15, .12, -.09]
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=evolution.tolist(),
                                         resistance_uses_pressure=resistance_uses_pressure),
                  active_material_parameters=["matrix_friction", "damage_resistance"],
                  train_network=False, train_evolution=False, channels=["q_MPa", "pressure_MPa"],
                  increment_weight=1.)
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    x = wt.encode(p, objective.names)
    residual = objective.residual_vector(x)
    assert json.loads((tmp_path/"checkpoint.json").read_text())["evolution_network"][
        "resistance_uses_pressure"] is resistance_uses_pressure
    jacobian = trainer.history_jacobian(objective, x, wt.coordinate_bounds(objective.names),
                                       step=difference_step)
    batched = trainer.history_jacobian(objective, x, wt.coordinate_bounds(objective.names),
                                     step=difference_step, batch_size=2)
    np.testing.assert_array_equal(batched, jacobian)
    delta = np.array([2e-5, -1e-5])
    actual = objective.residual_vector(x+delta)-residual
    np.testing.assert_allclose(jacobian@delta, actual, rtol=.02, atol=2e-6)
    assert np.linalg.norm(actual) > 1e-5


def test_batched_candidate_failure_does_not_discard_other_histories(tmp_path, monkeypatch):
    trainer = load_trainer()
    p = wt.initial_parameters()
    raw = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=raw.tolist()), active_material_parameters=list(wt.NAMES),
                  train_network=False, channels=["q_MPa"])
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    x = wt.encode(p)
    baseline = objective.residual_vector(x)
    accepted_parameters = json.loads((tmp_path / "checkpoint.json").read_text())["material_parameters"]
    simulate = wt.simulate

    def selective_failure(parameters, *args, **kwargs):
        if parameters["E_parallel"] > p["E_parallel"]*1.01:
            raise wt.ConstitutiveSolveError("one unresolved batch member")
        return simulate(parameters, *args, **kwargs)

    monkeypatch.setattr(wt, "simulate", selective_failure)
    failed = x.copy()
    failed[0] += .1
    results = objective.residual_vectors([x, failed, x])
    np.testing.assert_array_equal(results[0], baseline)
    assert np.isinf(results[1]).all()
    np.testing.assert_array_equal(results[2], baseline)
    assert objective.evaluations == 4
    assert objective.unsolved_evaluations == 1
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["material_parameters"] == accepted_parameters
    assert saved["optimizer"]["last_unsolved_candidate"]["evaluation"] == 3


def test_batched_jacobian_handles_a_failed_side_and_a_bound():
    trainer = load_trainer()

    class History:
        def residual_vector(self, x):
            assert 0 <= x[1] <= 1
            if x[0] > .40005:
                return np.array([np.inf])
            return np.array([x[0]**2+2*x[1]])

        def residual_vectors(self, coordinates):
            return [self.residual_vector(x) for x in coordinates]

    jacobian = trainer.history_jacobian(History(), np.array([.4, 1.]),
                                      [(0., 1.), (0., 1.)], batch_size=2)
    np.testing.assert_allclose(jacobian, [[.7999, 2.]], rtol=1e-9)


@pytest.mark.parametrize("coordinate_origin", ["absolute", "checkpoint"])
def test_history_optimizer_rejects_unresolved_trials_and_respects_bounds(coordinate_origin):
    trainer = load_trainer()

    class BoundedHistory:
        rejected = 0

        def residual_vector(self, x):
            assert 0 <= x[0] <= 1.
            if x[0] > .4:
                self.rejected += 1
                return np.array([np.inf])
            return np.array([np.exp(4*x[0])-np.exp(1.6)])

    objective = BoundedHistory()
    result = trainer.fit_history_least_squares(objective, np.array([.05]), [(0., 1.)], 30,
                                              coordinate_origin=coordinate_origin)
    assert result.success
    assert result.x[0] == pytest.approx(.4, abs=1e-6)
    assert objective.rejected > 0


@pytest.mark.parametrize("trust_solver", ["exact", "lsmr"])
def test_checkpoint_origin_does_not_collapse_near_a_material_bound(trust_solver):
    trainer = load_trainer()

    class History:
        def residual_vector(self, x):
            assert .01 <= x[0] <= .48
            return x-np.array([.3, -2.])

    start = np.array([.0100000001, -4.751712965572927])
    result = trainer.fit_history_least_squares(History(), start, [(.01, .48), (None, None)],
        20, trust_solver=trust_solver, coordinate_scaling="unit", coordinate_origin="checkpoint")
    np.testing.assert_allclose(result.x, [.3, -2.], atol=1e-7)


def test_saved_history_jacobian_requires_the_same_point_residual_bounds_and_step(tmp_path):
    trainer = load_trainer()

    class History:
        def __init__(self):
            self.calls = 0
            self.offset = 0.

        def residual_vector(self, x):
            self.calls += 1
            return np.array([x[0]**2+self.offset])

    objective = History()
    destination = tmp_path/"linearization.npz"
    options = dict(linearization_path=destination, step=1e-5)
    x = np.array([.2])
    first = trainer.history_jacobian(objective, x, [(0., 1.)], **options)
    assert objective.calls == 2
    reused = trainer.history_jacobian(objective, x, [(0., 1.)], **options)
    np.testing.assert_array_equal(reused, first)
    assert objective.calls == 3  # Current residual still checked against the saved one.
    objective.offset = 1.
    trainer.history_jacobian(objective, x, [(0., 1.)], **options)
    assert objective.calls == 5
    trainer.history_jacobian(objective, x, [(0., 2.)], **options)
    assert objective.calls == 7
    trainer.history_jacobian(objective, x+.1, [(0., 2.)], **options)
    assert objective.calls == 9
    trainer.history_jacobian(objective, x+.1, [(0., 2.)], **{**options, "step": 1e-6})
    assert objective.calls == 11


@pytest.mark.parametrize("trust_solver,coordinate_origin", [("lsmr", "absolute"),
                                                           ("exact", "checkpoint")])
def test_regularized_fit_resolves_material_directions_with_saturated_storage(tmp_path, trust_solver, coordinate_origin):
    trainer = load_trainer()
    p = wt.initial_parameters()
    raw = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    raw[:4] = -300.
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=raw.tolist()),
                  active_material_parameters=["matrix_friction", "damage_resistance"],
                  train_network=True, channels=["q_MPa", "pressure_MPa"])
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    start = np.r_[wt.encode(p, objective.names), raw]
    bounds = wt.coordinate_bounds(objective.names)+[(None, None)]*len(raw)
    before = objective(start)
    jacobian = trainer.history_jacobian(objective, start, bounds)
    # The complete physical histories are insensitive to the saturated
    # storage weights, while friction and resistance remain observable.
    np.testing.assert_array_equal(jacobian[:, 2:], 0.)
    assert np.linalg.matrix_rank(jacobian[:, :2]) == 2
    result = trainer.fit_history_least_squares(objective, start, bounds, 5,
                                              trust_solver=trust_solver, coordinate_scaling="unit",
                                              coordinate_origin=coordinate_origin)
    assert np.isfinite(result.fun).all()
    assert objective.best_loss < before
    accepted = json.loads((tmp_path / "checkpoint.json").read_text())
    checks = next(iter(accepted["metrics"].values()))["physical_checks"]
    assert checks["min_density_increment"] >= -1e-10
    assert checks["min_dissipation_MPa"] >= -1e-10
    assert checks["max_fluid_balance"] < 1e-10
    assert checks["max_pressure_partition_MPa"] < 1e-9
