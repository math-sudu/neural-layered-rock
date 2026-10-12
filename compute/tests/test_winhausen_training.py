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


def test_full_history_preserves_state_through_unloading_and_fluid_constraint():
    path = wt.training_paths(25)[1]
    indices = np.array([0, 1, 2, 3, 4])
    path["observations"] = {name: value[indices] for name, value in path["observations"].items()}
    path["control_strain"] = np.array([0., .004, .009, .004, .011])
    p = wt.initial_parameters()
    net = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    result = wt.simulate(p, net.raw_parameters, path)
    rows = result["rows"]
    assert [row["axial_strain"] for row in rows] == list(path["control_strain"])
    assert rows[3]["density"] >= rows[2]["density"]-1e-8
    assert result["diagnostics"]["min_dissipation_MPa"] > -1e-7
    assert result["diagnostics"]["max_fluid_balance"] < 1e-10
    assert result["diagnostics"]["max_pressure_partition_MPa"] < 1e-9
    for row in rows:
        parts = [row[f"pressure_{part}_MPa"] for part in ("stress", "matrix", "plane", "compaction")]
        assert sum(parts) == pytest.approx(row["pressure_MPa"], abs=1e-9)
        angle = np.deg2rad(path["angle"])
        physical_shear = abs(row["q_MPa"]*np.sin(angle)*np.cos(angle))
        assert row["plane_shear_traction_MPa"] == pytest.approx(physical_shear, abs=1e-9)
        assert row["plane_yield_MPa"] <= 1e-7
    assert result["diagnostics"]["min_cap_slack_MPa"] > 0.
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


def test_stress_objective_does_not_consume_pressure_targets():
    path = wt.training_paths(25)[0]
    rows = [{"q_MPa": float(x), "pressure_MPa": 0.} for x in path["observations"]["q_MPa"]]
    result = dict(rows=rows, q_peak=(path["full_observed_peaks"]["q_MPa"], 0.), pressure_peak=(0., 0.))
    before = wt.residuals(path, result, ["q_MPa"])
    path["observations"]["pressure_MPa"][:] = 1234.
    path["full_observed_peaks"]["pressure_MPa"] = 1234.
    np.testing.assert_array_equal(wt.residuals(path, result, ["q_MPa"]), before)


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

    def unsolved(*args):
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

    def programming_error(*args):
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


def test_joint_coordinates_reach_both_networks_and_failed_candidate_record(tmp_path, monkeypatch):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=evolution.tolist()),
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

    def unsolved(parameters, raw_storage, path, increment, raw_evolution):
        assert parameters["matrix_friction"] == candidate[0]
        np.testing.assert_array_equal(raw_storage, candidate[1:13])
        np.testing.assert_array_equal(raw_evolution, candidate[13:])
        raise wt.ConstitutiveSolveError("test unresolved coupled candidate")

    monkeypatch.setattr(wt, "simulate", unsolved)
    assert np.isinf(objective(candidate))
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    np.testing.assert_array_equal(saved["evolution_network"]["raw_parameters"], evolution)
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


@pytest.mark.parametrize("field,value", [("dataset", "different material"),
                                        ("training_specimens", ["OPA23"])])
def test_adding_pressure_keeps_training_population_fixed(tmp_path, monkeypatch, capsys, field, value):
    source = dict(channels=["q_MPa"], dataset="Winhausen2023 Mont Terri",
                  training_specimens=[path["specimen_id"] for path in load_specimens("train")])
    source[field] = value
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "joint"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
        "--channels", "joint", "--output", str(output), "--initialize-from", str(source_file)])
    with pytest.raises(SystemExit) as exc:
        load_trainer().main()
    assert exc.value.code == 2
    assert "same dataset and training specimens" in capsys.readouterr().err
    assert not (output / "checkpoint.json").exists()


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


def test_history_jacobian_predicts_coupled_loading_residual_changes(tmp_path):
    trainer = load_trainer()
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"]).raw_parameters
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters
    record = dict(samples=10, max_increment=.0005, material_parameters=p,
                  network=dict(raw_parameters=storage.tolist()),
                  evolution_network=dict(raw_parameters=evolution.tolist()),
                  active_material_parameters=["matrix_friction", "damage_resistance"],
                  train_network=False, train_evolution=False, channels=["q_MPa", "pressure_MPa"])
    objective = trainer.TrainingLoss(record, tmp_path, map)
    objective.paths = objective.paths[:1]
    x = wt.encode(p, objective.names)
    residual = objective.residual_vector(x)
    jacobian = trainer.history_jacobian(objective, x, wt.coordinate_bounds(objective.names))
    batched = trainer.history_jacobian(objective, x, wt.coordinate_bounds(objective.names),
                                     batch_size=2)
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

    def selective_failure(parameters, *args):
        if parameters["E_parallel"] > p["E_parallel"]*1.01:
            raise wt.ConstitutiveSolveError("one unresolved batch member")
        return simulate(parameters, *args)

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


def test_history_optimizer_rejects_unresolved_trials_and_respects_bounds():
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
    result = trainer.fit_history_least_squares(objective, np.array([.05]), [(0., 1.)], 30)
    assert result.success
    assert result.x[0] == pytest.approx(.4, abs=1e-6)
    assert objective.rejected > 0
