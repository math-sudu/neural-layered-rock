"""Evaluate portable parameter inputs without changing fitted states."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.neural_storage import MonotoneStorageNetwork
from tfenn_invcalib.neural_evolution import MatrixEvolutionNetwork
from tfenn_invcalib.winhausen_data import load_specimens


def trainer_module():
    script = Path(__file__).resolve().parents[1] / "scripts/train_winhausen_neural.py"
    spec = importlib.util.spec_from_file_location("neural_trainer", script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_portable_input_evaluates_real_history_and_preserves_parameters(tmp_path, monkeypatch):
    trainer = trainer_module()
    parameters = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(parameters["initial_density"],
                                               parameters["damage_peak_density"])
    evolution = MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"])
    record = dict(channels=["q_MPa"], material_parameters=parameters,
                  network=dict(raw_parameters=storage.raw_parameters.tolist(), hidden_width=4),
                  evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                                         hidden_width=4, pressure_scale_MPa=10.),
                  samples=5, max_increment=.0001)
    source = tmp_path / "parameters.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    original = source.read_bytes()
    specimen = load_specimens("test")[0]
    specimen["history"] = {key: value[:6] for key, value in specimen["history"].items()}
    monkeypatch.setattr(trainer, "load_specimens", lambda role: [specimen])
    output = tmp_path / "predictions"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "evaluate",
        "--parameters", str(source), "--output", str(output),
        "--channels", "stress", "--role", "test", "--workers", "1"])
    trainer.main()
    saved = json.loads((output / "test_evaluation.json").read_text())
    assert source.read_bytes() == original
    assert not (output / "checkpoint.json").exists()
    for key in ("material_parameters", "network", "evolution_network"):
        assert saved[key] == record[key]
    path = wt.prepare_path(specimen, 5)
    expected = wt.simulate(parameters, storage.raw_parameters, path, .0001,
                           evolution.raw_parameters)
    assert saved["metrics"] == {specimen["specimen_id"]: wt.path_metrics(path, expected)}
    rows = np.genfromtxt(output / "test_evaluation_paths.csv", delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    np.testing.assert_array_equal(rows["source_row"], path["observations"]["source_row"])
    np.testing.assert_array_equal(rows["observed_pressure_MPa"], path["observations"]["pressure_MPa"])


def test_portable_parameters_cannot_silently_replace_a_training_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "train",
        "--parameters", str(tmp_path / "parameters.json"), "--output", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as exc:
        trainer_module().main()
    assert exc.value.code == 2
    assert "applies only to evaluate" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
