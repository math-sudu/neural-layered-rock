"""Evaluate portable parameter inputs without changing fitted states."""

import importlib.util
import csv
import gzip
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


@pytest.mark.parametrize("channel,volume_weight", [("stress", 0.), ("joint", 1.)])
@pytest.mark.parametrize("storage_form,fabric_source", [("neural", "plastic_stress"),
    ("identity", "plastic_stress"), ("identity", "matrix_force")])
def test_portable_input_evaluates_real_history_and_preserves_parameters(
        tmp_path, monkeypatch, channel, volume_weight, storage_form, fabric_source):
    trainer = trainer_module()
    parameters = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(parameters["initial_density"],
                                               parameters["damage_peak_density"])
    input_size = 3 if fabric_source == "matrix_force" else 2
    evolution = MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"], input_size=input_size)
    record = dict(channels=["q_MPa"] if channel == "stress" else ["q_MPa", "pressure_MPa"],
                  volume_stage_weight=volume_weight, response_measure="cell", material_parameters=parameters,
                  network=dict(raw_parameters=storage.raw_parameters.tolist(), hidden_width=4),
                  evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                                         hidden_width=4, pressure_scale_MPa=10.,
                                         input_size=input_size, fabric_source=fabric_source),
                  samples=5, max_increment=.0001)
    if storage_form == "identity":
        record.update(storage_form=storage_form, network=dict(raw_parameters=[], hidden_width=0))
    source = tmp_path / "parameters.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    original = source.read_bytes()
    specimen = load_specimens("test")[0]
    end = int(np.flatnonzero(specimen["history"]["axial_strain"] > .001)[0])+1
    specimen["history"] = {key: value[:end] for key, value in specimen["history"].items()}
    monkeypatch.setattr(trainer, "load_specimens", lambda role, **kwargs: [specimen])
    output = tmp_path / "predictions"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "evaluate",
        "--parameters", str(source), "--output", str(output),
        "--channels", channel, "--role", "test", "--workers", "1"])
    trainer.main()
    saved = json.loads((output / "test_evaluation.json").read_text())
    assert source.read_bytes() == original
    assert not (output / "checkpoint.json").exists()
    for key in ("material_parameters", "network", "evolution_network"):
        assert saved[key] == record[key]
    assert saved["storage_form"] == storage_form
    path = wt.prepare_path(specimen, 5)
    expected = wt.simulate(parameters, storage.raw_parameters, path, .0001,
                           evolution.raw_parameters, input_size, storage_form=storage_form,
                           fabric_source=fabric_source)
    assert saved["metrics"] == {specimen["specimen_id"]: wt.path_metrics(path, expected)}
    residual = wt.residuals(path, expected, record["channels"], response_measure="cell",
                            volume_stage_weight=volume_weight)
    assert saved["loss"] == pytest.approx(residual@residual)
    assert saved["volume_stage_weight"] == volume_weight
    rows = np.genfromtxt(output / "test_evaluation_paths.csv", delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    np.testing.assert_array_equal(rows["source_row"], path["observations"]["source_row"])
    np.testing.assert_array_equal(rows["observed_pressure_MPa"], path["observations"]["pressure_MPa"])


@pytest.mark.parametrize("fabric_source", ["plastic_stress", "matrix_force"])
def test_analytical_training_coordinates_and_inherited_closure(tmp_path, monkeypatch, fabric_source):
    trainer = trainer_module()
    parameters = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(parameters["initial_density"],
                                               parameters["damage_peak_density"])
    evolution = MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"])
    source = dict(dataset="Winhausen2023 Mont Terri", channels=["q_MPa"],
                  training_specimens=[p["specimen_id"] for p in load_specimens("train")],
                  material_parameters=parameters,
                  network=dict(raw_parameters=storage.raw_parameters.tolist(), hidden_width=4),
                  evolution_network=dict(raw_parameters=evolution.raw_parameters.tolist(),
                                         hidden_width=4, input_size=2, pressure_scale_MPa=10.),
                  samples=5, max_increment=.0001)
    source_file = tmp_path / "neural.json"
    source_file.write_text(json.dumps(source), encoding="utf-8")
    specimen = load_specimens("train")[0]
    specimen["history"] = {key: value[:6] for key, value in specimen["history"].items()}
    path = wt.prepare_path(specimen, 5)
    monkeypatch.setattr(wt, "training_paths", lambda *args: [path])
    original_call = trainer.TrainingLoss.__call__

    def check_coordinates(self, coordinates):
        input_size = 3 if fabric_source == "matrix_force" else 2
        assert len(coordinates) == 22+4*(input_size+3)
        assert len(self.names) == 22
        assert not self.train_network
        p, raw_storage, raw_evolution = self.unpack(coordinates)
        assert len(raw_storage) == 0
        assert len(raw_evolution) == 4*(input_size+3)
        assert self.fabric_source == fabric_source
        direct = wt.simulate(p, raw_storage, path, .0001, raw_evolution, input_size,
                             storage_form="identity", fabric_source=fabric_source)
        expected = wt.residuals(path, direct, self.record["channels"])
        assert original_call(self, coordinates) == pytest.approx(expected @ expected)
        return float(expected @ expected)

    monkeypatch.setattr(trainer.TrainingLoss, "__call__", check_coordinates)
    architecture = (["--fabric-evolution", "--fabric-source", "matrix_force"]
                    if fabric_source == "matrix_force" else [])
    for channel, parent, options in (
            ("stress", source_file, ["--storage-form", "identity", *architecture]),
            ("joint", tmp_path / "stress/checkpoint.json", [])):
        output = tmp_path / channel
        monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "initialize",
            "--channels", channel, "--initialize-from", str(parent), "--output", str(output),
            "--workers", "1", *options])
        trainer.main()
        saved = json.loads((output / "checkpoint.json").read_text())
        assert saved["storage_form"] == "identity"
        assert saved["network"] == dict(raw_parameters=[], hidden_width=0)
        assert saved["evolution_network"]["fabric_source"] == fabric_source
        assert saved["training_specimens"] == source["training_specimens"]
    assert json.loads(source_file.read_text()) == source


def test_mechanism_trace_uses_saved_analytical_storage(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/trace_winhausen_mechanisms.py"
    spec = importlib.util.spec_from_file_location("mechanism_tracer", script)
    tracer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracer)
    parameters = wt.initial_parameters()
    source = tmp_path / "parameters.json"
    source.write_text(json.dumps(dict(material_parameters=parameters, storage_form="identity",
                                     network=dict(raw_parameters=[], hidden_width=0))), encoding="utf-8")
    specimen = load_specimens("train")[0]
    specimen["history"] = {key: value[:6] for key, value in specimen["history"].items()}
    tracer.trace_one((source, specimen, 5, .0001, tmp_path, None, None, None))
    saved = json.loads((tmp_path / f"{specimen['specimen_id']}.json").read_text())
    assert saved["storage_form"] == "identity"
    with gzip.open(tmp_path / f"{specimen['specimen_id']}.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        assert float(row["storage_coordinate"]) == float(row["density"])


def test_portable_parameters_cannot_silently_replace_a_training_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "train",
        "--parameters", str(tmp_path / "parameters.json"), "--output", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as exc:
        trainer_module().main()
    assert exc.value.code == 2
    assert "applies only to evaluate" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_evaluation_uses_saved_experiment_split(tmp_path, monkeypatch):
    trainer = trainer_module()
    parameters = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(parameters["initial_density"],
                                               parameters["damage_peak_density"])
    selected = [s["specimen_id"] for s in load_specimens("train")]+["OPA15", "OPA24"]
    record = dict(channels=["q_MPa"], training_specimens=selected,
                  material_parameters=parameters,
                  network=dict(raw_parameters=storage.raw_parameters.tolist(), hidden_width=4),
                  samples=5, max_increment=.0001)
    source = tmp_path / "parameters.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    def short_histories(role, **kwargs):
        specimens = load_specimens(role, **kwargs)
        for specimen in specimens:
            specimen["history"] = {k: v[:6] for k, v in specimen["history"].items()}
        return specimens

    monkeypatch.setattr(trainer, "load_specimens", short_histories)
    output = tmp_path / "predictions"
    monkeypatch.setattr(sys, "argv", ["train_winhausen_neural.py", "evaluate",
        "--parameters", str(source), "--output", str(output),
        "--channels", "stress", "--role", "test", "--workers", "1"])
    trainer.main()
    saved = json.loads((output / "test_evaluation.json").read_text())
    assert saved["training_specimens"] == selected
    assert set(saved["metrics"]) == {"OPA17", "OPA18", "OPA23"}
    assert json.loads(source.read_text()) == record
    rows = np.genfromtxt(output / "test_evaluation_paths.csv", delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    assert set(rows["specimen_id"]) == set(saved["metrics"])
    assert set(rows["role"]) == {"test"}
