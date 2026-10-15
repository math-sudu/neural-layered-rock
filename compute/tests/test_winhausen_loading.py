"""Conditional normal-bedding reconstruction on a complete forward history."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from diagnose_winhausen_loading import normal_matrix_state, stage_projections
from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.neural_evolution import MatrixEvolutionNetwork
from tfenn_invcalib.winhausen_data import load_specimens
import trace_winhausen_mechanisms as tracing


def test_stage_projections_separate_mean_stress_contact_reaction_and_matrix_force():
    # At 30 degrees, q increases p' but decreases normal contact reaction
    # through the stress-induced pressure. Volume change offsets both.
    rows = []
    for axial in (0., .002):
        t = axial/.002
        q = 5.+2*t
        matrix_pressure, plane_pressure, cap_pressure = -.3*t, .08*t, .02*t
        pressure = 1.+.3*(q-5.)+matrix_pressure+plane_pressure+cap_pressure
        backstress = .04*t
        rows.append(dict(axial_strain=axial,
            effective_mean_MPa=10.+q/3-pressure,
            matrix_mean_force_MPa=2.+.1*t,
            plane_normal_force_MPa=10.+.25*q-pressure-backstress,
            plane_backstress_MPa_0=backstress,
            pressure_matrix_MPa=matrix_pressure, pressure_plane_MPa=plane_pressure,
            pressure_compaction_MPa=cap_pressure))
    interval = dict(start_strain=.0005, end_strain=.0015, predicted_delta_q_MPa=1.)
    result = stage_projections(interval, rows, dict(angle_deg=30., pressure_stress_coefficient=.3))
    np.testing.assert_allclose([result["delta_effective_mean_MPa"],
        result["delta_plane_reaction_MPa"], result["delta_matrix_mean_force_MPa"]],
        [1/30+.1, .03, .05])
    np.testing.assert_allclose([result["plane_reaction_stress_contribution_MPa"],
        result["plane_reaction_matrix_volume_contribution_MPa"],
        result["plane_reaction_backstress_contribution_MPa"]], [-.05, .15, -.02])
    zero = dict(interval, end_strain=interval["start_strain"], predicted_delta_q_MPa=0.)
    assert stage_projections(zero, rows, dict(angle_deg=30., pressure_stress_coefficient=.3))[
        "delta_plane_reaction_MPa"] == 0.


def test_conditional_normal_state_recovers_forward_damage_and_matrix_strain():
    parameters = wt.initial_parameters()
    path = next(p for p in wt.training_paths(8) if p["specimen_id"] == "OPA22")
    weights = MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"]).raw_parameters
    model = wt.material(parameters, path, [], weights, storage_form="identity")
    initial = wt.initial_state(model, parameters, path)
    result = wt.simulate(parameters, [], path, .0005, weights, storage_form="identity")
    active = [row for row in result["rows"] if row["density"] > parameters["initial_density"]*1.1
              and abs(row["matrix_yield_MPa"]) < 1e-8 and row["matrix_mean_force_MPa"] > 0]
    assert active
    for row in (active[0], active[-1]):
        matrix, density, resistance, _ = normal_matrix_state(model, initial, path,
            row["axial_strain"], row["q_MPa"], row["pressure_MPa"])
        np.testing.assert_allclose([matrix[0], sum(matrix[:3])],
                                   [row["matrix_axial"], row["matrix_volume"]], atol=1e-12)
        assert density == pytest.approx(row["density"], rel=1e-7)
        assert resistance == pytest.approx(row["matrix_release_MPa"], rel=2e-7)


def test_failed_trace_replay_removes_stale_completion_metadata(tmp_path, monkeypatch):
    parameters = wt.initial_parameters()
    weights = MatrixEvolutionNetwork.initialize(parameters["damage_peak_density"]).raw_parameters
    source = tmp_path/"parameters.json"
    source.write_text(json.dumps(dict(material_parameters=parameters, storage_form="identity",
        network=dict(raw_parameters=[]), evolution_network=dict(raw_parameters=weights.tolist(), input_size=2))))
    specimen = load_specimens(None)[0]
    metadata = tmp_path/f"{specimen['specimen_id']}.json"
    metadata.write_text('{"parameters": "previous-parameters.json"}')

    def fail(*args, **kwargs):
        raise RuntimeError("Unresolved constitutive path")

    monkeypatch.setattr(tracing.wt, "simulate", fail)
    with pytest.raises(RuntimeError, match="Unresolved constitutive path"):
        tracing.trace_one((source, specimen, 8, .0005, tmp_path, None, None, None))
    assert not metadata.exists()
