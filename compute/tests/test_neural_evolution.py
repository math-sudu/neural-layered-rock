"""Pressure/history derivatives, objectivity and dissipation in matrix learning."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib import coupled_friction_damage as coupled
from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.neural_evolution import MatrixEvolutionNetwork, NeuralFrictionDamageLaw
from tfenn_invcalib.neural_storage import MonotoneStorageNetwork
from tfenn_invcalib.poromechanical_coupling import mandel_rotation


def model_with_network():
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    raw = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters.copy()
    raw[12:] = [.12, -.15, .1, -.08, -.1, .15, .12, -.09]
    return wt.material(p, dict(angle=30., confinement=10.), storage.raw_parameters, raw)


def test_neural_evolution_partials_and_admissibility():
    model = model_with_network()
    law = model.law
    for d in (.08, .4, 3., 40.):
        for pressure in (-2., 4., 16.):
            r, rd, rp, beta, bd, bp = law.evolution(d, pressure)
            assert r > 0 and beta <= law.friction(d)[0]
            hd, hp = d*1e-5, 1e-4
            left, right = law.evolution(d-hd, pressure), law.evolution(d+hd, pressure)
            np.testing.assert_allclose([rd, bd],
                (np.array(right)[[0, 3]]-np.array(left)[[0, 3]])/(2*hd), rtol=2e-6, atol=1e-9)
            left, right = law.evolution(d, pressure-hp), law.evolution(d, pressure+hp)
            np.testing.assert_allclose([rp, bp],
                (np.array(right)[[0, 3]]-np.array(left)[[0, 3]])/(2*hp), rtol=2e-6, atol=1e-9)
    g, _, _ = law.network.value_derivatives(law.damage_peak_density, law.network.pressure_scale)
    np.testing.assert_array_equal(g, [0., 0.])
    with pytest.raises(ValueError, match="effective mean stress"):
        law.evolution(.1, None)


def test_initial_outputs_recover_matrix_law_without_removing_trainable_features():
    neural = model_with_network().law
    initial = replace(neural, network=MatrixEvolutionNetwork.initialize(neural.damage_peak_density))
    for d in (.06, .3, 2.):
        base_r, base_rd = initial.resistance(d)
        base_beta, base_bd = initial.dilatancy(d)
        np.testing.assert_allclose(initial.evolution(d, 14.),
                                   [base_r, base_rd, 0., base_beta, base_bd, 0.], atol=1e-15)
    assert neural.evolution(.3, 4.)[0] != neural.evolution(.3, 14.)[0]
    assert neural.evolution(.3, 4.)[3] != neural.evolution(.3, 14.)[3]


@pytest.mark.parametrize("drained", [False, True])
def test_pressure_dependent_implicit_jacobian(monkeypatch, drained):
    model = model_with_network()
    original = coupled.least_squares
    initial = model.consolidated_state(10., .05)

    def checked(fun, x0, *, jac, **kwargs):
        probe = x0+np.array([.5, -.1, -.2, .05, .03, .02, .2, -.1, -.02, .2, .3, .1, -.2])
        increments = 2e-6*np.eye(len(x0))
        numerical = np.column_stack([(fun(probe+h)-fun(probe-h))/(4e-6) for h in increments])
        np.testing.assert_allclose(jac(probe), numerical, rtol=3e-5, atol=2e-6)
        return original(fun, x0, jac=jac, **kwargs)

    monkeypatch.setattr(coupled, "least_squares", checked)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    model.step(initial["strain"][0]+.003, [10., 10., 0., 0., 0.], initial, **boundary)


def test_recorded_opa06_activation_reaches_an_admissible_damage_branch(monkeypatch):
    case = json.loads((Path(__file__).parent / "data/winhausen_activation.json").read_text())
    model = wt.material(case["material_parameters"], case["path"],
                        np.array(case["storage_parameters"]), np.array(case["evolution_parameters"]))
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    original = coupled.least_squares

    def checked(fun, x0, *, jac, **kwargs):
        result = original(fun, x0, jac=jac, **kwargs)
        if max(abs(result.fun)) < 1e-7:
            increments = 2e-6*np.eye(len(x0))
            numerical = np.column_stack([(fun(result.x+h)-fun(result.x-h))/(4e-6)
                                         for h in increments])
            np.testing.assert_allclose(jac(result.x), numerical, rtol=3e-5, atol=2e-6)
        return result

    monkeypatch.setattr(coupled, "least_squares", checked)
    p0 = case["path"]["confinement"]
    state = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=case["fluid_content"])
    f = state["response"]
    assert np.log(state["d"]/previous["d"]) == pytest.approx(.04659657763352, abs=1e-8)
    assert f["total_stress"][0]-p0 == pytest.approx(6.59262245702, abs=1e-7)
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    assert state["matrix_yield"] <= 1e-7 and state["plane_yield"] <= 1e-7
    assert min(state["dissipation_increments"].values()) >= -1e-10
    assert state["fluid_content"] == case["fluid_content"]


def test_evolution_and_flow_are_objective_with_anisotropic_history():
    model = model_with_network()
    axis = np.array([1., 2., -1.]); axis /= np.linalg.norm(axis)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+np.sin(.73)*skew+(1-np.cos(.73))*(skew@skew)
    Q = mandel_rotation(rotation)
    m = model.energy
    from tfenn_invcalib.neural_storage import NeuralPorousEnergy
    rotated_energy = NeuralPorousEnergy(wt.elasticity(wt.initial_parameters(), 10.),
        rotation@np.array([.5, 0., np.sqrt(.75)]), network=m.network, **wt.FIXED)
    rotated = coupled.CoupledFrictionDamage(rotated_energy, model.law, model.D)
    strain = np.array([.004, .001, -.0002, .0004, -.0001, .0003])
    cm, z = strain/5, np.array([-.0001, .0003, -.0002])
    zr = rotated_energy.B.T @ Q @ m.B @ z
    a = m.response(strain, cm, z, 0., .4, model.D, -.0001)
    b = rotated_energy.response(Q@strain, Q@cm, zr, 0., .4, model.D, -.0001)
    pa, pb = VOLUME@a["plastic_stress"]/3, VOLUME@b["plastic_stress"]/3
    np.testing.assert_allclose(model.law.evolution(.4, pa), rotated.law.evolution(.4, pb), atol=1e-12)
    fa = model.matrix_surface(a["matrix_force"], .4, effective_mean=pa)
    fb = rotated.matrix_surface(b["matrix_force"], .4, effective_mean=pb)
    assert fb[0] == pytest.approx(fa[0], abs=1e-10)
    np.testing.assert_allclose(fb[1], Q@fa[1], atol=1e-10)


@pytest.mark.parametrize("filename", ["winhausen_matrix_corner.json", "winhausen_matrix_corner_transition.json",
                                      "winhausen_refined_matrix_activation.json",
                                      "winhausen_refined_matrix_activation_inclined.json",
                                      "winhausen_corner_nonnegative_multiplier.json"])
def test_recorded_matrix_mean_corner_resolves_partial_contraction(monkeypatch, filename):
    case = json.loads((Path(__file__).parent / "data" / filename).read_text())
    model = wt.material(case["material_parameters"], case["path"],
                        np.array(case["storage_parameters"]), np.array(case["evolution_parameters"]))
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    original = coupled.least_squares
    checked_corner = []

    def checked(fun, x0, *, jac, **kwargs):
        result = original(fun, x0, jac=jac, **kwargs)
        if max(abs(result.fun)) < 1e-7 and 0 < result.x[12] < 1:
            probe = result.x.copy()
            density = previous["d"]*np.exp(probe[10])
            at_origin = abs(density-case["material_parameters"]["initial_density"]) < 1e-10
            if at_origin:
                probe[10] = np.log(case["material_parameters"]["initial_density"]/previous["d"])
            increments = 2e-6*np.eye(len(x0))
            numerical = np.column_stack([(fun(probe+h)-fun(probe-h))/(4e-6)
                                         for h in increments])
            # Undamaged corners can lie at the initial-density dilatancy
            # knot; only there use the increasing-density derivative.
            if at_origin:
                numerical[:, 10] = (-3*fun(probe)+4*fun(probe+increments[10])
                                     -fun(probe+2*increments[10]))/(4e-6)
            # Flow and the corner equation are smooth here. The yield row
            # retains the valid one-sided derivative of positive-part pressure.
            rows = [0, 1, 2, 3, 4, 5, 10, 12]
            np.testing.assert_allclose(jac(probe)[rows], numerical[rows], rtol=3e-5, atol=2e-6)
            checked_corner.append(True)
        return result

    monkeypatch.setattr(coupled, "least_squares", checked)
    p0 = case["path"]["confinement"]
    state = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=case["fluid_content"])
    assert abs(state["matrix_mean_force"]) < 1e-7
    assert 0 < state["matrix_flow_fraction"] < 1
    assert state["matrix_dilatancy"] < 0 and state["matrix_multiplier"] > 0
    increment = state["matrix_strain"]-previous["matrix_strain"]
    assert VOLUME@increment == pytest.approx(-state["matrix_multiplier"]
        *state["matrix_dilatancy"]*state["matrix_flow_fraction"], abs=1e-11)
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-7)
    assert state["d"] >= previous["d"]-1e-10
    release = state["response"]["matrix_release"]
    assert release <= state["damage_resistance"]+1e-10
    assert (state["d"]-previous["d"])*(release-state["damage_resistance"]) == pytest.approx(0., abs=1e-10)
    if filename == "winhausen_refined_matrix_activation.json":
        assert state["d"] > previous["d"]+.01
    assert min(state["dissipation_increments"].values()) >= -1e-10
    np.testing.assert_allclose(state["response"]["total_stress"][1:], [p0, p0, 0., 0., 0.], atol=1e-9)
    assert state["fluid_content"] == case["fluid_content"]
    assert checked_corner


def test_corner_seed_reaches_adjacent_positive_mean_force_damage_jump():
    case = json.loads((Path(__file__).parent / "data/winhausen_post_corner_release.json").read_text())
    model = wt.material(case["material_parameters"], case["path"],
                        np.array(case["storage_parameters"]), np.array(case["evolution_parameters"]))
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    p0 = case["path"]["confinement"]
    state = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=case["fluid_content"])
    assert state["matrix_mean_force"] > .01
    assert state["matrix_flow_fraction"] == pytest.approx(1., abs=1e-9)
    assert state["matrix_dilatancy"] < 0 and state["matrix_multiplier"] > 0
    assert state["d"] > 2*previous["d"]
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-7)
    assert state["response"]["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    increment = state["matrix_strain"]-previous["matrix_strain"]
    assert VOLUME@increment == pytest.approx(-state["matrix_multiplier"]*state["matrix_dilatancy"], abs=1e-11)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    np.testing.assert_allclose(state["response"]["total_stress"][1:], [p0, p0, 0., 0., 0.], atol=1e-9)
    assert state["fluid_content"] == case["fluid_content"]


@pytest.mark.parametrize("drained", [False, True])
def test_neural_evolution_loading_and_unloading_preserve_physical_state(drained):
    model = model_with_network()
    initial = previous = model.consolidated_state(10., .05)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    history = np.r_[np.linspace(0., .009, 30), np.linspace(.009, .004, 12)[1:],
                    np.linspace(.004, .012, 24)[1:]]
    for axial in history:
        state = model.step(initial["strain"][0]+axial, [10., 10., 0., 0., 0.], previous, **boundary)
        f = state["response"]
        assert state["d"] >= previous["d"]-1e-8
        assert state["compaction"] >= previous["compaction"]-1e-10
        assert min(state["dissipation_increments"].values()) > -1e-7
        assert state["damage_resistance"] > 0
        assert state["matrix_dilatancy"] <= model.law.friction(state["d"])[0]
        assert f["matrix_release"] <= state["damage_resistance"]+1e-7
        if drained:
            assert abs(f["pressure"]) < 1e-9
        else:
            assert state["fluid_content"] == initial["fluid_content"]
        previous = state
    assert state["d"] > initial["d"]+.01
