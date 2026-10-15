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


def model_with_network(fabric=False, resistance_uses_pressure=True, fabric_source="plastic_stress"):
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    raw = MatrixEvolutionNetwork.initialize(p["damage_peak_density"]).raw_parameters.copy()
    raw[12:] = [.12, -.15, .1, -.08, -.1, .15, .12, -.09]
    if fabric:
        raw = MatrixEvolutionNetwork(raw, p["damage_peak_density"]).with_fabric_input().raw_parameters
        raw[[2, 5, 8, 11]] = [.4, -.6, .2, .7]
    return wt.material(p, dict(angle=30., confinement=10.), storage.raw_parameters, raw,
                       3 if fabric else 2, resistance_uses_pressure, fabric_source=fabric_source)


def test_neural_evolution_partials_and_admissibility():
    model = model_with_network()
    law = model.law
    for d in (.08, .4, 3., 40.):
        for pressure in (-2., 4., 16.):
            r, rd, rp, beta, bd, bp, rf, bf = law.evolution(d, pressure)
            assert r > 0 and beta <= law.friction(d)[0]
            hd, hp = d*1e-5, 1e-4
            left, right = law.evolution(d-hd, pressure), law.evolution(d+hd, pressure)
            np.testing.assert_allclose([rd, bd],
                (np.array(right)[[0, 3]]-np.array(left)[[0, 3]])/(2*hd), rtol=2e-6, atol=1e-9)
            left, right = law.evolution(d, pressure-hp), law.evolution(d, pressure+hp)
            np.testing.assert_allclose([rp, bp],
                (np.array(right)[[0, 3]]-np.array(left)[[0, 3]])/(2*hp), rtol=2e-6, atol=1e-9)
    g, *_ = law.network.value_derivatives(law.damage_peak_density, law.network.pressure_scale)
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
                                   [base_r, base_rd, 0., base_beta, base_bd, 0., 0., 0.], atol=1e-15)
    assert neural.evolution(.3, 4.)[0] != neural.evolution(.3, 14.)[0]
    assert neural.evolution(.3, 4.)[3] != neural.evolution(.3, 14.)[3]


@pytest.mark.parametrize("resistance_uses_pressure", [False, True])
def test_fabric_conditioning_preserves_reference_and_all_state_partials(resistance_uses_pressure):
    law = model_with_network(True, resistance_uses_pressure).law
    original = model_with_network().law.network
    embedded = original.with_fabric_input()
    for fabric in (-.5, 0., 1.):
        g, *_ = law.network.value_derivatives(law.damage_peak_density,
                                               law.network.pressure_scale, fabric)
        if fabric == 0.:
            np.testing.assert_allclose(g, 0., atol=1e-15)
        else:
            assert np.linalg.norm(g) > 0.
        np.testing.assert_allclose(embedded.value_derivatives(.4, 6., fabric),
                                   original.value_derivatives(.4, 6.), atol=1e-15)
        for d, pressure in ((.08, 4.), (.4, 16.), (3., 8.)):
            values = np.array(law.evolution(d, pressure, fabric))
            assert values[0] > 0 and values[3] <= law.friction(d)[0]
            for j, (step, indices) in enumerate(((d*1e-5, [1, 4]),
                                                 (1e-4, [2, 5]), (1e-4, [6, 7]))):
                left, right = np.array([d, pressure, fabric]), np.array([d, pressure, fabric])
                left[j] -= step
                right[j] += step
                numerical = (np.array(law.evolution(*right))[[0, 3]]
                             -np.array(law.evolution(*left))[[0, 3]])/(2*step)
                np.testing.assert_allclose(values[indices], numerical, rtol=2e-6, atol=1e-9)


def test_density_fabric_resistance_preserves_flow_pressure_dependence_and_reference():
    law = model_with_network(True, False).law
    low, high = law.evolution(.4, 4., .6), law.evolution(.4, 16., .6)
    np.testing.assert_array_equal(np.array(low)[[0, 1, 2, 6]], np.array(high)[[0, 1, 2, 6]])
    assert low[2] == 0. and low[3] != high[3]
    anchor = law.evolution(law.damage_peak_density, law.network.pressure_scale, 0.)
    np.testing.assert_allclose([anchor[0], anchor[3]],
        [law.resistance(law.damage_peak_density)[0], law.dilatancy(law.damage_peak_density)[0]], atol=1e-15)


def test_fabric_direction_is_objective_and_independent_of_triaxial_stress_magnitude():
    p = wt.initial_parameters()
    storage = MonotoneStorageNetwork.initialize(p["initial_density"], p["damage_peak_density"])
    evolution = MatrixEvolutionNetwork.initialize(p["damage_peak_density"], input_size=3)
    for angle in (0., 15., 45., 75., 90.):
        model = wt.material(p, dict(angle=angle, confinement=10.), storage.raw_parameters,
                            evolution.raw_parameters, 3)
        value, derivative = model.fabric_direction(10*VOLUME)
        assert value == 0.
        np.testing.assert_array_equal(derivative, 0.)
        for q in (.001, 12., 25.):
            stress = 10*VOLUME+np.array([q, 0., 0., 0., 0., 0.])
            expected = (3*np.sin(np.deg2rad(angle))**2-1)/2
            assert model.fabric_direction(stress)[0] == pytest.approx(expected, abs=1e-12)
        stress = np.array([14., 8., 6., 1., .5, -1.])
        _, gradient = model.fabric_direction(stress)
        increments = 1e-5*np.eye(6)
        numerical = [(model.fabric_direction(stress+h)[0]-model.fabric_direction(stress-h)[0])/2e-5
                     for h in increments]
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("drained", [False, True])
@pytest.mark.parametrize("fabric,fabric_source", [(False, "plastic_stress"), (True, "plastic_stress"),
                                                  (True, "matrix_force")])
@pytest.mark.parametrize("resistance_uses_pressure", [False, True])
def test_pressure_dependent_implicit_jacobian(monkeypatch, drained, fabric, fabric_source,
                                             resistance_uses_pressure):
    model = model_with_network(fabric, resistance_uses_pressure, fabric_source)
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
    transverse = [9.5, 10.5, .2, -.1, .15] if fabric else [10., 10., 0., 0., 0.]
    state = model.step(initial["strain"][0]+.003, transverse, initial, **boundary)
    expected = model.fabric_direction(state["response"][fabric_source])[0]
    assert state["fabric_projection"] == pytest.approx(expected, abs=1e-12)


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


@pytest.mark.parametrize("fabric,fabric_source", [(False, "plastic_stress"), (True, "plastic_stress"),
                                                  (True, "matrix_force")])
def test_evolution_and_flow_are_objective_with_anisotropic_history(fabric, fabric_source):
    model = model_with_network(fabric, fabric_source=fabric_source)
    axis = np.array([1., 2., -1.]); axis /= np.linalg.norm(axis)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+np.sin(.73)*skew+(1-np.cos(.73))*(skew@skew)
    Q = mandel_rotation(rotation)
    m = model.energy
    from tfenn_invcalib.neural_storage import NeuralPorousEnergy
    rotated_energy = NeuralPorousEnergy(wt.elasticity(wt.initial_parameters(), 10.),
        rotation@np.array([.5, 0., np.sqrt(.75)]), network=m.network, **wt.FIXED)
    rotated = coupled.CoupledFrictionDamage(rotated_energy, model.law, model.D, fabric_source=fabric_source)
    strain = np.array([.004, .001, -.0002, .0004, -.0001, .0003])
    cm, z = strain/5, np.array([-.0001, .0003, -.0002])
    zr = rotated_energy.B.T @ Q @ m.B @ z
    a = m.response(strain, cm, z, 0., .4, model.D, -.0001)
    b = rotated_energy.response(Q@strain, Q@cm, zr, 0., .4, model.D, -.0001)
    pa, pb = VOLUME@a["plastic_stress"]/3, VOLUME@b["plastic_stress"]/3
    ia = model.fabric_direction(a[fabric_source])[0]
    ib = rotated.fabric_direction(b[fabric_source])[0]
    if fabric_source == "matrix_force":
        assert ia != pytest.approx(model.fabric_direction(a["plastic_stress"])[0], abs=1e-8)
    assert ia == pytest.approx(ib, abs=1e-12)
    np.testing.assert_allclose(model.law.evolution(.4, pa, ia), rotated.law.evolution(.4, pb, ib), atol=1e-12)
    fa = model.matrix_surface(a["matrix_force"], .4, effective_mean=pa, fabric_projection=ia)
    fb = rotated.matrix_surface(b["matrix_force"], .4, effective_mean=pb, fabric_projection=ib)
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
    seeded = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                        fluid_content=case["fluid_content"], initial_guess=state)
    assert seeded["solver_route"] == "supplied_guess"
    np.testing.assert_allclose(seeded["matrix_strain"], state["matrix_strain"], atol=1e-11)
    assert seeded["d"] == pytest.approx(state["d"], abs=1e-9)
    assert seeded["matrix_multiplier"] == pytest.approx(state["matrix_multiplier"], abs=1e-11)


@pytest.mark.parametrize("pressure_weight_increment", [0., 1e-4, 2e-4])
@pytest.mark.parametrize("filename", ["winhausen_fabric_damage_predictor.json",
                                      "winhausen_fabric_activation_predictor.json"])
def test_failed_fixed_density_subproblem_retains_coupled_damage_root(monkeypatch, pressure_weight_increment, filename):
    case = json.loads((Path(__file__).parent / "data" / filename).read_text())
    raw = np.array(case["evolution_parameters"])
    raw[1] += pressure_weight_increment
    model = wt.material(case["material_parameters"], case["path"],
                        case["storage_parameters"], raw, case["evolution_inputs"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    original = coupled.least_squares
    checked_roots = []

    def checked(fun, x0, *, jac, **kwargs):
        result = original(fun, x0, jac=jac, **kwargs)
        if max(abs(result.fun)) < 1e-7:
            increments = 2e-6*np.eye(len(x0))
            numerical = np.column_stack([(fun(result.x+h)-fun(result.x-h))/(4e-6)
                                         for h in increments])
            np.testing.assert_allclose(jac(result.x), numerical, rtol=3e-5, atol=2e-6)
            checked_roots.append(True)
        return result

    monkeypatch.setattr(coupled, "least_squares", checked)
    state = model.step(case["axial_strain"], case["transverse_stress"], previous,
                       fluid_content=case["fluid_content"])
    f = state["response"]
    assert state["d"] > previous["d"]+.01
    assert state["matrix_multiplier"] > 0 and state["plane_multiplier"] > 0
    assert state["matrix_mean_force"] > 0
    assert state["matrix_flow_fraction"] == pytest.approx(1., abs=1e-9)
    assert state["matrix_dilatancy"] < 0
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-7)
    assert state["plane_yield"] == pytest.approx(0., abs=1e-7)
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    increment = state["matrix_strain"]-previous["matrix_strain"]
    assert VOLUME@increment == pytest.approx(-state["matrix_multiplier"]*state["matrix_dilatancy"], abs=1e-11)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    np.testing.assert_allclose(f["total_stress"][1:], case["transverse_stress"], atol=1e-9)
    assert state["fluid_content"] == case["fluid_content"]
    assert checked_roots


@pytest.mark.parametrize("case_index", [0, 1])
def test_history_predictor_recovers_activation_with_sticking_plane(case_index):
    record = json.loads((Path(__file__).parent / "data/winhausen_history_activation_predictor.json").read_text())
    case = record["cases"][case_index]
    model = wt.material(record["material_parameters"], record["path"],
                        record["storage_parameters"], record["evolution_parameters"],
                        record["evolution_inputs"], record["resistance_uses_pressure"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    p0 = record["path"]["confinement"]
    state = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=previous["fluid_content"])
    f = state["response"]
    assert state["solver_route"] == "released_plane_predictor"
    assert state["d"] > previous["d"]
    assert state["matrix_multiplier"] > 0 and state["matrix_mean_force"] > 0
    assert state["matrix_flow_fraction"] == pytest.approx(1., abs=1e-9)
    assert state["matrix_dilatancy"] < 0
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-7)
    assert state["plane_yield"] < 0
    np.testing.assert_allclose(state["plane_modes"], previous["plane_modes"], atol=1e-11)
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    _, flow, _, _ = model.matrix_surface(f["matrix_force"], state["d"],
        effective_mean=state["effective_mean_stress"], flow_fraction=state["matrix_flow_fraction"],
        fabric_projection=state["fabric_projection"])
    np.testing.assert_allclose(state["matrix_strain"]-previous["matrix_strain"],
                               state["matrix_multiplier"]*flow, atol=1e-11)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    np.testing.assert_allclose(f["total_stress"][1:], [p0, p0, 0., 0., 0.], atol=1e-9)
    assert state["fluid_content"] == previous["fluid_content"]


@pytest.mark.parametrize("drained", [False, True])
@pytest.mark.parametrize("fabric", [False, True])
@pytest.mark.parametrize("resistance_uses_pressure", [False, True])
def test_neural_evolution_loading_and_unloading_preserve_physical_state(drained, fabric, resistance_uses_pressure):
    model = model_with_network(fabric, resistance_uses_pressure)
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


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_near_fold_minimum_does_not_replace_a_constitutive_root(case_index):
    record = json.loads((Path(__file__).parent / "data/winhausen_near_fold_root.json").read_text())
    case = record["cases"][case_index]
    parameters = record["material_parameters"].copy()
    parameters["matrix_contraction"] = case["matrix_contraction"]
    model = wt.material(parameters, record["path"], record["storage_parameters"],
                        record["evolution_parameters"], record["evolution_inputs"],
                        record["resistance_uses_pressure"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in case["previous"].items()}
    p0 = record["path"]["confinement"]
    state = model.step(case["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=previous["fluid_content"])
    assert state["d"] == pytest.approx(case["expected_density"], abs=1e-7)
    assert state["response"]["total_stress"][0]-p0 == pytest.approx(case["expected_q_MPa"], abs=1e-7)
    assert state["matrix_yield"] <= 1e-8
    assert state["matrix_multiplier"] >= -1e-12
    assert state["d"] >= previous["d"]
    assert state["response"]["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-11)
    _, flow, _, _ = model.matrix_surface(state["response"]["matrix_force"], state["d"],
        effective_mean=state["effective_mean_stress"], flow_fraction=state["matrix_flow_fraction"],
        fabric_projection=state["fabric_projection"])
    np.testing.assert_allclose(state["matrix_strain"]-previous["matrix_strain"],
                               state["matrix_multiplier"]*flow, atol=1e-11)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    np.testing.assert_allclose(state["response"]["total_stress"][1:], [p0, p0, 0., 0., 0.], atol=1e-9)
    assert state["fluid_content"] == previous["fluid_content"]


@pytest.mark.parametrize("pressure_weight_increment", [0., -1e-7, 1e-7])
def test_fixed_density_branch_switch_releases_to_coupled_root(monkeypatch, pressure_weight_increment):
    record = json.loads((Path(__file__).parent / "data/winhausen_pressure_density_release.json").read_text())
    evolution = np.array(record["evolution_parameters"])
    evolution[1] += pressure_weight_increment
    model = wt.material(record["material_parameters"], record["path"],
                        record["storage_parameters"], evolution,
                        record["evolution_inputs"], record["resistance_uses_pressure"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in record["previous"].items()}
    original = coupled._resolve_candidate
    rejected_endpoint = []

    def inspect(solution, fun, jac):
        resolved = original(solution, fun, jac)
        residual = fun(solution.x)
        if not resolved and abs(residual[10]) > .01 and max(abs(residual[:10])) < 1e-7:
            rejected_endpoint.append(solution.x.copy())
        if resolved:
            h = 1e-7
            numerical = np.column_stack([(fun(solution.x+h*e)-fun(solution.x-h*e))/(2*h)
                                         for e in np.eye(13)])
            np.testing.assert_allclose(jac(solution.x), numerical, rtol=3e-5, atol=2e-6)
        return resolved

    monkeypatch.setattr(coupled, "_resolve_candidate", inspect)
    state = model.step(record["axial_strain"], record["transverse_stress"], previous,
                       fluid_content=record["fluid_content"])
    assert rejected_endpoint
    assert state["solver_route"] == "released_fixed_density"
    assert state["d"] > previous["d"]
    assert state["matrix_multiplier"] > 0 and state["matrix_mean_force"] > 0
    assert state["matrix_flow_fraction"] == pytest.approx(1., abs=1e-9)
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-8)
    assert state["plane_yield"] < 0
    f = state["response"]
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    _, flow, _, _ = model.matrix_surface(f["matrix_force"], state["d"],
        effective_mean=state["effective_mean_stress"], flow_fraction=state["matrix_flow_fraction"],
        fabric_projection=state["fabric_projection"])
    np.testing.assert_allclose(state["matrix_strain"]-previous["matrix_strain"],
                               state["matrix_multiplier"]*flow, atol=1e-11)
    np.testing.assert_allclose(state["plane_modes"], previous["plane_modes"], atol=1e-11)
    np.testing.assert_allclose(f["total_stress"][1:], record["transverse_stress"], atol=1e-9)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    assert state["fluid_content"] == record["fluid_content"]


@pytest.mark.parametrize("axial_increment", [0., 1e-8])
@pytest.mark.parametrize("filename", ["winhausen_pressure_normal_activation.json",
                                      "winhausen_pressure_inclined_activation.json"])
def test_pressure_conditioned_activation_resolves_finite_jump(axial_increment, filename):
    record = json.loads((Path(__file__).parent / "data" / filename).read_text())
    model = wt.material(record["material_parameters"], record["path"],
                        record["storage_parameters"], record["evolution_parameters"],
                        record["evolution_inputs"], record["resistance_uses_pressure"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in record["previous"].items()}
    state = model.step(record["axial_strain"]+axial_increment, record["transverse_stress"],
                       previous, fluid_content=record["fluid_content"])
    assert state["d"] > previous["d"]
    assert state["matrix_multiplier"] > 0 and state["matrix_mean_force"] > 0
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-8)
    assert state["plane_yield"] < 0
    f = state["response"]
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    _, flow, _, _ = model.matrix_surface(f["matrix_force"], state["d"],
        effective_mean=state["effective_mean_stress"], flow_fraction=state["matrix_flow_fraction"],
        fabric_projection=state["fabric_projection"])
    np.testing.assert_allclose(state["matrix_strain"]-previous["matrix_strain"],
                               state["matrix_multiplier"]*flow, atol=1e-11)
    np.testing.assert_allclose(state["plane_modes"], previous["plane_modes"], atol=1e-11)
    np.testing.assert_allclose(f["total_stress"][1:], record["transverse_stress"], atol=1e-9)
    assert min(state["dissipation_increments"].values()) >= -1e-10
    assert state["fluid_content"] == record["fluid_content"]
    if axial_increment == 0. and filename == "winhausen_pressure_normal_activation.json":
        assert state["d"] == pytest.approx(.219248314417492, abs=1e-8)
        assert f["total_stress"][0]-10. == pytest.approx(5.0503554205272785, abs=1e-7)


@pytest.mark.parametrize("axial_increment", [0., 1e-8])
def test_state_map_activation_resolves_coupled_matrix_and_plane_slip(axial_increment):
    record = json.loads((Path(__file__).parent / "data/winhausen_state_map_inclined_activation.json").read_text())
    model = wt.material(record["material_parameters"], record["path"],
                        record["storage_parameters"], record["evolution_parameters"],
                        record["evolution_inputs"], record["resistance_uses_pressure"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in record["previous"].items()}
    state = model.step(record["axial_strain"]+axial_increment, record["transverse_stress"],
                       previous, fluid_content=record["fluid_content"])
    f = state["response"]
    assert state["d"] > previous["d"]
    assert state["matrix_multiplier"] > 0 and state["plane_multiplier"] > 0
    assert state["matrix_mean_force"] > 0
    assert state["matrix_yield"] == pytest.approx(0., abs=1e-8)
    assert state["plane_yield"] == pytest.approx(0., abs=1e-8)
    assert f["matrix_release"] == pytest.approx(state["damage_resistance"], abs=1e-10)
    _, flow, _, _ = model.matrix_surface(f["matrix_force"], state["d"],
        effective_mean=state["effective_mean_stress"], flow_fraction=state["matrix_flow_fraction"],
        fabric_projection=state["fabric_projection"])
    np.testing.assert_allclose(state["matrix_strain"]-previous["matrix_strain"],
                               state["matrix_multiplier"]*flow, atol=1e-11)
    tangent = f["plane_force"][1:]
    np.testing.assert_allclose(state["plane_modes"][1:]-previous["plane_modes"][1:],
                               state["plane_multiplier"]*tangent/np.linalg.norm(tangent), atol=1e-11)
    np.testing.assert_allclose(f["total_stress"][1:], record["transverse_stress"], atol=1e-9)
    fluid = model.energy.fluid_content(state["strain"], state["matrix_strain"],
                                       state["plane_modes"], state["compaction"], f["pressure"])
    assert fluid == pytest.approx(record["fluid_content"], abs=1e-12)
    assert min(state["dissipation_increments"].values()) >= -1e-10


def test_inactive_matrix_corner_retains_undetermined_volume_fraction():
    record = json.loads((Path(__file__).parent / "data/winhausen_inactive_corner.json").read_text())
    model = wt.material(record["material_parameters"], record["path"],
                        record["storage_parameters"], record["evolution_parameters"])
    previous = {k: np.array(v) if isinstance(v, list) else v for k, v in record["previous"].items()}
    p0 = record["path"]["confinement"]
    state = model.step(record["axial_strain"], [p0, p0, 0., 0., 0.], previous,
                       fluid_content=previous["fluid_content"])
    assert abs(state["matrix_multiplier"]) < 1e-12
    assert abs(state["matrix_mean_force"]) < 1e-9
    np.testing.assert_allclose(state["matrix_strain"], previous["matrix_strain"], atol=1e-11)
    assert state["d"] == pytest.approx(previous["d"], abs=1e-10)
    assert 0 <= state["matrix_flow_fraction"] <= 1
    assert state["response"]["total_stress"][0]-p0 == pytest.approx(record["expected_q_MPa"], abs=1e-8)
    assert state["response"]["pressure"] == pytest.approx(record["expected_pressure_MPa"], abs=1e-8)
    assert min(state["dissipation_increments"].values()) >= -1e-10
