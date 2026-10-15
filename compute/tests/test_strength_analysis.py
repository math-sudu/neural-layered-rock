"""Test the product split and the boundaries of continuous sliding analysis."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_winhausen_strength import (classify_step, normal_matrix_increment,
                                       product_increments, peak_event,
                                       turning_rates, interval_budget, window_budget,
                                       orientation_coefficients)
from tfenn_invcalib import winhausen_training as wt
from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.neural_storage import MonotoneStorageNetwork


def test_product_split_recovers_backstress_change_from_zero_slip():
    slip, damage = product_increments(100., 50., 0., .02)
    np.testing.assert_allclose([slip, damage, slip+damage], [1.5, -.5, 1.])


def test_monotone_slip_can_harden_or_soften_during_damage():
    for z_after, sign in ((.03, 1), (.015, -1)):
        slip, damage = product_increments(100., 50., .01, z_after)
        assert slip > 0 and damage < 0
        assert np.sign(slip+damage) == sign
        np.testing.assert_allclose(slip+damage, 50*z_after-100*.01)


def test_product_split_approximates_continuous_storage_integrals():
    # z=d, k=1/(1+d): integral k dz=log(2), integral z dk=1/2-log(2).
    d = np.linspace(0., 1., 401)
    k = 1/(1+d)
    slip, damage = product_increments(k[:-1], k[1:], d[:-1], d[1:])
    np.testing.assert_allclose([slip.sum(), damage.sum()],
                              [np.log(2), .5-np.log(2)], rtol=3e-6)


def test_jump_entry_and_symmetry_are_not_continuous_sliding():
    active = dict(plane_multiplier=.001, sliding_balance=True, open_contact=False)
    inactive = dict(plane_multiplier=0., sliding_balance=False, open_contact=False)
    assert classify_step(active, active, .2, False) == "continuous_sliding"
    assert classify_step(active, active, .2, True) == "finite_jump"
    assert classify_step(active, inactive, .2, False) == "sliding_entry"
    assert classify_step(inactive, active, .2, False) == "non_sliding"
    assert classify_step(active, active, 0., False) == "no_plane_strength"
    opened = dict(plane_multiplier=.001, sliding_balance=False, open_contact=True)
    assert classify_step(opened, active, .2, False) == "open_contact"
    assert classify_step(opened, active, .2, True) == "finite_jump"


def test_peak_events_distinguish_turn_plateau_repeated_readings_and_endpoint():
    strain = np.arange(5)*.001
    assert peak_event(strain, [0, 1, 3, 2, 1])["kind"] == "interior_turn"
    plateau = peak_event(strain, [0, 3, 3, 2, 1])
    assert plateau["kind"] == "flat_maximum"
    assert (plateau["first_index"], plateau["last_index"]) == (1, 2)
    repeated = peak_event(strain, [0, 3, 2.99, 3, 1])
    assert repeated["kind"] == "repeated_maximum"
    assert repeated["maximum_count"] == 2
    assert peak_event(strain, [0, 1, 2, 3, 3])["kind"] == "boundary_maximum"
    kinds = ["non_sliding", "continuous_sliding", "continuous_sliding",
             "finite_jump", "continuous_sliding"]
    assert peak_event(strain, [0, 1, 3, 2, 1], kinds)["kind"] == "jump_maximum"


def test_dilation_allows_pressure_decline_before_strength_turning():
    bq, ctheta, mu, v = .1, .3, .2, 2.
    # The pressure turns at H=C_theta*V/B_q; strength turns at H=-mu*V.
    qr, ur = turning_rates(ctheta*v/bq, v, bq, ctheta, mu)
    assert qr > 0
    np.testing.assert_allclose(ur, 0., atol=1e-14)
    qr, ur = turning_rates(-mu*v, v, bq, ctheta, mu)
    np.testing.assert_allclose(qr, 0., atol=1e-14)
    assert ur < 0
    for h in (1., -.2):
        qr, ur = turning_rates(h, v, bq, ctheta, mu)
        assert qr > 0 and ur < 0
        np.testing.assert_allclose(ur, bq*qr-v)


def test_event_budget_keeps_finite_jumps_outside_slip_damage_integrals():
    rows = [dict(step=0, axial_strain=0., q_MPa=0., pressure_MPa=0.)]
    for i, (kind, dq, du) in enumerate([
            ("continuous_sliding", 1., -.2), ("finite_jump", -2., .3),
            ("no_plane_strength", .5, -.1)], 1):
        rows.append(dict(step=i, axial_strain=.001*i, kind=kind,
            q_MPa=rows[-1]["q_MPa"]+dq, pressure_MPa=rows[-1]["pressure_MPa"]+du,
            delta_q_MPa=dq, delta_pressure_MPa=du,
            pressure_stress_increment_MPa=.1*dq,
            pressure_matrix_increment_MPa=du-.1*dq,
            pressure_other_volume_increment_MPa=0.,
            slip_contribution_MPa=2., damage_contribution_MPa=-1.2,
            volume_contribution_MPa=.2))
    budget = interval_budget(rows, 0, 3)
    np.testing.assert_allclose([budget["delta_q_MPa"], budget["continuous_plane_delta_q_MPa"],
                               budget["jump_delta_q_MPa"], budget["other_delta_q_MPa"]],
                              [-.5, 1., -2., .5])
    assert budget["jump_steps"] == 1
    assert budget["slip_contribution_MPa"] == 2.
    assert np.isnan(interval_budget(rows, 2, 3)["slip_contribution_MPa"])


def test_normal_matrix_split_recovers_actual_undrained_strength_during_friction_evolution():
    parameters = wt.initial_parameters()
    raw = MonotoneStorageNetwork.initialize(parameters["initial_density"],
                                            parameters["damage_peak_density"]).raw_parameters
    model = wt.material(parameters, dict(angle=90., confinement=10.), raw)
    m = model.energy
    modulus = 1/(1/m.M+m.biot @ m.S0 @ m.biot)
    coefficient = modulus*(m.biot @ m.S0)[0]
    initial = previous = model.consolidated_state(10., parameters["initial_density"])
    preceding = None
    contributions = []
    for axial in np.linspace(0., .018, 81):
        state = model.step(initial["strain"][0]+axial, [10., 10., 0., 0., 0.],
                           previous, fluid_content=initial["fluid_content"])
        row = dict(density=state["d"], matrix_volume=VOLUME @ state["matrix_strain"],
                   matrix_friction=model.law.friction(state["d"])[0],
                   matrix_mean_force_MPa=state["matrix_mean_force"],
                   **{f"matrix_strain_{i}": value
                      for i, value in enumerate(state["matrix_strain"])})
        if (preceding is not None and abs(previous["matrix_yield"]) < 1e-8
                and abs(state["matrix_yield"]) < 1e-8
                and min(previous["matrix_mean_force"], state["matrix_mean_force"]) > 0):
            terms = normal_matrix_increment(preceding, row, m.Hc, modulus, coefficient)
            dq = state["response"]["total_stress"][0]-previous["response"]["total_stress"][0]
            np.testing.assert_allclose(terms.sum(), dq, atol=2e-8)
            contributions.append(terms)
        preceding, previous = row, state
    terms = np.array(contributions)
    assert len(terms) > 10
    assert np.all(np.max(abs(terms), axis=0) > 1e-4)
    assert terms[:, 3].min() < -1e-4


def test_observed_budget_clips_continuous_edges_without_smearing_a_jump():
    rows = [dict(step=0, axial_strain=0., q_MPa=0., pressure_MPa=0.)]
    for i in (1, 2, 3):
        dq = float(i)
        rows.append(dict(step=i, axial_strain=.001*i, kind="continuous_sliding",
            q_MPa=rows[-1]["q_MPa"]+dq, pressure_MPa=rows[-1]["pressure_MPa"]-.2*dq,
            delta_q_MPa=dq, delta_pressure_MPa=-.2*dq,
            pressure_stress_increment_MPa=.1*dq, pressure_matrix_increment_MPa=-.3*dq,
            pressure_other_volume_increment_MPa=0., slip_contribution_MPa=2*dq,
            damage_contribution_MPa=-1.2*dq, volume_contribution_MPa=.2*dq))
    result = window_budget(rows, .0005, .0025)
    np.testing.assert_allclose([result["delta_q_MPa"], result["delta_pressure_MPa"],
        result["slip_contribution_MPa"], result["damage_contribution_MPa"],
        result["volume_contribution_MPa"]], [4., -.8, 8., -4.8, .8])
    reverse = window_budget(rows, .0025, .0005)
    np.testing.assert_allclose(reverse["delta_q_MPa"], -4.)
    rows[2]["kind"] = "finite_jump"
    assert window_budget(rows, .0005, .0025)["jump_delta_q_MPa"] == 2.
    with pytest.raises(ValueError, match="intersects a finite jump"):
        window_budget(rows, .0015, .0025)


def test_ti_pressure_sign_angles_preserve_preparation_and_rotation():
    parameters = wt.initial_parameters()
    previous = None
    for angle in (0., 17., 53., 90.):
        model = wt.material(parameters, dict(angle=angle, confinement=4.123), [], storage_form="identity")
        result = orientation_coefficients(model)
        parallel, normal = result["pressure_coefficient_parallel"], result["pressure_coefficient_normal"]
        m = model.energy
        bq = (m.biot @ m.S0)[0]/(1/m.M+m.biot @ m.S0 @ m.biot)
        np.testing.assert_allclose(bq, parallel+(normal-parallel)*np.sin(np.deg2rad(angle))**2, atol=1e-14)
        if previous is not None:
            np.testing.assert_allclose([parallel, normal], previous, atol=1e-14)
        previous = [parallel, normal]
        for name, target in (("effective_mean", 1/3), ("plane_reaction", None)):
            if result[f"{name}_stress_switch_status"] == "root":
                square = np.sin(np.deg2rad(result[f"{name}_stress_switch_angle_deg"]))**2
                np.testing.assert_allclose(parallel+(normal-parallel)*square,
                                           square if target is None else target, atol=1e-14)


def test_isotropic_pressure_coefficients_distinguish_no_root_and_all_angles():
    energy = SimpleNamespace(S0=np.eye(6), biot=VOLUME, M=.5,
                             B=np.array([[1., 0., 0., 0., 0., 0.]]).T)
    model = SimpleNamespace(energy=energy)
    result = orientation_coefficients(model)
    assert result["effective_mean_stress_switch_status"] == "no_root"
    assert np.isnan(result["effective_mean_stress_switch_angle_deg"])
    energy.M = np.inf
    result = orientation_coefficients(model)
    assert result["effective_mean_stress_switch_status"] == "all_orientations"
    assert np.isnan(result["effective_mean_stress_switch_angle_deg"])
