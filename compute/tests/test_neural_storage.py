"""Conjugate forces, observer changes and path constraints for neural storage."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.neural_storage import IdentityStorage, MonotoneStorageNetwork, NeuralPorousEnergy
from tfenn_invcalib.poromechanical_coupling import TIElasticity, mandel_rotation
from tfenn_invcalib.coupled_friction_damage import CoupledFrictionDamage, FrictionDamageLaw


def energy(normal=(.5, 0., np.sqrt(.75))):
    net = MonotoneStorageNetwork.initialize(.1, .7)
    return NeuralPorousEnergy(
        TIElasticity(9000., 4500., .2, .25, 3000.), normal,
        network=net, grain_bulk=50000., fluid_bulk=2200., porosity=.15,
        matrix_slope=2., plane_normal_factor=5., plane_shear_factor=3.,
        compaction_onset=16., compaction_hardening=1500.)


def test_network_anchors_and_positive_storage_for_learned_weights():
    rng = np.random.default_rng(14)
    for _ in range(6):
        m = energy()
        m.network = MonotoneStorageNetwork(rng.normal(size=12), .1, .7)
        value, derivative, _ = m.network.value_derivatives(.1)
        assert value == pytest.approx(.1)
        assert derivative == pytest.approx(1.)
        for d in (.1, .7, 4., 40.):
            h, hp, _ = m.network.value_derivatives(d)
            H, Hprime, _ = m.plane_storage_derivatives(d)
            assert h >= .1 and hp > 0
            assert np.linalg.eigvalsh(H).min() > 0
            assert np.linalg.eigvalsh(-Hprime).min() > 0


def test_storage_curvature_used_by_implicit_jacobian():
    m = energy()
    m.network = MonotoneStorageNetwork(np.linspace(-1.2, 1.3, 12), .1, .7)
    for d in (.11, .8, 3.):
        step = 1e-5
        H, Hp, Hpp = m.plane_storage_derivatives(d)
        left = m.plane_storage_derivatives(d-step)
        right = m.plane_storage_derivatives(d+step)
        np.testing.assert_allclose(Hp, (right[0]-left[0])/(2*step), rtol=2e-7, atol=1e-7)
        np.testing.assert_allclose(Hpp, (right[1]-left[1])/(2*step), rtol=2e-7, atol=1e-7)


def test_identity_storage_recovers_projected_host_closure_and_both_derivatives():
    m = energy()
    m.network = IdentityStorage()
    densities = np.array([.1, .7, 4., 40.])
    h, hp, hpp = m.network.value_derivatives(densities)
    np.testing.assert_array_equal(h, densities)
    np.testing.assert_array_equal(hp, np.ones_like(densities))
    np.testing.assert_array_equal(hpp, np.zeros_like(densities))
    for d in densities:
        H = np.linalg.inv(m.plane_S0+d*m.plane_Sprime)
        Hp = -H @ m.plane_Sprime @ H
        Hpp = 2*H @ m.plane_Sprime @ H @ m.plane_Sprime @ H
        for actual, expected in zip(m.plane_storage_derivatives(d), (H, Hp, Hpp)):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("identity", [False, True])
def test_all_forces_and_reciprocity_derive_from_one_energy(identity):
    m, D = energy(), .2
    if identity:
        m.network = IdentityStorage()
    eps = np.array([.003, .001, -.0005, .0003, -.0004, .0002])
    cm = np.array([.0005, -.0002, -.0001, .0001, .0002, -.0001])
    z, kappa, d, zeta = np.array([-.0001, .0007, -.0003]), .0002, .6, -.0001
    point = np.r_[eps, cm, z, kappa, d, zeta]

    def response(x):
        return m.response(x[:6], x[6:12], x[12:15], x[15], x[16], D, x[17])

    f = response(point)
    expected = np.r_[f["total_stress"], -f["matrix_force"], -f["plane_force"],
                     -f["compaction_force"], -f["matrix_release"], f["pressure"]]
    numerical = []
    for i in range(18):
        step = 1e-7 if i == 16 else 1e-8
        delta = np.eye(18)[i]*step
        numerical.append((response(point+delta)["energy"]-response(point-delta)["energy"])/(2*step))
    np.testing.assert_allclose(numerical, expected, rtol=1e-6, atol=1e-7)
    delta = np.eye(18)[16]*1e-6
    dp = (response(point+delta)["plane_force"]-response(point-delta)["plane_force"])/2e-6
    dy = []
    for i in range(12, 15):
        delta = np.eye(18)[i]*1e-8
        dy.append((response(point+delta)["matrix_release"]-response(point-delta)["matrix_release"])/2e-8)
    np.testing.assert_allclose(dp, dy, rtol=1e-7, atol=1e-8)


def test_objectivity_rotates_the_material_axis_and_history_together():
    m = energy()
    axis = np.array([1., 2., -1.]); axis /= np.linalg.norm(axis)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    R = np.eye(3)+np.sin(.73)*skew+(1-np.cos(.73))*(skew@skew)
    Q = mandel_rotation(R)
    rotated = energy(R@np.array([.5, 0., np.sqrt(.75)]))
    eps = np.array([.002, -.001, .003, .0002, .0007, -.0003])
    cm = eps/5
    z = np.array([-.0001, .0004, .0002])
    zr = rotated.B.T@Q@m.B@z
    a = m.response(eps, cm, z, .0002, .6, .2, -.0001)
    b = rotated.response(Q@eps, Q@cm, zr, .0002, .6, .2, -.0001)
    for name in ("energy", "pressure", "matrix_release", "plane_to_matrix_release"):
        assert b[name] == pytest.approx(a[name], rel=1e-11, abs=1e-10)
    np.testing.assert_allclose(b["total_stress"], Q@a["total_stress"], atol=1e-10)
    np.testing.assert_allclose(b["matrix_force"], Q@a["matrix_force"], atol=1e-10)


@pytest.mark.parametrize("drained", [False, True])
def test_neural_material_loading_unloading_contact_and_fluid_balance(drained):
    m = energy()
    law = FrictionDamageLaw(5., .6, .1, .5, 1., .25, .015)
    model = CoupledFrictionDamage(m, law, .2)
    initial = previous = model.consolidated_state(5., .1)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    history = np.r_[np.linspace(0., .008, 25), np.linspace(.008, .004, 10)[1:],
                    np.linspace(.004, .010, 15)[1:]]
    states = []
    for axial in history:
        state = model.step(initial["strain"][0]+axial, [5., 5., 0., 0., 0.], previous, **boundary)
        f = state["response"]
        assert state["d"] >= previous["d"]-1e-8
        assert state["compaction"] >= previous["compaction"]-1e-10
        assert state["plane_modes"][0] <= 1e-9
        assert f["plane_force"][0] >= -1e-6
        assert abs(state["plane_modes"][0]*f["plane_force"][0]) < 1e-7
        assert min(state["dissipation_increments"].values()) >= -1e-7
        assert state["matrix_yield"] <= 1e-5 and state["plane_yield"] <= 1e-5
        if drained:
            assert abs(f["pressure"]) < 1e-9
        else:
            assert state["fluid_content"] == initial["fluid_content"]
            volume = VOLUME@(state["strain"]-initial["strain"])
            q = f["total_stress"][0]-5.
            assert volume == pytest.approx(q/(3*50000.)+.15*(1/2200.-1/50000.)*f["pressure"], abs=1e-10)
        states.append(state)
        previous = state
    assert max(s["d"] for s in states) > initial["d"]+.01
    assert max(np.linalg.norm(s["plane_modes"][1:]) for s in states) > 1e-5
