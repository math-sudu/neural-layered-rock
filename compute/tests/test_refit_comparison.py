"""Observed intervals and conditional volume retain their reference states."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_winhausen_refits import volume_coordinate, volume_history, observed_intervals, compare_stages


def test_volume_coordinate_keeps_prestress_and_separates_coefficient_changes():
    q = np.array([.4, 2., 5., 4.])
    volume = np.array([0., .001, -.0002, -.0005])
    pressure = 3.+.4*(q-q[0])+6000.*volume
    np.testing.assert_allclose(volume_coordinate(q, pressure, .4, 6000.), volume, atol=1e-18)
    changed = volume_coordinate(q, pressure, .5, 8000.)
    np.testing.assert_allclose(changed, (.4-.5)*(q-q[0])/8000.+6000.*volume/8000.)


def test_repeated_observed_pressure_maxima_preserve_both_windows_and_overlap():
    specimen = dict(specimen_id="example", history=dict(axial_strain=np.arange(5.)*.001,
        q_MPa=np.array([0., 1., 2., 3., 4.]), pressure_MPa=np.array([0., 3., 2., 3., 1.]),
        source_row=np.arange(5)+10))
    intervals = observed_intervals(specimen)
    assert [row["start_source_row"] for row in intervals] == [11, 13]
    assert [row["observed_delta_q_MPa"] for row in intervals] == [3., 1.]
    assert all(row["status"] == "pressure_before_strength" for row in intervals)
    specimen["history"]["q_MPa"] = np.array([0., 1., 4., 3., 2.])
    assert all(row["status"] == "overlapping_maximum_ranges" for row in observed_intervals(specimen))


def test_volume_history_separates_common_coordinates_and_internal_components():
    x = np.arange(4.)*.001
    q = np.array([.4, 2., 5., 4.])
    matrix = np.array([0., -.001, -.0005, -.0002])
    plane = np.array([0., -.0001, -.0002, -.0002])
    cap = np.array([0., .0001, .0003, .0004])
    pressure = 3.+.4*(q-q[0])+6000.*(matrix+plane+cap)
    specimen = dict(specimen_id="example", history=dict(axial_strain=x, q_MPa=q,
        pressure_MPa=pressure, source_row=np.arange(4)+10))
    rows = [dict(axial_strain=a, q_MPa=b, pressure_MPa=c, matrix_volume=m+.003,
                 plane_mode_0=p-.002, compaction=k+.001)
            for a,b,c,m,p,k in zip(x,q,pressure,matrix,plane,cap)]
    metadata = dict(pressure_stress_coefficient=.4, pressure_modulus_MPa=6000.,
        initial_state=dict(matrix_strain=[.001,.001,.001,0.,0.,0.],
                           plane_modes=[-.002,0.,0.], compaction=.001))
    reference = dict(pressure_stress_coefficient=.5, pressure_modulus_MPa=8000.)
    output = volume_history(specimen, rows, metadata, reference)
    np.testing.assert_allclose([r["predicted_volume_own_coefficients"] for r in output], matrix+plane+cap, atol=1e-18)
    np.testing.assert_allclose([r["predicted_total_inelastic_volume"] for r in output], matrix+plane+cap, atol=1e-18)
    np.testing.assert_allclose([r["observed_volume_common_coefficients"] for r in output],
                               (pressure-pressure[0]-.5*(q-q[0]))/8000.)
    assert [r["source_row"] for r in output] == [10,11,12,13]
    assert output[2]["predicted_matrix_volume"] < 0.
    assert output[2]["predicted_matrix_volume_increment"] > 0.


def test_stage_comparison_uses_common_observed_endpoints_and_joint_reference():
    specimen = dict(specimen_id="example", history=dict(axial_strain=np.arange(4.)*.001,
        q_MPa=np.array([.4, 2., 3., 4.]), pressure_MPa=np.array([0., 2., 1.5, 1.]),
        source_row=np.arange(4)))
    rows = [dict(axial_strain=x, q_MPa=5.-100.*x, pressure_MPa=2.-50.*x,
        matrix_volume=-.01*x, matrix_axial=.2*x, plane_axial=.3*x,
        matrix_mean_force_MPa=10.+x, plane_normal_force_MPa=9.+2*x)
        for x in [0., .0008, .0012, .0028, .0032]]
    metadata = dict(pressure_stress_coefficient=.4, pressure_modulus_MPa=6000.)
    reference = dict(pressure_stress_coefficient=.5, pressure_modulus_MPa=8000.)
    row = compare_stages(specimen, rows, metadata, reference)[0]
    assert row["observed_delta_q_MPa"] == 2.
    np.testing.assert_allclose(row["predicted_delta_q_MPa"], -.2)
    np.testing.assert_allclose(row["observed_volume_change_own_coefficients"], -1.8/6000.)
    np.testing.assert_allclose(row["observed_volume_change_common_coefficients"], -2./8000.)
    np.testing.assert_allclose([row["start_bracket_width"], row["end_bracket_width"]], .0004)
