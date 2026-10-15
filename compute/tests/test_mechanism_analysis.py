"""Mechanism activity uses increments; axial partitions use signed projections."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("mechanism_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_winhausen_mechanisms.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_volume_activity_is_independent_of_cumulative_sign_and_dilatancy_alone():
    group = dict(matrix_multiplier=1e-4, plane_multiplier=0.,
                 delta_matrix_volume=2e-6, matrix_volume=-3e-5, matrix_dilatancy=-.02)
    assert analysis.activity(group)["matrix_volume"] == "contraction"
    group.update(matrix_multiplier=0., delta_matrix_volume=1e-15)
    assert analysis.activity(group) == dict(matrix_flow="inactive", plane_slip="inactive",
                                           matrix_volume="neutral")
    group.update(matrix_multiplier=1e-4, matrix_dilatancy=.1, matrix_flow_fraction=0.)
    assert analysis.activity(group)["matrix_volume"] == "neutral"


def test_repeated_strain_preserves_increments_and_does_not_split_activity():
    rows = []
    for i, (strain, increment, multiplier, volume) in enumerate([
            (.001, .001, .0001, 1e-5), (.001, 0., 0., 1e-13),
            (.002, .001, .0001, 1e-5), (.003, .001, .0001, -1e-5)]):
        rows.append(dict(step=i, axial_strain=strain, axial_increment=increment,
            matrix_multiplier=multiplier, plane_multiplier=0., delta_matrix_volume=volume,
            delta_matrix_axial=multiplier, delta_plane_axial=0., delta_density=0., delta_q_MPa=0.))
    intervals = analysis.activity_intervals(rows)
    matrix = [row for row in intervals if row["mechanism"] == "matrix_flow"]
    assert len(matrix) == 1
    assert matrix[0]["start_strain"] == 0
    assert matrix[0]["end_strain"] == .003
    volume = [row for row in intervals if row["mechanism"] == "matrix_volume"]
    assert [row["activity"] for row in volume] == ["contraction", "dilation"]
    np.testing.assert_allclose(volume[0]["delta_matrix_volume"], 2e-5+1e-13, rtol=1e-14)
    assert volume[1]["start_strain"] == .002


def test_axial_partition_subtracts_preload_and_preserves_negative_projection():
    metadata = dict(angle_deg=0., initial_state=dict(strain=[.002, 0, 0, 0, 0, 0],
        matrix_strain=[.0001, 0, 0, 0, 0, 0], plane_modes=[0, 0, 0], compaction=.0003))
    row = dict(axial_strain=.01, strain_0=.012, matrix_axial=.0031,
               plane_axial=-.0002, compaction=.0006, matrix_volume=.0005)
    partition = analysis.partition(row, metadata)
    np.testing.assert_allclose([partition[key] for key in
        ("total_axial", "matrix_axial", "plane_axial", "cap_axial", "elastic_axial", "matrix_volume")],
        [.01, .003, -.0002, .0001, .0071, .0004])
