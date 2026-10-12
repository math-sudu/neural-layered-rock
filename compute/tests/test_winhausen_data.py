"""Regression checks for the actual public-data import ambiguities."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfenn_invcalib.winhausen_data import DATA_ROOT, load_specimens, read_raw


@pytest.fixture(scope="module")
def specimens():
    return {p["specimen_id"]: p for p in load_specimens(None)}


@pytest.mark.parametrize("specimen, peak_q, peak_du", [
    ("OPA07", 20.01, 2.67),  # UTF-8, pressure precedes effective mean stress.
    ("OPA11", 13.47, 3.41),  # UTF-16, the two pressure-related columns reverse.
])
def test_original_encodings_and_column_orders(specimens, specimen, peak_q, peak_du):
    path = specimens[specimen]
    h = path["history"]
    assert path["initial_pressure"] == pytest.approx(2.49)
    assert h["mean_effective_stress_MPa"][0] == pytest.approx(10.01)
    assert h["q_MPa"].max() == pytest.approx(peak_q)
    assert h["pressure_MPa"].max() == pytest.approx(peak_du)
    assert h["pressure_MPa"][0] == 0.


@pytest.mark.parametrize("specimen, initial_u, cell, angle", [
    ("OPA10", 3.49, 13.5, 60.),
    ("OPA18", 2.00, 12.0, 65.),
])
def test_raw_reference_resolves_table_conflict(specimens, specimen, initial_u, cell, angle):
    path = specimens[specimen]
    assert path["initial_pressure"] == pytest.approx(initial_u)
    assert path["total_confinement"] == pytest.approx(cell)
    assert path["confinement"] == pytest.approx(cell-initial_u)
    assert path["angle"] == angle
    assert path["nominal_confinement"] == 10


def test_loading_segments_keep_original_terminal_records(specimens):
    z15 = specimens["OPA24"]
    assert z15["raw_count"] == 20583
    assert len(z15["history"]["time_h"]) == 20547
    assert z15["history"]["axial_strain"][-1] == pytest.approx(.029)
    raw = read_raw((DATA_ROOT / "raw" / z15["source_file"]).read_bytes())
    assert raw["axial_strain"][-1] == pytest.approx(.0154)
    z30 = specimens["OPA10"]
    assert z30["raw_count"] == 21394
    assert len(z30["history"]["time_h"]) == 21393
    assert z30["history"]["pressure_MPa"][-1] == pytest.approx(3.07)
    assert z30["history"]["pressure_MPa"].max() == pytest.approx(5.19)


def test_strain_reversals_are_not_sorted_away(specimens):
    h = specimens["OPA15"]["history"]
    assert h["axial_strain"][1295] == pytest.approx(.0027)
    assert h["axial_strain"][1296] == pytest.approx(.0025)
    assert np.count_nonzero(np.diff(h["axial_strain"]) < 0) == 91
    assert h["axial_strain"][np.argmax(h["q_MPa"])] == pytest.approx(.0054)
    np.testing.assert_array_equal(h["source_row"], np.arange(8059))


def test_constancy_of_reconstructed_radial_stress_at_source_resolution(specimens):
    # Each reported stress is rounded to 0.01 MPa. The reconstruction uses
    # p' - q/3 + u; comparison to its sample median doubles this rounding span.
    rounding_span = 2 * .005 * (1 + 1/3 + 1)
    for path in specimens.values():
        h = path["history"]
        cell = h["mean_effective_stress_MPa"] - h["q_MPa"]/3 + h["pressure_absolute_MPa"]
        np.testing.assert_allclose(cell, path["total_confinement"], rtol=0, atol=rounding_span)
        assert "radial_strain_single" in h and "volume" not in h


def test_complete_orientation_groups_are_withheld(specimens):
    train = load_specimens("train")
    test = load_specimens("test")
    assert {p["specimen_id"] for p in train} == {
        "OPA06", "OPA07", "OPA09", "OPA10", "OPA11", "OPA12", "OPA22"}
    assert {p["specimen_id"] for p in test} == {
        "OPA15", "OPA17", "OPA18", "OPA23", "OPA24"}
    assert {p["angle"] for p in train}.isdisjoint({p["angle"] for p in test})
    assert {p["nominal_confinement"] for p in train} == {4, 5, 10}
    assert specimens["OPA07"]["angle"] == 0.
    assert specimens["OPA22"]["angle"] == 90.


def test_downloaded_original_sizes_match_source_metadata():
    source = json.loads((DATA_ROOT / "source.json").read_text(encoding="utf-8"))
    for item in source["files"]:
        assert (DATA_ROOT / "raw" / item["name"]).stat().st_size == item["bytes"]
