"""Public saturated triaxial histories from Winhausen et al. (2023).

Read the original OSF bytes by column name, retaining acquisition order.
The single measured radial direction is not a volumetric-strain channel.
Specimen roles and terminal-record exclusions are in the source metadata.
"""

import io
import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[2] / "results/data/winhausen2023"
COLUMNS = {
    "time[h]": "time_h",
    "axialstrain[%]": "axial_strain",
    "radialstrain[%]": "radial_strain_single",
    "diff.stress[MPa]": "q_MPa",
    "meanu[MPa]": "pressure_absolute_MPa",
    "eff.meanstress[MPa]": "mean_effective_stress_MPa",
}


def read_raw(raw):
    """Decode both source encodings and both observed pressure-column orders."""
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    text = raw.decode(encoding)
    header = [item.strip().strip('"').replace(" ", "")
              for item in text.splitlines()[0].split("\t") if item.strip()]
    if len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
        raise ValueError(f"Unrecognized Winhausen data columns: {header}")
    values = np.loadtxt(io.StringIO(text), skiprows=1, ndmin=2)
    if values.shape[1] != len(header) or not np.isfinite(values).all():
        raise ValueError("Expected six finite numeric columns in the source file.")
    columns = {COLUMNS[name]: values[:, i].copy() for i, name in enumerate(header)}
    if np.any(np.diff(columns["time_h"]) < 0):
        raise ValueError("Acquisition times must remain in their recorded order.")
    columns["axial_strain"] /= 100.
    columns["radial_strain_single"] /= 100.
    return columns


def load_specimen(metadata, root=DATA_ROOT):
    """Return the retained loading history without sorting or resampling.

    Pressure increments use the first recorded pore pressure. Total cell
    pressure is reconstructed as p' - q/3 + u; its median fixes the constant
    radial traction. The corresponding effective preparation pressure is
    this cell pressure minus the recorded initial pore pressure. Nominal
    preparation pressure and the original absolute observations are retained.
    """
    raw = read_raw((Path(root) / "raw" / metadata["name"]).read_bytes())
    count = len(raw["time_h"])
    stop = metadata.get("loading_stop", count)
    if not 0 < stop <= count:
        raise ValueError(f"Invalid loading segment for {metadata['name']}: {stop}")
    history = {name: values[:stop].copy() for name, values in raw.items()}
    history["source_row"] = np.arange(stop)
    u0 = float(history["pressure_absolute_MPa"][0])
    cell = (history["mean_effective_stress_MPa"] - history["q_MPa"] / 3.
            + history["pressure_absolute_MPa"])
    total_confinement = float(np.median(cell))
    history["pressure_MPa"] = history["pressure_absolute_MPa"] - u0
    history["axial_strain"] -= history["axial_strain"][0]
    history["radial_strain_single"] -= history["radial_strain_single"][0]
    return dict(
        key=f"winhausen2023_{metadata['specimen_id']}",
        specimen_id=metadata["specimen_id"], source_file=metadata["name"],
        role=metadata["role"], drainage="CU",
        angle=90. - metadata["source_angle_deg"],
        confinement=total_confinement-u0,
        nominal_confinement=metadata["effective_consolidation_MPa"],
        total_confinement=total_confinement, initial_pressure=u0,
        strain_rate_per_s=metadata["strain_rate_per_s"],
        raw_count=count, history=history,
    )


def load_specimens(role="train", root=DATA_ROOT):
    """Load complete specimens from the fixed split; None selects all records."""
    if role not in ("train", "test", None):
        raise ValueError("role must be 'train', 'test', or None.")
    root = Path(root)
    source = json.loads((root / "source.json").read_text(encoding="utf-8"))
    return [load_specimen(item, root) for item in source["files"]
            if role is None or item["role"] == role]
