"""Run named membrane cases and compare only against supplied cited data.

The user-provided request names papers but does not include their numerical
purity, recovery, area, or SEC values. This script intentionally does not invent
those references. Supply a CSV with the documented columns to calculate errors.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from multistage_model import (
    CascadeConfig,
    MembraneCascade,
    ModuleConfig,
    StageConfig,
)

BENCHMARKS = (
    ("Polaris 1st generation", 2200.0, 50.0, "Merkel 2010"),
    ("Polaris 2nd generation", 2700.0, 50.0, "White 2018"),
    ("PolyActive", 1200.0, 40.0, "Brinkmann 2017"),
    ("NTNU FTM pilot", 1000.0, 100.0, "Sandru 2010"),
)
REFERENCE_FIELDS = {
    "benchmark",
    "literature_purity_pct",
    "literature_recovery_pct",
    "literature_area_m2",
    "literature_sec_kwh_t",
    "citation",
}


def load_references(path: Path | None) -> dict[str, dict[str, str]]:
    """Load cited values from CSV; return an empty mapping when none is supplied."""
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REFERENCE_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {', '.join(sorted(missing))}")
        return {row["benchmark"]: row for row in reader}


def relative_error(model_value: float, reference_value: str) -> str:
    """Format relative error for a valid positive reference."""
    try:
        reference = float(reference_value)
    except (TypeError, ValueError):
        return "N/A"
    if reference == 0:
        return "N/A"
    return f"{abs(model_value - reference) / abs(reference) * 100:.1f}%"


def main() -> None:
    """Run the four benchmark materials and print model/reference comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references",
        type=Path,
        help="CSV with literature values and citations; no references are fabricated.",
    )
    arguments = parser.parse_args()
    references = load_references(arguments.references)
    for label, co2_gpu, alpha, paper in BENCHMARKS:
        module = ModuleConfig(
            permeance_CO2_GPU=co2_gpu,
            permeance_N2_GPU=co2_gpu / alpha,
            thickness_um=0.5,
            flow_pattern="countercurrent",
            segments=40,
        )
        cascade = CascadeConfig(
            layout="SINGLE_STAGE",
            stages=(StageConfig(
                P_feed_bar=1.0,
                P_perm_bar=0.2,
                target_stage_cut=0.3,
            ),),
            feed_flow_kmol_h=1000.0,
            feed_CO2_molfrac=0.10,
        )
        result = MembraneCascade(module, cascade).solve()
        reference = references.get(label, {})
        print(f"\n{label} ({paper})")
        print("Metric                 Model       Literature  Relative error")
        values = (
            ("Purity (%)", result.purity * 100, "literature_purity_pct"),
            ("Recovery (%)", result.recovery * 100, "literature_recovery_pct"),
            ("Area (m2)", result.total_area_m2, "literature_area_m2"),
            ("SEC (kWh/t)", result.specific_energy_kWh_per_t, "literature_sec_kwh_t"),
        )
        for metric, model_value, key in values:
            reference_value = reference.get(key, "")
            reported = f"{float(reference_value):.3f}" if reference_value else "N/A"
            error = relative_error(model_value, reference_value)
            print(f"{metric:<22} {model_value:>10.3f} {reported:>12} {error:>15}")
            try:
                if float(error.removesuffix("%")) > 20:
                    print(f"  Diagnostic: {metric} differs by over 20%; check basis, flow pattern, "
                          "operating pressures, module geometry, and reference conditions.")
            except ValueError:
                pass
        if not reference:
            print("  Literature values unavailable; no comparison is claimed.")
        elif reference.get("citation"):
            print(f"  Citation supplied: {reference['citation']}")


if __name__ == "__main__":
    main()
