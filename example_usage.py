"""End-to-end example using a selected Stage 1 regressor and membrane cascade."""

from __future__ import annotations

import numpy as np

from backend.api import MODEL_BUNDLE_PATH, load_model_bundle
from backend.ml_pipeline import MODEL_OPTIONS, build_feature_frame
from multistage_model import CascadeConfig, MembraneCascade, ModuleConfig, StageConfig


def main() -> None:
    """Predict material properties, then evaluate a configured two-stage cascade."""
    bundle = load_model_bundle()
    if bundle is None:
        raise RuntimeError(
            f"No trained model bundle at {MODEL_BUNDLE_PATH}; run `npm run train:model` first."
        )
    smiles = "*CC*"
    selected_model = "xg"
    if selected_model not in MODEL_OPTIONS:
        raise ValueError("selected_model must be lr, rf, or xg")
    feature_names = bundle["metadata"]["feature_names"]
    features, descriptors = build_feature_frame(smiles, feature_names)
    model_pair = bundle["models"][selected_model]
    permeability = float(np.power(10, model_pair["permeability"].predict(features)[0]))
    selectivity = float(np.power(10, model_pair["selectivity"].predict(features)[0]))
    print(f"SMILES: {smiles}; selected model: {MODEL_OPTIONS[selected_model]['label']}")
    print(f"Predicted CO2 permeability: {permeability:.2f} Barrer")
    print(f"Predicted CO2/N2 selectivity: {selectivity:.2f}")
    print(f"RDKit descriptors calculated: {len(descriptors)}")

    module = ModuleConfig(
        thickness_um=0.5,
        P_CO2_Barrer=permeability,
        alpha_CO2_N2=selectivity,
        flow_pattern="countercurrent",
    )
    cascade = CascadeConfig(
        layout="TWO_STAGE_RECYCLE",
        stages=(
            StageConfig(P_feed_bar=6.0, P_perm_bar=0.1, target_stage_cut=0.3),
            StageConfig(P_feed_bar=6.0, P_perm_bar=0.05, target_stage_cut=0.2),
        ),
        feed_flow_kmol_h=1000.0,
        feed_CO2_molfrac=0.10,
    )
    result = MembraneCascade(module, cascade).solve()
    print(result.summary())
    tea = result.compute_tea(electricity_price=cascade.electricity_price)
    print(tea.summary())
    print("TEA sensitivity ($/t):", tea.sensitivity)


if __name__ == "__main__":
    main()
