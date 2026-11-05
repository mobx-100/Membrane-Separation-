import unittest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from backend.ml_pipeline import (
    BASE_FEATURE_NAMES,
    MODEL_OPTIONS,
    build_feature_frame,
    build_feature_row,
    validate_model_bundle,
)


class FeaturePipelineTests(unittest.TestCase):
    def test_feature_vector_matches_notebook_schema(self):
        features, descriptors = build_feature_row(
            "*/C(=C(/*)c1ccc([Si](C)(C)C)cc1)c1ccccc1"
        )

        self.assertEqual(len(features), 148)
        self.assertEqual(tuple(features), BASE_FEATURE_NAMES)
        self.assertEqual(descriptors["NumHeavyAtoms"], 20)
        self.assertEqual(descriptors["NumAromaticRings"], 2)

    def test_inference_frame_preserves_saved_model_feature_order(self):
        selected_features = ["FP_17", "MolWt", "NumAromaticRings"]

        frame, _ = build_feature_frame("*CC*", selected_features)

        self.assertEqual(list(frame.columns), selected_features)
        self.assertEqual(frame.shape, (1, 3))

    def test_invalid_smiles_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not parse"):
            build_feature_row("C(")

    def test_bundle_contains_all_three_model_choices_for_both_targets(self):
        features = pd.DataFrame({
            "MolWt": [12.0, 24.0, 36.0, 48.0],
            "NumRings": [0, 1, 0, 1],
        })
        targets = [1.0, 1.5, 2.0, 2.5]
        models = {
            "lr": {
                target: LinearRegression().fit(features, targets)
                for target in ("permeability", "selectivity")
            },
            "rf": {
                target: RandomForestRegressor(n_estimators=2, random_state=42).fit(features, targets)
                for target in ("permeability", "selectivity")
            },
            "xg": {
                target: XGBRegressor(
                    n_estimators=2,
                    max_depth=2,
                    n_jobs=1,
                    objective="reg:squarederror",
                ).fit(features, targets)
                for target in ("permeability", "selectivity")
            },
        }
        bundle = {
            "schema_version": 2,
            "metadata": {
                "targets": {
                    "permeability": "log10_CO2_permeability_barrer",
                    "selectivity": "log10_CO2_N2_selectivity",
                },
                "feature_names": list(features.columns),
                "feature_count": len(features.columns),
                "model_options": {
                    model_id: {"label": option["label"]}
                    for model_id, option in MODEL_OPTIONS.items()
                },
            },
            "models": models,
        }

        self.assertIs(validate_model_bundle(bundle), bundle)

    def test_bundle_rejects_missing_model_choice(self):
        metadata = {
            "targets": {
                "permeability": "log10_CO2_permeability_barrer",
                "selectivity": "log10_CO2_N2_selectivity",
            },
            "feature_names": ["MolWt"],
            "feature_count": 1,
            "model_options": {
                model_id: {"label": option["label"]}
                for model_id, option in MODEL_OPTIONS.items()
            },
        }
        with self.assertRaisesRegex(ValueError, "all supported model choices"):
            validate_model_bundle({
                "schema_version": 2,
                "models": {"lr": {}, "rf": {}},
                "metadata": metadata,
            })


if __name__ == "__main__":
    unittest.main()
