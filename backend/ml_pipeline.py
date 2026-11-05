from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "NumHeavyAtoms": Descriptors.HeavyAtomCount,
    "NumRings": Descriptors.RingCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "MolLogP": Descriptors.MolLogP,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "MolMR": Descriptors.MolMR,
    "LabuteASA": Descriptors.LabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
    "NumSpiroAtoms": Descriptors.NumSpiroAtoms,
}
BASE_FEATURE_NAMES = tuple(DESCRIPTOR_FUNCTIONS) + tuple(f"FP_{index}" for index in range(128))
MODEL_OPTIONS = {
    "lr": {"label": "Linear Regression", "class_name": "LinearRegression"},
    "rf": {"label": "Random Forest", "class_name": "RandomForestRegressor"},
    "xg": {"label": "XGBoost", "class_name": "XGBRegressor"},
}
TARGET_NAMES = ("permeability", "selectivity")


def build_feature_row(smiles: str) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Create the notebook's ordered 20 RDKit descriptors and 128 Morgan bits."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Provide a non-empty SMILES string")

    molecule = Chem.MolFromSmiles(smiles.strip().replace("*", "C"))
    if molecule is None:
        raise ValueError("RDKit could not parse this SMILES string")

    descriptors: dict[str, float | int] = {
        name: function(molecule) for name, function in DESCRIPTOR_FUNCTIONS.items()
    }
    fingerprint = MORGAN_GENERATOR.GetFingerprint(molecule)
    bits = np.zeros((128,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    features = {**descriptors, **{f"FP_{index}": int(value) for index, value in enumerate(bits)}}
    return features, descriptors


def build_feature_frame(
    smiles: str, feature_names: list[str] | tuple[str, ...] | None = None
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build a named, ordered feature frame for training or model inference."""
    features, descriptors = build_feature_row(smiles)
    columns = list(feature_names) if feature_names is not None else list(BASE_FEATURE_NAMES)
    missing = [name for name in columns if name not in features]
    if missing:
        raise ValueError(f"Unknown model feature(s): {', '.join(missing)}")
    return pd.DataFrame([{name: features[name] for name in columns}], columns=columns), descriptors


def validate_model_bundle(bundle: Any) -> dict[str, Any]:
    """Fail closed on stale or incompatible model artifacts."""
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must be a model bundle dictionary")
    if bundle.get("schema_version") != 2:
        raise ValueError("Unsupported model artifact schema version")
    metadata = bundle.get("metadata")
    models = bundle.get("models")
    if not isinstance(metadata, dict) or not isinstance(models, dict):
        raise ValueError("Model artifact is missing metadata or fitted models")
    if metadata.get("targets") != {
        "permeability": "log10_CO2_permeability_barrer",
        "selectivity": "log10_CO2_N2_selectivity",
    }:
        raise ValueError("Model artifact target schema does not match the API")
    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Model artifact has no ordered feature schema")
    if not all(isinstance(name, str) for name in feature_names):
        raise ValueError("Model artifact feature names must be strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Model artifact feature schema contains duplicate names")
    ordered_names = [name for name in BASE_FEATURE_NAMES if name in set(feature_names)]
    if ordered_names != feature_names:
        raise ValueError("Model artifact features do not match the canonical notebook order")
    if metadata.get("feature_count") != len(feature_names):
        raise ValueError("Model artifact feature count does not match its schema")

    if set(models) != set(MODEL_OPTIONS):
        raise ValueError("Model artifact does not contain all supported model choices")
    model_options = metadata.get("model_options")
    if not isinstance(model_options, dict) or set(model_options) != set(MODEL_OPTIONS):
        raise ValueError("Model artifact is missing model option metadata")
    for model_id, option in MODEL_OPTIONS.items():
        model_metadata = model_options[model_id]
        if not isinstance(model_metadata, dict) or model_metadata.get("label") != option["label"]:
            raise ValueError(f"Model artifact has an invalid label for {model_id}")
        model_pair = models[model_id]
        if not isinstance(model_pair, dict) or set(model_pair) != set(TARGET_NAMES):
            raise ValueError(f"Model artifact is missing {model_id} target models")
        for target in TARGET_NAMES:
            model = model_pair[target]
            if model.__class__.__name__ != option["class_name"]:
                raise ValueError(f"Model artifact has an invalid {model_id} {target} model")
            feature_count = getattr(model, "n_features_in_", None)
            if feature_count != len(feature_names):
                raise ValueError(f"{model_id} {target} feature count does not match the saved schema")
            model_feature_names = getattr(model, "feature_names_in_", None)
            if model_feature_names is not None and list(model_feature_names) != feature_names:
                raise ValueError(f"{model_id} {target} feature order does not match the saved schema")
    return bundle
