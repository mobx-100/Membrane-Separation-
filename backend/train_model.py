from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .ml_pipeline import (
    BASE_FEATURE_NAMES,
    MODEL_OPTIONS,
    TARGET_NAMES,
    build_feature_row,
    validate_model_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CO2_DATA_URL = (
    "https://raw.githubusercontent.com/sptiwari/polymer_backbones_paper_data/main/"
    "Experimental_permeabilities_selectivities_and_SMILES_of_polymers/CO2_permeability_data.csv"
)
SELECTIVITY_DATA_URL = (
    "https://raw.githubusercontent.com/sptiwari/polymer_backbones_paper_data/main/"
    "Experimental_permeabilities_selectivities_and_SMILES_of_polymers/co2_n2_selectivity_data.csv"
)
TARGETS = {
    "permeability": "log10_CO2_permeability_barrer",
    "selectivity": "log10_CO2_N2_selectivity",
}
RANDOM_STATE = 42
MODEL_PARAMETERS = {
    "lr": {},
    "rf": {
        "n_estimators": 300,
        "max_depth": 15,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    "xg": {
            "learning_rate": 0.03,
            "max_depth": 8,
            "n_estimators": 500,
            "subsample": 0.7,
    },
}


def read_csv(url: str) -> tuple[pd.DataFrame, str]:
    with urlopen(url, timeout=30) as response:
        content = response.read()
    return pd.read_csv(io.BytesIO(content)), hashlib.sha256(content).hexdigest()


def load_training_data() -> tuple[pd.DataFrame, dict[str, str]]:
    co2, co2_hash = read_csv(CO2_DATA_URL)
    selectivity, selectivity_hash = read_csv(SELECTIVITY_DATA_URL)
    co2.columns = co2.columns.str.strip()
    selectivity.columns = selectivity.columns.str.strip()

    required_co2 = {"SMILES", "CO2 permeability (barrer)"}
    required_selectivity = {"SMILES", "CO2/N2 Selectivity"}
    if not required_co2.issubset(co2.columns) or not required_selectivity.issubset(selectivity.columns):
        raise ValueError("The downloaded CSV files do not match the notebook's expected columns")

    data = co2.rename(columns={"CO2 permeability (barrer)": "CO2_perm"})[
        ["SMILES", "CO2_perm"]
    ].merge(
        selectivity.rename(columns={"CO2/N2 Selectivity": "CO2_N2_sel"})[
            ["SMILES", "CO2_N2_sel"]
        ],
        on="SMILES",
        how="inner",
    )
    data = data.dropna(subset=["SMILES", "CO2_perm", "CO2_N2_sel"])
    data = data[
        np.isfinite(data["CO2_perm"])
        & np.isfinite(data["CO2_N2_sel"])
        & (data["CO2_perm"] > 0)
        & (data["CO2_N2_sel"] > 0)
    ]
    data = data.drop_duplicates(subset="SMILES", keep="first").reset_index(drop=True)
    if data.empty:
        raise ValueError("No paired positive permeability/selectivity records were found")
    return data, {"co2_permeability_sha256": co2_hash, "co2_n2_selectivity_sha256": selectivity_hash}


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    feature_rows: list[dict[str, float | int]] = []
    target_rows: list[tuple[int, float, float]] = []
    invalid_rows = 0
    for row in data.itertuples(index=False):
        try:
            features, _ = build_feature_row(row.SMILES)
        except ValueError:
            invalid_rows += 1
            continue
        feature_rows.append(features)
        target_rows.append((len(feature_rows) - 1, np.log10(row.CO2_perm), np.log10(row.CO2_N2_sel)))
    if not feature_rows:
        raise ValueError("RDKit could not extract features from any training records")

    features_frame = pd.DataFrame(feature_rows, columns=BASE_FEATURE_NAMES)
    targets_frame = pd.DataFrame(
        [(permeability, selectivity) for _, permeability, selectivity in target_rows],
        columns=TARGETS,
    )
    zero_variance = features_frame.columns[features_frame.var() == 0].tolist()
    features_frame = features_frame.drop(columns=zero_variance)
    correlations = features_frame.corr().abs()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    correlated = [column for column in upper.columns if any(upper[column] > 0.98)]
    features_frame = features_frame.drop(columns=correlated)
    return features_frame, targets_frame, {
        "invalid_smiles": invalid_rows,
        "zero_variance_features_removed": len(zero_variance),
        "correlated_features_removed": len(correlated),
    }


def score_model(model: object, features: pd.DataFrame, actual: pd.Series) -> dict[str, float]:
    prediction = model.predict(features)
    return {
        "r2": float(r2_score(actual, prediction)),
        "mae_log10": float(mean_absolute_error(actual, prediction)),
        "rmse_log10": float(np.sqrt(mean_squared_error(actual, prediction))),
    }


def build_regressor(model_id: str) -> object:
    if model_id == "lr":
        return LinearRegression(**MODEL_PARAMETERS[model_id])
    if model_id == "rf":
        return RandomForestRegressor(
            **MODEL_PARAMETERS[model_id],
            random_state=RANDOM_STATE,
            n_jobs=min(4, os.cpu_count() or 1),
        )
    if model_id == "xg":
        return XGBRegressor(
            **MODEL_PARAMETERS[model_id],
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=min(4, os.cpu_count() or 1),
            verbosity=0,
        )
    raise ValueError(f"Unsupported model choice: {model_id}")


def train(output_dir: Path) -> dict[str, object]:
    data, hashes = load_training_data()
    features, targets, cleaning = prepare_features(data)
    indices = np.arange(len(features))
    train_validation, test = train_test_split(
        indices, test_size=0.15, random_state=RANDOM_STATE
    )
    train_indices, validation_indices = train_test_split(
        train_validation, test_size=0.176, random_state=RANDOM_STATE
    )
    fit_indices = np.concatenate((train_indices, validation_indices))
    train_validation_features = features.iloc[fit_indices]
    test_features = features.iloc[test]

    models: dict[str, dict[str, object]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_id in MODEL_OPTIONS:
        models[model_id] = {}
        metrics[model_id] = {}
        for target in TARGET_NAMES:
            model = build_regressor(model_id)
            model.fit(train_validation_features, targets[target].iloc[fit_indices])
            models[model_id][target] = model
            metrics[model_id][target] = score_model(model, test_features, targets[target].iloc[test])

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": int(len(features)),
        "training_size": int(len(train_indices)),
        "validation_size": int(len(validation_indices)),
        "train_validation_size": int(len(train_validation)),
        "test_size": int(len(test)),
        "feature_names": list(features.columns),
        "feature_count": int(features.shape[1]),
        "targets": TARGETS,
        "model_options": {
            model_id: {
                **option,
                "parameters": MODEL_PARAMETERS[model_id],
            }
            for model_id, option in MODEL_OPTIONS.items()
        },
        "test_metrics": metrics,
        "cleaning": cleaning,
        "dataset_source": {
            "co2_permeability_csv": CO2_DATA_URL,
            "co2_n2_selectivity_csv": SELECTIVITY_DATA_URL,
            **hashes,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "random_state": RANDOM_STATE,
        "target_transform": "log10; API returns inverse-transformed values",
    }
    bundle = validate_model_bundle({
        "schema_version": 2,
        "metadata": metadata,
        "models": models,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "membrane_models.joblib"
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".joblib.tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    data.to_csv(output_dir / "clean_data.csv", index=False)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(metadata, summary_file, indent=2)
    return {"artifact": str(artifact_path), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the notebook's polymer membrane models.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory for the model bundle, cleaned CSV and training summary.",
    )
    arguments = parser.parse_args()
    result = train(arguments.output_dir)
    metadata = result["metadata"]
    print(f"Trained model bundle: {result['artifact']}")
    print(
        f"Dataset: {metadata['dataset_size']} polymers; "
        f"{metadata['feature_count']} ordered features"
    )
    for model_id, target_metrics in metadata["test_metrics"].items():
        for target, metrics in target_metrics.items():
            print(
                f"{model_id.upper()} {target}: test R²={metrics['r2']:.3f}, "
                f"MAE={metrics['mae_log10']:.3f} log10"
            )


if __name__ == "__main__":
    main()
