import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
from xgboost.core import XGBoostError
from .ml_pipeline import MODEL_OPTIONS, build_feature_frame, validate_model_bundle
from multistage_model import (
    CascadeConfig,
    MembraneCascade,
    ModuleConfig,
    PretreatmentUnit,
    StageConfig,
)
from rigorous_simulation import solve_cascade, solve_module


ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = Path(os.environ.get("MODEL_BUNDLE_PATH", ROOT / "artifacts" / "membrane_models.joblib"))
MODEL_BUNDLE = None


def load_model_bundle():
    if not MODEL_BUNDLE_PATH.is_file():
        return None
    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except (OSError, ValueError, ImportError, EOFError) as error:
        raise RuntimeError(f"Could not load model artifact {MODEL_BUNDLE_PATH}: {error}") from error
    return validate_model_bundle(bundle)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(204, {})

    def do_GET(self):
        if self.path != "/api/health":
            self.send_json(404, {"error": "Not found"})
            return
        if MODEL_BUNDLE is None:
            self.send_json(503, {
                "model_ready": False,
                "error": "Trained model bundle is unavailable",
                "train_command": "python -m backend.train_model",
            })
            return
        metadata = MODEL_BUNDLE["metadata"]
        self.send_json(200, {
            "model_ready": True,
            "available_models": [
                {
                    "id": model_id,
                    "label": option["label"],
                    "test_metrics": metadata["test_metrics"][model_id],
                }
                for model_id, option in MODEL_OPTIONS.items()
            ],
            "dataset_size": metadata["dataset_size"],
            "feature_count": metadata["feature_count"],
            "trained_at": metadata["trained_at"],
        })

    def do_POST(self):
        if self.path not in {"/api/predict", "/api/process"}:
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Content-Length must be an integer"})
            return
        if length > 16_384:
            self.send_json(413, {"error": "SMILES input is too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Request body must be valid JSON"})
            return
        if not isinstance(payload, dict):
            self.send_json(400, {"error": "Request JSON must be an object"})
            return
        if self.path == "/api/process":
            self.do_process(payload)
            return
        model_id = payload.get("model")
        if not isinstance(model_id, str) or model_id not in MODEL_OPTIONS:
            self.send_json(400, {
                "error": "Choose a supported model: lr, rf, or xg",
            })
            return
        smiles = payload.get("smiles", "")
        if not isinstance(smiles, str) or not smiles.strip():
            self.send_json(400, {"error": "Provide a non-empty SMILES string"})
            return
        if MODEL_BUNDLE is None:
            self.send_json(503, {
                "error": "Trained model bundle is unavailable. Run `python -m backend.train_model` first.",
            })
            return
        metadata = MODEL_BUNDLE["metadata"]
        model_label = MODEL_OPTIONS[model_id]["label"]
        try:
            features, descriptors = build_feature_frame(
                smiles.strip(), metadata["feature_names"]
            )
        except ValueError as error:
            self.send_json(422, {"error": str(error)})
            return
        try:
            permeability_log10 = float(
                MODEL_BUNDLE["models"][model_id]["permeability"].predict(features)[0]
            )
            selectivity_log10 = float(
                MODEL_BUNDLE["models"][model_id]["selectivity"].predict(features)[0]
            )
            permeability = float(np.power(10, permeability_log10))
            selectivity = float(np.power(10, selectivity_log10))
            if not math.isfinite(permeability) or not math.isfinite(selectivity):
                raise ValueError("The trained model returned a non-finite result")
            self.send_json(200, {
                "permeability": round(permeability, 2),
                "selectivity": round(selectivity, 2),
                "log10_permeability": round(permeability_log10, 6),
                "log10_selectivity": round(selectivity_log10, 6),
                "selected_model": {"id": model_id, "label": model_label},
                "descriptors": descriptors,
                "models": {
                    "permeability": model_label,
                    "selectivity": model_label,
                },
                "test_metrics": metadata["test_metrics"][model_id],
            })
        except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError, XGBoostError) as error:
            print(f"Model inference failed for {smiles!r}: {error}")
            self.send_json(500, {"error": f"Model inference failed: {error}"})

    def do_process(self, payload):
        """Validate process inputs and run the selected process model."""
        solver = payload.get("solver", "screening")
        if solver == "rigorous":
            self.do_rigorous_process(payload)
            return
        if solver != "screening":
            self.send_json(400, {"error": "Choose a supported process solver"})
            return

        try:
            permeability = float(payload["permeability_barrer"])
            selectivity = float(payload["selectivity"])
            design = payload["design"]
            if not isinstance(design, dict):
                raise ValueError("design must be a JSON object")
            layout = design["layout"]
            flow_pattern = design["flow_pattern"]
            feed_flow = float(design["feed_flow_kmol_h"])
            feed_fraction = float(design["feed_co2_fraction"])
            feed_pressure = float(design["feed_pressure_bar"])
            permeate_pressure = float(design["permeate_pressure_bar"])
            thickness = float(design["thickness_um"])
            stage_cut = float(design["target_stage_cut"])
            pipeline_pressure = float(design["pipeline_pressure_bar"])
            electricity_price = float(design["electricity_price"])
            if not all(math.isfinite(value) for value in (
                permeability, selectivity, feed_flow, feed_fraction, feed_pressure,
                permeate_pressure, thickness, stage_cut, pipeline_pressure, electricity_price,
            )):
                raise ValueError("process inputs must be finite numbers")
            if permeability <= 0 or selectivity <= 0:
                raise ValueError("Stage 1 permeability and selectivity must be positive")
            if layout not in {"SINGLE_STAGE", "TWO_STAGE_RECYCLE", "THREE_STAGE"}:
                raise ValueError("Choose a supported cascade layout")
            stage_count = {"SINGLE_STAGE": 1, "TWO_STAGE_RECYCLE": 2, "THREE_STAGE": 3}[layout]
            stages = tuple(
                StageConfig(
                    P_feed_bar=feed_pressure,
                    P_perm_bar=permeate_pressure if index == 0 else min(
                        permeate_pressure, feed_pressure * 0.1
                    ),
                    target_stage_cut=stage_cut if index == 0 else min(stage_cut, 0.3),
                )
                for index in range(stage_count)
            )
            module = ModuleConfig(
                thickness_um=thickness,
                P_CO2_Barrer=permeability,
                alpha_CO2_N2=selectivity,
                flow_pattern=flow_pattern,
                segments=8,
            )
            cascade_config = CascadeConfig(
                layout=layout,
                stages=stages,
                feed_flow_kmol_h=feed_flow,
                feed_CO2_molfrac=feed_fraction,
                pipeline_pressure_bar=pipeline_pressure,
                electricity_price=electricity_price,
            )
            result = MembraneCascade(
                module, cascade_config, PretreatmentUnit()
            ).solve()
            tea = result.compute_tea(electricity_price=electricity_price)
            per_stage = [
                {
                    key: value
                    for key, value in stage.items()
                    if key not in {"retentate_components", "permeate_components"}
                }
                for stage in result.per_stage
            ]
            self.send_json(200, {
                "purity": result.purity,
                "recovery": result.recovery,
                "stage_cut": result.stage_cut,
                "total_area_m2": result.total_area_m2,
                "total_compression_kW": result.total_compression_kW,
                "specific_energy_kWh_per_t": result.specific_energy_kWh_per_t,
                "captured_CO2_kmol_h": result.captured_CO2_kmol_h,
                "converged": result.converged,
                "n_iterations": result.n_iterations,
                "per_stage": per_stage,
                "energy_details": result.energy_details,
                "tea": {
                    "total_capital_cost_usd": tea.total_capital_cost_usd,
                    "annual_fixed_opex_usd": tea.annual_fixed_opex_usd,
                    "annual_variable_opex_usd": tea.annual_variable_opex_usd,
                    "levelized_cost_usd_per_tonne": tea.levelized_cost_usd_per_tonne,
                    "cost_breakdown": tea.cost_breakdown,
                    "sensitivity": tea.sensitivity,
                },
                "assumption_note": (
                    "Screening estimates using idealized flow-pattern treatment and "
                    "configurable cost allowances; not literature-validated."
                ),
            })
        except (KeyError, TypeError, ValueError) as error:
            self.send_json(400, {"error": f"Invalid process input: {error}"})
        except (RuntimeError, FloatingPointError, OverflowError) as error:
            self.send_json(422, {"error": f"Process model could not converge: {error}"})

    def do_rigorous_process(self, payload):
        """Run the standalone 1D module solver for single- or two-stage designs."""
        try:
            permeability = float(payload["permeability_barrer"])
            selectivity = float(payload["selectivity"])
            design = payload["design"]
            if not isinstance(design, dict):
                raise ValueError("design must be a JSON object")

            layout = design["layout"]
            flow_pattern = design["flow_pattern"]
            feed_flow = float(design["feed_flow_kmol_h"])
            feed_fraction = float(design["feed_co2_fraction"])
            feed_pressure = float(design["feed_pressure_bar"])
            permeate_pressure = float(design["permeate_pressure_bar"])
            thickness = float(design["thickness_um"])
            area_stage1 = float(design["area_stage1_m2"])
            values = (
                permeability, selectivity, feed_flow, feed_fraction,
                feed_pressure, permeate_pressure, thickness, area_stage1,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("process inputs must be finite numbers")
            if permeability <= 0 or selectivity <= 0:
                raise ValueError("Stage 1 permeability and selectivity must be positive")
            if layout not in {"SINGLE_STAGE", "TWO_STAGE_RECYCLE"}:
                raise ValueError(
                    "The rigorous solver supports single-stage and two-stage recycle layouts"
                )
            if flow_pattern not in {"cocurrent", "countercurrent"}:
                raise ValueError(
                    "The rigorous solver supports cocurrent or countercurrent flow"
                )

            if layout == "SINGLE_STAGE":
                module = solve_module(
                    area=area_stage1,
                    P_CO2_barrer=permeability,
                    alpha=selectivity,
                    P_feed=feed_pressure,
                    P_perm=permeate_pressure,
                    x_feed=feed_fraction,
                    F_feed=feed_flow,
                    flow_pattern=flow_pattern,
                    thickness_um=thickness,
                )
                stages = [module]
                purity = module.purity_y
                recovery = module.recovery
                product_flow = module.permeate_flow
                product_co2_flow = module.permeate_co2_flow
                total_area = module.area
                n_iterations = 0
            else:
                area_stage2 = float(design["area_stage2_m2"])
                if not math.isfinite(area_stage2):
                    raise ValueError("area_stage2_m2 must be a finite number")
                cascade = solve_cascade(
                    P_CO2_barrer=permeability,
                    alpha=selectivity,
                    P_feed=feed_pressure,
                    P_perm=permeate_pressure,
                    x_feed=feed_fraction,
                    F_feed=feed_flow,
                    flow_pattern=flow_pattern,
                    area_stage1=area_stage1,
                    area_stage2=area_stage2,
                    thickness_um=thickness,
                )
                stages = [cascade.stage1, cascade.stage2]
                purity = cascade.product_purity
                recovery = cascade.overall_recovery
                product_flow = cascade.product_flow
                product_co2_flow = cascade.product_co2_flow
                total_area = cascade.stage1.area + cascade.stage2.area
                n_iterations = cascade.iterations

            per_stage = [
                {
                    "stage": index,
                    "feed_flow_kmol_h": stage.feed_flow,
                    "permeate_flow_kmol_h": stage.permeate_flow,
                    "permeate_composition": [
                        stage.purity_y,
                        1.0 - stage.purity_y,
                    ],
                    "retentate_flow_kmol_h": stage.retentate_flow,
                    "retentate_composition": [
                        stage.retentate_x,
                        1.0 - stage.retentate_x,
                    ],
                    "area_m2": stage.area,
                    "stage_cut": stage.stage_cut,
                    "recovery": stage.recovery,
                }
                for index, stage in enumerate(stages, start=1)
            ]
            self.send_json(200, {
                "solver": "rigorous_1d",
                "purity": purity,
                "recovery": recovery,
                "stage_cut": stages[0].stage_cut,
                "total_area_m2": total_area,
                "captured_CO2_kmol_h": product_co2_flow,
                "product_flow_kmol_h": product_flow,
                "converged": True,
                "n_iterations": n_iterations,
                "per_stage": per_stage,
                "assumption_note": (
                    "Binary CO₂/N₂ solution-diffusion model with fixed stage pressures "
                    "and a 100-segment area integration. Energy and economic estimates "
                    "are not calculated by the rigorous solver."
                ),
            })
        except (KeyError, TypeError, ValueError) as error:
            self.send_json(400, {"error": f"Invalid process input: {error}"})
        except (RuntimeError, FloatingPointError, OverflowError) as error:
            self.send_json(422, {"error": f"Process model could not converge: {error}"})

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


if __name__ == "__main__":
    MODEL_BUNDLE = load_model_bundle()
    if MODEL_BUNDLE is None:
        print(f"Model artifact not found at {MODEL_BUNDLE_PATH}. Run `python -m backend.train_model` first.")
    else:
        details = MODEL_BUNDLE["metadata"]
        print(
            f"Loaded {details['dataset_size']}-polymer model bundle "
            f"with {details['feature_count']} features from {MODEL_BUNDLE_PATH}."
        )
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving prediction API at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
