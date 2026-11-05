"""FastAPI service for Stage 1 polymer predictions and rigorous Stage 2 design."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from xgboost.core import XGBoostError

from backend.ml_pipeline import (
    MODEL_OPTIONS,
    build_feature_frame,
    validate_model_bundle,
)
from stage2_engine import (
    EnergyCostResult,
    ModuleResult,
    calculate_energy_and_cost,
    find_required_area,
    solve_module,
)


LOGGER = logging.getLogger("membrane_lab.api")
ROOT = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = Path(
    os.environ.get("MODEL_BUNDLE_PATH", ROOT / "artifacts" / "membrane_models.joblib")
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model_bundle = None
    application.state.model_load_error = None
    if MODEL_BUNDLE_PATH.is_file():
        try:
            application.state.model_bundle = validate_model_bundle(
                joblib.load(MODEL_BUNDLE_PATH)
            )
        except (OSError, ValueError, ImportError, EOFError) as error:
            application.state.model_load_error = str(error)
            LOGGER.exception("Could not load the Stage 1 model bundle")
    else:
        application.state.model_load_error = (
            f"Model artifact not found at {MODEL_BUNDLE_PATH}"
        )
        LOGGER.warning(
            "%s; train models with `python -m backend.train_model`",
            application.state.model_load_error,
        )
    yield


app = FastAPI(
    title="Membrane Lab API",
    description="Stage 1 polymer inference and rigorous countercurrent Stage 2 simulation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: str = Field(min_length=1, max_length=16_384)
    model: Literal["lr", "rf", "xg"]


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    P_CO2: float = Field(gt=0, allow_inf_nan=False)
    alpha: float = Field(gt=0, allow_inf_nan=False)
    thickness: float = Field(gt=0, allow_inf_nan=False)
    P_feed: float = Field(gt=0, allow_inf_nan=False)
    P_perm: float = Field(ge=0, allow_inf_nan=False)
    F_feed: float = Field(gt=0, allow_inf_nan=False)
    x_feed: float = Field(gt=0, lt=1, allow_inf_nan=False)
    mode: Literal["design", "simulation"]
    target_value: float = Field(gt=0, allow_inf_nan=False)


class AreaSweepPoint(BaseModel):
    area: float
    recovery: float | None
    purity: float | None


class PressureSweepPoint(BaseModel):
    pressure: float
    area: float | None


class GraphData(BaseModel):
    area_sweep: list[AreaSweepPoint]
    pressure_sweep: list[PressureSweepPoint]


class SimulationResponse(BaseModel):
    purity: float
    recovery: float
    area: float
    specific_energy: float
    cost: float
    product_co2_flow: float
    product_flow: float
    graphs: GraphData


@lru_cache(maxsize=4096)
def _cached_module(
    area: float,
    permeability: float,
    selectivity: float,
    feed_pressure: float,
    permeate_pressure: float,
    feed_fraction: float,
    feed_flow: float,
    thickness: float,
) -> ModuleResult:
    return solve_module(
        area=area,
        P_CO2_barrer=permeability,
        alpha=selectivity,
        P_feed=feed_pressure,
        P_perm=permeate_pressure,
        x_feed=feed_fraction,
        F_feed=feed_flow,
        thickness_um=thickness,
    )


@lru_cache(maxsize=1024)
def _cached_design_area(
    permeability: float,
    selectivity: float,
    feed_pressure: float,
    permeate_pressure: float,
    feed_fraction: float,
    feed_flow: float,
    recovery: float,
    thickness: float,
) -> float:
    return find_required_area(
        P_CO2_barrer=permeability,
        alpha=selectivity,
        P_feed=feed_pressure,
        P_perm=permeate_pressure,
        x_feed=feed_fraction,
        F_feed=feed_flow,
        target_recovery=recovery,
        thickness_um=thickness,
    )


def _solve_module(request: SimulationRequest, area: float) -> ModuleResult:
    return _cached_module(
        area,
        request.P_CO2,
        request.alpha,
        request.P_feed,
        request.P_perm,
        request.x_feed,
        request.F_feed,
        request.thickness,
    )


def _energy_result(request: SimulationRequest, result: ModuleResult) -> EnergyCostResult:
    return calculate_energy_and_cost(
        feed_flow_kmol_h=request.F_feed,
        feed_pressure_bar=request.P_feed,
        permeate_pressure_bar=request.P_perm,
        permeate_flow_kmol_h=result.permeate_flow_kmol_h,
        captured_co2_kmol_h=result.permeate_co2_flow_kmol_h,
        total_membrane_area_m2=result.area_m2,
    )


def _build_graphs(
    request: SimulationRequest,
    base_area: float,
) -> GraphData:
    area_values = np.unique(
        np.append(np.linspace(base_area * 0.1, base_area * 2.0, 8), base_area)
    )
    area_points: list[AreaSweepPoint] = []
    for area in area_values:
        try:
            result = _solve_module(request, float(area))
        except (ValueError, RuntimeError):
            area_points.append(
                AreaSweepPoint(area=float(area), recovery=None, purity=None)
            )
        else:
            area_points.append(
                AreaSweepPoint(
                    area=float(area),
                    recovery=result.recovery,
                    purity=result.purity_co2,
                )
            )

    max_pressure = min(0.9, request.P_feed * 0.99)
    pressure_values = np.linspace(min(0.02, max_pressure * 0.1), max_pressure, 6)
    pressure_values = np.unique(
        np.append(pressure_values, [request.P_perm, min(0.1, max_pressure), min(0.2, max_pressure)])
    )
    pressure_points: list[PressureSweepPoint] = []
    for pressure in pressure_values:
        pressure = float(pressure)
        if pressure >= request.P_feed:
            pressure_points.append(PressureSweepPoint(pressure=pressure, area=None))
            continue
        try:
            required_area = _cached_design_area(
                request.P_CO2,
                request.alpha,
                request.P_feed,
                pressure,
                request.x_feed,
                request.F_feed,
                0.9,
                request.thickness,
            )
        except (ValueError, RuntimeError):
            required_area = None
        pressure_points.append(
            PressureSweepPoint(pressure=pressure, area=required_area)
        )
    return GraphData(area_sweep=area_points, pressure_sweep=pressure_points)


@app.get("/api/health")
def health(request: Request) -> dict[str, object]:
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_load_error
            or "Trained model bundle is unavailable",
        )
    metadata = bundle["metadata"]
    return {
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
    }


@app.get("/")
def root() -> dict[str, str]:
    """Provide a useful landing response when the API host is opened directly."""
    return {
        "service": "Membrane Lab API",
        "status": "ok",
        "docs": "/docs",
        "simulate": "POST /api/simulate",
        "predict": "POST /api/predict",
    }


@app.post("/api/predict")
def predict(payload: PredictionRequest, request: Request) -> dict[str, object]:
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                request.app.state.model_load_error
                or "Train models with `python -m backend.train_model` first."
            ),
        )
    try:
        features, descriptors = build_feature_frame(
            payload.smiles.strip(), bundle["metadata"]["feature_names"]
        )
        permeability_log10 = float(
            bundle["models"][payload.model]["permeability"].predict(features)[0]
        )
        selectivity_log10 = float(
            bundle["models"][payload.model]["selectivity"].predict(features)[0]
        )
        permeability = float(np.power(10.0, permeability_log10))
        selectivity = float(np.power(10.0, selectivity_log10))
        if not math.isfinite(permeability) or not math.isfinite(selectivity):
            raise ValueError("The trained model returned a non-finite result")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (TypeError, RuntimeError, FloatingPointError, OverflowError, XGBoostError) as error:
        LOGGER.exception("Stage 1 inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from error

    metadata = bundle["metadata"]
    return {
        "permeability": round(permeability, 2),
        "selectivity": round(selectivity, 2),
        "log10_permeability": round(permeability_log10, 6),
        "log10_selectivity": round(selectivity_log10, 6),
        "selected_model": {
            "id": payload.model,
            "label": MODEL_OPTIONS[payload.model]["label"],
        },
        "descriptors": descriptors,
        "models": {
            "permeability": MODEL_OPTIONS[payload.model]["label"],
            "selectivity": MODEL_OPTIONS[payload.model]["label"],
        },
        "test_metrics": metadata["test_metrics"][payload.model],
    }


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest) -> SimulationResponse:
    if payload.P_perm >= payload.P_feed:
        raise HTTPException(
            status_code=422,
            detail="Permeate pressure must be below feed pressure.",
        )
    if payload.mode == "design" and not 0 < payload.target_value < 0.99:
        raise HTTPException(
            status_code=422,
            detail="Design recovery must be greater than 0 and less than 99%.",
        )

    try:
        if payload.mode == "design":
            area = _cached_design_area(
                payload.P_CO2,
                payload.alpha,
                payload.P_feed,
                payload.P_perm,
                payload.x_feed,
                payload.F_feed,
                payload.target_value,
                payload.thickness,
            )
        else:
            area = payload.target_value
        result = _solve_module(payload, area)
        energy = _energy_result(payload, result)
        graphs = _build_graphs(payload, area)
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return SimulationResponse(
        purity=result.purity_co2,
        recovery=float(np.clip(result.permeate_co2_flow_kmol_h / result.feed_co2_flow_kmol_h, 0.0, 1.0)),
        area=result.area_m2,
        specific_energy=energy.specific_energy_kwh_per_tonne_co2,
        cost=energy.levelized_capture_cost_usd_per_tonne,
        product_co2_flow=result.permeate_co2_flow_kmol_h,
        product_flow=result.permeate_flow_kmol_h,
        graphs=graphs,
    )
