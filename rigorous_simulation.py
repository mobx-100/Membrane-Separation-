"""One-dimensional solution-diffusion membrane module and recycle simulation.

Flows are molar rates in kmol/h, pressures are in bar, permeability is in
Barrer, and membrane thickness is in micrometers. The default thickness is
1 micrometer because it is not part of the compact public API requested for
``solve_module``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

GPU_TO_SI = 3.348e-10
BAR_TO_PA = 1e5
MOL_S_TO_KMOL_H = 3.6
N_SEGMENTS = 100
FlowPattern = Literal["cocurrent", "countercurrent"]


@dataclass(frozen=True)
class ModuleResult:
    """Steady-state material balance for one membrane module."""

    stage_cut: float
    recovery: float
    purity_y: float
    retentate_flow: float
    retentate_x: float
    area: float
    feed_flow: float
    feed_x: float
    permeate_flow: float
    permeate_co2_flow: float
    permeate_n2_flow: float
    retentate_co2_flow: float
    retentate_n2_flow: float


@dataclass(frozen=True)
class CascadeResult:
    """Two-stage permeate-recycle solution and its external material balance."""

    stage1: ModuleResult
    stage2: ModuleResult
    recycle_flow: float
    recycle_x: float
    product_flow: float
    product_co2_flow: float
    product_purity: float
    overall_recovery: float
    final_retentate_flow: float
    final_retentate_x: float
    iterations: int


def _validate_inputs(
    area: float,
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    flow_pattern: str,
    thickness_um: float,
) -> None:
    values = (area, P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed, thickness_um)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all module inputs must be finite")
    if area <= 0 or P_CO2_barrer <= 0 or alpha <= 0 or F_feed <= 0 or thickness_um <= 0:
        raise ValueError("area, permeability, selectivity, feed flow, and thickness must be positive")
    if P_feed <= 0 or P_perm < 0:
        raise ValueError("feed pressure must be positive and permeate pressure nonnegative")
    if P_perm >= P_feed:
        raise ValueError("pressure reversal: permeate pressure must be below feed pressure")
    if not 0 < x_feed < 1:
        raise ValueError("x_feed must be strictly between 0 and 1")
    if flow_pattern not in {"cocurrent", "countercurrent"}:
        raise ValueError("flow_pattern must be 'cocurrent' or 'countercurrent'")


def _permeation_flux(
    state: np.ndarray,
    P_feed_pa: float,
    P_perm_pa: float,
    permeances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feed = np.maximum(state[:2], 0.0)
    permeate = np.maximum(state[2:], 0.0)
    feed_total = max(float(feed.sum()), 1e-30)
    x = feed / feed_total
    perm_total = float(permeate.sum())
    if perm_total > 1e-20:
        y = permeate / perm_total
    else:
        initial_flux = permeances * (P_feed_pa * x)
        total_initial_flux = float(initial_flux.sum())
        y = initial_flux / total_initial_flux if total_initial_flux > 0 else x
    driving_force = P_feed_pa * x - P_perm_pa * y
    return permeances * driving_force, driving_force


def _check_pressure_reversal(
    feed: np.ndarray,
    permeate: np.ndarray,
    P_feed_pa: float,
    P_perm_pa: float,
    permeances: np.ndarray,
) -> None:
    state = np.concatenate((feed, permeate))
    _, driving_force = _permeation_flux(state, P_feed_pa, P_perm_pa, permeances)
    tolerance = 1e-9 * max(P_feed_pa, P_perm_pa, 1.0)
    if np.any(driving_force < -tolerance):
        raise ValueError(
            "pressure reversal: permeate-side partial pressure exceeds "
            "feed-side partial pressure for at least one component"
        )


def solve_module(
    area: float,
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    flow_pattern: FlowPattern,
    *,
    thickness_um: float = 1.0,
) -> ModuleResult:
    """Solve a binary CO2/N2 membrane module using 100 area segments.

    Cocurrent operation is an initial-value integration. Countercurrent
    operation is a two-point boundary-value problem: feed enters at area 0
    and the zero-flow permeate boundary is imposed at area ``A``.
    """
    _validate_inputs(
        area, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, flow_pattern, thickness_um,
    )
    P_feed_pa = P_feed * BAR_TO_PA
    P_perm_pa = P_perm * BAR_TO_PA
    permeances = np.array(
        [P_CO2_barrer / thickness_um, P_CO2_barrer / (alpha * thickness_um)],
        dtype=float,
    ) * GPU_TO_SI
    feed_in = np.array([F_feed * x_feed, F_feed * (1.0 - x_feed)], dtype=float)
    area_grid = np.linspace(0.0, area, N_SEGMENTS + 1)

    if flow_pattern == "cocurrent":
        initial_state = np.concatenate((feed_in, np.zeros(2)))

        def cocurrent_ode(area_position: float, state: np.ndarray) -> np.ndarray:
            del area_position
            flux, _ = _permeation_flux(state, P_feed_pa, P_perm_pa, permeances)
            return np.concatenate((-flux, flux)) * MOL_S_TO_KMOL_H

        solution = solve_ivp(
            cocurrent_ode,
            (0.0, area),
            initial_state,
            method="RK45",
            t_eval=area_grid,
            rtol=1e-9,
            atol=1e-12,
            max_step=area / N_SEGMENTS,
        )
        if not solution.success:
            raise RuntimeError(f"cocurrent module integration failed: {solution.message}")
        profiles = solution.y
        retentate = profiles[:2, -1]
        permeate = profiles[2:, -1]
        validation_indices = range(profiles.shape[1])
        states_to_check = profiles
    else:
        mesh = area_grid
        initial_co_current = solve_ivp(
            lambda area_position, state: np.concatenate(
                (-_permeation_flux(state, P_feed_pa, P_perm_pa, permeances)[0],
                 _permeation_flux(state, P_feed_pa, P_perm_pa, permeances)[0])
            ) * MOL_S_TO_KMOL_H,
            (0.0, area),
            np.concatenate((feed_in, np.zeros(2))),
            method="RK45",
            t_eval=mesh,
            rtol=1e-8,
            atol=1e-11,
            max_step=area / N_SEGMENTS,
        )
        if not initial_co_current.success:
            raise RuntimeError(
                f"countercurrent initial estimate failed: {initial_co_current.message}"
            )
        initial_permeate_out = np.maximum(initial_co_current.y[2:, -1], F_feed * 1e-10)

        def integrate_countercurrent(permeate_out: np.ndarray):
            initial_state = np.concatenate((feed_in, permeate_out))

            def countercurrent_ode(area_position: float, state: np.ndarray) -> np.ndarray:
                del area_position
                flux, _ = _permeation_flux(
                    state, P_feed_pa, P_perm_pa, permeances
                )
                return np.concatenate((-flux, -flux)) * MOL_S_TO_KMOL_H

            return solve_ivp(
                countercurrent_ode,
                (0.0, area),
                initial_state,
                method="RK45",
                t_eval=mesh,
                rtol=1e-8,
                atol=1e-11,
                max_step=area / N_SEGMENTS,
            )

        def boundary_residual(log_permeate_out: np.ndarray) -> np.ndarray:
            permeate_out = np.exp(np.clip(log_permeate_out, -30.0, 10.0)) * F_feed
            trial = integrate_countercurrent(permeate_out)
            if not trial.success:
                return np.full(2, 1e3)
            return trial.y[2:, -1] / F_feed

        guess = np.log(np.maximum(initial_permeate_out / F_feed, 1e-12))
        shooting = root(boundary_residual, guess, method="hybr")
        permeate_out = np.exp(np.clip(shooting.x, -30.0, 10.0)) * F_feed
        solution = integrate_countercurrent(permeate_out)
        if (
            not shooting.success
            or not solution.success
            or np.linalg.norm(solution.y[2:, -1] / F_feed, ord=np.inf) > 1e-7
        ):
            raise RuntimeError(
                f"countercurrent module solve failed: {shooting.message}"
            )
        states_to_check = solution.y
        retentate = states_to_check[:2, -1]
        permeate = states_to_check[2:, 0]
        validation_indices = range(states_to_check.shape[1])

    numerical_tolerance = 1e-8 * max(F_feed, 1.0)
    if np.any(retentate < -numerical_tolerance) or np.any(permeate < -numerical_tolerance):
        raise RuntimeError("module solution contains a negative component flow")
    retentate = np.maximum(retentate, 0.0)
    permeate = feed_in - retentate
    if np.any(permeate < -numerical_tolerance):
        raise RuntimeError("module solution transfers more material than the feed contains")
    permeate = np.maximum(permeate, 0.0)

    for column in validation_indices:
        _check_pressure_reversal(
            np.maximum(states_to_check[:2, column], 0.0),
            np.maximum(states_to_check[2:, column], 0.0),
            P_feed_pa,
            P_perm_pa,
            permeances,
        )

    retentate_flow = float(retentate.sum())
    permeate_flow = float(permeate.sum())
    permeate_co2 = float(permeate[0])
    feed_co2 = float(feed_in[0])
    if permeate_flow <= 0:
        raise ValueError("operating conditions produce no net permeate")
    recovery = float(np.clip(permeate_co2 / feed_co2, 0.0, 1.0))
    return ModuleResult(
        stage_cut=permeate_flow / F_feed,
        recovery=recovery,
        purity_y=permeate_co2 / permeate_flow,
        retentate_flow=retentate_flow,
        retentate_x=float(retentate[0] / retentate_flow) if retentate_flow else 0.0,
        area=area,
        feed_flow=F_feed,
        feed_x=x_feed,
        permeate_flow=permeate_flow,
        permeate_co2_flow=permeate_co2,
        permeate_n2_flow=float(permeate[1]),
        retentate_co2_flow=float(retentate[0]),
        retentate_n2_flow=float(retentate[1]),
    )


def find_required_area(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    flow_pattern: FlowPattern,
    target_recovery: float,
    *,
    thickness_um: float = 1.0,
) -> float:
    """Use Brent's method to find area for a physically attainable recovery.

    Targets at or above 99% are rejected because they are generally poorly
    conditioned and may be thermodynamically unattainable at finite permeate
    pressure.
    """
    if not math.isfinite(target_recovery) or not 0 < target_recovery < 0.99:
        raise ValueError("target_recovery must be greater than 0 and less than 0.99")
    _validate_inputs(
        1.0, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, flow_pattern, thickness_um,
    )

    permeance_co2 = (P_CO2_barrer / thickness_um) * GPU_TO_SI
    initial_flux = permeance_co2 * P_feed * BAR_TO_PA * x_feed * MOL_S_TO_KMOL_H
    if initial_flux <= 0:
        raise ValueError("operating conditions have no initial CO2 driving force")
    area_low = max(F_feed * target_recovery / initial_flux * 1e-5, 1e-10)
    area_high = max(F_feed * target_recovery / initial_flux, area_low * 2.0)

    def recovery_error(area: float) -> float:
        return solve_module(
            area, P_CO2_barrer, alpha, P_feed, P_perm,
            x_feed, F_feed, flow_pattern, thickness_um=thickness_um,
        ).recovery - target_recovery

    low_error = recovery_error(area_low)
    if low_error >= 0:
        return area_low
    for _ in range(80):
        try:
            high_error = recovery_error(area_high)
        except ValueError as error:
            if "pressure reversal" in str(error):
                raise ValueError(
                    f"target recovery {target_recovery:.3f} is not attainable "
                    "before pressure reversal"
                ) from error
            raise
        if high_error >= 0:
            return float(brentq(
                recovery_error, area_low, area_high, xtol=1e-10, rtol=1e-10
            ))
        area_high *= 2.0
        if area_high > 1e12:
            break
    raise ValueError(
        f"target recovery {target_recovery:.3f} is not achievable for these conditions"
    )


def solve_cascade(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    flow_pattern: FlowPattern,
    area_stage1: float,
    area_stage2: float,
    *,
    thickness_um: float = 1.0,
    recycle_tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> CascadeResult:
    """Solve a two-stage cascade with Stage 2 permeate recycled to Stage 1.

    ``scipy.optimize.root`` closes the two component recycle-flow equations.
    The external product is Stage 1 permeate; the external residual is Stage
    2 retentate.
    """
    _validate_inputs(
        area_stage1, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, flow_pattern, thickness_um,
    )
    _validate_inputs(
        area_stage2, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, flow_pattern, thickness_um,
    )
    if recycle_tolerance <= 0 or not math.isfinite(recycle_tolerance):
        raise ValueError("recycle_tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    fresh = np.array([F_feed * x_feed, F_feed * (1.0 - x_feed)])
    initial_recycle = fresh * 0.05

    def recycle_residual(recycle: np.ndarray) -> np.ndarray:
        if np.any(recycle < 0) or not np.all(np.isfinite(recycle)):
            return np.full(2, 1e6) + np.abs(np.nan_to_num(recycle, nan=1e6))
        stage1_feed = fresh + recycle
        stage1 = solve_module(
            area_stage1, P_CO2_barrer, alpha, P_feed, P_perm,
            float(stage1_feed[0] / stage1_feed.sum()),
            float(stage1_feed.sum()),
            flow_pattern,
            thickness_um=thickness_um,
        )
        stage2 = solve_module(
            area_stage2, P_CO2_barrer, alpha, P_feed, P_perm,
            stage1.retentate_x,
            stage1.retentate_flow,
            flow_pattern,
            thickness_um=thickness_um,
        )
        recycle_out = np.array(
            [stage2.permeate_co2_flow, stage2.permeate_n2_flow], dtype=float
        )
        return (recycle - recycle_out) / F_feed

    solution = root(
        recycle_residual,
        initial_recycle,
        method="hybr",
        options={"xtol": recycle_tolerance, "maxfev": max_iterations * 10},
    )
    recycle = np.asarray(solution.x, dtype=float)
    residual = recycle_residual(recycle)
    if (
        not solution.success
        or np.any(recycle < 0)
        or np.linalg.norm(residual, ord=np.inf) > recycle_tolerance
    ):
        raise RuntimeError(
            f"cascade recycle did not converge: {solution.message}; "
            f"normalized residual {np.linalg.norm(residual, ord=np.inf):.3g}"
        )

    stage1_feed = fresh + recycle
    stage1 = solve_module(
        area_stage1, P_CO2_barrer, alpha, P_feed, P_perm,
        float(stage1_feed[0] / stage1_feed.sum()),
        float(stage1_feed.sum()),
        flow_pattern,
        thickness_um=thickness_um,
    )
    stage2 = solve_module(
        area_stage2, P_CO2_barrer, alpha, P_feed, P_perm,
        stage1.retentate_x,
        stage1.retentate_flow,
        flow_pattern,
        thickness_um=thickness_um,
    )
    external_feed_co2 = F_feed * x_feed
    overall_recovery = float(
        np.clip(stage1.permeate_co2_flow / external_feed_co2, 0.0, 1.0)
    )
    return CascadeResult(
        stage1=stage1,
        stage2=stage2,
        recycle_flow=float(recycle.sum()),
        recycle_x=float(recycle[0] / recycle.sum()) if recycle.sum() else 0.0,
        product_flow=stage1.permeate_flow,
        product_co2_flow=stage1.permeate_co2_flow,
        product_purity=stage1.purity_y,
        overall_recovery=overall_recovery,
        final_retentate_flow=stage2.retentate_flow,
        final_retentate_x=stage2.retentate_x,
        iterations=int(solution.nfev),
    )
