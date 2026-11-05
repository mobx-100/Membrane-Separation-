"""Countercurrent 1D CO2/N2 membrane solver and screening post-processing.

The module uses component molar flows as its state variables and integrates
over membrane area. The membrane is treated as an ideal binary separator at
constant feed/permeate pressure. Geometry, pressure drop, concentration
polarization, gas non-ideality, and detailed equipment correlations are not
included.

Units: permeability in Barrer, thickness in micrometers, pressure in bar,
area in m2, component flows in kmol/h, power in kW, and cost in USD.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root
from scipy.special import expit, logit

GPU_TO_SI = 3.348e-10
BAR_TO_PA = 1.0e5
MOL_S_TO_KMOL_H = 3.6
N_SEGMENTS = 100
R_GAS_J_PER_MOL_K = 8.314462618
DEFAULT_TEMPERATURE_K = 298.15
CO2_KMOL_TO_TONNE = 0.04401
GAS_HEAT_CAPACITY_RATIO = 1.4
COMPRESSION_STAGES = 3


@dataclass(frozen=True)
class ModuleResult:
    """Steady-state results and axial profiles for one countercurrent module."""

    area_m2: float
    stage_cut: float
    recovery: float
    purity_co2: float
    feed_flow_kmol_h: float
    feed_co2_fraction: float
    retentate_flow_kmol_h: float
    retentate_co2_fraction: float
    permeate_flow_kmol_h: float
    permeate_co2_fraction: float
    feed_co2_flow_kmol_h: float
    feed_n2_flow_kmol_h: float
    retentate_co2_flow_kmol_h: float
    retentate_n2_flow_kmol_h: float
    permeate_co2_flow_kmol_h: float
    permeate_n2_flow_kmol_h: float
    area_profile_m2: tuple[float, ...]
    feed_co2_profile: tuple[float, ...]
    permeate_co2_profile: tuple[float, ...]


@dataclass(frozen=True)
class SingleStageResult:
    """A single module sized to the requested feed-CO2 recovery."""

    module: ModuleResult
    target_recovery: float

    @property
    def area_m2(self) -> float:
        return self.module.area_m2


@dataclass(frozen=True)
class CascadeResult:
    """Two-module retentate-recycle cascade.

    Stage 1 permeate is the captured product. Stage 1 retentate feeds Stage 2.
    Stage 2 retentate returns to the Stage 1 mixed feed; Stage 2 permeate is a
    separate non-product outlet and is included in the external balances.
    """

    stage1: ModuleResult
    stage2: ModuleResult
    recycle_flow_kmol_h: float
    recycle_co2_fraction: float
    product_flow_kmol_h: float
    product_co2_flow_kmol_h: float
    product_purity: float
    overall_recovery: float
    stage1_area_m2: float
    stage2_area_m2: float
    iterations: int

    @property
    def total_area_m2(self) -> float:
        return self.stage1_area_m2 + self.stage2_area_m2


@dataclass(frozen=True)
class EnergyCostResult:
    """Transparent ideal-gas energy and annualized membrane-cost estimate."""

    feed_compression_kw: float
    vacuum_pumping_kw: float
    total_power_kw: float
    specific_energy_kwh_per_tonne_co2: float
    annual_capture_tonnes: float
    membrane_capital_usd: float
    annualized_capital_usd: float
    annual_fixed_opex_usd: float
    annual_electricity_cost_usd: float
    levelized_capture_cost_usd_per_tonne: float


def _validate_module_inputs(
    area_m2: float,
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    thickness_um: float,
    n_segments: int,
) -> None:
    numeric = (
        area_m2, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, thickness_um,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("all module inputs must be finite numbers")
    if area_m2 <= 0 or P_CO2_barrer <= 0 or alpha <= 0 or F_feed <= 0 or thickness_um <= 0:
        raise ValueError("area, permeability, selectivity, feed flow, and thickness must be positive")
    if P_feed <= 0 or P_perm < 0:
        raise ValueError("feed pressure must be positive and permeate pressure nonnegative")
    if P_perm >= P_feed:
        raise ValueError("pressure reversal: permeate pressure must be below feed pressure")
    if not 0 < x_feed < 1:
        raise ValueError("feed CO2 fraction must be strictly between 0 and 1")
    if not isinstance(n_segments, int) or n_segments < 2:
        raise ValueError("n_segments must be an integer of at least 2")


def _zero_flow_permeate_fraction(
    feed_fraction: np.ndarray,
    permeances: np.ndarray,
    P_feed_pa: float,
    P_perm_pa: float,
) -> np.ndarray:
    """Return the limiting permeate composition at the zero-flow boundary."""
    component_driving_scale = permeances * P_feed_pa * feed_fraction
    if float(component_driving_scale.sum()) <= 1e-300:
        return np.array([0.5, 0.5])
    if P_perm_pa == 0:
        return component_driving_scale / component_driving_scale.sum()

    def composition_sum(total_flux: float) -> float:
        return float(np.sum(
            component_driving_scale / (total_flux + permeances * P_perm_pa)
        ))

    total_flux = brentq(
        lambda value: composition_sum(value) - 1.0,
        0.0,
        2.0 * float(component_driving_scale.sum()),
    )
    return component_driving_scale / (total_flux + permeances * P_perm_pa)


def solve_module(
    area: float,
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    *,
    thickness_um: float = 1.0,
    n_segments: int = N_SEGMENTS,
) -> ModuleResult:
    """Solve the countercurrent module by a 100-segment shooting integration.

    The feed coordinate increases from the feed inlet to the retentate outlet.
    Permeate flow travels in the opposite direction. Both component balances
    therefore have a negative area derivative in this coordinate. The BVP
    boundary conditions are the known feed at area zero and zero permeate flow
    at the retentate end. ``scipy.optimize.root`` shoots on the two unknown
    retentate component rates at that end.

    Recovery is based on the CO2 component balance, never supplied as an
    independent process input.
    """
    _validate_module_inputs(
        area, P_CO2_barrer, alpha, P_feed, P_perm,
        x_feed, F_feed, thickness_um, n_segments,
    )
    P_feed_pa = P_feed * BAR_TO_PA
    P_perm_pa = P_perm * BAR_TO_PA
    permeances = np.array(
        [P_CO2_barrer / thickness_um, P_CO2_barrer / (alpha * thickness_um)],
        dtype=float,
    ) * GPU_TO_SI
    feed_in = np.array(
        [F_feed * x_feed, F_feed * (1.0 - x_feed)],
        dtype=float,
    )
    feed_total_in = float(feed_in.sum())
    area_grid = np.linspace(0.0, area, n_segments + 1)
    reverse_grid = area_grid[::-1]

    def flux_and_driving_force(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feed_components = np.maximum(state[:2], 0.0)
        permeate_components = np.maximum(state[2:], 0.0)
        feed_total = max(float(feed_components.sum()), 1e-24)
        feed_fraction_local = feed_components / feed_total
        permeate_total = float(permeate_components.sum())
        if permeate_total > 1e-14 * max(F_feed, 1.0):
            permeate_fraction_local = permeate_components / permeate_total
        else:
            permeate_fraction_local = _zero_flow_permeate_fraction(
                feed_fraction_local, permeances, P_feed_pa, P_perm_pa
            )
        driving_force = (
            P_feed_pa * feed_fraction_local
            - P_perm_pa * permeate_fraction_local
        )
        flux = permeances * driving_force * MOL_S_TO_KMOL_H
        return flux, driving_force

    def countercurrent_ode(_area_position: float, state: np.ndarray) -> np.ndarray:
        local_flux, _ = flux_and_driving_force(state)
        return np.concatenate((-local_flux, -local_flux))

    exposure = (
        permeances * P_feed_pa * MOL_S_TO_KMOL_H * area / feed_total_in
    )
    initial_retentate_fraction = np.clip(np.exp(-exposure), 1e-8, 1.0 - 1e-8)
    initial_guess = logit(initial_retentate_fraction)

    def shoot(log_retentate_fraction: np.ndarray) -> np.ndarray:
        retentate_fraction = expit(np.clip(log_retentate_fraction, -35.0, 35.0))
        retentate_at_end = feed_in * retentate_fraction
        state_at_end = np.concatenate((retentate_at_end, np.zeros(2)))
        trial = solve_ivp(
            countercurrent_ode,
            (area, 0.0),
            state_at_end,
            method="RK45",
            t_eval=[0.0],
            rtol=2e-8,
            atol=1e-11,
            max_step=area / n_segments,
        )
        if not trial.success or not np.all(np.isfinite(trial.y[:, -1])):
            return np.full(2, 1e3)
        return (trial.y[:2, -1] - feed_in) / feed_total_in

    shooting = root(
        shoot,
        initial_guess,
        method="hybr",
        options={"xtol": 1e-9, "maxfev": 300},
    )
    shooting_residual = shoot(shooting.x)
    if np.linalg.norm(shooting_residual, ord=np.inf) > 2e-7:
        raise RuntimeError(
            "countercurrent shooting failed to satisfy the feed boundary "
            f"(normalized residual {np.linalg.norm(shooting_residual, ord=np.inf):.3g}; "
            f"solver status: {shooting.message})"
        )

    retentate_fraction_at_end = expit(np.clip(shooting.x, -35.0, 35.0))
    retentate_at_end = feed_in * retentate_fraction_at_end
    state_at_end = np.concatenate((retentate_at_end, np.zeros(2)))
    profile = solve_ivp(
        countercurrent_ode,
        (area, 0.0),
        state_at_end,
        method="RK45",
        t_eval=reverse_grid,
        rtol=2e-9,
        atol=1e-12,
        max_step=area / n_segments,
    )
    if not profile.success:
        raise RuntimeError(f"countercurrent profile integration failed: {profile.message}")

    states = profile.y[:, ::-1]
    numerical_tolerance = 2e-7 * max(feed_total_in, 1.0)
    if np.any(states < -numerical_tolerance):
        raise RuntimeError("countercurrent profile contains a negative component flow")

    reversal_tolerance_pa = 1e-8 * max(P_feed_pa, P_perm_pa, 1.0)
    for column in range(states.shape[1]):
        _, local_driving_force = flux_and_driving_force(states[:, column])
        if np.any(local_driving_force < -reversal_tolerance_pa):
            raise ValueError(
                "pressure reversal: permeate-side partial pressure exceeds "
                "feed-side partial pressure within the module"
            )

    # Component balances are evaluated from the solved boundary streams.
    retentate = np.maximum(retentate_at_end, 0.0)
    permeate_from_balance = feed_in - retentate
    permeate_from_profile = states[2:, 0]
    if np.any(permeate_from_balance < -numerical_tolerance):
        raise RuntimeError("module transferred more of a component than the feed contains")
    component_error = np.max(
        np.abs(permeate_from_profile - permeate_from_balance)
        / np.maximum(feed_in, 1.0)
    )
    if component_error > 2e-6:
        raise RuntimeError(
            f"module component balance did not close (relative error {component_error:.3g})"
        )
    permeate = np.clip(permeate_from_balance, 0.0, feed_in)

    retentate_flow = float(retentate.sum())
    permeate_flow = float(permeate.sum())
    if permeate_flow <= 0.0:
        raise ValueError("operating conditions produce no net permeate")
    recovery = float(np.clip(permeate[0] / feed_in[0], 0.0, 1.0))
    assert recovery <= 1.0
    retentate_fraction = (
        float(retentate[0] / retentate_flow) if retentate_flow > 0 else 0.0
    )
    permeate_fraction = float(permeate[0] / permeate_flow)
    assert abs(feed_total_in - retentate_flow - permeate_flow) <= numerical_tolerance
    assert abs(feed_in[0] - retentate[0] - permeate[0]) <= numerical_tolerance

    feed_co2_profile = tuple(
        float(states[0, index] / max(states[:2, index].sum(), 1e-24))
        for index in range(states.shape[1])
    )
    permeate_co2_profile = tuple(
        float(states[2, index] / max(states[2:, index].sum(), 1e-24))
        for index in range(states.shape[1])
    )
    return ModuleResult(
        area_m2=area,
        stage_cut=permeate_flow / feed_total_in,
        recovery=recovery,
        purity_co2=permeate_fraction,
        feed_flow_kmol_h=feed_total_in,
        feed_co2_fraction=x_feed,
        retentate_flow_kmol_h=retentate_flow,
        retentate_co2_fraction=retentate_fraction,
        permeate_flow_kmol_h=permeate_flow,
        permeate_co2_fraction=permeate_fraction,
        feed_co2_flow_kmol_h=float(feed_in[0]),
        feed_n2_flow_kmol_h=float(feed_in[1]),
        retentate_co2_flow_kmol_h=float(retentate[0]),
        retentate_n2_flow_kmol_h=float(retentate[1]),
        permeate_co2_flow_kmol_h=float(permeate[0]),
        permeate_n2_flow_kmol_h=float(permeate[1]),
        area_profile_m2=tuple(float(value) for value in area_grid),
        feed_co2_profile=feed_co2_profile,
        permeate_co2_profile=permeate_co2_profile,
    )


def _area_estimate(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    target_recovery: float,
    thickness_um: float,
) -> float:
    """Provide a scale for bracketing; it is not a substitute for the ODE."""
    permeances = np.array(
        [P_CO2_barrer / thickness_um, P_CO2_barrer / (alpha * thickness_um)]
    ) * GPU_TO_SI
    feed_fraction = np.array([x_feed, 1.0 - x_feed])
    limit_y = _zero_flow_permeate_fraction(
        feed_fraction, permeances, P_feed * BAR_TO_PA, P_perm * BAR_TO_PA
    )
    co2_inlet_flux = (
        permeances[0]
        * (P_feed * BAR_TO_PA * x_feed - P_perm * BAR_TO_PA * limit_y[0])
        * MOL_S_TO_KMOL_H
    )
    if co2_inlet_flux <= 0:
        raise ValueError("operating conditions have no initial CO2 driving force")
    return max(F_feed * x_feed * target_recovery / co2_inlet_flux, 1e-8)


def find_required_area(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    target_recovery: float,
    *,
    thickness_um: float = 1.0,
) -> float:
    """Find the countercurrent area that achieves a feasible recovery target.

    Brent's method evaluates the spatial module solver at every trial area.
    Targets at or above 99% are rejected because they are poorly conditioned;
    targets that encounter a pressure pinch/reversal before the requested
    recovery are reported as physically unattainable for the supplied case.
    """
    if not math.isfinite(target_recovery) or not 0.0 < target_recovery < 0.99:
        raise ValueError("target recovery must be greater than 0 and less than 99%")
    estimate = _area_estimate(
        P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
        target_recovery, thickness_um,
    )

    def result_at(area_m2: float) -> ModuleResult:
        return solve_module(
            area_m2, P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
            thickness_um=thickness_um,
        )

    low_area = max(estimate * 1e-5, 1e-9)
    low_result = result_at(low_area)
    if low_result.recovery >= target_recovery:
        return low_area

    high_area = estimate
    high_result: ModuleResult | None = None
    for _ in range(32):
        try:
            high_result = result_at(high_area)
        except ValueError as error:
            if "pressure reversal" not in str(error):
                raise
            # Locate the largest valid area and determine whether the target
            # was reached before the local partial-pressure pinch.
            invalid_area = high_area
            valid_area = low_area
            valid_result = low_result
            for _ in range(32):
                trial_area = 0.5 * (valid_area + invalid_area)
                try:
                    trial_result = result_at(trial_area)
                except ValueError as trial_error:
                    if "pressure reversal" not in str(trial_error):
                        raise
                    invalid_area = trial_area
                else:
                    if trial_result.recovery >= target_recovery:
                        return float(brentq(
                            lambda area: result_at(area).recovery - target_recovery,
                            valid_area,
                            trial_area,
                            xtol=1e-8,
                            rtol=1e-9,
                        ))
                    valid_area, valid_result = trial_area, trial_result
            raise ValueError(
                f"target recovery {target_recovery:.1%} is not attainable "
                f"before pressure reversal (maximum found {valid_result.recovery:.1%})"
            ) from error

        if high_result.recovery >= target_recovery:
            return float(brentq(
                lambda area: result_at(area).recovery - target_recovery,
                low_area,
                high_area,
                xtol=1e-8,
                rtol=1e-9,
            ))
        low_area, low_result = high_area, high_result
        high_area *= 2.0
        if high_area > 1e9:
            break
    maximum = high_result.recovery if high_result is not None else low_result.recovery
    raise ValueError(
        f"target recovery {target_recovery:.1%} is not achievable for these conditions "
        f"(maximum found {maximum:.1%})"
    )


def solve_single_stage(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    target_recovery: float,
    *,
    thickness_um: float = 1.0,
) -> SingleStageResult:
    """Size one countercurrent module and return its calculated performance."""
    area_m2 = find_required_area(
        P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
        target_recovery, thickness_um=thickness_um,
    )
    module = solve_module(
        area_m2, P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
        thickness_um=thickness_um,
    )
    if abs(module.recovery - target_recovery) > 2e-6:
        raise RuntimeError("inverse area result did not meet its recovery target")
    return SingleStageResult(module=module, target_recovery=target_recovery)


def _solve_fixed_area_cascade(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    area_stage1_m2: float,
    area_stage2_m2: float,
    thickness_um: float,
    recycle_tolerance: float,
    max_iterations: int,
) -> CascadeResult:
    """Converge recycle component rates for a fixed pair of stage areas."""
    fresh = np.array([F_feed * x_feed, F_feed * (1.0 - x_feed)], dtype=float)
    scale = max(float(fresh.sum()), 1.0)
    
    def evaluate(
        recycle_components: np.ndarray,
    ) -> tuple[np.ndarray, ModuleResult, ModuleResult]:
        mixed = fresh + recycle_components
        stage1_result = solve_module(
            area_stage1_m2, P_CO2_barrer, alpha, P_feed, P_perm,
            float(mixed[0] / mixed.sum()), float(mixed.sum()),
            thickness_um=thickness_um,
        )
        if stage1_result.retentate_flow_kmol_h <= 0:
            raise ValueError("Stage 1 has no retentate flow for Stage 2")
        stage2_result = solve_module(
            area_stage2_m2, P_CO2_barrer, alpha, P_feed, P_perm,
            stage1_result.retentate_co2_fraction,
            stage1_result.retentate_flow_kmol_h,
            thickness_um=thickness_um,
        )
        next_recycle = np.array(
            [
                stage2_result.retentate_co2_flow_kmol_h,
                stage2_result.retentate_n2_flow_kmol_h,
            ],
            dtype=float,
        )
        return recycle_components - next_recycle, stage1_result, stage2_result

    _, zero_stage1, zero_stage2 = evaluate(np.zeros(2))
    large_trial_recycle = fresh * 1e4
    large_trial_residual, _, _ = evaluate(large_trial_recycle)
    if np.any(large_trial_residual < -recycle_tolerance * scale):
        raise ValueError(
            "retentate recycle is not physically closable at these areas: "
            "Stage 2 permeate cannot purge every fresh-feed component load"
        )
    stage2_feed_at_zero = np.array([
        zero_stage1.retentate_co2_flow_kmol_h,
        zero_stage1.retentate_n2_flow_kmol_h,
    ])
    recycle_fraction = np.divide(
        [
            zero_stage2.retentate_co2_flow_kmol_h,
            zero_stage2.retentate_n2_flow_kmol_h,
        ],
        stage2_feed_at_zero,
        out=np.zeros(2),
        where=stage2_feed_at_zero > 0,
    )
    one_pass_retention = np.clip(
        (stage2_feed_at_zero / fresh) * recycle_fraction,
        0.0,
        1.0 - 1e-5,
    )
    recycle_guess = fresh * one_pass_retention / (1.0 - one_pass_retention)
    recycle = recycle_guess
    jacobian = np.diag(np.maximum(1.0 - one_pass_retention, 1e-5))
    residual, stage1, stage2 = evaluate(recycle)
    iteration = 0
    converged = False
    while iteration < max_iterations:
        if np.linalg.norm(residual, ord=np.inf) <= recycle_tolerance * scale:
            converged = True
            break
        iteration += 1
        try:
            newton_step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            newton_step = -residual

        current_norm = float(np.linalg.norm(residual, ord=2))
        accepted = False
        damping = 1.0
        for _ in range(8):
            trial_recycle = np.maximum(recycle + damping * newton_step, 0.0)
            try:
                trial_residual, trial_stage1, trial_stage2 = evaluate(trial_recycle)
            except ValueError as error:
                if "pressure reversal" not in str(error):
                    raise
                damping *= 0.5
                continue
            if np.linalg.norm(trial_residual, ord=2) < current_norm:
                displacement = trial_recycle - recycle
                residual_change = trial_residual - residual
                denominator = float(displacement @ displacement)
                if denominator > 0:
                    jacobian += np.outer(
                        residual_change - jacobian @ displacement,
                        displacement,
                    ) / denominator
                recycle = trial_recycle
                residual = trial_residual
                stage1, stage2 = trial_stage1, trial_stage2
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            # The fixed-point update is a safe, nonnegative fallback if the
            # local Broyden step does not reduce the balance residual.
            trial_recycle = np.maximum(recycle - residual, 0.0)
            try:
                trial_residual, trial_stage1, trial_stage2 = evaluate(trial_recycle)
            except ValueError as error:
                if "pressure reversal" not in str(error):
                    raise
                break
            if np.linalg.norm(trial_residual, ord=2) >= current_norm:
                break
            recycle = trial_recycle
            residual = trial_residual
            stage1, stage2 = trial_stage1, trial_stage2
            jacobian = np.diag(np.maximum(1.0 - one_pass_retention, 1e-5))

    if np.linalg.norm(residual, ord=np.inf) <= recycle_tolerance * scale:
        converged = True
    if not converged:
        raise RuntimeError(
            "retentate recycle did not converge within "
            f"{max_iterations} iterations (normalized residual "
            f"{np.linalg.norm(residual, ord=np.inf) / scale:.3g})"
        )

    recycle_final = np.array([
        stage2.retentate_co2_flow_kmol_h,
        stage2.retentate_n2_flow_kmol_h,
    ])
    if np.linalg.norm(recycle_final - recycle, ord=np.inf) > recycle_tolerance * scale:
        raise RuntimeError("retentate recycle changed after final balance evaluation")

    product = np.array([
        stage1.permeate_co2_flow_kmol_h,
        stage1.permeate_n2_flow_kmol_h,
    ])
    stage2_permeate = np.array([
        stage2.permeate_co2_flow_kmol_h,
        stage2.permeate_n2_flow_kmol_h,
    ])
    external_error = fresh - product - stage2_permeate
    if np.linalg.norm(external_error, ord=np.inf) > 5e-6 * scale:
        raise RuntimeError("cascade external component balance did not close")
    recovery = float(np.clip(product[0] / fresh[0], 0.0, 1.0))
    assert recovery <= 1.0
    product_flow = float(product.sum())
    return CascadeResult(
        stage1=stage1,
        stage2=stage2,
        recycle_flow_kmol_h=float(recycle.sum()),
        recycle_co2_fraction=float(recycle[0] / recycle.sum()) if recycle.sum() else 0.0,
        product_flow_kmol_h=product_flow,
        product_co2_flow_kmol_h=float(product[0]),
        product_purity=float(product[0] / product_flow) if product_flow else 0.0,
        overall_recovery=recovery,
        stage1_area_m2=area_stage1_m2,
        stage2_area_m2=area_stage2_m2,
        iterations=iteration,
    )


def solve_cascade(
    P_CO2_barrer: float,
    alpha: float,
    P_feed: float,
    P_perm: float,
    x_feed: float,
    F_feed: float,
    target_recovery: float,
    *,
    thickness_um: float = 1.0,
    stage2_area_ratio: float = 5.0,
    recycle_tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> CascadeResult:
    """Solve a two-stage retentate-recycle cascade.

    Stage 1 area is sized to the supplied recovery target with the inverse
    module solver. Stage 2 area is ``stage2_area_ratio * Stage 1 area``.
    The reported cascade recovery is calculated from Stage 1 product and the
    fresh feed; it need not equal the single-stage sizing target because Stage 2
    permeate is an external non-product outlet. The two component recycle
    rates are iterated with a damped Broyden method, subject to the supplied
    iteration limit. A recycle that cannot purge its incoming component load
    is rejected instead of producing an unbounded recycle rate.
    """
    if not 0 < target_recovery < 0.99 or not math.isfinite(target_recovery):
        raise ValueError("target recovery must be greater than 0 and less than 99%")
    if not math.isfinite(stage2_area_ratio) or stage2_area_ratio <= 0:
        raise ValueError("stage2_area_ratio must be finite and positive")
    if not math.isfinite(recycle_tolerance) or recycle_tolerance <= 0:
        raise ValueError("recycle_tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    sized_area = find_required_area(
        P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
        target_recovery, thickness_um=thickness_um,
    )
    return _solve_fixed_area_cascade(
        P_CO2_barrer, alpha, P_feed, P_perm, x_feed, F_feed,
        sized_area, sized_area * stage2_area_ratio, thickness_um,
        recycle_tolerance, max_iterations,
    )


def calculate_energy_and_cost(
    *,
    feed_flow_kmol_h: float,
    feed_pressure_bar: float,
    permeate_pressure_bar: float,
    permeate_flow_kmol_h: float,
    captured_co2_kmol_h: float,
    total_membrane_area_m2: float,
    electricity_price_usd_kwh: float = 0.07,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    compressor_efficiency: float = 0.75,
    vacuum_efficiency: float = 0.65,
    annual_operating_hours: float = 8000.0,
    membrane_cost_usd_m2: float = 100.0,
    membrane_lifetime_years: float = 5.0,
    discount_rate: float = 0.08,
    fixed_opex_fraction: float = 0.03,
) -> EnergyCostResult:
    """Estimate compression, vacuum work, SEC, and membrane cost.

    The compressor raises fresh feed from 1 bar to ``feed_pressure_bar``.
    The vacuum pump raises the combined permeate stream from its module
    pressure to 1 bar. Work uses a three-stage, perfectly intercooled ideal-gas
    isentropic estimate with the user-supplied overall isentropic efficiency.
    The kmol/h-to-mol/s and kmol-CO2-to-tonne conversions are explicit.
    Capture cost includes annualized membrane capital, fixed OPEX, and
    electricity; the membrane price, lifetime, discount rate, and OPEX fraction
    are explicit screening assumptions, not vendor cost correlations.
    """
    numeric = (
        feed_flow_kmol_h, feed_pressure_bar, permeate_pressure_bar,
        permeate_flow_kmol_h, captured_co2_kmol_h, total_membrane_area_m2,
        electricity_price_usd_kwh, temperature_k, compressor_efficiency,
        vacuum_efficiency, annual_operating_hours, membrane_cost_usd_m2,
        membrane_lifetime_years, discount_rate, fixed_opex_fraction,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("energy and cost inputs must be finite numbers")
    if feed_flow_kmol_h < 0 or permeate_flow_kmol_h < 0 or captured_co2_kmol_h <= 0:
        raise ValueError("flows must be nonnegative and captured CO2 flow must be positive")
    if feed_pressure_bar <= 0 or permeate_pressure_bar <= 0:
        raise ValueError("pressures must be positive for ideal-gas work calculations")
    if permeate_pressure_bar >= 1.0:
        raise ValueError("vacuum work requires permeate pressure below 1 bar")
    if total_membrane_area_m2 <= 0 or temperature_k <= 0:
        raise ValueError("area and temperature must be positive")
    if not 0 < compressor_efficiency <= 1 or not 0 < vacuum_efficiency <= 1:
        raise ValueError("compressor and vacuum efficiencies must be in (0, 1]")
    if annual_operating_hours <= 0 or membrane_lifetime_years <= 0:
        raise ValueError("annual hours and membrane lifetime must be positive")
    if not 0 <= discount_rate or membrane_cost_usd_m2 < 0 or fixed_opex_fraction < 0:
        raise ValueError("cost assumptions must be nonnegative")

    def intercooled_isentropic_compression_kw(
        flow_kmol_h: float, inlet_bar: float, outlet_bar: float, efficiency: float
    ) -> float:
        if flow_kmol_h <= 0 or outlet_bar <= inlet_bar:
            return 0.0
        pressure_ratio_per_stage = (
            outlet_bar / inlet_bar
        ) ** (1.0 / COMPRESSION_STAGES)
        molar_work_j_per_mol = (
            COMPRESSION_STAGES
            * GAS_HEAT_CAPACITY_RATIO
            / (GAS_HEAT_CAPACITY_RATIO - 1.0)
            * R_GAS_J_PER_MOL_K
            * temperature_k
            * (
                pressure_ratio_per_stage
                ** ((GAS_HEAT_CAPACITY_RATIO - 1.0) / GAS_HEAT_CAPACITY_RATIO)
                - 1.0
            )
            / efficiency
        )
        work_j_h = (
            flow_kmol_h * 1000.0 * molar_work_j_per_mol
        )
        return work_j_h / 3.6e6

    feed_work_kw = intercooled_isentropic_compression_kw(
        feed_flow_kmol_h, 1.0, feed_pressure_bar, compressor_efficiency
    )
    vacuum_work_kw = intercooled_isentropic_compression_kw(
        permeate_flow_kmol_h, permeate_pressure_bar, 1.0, vacuum_efficiency
    )
    total_power_kw = feed_work_kw + vacuum_work_kw
    captured_tonnes_h = captured_co2_kmol_h * CO2_KMOL_TO_TONNE
    sec = total_power_kw / captured_tonnes_h
    annual_capture_tonnes = captured_tonnes_h * annual_operating_hours

    membrane_capital = total_membrane_area_m2 * membrane_cost_usd_m2
    if discount_rate == 0:
        capital_recovery_factor = 1.0 / membrane_lifetime_years
    else:
        factor = (1.0 + discount_rate) ** membrane_lifetime_years
        capital_recovery_factor = (
            discount_rate * factor / (factor - 1.0)
        )
    annualized_capital = membrane_capital * capital_recovery_factor
    annual_fixed_opex = membrane_capital * fixed_opex_fraction
    annual_electricity_cost = (
        total_power_kw * annual_operating_hours * electricity_price_usd_kwh
    )
    levelized_cost = (
        annualized_capital + annual_fixed_opex + annual_electricity_cost
    ) / annual_capture_tonnes
    return EnergyCostResult(
        feed_compression_kw=feed_work_kw,
        vacuum_pumping_kw=vacuum_work_kw,
        total_power_kw=total_power_kw,
        specific_energy_kwh_per_tonne_co2=sec,
        annual_capture_tonnes=annual_capture_tonnes,
        membrane_capital_usd=membrane_capital,
        annualized_capital_usd=annualized_capital,
        annual_fixed_opex_usd=annual_fixed_opex,
        annual_electricity_cost_usd=annual_electricity_cost,
        levelized_capture_cost_usd_per_tonne=levelized_cost,
    )
