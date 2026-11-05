"""Screening model for membrane modules, cascades, energy, and capture economics.

The model is deliberately explicit about its assumptions. It is not a validated
plant simulator: module geometry, concentration polarization, gas non-ideality,
pretreatment, and equipment cost correlations are represented by configurable
screening approximations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

LOGGER = logging.getLogger(__name__)

GPU_TO_SI = 3.348e-10
BAR_TO_PA = 1.0e5
KMOL_H_TO_MOL_S = 1.0 / 3.6
R_GAS = 8.314462618
T_AMBIENT_K = 298.15
T_INTERCOOL_K = 313.15
MAX_DISCHARGE_K = 423.15
CO2_MW = 44.01
N2_MW = 28.014
CO2_CRITICAL_K = 304.1282
CO2_CRITICAL_PA = 7.3773e6
CO2_ACENTRIC = 0.22394
HOURS_PER_YEAR = 8000.0
TONNE_PER_KMOL_CO2 = CO2_MW / 1000.0
DEFAULT_LAYOUTS = ("SINGLE_STAGE", "TWO_STAGE_RECYCLE", "THREE_STAGE")


@dataclass(frozen=True)
class ModuleConfig:
    """Membrane transport and module configuration.

    Args:
        thickness_um: Selective-layer thickness in micrometers.
        permeance_CO2_GPU: Optional CO2 permeance; overrides permeability conversion.
        permeance_N2_GPU: Optional N2 permeance; overrides selectivity conversion.
        P_CO2_Barrer: Stage 1 CO2 permeability.
        alpha_CO2_N2: Stage 1 CO2/N2 selectivity.
        module_pressure_drop_bar: Feed-side pressure drop across one module.
        CP_modulus_CO2: Concentration-polarization factor (>= 1).
        CP_modulus_N2: Concentration-polarization factor (>= 1).
        stage_cut_max: Maximum feasible stage cut.
        flow_pattern: Cocurrent, countercurrent, or crossflow.
        segments: Finite-area integration resolution.
    """

    thickness_um: float = 0.5
    permeance_CO2_GPU: float | None = None
    permeance_N2_GPU: float | None = None
    P_CO2_Barrer: float | None = None
    alpha_CO2_N2: float | None = None
    module_pressure_drop_bar: float = 0.1
    CP_modulus_CO2: float = 1.0
    CP_modulus_N2: float = 1.0
    stage_cut_max: float = 0.85
    flow_pattern: str = "countercurrent"
    segments: int = 100

    def __post_init__(self) -> None:
        """Validate membrane material and numerical settings."""
        if self.thickness_um <= 0 or not math.isfinite(self.thickness_um):
            raise ValueError("thickness_um must be finite and positive")
        if self.permeance_CO2_GPU is None and (self.P_CO2_Barrer is None or self.P_CO2_Barrer <= 0):
            raise ValueError("Provide positive permeance_CO2_GPU or P_CO2_Barrer")
        if self.permeance_N2_GPU is None and (
            self.alpha_CO2_N2 is None or self.alpha_CO2_N2 <= 0
        ):
            raise ValueError("Provide positive permeance_N2_GPU or alpha_CO2_N2")
        if self.permeance_CO2_GPU is not None and self.permeance_CO2_GPU <= 0:
            raise ValueError("permeance_CO2_GPU must be positive")
        if self.permeance_N2_GPU is not None and self.permeance_N2_GPU <= 0:
            raise ValueError("permeance_N2_GPU must be positive")
        if self.P_CO2_Barrer is not None and self.P_CO2_Barrer <= 0:
            raise ValueError("P_CO2_Barrer must be positive")
        if self.alpha_CO2_N2 is not None and self.alpha_CO2_N2 <= 0:
            raise ValueError("alpha_CO2_N2 must be positive")
        if self.module_pressure_drop_bar < 0:
            raise ValueError("module_pressure_drop_bar cannot be negative")
        if self.CP_modulus_CO2 < 1 or self.CP_modulus_N2 < 1:
            raise ValueError("concentration-polarization moduli must be at least 1")
        if not 0 < self.stage_cut_max < 1:
            raise ValueError("stage_cut_max must be between 0 and 1")
        if self.flow_pattern not in {"cocurrent", "countercurrent", "crossflow"}:
            raise ValueError("flow_pattern must be cocurrent, countercurrent, or crossflow")
        if self.segments < 8:
            raise ValueError("segments must be at least 8")

    @property
    def co2_gpu(self) -> float:
        """Return the effective CO2 permeance in GPU."""
        if self.permeance_CO2_GPU is not None:
            return self.permeance_CO2_GPU / self.CP_modulus_CO2
        return (self.P_CO2_Barrer / self.thickness_um) / self.CP_modulus_CO2

    @property
    def n2_gpu(self) -> float:
        """Return the effective N2 permeance in GPU."""
        if self.permeance_N2_GPU is not None:
            return self.permeance_N2_GPU / self.CP_modulus_N2
        return self.co2_gpu * self.CP_modulus_CO2 / self.alpha_CO2_N2 / self.CP_modulus_N2


@dataclass(frozen=True)
class StageConfig:
    """Operating conditions for one membrane stage."""

    P_feed_bar: float = 1.0
    P_perm_bar: float = 0.2
    target_stage_cut: float = 0.3
    area_m2: float | None = None

    def __post_init__(self) -> None:
        """Validate stage pressures, area, and stage-cut target."""
        if self.P_feed_bar <= 0 or self.P_perm_bar <= 0:
            raise ValueError("stage pressures must be positive")
        if self.P_perm_bar >= self.P_feed_bar:
            raise ValueError("P_perm_bar must be below P_feed_bar")
        if not 0 < self.target_stage_cut < 1:
            raise ValueError("target_stage_cut must be between 0 and 1")
        if self.area_m2 is not None and self.area_m2 <= 0:
            raise ValueError("area_m2 must be positive when supplied")


@dataclass(frozen=True)
class CascadeConfig:
    """Feed and stage arrangement for a membrane cascade."""

    layout: str = "SINGLE_STAGE"
    stages: tuple[StageConfig, ...] | list[StageConfig] = field(
        default_factory=lambda: (StageConfig(),)
    )
    feed_flow_kmol_h: float = 1000.0
    feed_CO2_molfrac: float = 0.1
    recycle_tolerance: float = 1e-6
    max_iterations: int = 100
    pipeline_pressure_bar: float = 110.0
    electricity_price: float = 0.07
    annual_operating_hours: float = HOURS_PER_YEAR

    def __post_init__(self) -> None:
        """Validate cascade layout and feed specification."""
        if self.layout not in DEFAULT_LAYOUTS:
            raise ValueError(f"layout must be one of {', '.join(DEFAULT_LAYOUTS)}")
        if not self.stages:
            raise ValueError("at least one StageConfig is required")
        if self.feed_flow_kmol_h <= 0:
            raise ValueError("feed_flow_kmol_h must be positive")
        if not 0 <= self.feed_CO2_molfrac <= 1:
            raise ValueError("feed_CO2_molfrac must be between 0 and 1")
        if self.recycle_tolerance <= 0 or self.max_iterations < 1:
            raise ValueError("recycle_tolerance and max_iterations must be positive")
        if self.pipeline_pressure_bar <= 0 or self.electricity_price < 0:
            raise ValueError("pipeline pressure must be positive and electricity price nonnegative")
        required = {"SINGLE_STAGE": 1, "TWO_STAGE_RECYCLE": 2, "THREE_STAGE": 3}[self.layout]
        if len(self.stages) != required:
            raise ValueError(f"{self.layout} requires exactly {required} stage configurations")

    @classmethod
    def single_stage(cls, **kwargs: Any) -> "CascadeConfig":
        """Create a one-stage cascade with default operating conditions."""
        return cls(layout="SINGLE_STAGE", stages=(StageConfig(),), **kwargs)

    @classmethod
    def two_stage_recycle(cls, **kwargs: Any) -> "CascadeConfig":
        """Create a two-stage arrangement recycling second-stage permeate."""
        return cls(
            layout="TWO_STAGE_RECYCLE",
            stages=(StageConfig(), StageConfig(P_feed_bar=1.0, P_perm_bar=0.05)),
            **kwargs,
        )

    @classmethod
    def three_stage(cls, **kwargs: Any) -> "CascadeConfig":
        """Create a three-stage series cascade."""
        return cls(
            layout="THREE_STAGE",
            stages=(StageConfig(), StageConfig(), StageConfig()),
            **kwargs,
        )


@dataclass(frozen=True)
class PretreatmentUnit:
    """Configurable screening allowance for flue-gas pretreatment."""

    cost_per_tonne_co2: float = 10.0
    energy_fraction_of_plant_output: float = 0.0075
    description: str = "FGD, SCR, dehydration, and particulate removal allowance"

    def __post_init__(self) -> None:
        """Validate screening values against documented ranges."""
        if not 5 <= self.cost_per_tonne_co2 <= 15:
            raise ValueError("pretreatment cost must be between $5 and $15 per tonne CO2")
        if not 0.005 <= self.energy_fraction_of_plant_output <= 0.01:
            raise ValueError("pretreatment energy fraction must be between 0.005 and 0.01")


class CompressionTrain:
    """Intercooled ideal-gas compressor and Peng-Robinson CO2 screening model."""

    def __init__(
        self,
        compressor_efficiency: float = 0.75,
        vacuum_efficiency: float = 0.65,
        intercool_temperature_k: float = T_INTERCOOL_K,
        max_discharge_temperature_k: float = MAX_DISCHARGE_K,
        heat_capacity_ratio: float = 1.3,
    ) -> None:
        """Initialize efficiencies and thermal limits."""
        if not 0 < compressor_efficiency <= 1 or not 0 < vacuum_efficiency <= 1:
            raise ValueError("compression efficiencies must be in (0, 1]")
        if intercool_temperature_k <= 0 or max_discharge_temperature_k <= intercool_temperature_k:
            raise ValueError("invalid compressor temperature limits")
        self.compressor_efficiency = compressor_efficiency
        self.vacuum_efficiency = vacuum_efficiency
        self.intercool_temperature_k = intercool_temperature_k
        self.max_discharge_temperature_k = max_discharge_temperature_k
        self.heat_capacity_ratio = heat_capacity_ratio

    @staticmethod
    def co2_compressibility(pressure_pa: float, temperature_k: float) -> float:
        """Calculate pure CO2 Z using the vapor root of Peng-Robinson EOS."""
        if pressure_pa <= 0 or temperature_k <= 0:
            raise ValueError("pressure and temperature must be positive")
        a = 0.45724 * R_GAS**2 * CO2_CRITICAL_K**2 / CO2_CRITICAL_PA
        b = 0.07780 * R_GAS * CO2_CRITICAL_K / CO2_CRITICAL_PA
        kappa = 0.37464 + 1.54226 * CO2_ACENTRIC - 0.26992 * CO2_ACENTRIC**2
        alpha = (1 + kappa * (1 - math.sqrt(temperature_k / CO2_CRITICAL_K))) ** 2
        capital_a = a * alpha * pressure_pa / (R_GAS**2 * temperature_k**2)
        capital_b = b * pressure_pa / (R_GAS * temperature_k)
        roots = np.roots([
            1.0,
            -(1.0 - capital_b),
            capital_a - 3.0 * capital_b**2 - 2.0 * capital_b,
            -(capital_a * capital_b - capital_b**2 - capital_b**3),
        ])
        real_roots = [float(root.real) for root in roots if abs(root.imag) < 1e-8]
        if not real_roots:
            raise ValueError("Peng-Robinson EOS returned no real compressibility root")
        return max(real_roots)

    def _compress(self, flow_kmol_h: float, molecular_weight: float, inlet_bar: float,
                  outlet_bar: float, efficiency: float, z_factor: float = 1.0) -> dict[str, float]:
        if flow_kmol_h <= 0 or outlet_bar <= inlet_bar:
            return {"power_kW": 0.0, "cooling_kW": 0.0, "discharge_temperature_K": T_AMBIENT_K, "stages": 0}
        exponent = (self.heat_capacity_ratio - 1) / self.heat_capacity_ratio
        ratio = outlet_bar / inlet_bar
        stages = 1
        while True:
            stage_ratio = ratio ** (1 / stages)
            discharge_k = T_AMBIENT_K * (
                1 + (stage_ratio**exponent - 1) / efficiency
            )
            if discharge_k <= self.max_discharge_temperature_k or stages >= 12:
                break
            stages += 1
        flow_mol_s = flow_kmol_h * 1000 / 3600
        molar_mass_kg_mol = molecular_weight / 1000
        work_j_mol = (
            z_factor * R_GAS * T_AMBIENT_K / molar_mass_kg_mol
            * self.heat_capacity_ratio / (self.heat_capacity_ratio - 1)
            * stages * (stage_ratio**exponent - 1) / efficiency
        )
        mass_kg_s = flow_mol_s * molar_mass_kg_mol
        power_kw = mass_kg_s * work_j_mol / 1000
        cp_j_kg_k = 1100.0
        cooling_kw = mass_kg_s * cp_j_kg_k * max(0.0, discharge_k - self.intercool_temperature_k) / 1000 * stages
        return {
            "power_kW": power_kw,
            "cooling_kW": cooling_kw,
            "discharge_temperature_K": discharge_k,
            "stages": stages,
        }

    def feed_compression(self, flow_kmol_h: float, co2_fraction: float,
                         feed_pressure_bar: float, ambient_bar: float = 1.01325) -> dict[str, float]:
        """Compress flue gas from ambient to feed pressure."""
        mw = co2_fraction * CO2_MW + (1 - co2_fraction) * N2_MW
        return self._compress(flow_kmol_h, mw, ambient_bar, feed_pressure_bar, self.compressor_efficiency)

    def vacuum_pumping(self, flow_kmol_h: float, co2_fraction: float,
                       permeate_pressure_bar: float, ambient_bar: float = 1.01325) -> dict[str, float]:
        """Pump permeate gas from module pressure to ambient."""
        mw = co2_fraction * CO2_MW + (1 - co2_fraction) * N2_MW
        return self._compress(flow_kmol_h, mw, permeate_pressure_bar, ambient_bar, self.vacuum_efficiency)

    def product_compression(self, co2_flow_kmol_h: float, inlet_bar: float,
                            pipeline_bar: float = 110.0) -> dict[str, float]:
        """Compress captured CO2 to pipeline pressure with a PR Z correction."""
        if co2_flow_kmol_h <= 0:
            return {"power_kW": 0.0, "cooling_kW": 0.0, "discharge_temperature_K": T_AMBIENT_K, "stages": 0}
        average_pressure_pa = math.sqrt(inlet_bar * pipeline_bar) * BAR_TO_PA
        z_factor = self.co2_compressibility(average_pressure_pa, T_AMBIENT_K)
        return self._compress(
            co2_flow_kmol_h, CO2_MW, inlet_bar, pipeline_bar,
            self.compressor_efficiency, z_factor=z_factor,
        )


@dataclass
class TEAReport:
    """Annualized screening techno-economic analysis."""

    total_capital_cost_usd: float
    annual_fixed_opex_usd: float
    annual_variable_opex_usd: float
    levelized_cost_usd_per_tonne: float
    cost_breakdown: dict[str, float]
    sensitivity: dict[float, float]

    def summary(self) -> str:
        """Format a concise TEA summary."""
        return (
            f"Total capital: ${self.total_capital_cost_usd / 1e6:,.2f}M; "
            f"fixed OPEX: ${self.annual_fixed_opex_usd / 1e6:,.2f}M/yr; "
            f"variable OPEX: ${self.annual_variable_opex_usd / 1e6:,.2f}M/yr; "
            f"levelized capture cost: ${self.levelized_cost_usd_per_tonne:,.2f}/t CO2"
        )


@dataclass
class CascadeResult:
    """Cascade material and energy balance."""

    purity: float
    recovery: float
    stage_cut: float
    total_area_m2: float
    total_compression_kW: float
    specific_energy_kWh_per_t: float
    per_stage: list[dict[str, Any]]
    converged: bool
    n_iterations: int
    fresh_feed_kmol_h: float
    fresh_feed_CO2_molfrac: float
    captured_CO2_kmol_h: float
    retentate_kmol_h: float
    energy_details: dict[str, float]
    module: ModuleConfig
    cascade: CascadeConfig
    pretreatment: PretreatmentUnit

    def summary(self) -> str:
        """Return key cascade performance indicators."""
        return (
            f"Purity {self.purity * 100:.2f}%; recovery {self.recovery * 100:.2f}%; "
            f"stage cut {self.stage_cut * 100:.2f}%; area {self.total_area_m2:,.1f} m2; "
            f"power {self.total_compression_kW:,.1f} kW; "
            f"SEC {self.specific_energy_kWh_per_t:,.1f} kWh/t"
        )

    def compute_tea(self, electricity_price: float = 0.07) -> TEAReport:
        """Calculate configurable screening economics.

        Args:
            electricity_price: Electricity price in USD/kWh.

        Returns:
            Annualized capital, operating cost, and price sensitivity report.
        """
        if electricity_price < 0:
            raise ValueError("electricity_price cannot be negative")
        annual_capture_t = self.captured_CO2_kmol_h * TONNE_PER_KMOL_CO2 * self.cascade.annual_operating_hours
        if annual_capture_t <= 0:
            raise ValueError("TEA requires a positive captured CO2 rate")
        membrane_capital = self.total_area_m2 * 100.0
        power_equipment = self.total_compression_kW * 1200.0
        pretreatment_capital = annual_capture_t * self.pretreatment.cost_per_tonne_co2 * 0.15
        equipment = membrane_capital + power_equipment + pretreatment_capital
        balance_of_plant = equipment * 0.25
        overnight = (equipment + balance_of_plant) * 1.20
        crf = 0.08 * (1.08**20) / ((1.08**20) - 1)
        replacement = membrane_capital / 5.0
        maintenance = overnight * 0.03
        labor = 2 * 45.0 * 2080.0
        fixed_opex = replacement + maintenance + labor
        pretreatment_variable = annual_capture_t * self.pretreatment.cost_per_tonne_co2
        energy_kwh_year = self.total_compression_kW * self.cascade.annual_operating_hours
        cooling_cost = self.energy_details.get("cooling_kW", 0.0) * self.cascade.annual_operating_hours * 0.001

        def cost_at(price: float) -> float:
            variable = energy_kwh_year * price + pretreatment_variable + cooling_cost
            return (overnight * crf + fixed_opex + variable) / annual_capture_t

        breakdown = {
            "annualized capital": overnight * crf,
            "fixed OPEX": fixed_opex,
            "electricity": energy_kwh_year * electricity_price,
            "pretreatment": pretreatment_variable,
            "cooling": cooling_cost,
        }
        return TEAReport(
            total_capital_cost_usd=overnight,
            annual_fixed_opex_usd=fixed_opex,
            annual_variable_opex_usd=sum(
                (energy_kwh_year * electricity_price, pretreatment_variable, cooling_cost)
            ),
            levelized_cost_usd_per_tonne=cost_at(electricity_price),
            cost_breakdown=breakdown,
            sensitivity={price: cost_at(price) for price in (0.04, 0.07, 0.10, 0.13)},
        )


class MembraneCascade:
    """Solve membrane module(s), recycle, energy, and screening economics."""

    def __init__(
        self,
        module: ModuleConfig,
        cascade: CascadeConfig,
        pretreatment: PretreatmentUnit | None = None,
    ) -> None:
        """Create a cascade from material, process, and optional pretreatment data."""
        self.module = module
        self.cascade = cascade
        self.pretreatment = pretreatment or PretreatmentUnit()

    def _flux(self, f: np.ndarray, p: np.ndarray, area: float,
              stage: StageConfig, index: float, cross_y: np.ndarray | None = None) -> np.ndarray:
        f = np.maximum(f, 0.0)
        p = np.maximum(p, 0.0)
        f_total = max(float(np.sum(f)), 1e-14)
        p_total = max(float(np.sum(p)), 1e-14)
        x = f / f_total
        y = cross_y if cross_y is not None else p / p_total
        if np.sum(p) <= 1e-12 and cross_y is None:
            y = np.array([0.5, 0.5])
        p_feed = (stage.P_feed_bar - self.module.module_pressure_drop_bar * index) * BAR_TO_PA
        p_perm = stage.P_perm_bar * BAR_TO_PA
        permeances = np.array([self.module.co2_gpu, self.module.n2_gpu]) * GPU_TO_SI
        flux = permeances * (p_feed * x - p_perm * y)
        forward = flux > 0
        reverse = flux < 0
        flux[forward] *= np.clip(f[forward] / 1e-8, 0.0, 1.0)
        flux[reverse] *= np.clip(p[reverse] / 1e-8, 0.0, 1.0)
        return flux

    def _integrate_module(self, feed: np.ndarray, stage: StageConfig,
                          area: float) -> tuple[np.ndarray, np.ndarray]:
        if np.sum(feed) <= 0:
            return np.zeros(2), np.zeros(2)
        fixed_y = None
        if self.module.flow_pattern in {"crossflow", "countercurrent"}:
            fixed_y = np.array([0.5, 0.5])
        initial_state = np.concatenate((feed, np.zeros(2)))
        state = initial_state
        fixed_point_iterations = 4
        for _ in range(fixed_point_iterations if fixed_y is not None else 1):
            def equations(s: float, values: np.ndarray) -> np.ndarray:
                feed_flow = np.maximum(values[:2], 0.0)
                perm_flow = np.maximum(values[2:], 0.0)
                flux = self._flux(feed_flow, perm_flow, area, stage, s / area, fixed_y)
                return np.concatenate((-flux, flux)) * 3.6

            solution = solve_ivp(
                equations, (0.0, area), initial_state, method="BDF",
                t_eval=np.linspace(0.0, area, self.module.segments + 1),
                rtol=2e-6, atol=1e-9, max_step=area / self.module.segments,
            )
            if not solution.success:
                raise RuntimeError(f"module integration failed: {solution.message}")
            state = solution.y[:, -1]
            if fixed_y is None:
                break
            new_y = np.maximum(state[2:], 0.0)
            total = float(np.sum(new_y))
            if total <= 1e-12:
                break
            new_y /= total
            if np.linalg.norm(new_y - fixed_y, ord=1) < 1e-3:
                fixed_y = new_y
                break
            fixed_y = 0.5 * (fixed_y + new_y)
        return np.maximum(state[:2], 0.0), np.maximum(state[2:], 0.0)

    def _solve_stage(self, feed: np.ndarray, stage: StageConfig, index: int) -> dict[str, Any]:
        if self.module.co2_gpu == 0 or self.module.n2_gpu == 0:
            raise ValueError("zero permeance cannot be used to size a membrane stage")
        if self.module.module_pressure_drop_bar >= stage.P_feed_bar:
            raise ValueError("module pressure drop must be below feed pressure")
        feed_total = float(np.sum(feed))
        if feed_total <= 0:
            raise ValueError("stage feed must have positive flow")
        target = min(stage.target_stage_cut, self.module.stage_cut_max)

        def calculate(area: float) -> tuple[float, np.ndarray, np.ndarray]:
            retentate, permeate = self._integrate_module(feed, stage, area)
            cut = float(np.sum(permeate) / feed_total)
            return cut, retentate, permeate

        if stage.area_m2 is not None:
            area = stage.area_m2
            cut, retentate, permeate = calculate(area)
        else:
            composition = feed / feed_total
            inlet_flux = self._flux(feed, feed, 1.0, stage, 0.0, composition)
            available_flux = float(np.sum(np.maximum(inlet_flux, 0.0))) * 3.6
            if available_flux <= 1e-14:
                raise ValueError(f"stage {index + 1} has no forward permeation driving force")
            area = max(target * feed_total / available_flux, 1e-6)
            for _ in range(8):
                cut, retentate, permeate = calculate(area)
                error = target - cut
                if abs(error) <= 1e-2:
                    break
                if cut <= 1e-9:
                    area *= 2.0
                else:
                    area *= float(np.clip(target / cut, 0.5, 2.0))
            else:
                raise RuntimeError(f"stage {index + 1} could not size area to target stage cut")
        if np.any(~np.isfinite(retentate)) or np.any(~np.isfinite(permeate)):
            raise RuntimeError(f"stage {index + 1} returned a non-finite stream")
        if np.any(retentate < -1e-7) or np.any(permeate < -1e-7):
            raise RuntimeError(f"stage {index + 1} produced negative component flow")
        r_total = max(float(np.sum(retentate)), 0.0)
        p_total = max(float(np.sum(permeate)), 0.0)
        return {
            "stage": index + 1,
            "feed_flow_kmol_h": feed_total,
            "feed_composition": (feed / feed_total).tolist(),
            "permeate_flow_kmol_h": p_total,
            "permeate_composition": (permeate / p_total).tolist() if p_total else [0.0, 0.0],
            "retentate_flow_kmol_h": r_total,
            "retentate_composition": (retentate / r_total).tolist() if r_total else [0.0, 0.0],
            "area_m2": area,
            "P_feed_bar": stage.P_feed_bar,
            "P_perm_bar": stage.P_perm_bar,
            "stage_cut": cut,
            "retentate_components": retentate,
            "permeate_components": permeate,
        }

    def solve(self) -> CascadeResult:
        """Solve all membrane stages and the specified recycle arrangement.

        Returns:
            A material/energy-balance result with screening TEA support.

        Raises:
            RuntimeError: When the recycle loop or module solver fails.
            ValueError: When the configuration cannot produce captured CO2.
        """
        fresh = self.cascade.feed_flow_kmol_h * np.array([
            self.cascade.feed_CO2_molfrac, 1 - self.cascade.feed_CO2_molfrac
        ])
        recycle = np.zeros(2)
        per_stage: list[dict[str, Any]] = []
        iterations = 1
        for iterations in range(1, self.cascade.max_iterations + 1):
            stage_feed = fresh + recycle
            first = self._solve_stage(stage_feed, self.cascade.stages[0], 0)
            per_stage = [first]
            if len(self.cascade.stages) == 1:
                product = first["permeate_components"]
                retentate = first["retentate_components"]
                break
            current_feed = first["retentate_components"]
            second = self._solve_stage(current_feed, self.cascade.stages[1], 1)
            per_stage.append(second)
            if self.cascade.layout == "TWO_STAGE_RECYCLE":
                new_recycle = second["permeate_components"]
                if np.linalg.norm(new_recycle - recycle, ord=1) <= self.cascade.recycle_tolerance * max(
                    float(np.sum(new_recycle)), 1.0
                ):
                    recycle = new_recycle
                    product = first["permeate_components"]
                    retentate = second["retentate_components"]
                    break
                recycle = new_recycle
                continue
            third = self._solve_stage(second["retentate_components"], self.cascade.stages[2], 2)
            per_stage.append(third)
            product = (
                first["permeate_components"]
                + second["permeate_components"]
                + third["permeate_components"]
            )
            retentate = third["retentate_components"]
            break
        else:
            raise RuntimeError(
                f"recycle did not converge in {self.cascade.max_iterations} iterations"
            )
        if self.cascade.layout == "TWO_STAGE_RECYCLE":
            LOGGER.info("Recycle converged after %d iterations", iterations)
        product_flow = float(np.sum(product))
        captured = float(product[0])
        purity = captured / product_flow if product_flow > 0 else 0.0
        feed_co2 = float(fresh[0])
        recovery = captured / feed_co2 if feed_co2 > 0 else 0.0
        stage_cut = product_flow / self.cascade.feed_flow_kmol_h
        retentate_flow = float(np.sum(retentate))
        closure = abs(self.cascade.feed_flow_kmol_h - product_flow - retentate_flow)
        if closure > 1e-6 * max(self.cascade.feed_flow_kmol_h, 1.0):
            raise RuntimeError(f"cascade material balance failed to close (error {closure:g} kmol/h)")

        compression = CompressionTrain()
        feed_work = compression.feed_compression(
            self.cascade.feed_flow_kmol_h, self.cascade.feed_CO2_molfrac,
            self.cascade.stages[0].P_feed_bar,
        )
        permeate_composition = purity
        vacuum = compression.vacuum_pumping(
            product_flow, permeate_composition, self.cascade.stages[0].P_perm_bar,
        )
        product_work = compression.product_compression(
            captured, self.cascade.stages[0].P_perm_bar, self.cascade.pipeline_pressure_bar,
        )
        total_power = feed_work["power_kW"] + vacuum["power_kW"] + product_work["power_kW"]
        tonnes_h = captured * TONNE_PER_KMOL_CO2
        sec = total_power / tonnes_h if tonnes_h > 0 else 0.0
        energy_details = {
            "feed_compression_kW": feed_work["power_kW"],
            "vacuum_kW": vacuum["power_kW"],
            "product_compression_kW": product_work["power_kW"],
            "cooling_kW": feed_work["cooling_kW"] + vacuum["cooling_kW"] + product_work["cooling_kW"],
            "captured_flow_kmol_h": captured,
        }
        return CascadeResult(
            purity=purity,
            recovery=recovery,
            stage_cut=stage_cut,
            total_area_m2=sum(item["area_m2"] for item in per_stage),
            total_compression_kW=total_power,
            specific_energy_kWh_per_t=sec,
            per_stage=per_stage,
            converged=True,
            n_iterations=iterations,
            fresh_feed_kmol_h=self.cascade.feed_flow_kmol_h,
            fresh_feed_CO2_molfrac=self.cascade.feed_CO2_molfrac,
            captured_CO2_kmol_h=captured,
            retentate_kmol_h=retentate_flow,
            energy_details=energy_details,
            module=self.module,
            cascade=self.cascade,
            pretreatment=self.pretreatment,
        )
