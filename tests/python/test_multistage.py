import math
import unittest

import numpy as np

from multistage_model import (
    CascadeConfig,
    CompressionTrain,
    MembraneCascade,
    ModuleConfig,
    StageConfig,
)


class MultiStageModelTests(unittest.TestCase):
    def module(self, pattern="cocurrent", alpha=50, segments=12):
        return ModuleConfig(
            P_CO2_Barrer=2200,
            alpha_CO2_N2=alpha,
            flow_pattern=pattern,
            segments=segments,
        )

    def stage(self, **kwargs):
        values = {
            "P_feed_bar": 2.0,
            "P_perm_bar": 0.1,
            "target_stage_cut": 0.2,
        }
        values.update(kwargs)
        return StageConfig(**values)

    def test_module_mass_balance_closes(self):
        model = MembraneCascade(
            self.module("cocurrent"),
            CascadeConfig(
                stages=(self.stage(area_m2=40),),
                feed_flow_kmol_h=10,
                feed_CO2_molfrac=0.1,
            ),
        )
        result = model.solve()
        stage = result.per_stage[0]
        self.assertAlmostEqual(
            stage["feed_flow_kmol_h"],
            stage["permeate_flow_kmol_h"] + stage["retentate_flow_kmol_h"],
            places=6,
        )

    def test_module_no_driving_force_returns_zero(self):
        module = ModuleConfig(
            P_CO2_Barrer=1000,
            alpha_CO2_N2=1,
            flow_pattern="crossflow",
            segments=8,
        )
        model = MembraneCascade(
            module,
            CascadeConfig(stages=(self.stage(P_feed_bar=1, P_perm_bar=0.2, area_m2=1),)),
        )
        flux = model._flux(
            np.array([1.0, 9.0]),
            np.array([0.5, 0.5]),
            1.0,
            model.cascade.stages[0],
            0.0,
            np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(float(flux[0]), 0.0, places=10)
        self.assertGreater(float(flux[1]), 0.0)

    def test_module_flow_patterns_size_to_stage_cut(self):
        for pattern in ("cocurrent", "crossflow", "countercurrent"):
            model = MembraneCascade(self.module(pattern), CascadeConfig(stages=(self.stage(),)))
            result = model._solve_stage(np.array([1.0, 9.0]), self.stage(), 0)
            self.assertGreater(result["area_m2"], 0)
            self.assertAlmostEqual(result["stage_cut"], 0.2, delta=0.01)

    def test_cascade_single_stage_recovers_expected(self):
        config = CascadeConfig(
            layout="SINGLE_STAGE",
            stages=(self.stage(area_m2=40),),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
        )
        result = MembraneCascade(self.module(), config).solve()
        self.assertTrue(result.converged)
        self.assertGreater(result.recovery, 0)
        self.assertGreater(result.purity, config.feed_CO2_molfrac)

    def test_cascade_two_stage_achieves_higher_purity(self):
        module = self.module("crossflow")
        one = CascadeConfig(
            stages=(self.stage(area_m2=40),),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
        )
        two = CascadeConfig(
            layout="TWO_STAGE_RECYCLE",
            stages=(
                self.stage(area_m2=40),
                self.stage(P_perm_bar=0.03, target_stage_cut=0.2, area_m2=20),
            ),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
        )
        self.assertGreaterEqual(
            MembraneCascade(module, two).solve().purity,
            MembraneCascade(module, one).solve().purity,
        )

    def test_three_stage_series_closes_with_all_permeates_in_product(self):
        feed_flow = 1000
        config = CascadeConfig(
            layout="THREE_STAGE",
            stages=(
                self.stage(P_feed_bar=6, P_perm_bar=0.1, target_stage_cut=0.3),
                self.stage(P_feed_bar=6, P_perm_bar=0.1, target_stage_cut=0.3),
                self.stage(P_feed_bar=6, P_perm_bar=0.1, target_stage_cut=0.3),
            ),
            feed_flow_kmol_h=feed_flow,
            feed_CO2_molfrac=0.1,
        )
        result = MembraneCascade(
            ModuleConfig(
                thickness_um=0.5,
                P_CO2_Barrer=4836.8,
                alpha_CO2_N2=8.77,
                flow_pattern="countercurrent",
                segments=8,
            ),
            config,
        ).solve()

        permeate_total = sum(
            stage["permeate_flow_kmol_h"] for stage in result.per_stage
        )
        self.assertAlmostEqual(
            feed_flow,
            permeate_total + result.retentate_kmol_h,
            places=6,
        )
        self.assertAlmostEqual(result.stage_cut, permeate_total / feed_flow, places=6)
        self.assertGreater(result.recovery, 0)

    def test_cascade_recycle_converges(self):
        config = CascadeConfig(
            layout="TWO_STAGE_RECYCLE",
            stages=(
                self.stage(area_m2=30),
                self.stage(P_perm_bar=0.03, target_stage_cut=0.15, area_m2=10),
            ),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
        )
        result = MembraneCascade(self.module("crossflow"), config).solve()
        self.assertTrue(result.converged)
        self.assertLess(result.n_iterations, 100)

    def test_cascade_recycle_fails_gracefully_if_no_convergence(self):
        config = CascadeConfig(
            layout="TWO_STAGE_RECYCLE",
            stages=(self.stage(area_m2=30), self.stage(area_m2=10)),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
            recycle_tolerance=1e-14,
            max_iterations=1,
        )
        with self.assertRaisesRegex(RuntimeError, "recycle did not converge"):
            MembraneCascade(self.module("crossflow"), config).solve()

    def test_compression_isentropic_work_matches_thermodynamics(self):
        train = CompressionTrain(compressor_efficiency=1.0)
        result = train._compress(10, 28.014, 1, 1.1, 1.0)
        mass_flow = 10 * 28.014 / 3600
        exponent = (1.3 - 1) / 1.3
        expected = (
            mass_flow * 8.314462618 * 298.15 / 0.028014
            * 1.3 / 0.3 * (1.1**exponent - 1) / 1000
        )
        self.assertAlmostEqual(result["power_kW"], expected, places=8)

    def test_vacuum_pump_power_reasonable(self):
        result = CompressionTrain().vacuum_pumping(10, 0.1, 0.2)
        self.assertGreater(result["power_kW"], 0)
        self.assertLess(result["power_kW"], 100)

    def test_co2_compression_to_100_bar_z_correction(self):
        train = CompressionTrain()
        z = train.co2_compressibility(100e5, 313.15)
        self.assertTrue(math.isfinite(z))
        self.assertGreater(z, 0)
        self.assertNotAlmostEqual(z, 1.0, places=3)
        result = train.product_compression(1, 1, 100)
        self.assertGreater(result["power_kW"], 0)

    def test_tea_cost_breakdown_sums_correctly(self):
        config = CascadeConfig(
            stages=(self.stage(area_m2=40),),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=0.1,
        )
        result = MembraneCascade(self.module(), config).solve()
        tea = result.compute_tea()
        crf = 0.08 * 1.08**20 / (1.08**20 - 1)
        self.assertAlmostEqual(
            sum(tea.cost_breakdown.values()),
            tea.total_capital_cost_usd * crf + tea.annual_fixed_opex_usd
            + tea.annual_variable_opex_usd,
            delta=1e-6,
        )
        self.assertEqual(set(tea.sensitivity), {0.04, 0.07, 0.10, 0.13})

    def test_edge_case_pure_CO2_feed(self):
        config = CascadeConfig(
            stages=(self.stage(area_m2=10),),
            feed_flow_kmol_h=10,
            feed_CO2_molfrac=1.0,
        )
        result = MembraneCascade(self.module("crossflow"), config).solve()
        self.assertAlmostEqual(result.purity, 1.0, places=6)
        self.assertAlmostEqual(result.recovery, result.stage_cut, places=5)

    def test_edge_case_alpha_equals_1(self):
        result = MembraneCascade(
            self.module("crossflow", alpha=1),
            CascadeConfig(
                stages=(self.stage(area_m2=1),),
                feed_flow_kmol_h=10,
                feed_CO2_molfrac=0.1,
            ),
        ).solve()
        self.assertAlmostEqual(result.purity, 0.1, delta=0.02)

    def test_edge_case_zero_permeability(self):
        with self.assertRaisesRegex(ValueError, "P_CO2_Barrer"):
            ModuleConfig(P_CO2_Barrer=0, alpha_CO2_N2=50)


if __name__ == "__main__":
    unittest.main()
