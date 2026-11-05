import unittest

from stage2_engine import (
    calculate_energy_and_cost,
    find_required_area,
    solve_module,
    solve_single_stage,
    solve_cascade,
)


class StageTwoEngineTests(unittest.TestCase):
    def test_countercurrent_module_closes_component_balances(self):
        result = solve_module(
            area=250.0,
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
        )

        self.assertLessEqual(result.recovery, 1.0)
        self.assertAlmostEqual(
            result.feed_flow_kmol_h,
            result.retentate_flow_kmol_h + result.permeate_flow_kmol_h,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            result.feed_co2_flow_kmol_h,
            result.retentate_co2_flow_kmol_h + result.permeate_co2_flow_kmol_h,
            delta=1e-6,
        )
        self.assertEqual(len(result.area_profile_m2), 101)

    def test_inverse_area_and_single_stage_hit_target(self):
        area = find_required_area(
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            target_recovery=0.9,
        )
        result = solve_single_stage(
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            target_recovery=0.9,
        )

        self.assertGreater(area, 0)
        self.assertAlmostEqual(result.area_m2, area, delta=1e-5)
        self.assertAlmostEqual(result.module.recovery, 0.9, places=6)

        lower_target = solve_single_stage(
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            target_recovery=0.8,
        )
        self.assertAlmostEqual(lower_target.module.recovery, 0.8, places=6)

    def test_default_app_conditions_are_physically_solved(self):
        result = solve_single_stage(
            P_CO2_barrer=4836.77,
            alpha=9.12,
            P_feed=1.0,
            P_perm=0.1,
            x_feed=0.1,
            F_feed=1000.0,
            target_recovery=0.9,
        )

        self.assertAlmostEqual(result.module.recovery, 0.9, places=6)
        self.assertLessEqual(result.module.recovery, 1.0)

    def test_selectivity_limits_purity_and_vacuum_pressure_increases_area(self):
        baseline = solve_single_stage(
            P_CO2_barrer=2000.0,
            alpha=12.0,
            P_feed=1.0,
            P_perm=0.1,
            x_feed=0.1,
            F_feed=1000.0,
            target_recovery=0.9,
        )
        higher_pressure_area = find_required_area(
            P_CO2_barrer=2000.0,
            alpha=12.0,
            P_feed=1.0,
            P_perm=0.2,
            x_feed=0.1,
            F_feed=1000.0,
            target_recovery=0.9,
        )
        half_permeability_area = find_required_area(
            P_CO2_barrer=1000.0,
            alpha=12.0,
            P_feed=1.0,
            P_perm=0.1,
            x_feed=0.1,
            F_feed=1000.0,
            target_recovery=0.9,
        )

        self.assertAlmostEqual(baseline.module.recovery, 0.9, places=6)
        # With 10% CO2 feed and alpha=12, the zero-permeate-pressure limit
        # for local permeate is about 57%; finite pressure and high recovery
        # lower the integrated product purity further.
        self.assertLess(baseline.module.purity_co2, 0.8)
        self.assertGreater(higher_pressure_area, baseline.area_m2)
        self.assertAlmostEqual(
            half_permeability_area,
            2.0 * baseline.area_m2,
            delta=baseline.area_m2 * 0.01,
        )

    def test_invalid_target_and_pressure_reversal_are_reported(self):
        with self.assertRaisesRegex(ValueError, "less than 99%"):
            find_required_area(1000, 30, 10, 0.05, 0.15, 100, 0.99)
        with self.assertRaisesRegex(ValueError, "pressure reversal"):
            solve_module(10, 1000, 30, 1, 2, 0.15, 100)

    def test_retentate_recycle_rejects_insufficient_purge(self):
        with self.assertRaisesRegex(ValueError, "cannot purge"):
            solve_cascade(
                P_CO2_barrer=1000.0,
                alpha=30.0,
                P_feed=10.0,
                P_perm=0.05,
                x_feed=0.15,
                F_feed=100.0,
                target_recovery=0.5,
                stage2_area_ratio=0.5,
            )

    def test_retentate_recycle_closes_when_all_components_can_be_purged(self):
        result = solve_cascade(
            P_CO2_barrer=1000.0,
            alpha=1.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            target_recovery=0.5,
            stage2_area_ratio=1.0,
        )

        self.assertAlmostEqual(result.overall_recovery, 0.5, places=6)
        self.assertLessEqual(result.overall_recovery, 1.0)
        self.assertGreaterEqual(result.recycle_flow_kmol_h, 0.0)
        self.assertLessEqual(result.iterations, 100)

    def test_energy_and_cost_postprocessing(self):
        result = calculate_energy_and_cost(
            feed_flow_kmol_h=100.0,
            feed_pressure_bar=6.0,
            permeate_pressure_bar=0.1,
            permeate_flow_kmol_h=20.0,
            captured_co2_kmol_h=10.0,
            total_membrane_area_m2=500.0,
        )

        self.assertGreater(result.feed_compression_kw, 0)
        self.assertGreater(result.vacuum_pumping_kw, 0)
        self.assertAlmostEqual(
            result.total_power_kw,
            result.feed_compression_kw + result.vacuum_pumping_kw,
        )
        self.assertGreater(result.specific_energy_kwh_per_tonne_co2, 0)
        self.assertGreater(result.levelized_capture_cost_usd_per_tonne, 0)
        gamma = 1.4
        stages = 3
        temperature = 298.15
        efficiency = 0.65
        pressure_ratio_per_stage = (1.0 / 0.1) ** (1.0 / stages)
        expected_vacuum_kw = (
            20.0
            * 1000.0
            * stages
            * gamma
            / (gamma - 1.0)
            * 8.314462618
            * temperature
            * (
                pressure_ratio_per_stage ** ((gamma - 1.0) / gamma)
                - 1.0
            )
            / efficiency
            / 3.6e6
        )
        self.assertAlmostEqual(result.vacuum_pumping_kw, expected_vacuum_kw)
        self.assertAlmostEqual(
            result.specific_energy_kwh_per_tonne_co2,
            result.total_power_kw / (10.0 * 0.04401),
        )


if __name__ == "__main__":
    unittest.main()
