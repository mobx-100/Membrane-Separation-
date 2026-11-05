import unittest

from rigorous_simulation import find_required_area, solve_cascade, solve_module


class RigorousSimulationTests(unittest.TestCase):
    def module(self, area, flow_pattern="cocurrent", **kwargs):
        values = {
            "area": area,
            "P_CO2_barrer": 1000.0,
            "alpha": 30.0,
            "P_feed": 10.0,
            "P_perm": 0.05,
            "x_feed": 0.15,
            "F_feed": 100.0,
            "flow_pattern": flow_pattern,
        }
        values.update(kwargs)
        return solve_module(**values)

    def test_recovery_limit(self):
        result = self.module(
            area=1e6,
            P_perm=0.0,
            flow_pattern="cocurrent",
        )
        self.assertGreater(result.recovery, 0.999)
        self.assertLessEqual(result.recovery, 1.0)

    def test_mass_balance(self):
        result = self.module(area=250.0, flow_pattern="cocurrent")
        feed_total = result.feed_flow
        permeate_total = result.permeate_flow
        retentate_total = result.retentate_flow
        total_relative_error = abs(
            feed_total - retentate_total - permeate_total
        ) / feed_total
        co2_relative_error = abs(
            feed_total * result.feed_x
            - retentate_total * result.retentate_x
            - result.permeate_co2_flow
        ) / (feed_total * result.feed_x)
        self.assertLessEqual(total_relative_error, 1e-6)
        self.assertLessEqual(co2_relative_error, 1e-6)
        self.assertAlmostEqual(
            feed_total * (1 - result.feed_x),
            result.retentate_n2_flow + result.permeate_n2_flow,
            delta=1e-6 * feed_total,
        )

    def test_flow_pattern_difference(self):
        cocurrent = self.module(area=250.0, flow_pattern="cocurrent")
        countercurrent = self.module(area=250.0, flow_pattern="countercurrent")
        self.assertGreater(countercurrent.purity_y, cocurrent.purity_y)
        self.assertGreater(countercurrent.recovery, cocurrent.recovery)

    def test_inverse_area_hits_requested_recovery(self):
        area = find_required_area(
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            flow_pattern="countercurrent",
            target_recovery=0.9,
        )
        result = self.module(area=area, flow_pattern="countercurrent")
        self.assertAlmostEqual(result.recovery, 0.9, places=6)

    def test_recovery_target_must_be_below_99_percent(self):
        with self.assertRaisesRegex(ValueError, "less than 0.99"):
            find_required_area(
                P_CO2_barrer=1000.0,
                alpha=30.0,
                P_feed=10.0,
                P_perm=0.05,
                x_feed=0.15,
                F_feed=100.0,
                flow_pattern="cocurrent",
                target_recovery=0.99,
            )

    def test_pressure_reversal_is_reported(self):
        with self.assertRaisesRegex(ValueError, "pressure reversal"):
            solve_module(
                area=10.0,
                P_CO2_barrer=1000.0,
                alpha=30.0,
                P_feed=1.0,
                P_perm=2.0,
                x_feed=0.15,
                F_feed=100.0,
                flow_pattern="cocurrent",
            )

    def test_two_stage_recycle_closes_external_balance(self):
        result = solve_cascade(
            P_CO2_barrer=1000.0,
            alpha=30.0,
            P_feed=10.0,
            P_perm=0.05,
            x_feed=0.15,
            F_feed=100.0,
            flow_pattern="cocurrent",
            area_stage1=200.0,
            area_stage2=80.0,
        )
        self.assertGreaterEqual(result.overall_recovery, 0.0)
        self.assertLessEqual(result.overall_recovery, 1.0)
        self.assertGreaterEqual(result.recycle_flow, 0.0)
        self.assertAlmostEqual(
            result.stage1.feed_flow,
            result.stage1.permeate_flow + result.stage1.retentate_flow,
            delta=1e-6 * result.stage1.feed_flow,
        )
        self.assertAlmostEqual(
            result.stage2.feed_flow,
            result.stage2.permeate_flow + result.stage2.retentate_flow,
            delta=1e-6 * result.stage2.feed_flow,
        )
        self.assertAlmostEqual(
            100.0,
            result.product_flow + result.final_retentate_flow,
            delta=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
