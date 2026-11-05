import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class FastAPISimulationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.payload = {
            "P_CO2": 2000.0,
            "alpha": 12.0,
            "thickness": 1.0,
            "P_feed": 1.0,
            "P_perm": 0.1,
            "F_feed": 1000.0,
            "x_feed": 0.1,
            "mode": "simulation",
            "target_value": 2500.0,
        }

    def tearDown(self):
        self.client.close()

    def test_api_root_has_useful_landing_response(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["simulate"], "POST /api/simulate")

    def test_simulation_mode_returns_calculated_performance_and_graphs(self):
        with patch.object(
            main,
            "_build_graphs",
            return_value=main.GraphData(
                area_sweep=[
                    main.AreaSweepPoint(area=250.0, recovery=0.1, purity=0.4)
                ],
                pressure_sweep=[
                    main.PressureSweepPoint(pressure=0.1, area=2500.0)
                ],
            ),
        ):
            response = self.client.post("/api/simulate", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["area"], self.payload["target_value"])
        self.assertGreaterEqual(result["recovery"], 0.0)
        self.assertLessEqual(result["recovery"], 1.0)
        self.assertGreater(result["purity"], 0.0)
        self.assertGreater(result["product_co2_flow"], 0.0)
        self.assertGreater(result["product_flow"], result["product_co2_flow"])
        self.assertGreater(result["specific_energy"], 0.0)
        self.assertGreater(result["cost"], 0.0)
        self.assertEqual(result["graphs"]["area_sweep"][0]["area"], 250.0)
        self.assertEqual(result["graphs"]["pressure_sweep"][0]["pressure"], 0.1)

    def test_design_mode_rejects_invalid_recovery_and_pressure_reversal(self):
        invalid_recovery = {**self.payload, "mode": "design", "target_value": 0.99}
        response = self.client.post("/api/simulate", json=invalid_recovery)
        self.assertEqual(response.status_code, 422)

        reversed_pressure = {**self.payload, "P_perm": 1.0}
        response = self.client.post("/api/simulate", json=reversed_pressure)
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
