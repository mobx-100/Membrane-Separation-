import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.api import Handler


class RigorousProcessApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/api/process"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def request_process(self, payload):
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    @staticmethod
    def payload(layout):
        design = {
            "layout": layout,
            "flow_pattern": "cocurrent",
            "feed_flow_kmol_h": 100.0,
            "feed_co2_fraction": 0.15,
            "feed_pressure_bar": 10.0,
            "permeate_pressure_bar": 0.05,
            "thickness_um": 1.0,
            "area_stage1_m2": 200.0,
        }
        if layout == "TWO_STAGE_RECYCLE":
            design["area_stage2_m2"] = 80.0
        return {
            "solver": "rigorous",
            "permeability_barrer": 1000.0,
            "selectivity": 30.0,
            "design": design,
        }

    def test_two_stage_request_uses_rigorous_recycle_solver(self):
        status, result = self.request_process(self.payload("TWO_STAGE_RECYCLE"))

        self.assertEqual(status, 200, result)
        self.assertEqual(result["solver"], "rigorous_1d")
        self.assertEqual(len(result["per_stage"]), 2)
        self.assertGreaterEqual(result["recovery"], 0.0)
        self.assertLessEqual(result["recovery"], 1.0)
        self.assertAlmostEqual(result["total_area_m2"], 280.0)
        self.assertAlmostEqual(
            100.0,
            result["product_flow_kmol_h"] + result["per_stage"][1]["retentate_flow_kmol_h"],
            delta=1e-5,
        )

    def test_single_stage_request_uses_rigorous_module_solver(self):
        status, result = self.request_process(self.payload("SINGLE_STAGE"))

        self.assertEqual(status, 200, result)
        self.assertEqual(result["solver"], "rigorous_1d")
        self.assertEqual(len(result["per_stage"]), 1)
        self.assertAlmostEqual(result["per_stage"][0]["area_m2"], 200.0)

    def test_default_material_prediction_and_countercurrent_case(self):
        payload = self.payload("TWO_STAGE_RECYCLE")
        payload["permeability_barrer"] = 4836.77
        payload["selectivity"] = 8.77
        payload["design"].update({
            "flow_pattern": "countercurrent",
            "feed_flow_kmol_h": 1000.0,
            "feed_co2_fraction": 0.1,
            "feed_pressure_bar": 6.0,
            "permeate_pressure_bar": 0.1,
            "thickness_um": 0.5,
            "area_stage1_m2": 100.0,
            "area_stage2_m2": 25.0,
        })

        status, result = self.request_process(payload)

        self.assertEqual(status, 200, result)
        self.assertEqual(result["solver"], "rigorous_1d")
        self.assertGreater(result["recovery"], 0.0)
        self.assertLessEqual(result["recovery"], 1.0)

    def test_rigorous_mode_rejects_unsupported_crossflow(self):
        payload = self.payload("SINGLE_STAGE")
        payload["design"]["flow_pattern"] = "crossflow"

        status, result = self.request_process(payload)

        self.assertEqual(status, 400)
        self.assertIn("cocurrent or countercurrent", result["error"])

    def test_screening_solver_remains_available(self):
        payload = self.payload("SINGLE_STAGE")
        payload["solver"] = "screening"
        payload["design"].update({
            "target_stage_cut": 0.3,
            "pipeline_pressure_bar": 110.0,
            "electricity_price": 0.07,
        })

        status, result = self.request_process(payload)

        self.assertEqual(status, 200, result)
        self.assertIn("tea", result)
        self.assertIn("total_compression_kW", result)


if __name__ == "__main__":
    unittest.main()
