import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_PROCESS, simulateProcess } from "../../src/stage2.js";

const design = {
  feedCo2Fraction: 0.01,
  feedPressureBar: 6,
  permeatePressureBar: 0.1,
  thicknessUm: 1,
  feedFlowKmolH: 10,
  targetRecovery: 0.65,
  compressorEfficiency: 0.72,
};

const result = simulateProcess(4836.77, 9.14, design);

test("CO2 mass balance applies the recovery target to feed CO2 flow", () => {
  assert.ok(Math.abs(result.feedCo2 - 0.1) < 1e-12);
  assert.ok(Math.abs(result.co2Captured - 0.065) < 1e-12);
  assert.ok(Math.abs(result.actualRecovery - 0.65) < 1e-12);
  assert.ok(Math.abs(result.retentateFlow * result.retentateCo2
    + result.co2Captured - result.feedCo2) < 1e-12);
});

test("Stage 1 properties produce CO2 and N2 permeances for solution-diffusion", () => {
  assert.ok(Math.abs(result.permeanceCo2Gpu - 4836.77) < 1e-9);
  assert.ok(Math.abs(result.permeanceN2Gpu - 529.187) < 1e-3);
  assert.ok(result.area > 0 && result.area < 100_000);
});

test("total power includes feed compression and permeate vacuum", () => {
  assert.ok(Math.abs(result.compressorPower - 19.0351) < 0.01);
  assert.ok(result.vacuumPower > 0);
  assert.ok(Math.abs(result.totalPower - result.compressorPower - result.vacuumPower) < 1e-12);
  assert.ok(Math.abs(result.specificEnergy
    - result.totalPower / result.co2TonnesPerHour) < 1e-9);
});

test("energy output exposes power and per-tonne consumption without cost estimates", () => {
  assert.ok(Math.abs(result.specificEnergy - 6996.11) < 0.01);
  assert.ok(result.totalPower > 0);
  assert.equal("captureCost" in result, false);
  assert.equal("costBreakdown" in result, false);
});

test("corrected mock case gives plausible membrane area and SEC", () => {
  const mock = simulateProcess(1000, 30, DEFAULT_PROCESS);
  assert.ok(mock.area >= 1_000 && mock.area <= 50_000);
  assert.ok(mock.specificEnergy < 130);
  assert.ok(Math.abs(mock.actualRecovery - 0.9) < 1e-12);
});

test("reversed partial-pressure driving force returns an explicit model error", () => {
  const invalid = simulateProcess(1000, 30, {
    ...DEFAULT_PROCESS,
    feedCo2Fraction: 0.01,
    permeatePressureBar: 0.5,
  });
  assert.match(invalid.error, /driving force/);
});
