const GPU_TO_SI = 3.348e-10;
const BAR_TO_PA = 1e5;
const GAS_CONSTANT = 8.314;
const FEED_TEMPERATURE_K = 298.15;
const CO2_RICH_HEAT_CAPACITY_RATIO = 1.3;
const COMPRESSOR_STAGES = 2;
const CO2_MOLAR_MASS_KG_PER_KMOL = 44.01;
const N2_MOLAR_MASS_KG_PER_KMOL = 28.014;
const REFERENCE_SUCTION_PRESSURE_BAR = 1;
const ATMOSPHERIC_PRESSURE_BAR = 1.01325;
const Z_COMPRESSIBILITY = 1;
export const DEFAULT_PROCESS = {
  feedCo2Fraction: 0.1,
  feedPressureBar: 1,
  permeatePressureBar: 0.1,
  thicknessUm: 1,
  feedFlowKmolH: 1000,
  targetRecovery: 0.9,
  compressorEfficiency: 0.72,
  vacuumEfficiency: 0.72,
};

function logMeanPressureDifference(feedPartialPressure, permeatePartialPressure) {
  if (!Number.isFinite(feedPartialPressure) || !Number.isFinite(permeatePartialPressure)
    || feedPartialPressure <= 0 || permeatePartialPressure <= 0
    || feedPartialPressure < permeatePartialPressure) {
    return null;
  }
  const difference = feedPartialPressure - permeatePartialPressure;
  if (difference <= 1e-12 * Math.max(feedPartialPressure, permeatePartialPressure)) {
    return (feedPartialPressure + permeatePartialPressure) / 2;
  }
  const logRatio = Math.log(feedPartialPressure / permeatePartialPressure);
  return logRatio > 0 ? difference / logRatio : null;
}

function isentropicCompressionPowerKw(flowKmolH, molecularWeightKgPerKmol, pressureRatio, efficiency, stages = 1) {
  if (pressureRatio <= 1 || flowKmolH <= 0) return 0;
  const massFlowKgPerSecond = flowKmolH * molecularWeightKgPerKmol / 3600;
  const molecularWeightKgPerMol = molecularWeightKgPerKmol / 1000;
  const stagePressureRatio = pressureRatio ** (1 / stages);
  const workPerSecond = massFlowKgPerSecond
    * Z_COMPRESSIBILITY
    * GAS_CONSTANT
    * FEED_TEMPERATURE_K
    / (molecularWeightKgPerMol * efficiency)
    * CO2_RICH_HEAT_CAPACITY_RATIO / (CO2_RICH_HEAT_CAPACITY_RATIO - 1)
    * stages
    * (stagePressureRatio ** ((CO2_RICH_HEAT_CAPACITY_RATIO - 1) / CO2_RICH_HEAT_CAPACITY_RATIO) - 1);
  return workPerSecond / 1000;
}

export function simulateProcess(permeability, selectivity, design) {
  const {
    feedCo2Fraction: xFeed,
    feedPressureBar: feedPressure,
    permeatePressureBar: permeatePressure,
    thicknessUm: thickness,
    feedFlowKmolH: feedFlow,
    targetRecovery: recovery,
    compressorEfficiency: efficiency,
    vacuumEfficiency = efficiency,
  } = design;

  if (![permeability, selectivity, xFeed, feedPressure, permeatePressure, thickness, feedFlow, recovery, efficiency, vacuumEfficiency].every(Number.isFinite)
    || permeability <= 0 || selectivity <= 0 || xFeed <= 0 || xFeed >= 1
    || feedPressure <= 0 || permeatePressure <= 0 || permeatePressure >= feedPressure
    || thickness <= 0 || feedFlow <= 0 || recovery <= 0 || recovery > 1
    || efficiency <= 0 || efficiency > 1 || vacuumEfficiency <= 0 || vacuumEfficiency > 1) {
    return { error: "Check inputs: pressures, thickness, flow and efficiency must be positive; permeate pressure must be below feed pressure." };
  }

  const thicknessCm = thickness * 1e-4;
  const permeanceCo2Gpu = (permeability / thicknessCm) * 1e-4;
  const permeanceN2Gpu = (permeability / selectivity / thicknessCm) * 1e-4;
  const permeanceCo2Si = permeanceCo2Gpu * GPU_TO_SI;
  const permeanceN2Si = permeanceN2Gpu * GPU_TO_SI;
  let yCo2 = 0.5;
  let dpCo2 = 0;
  let dpN2 = 0;
  let fluxCo2 = 0;
  let fluxN2 = 0;
  let converged = false;

  for (let iteration = 0; iteration < 200; iteration += 1) {
    const pCo2Feed = xFeed * feedPressure;
    const pCo2Permeate = yCo2 * permeatePressure;
    dpCo2 = logMeanPressureDifference(pCo2Feed, pCo2Permeate);

    const pN2Feed = (1 - xFeed) * feedPressure;
    const pN2Permeate = (1 - yCo2) * permeatePressure;
    dpN2 = logMeanPressureDifference(pN2Feed, pN2Permeate);
    if (dpCo2 === null || dpN2 === null) {
      return { error: "Partial-pressure driving force is zero, reversed, or invalid." };
    }

    // Fick's law uses SI permeance and the log-mean partial-pressure gradient in Pa.
    fluxCo2 = permeanceCo2Si * dpCo2 * BAR_TO_PA;
    fluxN2 = permeanceN2Si * dpN2 * BAR_TO_PA;
    const totalFlux = fluxCo2 + fluxN2;
    if (totalFlux <= 1e-30) return { error: "Total membrane flux is too small to solve permeate composition." };
    const nextYCo2 = fluxCo2 / totalFlux;
    if (Math.abs(nextYCo2 - yCo2) < 1e-7) {
      yCo2 = nextYCo2;
      converged = true;
      break;
    }
    yCo2 = nextYCo2;
  }

  if (!converged) return { error: "Permeate composition did not converge." };
  if (yCo2 <= 1e-9) return { error: "Predicted permeate CO₂ concentration is near zero." };

  const feedCo2 = feedFlow * xFeed;
  const targetCo2Captured = feedCo2 * recovery;
  const co2Captured = Math.min(targetCo2Captured, feedFlow * yCo2);
  const permeateFlow = co2Captured / yCo2;
  const stageCut = Math.min(1, permeateFlow / feedFlow);
  const retentateFlow = (1 - stageCut) * feedFlow;
  const actualRecovery = feedCo2 > 0 ? co2Captured / feedCo2 : 0;
  const retentateCo2 = retentateFlow > 1e-12
    ? Math.max(0, Math.min(1, (feedFlow * xFeed - co2Captured) / retentateFlow))
    : 0;
  const capturedMolPerSecond = co2Captured * 1000 / 3600;
  const areaDenominator = permeanceCo2Si * dpCo2 * BAR_TO_PA;
  const area = areaDenominator > 1e-30 ? capturedMolPerSecond / areaDenominator : Infinity;
  const feedMolecularWeightKgPerKmol =
    xFeed * CO2_MOLAR_MASS_KG_PER_KMOL
    + (1 - xFeed) * N2_MOLAR_MASS_KG_PER_KMOL;
  const compressorPower = isentropicCompressionPowerKw(
    feedFlow,
    feedMolecularWeightKgPerKmol,
    feedPressure / REFERENCE_SUCTION_PRESSURE_BAR,
    efficiency,
    COMPRESSOR_STAGES,
  );
  const permeateMolecularWeight =
    yCo2 * CO2_MOLAR_MASS_KG_PER_KMOL
    + (1 - yCo2) * N2_MOLAR_MASS_KG_PER_KMOL;
  const permeateMassFlowKgH = permeateFlow * permeateMolecularWeight;
  const vacuumPower = permeatePressure < ATMOSPHERIC_PRESSURE_BAR
    ? isentropicCompressionPowerKw(
      permeateFlow,
      permeateMolecularWeight,
      ATMOSPHERIC_PRESSURE_BAR / permeatePressure,
      vacuumEfficiency,
    )
    : 0;
  const totalPower = compressorPower + vacuumPower;
  const co2TonnesPerHour = co2Captured * CO2_MOLAR_MASS_KG_PER_KMOL / 1000;
  const specificEnergy = co2TonnesPerHour > 0 ? totalPower / co2TonnesPerHour : Infinity;

  return {
    purity: yCo2,
    recovery,
    targetRecovery: recovery,
    actualRecovery,
    feedCo2,
    targetCo2Captured,
    unmetCo2Capture: Math.max(0, targetCo2Captured - co2Captured),
    stageCut,
    permeateFlow,
    retentateFlow,
    co2Captured,
    retentateCo2,
    area,
    permeanceCo2Gpu,
    permeanceN2Gpu,
    compressorPower,
    vacuumPower,
    totalPower,
    specificEnergy,
    compressorStages: COMPRESSOR_STAGES,
    compressorEfficiency: efficiency,
    vacuumEfficiency,
    permeateMassFlowKgH,
    co2TonnesPerHour,
  };
}

export function makeStage2Studies(permeability, selectivity, design) {
  const thicknesses = [0.2, 0.5, 1, 2, 5, 10];
  const thicknessSweep = thicknesses.map((thicknessUm) => ({
    x: thicknessUm,
    result: simulateProcess(permeability, selectivity, { ...design, thicknessUm }),
  }));
  const recoveries = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
  const recoverySweep = recoveries.map((targetRecovery) => ({
    x: targetRecovery * 100,
    result: simulateProcess(permeability, selectivity, { ...design, targetRecovery }),
  }));
  const pressures = [1, 1.5, 2, 3, 4, 5].filter((pressure) => pressure > design.permeatePressureBar);
  const pressureSweep = pressures.map((feedPressureBar) => ({
    x: feedPressureBar,
    result: simulateProcess(permeability, selectivity, {
      ...design,
      feedPressureBar: Math.max(feedPressureBar, design.permeatePressureBar + 0.01),
    }),
  }));
  return { thicknessSweep, recoverySweep, pressureSweep };
}
