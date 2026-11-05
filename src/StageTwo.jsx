import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = {
  recovery: "#176b78",
  purity: "#6b7280",
  area: "#176b78",
};

function format(value, digits = 2) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value);
}

function RangeInput({ label, unit, value, min, max, step, onChange }) {
  const progress = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  return (
    <label className="process-field">
      <span>{label} <small>{format(value, 2)} {unit}</small></span>
      <input
        type="range"
        className="process-range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--range-progress": `${progress}%` }}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={`${label} (${unit})`}
      />
    </label>
  );
}

function PerformanceGraph({ data, targetRecovery }) {
  const chartData = useMemo(
    () => data.map((point) => ({
      area: point.area,
      recoveryPercent: point.recovery == null ? null : point.recovery * 100,
      purityPercent: point.purity == null ? null : point.purity * 100,
    })),
    [data],
  );

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis
          dataKey="area"
          type="number"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(value) => format(value, 0)}
          label={{ value: "Area (m²)", position: "insideBottom", offset: -3 }}
        />
        <YAxis
          domain={[0, 100]}
          tickFormatter={(value) => `${value}%`}
          width={45}
        />
        <Tooltip
          labelFormatter={(value) => `Area: ${format(Number(value), 2)} m²`}
          formatter={(value, name) => [`${format(Number(value), 2)}%`, name]}
        />
        <Legend verticalAlign="top" height={28} />
        <Line
          dataKey="recoveryPercent"
          name="CO₂ recovery"
          stroke={COLORS.recovery}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
        />
        <Line
          dataKey="purityPercent"
          name="Permeate purity"
          stroke={COLORS.purity}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
        />
        <ReferenceLine
          y={targetRecovery * 100}
          stroke="#b36b20"
          strokeDasharray="5 4"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function PressureGraph({ data, currentPressure }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis
          dataKey="pressure"
          type="number"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(value) => format(value, 2)}
          label={{ value: "Permeate pressure (bar)", position: "insideBottom", offset: -3 }}
        />
        <YAxis
          dataKey="area"
          type="number"
          domain={["auto", "auto"]}
          tickFormatter={(value) => format(value, 0)}
          width={55}
        />
        <Tooltip
          labelFormatter={(value) => `Permeate pressure: ${format(Number(value), 3)} bar`}
          formatter={(value) => [value == null ? "Not attainable" : `${format(Number(value), 2)} m²`, "Required area"]}
        />
        <Legend verticalAlign="top" height={28} />
        <Line
          dataKey="area"
          name="Area for 90% recovery"
          stroke={COLORS.area}
          strokeWidth={2}
          dot={{ r: 3 }}
          connectNulls={false}
        />
        <ReferenceLine x={currentPressure} stroke="#b36b20" strokeDasharray="5 4" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function StageTwo({ prediction, design, setDesign, result, setResult }) {
  const [mode, setMode] = useState("design");
  const [targetRecovery, setTargetRecovery] = useState(0.9);
  const [targetArea, setTargetArea] = useState(Math.max(1, design.areaStage1M2 || 2000));
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!prediction) {
      setResult(null);
      setIsRunning(false);
      setError("");
      return undefined;
    }

    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      setIsRunning(true);
      setError("");
      try {
        const response = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            P_CO2: prediction.permeability,
            alpha: prediction.selectivity,
            thickness: design.thicknessUm,
            P_feed: design.feedPressureBar,
            P_perm: design.permeatePressureBar,
            F_feed: design.feedFlowKmolH,
            x_feed: design.feedCo2Fraction,
            mode,
            target_value: mode === "design" ? targetRecovery : targetArea,
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? payload.error ?? "The membrane simulation failed.");
        }
        setResult({ ...payload, design: { ...design, mode } });
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          setResult(null);
          setError(requestError instanceof Error ? requestError.message : "The membrane simulation failed.");
        }
      } finally {
        if (!controller.signal.aborted) setIsRunning(false);
      }
    }, 180);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [
    design.feedCo2Fraction,
    design.feedFlowKmolH,
    design.feedPressureBar,
    design.permeatePressureBar,
    design.thicknessUm,
    mode,
    prediction,
    setResult,
    targetArea,
    targetRecovery,
  ]);

  function updateDesign(key, value) {
    setDesign((previous) => ({ ...previous, [key]: value }));
  }

  const graphs = result?.graphs;
  const pressureData = graphs?.pressure_sweep ?? [];

  return (
    <section className="stage-two-section" id="stage2">
      <div className="stage-two-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> STAGE 02 · PROCESS DESIGN</div>
          <h2>Membrane CO₂ Capture · Stage 2</h2>
          <p>Countercurrent 1D solution-diffusion simulation, sized or evaluated at a specified area.</p>
        </div>
      </div>

      <div className="process-layout">
        <div className="panel process-controls">
          <div className="panel-heading">
            <div><h3>Material and process inputs</h3><p>Stage 1 properties pass through automatically.</p></div>
          </div>
          {!prediction && (
            <div className="stage-two-empty">
              Run a Stage 1 prediction to populate permeability and selectivity.
            </div>
          )}
          <div className="process-inputs">
            <label className="process-field">
              <span>CO₂ permeability <small>Barrer</small></span>
              <input type="number" value={prediction?.permeability ?? ""} readOnly aria-label="CO2 permeability in Barrer" />
            </label>
            <label className="process-field">
              <span>CO₂/N₂ selectivity <small>α</small></span>
              <input type="number" value={prediction?.selectivity ?? ""} readOnly aria-label="CO2 N2 selectivity" />
            </label>
            <RangeInput
              label="Selective-layer thickness"
              unit="µm"
              min={0.05}
              max={20}
              step={0.05}
              value={design.thicknessUm}
              onChange={(value) => updateDesign("thicknessUm", value)}
            />
            <RangeInput
              label="Feed pressure"
              unit="bar"
              min={1}
              max={20}
              step={0.1}
              value={design.feedPressureBar}
              onChange={(value) => updateDesign("feedPressureBar", value)}
            />
            <RangeInput
              label="Permeate pressure"
              unit="bar"
              min={0.02}
              max={0.9}
              step={0.01}
              value={design.permeatePressureBar}
              onChange={(value) => updateDesign("permeatePressureBar", value)}
            />
            <RangeInput
              label="Fresh feed flow"
              unit="kmol/h"
              min={1}
              max={10000}
              step={10}
              value={design.feedFlowKmolH}
              onChange={(value) => updateDesign("feedFlowKmolH", value)}
            />
            <RangeInput
              label="Feed CO₂ fraction"
              unit="%"
              min={1}
              max={90}
              step={1}
              value={design.feedCo2Fraction * 100}
              onChange={(value) => updateDesign("feedCo2Fraction", value / 100)}
            />
            <fieldset className="process-mode-field">
              <legend>Calculation mode</legend>
              <label>
                <input
                  type="radio"
                  name="stage2-mode"
                  value="design"
                  checked={mode === "design"}
                  onChange={() => setMode("design")}
                />
                Design Mode (Input Recovery)
              </label>
              <label>
                <input
                  type="radio"
                  name="stage2-mode"
                  value="simulation"
                  checked={mode === "simulation"}
                  onChange={() => {
                    if (result?.area) setTargetArea(result.area);
                    setMode("simulation");
                  }}
                />
                Simulation Mode (Input Area)
              </label>
            </fieldset>
            {mode === "design" ? (
              <RangeInput
                label="Target recovery"
                unit="%"
                min={1}
                max={98}
                step={1}
                value={targetRecovery * 100}
                onChange={(value) => setTargetRecovery(value / 100)}
              />
            ) : (
              <label className="process-field">
                <span>Membrane area <small>m²</small></span>
                <input
                  type="number"
                  min="0.01"
                  step="any"
                  value={targetArea}
                  onChange={(event) => {
                    const value = event.target.valueAsNumber;
                    if (Number.isFinite(value) && value > 0) setTargetArea(value);
                  }}
                  aria-label="Membrane area in square meters"
                />
              </label>
            )}
          </div>
          <p className="process-method-note">
            Results recalculate when process inputs change. Recovery is calculated from component balances.
          </p>
        </div>

        <div className="process-output" aria-live="polite">
          {isRunning && <div className="stage-two-empty">Solving the spatial membrane model and updating graph data…</div>}
          {error && <div className="process-error" role="alert">{error}</div>}
          {result && (
            <>
              <div className="process-kpis metrics-four">
                <article className="process-kpi"><span>PRODUCT CO₂ PURITY</span><strong>{format(result.purity * 100, 1)}<small> mol%</small></strong></article>
                <article className="process-kpi"><span>CO₂ RECOVERY</span><strong>{format(result.recovery * 100, 1)}<small> %</small></strong></article>
                <article className="process-kpi"><span>TOTAL MEMBRANE AREA</span><strong>{format(result.area, 1)}<small> m²</small></strong></article>
                <article className="process-kpi energy-kpi"><span>SPECIFIC ENERGY</span><strong>{format(result.specific_energy, 1)}<small> kWh/t CO₂</small></strong></article>
              </div>
              <article className="panel process-detail cost-panel">
                <div className="panel-heading">
                  <div><h3>Levelized capture cost · screening</h3><p>Annualized membrane capital, fixed OPEX and electricity.</p></div>
                  <strong>${format(result.cost, 1)}/t CO₂</strong>
                </div>
              </article>
              <div className="stage-charts-grid">
                <article className="panel process-detail">
                  <div className="panel-heading">
                    <div><h3>Area vs. module performance</h3><p>Recovery and purity from repeated ODE solves.</p></div>
                  </div>
                  <PerformanceGraph data={graphs.area_sweep} targetRecovery={targetRecovery} />
                </article>
                <article className="panel process-detail">
                  <div className="panel-heading">
                    <div><h3>Vacuum pressure vs. area</h3><p>Area required for 90% CO₂ recovery.</p></div>
                  </div>
                  <PressureGraph
                    data={pressureData}
                    currentPressure={design.permeatePressureBar}
                  />
                </article>
              </div>
              <p className="stage-two-caveat">
                Captured product: {format(result.product_co2_flow, 3)} kmol/h CO₂ in{" "}
                {format(result.product_flow, 3)} kmol/h total product.
                {result.purity < 0.8 && " Product purity follows the feed composition, selectivity and recovery; this single-stage model does not assume 80% purity."}
              </p>
            </>
          )}
          {!result && !error && !isRunning && (
            <div className="panel process-placeholder">
              {prediction ? "Adjust a process input to solve the membrane model." : "Stage 2 results appear after a Stage 1 prediction."}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
