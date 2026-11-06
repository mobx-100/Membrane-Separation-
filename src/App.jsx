import { lazy, Suspense, useEffect, useState } from "react";

const MultiStageTwo = lazy(() => import("./StageTwo.jsx"));

const EXAMPLE_SMILES = "*/C(=C(/*)c1ccc([Si](C)(C)C)cc1)c1ccccc1";
const EXAMPLES = [
  { label: "Silicon-rich", smiles: EXAMPLE_SMILES },
  { label: "Polyethylene", smiles: "*CC*" },
  { label: "Polyvinyl acetate", smiles: "*CC(=O)OC*" },
  { label: "Polyether ester", smiles: "*CCOC(=O)*" },
];
const MODEL_CHOICES = [
  { id: "lr", label: "LR · Linear Regression" },
  { id: "rf", label: "RF · Random Forest" },
  { id: "xg", label: "XG · XGBoost" },
];

function profileFromRdkit(rdkitDescriptors) {
  if (!rdkitDescriptors) return {};
  return {
    "Molecular weight": rdkitDescriptors.MolWt,
    "Polar surface": rdkitDescriptors.TPSA,
    Lipophilicity: rdkitDescriptors.MolLogP,
    "Aromatic rings": rdkitDescriptors.NumAromaticRings,
    "Rotatable bonds": rdkitDescriptors.NumRotatableBonds,
    Heteroatoms: rdkitDescriptors.NumHeteroatoms,
    heavyAtoms: rdkitDescriptors.NumHeavyAtoms,
    ringCount: rdkitDescriptors.NumRings,
  };
}

function formatValue(value, digits = 1, minimumDigits = 0) {
  if (!Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: minimumDigits,
  }).format(value);
}

function loadSavedPrediction() {
  try {
    const prediction = JSON.parse(
      window.localStorage.getItem("membrane-lab-stage1-prediction") ?? "null",
    );
    if (
      prediction
      && typeof prediction.smiles === "string"
      && Number.isFinite(prediction.permeability)
      && prediction.permeability > 0
      && Number.isFinite(prediction.selectivity)
      && prediction.selectivity > 0
    ) {
      return prediction;
    }
  } catch {
    return null;
  }
  return null;
}

function clearSavedPrediction() {
  try {
    window.localStorage.removeItem("membrane-lab-stage1-prediction");
    window.localStorage.removeItem("P_CO2");
    window.localStorage.removeItem("alpha");
  } catch {
    return;
  }
}

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  const paths = {
    flask: <><path d="M9 3h6M10 3v6l-5.8 9.2A2 2 0 0 0 5.9 21h12.2a2 2 0 0 0 1.7-2.8L14 9V3" /><path d="M7.5 16h9" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
    chart: <><path d="M4 19V5M4 19h17" /><path d="m7 15 4-4 3 2 5-6" /><path d="M17 7h2v2" /></>,
    atom: <><circle cx="12" cy="12" r="1.5" /><ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(30 12 12)" /><ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(-30 12 12)" /></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5m0-8h.01" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M5 21h14" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></>,
    moon: <path d="M20.8 13A8.5 8.5 0 0 1 11 3.2 8.5 8.5 0 1 0 20.8 13Z" />,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function PerformanceMap({ prediction }) {
  const x = Math.max(6, Math.min(94, ((Math.log10(Math.max(prediction.permeability, 0.1)) + 1) / 5.8) * 100));
  const y = Math.max(8, Math.min(92, 100 - ((Math.log10(Math.max(prediction.selectivity, 1)) / 2.1) * 100)));
  return (
    <div className="map-wrap">
      <div className="map-y-label">CO₂ / N₂ selectivity</div>
      <div className="performance-map">
        {[25, 50, 75].map((position) => (
          <span className="map-grid-line vertical" key={`vertical-${position}`} style={{ left: `${position}%` }} />
        ))}
        {[25, 50, 75].map((position) => (
          <span className="map-grid-line horizontal" key={`horizontal-${position}`} style={{ top: `${position}%` }} />
        ))}
        <div className="candidate-point" style={{ left: `${x}%`, top: `${y}%` }} title="Current polymer">
          <span />
          <b>Current polymer</b>
        </div>
      </div>
      <div className="map-x-label">CO₂ permeability · Barrer <span>Log scale</span></div>
      <p className="chart-footnote">Log-scaled position of the current model prediction; no reference dataset is shown.</p>
    </div>
  );
}

function DescriptorChart({ descriptors }) {
  const entries = [
  ["Molecular weight", descriptors["Molecular weight"], "g/mol", 1],
  ["Polar surface area", descriptors["Polar surface"], "Å²", 1],
  ["Lipophilicity", descriptors.Lipophilicity, "LogP", 2],
  ["Aromatic rings", descriptors["Aromatic rings"], "", 0],
  ["Rotatable bonds", descriptors["Rotatable bonds"], "", 0],
  ["Heteroatoms", descriptors.Heteroatoms, "", 0],
  ["Heavy atoms", descriptors.heavyAtoms, "", 0],
  ["Rings", descriptors.ringCount, "", 0],
  ];
  return (
  <div className="descriptor-table-wrap">
    <table className="descriptor-table">
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>
        {entries.map(([label, value, unit, digits]) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            <td>{value === undefined ? "—" : `${formatValue(value, digits)}${unit ? ` ${unit}` : ""}`}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <p className="chart-footnote">Descriptors are calculated by RDKit in the model API.</p>
  </div>
  );
}

function PredictionBars({ prediction }) {
  const permWidth = Math.max(2, Math.min(100, (Math.log10(Math.max(prediction.permeability, 0.1) + 1) / Math.log10(47001)) * 100));
  const selectivityWidth = Math.max(2, Math.min(100, (Math.log10(Math.max(prediction.selectivity, 1)) / Math.log10(120)) * 100));
  return (
    <div className="prediction-bars">
      <div className="prediction-bar">
        <div><span>CO₂ permeability</span><strong>{formatValue(prediction.permeability)} <small>Barrer</small></strong></div>
        <div className="bar-track"><i className="bar-cyan" style={{ width: `${permWidth}%` }} /></div>
        <div className="bar-scale"><span>0.1</span><span>Log scale · 47,000</span></div>
      </div>
      <div className="prediction-bar">
        <div><span>CO₂ / N₂ selectivity</span><strong>{formatValue(prediction.selectivity, 2)} <small>α</small></strong></div>
        <div className="bar-track"><i className="bar-violet" style={{ width: `${selectivityWidth}%` }} /></div>
        <div className="bar-scale"><span>1</span><span>Log scale · 120</span></div>
      </div>
    </div>
  );
}

const NOTEBOOK_GRAPHS = [
  { title: "Test-set parity plots", file: "parity-plots.png", description: "Actual vs. predicted CO₂ permeability and selectivity for the notebook’s evaluated models." },
  { title: "Residual analysis", file: "residual-analysis.png", description: "Residuals vs. predicted values, residual distribution, Q–Q plot, and residuals vs. actual." },
  { title: "Learning curve", file: "learning-curve.png", description: "Random Forest permeability training and validation scores by training-set size." },
  { title: "Robeson plot", file: "robeson-plot.png", description: "CO₂ permeability vs. CO₂/N₂ selectivity with Robeson upper-bound references." },
  { title: "Permutation feature importance", file: "feature-importance.png", description: "Top 20 features ranked by permutation importance for the permeability model." },
  { title: "Notebook Stage 2 thickness trade-off", file: "stage2-thickness-tradeoff.png", description: "Saved notebook figure comparing membrane area and energy across layer thicknesses." },
];

function NotebookGraphGallery() {
  return (
    <section className="notebook-gallery" id="notebook-graphs">
      <div className="section-heading">
        <div><h2>Notebook graphs</h2><p>Original output plots captured from Carbon_Capture_ml.ipynb</p></div>
        <span className="static-tag">SAVED NOTEBOOK OUTPUTS</span>
      </div>
      <div className="notebook-graph-grid">
        {NOTEBOOK_GRAPHS.map((graph) => (
          <figure className="panel notebook-graph" key={graph.file}>
            <figcaption><strong>{graph.title}</strong><span>{graph.description}</span></figcaption>
            <a href={`/notebook-graphs/${graph.file}`} target="_blank" rel="noreferrer" aria-label={`Open full-size graph: ${graph.title}`}>
              <img src={`/notebook-graphs/${graph.file}`} alt={`${graph.title}. ${graph.description}`} loading="lazy" />
            </a>
          </figure>
        ))}
      </div>
      <p className="notebook-gallery-note"><Icon name="info" size={14} /> Notebook figures are saved outputs from the original run; they are not recalculated when SMILES or process settings change. Stage 2 process results are recalculated when you run the cascade model.</p>
    </section>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    try {
      return window.localStorage.getItem("membrane-lab-theme") === "dark" ? "dark" : "light";
    } catch {
      return "light";
    }
  });
  const [smiles, setSmiles] = useState(
    () => loadSavedPrediction()?.smiles ?? EXAMPLE_SMILES,
  );
  const [selectedModel, setSelectedModel] = useState(
    () => loadSavedPrediction()?.selected_model?.id ?? "xg",
  );
  const [prediction, setPrediction] = useState(loadSavedPrediction);
  const [processDesign, setProcessDesign] = useState({
    layout: "TWO_STAGE_RECYCLE",
    flowPattern: "countercurrent",
    feedCo2Fraction: 0.1,
    feedPressureBar: 6,
    permeatePressureBar: 0.1,
    thicknessUm: 0.5,
    feedFlowKmolH: 1000,
    areaStage1M2: 100,
    areaStage2M2: 25,
    targetStageCut: 0.3,
    pipelinePressureBar: 110,
    electricityPrice: 0.07,
  });
  const [processResult, setProcessResult] = useState(null);
  const [modelStatus, setModelStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem("membrane-lab-theme", theme);
    } catch {
      // Theme remains active for this session when storage is unavailable.
    }
  }, [theme]);

  async function refreshModelHealth() {
    setModelStatus("checking");
    try {
      const response = await fetch("/api/health");
      const health = await response.json();
      if (!response.ok || !health.model_ready) {
        setModelStatus("missing");
        setModelInfo(null);
        return;
      }
      const availableIds = new Set(health.available_models?.map((model) => model.id));
      if (!MODEL_CHOICES.every((model) => availableIds.has(model.id))) {
        setModelStatus("incompatible");
        setModelInfo(null);
        return;
      }
      setModelInfo(health);
      setModelStatus("ready");
    } catch {
      setModelStatus("unavailable");
      setModelInfo(null);
    }
  }

  useEffect(() => {
    refreshModelHealth();
  }, []);

  async function runPrediction(event) {
    event?.preventDefault();
    const trimmed = smiles.trim();
    if (!trimmed) {
      setError("Enter a polymer SMILES string to run a prediction.");
      return;
    }
    if (/[<>{}]/.test(trimmed) || !/[A-Za-z*]/.test(trimmed)) {
      setError("Enter a polymer SMILES string using supported atom symbols.");
      return;
    }
    setError("");
    setPrediction(null);
    setProcessResult(null);
    setIsPredicting(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: trimmed, model: selectedModel }),
      });
      const result = await response.json();
      if (!response.ok) {
        if (response.status === 503) {
          setModelStatus("missing");
          setModelInfo(null);
        }
        throw new Error(result.error ?? "The prediction service could not process this SMILES.");
      }
      if (![result.permeability, result.selectivity].every((value) => Number.isFinite(value) && value > 0)) {
        throw new Error("The model API returned invalid prediction values.");
      }
      if (result.selected_model?.id !== selectedModel) {
        setModelStatus("incompatible");
        throw new Error("The running API does not support model selection. Restart it with `npm run api`.");
      }
      const predictionResult = {
        ...result,
        smiles: trimmed,
        rdkitDescriptors: result.descriptors,
        mode: "model",
      };
      setPrediction(predictionResult);
      try {
        window.localStorage.setItem(
          "membrane-lab-stage1-prediction",
          JSON.stringify(predictionResult),
        );
        window.localStorage.setItem("P_CO2", String(result.permeability));
        window.localStorage.setItem("alpha", String(result.selectivity));
      } catch {
        // The current React state still supplies Stage 2 if storage is unavailable.
      }
      setModelStatus("ready");
      setModelInfo((previous) => previous ? {
        ...previous,
        selected_model: result.selected_model,
        test_metrics: result.test_metrics,
      } : previous);
    } catch (predictionError) {
      if (predictionError instanceof TypeError) {
        setModelStatus("unavailable");
        setError("The model API is not reachable. Start it with `npm run api` and retry.");
      } else {
        setError(predictionError instanceof Error ? predictionError.message : "The model API returned an invalid response.");
      }
    } finally {
      setIsPredicting(false);
    }
  }

  function chooseExample(value) {
    setSmiles(value);
    setError("");
    setPrediction(null);
    setProcessResult(null);
    clearSavedPrediction();
  }

  function downloadResult() {
    if (!prediction) return;
    const result = {
      smiles,
      permeability_barrer: prediction.permeability,
      co2_n2_selectivity: prediction.selectivity,
      log10_permeability: prediction.log10_permeability,
      log10_selectivity: prediction.log10_selectivity,
      selected_model: prediction.selected_model,
      mode: "trained_notebook_models",
      descriptors: structureProfile,
      models: prediction.models,
      test_metrics: prediction.test_metrics,
      process_design: processDesign,
      stage2_results: processResult,
    };
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "membrane-prediction.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  const activePrediction = prediction?.smiles === smiles.trim() ? prediction : null;
  const structureProfile = profileFromRdkit(activePrediction?.rdkitDescriptors);
  const selectedModelInfo = modelInfo?.available_models?.find((model) => model.id === selectedModel);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Membrane Lab home">
          <span className="brand-mark"><Icon name="flask" size={18} /></span>
          <span>Membrane Lab<small>CARBON CAPTURE</small></span>
        </a>
        <nav className="menu-bar" aria-label="Main navigation">
          <a className="menu-link active" href="#overview">Overview</a>
          <a className="menu-link" href="#predictor">Polymer predictor</a>
          <a className="menu-link" href="#analytics">Analytics</a>
          <a className="menu-link" href="#stage2">Stage 2 process</a>
          <a className="menu-link" href="#notebook-graphs">Notebook graphs</a>
        </nav>
        <div className="topbar-right">
          <span className={`model-status ${modelStatus === "ready" ? "online" : modelStatus === "checking" ? "checking" : "offline"}`}>
            <i />
            <span>{modelStatus === "ready" ? `Model: Connected${selectedModelInfo ? ` · ${selectedModelInfo.label}` : ""}` : modelStatus === "checking" ? "Model: Connecting" : "Model: Offline"}</span>
          </span>
          <span className="model-context">
            {modelInfo?.dataset_size ?? "—"} samples
          </span>
          <button
            type="button"
            className="theme-toggle"
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            onClick={() => setTheme((current) => current === "light" ? "dark" : "light")}
          >
            <Icon name={theme === "light" ? "moon" : "sun"} size={17} />
            <span>{theme === "light" ? "Dark" : "Light"}</span>
          </button>
          <span className="avatar">ML</span>
        </div>
      </header>

      <main className="main-content" id="overview">
        <div className="content-inner">
        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> POLYMER DISCOVERY</div>
            <h1>Membrane performance</h1>
            <p>Explore structure–property predictions for next-generation CO₂ capture materials.</p>
          </div>
          <div className="heading-tag"><span className="live-dot" /> STAGE 01 <b>·</b> MODEL INFERENCE</div>
        </section>

        {(modelStatus === "missing" || modelStatus === "unavailable" || modelStatus === "incompatible") && (
          <div className="preview-notice model-unavailable-notice">
            <Icon name="info" size={17} />
            <span><strong>No compatible model prediction is available.</strong> {modelStatus === "missing" ? "Train the models with `npm run train:model`, then restart the API." : modelStatus === "incompatible" ? "The running API is outdated. Stop it and restart with `npm run api`." : "Start the backend with `npm run api` and retry."} The app never substitutes preview values.</span>
            <button onClick={refreshModelHealth}>Retry connection <Icon name="arrow" size={15} /></button>
          </div>
        )}

        <section className="input-card" id="predictor">
          <div className="card-title-row">
            <div className="section-title">
              <span className="title-icon"><Icon name="atom" /></span>
              <div><h2>Polymer structure</h2><p>Enter a polymer repeat unit for trained-model inference</p></div>
            </div>
            <span className="input-mode"><span className="connection-dot connected" /> STRUCTURE INPUT</span>
          </div>
          <form onSubmit={runPrediction}>
            <label htmlFor="smiles-input" className="sr-only">Polymer SMILES string</label>
            <div className="input-wrap">
              <span className="input-prefix">SMILES</span>
              <input id="smiles-input" value={smiles}               onChange={(event) => { setSmiles(event.target.value); setPrediction(null); setProcessResult(null); setError(""); clearSavedPrediction(); }} spellCheck="false" autoComplete="off" />
              <span className="input-valid"><Icon name="check" size={16} /> READY</span>
            </div>
            <div className="model-select-row">
              <label htmlFor="model-select">Prediction model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(event) => {
                  setSelectedModel(event.target.value);
                  setPrediction(null);
                  setProcessResult(null);
                  clearSavedPrediction();
                }}
              >
                {MODEL_CHOICES.map((model) => (
                  <option value={model.id} key={model.id}>{model.label}</option>
                ))}
              </select>
              <span>One selected algorithm predicts both permeability and selectivity.</span>
            </div>
            <div className="form-bottom">
              <div className="examples">
                <span>TRY AN EXAMPLE</span>
                {EXAMPLES.map((example) => (
                  <button type="button" className={smiles === example.smiles ? "example active" : "example"} key={example.label} onClick={() => chooseExample(example.smiles)}>{example.label}</button>
                ))}
              </div>
              <button className="predict-button" type="submit" disabled={isPredicting}>
                {isPredicting ? "Predicting…" : "Predict properties"} <Icon name="arrow" size={17} />
              </button>
            </div>
            {error && <div className="form-error" role="alert">{error}</div>}
          </form>
        </section>

        <div className="section-heading" id="analytics">
          <div><h2>Model prediction</h2><p>CO₂ permeability, selectivity, and RDKit descriptors from the trained model API</p></div>
          <div className="analytics-actions">
            {prediction && <button className="stage-continue-button" onClick={() => document.getElementById("stage2")?.scrollIntoView({ behavior: "smooth" })}>Continue to Stage 2 <Icon name="arrow" size={15} /></button>}
            <button className="export-button" onClick={downloadResult} disabled={!prediction}><Icon name="download" size={16} /> Export results</button>
          </div>
        </div>

        <section className="metrics-grid" aria-live="polite">
          <article className="metric-card">
            <div className="metric-top"><span>CO₂ PERMEABILITY</span></div>
            <div className="metric-number">{prediction ? formatValue(prediction.permeability) : "—"} <small>Barrer</small></div>
            <div className="metric-foot">{prediction ? `${prediction.selected_model?.label ?? prediction.models?.permeability} · trained model output` : "Run inference to calculate"}</div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>CO₂ / N₂ SELECTIVITY</span></div>
            <div className="metric-number">{prediction ? formatValue(prediction.selectivity, 2) : "—"} <small>selectivity</small></div>
            <div className="metric-foot">{prediction ? `${prediction.selected_model?.label ?? prediction.models?.selectivity} · trained model output` : "Run inference to calculate"}</div>
          </article>
          <article className="metric-card small-metric">
            <div className="metric-top"><span>STRUCTURE SIZE</span></div>
            <div className="metric-number">{structureProfile.heavyAtoms === undefined ? "—" : formatValue(structureProfile.heavyAtoms, 0)} <small>heavy atoms</small></div>
            <div className="metric-foot">{structureProfile.ringCount === undefined ? "RDKit model descriptors" : `${structureProfile.ringCount} rings · RDKit`}</div>
          </article>
        </section>

        <section className="charts-grid">
          <article className="panel performance-panel">
            <div className="panel-heading">
              <div><h3>Permeability · selectivity map</h3><p>Candidate position across the two target properties</p></div>
              <span className="panel-badge"><i /> CURRENT CANDIDATE</span>
            </div>
            {prediction ? <PerformanceMap prediction={prediction} /> : <div className="empty-chart"><Icon name="chart" size={24} /><span>Run a prediction to place this polymer on the map</span></div>}
          </article>
          <article className="panel descriptor-panel">
            <div className="panel-heading">
              <div><h3>Structure profile</h3><p>RDKit features returned by the trained-model API</p></div>
              <span className="chart-menu">•••</span>
            </div>
            <DescriptorChart descriptors={structureProfile} />
          </article>
        </section>

        <section className="charts-grid lower-grid">
          <article className="panel output-panel">
            <div className="panel-heading">
              <div><h3>Predicted property range</h3><p>Log-scaled position within notebook dataset bounds</p></div>
              <span className="panel-badge muted">MODEL OUTPUT</span>
            </div>
            {prediction ? <PredictionBars prediction={prediction} /> : <div className="empty-chart compact-empty"><Icon name="chart" size={22} /><span>Output ranges appear after prediction</span></div>}
          </article>
          <article className="panel method-panel">
            <div className="panel-heading">
            <div><h3>Model & dataset</h3><p>Reference context from your notebook</p></div>
            </div>
            <div className="method-stats">
              <div><span>Training polymers</span><strong>{modelInfo?.dataset_size ?? "—"}</strong></div>
              <div><span>{selectedModelInfo?.label ?? "Selected model"} · permeability test R²</span><strong>{selectedModelInfo?.test_metrics?.permeability ? formatValue(selectedModelInfo.test_metrics.permeability.r2, 3) : "—"}</strong></div>
              <div><span>{selectedModelInfo?.label ?? "Selected model"} · selectivity test R²</span><strong>{selectedModelInfo?.test_metrics?.selectivity ? formatValue(selectedModelInfo.test_metrics.selectivity.r2, 3) : "—"}</strong></div>
              <div><span>Feature schema</span><strong>{modelInfo?.feature_count ?? "—"} RDKit + Morgan</strong></div>
              <div><span>Targets</span><strong>CO₂ · CO₂/N₂</strong></div>
            </div>
            <div className="method-note"><Icon name="info" size={15} /><span>Dataset values are condition-averaged; operating temperature, pressure, and humidity are not inputs.</span></div>
          </article>
        </section>

        <Suspense fallback={<section className="stage-two-section" id="stage2">Loading Stage 2 process tools…</section>}>
          <MultiStageTwo
            prediction={activePrediction}
            design={processDesign}
            setDesign={setProcessDesign}
            result={processResult}
            setResult={setProcessResult}
          />
        </Suspense>
        <NotebookGraphGallery />

        <footer className="page-footer">
          <span><i /> MEMBRANE LAB <b>·</b> STRUCTURE-BASED SCREENING</span>
          <span>Use predictions as screening signals, not experimental guarantees.</span>
        </footer>
        </div>
      </main>
    </div>
  );
}
