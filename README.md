# Membrane Lab

A React dashboard for screening polymer repeat-unit SMILES against the CO₂
permeability and CO₂/N₂ selectivity targets in `Carbon_Capture_ml.ipynb`.

## Project layout

- `src/` contains the React dashboard.
- `main.py` provides the FastAPI endpoints used by the React dashboard.
- `backend/` contains the Stage 1 feature pipeline and model-training entry point.
- `stage2_engine.py` contains the countercurrent 1D ODE solver, inverse design,
  and energy/cost calculations.
- `multistage_model.py` contains the multi-stage module, cascade, compression, and TEA screening model.
- `rigorous_simulation.py` contains a separate 1D solution-diffusion module solver, inverse area sizing, and two-stage permeate-recycle solver.
- `validate_literature.py` compares model results with literature values supplied by the user.
- `example_usage.py` demonstrates Stage 1 inference followed by the cascade model.
- `tests/javascript/` and `tests/python/` contain the Node.js and Python tests.
- `public/notebook-graphs/` contains the notebook plots displayed in the dashboard.
- `Carbon_Capture_ml.ipynb` is the research notebook; generated model artifacts
  and training data are kept in the ignored `artifacts/` directory.

## Run the dashboard

```bash
npm install
npm start
```

The dashboard only displays trained-model predictions. It does not invent a
fallback prediction when the model API or its trained artifact is unavailable.

## Train the model and start the API

Use the Python environment that has the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
npm run train:model
npm run api
```

In another terminal, run the React app:

```bash
npm start
```

`backend/train_model.py` downloads the same paired experimental CO₂ permeability
and CO₂/N₂ selectivity CSVs referenced in the notebook, applies its SMILES
cleanup, and creates the same 20 RDKit descriptors plus 128-bit Morgan
fingerprints. It trains Linear Regression (LR), Random Forest (RF), and XGBoost
(XG) models for both targets on the 85% train-plus-validation partition, and
reports per-model metrics on a held-out 15% test partition. Random seed 42 keeps
the split reproducible.

Training writes `artifacts/membrane_models.joblib`, `artifacts/clean_data.csv`,
and `artifacts/training_summary.json`. Retrain after upgrading this project so
the artifact contains all six selectable regressors. The bundle stores the
fitted models, ordered feature names, target/log-transform definitions, data
hashes, package versions, and held-out test metrics. The API validates the
bundle's schema and model feature counts before serving predictions.

The API binds to localhost on port 8000 by default. Override with `HOST` or
`PORT` only when intentionally exposing it beyond localhost.

## Stage 2 process application

The React navigation links Stage 1 polymer prediction to the Stage 2 process
panel. A successful Stage 1 prediction is saved in browser storage, then passed
to the process API as permeability and selectivity. Stage 2 offers design mode
(inverse area sizing for a target recovery) and simulation mode (calculated
performance at a specified area). Operating-input changes trigger a debounced
API request, and the returned ODE data updates the performance and pressure
plots. A recovery target is used only for design sizing; the reported recovery
is always calculated from the module component flows.

The Stage 2 model is a single countercurrent binary CO₂/N₂ membrane. Reported
purity and area are conditional on the supplied feed, membrane, pressures, and
selective-layer thickness; the UI does not force a desired purity or a fixed
area range.

`multistage_model.py` provides validated dataclass configurations, finite-area
`solve_ivp` integration, single-stage, second-stage permeate-recycle, and
three-stage series layouts, recycle convergence checks, compressor staging,
Peng–Robinson CO₂ compressibility, pretreatment allowances, and annualized
cost/sensitivity calculations. The defaults are explicit screening
assumptions—not a validated design model. The three-stage series layout
collects all stage permeates as product;
the two-stage layout recycles the second-stage permeate to the first stage.
Every stage outlet is included in the cascade material balance. The model
does not resolve hollow-fiber geometry, detailed crossflow/countercurrent
boundary-value profiles, concentration-polarization correlations, flue-gas
contaminant compositions, pressure-drop hydraulics, or vendor equipment
curves. Its TEA is a configurable screening estimate, not a NETL estimate.

The supplied validation request named four papers but did not give their
numerical results or complete operating bases. No literature values or
validation claims are fabricated. `validate_literature.py` reports model
results and relative errors only for reference values provided in a CSV. The
dashboard notebook-graph gallery contains saved notebook figures; they are not
recomputed by the cascade model.

Create a reference CSV with columns `benchmark`, `literature_purity_pct`,
`literature_recovery_pct`, `literature_area_m2`, `literature_sec_kwh_t`, and
`citation`, then run:

```bash
python validate_literature.py --references literature_benchmarks.csv
```

Run all JavaScript and Python tests with `npm test`. Python tests can also be
run independently with `npm run test:python`. To run the multi-stage tests
directly with pytest, use `python -m pytest -v tests/python/test_multistage.py`.
Run `python example_usage.py` for an end-to-end Stage 1-to-Stage 2 example.

## 1D membrane simulation engine

`stage2_engine.py` integrates local CO₂/N₂ solution-diffusion flux with
`solve_ivp` over 100 membrane-area segments and uses a `root` shooting solve for
the countercurrent boundary condition. The FastAPI route calls `solve_module`
for simulation mode and `find_required_area` (which uses Brent root finding)
for design mode. Component flows determine the reported purity, recovery,
stage cut, and energy/cost inputs; recovery is constrained to [0, 1].

`rigorous_simulation.py` remains a separate Python module with cocurrent,
countercurrent, and cascade solvers. It is not the engine used by the current
React Stage 2 panel.

The Stage 2 API accepts permeability in Barrer, thickness in µm, pressures in
bar, flow in kmol/h, and area in m². It calculates compressor and vacuum work,
specific energy, and a screening levelized cost using the solved product and
permeate flows. The cost and equipment equations are transparent screening
assumptions, not a validated equipment design or vendor estimate.

Run its validation tests with:

```bash
python -m pytest -v tests/python/test_simulation.py
```

## API

- `POST /api/simulate` accepts `P_CO2`, `alpha`, `thickness`, `P_feed`,
  `P_perm`, `F_feed`, `x_feed`, `mode` (`design` or `simulation`), and
  `target_value`. It returns calculated purity, recovery, area, specific energy,
  levelized cost, and arrays for the area and permeate-pressure plots.
- `GET /api/health` reports whether a validated model bundle is loaded, the
  available model choices and their held-out test metrics, dataset size, feature
  count, and training timestamp.
- `POST /api/predict` accepts `{"smiles":"*CC*","model":"xg"}` (model is `lr`,
  `rf`, or `xg`) and returns model-derived
  permeability (Barrer), CO₂/N₂ selectivity, RDKit descriptors, model names,
  and test metrics. Invalid SMILES returns HTTP 422. If training has not been
  run, the API returns HTTP 503 rather than making up a prediction.
Start the FastAPI backend and React app with `npm run api` and `npm start` in
separate terminals. The UI uses Vite's `/api` proxy to reach `127.0.0.1:8000`.
Run the API directly with `uvicorn main:app --reload`; train with
`python -m backend.train_model` when not using the npm scripts.

`joblib` model files are Python pickle artifacts. Only load bundles generated
locally by this training pipeline; do not load model files from untrusted
sources.
# Membrane-Separation-
