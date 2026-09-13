# Population Migration Forecast for NYC Census Tracts

A machine-learning pipeline that **forecasts population change in New York City census tracts from 2023 to 2033**. Models are trained on observed 2010→2020 tract-level population change against socio-economic and built-environment features, then applied to 2023 tract data to predict which tracts will grow or decline by 2033 — with probabilities, SHAP explanations, and choropleth maps.

## Overview

The project solves two related classification tasks:

| Task | Target | Classes |
|------|--------|---------|
| **Population Growth Classification** (binary) | `popchange10to20_binary` | `1` = population grew (change > 0), `0` = declined (change ≤ 0) |
| **Classification of Population Growth Levels** (3-class) | `growth_classify2` | `0` = decline (rate ≤ −0.038), `1` = moderate growth, `2` = high growth (rate ≥ 0.06) |

Both tasks are handled with **Random Forest** classifiers (benchmarked against KNN, Decision Tree, and SVM), trained with SMOTE oversampling and SelectKBest feature selection. The resulting 2023→2033 forecasts are explained with **SHAP** and visualized as choropleth maps.

## Repository Structure

| Path | Description |
|------|-------------|
| `predictive_model.ipynb` | Main training notebook: feature engineering, feature selection, SMOTE, model benchmarking, and saving of trained models |
| `prediction.ipynb` | Applies the saved models to 2023 tract data and produces the forecast (`Result.geojson`) |
| `SHAP prediction.ipynb` | SHAP explainability of both models (beeswarm, bar, heatmap, partial dependence, and waterfall plots) |
| `geometry/ct2010_3.geojson` | Training input: 2,158 NYC 2010 census tracts with population, socio-economic, and built-environment attributes |
| `Result.geojson` | Final output: 2,325 tracts (2020 boundaries) with 2023 data and forecast columns |
| `predictive model/random_forest_growth.pkl` | Trained binary Random Forest (Task 1) |
| `predictive model/forest_growth_type.pkl` | Trained 3-class Random Forest (Task 2) |
| `predictive model/svc_growth.pkl` | Binary SVM (rbf kernel), benchmarked but not used in the final forecast |
| `cleaned_sample_for_trainingModels.csv` | 100-row sample of the training property table (48 columns) |
| `cleaned_sample_for_prediction.csv` | 100-row sample of the prediction table, including model outputs |
| `Prediction result/` | Five choropleth maps of the 2023→2033 forecast (PNG) |

## Data

### Training data (`geometry/ct2010_3.geojson`)

2,158 NYC 2010 census tracts (Polygon geometry) with census identifiers (CT2010, GEOID, NTA, PUMA, borough codes) and:

- **Population**: 2010 and 2020 counts, 2010→2020 change and growth rate, population density
- **Socio-economic features**: education (`less than high school`), `commuting time`, `citizen ratio`, `unemployment`, `unrelated individual`, `per capita income`, `GINI index`, `gross rent`
- **Housing / built environment**: `%structure early built`, `median structure year`, `facility density`, `ComFAR`, `ResFAR`, `assesstot` (total assessed value), `NumFloors`, `Avr floor area`, `avr height`, `building density`
- **Trees**: `tree_count`, `good_tree_count`, plus engineered `tree density` (= tree_count / area_hectares) and `good tree density`

### Prediction data (`Result.geojson`)

2,325 features keyed to 2020 census tracts (`geoid`, `ct2020`, NTA/CDTA identifiers) with 2023 population data and the same feature set, plus the forecast columns:

- Binary task: `pop_change_forest`, `pop_change_forest_prob_1` (growth probability)
- 3-class task: `pop_change_type_forest`, `pop_change_type_forest_prob_2` (high-growth probability), `pop_change_type_forest_prob_0` (decline probability)

> **Note:** data sources are not cited in the repository (no NYC Open Data / ACS / PLUTO URLs appear in the notebooks). The intermediate files `ct2023_predict.geojson` (input to `prediction.ipynb`) and `2010_final.geojson` (output of `predictive_model.ipynb`) are also not included, so the full pipeline cannot be re-run from scratch without them.

## Methodology

1. **Feature engineering** — tree-density variables are derived from tree counts and tract area; binary and 3-class growth targets are constructed from the 2010→2020 population change.
2. **Feature selection** — `SelectKBest` (k=11) over 20 candidate features. The binary task uses 13 final features: `less than high school`, `pop_density`, `citizen ratio`, `unemployment`, `unrelated individual`, `GINI index`, `gross rent`, `assesstot`, `ResFAR`, `building density`, `tree density`, `good tree density`, `per capita income`. The 3-class task swaps `ComFAR` and `Avr floor area` in place of `ResFAR` and `tree density`.
3. **Class balancing** — SMOTE oversampling, followed by a VIF multicollinearity check.
4. **Train/test split** — 60/40 with `random_state=42`.

### Model benchmarking

| Model | Binary accuracy | 3-class accuracy |
|-------|----------------:|-----------------:|
| KNN (n=6) | benchmarked | benchmarked |
| Decision Tree (max_depth=3) | 0.402 | 0.506 |
| SVM (poly / linear / rbf) | 0.591 / 0.552 / 0.651 | ~0.53 (best) |
| **Random Forest (selected)** | **0.697** | **0.513** |

**Selected models:**

- **Binary task:** `RandomForestClassifier(n_estimators=500, max_depth=8, class_weight={0: 1.5, 1: 1})` — test accuracy **0.697** (growth-class F1 = 0.78). Top feature importances: `assesstot` (0.123), `pop_density` (0.122), `building density` (0.112).
- **3-class task:** `RandomForestClassifier(n_estimators=300, max_depth=6, class_weight={0: 1.5, 1: 2, 2: 2})` — test accuracy **0.513** (high-growth-class F1 = 0.61).

## Forecast Results (2023 → 2033)

| Task | Predicted distribution |
|------|------------------------|
| Binary growth | **1,685** tracts predicted to grow, **640** predicted to decline |
| Growth level | **431** decline, **1,236** moderate growth, **658** high growth |

### Forecast maps

![Binary growth/decline classification](Prediction%20result/23to33popchange.png)

![Three-level growth category](Prediction%20result/23to33pop_change_category.png)

![Population growth probability for binary classification](Prediction%20result/23to33population%20growth%20Probability.png)

Additional maps: [high-growth probability](Prediction%20result/23to33pop_increase_probability.png) · [decline probability](Prediction%20result/23to33pop_decline_probability.png)

## Model Explainability (SHAP)

`SHAP prediction.ipynb` explains both models using `shap.Explainer` with a 500-row background sample:

- **Global importance** — beeswarm, bar, clustered bar (hierarchical clustering cutoff 0.8), and heatmap plots for the binary model and per class (0 and 2) for the 3-class model
- **Feature effects** — partial dependence and scatter plots for key drivers such as building density and unemployment, plus a sub-analysis restricted to tracts with `assesstot < 1.5e6`
- **Individual predictions** — waterfall plots for the most / least / median-confidence tracts

## Usage

**Required Python libraries:** `pandas`, `geopandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn` (SMOTE), `statsmodels` (VIF), `joblib`, `shap`, `shapely`, `libpysal`.

Run the notebooks in order:

1. `predictive_model.ipynb` — trains and benchmarks the models, saves the two Random Forest classifiers to `predictive model/`
2. `prediction.ipynb` — applies the saved models to 2023 tract data and writes the forecast (`Result.geojson`)
3. `SHAP prediction.ipynb` — generates the SHAP explanations from `Result.geojson` and the saved models

> **Note:** step 2 requires `ct2023_predict.geojson` (2023 tract data), which is not included in this repository; step 1 writes `2010_final.geojson` (not included). The pre-trained models and the final `Result.geojson` are provided, so steps 2–3 and the result maps can be inspected without re-training.

## Limitations & Notes

- The 3-class task achieves only ~0.51 test accuracy; its forecasts should be interpreted cautiously.
- `svc_growth.pkl` was trained and benchmarked but is not used in the final forecast.
- Data provenance (NYC Open Data, ACS, PLUTO, tree surveys) is not documented in the repository.
- The five maps in `Prediction result/` are included as images only; the code that generated them is not in the notebooks.
