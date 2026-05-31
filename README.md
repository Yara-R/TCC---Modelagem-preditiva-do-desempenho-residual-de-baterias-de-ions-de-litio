# 🔋 Battery RUL — Predictive Modeling of Residual Performance of Lithium-Ion Batteries

> **Undergraduate Thesis (TCC) — Computer Science**
> Yara Rodrigues Inácio · Advisor: Diego de Freitas Bezerra

A data-centric framework for robust Remaining Useful Life (RUL) prediction of lithium-ion batteries, featuring a multi-domain OOD benchmark, physics-informed feature engineering, and a risk-aware evaluation metric — deployed as an interactive Streamlit dashboard.

---

## 📌 Overview

Standard battery prognostics models perform well in controlled settings but silently fail when deployed across different experimental domains — a phenomenon known as **domain shift**. Worse, conventional metrics like MAE and RMSE hide this failure by aggregating errors symmetrically, masking the most dangerous case: **overestimating remaining battery life**.

This project addresses these gaps through three contributions:

1. **Multi-domain OOD benchmark** — six heterogeneous public datasets unified under a canonical schema and standardized evaluation protocols (LOCO / LODO).
2. **Regime-Aware Ensemble** — three specialized XGBoost regressors routed by the *Shadow Impedance* proxy, a physics-informed estimate of internal resistance derived from raw voltage and current signals.
3. **Danger Bias metric** — a risk-aware evaluation indicator that quantifies the fraction of operationally dangerous overestimates, exposing risks invisible to R² and MAE.

**Key results:**
- Intra-domain: R² = 0.9662, MAE < 5% (LOCO protocol)
- Cross-domain: Danger Bias reaches **88.8%** on the EVERLASTING dataset (calendar aging), exposing critical OOD degradation

---

## 🗂️ Repository Structure

```
.
├── app.py                  # Streamlit dashboard (main entry point)
├── etl_pipeline.py         # ETL pipeline — dataset scanning and master index generation
├── configs/
│   └── config.yaml         # External data source paths (edit before running)
├── dataloaders/            # Dataset-specific parsers (.csv and .mat)
├── requirements.txt        # Python dependencies (pip)
├── environment.yml         # Conda environment specification
└── README.md
```

---

## 🧪 Datasets

The benchmark integrates six public datasets spanning diverse degradation regimes:

| Dataset | Chemistry | Application | Cells |
|---|---|---|---|
| [NASA Battery](https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset) | LiCoO₂ 18650 | Generic cycling | 52 |
| [Oxford Battery](https://doi.org/10.5287/bodleian:KO2kdmYGg) | NMC/Graphite pouch | Controlled lab | 8 |
| [Forklift](https://doi.org/10.17632/yz4pttm73n.2) | LFP prismatic 180 Ah | Industrial dynamic | 3 |
| [EVERLASTING](https://doi.org/10.4121/13804304.v1) | NCA/Si-Gr 18650 | Calendar aging | 2 |
| [eVTOL](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830) | NMC 18650 | Aeronautical high-rate | 22 |
| [Multi-Stage](https://doi.org/10.6084/m9.figshare.25975315) | NMC 21700 | Multi-condition lab | 279 |

> **Note:** The datasets are **not included** in this repository due to size and licensing constraints. Download each dataset from the links above and configure the paths in `configs/config.yaml` before running the pipeline.

---

## ⚙️ Setup

### Option 1 — pip

```bash
# Clone the repository
git clone https://github.com/Yara-R/TCC---Modelagem-preditiva-do-desempenho-residual-de-baterias-de-ions-de-litio.git
cd TCC---Modelagem-preditiva-do-desempenho-residual-de-baterias-de-ions-de-litio

# Install dependencies
pip install -r requirements.txt
```

### Option 2 — Conda

```bash
conda env create -f environment.yml
conda activate battery-rul
```

### Configure data paths

Edit `configs/config.yaml` to point to your local dataset directories:

```yaml
data_sources:
  nasa: /path/to/nasa_battery/
  oxford: /path/to/oxford_battery/
  forklift: /path/to/forklift/
  everlasting: /path/to/everlasting/
  evtol: /path/to/evtol/
  multistage: /path/to/multistage/
```

---

## 🚀 Running

### Step 1 — Build the master index

```bash
python etl_pipeline.py
```

This recursively scans configured directories, parses `.csv` and `.mat` files, and generates `master_battery_index.csv` with standardized metadata for each cell.

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

On first run, the Regime-Aware Ensemble trains automatically and is serialized to `moe_brain_v1.pkl`. Subsequent sessions load the cached model instantly.

---

## 📊 Dashboard Modules

| Tab | Description |
|---|---|
| **Degradation Curve** | Capacity vs. cycle with configurable end-of-life threshold (50–90% SOH) |
| **dQ/dV Analysis** | Incremental capacity derivative — electrochemical degradation signature |
| **Health Radar** | Five-dimensional health radar: capacity, resistance, RUL, stability, safety |
| **Second Life** | Tier-based (1–3) reuse allocation matrix with suggested applications |
| **Model Reliability** | Error histogram and real vs. predicted plot for the MoE ensemble |

The sidebar displays which specialist (aeronautical / industrial / stationary) is active for the selected cell, making the routing decision auditable.

---

## 🧠 Methodology

### Shadow Impedance

A physics-informed proxy for internal resistance derived from raw voltage and current signals, without requiring specialized electrochemical impedance spectroscopy (EIS) equipment:

```
Z_shadow = ΔV / ΔI
```

An exponential moving average (EMA) filter is applied before extraction to suppress sensor noise. Shadow Impedance serves a dual role: it enriches the feature space and acts as the routing signal for the ensemble.

### Regime-Aware Ensemble

Three independent XGBoost regressors are activated based on Shadow Impedance thresholds derived from training data quantile analysis:

| Regime | Z_shadow | Specialist |
|---|---|---|
| High-performance / aeronautical | < 0.06 Ω | E₁ |
| Industrial / standard cycling | 0.06 – 0.12 Ω | E₂ |
| Stationary / calendar aging | ≥ 0.12 Ω | E₃ |

### Danger Bias

A risk-asymmetric evaluation metric that quantifies the fraction of predictions that would lead an operator to delay maintenance past the true end of life:

```
DangerBias = (1/N) · Σ 𝟙[ŷᵢ > yᵢ + τ]
```

By construction: `SafeBias + Neutral + DangerBias = 1`. The study uses `τ = 0` (the most conservative threshold).

### Evaluation Protocols

- **LOCO** (Leave-One-Cell-Out): measures intra-domain cell-to-cell generalization
- **LODO** (Leave-One-Dataset-Out): measures cross-domain OOD generalization
- **Temporal Split**: evaluates performance in the end-of-life regime

---

## 📈 Results Summary

| Dataset | R² | MAE (%) | Danger Bias (%) |
|---|---|---|---|
| NASA | 0.98 | 3.63 | 21.1 |
| Forklift | 0.97 | 3.76 | 14.7 |
| Multi-Stage | 0.93 | 5.77 | 7.2 |
| eVTOL | −0.03 | 25.62 | 41.6 |
| Oxford | 0.00 | 25.15 | 45.0 |
| EVERLASTING | −2.60 | 47.24 | **88.8** |

Three structurally distinct OOD failure modes were identified: **mechanism shift** (EVERLASTING — calendar aging invisible to Z_shadow), **regime saturation** (eVTOL — extreme discharge rates below routing threshold), and **variance collapse** (Oxford — near-constant features under rigid constant-current protocol).

---

## 🔁 Reproducibility

All experiments are fully reproducible:

| Mechanism | Implementation |
|---|---|
| Random seeds | `np.random.seed(42)` and `random_state=42` throughout |
| Model persistence | Serialized via `joblib` (`moe_brain_v1.pkl`) |
| External configuration | Data paths in `configs/config.yaml` — no hardcoded absolute paths |
| Streamlit caching | `@st.cache_data` / `@st.cache_resource` prevent redundant reloads |
| Fixed hyperparameters | `n_estimators=150`, `max_depth=4`, `learning_rate=0.05` across all specialists |

---

## ⚠️ Limitations

- Second-Life Tier classification is **advisory only**. Deployment decisions must incorporate complementary safety tests (mechanical integrity checks, thermal abuse testing) not replaceable by predictive models alone.
- The routing thresholds are empirically derived from the training benchmark and may require recalibration for cell chemistries or operational regimes not represented in the six datasets.
- Calendar aging regimes (EVERLASTING) are currently outside the model's reliable generalization range. Features encoding time-temperature stress are required to address this failure mode.

---

## 🔭 Future Work

- Learned gating functions to replace hard-threshold routing
- Domain adaptation techniques to reduce Danger Bias under distribution shift
- Uncertainty quantification methods to support confidence-based abstention
- Validation of the dashboard artifact with real battery engineering users
- Extension of the benchmark to include solid-state and sodium-ion chemistries

---

## 📄 Reference

If you use this framework or benchmark in your research, please cite:

```
INÁCIO, Yara Rodrigues; BEZERRA, Diego de Freitas.
Modelagem preditiva do desempenho residual de baterias de íons de lítio
para reuso em aplicações de baixa demanda.
Trabalho de Conclusão de Curso — Ciência da Computação, 2026.
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).