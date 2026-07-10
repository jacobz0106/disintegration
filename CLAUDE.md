# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project runs on **Compute Canada (Cedar HPC cluster)** using a Gurobi-licensed Python environment. Local development uses the base Anaconda environment (`/Users/jacobzhu/opt/anaconda3`). The `requirements.txt` pins versions with `+computecanada` suffixes — do not strip these when editing for cluster use.

```bash
# Local: use system python
python script.py

# Cluster: activate virtualenv inside batch scripts
source ~/env_gurobi/bin/activate
```

## Running Experiments

All task scripts must be **run from `batchTask/`**, not from the repo root — relative paths like `../data/` and `../event_estimation.py` depend on this.

```bash
# Local single run (from batchTask/)
cd batchTask && python ../event_estimation.py <example> <model> <numIntervals> <sample_method>

# Example arguments:
#   example:       function2 | brusselator | lotka | SIR
#   model:         NN | PPSVMG
#   numIntervals:  5 | 10 | 20
#   sample_method: POF | Random

# Cluster: submit all jobs for an example
cd batchTask && bash sbatch_SIR.sh
```

Results are persisted in SQLite via `sqlitedict` at `../Results/<example>_<model>_<intervals>_<method>.sqlite`. Cached training data lives in `../data/<example>/`. Re-runs skip already-completed keys automatically.

## Architecture

### Core idea: Disintegration-based event probability estimation

Given a parameter space Λ (the domain), a quantity of interest Q(λ), and an event region E ⊆ Λ, the framework estimates P(λ ∈ E) by:
1. Building an empirical/KDE distribution of Q over Λ
2. Partitioning the Q range into `numIntervals` equivalence classes via `critical_values`
3. Training a classifier to label new λ points by which equivalence class Q(λ) falls in
4. Estimating P(λ ∈ E) = Σ P(class k) · P(λ ∈ E | class k)

### Key files

| File | Role |
|---|---|
| `event_estimation.py` | Main entry point. `main()` dispatches by example name, builds KDE, calls `accuracyComparison_parallel_repeat` |
| `dataGeneration.py` | `SIP_Data` / `SIP_Data_Multi` — sample generation (uniform or POF-Darts). Also defines all QoI functions: `function2`, `integral_3D` (Brusselator), elliptic PDE |
| `POFdarts.py` | Geometric adaptive sampler (POF-Darts algorithm) that concentrates samples near equivalence-class boundaries |
| `CBP.py` | Classifier implementations: `GMSVM_reduced` (gradient-augmented SVM with clustering), `ClassifierChainWrapper`, `OneVsRestWrapper` (wraps GMSVM_reduced for multi-class) |
| `SVM_Penalized.py` | Gradient-penalized SVM dual solver using SLSQP |
| `PSVM.py` | `MagKmeans` — class-balanced clustering used inside GMSVM to partition training data before fitting per-cluster SVMs |
| `SIR.py` | SIR epidemic model with Heun integrator. `quantity_interest` = (I(T+T₀) − I(T₀))/T |
| `lotkaVolterra.py` | 9-parameter Lotka-Volterra model |
| `make_plot.py` | Loads all SQLite results into a DataFrame for plotting |

### Two classifier modes

- **`NN`** — `MLPClassifier` with `GridSearchCV` (no gradient needed, parallelised across repeat/n)
- **`PPSVMG`** — `ClassifierChainWrapper(GMSVM_reduced)` — gradient-augmented, runs serially (max_workers=1)

### Adding a new example

1. Define `quantity_of_interest(lambda_vec)` and `gradientFunction(lambda_vec)` (returns array of partials)
2. Add `elif example == 'your_example':` block in `event_estimation.py:main()` setting `domains`, `event`, `quantity_of_interest`, `gradientFunction`
3. Add KDE-building logic in the `else:` branch (or let the non-SIR path handle it via 5000-sample uniform draw)
4. Create batch scripts in `batchTask/` following the naming pattern `{example}_{model}_{intervals}_{method}.sh`
5. Add data directory `data/{example}/` (created automatically on first run)

### SIR-specific notes

- Parameter space Λ₁ = [0, 0.35] × [0, 0.6]; β ~ Beta(12,30), γ ~ Beta(6,30); T₀=10, T=30
- KDE uses real Surge 2 empirical data (`data/SIR/empericalData/Surge2_QoI_30_days.csv`) with Scott's rule bandwidth — not the 5000-sample synthetic draw used by other examples
- Event option A (~18%): β ∈ [0.25, 0.35], γ ∈ [0.06, 0.14]

## Data / Results Layout

```
data/<example>/
    kde_source_n5000.csv          # cached uniform draw for KDE (non-SIR)
    df_Train_size{n}_interval_{k}_repeat{r}_{method}.csv
    dQ_Train_size{n}_interval_{k}_repeat{r}_{method}.csv  # gradients (PPSVMG only)

Results/
    <example>_<model>_<intervals>_<method>.sqlite   # SqliteDict of run results
```

SQLite keys follow: `{sample_method}_{example}_{model}_{n}_repeat_{r}_intervals_{k}`
