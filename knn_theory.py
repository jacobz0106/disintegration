"""1-NN theory-validation experiment for the composed-function example.

Protocol:
  - 2-D composed function (dataGeneration.function2) on Lambda = [-1,1]^2
  - N in {100, 200, 500, 1000, 2000, 5000, 10000, 20000}
  - m in {5, 10, 20}
  - R = 100 repetitions per (N, m)
  - 1-nearest-neighbor classifier (n_neighbors = 1, NOT sqrt(N))
  - Fixed T = 200_000 auxiliary uniform sample (seed T_SEED), reused across
    all repetitions to isolate training-set variability
  - Fixed J = 50_000 diagnostic uniform sample (seed J_SEED), reused
  - Two-stage MC baseline draws its own N samples independently of the 1-NN
    training set (no reuse, no POF-Darts)

Outputs (raw + aggregated) live under Plots/results/latex/ so the plotting
script and manuscript pick them up in one place:
  - 1nn_theory_results.csv  (one row per (N, m, repetition))
  - 1nn_theory_summary.csv  (one row per (N, m) with aggregate stats)

Parallelised across repetitions.
"""

from __future__ import annotations

import os
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

from dataGeneration import function2
from event_estimation import equivalenceSpaceProbability, kde_estimation

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DOMAIN = [[-1.0, 1.0], [-1.0, 1.0]]
EVENT  = [[0.0, 0.8], [-0.7, 0.5]]

N_VALUES  = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
INTERVALS = [5, 10, 20]
REPEATS   = 100
T_TEST_SIZE = 200_000
J_DIAG_SIZE = 50_000
KDE_N       = 5_000

T_SEED = 12345
J_SEED = 67890

OUT_DIR      = "Plots/results/latex"
RAW_CSV      = f"{OUT_DIR}/1nn_theory_results.csv"
SUMMARY_CSV  = f"{OUT_DIR}/1nn_theory_summary.csv"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _uniform(n: int, rng: np.random.Generator) -> np.ndarray:
	return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in DOMAIN])


def _q_of(X: np.ndarray) -> np.ndarray:
	"""Vectorised composed function (function2 with defaults A=10, B=1)."""
	x = X[:, 0]
	y = X[:, 1]
	A, B = 10.0, 1.0
	return 1.0 + np.tanh(B * (y - A * x * (x - 0.5) * (x + 0.5)))


def _in_event(X: np.ndarray) -> np.ndarray:
	lo = np.array([e[0] for e in EVENT])
	hi = np.array([e[1] for e in EVENT])
	return np.all((X >= lo) & (X <= hi), axis=1)


def _categorize(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
	"""Vectorised replacement for dataGeneration.categorize_values.

	Returns labels in {0, 1, ..., len(cuts)} where label i means the value
	falls in the i-th equivalence class (below cuts[0], between cuts, above
	cuts[-1]). Matches the "value < threshold" convention of the reference.
	"""
	# np.searchsorted with side='right' gives, for each v, the number of cuts
	# strictly less than or equal to v -- exactly the label i.
	return np.searchsorted(cuts, values, side="right").astype(np.int64)


def _build_kde_and_cuts(m: int) -> tuple:
	cache = "data/function2/kde_source_n5000.csv"
	if os.path.exists(cache):
		f_values = pd.read_csv(cache, index_col=0)["f"].values
	else:
		rng_kde = np.random.default_rng(seed=0)
		X_kde = _uniform(KDE_N, rng_kde)
		f_values = _q_of(X_kde)
		os.makedirs(os.path.dirname(cache), exist_ok=True)
		pd.DataFrame({"f": f_values}).to_csv(cache)
	kde_cdf = kde_estimation(f_values.reshape(-1, 1))
	out_range = (float(f_values.min()), float(f_values.max()))
	cuts = np.linspace(out_range[0], out_range[1], m + 1)[1:-1]
	return kde_cdf, cuts


def _disintegration_estimate(labels: np.ndarray, in_event: np.ndarray,
                             kde_cdf, cuts: np.ndarray) -> float:
	total = 0.0
	for k in np.unique(labels):
		mask = labels == k
		denom = int(mask.sum())
		if denom == 0:
			continue
		cond = float(np.logical_and(mask, in_event).sum()) / denom
		total += float(equivalenceSpaceProbability(kde_cdf, cuts, int(k))) * cond
	return total


# ----------------------------------------------------------------------------
# Per-worker state (built once per process; reused across all reps it processes)
# ----------------------------------------------------------------------------

_KDE_CACHE: dict = {}
_CUTS_CACHE: dict = {}
_T_TEST_CACHE = None
_IN_EVENT_T_CACHE = None
_J_DIAG_CACHE = None
_Y_J_Q_CACHE = None


def _ensure_worker_state():
	global _KDE_CACHE, _CUTS_CACHE
	global _T_TEST_CACHE, _IN_EVENT_T_CACHE, _J_DIAG_CACHE, _Y_J_Q_CACHE
	if _T_TEST_CACHE is not None:
		return
	for m in INTERVALS:
		kde_cdf, cuts = _build_kde_and_cuts(m)
		_KDE_CACHE[m] = kde_cdf
		_CUTS_CACHE[m] = cuts
	rng_T = np.random.default_rng(seed=T_SEED)
	_T_TEST_CACHE = _uniform(T_TEST_SIZE, rng_T)
	_IN_EVENT_T_CACHE = _in_event(_T_TEST_CACHE)
	rng_J = np.random.default_rng(seed=J_SEED)
	_J_DIAG_CACHE = _uniform(J_DIAG_SIZE, rng_J)
	_Y_J_Q_CACHE = _q_of(_J_DIAG_CACHE)


def _process_repeat(r: int) -> list:
	_ensure_worker_state()
	X_T = _T_TEST_CACHE
	in_event_T = _IN_EVENT_T_CACHE
	X_J = _J_DIAG_CACHE
	y_J_q = _Y_J_Q_CACHE

	rng = np.random.default_rng(seed=1000 + r)
	rows: list = []
	for N in N_VALUES:
		# 1-NN training design: N iid uniform points + evaluate true QoI.
		X_train = _uniform(N, rng)
		y_train_q = _q_of(X_train)

		# Two-stage MC baseline: SEPARATE N iid uniform points (no reuse).
		X_mc = _uniform(N, rng)
		y_mc_q = _q_of(X_mc)
		in_event_mc = _in_event(X_mc)

		# One cKDTree per training set covers all m -- 1-NN neighbor indices
		# are the same regardless of the label mapping, so we can share the
		# tree and just relabel per m.
		tree = cKDTree(X_train)
		_, nn_idx_T = tree.query(X_T, k=1)
		_, nn_idx_J = tree.query(X_J, k=1)

		for m in INTERVALS:
			cuts = _CUTS_CACHE[m]
			kde_cdf = _KDE_CACHE[m]

			y_train_lbl = _categorize(y_train_q, cuts)
			lbl_T = y_train_lbl[nn_idx_T]
			nn_est = _disintegration_estimate(lbl_T, in_event_T, kde_cdf, cuts)

			y_mc_lbl = _categorize(y_mc_q, cuts)
			mc_est = _disintegration_estimate(y_mc_lbl, in_event_mc,
			                                  kde_cdf, cuts)

			y_J_lbl = _categorize(y_J_q, cuts)
			lbl_J = y_train_lbl[nn_idx_J]
			err = float((lbl_J != y_J_lbl).mean())

			rows.append({
				"N": int(N), "m": int(m), "repetition": int(r),
				"nn_probability_estimate":       nn_est,
				"two_stage_probability_estimate": mc_est,
				"nn_classification_error":       err,
			})
	return rows


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_experiment() -> tuple:
	os.makedirs(OUT_DIR, exist_ok=True)
	n_workers = max(1, cpu_count() - 1)
	tasks = list(range(REPEATS))
	print(f"[1nn] launching {REPEATS} repeats across {n_workers} workers  "
	      f"(N up to {N_VALUES[-1]}, T={T_TEST_SIZE}, J={J_DIAG_SIZE})",
	      flush=True)
	t0 = time.time()
	all_rows: list = []
	with Pool(processes=n_workers) as pool:
		for i, rows in enumerate(pool.imap_unordered(_process_repeat, tasks), 1):
			all_rows.extend(rows)
			elapsed = time.time() - t0
			print(f"[1nn] repeat {i:>3}/{REPEATS} done  elapsed {elapsed/60:5.1f} min",
			      flush=True)

	df = pd.DataFrame(all_rows).sort_values(["m", "N", "repetition"]).reset_index(drop=True)
	df.to_csv(RAW_CSV, index=False)
	print(f"[1nn] wrote {RAW_CSV}  ({len(df)} rows)", flush=True)

	summary = (
		df.groupby(["m", "N"], as_index=False)
		  .agg(
			  repetitions=("repetition", "count"),
			  classification_error_mean=("nn_classification_error", "mean"),
			  classification_error_sd=("nn_classification_error", "std"),
			  nn_estimate_mean=("nn_probability_estimate", "mean"),
			  nn_variance=("nn_probability_estimate", lambda s: float(s.var(ddof=1))),
			  mc_estimate_mean=("two_stage_probability_estimate", "mean"),
			  mc_variance=("two_stage_probability_estimate", lambda s: float(s.var(ddof=1))),
		  )
	)
	summary["variance_ratio"] = summary["nn_variance"] / summary["mc_variance"]
	summary = summary[[
		"N", "m", "repetitions",
		"classification_error_mean", "classification_error_sd",
		"nn_estimate_mean", "nn_variance",
		"mc_estimate_mean", "mc_variance",
		"variance_ratio",
	]].sort_values(["m", "N"]).reset_index(drop=True)
	summary.to_csv(SUMMARY_CSV, index=False)
	print(f"[1nn] wrote {SUMMARY_CSV}  ({len(summary)} rows)", flush=True)
	print(f"[1nn] total runtime {(time.time()-t0)/60:.1f} min", flush=True)
	return df, summary


if __name__ == "__main__":
	run_experiment()
