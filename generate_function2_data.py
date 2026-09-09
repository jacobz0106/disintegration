"""Generate missing POF training data caches for the function2 example.

Skips (n, interval, repeat) combos that already have a df_Train CSV, so it's
safe to re-run and pick up where a previous invocation left off. Writes both
df_Train (features + labels) and dQ_Train (gradients) so downstream NN and
PPSVMG runs can both skip the slow sampling step.

Run from the repo root:
    python generate_function2_data.py
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dataGeneration import SIP_Data_Multi, function2, Gradient_f2
from event_estimation import kde_estimation

DOMAINS = [[-1, 1], [-1, 1]]
CACHE_DIR = "data/function2"
KDE_SOURCE = f"{CACHE_DIR}/kde_source_n5000.csv"

SIZES = [100, 120, 140, 160, 180, 200, 250, 300, 400, 600, 800, 1000, 1400, 1600, 2000]
INTERVALS = [5, 10, 20]
REPEATS = 30


def critical_values_for(num_intervals: int) -> np.ndarray:
	"""Recreate the same quantization the batch runs use."""
	if os.path.exists(KDE_SOURCE):
		f_values = pd.read_csv(KDE_SOURCE, index_col=0)["f"].values
	else:
		# Fall back to a fresh 5000-point uniform draw (matches event_estimation.py).
		os.makedirs(CACHE_DIR, exist_ok=True)
		seed_state = np.random.get_state()
		np.random.seed(0)
		src = SIP_Data_Multi(function2, Gradient_f2, np.array([0.0]), len(DOMAINS), *DOMAINS)
		src.generate_Uniform(5000, Gradient=False)
		np.random.set_state(seed_state)
		f_values = np.asarray(src.df["f"])
		pd.DataFrame({"f": f_values}).to_csv(KDE_SOURCE)
	# Populate the KDE just to make sure it matches the batch flow, even
	# though this function only returns cut points.
	_ = kde_estimation(f_values.reshape(-1, 1))
	rng = (float(f_values.min()), float(f_values.max()))
	return np.linspace(rng[0], rng[1], num_intervals + 1)[1:-1]


def _needs(n: int, k: int, r: int) -> tuple[bool, bool]:
	df_path = f"{CACHE_DIR}/df_Train_size{n}_interval_{k}_repeat{r}_POF.csv"
	dq_path = f"{CACHE_DIR}/dQ_Train_size{n}_interval_{k}_repeat{r}_POF.csv"
	return (not os.path.exists(df_path), not os.path.exists(dq_path))


def generate_one(n: int, k: int, r: int, cuts: np.ndarray) -> None:
	need_df, need_dq = _needs(n, k, r)
	if not need_df and not need_dq:
		return
	data = SIP_Data_Multi(function2, Gradient_f2, cuts, len(DOMAINS), *DOMAINS)
	data.generate_POF(n=n, CONST_a=2, iniPoints=5, sampleCriteria="k-dDarts")
	df_path = f"{CACHE_DIR}/df_Train_size{n}_interval_{k}_repeat{r}_POF.csv"
	dq_path = f"{CACHE_DIR}/dQ_Train_size{n}_interval_{k}_repeat{r}_POF.csv"
	if need_df:
		data.df.to_csv(df_path)
	if need_dq:
		rows = data.Gradient
		rows = rows.values if hasattr(rows, "values") else rows
		dq_arr = np.vstack([np.asarray(row, dtype=float) for row in rows])
		pd.DataFrame(dq_arr).to_csv(dq_path, header=False, index=False)


def main() -> None:
	os.makedirs(CACHE_DIR, exist_ok=True)
	tasks = []
	for k in INTERVALS:
		for n in SIZES:
			for r in range(REPEATS):
				need_df, need_dq = _needs(n, k, r)
				if need_df or need_dq:
					tasks.append((n, k, r))
	print(f"[gen] {len(tasks)} POF samples to generate for function2")
	# Precompute cuts per interval (fast; caches KDE source too).
	cuts_by_k = {k: critical_values_for(k) for k in INTERVALS}
	# Order: small n first so early progress is fast, then large-n stragglers.
	tasks.sort(key=lambda t: (t[0], t[1], t[2]))
	t0 = time.time()
	for i, (n, k, r) in enumerate(tasks, 1):
		ts = time.time()
		generate_one(n, k, r, cuts_by_k[k])
		dt = time.time() - ts
		elapsed = time.time() - t0
		print(f"[gen] {i:>4}/{len(tasks)}  n={n:>4} k={k:>2} r={r:>2}  "
		      f"{dt:6.2f}s   total elapsed {elapsed/60:6.1f} min",
		      flush=True)
	print(f"[gen] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
	main()
