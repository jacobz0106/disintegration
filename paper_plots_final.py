"""Final result figures for main.tex (single shared legend per figure).

Produces the four PDFs (and matching PNGs) that main.tex references:

    Plots/results/latex/knn_theory_validation.{pdf,png}
    Plots/results/latex/event_probability_boxplots.{pdf,png}
    Plots/results/latex/practical_variance_ratio.{pdf,png}
    Plots/results/latex/practical_rmse.{pdf,png}

Layout uses matplotlib subfigures so that each PDF contains one
figure-level legend, not one legend per panel. Consistent method aesthetics
are shared across every practical figure via _METHOD_STYLE below.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# Reuse the aggregation logic already written for the paper.
from paper_plots_latex import (
	load_practical_long,
	load_naive_long,
	compute_benchmarks,
	build_practical_summary,
	OUT_DIR,
	EXAMPLES,
	INTERVALS,
	METHOD_ORDER,
	METHOD_LABEL,
	COLOR_MAP,
	LINETYPE_MAP,
	SHAPE_MAP,
	BOX_COLOR_MAP,
)

_1NN_SUMMARY_CSV = f"{OUT_DIR}/1nn_theory_summary.csv"
_1NN_RAW_CSV     = f"{OUT_DIR}/1nn_theory_results.csv"


def load_1nn_summary() -> pd.DataFrame:
	if not os.path.exists(_1NN_SUMMARY_CSV):
		raise FileNotFoundError(
			f"missing 1-NN summary: {_1NN_SUMMARY_CSV}. "
			"Run `python knn_theory.py` first."
		)
	return pd.read_csv(_1NN_SUMMARY_CSV)


def load_1nn_raw() -> pd.DataFrame:
	if not os.path.exists(_1NN_RAW_CSV):
		raise FileNotFoundError(
			f"missing 1-NN raw results: {_1NN_RAW_CSV}. "
			"Run `python knn_theory.py` first."
		)
	return pd.read_csv(_1NN_RAW_CSV)


def compute_1nn_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate the R=100 replicates into the three curves the manuscript
	tracks: mean misclassification error, its empirical second moment, and
	the classifier / two-stage-MC variance ratio.

	The variance ratio uses matched (N, m) repetition sets and the ordinary
	unbiased sample variance (ddof=1) in both numerator and denominator.
	"""
	def _agg(sub: pd.DataFrame) -> pd.Series:
		eps = sub["nn_classification_error"].values
		p_nn = sub["nn_probability_estimate"].values
		p_mc = sub["two_stage_probability_estimate"].values
		return pd.Series({
			"repetitions":     len(sub),
			"mean_eps":        float(np.mean(eps)),
			"M_eps2":          float(np.mean(eps * eps)),
			"var_p_nn":        float(np.var(p_nn, ddof=1)) if len(p_nn) > 1 else float("nan"),
			"var_p_mc":        float(np.var(p_mc, ddof=1)) if len(p_mc) > 1 else float("nan"),
		})

	agg = (
		raw.groupby(["N", "m"], as_index=False)
		   .apply(_agg)
		   .reset_index(drop=True)
	)
	agg["variance_ratio"] = agg["var_p_nn"] / agg["var_p_mc"]
	return agg.sort_values(["m", "N"]).reset_index(drop=True)


def _fit_loglog_slope(x: np.ndarray, y: np.ndarray, n_min: int = 2000):
	"""Return (slope, intercept, n_used) from log-log fit for x >= n_min."""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	mask = (x >= n_min) & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
	if mask.sum() < 2:
		return (float("nan"), float("nan"), int(mask.sum()))
	slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
	return (float(slope), float(intercept), int(mask.sum()))

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

# Shared paper-style rcParams so line widths/fonts survive LaTeX rescaling.
mpl.rcParams.update({
	"figure.dpi": 120,
	"savefig.dpi": 300,
	"savefig.bbox": "tight",
	"pdf.fonttype": 42,     # embed fonts
	"ps.fonttype":  42,
	"font.family": "DejaVu Sans",
	"font.size": 10,
	"axes.labelsize": 10,
	"axes.titlesize": 10,
	"xtick.labelsize": 9,
	"ytick.labelsize": 9,
	"legend.fontsize": 9,
	"lines.linewidth": 1.4,
	"lines.markersize": 4.5,
	"axes.linewidth": 0.7,
	"grid.linewidth": 0.4,
	"grid.color": "#DDDDDD",
})

# matplotlib linestyle strings (plotnine's "dashed" etc. -> matplotlib codes)
_MPL_LS = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

# Marker mapping (plotnine string -> matplotlib code).
_MPL_MARKER = {"o": "o", "^": "^", "s": "s"}

PROBLEM_LABEL = {
	"function2":   "Composed function",
	"brusselator": "Brusselator",
	"lotka":       "Lotka-Volterra",
	"SIR":         "SIR",
}
PANEL_LETTER = {name: chr(ord("a") + i) for i, name in enumerate(EXAMPLES)}
N_TICKS = [100, 200, 400, 1000, 2000]


def _method_style(method: str) -> dict:
	"""Kwargs for ax.plot() so every figure uses identical method aesthetics."""
	return dict(
		color=COLOR_MAP[method],
		linestyle=_MPL_LS.get(LINETYPE_MAP[method], "-"),
		marker=_MPL_MARKER.get(SHAPE_MAP[method], "o"),
		markerfacecolor=COLOR_MAP[method],
		markeredgecolor=COLOR_MAP[method],
	)


def _legend_handles(methods: Sequence[str]) -> Tuple[List, List[str]]:
	handles, labels = [], []
	for m in methods:
		st = _method_style(m)
		handles.append(Line2D([0], [0], **st))
		labels.append(METHOD_LABEL[m])
	return handles, labels


def _box_legend_handles(methods: Sequence[str]) -> Tuple[List, List[str]]:
	handles, labels = [], []
	for m in methods:
		handles.append(Patch(facecolor=BOX_COLOR_MAP[m], edgecolor=BOX_COLOR_MAP[m],
		                     alpha=0.75))
		labels.append(METHOD_LABEL[m])
	return handles, labels


def _save(fig, base: str) -> str:
	pdf = f"{OUT_DIR}/{base}.pdf"
	png = f"{OUT_DIR}/{base}.png"
	fig.savefig(pdf)
	fig.savefig(png)
	plt.close(fig)
	print(f"  wrote {pdf}, {png}")
	return pdf


def _apply_line_axes(ax, xlabel: str = "", ylabel: str = "",
                     yscale: str = "log", title: str = "") -> None:
	ax.set_xscale("log")
	if yscale == "log":
		ax.set_yscale("log")
	ax.set_xticks(N_TICKS)
	ax.set_xticklabels([str(x) for x in N_TICKS])
	ax.grid(True, which="both", alpha=0.35)
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title, loc="left", pad=4)


# ----------------------------------------------------------------------------
# 1. K-NN theory validation
# ----------------------------------------------------------------------------

# Color triple for the three m values across theory figures.
_M_COLORS = {5: "#440154", 10: "#21918C", 20: "#FDE725"}   # viridis triple
_LARGE_N_MIN = 2000

# Wider N ticks needed since 1-NN experiment goes to N=20000.
_THEORY_N_TICKS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def _apply_theory_axes(ax, xlabel: str = "", ylabel: str = "",
                       yscale: str = "log", title: str = "") -> None:
	ax.set_xscale("log")
	if yscale == "log":
		ax.set_yscale("log")
	ax.set_xticks(_THEORY_N_TICKS)
	ax.set_xticklabels([str(x) for x in _THEORY_N_TICKS], rotation=0)
	ax.grid(True, which="both", alpha=0.35)
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if title:
		ax.set_title(title, loc="left", pad=4)


def _plot_mean_error_on(ax, agg: pd.DataFrame, *, panel_label: str = "") -> None:
	for m in INTERVALS:
		sub = agg[agg["m"] == m].sort_values("N")
		if sub.empty:
			continue
		ax.plot(sub["N"], sub["mean_eps"],
		        color=_M_COLORS[m], marker="o", linewidth=1.3)
	# N^{-1/2} slope guide, anchored at the mean_eps of m = 10 at the smallest
	# N so it sits alongside the data. On a linear y-axis the guide is just a
	# curve, not a straight line.
	Ns = np.array(sorted(agg["N"].unique()), dtype=float)
	if len(Ns) >= 2:
		ref_m = 10
		row0 = agg[(agg["m"] == ref_m) & (agg["N"] == int(Ns[0]))]
		anchor = float(row0["mean_eps"].iloc[0]) if not row0.empty else \
			float(agg[agg["N"] == int(Ns[0])]["mean_eps"].max())
		y = anchor * (Ns[0] / Ns) ** 0.5
		ax.plot(Ns, y, color="black", linestyle="--", linewidth=1.0,
		        label=r"$N^{-1/2}$ reference")
	_apply_theory_axes(ax, "Number of expensive function evaluations, N",
	                   "Mean 1-NN misclassification error",
	                   yscale="linear", title=panel_label)
	ax.set_ylim(0.0, 1.0)


def _plot_second_moment_on(ax, agg: pd.DataFrame, *, panel_label: str = "") -> None:
	for m in INTERVALS:
		sub = agg[agg["m"] == m].sort_values("N")
		if sub.empty:
			continue
		ax.plot(sub["N"], sub["M_eps2"],
		        color=_M_COLORS[m], marker="o", linewidth=1.3)
	# N^{-1} slope guide, anchored at the m = 10 second moment at the
	# smallest N so it can be visually compared against the data slope.
	Ns = np.array(sorted(agg["N"].unique()), dtype=float)
	if len(Ns) >= 2:
		ref_m = 10
		row0 = agg[(agg["m"] == ref_m) & (agg["N"] == int(Ns[0]))]
		anchor = float(row0["M_eps2"].iloc[0]) if not row0.empty else \
			float(agg[agg["N"] == int(Ns[0])]["M_eps2"].max())
		y = anchor * (Ns[0] / Ns)
		ax.plot(Ns, y, color="black", linestyle="--", linewidth=1.0,
		        label=r"$N^{-1}$ reference")
	_apply_theory_axes(ax, "Number of expensive function evaluations, N",
	                   r"$\mathbb{E}[(\varepsilon_N^{\mathrm{1NN}})^2]$",
	                   yscale="log", title=panel_label)


def _plot_variance_ratio_on(ax, agg: pd.DataFrame, *, panel_label: str = "") -> None:
	for m in INTERVALS:
		sub = agg[agg["m"] == m].sort_values("N")
		if sub.empty:
			continue
		ax.plot(sub["N"], sub["variance_ratio"],
		        color=_M_COLORS[m], marker="o", linewidth=1.3)
	ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
	_apply_theory_axes(ax, "Number of expensive function evaluations, N",
	                   "Variance ratio (1-NN / two-stage MC)",
	                   yscale="log", title=panel_label)


def _theory_legend_handles():
	"""Shared legend across all three theory panels."""
	handles = [Line2D([0], [0], color=_M_COLORS[m], marker="o", linewidth=1.3,
	                  label=f"m = {m}") for m in INTERVALS]
	handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
	                      label="reference slope"))
	return handles


def plot_1nn_mean_error(agg: pd.DataFrame) -> str:
	fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
	_plot_mean_error_on(ax, agg, panel_label="(a) Mean misclassification error")
	handles = [Line2D([0], [0], color=_M_COLORS[m], marker="o", linewidth=1.3,
	                  label=f"m = {m}") for m in INTERVALS]
	handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
	                      label=r"$N^{-1/2}$ reference"))
	fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
	return _save(fig, "1nn_theory_misclassification_error")


def plot_1nn_error_second_moment(agg: pd.DataFrame) -> str:
	fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
	_plot_second_moment_on(ax, agg, panel_label="(b) Second moment of misclassification error")
	handles = [Line2D([0], [0], color=_M_COLORS[m], marker="o", linewidth=1.3,
	                  label=f"m = {m}") for m in INTERVALS]
	handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
	                      label=r"$N^{-1}$ reference"))
	fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
	return _save(fig, "1nn_theory_error_second_moment")


def plot_1nn_variance_ratio_only(agg: pd.DataFrame) -> str:
	fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
	_plot_variance_ratio_on(ax, agg, panel_label="(c) Variance ratio")
	handles = [Line2D([0], [0], color=_M_COLORS[m], marker="o", linewidth=1.3,
	                  label=f"m = {m}") for m in INTERVALS]
	fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
	return _save(fig, "1nn_theory_variance_ratio")


def plot_1nn_theory_validation(agg: pd.DataFrame) -> str:
	"""Three-panel composite required by main.tex: mean error, second moment,
	variance ratio -- with one shared bottom legend."""
	fig = plt.figure(figsize=(14.5, 4.6), constrained_layout=True)
	gs = fig.add_gridspec(1, 3, wspace=0.22)
	axA = fig.add_subplot(gs[0])
	axB = fig.add_subplot(gs[1])
	axC = fig.add_subplot(gs[2])
	_plot_mean_error_on(axA, agg, panel_label="(a) Mean misclassification error")
	_plot_second_moment_on(axB, agg,
	                       panel_label="(b) Second moment of misclassification error")
	_plot_variance_ratio_on(axC, agg, panel_label="(c) Variance ratio")

	handles = [Line2D([0], [0], color=_M_COLORS[m], marker="o", linewidth=1.3,
	                  label=f"m = {m}") for m in INTERVALS]
	handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
	                      label=r"$N^{-1/2}$ or $N^{-1}$ reference slope"))
	fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
	return _save(fig, "1nn_theory_validation")


def plot_1nn_variance_rate(summary: pd.DataFrame) -> tuple:
	"""Raw variance vs N on log-log axes for 1-NN and two-stage MC.

	Also returns fitted large-N slopes per m (both estimators) so the driver
	can print them.
	"""
	fig = plt.figure(figsize=(11.5, 4.6), constrained_layout=True)
	ax = fig.add_subplot(111)

	slopes: dict = {}
	for m in INTERVALS:
		sub = summary[summary["m"] == m].sort_values("N")
		if sub.empty:
			continue
		N = sub["N"].values
		v_nn = sub["nn_variance"].values
		v_mc = sub["mc_variance"].values

		ax.plot(N, v_nn, color=_M_COLORS[m], marker="o", linewidth=1.3,
		        linestyle="-",  label=f"1-NN, m = {m}")
		ax.plot(N, v_mc, color=_M_COLORS[m], marker="s", linewidth=1.3,
		        linestyle="--", markerfacecolor="white",
		        label=f"Two-stage MC, m = {m}")

		s_nn = _fit_loglog_slope(N, v_nn, n_min=_LARGE_N_MIN)
		s_mc = _fit_loglog_slope(N, v_mc, n_min=_LARGE_N_MIN)
		slopes[m] = {"nn": s_nn, "mc": s_mc}

	_apply_theory_axes(ax, "Number of expensive function evaluations, N",
	                   "Sample variance of P(E) estimator",
	                   yscale="log")

	# One shared legend at the bottom.
	handles = []
	for m in INTERVALS:
		handles.append(Line2D([0], [0], color=_M_COLORS[m], marker="o",
		                      linewidth=1.3, linestyle="-",
		                      label=f"1-NN, m = {m}"))
	for m in INTERVALS:
		handles.append(Line2D([0], [0], color=_M_COLORS[m], marker="s",
		                      linewidth=1.3, linestyle="--",
		                      markerfacecolor="white",
		                      label=f"Two-stage MC, m = {m}"))
	fig.legend(handles=handles, loc="outside lower center", ncol=3,
	           frameon=False, handlelength=2.5, columnspacing=1.4)
	pdf = _save(fig, "1nn_theory_variance_rate")
	return pdf, slopes


# ----------------------------------------------------------------------------
# Shared helpers for the practical 2x2-problems x 3-m layout
# ----------------------------------------------------------------------------

def _practical_layout(figsize=(13.5, 10.5)):
	"""Outer 2x2 problem subfigures; each contains 1x3 m subplots.

	Returns (fig, axes_by_problem, subfig_by_problem) so callers can attach
	subfigure-level x-axis labels once per subfigure (avoids colliding labels
	on each inner axis).
	"""
	fig = plt.figure(figsize=figsize, constrained_layout=True)
	outer = fig.subfigures(2, 2, wspace=0.04, hspace=0.12)
	axes_by_problem: Dict[str, List[plt.Axes]] = {}
	subfig_by_problem: Dict[str, mpl.figure.SubFigure] = {}
	for subfig, prob in zip(outer.flat, EXAMPLES):
		# Place the panel label above the entire subfigure content using
		# subfig.text at y slightly above 1 so it sits over the reserved
		# top strip left by ax.set_title(pad=...) below. constrained_layout
		# will grow the subfigure top margin to accommodate.
		subfig.suptitle(
			f"({PANEL_LETTER[prob]}) {PROBLEM_LABEL[prob]}",
			fontsize=11, ha="left", x=0.01, y=1.02, fontweight="bold",
		)
		inner_axes = subfig.subplots(1, 3, sharey=True)
		axes_by_problem[prob] = list(inner_axes)
		subfig_by_problem[prob] = subfig
	return fig, axes_by_problem, subfig_by_problem


def _annotate_m(axes: Iterable[plt.Axes]) -> None:
	# Larger pad so the "m = 5" title of the leftmost axis does not collide
	# with the subfigure-level panel label sitting above at y=1.02.
	for ax, m in zip(axes, INTERVALS):
		ax.set_title(f"m = {m}", fontsize=9, pad=10)


# ----------------------------------------------------------------------------
# 2. Event-probability boxplots
# ----------------------------------------------------------------------------

def plot_event_probability_boxplots(cls_long: pd.DataFrame,
                                    naive_long: pd.DataFrame,
                                    benchmarks: pd.DataFrame) -> str:
	long = pd.concat([cls_long, naive_long], ignore_index=True, sort=False)
	fig, axes_by_prob, subfig_by_prob = _practical_layout(figsize=(15.0, 12.0))

	# per-N x-position with per-method dodge inside the position.
	n_order = sorted(long["n"].unique())
	n_index = {n: i for i, n in enumerate(n_order)}
	n_methods = len(METHOD_ORDER)
	# Width of the total dodge cluster at each N tick.
	cluster_w = 0.85
	method_w  = cluster_w / n_methods
	offsets = {
		m: (i - (n_methods - 1) / 2) * method_w
		for i, m in enumerate(METHOD_ORDER)
	}

	for problem, axes in axes_by_prob.items():
		d = long[long["problem"] == problem]
		bench_map = dict(zip(
			benchmarks[benchmarks["problem"] == problem]["m"],
			benchmarks[benchmarks["problem"] == problem]["benchmark"],
		))
		for ax, m in zip(axes, INTERVALS):
			dm = d[d["m"] == m]
			for method in METHOD_ORDER:
				vals_by_n = []
				positions = []
				for n in n_order:
					vals = dm[(dm["method"] == method) & (dm["n"] == n)]["estimation"].values
					if vals.size == 0:
						continue
					vals_by_n.append(vals)
					positions.append(n_index[n] + offsets[method])
				if not vals_by_n:
					continue
				box = ax.boxplot(
					vals_by_n, positions=positions, widths=method_w * 0.85,
					patch_artist=True, showfliers=True, manage_ticks=False,
					flierprops=dict(marker="o", markersize=1.4,
					                markerfacecolor=BOX_COLOR_MAP[method],
					                markeredgecolor=BOX_COLOR_MAP[method],
					                alpha=0.7),
					medianprops=dict(color="black", linewidth=0.6),
					boxprops=dict(facecolor=BOX_COLOR_MAP[method],
					              edgecolor=BOX_COLOR_MAP[method],
					              alpha=0.75, linewidth=0.4),
					whiskerprops=dict(color=BOX_COLOR_MAP[method], linewidth=0.5),
					capprops=dict(color=BOX_COLOR_MAP[method], linewidth=0.5),
				)
			bench = bench_map.get(m, np.nan)
			if np.isfinite(bench):
				ax.axhline(bench, color="black", linestyle=":", linewidth=0.8)
			ax.set_xticks(list(range(len(n_order))))
			ax.set_xticklabels([str(n) for n in n_order], rotation=45, ha="right",
			                   fontsize=7)
			ax.grid(True, axis="y", alpha=0.3)
		_annotate_m(axes)
		subfig_by_prob[problem].supylabel("Estimated event probability",
		                                  fontsize=9)

	# x-labels ONCE per subfigure on the bottom row
	for problem in EXAMPLES[-2:]:  # bottom row = lotka, SIR
		subfig_by_prob[problem].supxlabel(
			"Number of expensive function evaluations, N", fontsize=9)

	handles, labels = _box_legend_handles(METHOD_ORDER)
	handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=0.8))
	labels.append("Benchmark (two-stage MC at N=2000)")
	fig.legend(handles=handles, labels=labels, loc="outside lower center",
	           ncol=6, frameon=False,
	           handlelength=1.6, columnspacing=1.3)
	return _save(fig, "event_probability_boxplots")


# ----------------------------------------------------------------------------
# 3. Variance-ratio, 2x2 problems x 3 m, classifier methods only
# ----------------------------------------------------------------------------

def plot_practical_variance_ratio(agg: pd.DataFrame) -> str:
	fig, axes_by_prob, subfig_by_prob = _practical_layout(figsize=(13.5, 10.0))
	classifier_methods = METHOD_ORDER[1:]  # exclude Naive: ratio == 1

	for problem, axes in axes_by_prob.items():
		d = agg[(agg["problem"] == problem) & (agg["method"] != "Naive")]
		for ax, m in zip(axes, INTERVALS):
			ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, zorder=1)
			for method in classifier_methods:
				sub = d[(d["method"] == method) & (d["m"] == m)].sort_values("n")
				if sub.empty:
					continue
				vr = sub["variance_ratio_to_naive"].values.astype(float)
				vr[vr <= 0] = np.nan
				ax.plot(sub["n"].values, vr, **_method_style(method), zorder=2)
			_apply_line_axes(ax)
		_annotate_m(axes)
		subfig_by_prob[problem].supylabel(
			"Var(classifier) / Var(two-stage MC)", fontsize=9)

	for problem in EXAMPLES[-2:]:
		subfig_by_prob[problem].supxlabel(
			"Number of expensive function evaluations, N", fontsize=9)

	handles, labels = _legend_handles(classifier_methods)
	fig.legend(handles=handles, labels=labels, loc="outside lower center",
	           ncol=4, frameon=False,
	           handlelength=2.4, columnspacing=1.6)
	return _save(fig, "practical_variance_ratio")


# ----------------------------------------------------------------------------
# 4. RMSE, 2x2 problems x 3 m, all 5 methods including Naive
# ----------------------------------------------------------------------------

def plot_practical_accuracy(agg: pd.DataFrame) -> str:
	"""Mean multiclass test accuracy vs N, 2x2 problems x 3 m, shared legend.

	Uses the ordinary (overall) accuracy stored in the SQLite results because
	balanced accuracy is not recoverable from the aggregate scalars saved by
	event_estimation.py -- refitting all 21 600 classifiers was skipped per
	user direction.
	"""
	fig, axes_by_prob, subfig_by_prob = _practical_layout(figsize=(13.5, 10.0))
	classifier_methods = METHOD_ORDER[1:]  # exclude Naive: not a classifier

	for problem, axes in axes_by_prob.items():
		d = agg[(agg["problem"] == problem) & (agg["method"] != "Naive")]
		for ax, m in zip(axes, INTERVALS):
			for method in classifier_methods:
				sub = d[(d["method"] == method) & (d["m"] == m)].sort_values("n")
				if sub.empty:
					continue
				y = sub["classifier_accuracy_mean"].values.astype(float)
				ax.plot(sub["n"].values, y, **_method_style(method))
			# Bounded metric: use a common 0-1 scale for comparability.
			ax.set_ylim(0.0, 1.0)
			_apply_line_axes(ax, yscale="linear")
		_annotate_m(axes)
		subfig_by_prob[problem].supylabel(
			"Mean 5000-point test accuracy", fontsize=9)

	for problem in EXAMPLES[-2:]:
		subfig_by_prob[problem].supxlabel(
			"Number of expensive function evaluations, N", fontsize=9)

	handles, labels = _legend_handles(classifier_methods)
	fig.legend(handles=handles, labels=labels, loc="outside lower center",
	           ncol=4, frameon=False,
	           handlelength=2.4, columnspacing=1.6)
	return _save(fig, "practical_accuracy")


def plot_practical_rmse(agg: pd.DataFrame) -> str:
	fig, axes_by_prob, subfig_by_prob = _practical_layout(figsize=(13.5, 10.0))

	for problem, axes in axes_by_prob.items():
		d = agg[agg["problem"] == problem]
		for ax, m in zip(axes, INTERVALS):
			for method in METHOD_ORDER:
				sub = d[(d["method"] == method) & (d["m"] == m)].sort_values("n")
				if sub.empty:
					continue
				y = sub["rmse"].values.astype(float)
				y[y <= 0] = np.nan
				ax.plot(sub["n"].values, y, **_method_style(method))
			_apply_line_axes(ax)
		_annotate_m(axes)
		subfig_by_prob[problem].supylabel("RMSE", fontsize=9)

	for problem in EXAMPLES[-2:]:
		subfig_by_prob[problem].supxlabel(
			"Number of expensive function evaluations, N", fontsize=9)

	handles, labels = _legend_handles(METHOD_ORDER)
	fig.legend(handles=handles, labels=labels, loc="outside lower center",
	           ncol=5, frameon=False,
	           handlelength=2.4, columnspacing=1.6)
	return _save(fig, "practical_rmse")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
	os.makedirs(OUT_DIR, exist_ok=True)
	print("[load] practical results ...")
	cls_long   = load_practical_long()
	naive_long = load_naive_long()
	benchmarks = compute_benchmarks(naive_long)
	prac_agg   = build_practical_summary(cls_long, naive_long, benchmarks)

	print("[load] 1-NN raw results ...")
	raw_1nn = None
	try:
		raw_1nn = load_1nn_raw()
	except FileNotFoundError as e:
		print(f"[error] {e}")

	generated = []
	slope_all: dict = {}
	slope_tail: dict = {}
	summary_1nn = None
	if raw_1nn is not None:
		agg_1nn = compute_1nn_aggregates(raw_1nn)
		summary_1nn = agg_1nn
		# Individual panels + composite.
		generated.append(plot_1nn_mean_error(agg_1nn))
		generated.append(plot_1nn_error_second_moment(agg_1nn))
		generated.append(plot_1nn_variance_ratio_only(agg_1nn))
		generated.append(plot_1nn_theory_validation(agg_1nn))

		# Remove the stale k-NN-named composite so no inconsistent duplicate
		# survives in the LaTeX include directory.
		for stale in (f"{OUT_DIR}/knn_theory_validation.pdf",
		              f"{OUT_DIR}/knn_theory_validation.png"):
			if os.path.exists(stale):
				os.remove(stale)
				print(f"  removed stale {stale}")

	generated.append(plot_event_probability_boxplots(cls_long, naive_long, benchmarks))
	generated.append(plot_practical_variance_ratio(prac_agg))
	generated.append(plot_practical_rmse(prac_agg))
	generated.append(plot_practical_accuracy(prac_agg))

	print("\n[final figures]")
	for f in generated:
		print(f"  {f}")

	if summary_1nn is not None:
		# Slopes for the empirical second moment on the largest-N tail
		# (largest 4 N values = 2000, 5000, 10000, 20000).
		Ns_sorted = np.array(sorted(summary_1nn["N"].unique()), dtype=float)
		tail_min = float(Ns_sorted[-4]) if len(Ns_sorted) >= 4 else float(Ns_sorted[0])

		print("\n[1-NN empirical second-moment log-log slopes for M_eps2 vs N]")
		print(f"  (tail uses N >= {int(tail_min)}: largest 4 N values)")
		slope_rows = []
		for m in INTERVALS:
			row = summary_1nn[summary_1nn["m"] == m].sort_values("N")
			s_all = _fit_loglog_slope(row["N"].values, row["M_eps2"].values, n_min=int(Ns_sorted[0]))
			s_tail = _fit_loglog_slope(row["N"].values, row["M_eps2"].values, n_min=int(tail_min))
			slope_all[m] = s_all
			slope_tail[m] = s_tail
			slope_rows.append({"m": m,
			                   "slope_all_N": s_all[0],
			                   "slope_largest4_N": s_tail[0],
			                   "tail_N_min": int(tail_min)})
			print(f"  m = {m:>2}  slope (all N) = {s_all[0]:+.3f}   "
			      f"slope (largest 4 N) = {s_tail[0]:+.3f}")
		pd.DataFrame(slope_rows).to_csv(
			f"{OUT_DIR}/1nn_theory_second_moment_slopes.csv", index=False)
		print(f"  wrote {OUT_DIR}/1nn_theory_second_moment_slopes.csv")

		# Endpoint diagnostics at largest N.
		N_max = int(summary_1nn["N"].max())
		print(f"\n[1-NN endpoint diagnostics at N = {N_max}]")
		endpoint = summary_1nn[summary_1nn["N"] == N_max].sort_values("m")
		for _, r in endpoint.iterrows():
			print(f"  m = {int(r['m']):>2}  "
			      f"mean_eps = {r['mean_eps']:.4f}  "
			      f"M_eps2 = {r['M_eps2']:.4e}  "
			      f"var-ratio = {r['variance_ratio']:.4f}")

if __name__ == "__main__":
	main()
