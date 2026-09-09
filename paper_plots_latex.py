"""LaTeX-ready figures for the stochastic-inverse manuscript.

All output lives under Plots/results/latex/ and is generated in matched
PDF (vector, for LaTeX) + PNG (300 dpi, for inspection).

Produces:
  - Theory validation (composed function, k-NN only):
        knn_theory_classifier_error.{pdf,png}
        knn_theory_variance_ratio.{pdf,png}
        knn_theory_variance.{pdf,png}
  - Practical event-probability comparison, one PDF/PNG per problem:
        {problem}_variance_ratio.{pdf,png}
        {problem}_rmse.{pdf,png}
        {problem}_abs_bias.{pdf,png}
        {problem}_classifier_accuracy.{pdf,png}
        {problem}_est_box.{pdf,png}
  - Combined summary CSV: result_summary.csv

Naive baseline for every problem is the two-stage MC estimator with N iid
uniform Lambda samples -- verified in event_estimation.py.  POF-Darts only
changes the classifier training design; the Naive denominator is always the
uniform-sample baseline at matching (problem, m, N).

Benchmark probability per (problem, m) is the mean of the naive estimator
at N = 2000 across R = 30 repeats (~60k pooled samples), the same reference
we've been using.  Not fabricated: values come from
Results/Simulation/{ex}/Estimation_interval_{m}_Naive.csv.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from plotnine import (
	aes,
	element_text,
	element_rect,
	facet_grid,
	facet_wrap,
	geom_boxplot,
	geom_hline,
	geom_line,
	geom_point,
	geom_ribbon,
	ggplot,
	guide_legend,
	guides,
	labs,
	position_dodge,
	scale_color_manual,
	scale_fill_manual,
	scale_linetype_manual,
	scale_shape_manual,
	scale_x_log10,
	scale_y_log10,
	theme,
	theme_bw,
)

warnings.filterwarnings("ignore")

from make_plot import load_df

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OUT_DIR = "Plots/results/latex"
KNN_CSV = "Results/knn/function2_theory.csv"
EXAMPLES = ("function2", "brusselator", "lotka", "SIR")
INTERVALS = (5, 10, 20)
BENCHMARK_N = 2000

# Publication palette (Nature-inspired: red for MLP, navy for PPSVMG,
# black dotted for the two-stage MC baseline). Line/point figures rely on
# linetype/shape to distinguish POF vs Random; boxplots use BOX_COLOR_MAP
# below because fill is the only channel available for a box geom.
METHOD_ORDER = ["Naive", "Random_MLP", "Random_PPSVMG", "POF_MLP", "POF_PPSVMG"]
METHOD_LABEL = {
	"Naive":          "Two-stage MC",
	"Random_MLP":     "Random + MLP",
	"Random_PPSVMG":  "Random + PPSVMG",
	"POF_MLP":        "POF + MLP",
	"POF_PPSVMG":     "POF + PPSVMG",
}
COLOR_MAP = {
	"Naive":          "#000000",
	"Random_MLP":     "#D62728",
	"Random_PPSVMG":  "#1F4E79",
	"POF_MLP":        "#D62728",
	"POF_PPSVMG":     "#1F4E79",
}
# For boxplots: lighter shade for POF variants so the classifier family stays
# recognizable by hue while training design is distinguished by lightness.
BOX_COLOR_MAP = {
	"Naive":          "#4D4D4D",   # dark grey
	"Random_MLP":     "#D62728",   # red
	"POF_MLP":        "#F4A582",   # salmon (light red)
	"Random_PPSVMG":  "#1F4E79",   # navy
	"POF_PPSVMG":     "#6BAED6",   # sky blue (light navy)
}
LINETYPE_MAP = {
	"Naive":          "dotted",
	"Random_MLP":     "solid",
	"Random_PPSVMG":  "solid",
	"POF_MLP":        "dashed",
	"POF_PPSVMG":     "dashed",
}
SHAPE_MAP = {
	"Naive":          "s",
	"Random_MLP":     "o",
	"Random_PPSVMG":  "o",
	"POF_MLP":        "^",
	"POF_PPSVMG":     "^",
}


def _pub_theme():
	return theme_bw() + theme(
		figure_size=(12, 4.2),
		axis_text=element_text(size=9),
		axis_title=element_text(size=10),
		strip_text=element_text(size=10),
		strip_background=element_rect(fill="#EEEEEE"),
		legend_text=element_text(size=9),
		legend_title=element_text(size=9),
		legend_position="right",
		plot_title=element_text(size=10, ha="left"),
	)


def _facet_m_cat(series: pd.Series) -> pd.Categorical:
	return pd.Categorical("m = " + series.astype(str),
	                      categories=[f"m = {m}" for m in INTERVALS],
	                      ordered=True)


def _save(p, base: str) -> str:
	pdf = f"{OUT_DIR}/{base}.pdf"
	png = f"{OUT_DIR}/{base}.png"
	p.save(pdf, width=12, height=4.2, dpi=300, verbose=False)
	p.save(png, width=12, height=4.2, dpi=300, verbose=False)
	print(f"  wrote {pdf}, {png}")
	return pdf


# ----------------------------------------------------------------------------
# Data assembly for the practical comparison
# ----------------------------------------------------------------------------

def load_practical_long() -> pd.DataFrame:
	df = load_df().copy()
	df.loc[df["model"] == "NN", "model"] = "MLP"
	df["method"] = df["sample_method"].astype(str) + "_" + df["model"].astype(str)
	df = df.rename(columns={"example": "problem", "interval": "m"})
	df["n"] = df["n"].astype(int)
	return df[["problem", "m", "n", "r", "method", "estimation", "accuracy"]]


def load_naive_long() -> pd.DataFrame:
	frames = []
	for ex in EXAMPLES:
		for m in INTERVALS:
			p = f"Results/Simulation/{ex}/Estimation_interval_{m}_Naive.csv"
			if not os.path.exists(p):
				print(f"[warn] missing naive CSV: {p}")
				continue
			d = pd.read_csv(p)
			d.columns = d.columns.str.extract(r"(\d+)", expand=False).astype(float).astype(int)
			d["r"] = d.index
			d = d.melt(id_vars="r", var_name="n", value_name="estimation")
			d["problem"] = ex
			d["m"] = m
			d["method"] = "Naive"
			d["accuracy"] = np.nan
			frames.append(d)
	return pd.concat(frames, ignore_index=True)


def compute_benchmarks(naive_long: pd.DataFrame) -> pd.DataFrame:
	return (
		naive_long[naive_long["n"] == BENCHMARK_N]
		.groupby(["problem", "m"], as_index=False)["estimation"]
		.mean()
		.rename(columns={"estimation": "benchmark"})
	)


def build_practical_summary(classifier_long, naive_long, benchmarks) -> pd.DataFrame:
	all_long = pd.concat([classifier_long, naive_long], ignore_index=True, sort=False)

	agg = (
		all_long
		.groupby(["problem", "m", "n", "method"], as_index=False)
		.agg(
			mean_estimate=("estimation", "mean"),
			sample_variance=("estimation", "var"),
			classifier_accuracy_mean=("accuracy", "mean"),
			n_reps=("estimation", "count"),
		)
	)
	agg["sample_sd"] = np.sqrt(agg["sample_variance"])
	agg = agg.merge(benchmarks, on=["problem", "m"], how="left")
	agg["bias"] = agg["mean_estimate"] - agg["benchmark"]
	agg["abs_bias"] = agg["bias"].abs()
	agg["rmse"] = np.sqrt(agg["bias"] ** 2 + agg["sample_variance"])

	naive_var = (
		agg[agg["method"] == "Naive"][["problem", "m", "n", "sample_variance"]]
		.rename(columns={"sample_variance": "naive_variance"})
	)
	agg = agg.merge(naive_var, on=["problem", "m", "n"], how="left")
	agg["variance_ratio_to_naive"] = np.where(
		agg["naive_variance"] > 0,
		agg["sample_variance"] / agg["naive_variance"],
		np.nan,
	)
	# Naive by definition has ratio 1 (guard: 0/0 above becomes nan otherwise).
	agg.loc[agg["method"] == "Naive", "variance_ratio_to_naive"] = 1.0
	agg["classifier_error_mean"] = 1.0 - agg["classifier_accuracy_mean"]

	return agg


# ----------------------------------------------------------------------------
# Practical plots
# ----------------------------------------------------------------------------

def plot_practical_variance_ratio(agg: pd.DataFrame, problem: str) -> str:
	d = agg[(agg["problem"] == problem) & (agg["method"] != "Naive")].copy()
	d = d.dropna(subset=["variance_ratio_to_naive"])
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m"] = _facet_m_cat(d["m"])
	non_naive = METHOD_ORDER[1:]
	labels = [METHOD_LABEL[m] for m in non_naive]

	p = (
		ggplot(d, aes("n", "variance_ratio_to_naive",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_hline(yintercept=1.0, color="grey", linetype="dotted", size=0.5)
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_color_manual(values=[COLOR_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in non_naive],
		                        breaks=non_naive, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Variance ratio relative to two-stage Monte Carlo",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2),
		         shape=guide_legend(nrow=2))
		+ _pub_theme()
	)
	# facet as a grid of one row via facet_wrap done via _pub_theme size, so
	# rely on facet_wrap here.
	from plotnine import facet_wrap
	p = p + facet_wrap("~ m", nrow=1)
	return _save(p, f"{problem}_variance_ratio")


def _add_facet(p):
	from plotnine import facet_wrap
	return p + facet_wrap("~ m", nrow=1)


def plot_practical_rmse(agg: pd.DataFrame, problem: str) -> str:
	d = agg[agg["problem"] == problem].copy()
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m"] = _facet_m_cat(d["m"])
	labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
	p = (
		ggplot(d, aes("n", "rmse",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_color_manual(values=[COLOR_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in METHOD_ORDER],
		                        breaks=METHOD_ORDER, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="RMSE",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2),
		         shape=guide_legend(nrow=2))
		+ _pub_theme()
	)
	return _save(_add_facet(p), f"{problem}_rmse")


def plot_practical_abs_bias(agg: pd.DataFrame, problem: str) -> str:
	d = agg[agg["problem"] == problem].copy()
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m"] = _facet_m_cat(d["m"])
	eps = 1e-7
	d["abs_bias"] = d["abs_bias"].clip(lower=eps)
	labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
	p = (
		ggplot(d, aes("n", "abs_bias",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_color_manual(values=[COLOR_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in METHOD_ORDER],
		                        breaks=METHOD_ORDER, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Absolute bias  |mean(estimate) - benchmark|",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2),
		         shape=guide_legend(nrow=2))
		+ _pub_theme()
	)
	return _save(_add_facet(p), f"{problem}_abs_bias")


def plot_practical_classifier_accuracy(agg: pd.DataFrame, problem: str) -> str:
	d = agg[(agg["problem"] == problem) & (agg["method"] != "Naive")].copy()
	d = d.dropna(subset=["classifier_accuracy_mean"])
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m"] = _facet_m_cat(d["m"])
	non_naive = METHOD_ORDER[1:]
	labels = [METHOD_LABEL[m] for m in non_naive]
	p = (
		ggplot(d, aes("n", "classifier_accuracy_mean",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_color_manual(values=[COLOR_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in non_naive],
		                        breaks=non_naive, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Mean classifier accuracy on 5000-point test set",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2),
		         shape=guide_legend(nrow=2))
		+ _pub_theme()
	)
	return _save(_add_facet(p), f"{problem}_classifier_accuracy")


def plot_practical_est_box(classifier_long: pd.DataFrame,
                           naive_long: pd.DataFrame,
                           benchmarks: pd.DataFrame,
                           problem: str) -> str:
	"""Boxplot of raw R=30 estimate distributions, overlaid with benchmark."""
	long = pd.concat([classifier_long, naive_long], ignore_index=True, sort=False)
	d = long[long["problem"] == problem].copy()
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m"] = _facet_m_cat(d["m"])
	d["n_lbl"] = d["n"].astype(str)
	# Order n as categorical factor by numeric value.
	n_order = sorted(d["n"].unique())
	d["n_lbl"] = pd.Categorical(d["n_lbl"], categories=[str(x) for x in n_order],
	                            ordered=True)
	bench_here = benchmarks[benchmarks["problem"] == problem].copy()
	bench_here["m"] = _facet_m_cat(bench_here["m"])
	labels = [METHOD_LABEL[m] for m in METHOD_ORDER]

	p = (
		ggplot(d, aes("n_lbl", "estimation", fill="method", color="method"))
		+ geom_boxplot(position=position_dodge(0.85), width=0.75,
		               outlier_size=0.6, size=0.3, alpha=0.75)
		+ geom_hline(data=bench_here, mapping=aes(yintercept="benchmark"),
		             linetype="dotted", color="black", size=0.5,
		             inherit_aes=False)
		+ scale_fill_manual(values=[BOX_COLOR_MAP[m] for m in METHOD_ORDER],
		                    breaks=METHOD_ORDER, labels=labels)
		+ scale_color_manual(values=[BOX_COLOR_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Estimated event probability",
			fill="Method", color="Method",
			caption="Dotted horizontal line: benchmark from mean of two-stage MC at N=2000.",
		)
		+ guides(fill=guide_legend(nrow=2), color=guide_legend(nrow=2))
		+ _pub_theme()
	)
	from plotnine import facet_wrap
	p = p + facet_wrap("~ m", ncol=1, scales="free_y")
	# Boxplots want more vertical room; save with a taller canvas.
	pdf = f"{OUT_DIR}/{problem}_est_box.pdf"
	png = f"{OUT_DIR}/{problem}_est_box.png"
	p.save(pdf, width=12, height=9, dpi=300, verbose=False)
	p.save(png, width=12, height=9, dpi=300, verbose=False)
	print(f"  wrote {pdf}, {png}")
	return pdf


# ----------------------------------------------------------------------------
# Combined multi-problem figures (single shared legend across all 12 panels)
# ----------------------------------------------------------------------------

def _combined_theme():
	"""Theme tuned for the 4x3 combined-problem grids: legend at bottom so a
	single legend serves all panels without crowding any of them."""
	return theme_bw() + theme(
		figure_size=(13, 12),
		axis_text=element_text(size=8),
		axis_title=element_text(size=10),
		strip_text=element_text(size=9),
		strip_background=element_rect(fill="#EEEEEE"),
		legend_text=element_text(size=9),
		legend_title=element_text(size=9),
		legend_position="bottom",
		legend_box="horizontal",
		plot_title=element_text(size=11, ha="left"),
	)


def _save_combined(p, base: str, width=13.0, height=12.0) -> str:
	pdf = f"{OUT_DIR}/{base}.pdf"
	png = f"{OUT_DIR}/{base}.png"
	p.save(pdf, width=width, height=height, dpi=300, verbose=False)
	p.save(png, width=width, height=height, dpi=300, verbose=False)
	print(f"  wrote {pdf}, {png}")
	return pdf


def plot_combined_variance_ratio(agg: pd.DataFrame) -> str:
	d = agg[agg["method"] != "Naive"].copy()
	d = d.dropna(subset=["variance_ratio_to_naive"])
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m_lbl"] = _facet_m_cat(d["m"])
	d["problem_lbl"] = pd.Categorical(d["problem"], categories=EXAMPLES, ordered=True)
	non_naive = METHOD_ORDER[1:]
	labels = [METHOD_LABEL[m] for m in non_naive]
	p = (
		ggplot(d, aes("n", "variance_ratio_to_naive",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_hline(yintercept=1.0, color="grey", linetype="dotted", size=0.5)
		+ geom_line(size=0.8)
		+ geom_point(size=1.6)
		+ facet_grid("problem_lbl ~ m_lbl", scales="free_y")
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_color_manual(values=[COLOR_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in non_naive],
		                        breaks=non_naive, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in non_naive],
		                     breaks=non_naive, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Variance ratio relative to two-stage Monte Carlo",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=1), linetype=guide_legend(nrow=1),
		         shape=guide_legend(nrow=1))
		+ _combined_theme()
	)
	return _save_combined(p, "combined_variance_ratio")


def plot_combined_rmse(agg: pd.DataFrame) -> str:
	d = agg.copy()
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m_lbl"] = _facet_m_cat(d["m"])
	d["problem_lbl"] = pd.Categorical(d["problem"], categories=EXAMPLES, ordered=True)
	labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
	p = (
		ggplot(d, aes("n", "rmse",
		              color="method", linetype="method", shape="method", group="method"))
		+ geom_line(size=0.8)
		+ geom_point(size=1.6)
		+ facet_grid("problem_lbl ~ m_lbl", scales="free_y")
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_color_manual(values=[COLOR_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ scale_linetype_manual(values=[LINETYPE_MAP[m] for m in METHOD_ORDER],
		                        breaks=METHOD_ORDER, labels=labels)
		+ scale_shape_manual(values=[SHAPE_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="RMSE",
			color="Method", linetype="Method", shape="Method",
		)
		+ guides(color=guide_legend(nrow=1), linetype=guide_legend(nrow=1),
		         shape=guide_legend(nrow=1))
		+ _combined_theme()
	)
	return _save_combined(p, "combined_rmse")


def plot_combined_est_box(classifier_long: pd.DataFrame,
                          naive_long: pd.DataFrame,
                          benchmarks: pd.DataFrame) -> str:
	long = pd.concat([classifier_long, naive_long], ignore_index=True, sort=False)
	d = long.copy()
	d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER, ordered=True)
	d["m_lbl"] = _facet_m_cat(d["m"])
	d["problem_lbl"] = pd.Categorical(d["problem"], categories=EXAMPLES, ordered=True)
	n_order = sorted(d["n"].unique())
	d["n_lbl"] = pd.Categorical(d["n"].astype(str),
	                            categories=[str(x) for x in n_order], ordered=True)
	bench_here = benchmarks.copy()
	bench_here["m_lbl"] = _facet_m_cat(bench_here["m"])
	bench_here["problem_lbl"] = pd.Categorical(bench_here["problem"],
	                                            categories=EXAMPLES, ordered=True)
	labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
	p = (
		ggplot(d, aes("n_lbl", "estimation", fill="method", color="method"))
		+ geom_boxplot(position=position_dodge(0.85), width=0.75,
		               outlier_size=0.4, size=0.25, alpha=0.75)
		+ geom_hline(data=bench_here, mapping=aes(yintercept="benchmark"),
		             linetype="dotted", color="black", size=0.4,
		             inherit_aes=False)
		+ facet_grid("problem_lbl ~ m_lbl", scales="free_y")
		+ scale_fill_manual(values=[BOX_COLOR_MAP[m] for m in METHOD_ORDER],
		                    breaks=METHOD_ORDER, labels=labels)
		+ scale_color_manual(values=[BOX_COLOR_MAP[m] for m in METHOD_ORDER],
		                     breaks=METHOD_ORDER, labels=labels)
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Estimated event probability",
			fill="Method", color="Method",
			caption="Dotted horizontal line: benchmark from mean of two-stage MC at N=2000.",
		)
		+ guides(fill=guide_legend(nrow=1), color=guide_legend(nrow=1))
		+ _combined_theme()
		+ theme(axis_text_x=element_text(rotation=45, hjust=1, size=6))
	)
	return _save_combined(p, "combined_est_box", width=15.0, height=13.0)


# ----------------------------------------------------------------------------
# Theory validation
# ----------------------------------------------------------------------------

def load_knn_theory() -> pd.DataFrame:
	if not os.path.exists(KNN_CSV):
		raise FileNotFoundError(
			f"missing k-NN theory data: {KNN_CSV}. "
			"Run `python knn_theory.py` first."
		)
	return pd.read_csv(KNN_CSV)


def build_theory_summary(theory: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate per (m, N) with mean/sd/sem and variances."""
	agg = (
		theory
		.groupby(["m", "N"], as_index=False)
		.agg(
			knn_mean=("knn_estimate", "mean"),
			knn_var=("knn_estimate", "var"),
			knn_est_std=("knn_estimate", "std"),
			naive_mean=("naive_estimate", "mean"),
			naive_var=("naive_estimate", "var"),
			naive_est_std=("naive_estimate", "std"),
			knn_err_mean=("knn_error", "mean"),
			knn_err_std=("knn_error", "std"),
			n_reps=("knn_estimate", "count"),
		)
	)
	agg["variance_ratio"] = agg["knn_var"] / agg["naive_var"]
	agg["knn_err_sem"] = agg["knn_err_std"] / np.sqrt(agg["n_reps"].clip(lower=1))
	return agg


def plot_knn_classifier_error(theory_agg: pd.DataFrame) -> str:
	d = theory_agg.copy()
	d["m_lbl"] = pd.Categorical("m = " + d["m"].astype(str),
	                            categories=[f"m = {m}" for m in INTERVALS],
	                            ordered=True)
	d["lo"] = (d["knn_err_mean"] - 1.96 * d["knn_err_sem"]).clip(lower=0.0)
	d["hi"] =  d["knn_err_mean"] + 1.96 * d["knn_err_sem"]
	p = (
		ggplot(d, aes("N", "knn_err_mean", color="m_lbl", fill="m_lbl", group="m_lbl"))
		+ geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.20, color=None)
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ labs(
			x="Number of expensive function evaluations, N",
			y="k-NN classification error",
			color="Intervals", fill="Intervals",
		)
		+ _pub_theme()
	)
	return _save(p, "knn_theory_classifier_error")


def plot_knn_variance_ratio(theory_agg: pd.DataFrame) -> str:
	d = theory_agg.copy()
	d["m_lbl"] = pd.Categorical("m = " + d["m"].astype(str),
	                            categories=[f"m = {m}" for m in INTERVALS],
	                            ordered=True)
	p = (
		ggplot(d, aes("N", "variance_ratio", color="m_lbl", group="m_lbl"))
		+ geom_hline(yintercept=1.0, color="grey", linetype="dotted", size=0.5)
		+ geom_line(size=0.9)
		+ geom_point(size=2.0)
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Variance ratio relative to two-stage Monte Carlo",
			color="Intervals",
		)
		+ _pub_theme()
	)
	return _save(p, "knn_theory_variance_ratio")


def plot_knn_variance(theory_agg: pd.DataFrame) -> str:
	d = theory_agg[["m", "N", "knn_var", "naive_var"]].melt(
		id_vars=["m", "N"], var_name="src", value_name="variance")
	d["m_lbl"] = pd.Categorical("m = " + d["m"].astype(str),
	                            categories=[f"m = {m}" for m in INTERVALS],
	                            ordered=True)
	d["src_lbl"] = d["src"].map({"knn_var": "k-NN estimator",
	                             "naive_var": "Two-stage MC"})
	d["group"] = d["m_lbl"].astype(str) + " / " + d["src_lbl"]
	# Add a reference N^{-1} line anchored at the two-stage MC variance at
	# the smallest N per m so readers can eyeball MC rate.
	refs = []
	for m in INTERVALS:
		sub = d[(d["m"] == m) & (d["src"] == "naive_var")].sort_values("N")
		if sub.empty:
			continue
		N0 = int(sub["N"].iloc[0])
		v0 = float(sub["variance"].iloc[0])
		N_grid = np.geomspace(sub["N"].min(), sub["N"].max(), 12)
		refs.append(pd.DataFrame({
			"m": m, "N": N_grid, "variance": v0 * (N0 / N_grid),
			"src": "N^{-1} reference",
			"m_lbl": pd.Categorical([f"m = {m}"] * len(N_grid),
			                        categories=[f"m = {mi}" for mi in INTERVALS],
			                        ordered=True),
			"src_lbl": "N^{-1} reference",
			"group": [f"m = {m} / N^{{-1}} reference"] * len(N_grid),
		}))
	if refs:
		d = pd.concat([d] + refs, ignore_index=True, sort=False)

	# Colour by m, distinguish knn vs naive vs ref via linetype.
	linetype_map = {"k-NN estimator": "solid",
	                "Two-stage MC":   "dashed",
	                "N^{-1} reference": "dotted"}
	p = (
		ggplot(d, aes("N", "variance", color="m_lbl", linetype="src_lbl", group="group"))
		+ geom_line(size=0.8)
		+ geom_point(size=1.6, data=d[d["src"] != "N^{-1} reference"])
		+ scale_x_log10(breaks=[100, 200, 400, 1000, 2000])
		+ scale_y_log10()
		+ scale_linetype_manual(values=list(linetype_map.values()),
		                        breaks=list(linetype_map.keys()))
		+ labs(
			x="Number of expensive function evaluations, N",
			y="Sample variance of P(E) estimator",
			color="Intervals", linetype="Estimator",
		)
		+ _pub_theme()
	)
	return _save(p, "knn_theory_variance")


# ----------------------------------------------------------------------------
# Combined CSV
# ----------------------------------------------------------------------------

def _naive_denominator(practical_summary: pd.DataFrame,
                       problem: str, m: int, N: int) -> float:
	row = practical_summary[
		(practical_summary["problem"] == problem)
		& (practical_summary["m"] == m)
		& (practical_summary["n"] == N)
		& (practical_summary["method"] == "Naive")
	]
	if row.empty:
		return np.nan
	return float(row["sample_variance"].iloc[0])


def build_combined_csv(practical_summary: pd.DataFrame,
                       theory_agg: pd.DataFrame) -> pd.DataFrame:
	prac = practical_summary.copy()
	prac = prac.rename(columns={
		"n": "N",
		"benchmark": "benchmark_probability",
	})
	prac["repetitions"] = prac["n_reps"]
	cols = [
		"problem", "m", "N", "method",
		"repetitions", "mean_estimate", "benchmark_probability",
		"bias", "abs_bias", "sample_variance", "sample_sd", "rmse",
		"variance_ratio_to_naive",
		"classifier_accuracy_mean", "classifier_error_mean",
	]
	prac = prac[cols]

	# Add k-NN rows for function2 using the theory experiment.
	naive_var_from_theory = theory_agg[["m", "N", "naive_var"]]
	theory_extra = theory_agg.copy()
	theory_extra["problem"] = "function2"
	theory_extra["method"] = "kNN"
	theory_extra["repetitions"] = theory_extra["n_reps"]
	theory_extra["mean_estimate"] = theory_extra["knn_mean"]
	# Match benchmark to the same reference used for the practical function2 rows.
	fbench = (prac[(prac["problem"] == "function2") & (prac["method"] == "Naive")]
	          .drop_duplicates(subset=["m", "N"])
	          [["m", "N", "benchmark_probability"]])
	theory_extra = theory_extra.merge(fbench, on=["m", "N"], how="left")
	theory_extra["bias"] = theory_extra["mean_estimate"] - theory_extra["benchmark_probability"]
	theory_extra["abs_bias"] = theory_extra["bias"].abs()
	theory_extra["sample_variance"] = theory_extra["knn_var"]
	theory_extra["sample_sd"] = theory_extra["knn_est_std"]
	theory_extra["rmse"] = np.sqrt(theory_extra["bias"] ** 2 + theory_extra["sample_variance"])
	# Ratio is against the theory experiment's own naive baseline (matched N/m).
	theory_extra = theory_extra.merge(naive_var_from_theory, on=["m", "N"], how="left",
	                                  suffixes=("", "_naive"))
	theory_extra["variance_ratio_to_naive"] = np.where(
		theory_extra["naive_var"] > 0,
		theory_extra["sample_variance"] / theory_extra["naive_var"],
		np.nan,
	)
	theory_extra["classifier_accuracy_mean"] = 1.0 - theory_extra["knn_err_mean"]
	theory_extra["classifier_error_mean"] = theory_extra["knn_err_mean"]
	theory_extra = theory_extra[cols]

	combined = pd.concat([prac, theory_extra], ignore_index=True)
	combined = combined.sort_values(["problem", "m", "N", "method"]).reset_index(drop=True)
	return combined


# ----------------------------------------------------------------------------
# Numerical report
# ----------------------------------------------------------------------------

def _fit_loglog_slope(N: np.ndarray, y: np.ndarray) -> float:
	mask = np.isfinite(N) & np.isfinite(y) & (N > 0) & (y > 0)
	if mask.sum() < 3:
		return np.nan
	x = np.log(N[mask])
	z = np.log(y[mask])
	return float(np.polyfit(x, z, 1)[0])


def report_theory(theory_agg: pd.DataFrame) -> None:
	print("\n----- THEORY VALIDATION (composed function, k-NN) -----")
	for m in INTERVALS:
		sub = theory_agg[theory_agg["m"] == m].sort_values("N")
		if sub.empty:
			continue
		err0 = sub["knn_err_mean"].iloc[0]
		errL = sub["knn_err_mean"].iloc[-1]
		ratio_min = sub["variance_ratio"].min()
		ratio_max = sub["variance_ratio"].max()
		N_min_ratio = int(sub[sub["variance_ratio"] < 1]["N"].min()) \
			if (sub["variance_ratio"] < 1).any() else -1
		naive_slope = _fit_loglog_slope(sub["N"].values, sub["naive_var"].values)
		knn_slope   = _fit_loglog_slope(sub["N"].values, sub["knn_var"].values)
		print(f"  m={m:>2}  "
		      f"class-err {err0:.3f} -> {errL:.3f}  |  "
		      f"var-ratio {ratio_min:.2f}..{ratio_max:.2f}  |  "
		      f"first N with ratio<1: {N_min_ratio}  |  "
		      f"log-log slope naive_var vs N: {naive_slope:+.2f}  "
		      f"knn_var vs N: {knn_slope:+.2f}")


def report_practical(agg: pd.DataFrame) -> None:
	print("\n----- PRACTICAL COMPARISON -----")
	target_Ns = [200, 600, 2000]
	for problem in EXAMPLES:
		print(f"\n  {problem}")
		for m in INTERVALS:
			sub = agg[(agg["problem"] == problem) & (agg["m"] == m)]
			if sub.empty:
				continue
			best_var  = sub[sub["method"] != "Naive"].loc[
				sub[sub["method"] != "Naive"]["sample_variance"].idxmin()]
			best_rmse = sub.loc[sub["rmse"].idxmin()]
			print(f"    m={m:>2}  "
			      f"lowest-var method: {best_var['method']:<15s} @ N={int(best_var['n']):>4}  "
			      f"(Var={best_var['sample_variance']:.2e})  |  "
			      f"lowest-RMSE method: {best_rmse['method']:<15s} @ N={int(best_rmse['n']):>4}  "
			      f"(RMSE={best_rmse['rmse']:.2e})")
			for N in target_Ns:
				row = sub[sub["n"] == N]
				if row.empty:
					print(f"      N={N:>4}: no data")
					continue
				ratios = []
				for meth in METHOD_ORDER[1:]:
					r = row[row["method"] == meth]
					if r.empty:
						continue
					ratios.append(f"{meth}={float(r['variance_ratio_to_naive'].iloc[0]):.2f}")
				print(f"      N={N:>4}: variance-ratios: " + "  ".join(ratios))
			# POF vs Random summary at N=2000 (or largest available).
			for classifier in ("MLP", "PPSVMG"):
				sub_c = sub[sub["method"].isin([f"POF_{classifier}",
				                                f"Random_{classifier}"])]
				merged = (
					sub_c.pivot(index="n", columns="method", values="sample_variance")
					.dropna()
				)
				if merged.empty:
					continue
				pof_col = f"POF_{classifier}"
				ran_col = f"Random_{classifier}"
				if pof_col not in merged or ran_col not in merged:
					continue
				pof_wins = int((merged[pof_col] < merged[ran_col]).sum())
				total    = int(len(merged))
				print(f"      POF_{classifier} vs Random_{classifier}: "
				      f"POF has lower variance at {pof_wins}/{total} N values")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
	os.makedirs(OUT_DIR, exist_ok=True)

	print("[load] practical results ...")
	cls_long = load_practical_long()
	naive_long = load_naive_long()
	benchmarks = compute_benchmarks(naive_long)
	practical_summary = build_practical_summary(cls_long, naive_long, benchmarks)
	print(f"       {len(practical_summary)} (problem, m, N, method) aggregate rows")

	print("[load] k-NN theory results ...")
	try:
		theory = load_knn_theory()
	except FileNotFoundError as e:
		print(f"[error] {e}")
		theory_agg = None
	else:
		theory_agg = build_theory_summary(theory)
		print(f"       theory: {len(theory_agg)} (m, N) aggregate rows; "
		      f"{theory['r'].nunique()} repeats")

	generated = []

	if theory_agg is not None:
		print("\n[plots] theory validation")
		generated.append(plot_knn_classifier_error(theory_agg))
		generated.append(plot_knn_variance_ratio(theory_agg))
		generated.append(plot_knn_variance(theory_agg))

	for problem in EXAMPLES:
		print(f"\n[plots] practical -- {problem}")
		generated.append(plot_practical_variance_ratio(practical_summary, problem))
		generated.append(plot_practical_rmse(practical_summary, problem))
		generated.append(plot_practical_abs_bias(practical_summary, problem))
		generated.append(plot_practical_classifier_accuracy(practical_summary, problem))
		generated.append(plot_practical_est_box(cls_long, naive_long, benchmarks, problem))

	print("\n[plots] combined multi-problem figures (single shared legend)")
	generated.append(plot_combined_variance_ratio(practical_summary))
	generated.append(plot_combined_rmse(practical_summary))
	generated.append(plot_combined_est_box(cls_long, naive_long, benchmarks))

	# Combined CSV.
	if theory_agg is not None:
		combined = build_combined_csv(practical_summary, theory_agg)
	else:
		combined = build_combined_csv(
			practical_summary,
			pd.DataFrame(columns=["m", "N", "knn_mean", "knn_var", "knn_est_std",
			                      "naive_mean", "naive_var", "naive_est_std",
			                      "knn_err_mean", "knn_err_std", "n_reps",
			                      "variance_ratio", "knn_err_sem"]),
		)
	csv_path = f"{OUT_DIR}/result_summary.csv"
	combined.to_csv(csv_path, index=False)
	print(f"\n[csv] wrote {csv_path}  ({len(combined)} rows)")
	generated.append(csv_path)

	if theory_agg is not None:
		report_theory(theory_agg)
	report_practical(practical_summary)

	print("\n----- GENERATED FILES -----")
	for f in generated:
		print(f"  {f}")

if __name__ == "__main__":
	main()
