#!/usr/bin/env python3
"""Stage 09: publication figures and tables.

Gathers numerical results from stages 02-08 into the final set of
publication-quality figures (outputs/figures/) and tables (outputs/tables/),
each reproducible from a specific git commit (recorded in
outputs/tables/reproducibility_manifest.json).

Figures:
    mofa_variance_explained.(png|pdf)
    clustering_k_selection.(png|pdf)
    km_test_by_cluster.(png|pdf)
    km_tcga_by_cluster.(png|pdf)
    cluster_pathway_heatmap.(png|pdf)
    alluvial_cluster_pam50.(png|pdf)
    forest_plot_treatment_effect.(png|pdf)

Tables:
    table1_cluster_characteristics.csv
    reproducibility_manifest.json
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

from common import load_config, get_logger, resolve, get_git_commit

sns.set_theme(style="whitegrid", context="paper")
PALETTE = sns.color_palette("colorblind")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "axes.edgecolor": "0.3",
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})


def savefig(fig, path_no_ext, formats):
    for fmt in formats:
        fig.savefig(f"{path_no_ext}.{fmt}", bbox_inches="tight")
    plt.close(fig)


VIEW_LABELS = {"mrna_pathway": "mRNA pathway", "cna_pathway": "CNA pathway", "mutation": "Mutation"}


def plot_mofa_variance(tables_dir, figures_dir, formats, dpi):
    df = pd.read_csv(tables_dir / "mofa_variance_explained.csv")
    labels = [VIEW_LABELS.get(v, v) for v in df["view"]]
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)
    ax.bar(labels, df["R2_total"], color=PALETTE[:len(df)], edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Variance explained (R$^2$)")
    ax.set_title("MOFA variance explained per view")
    ax.set_ylim(0, 1)
    for i, v in enumerate(df["R2_total"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, figures_dir / "mofa_variance_explained", formats)


def plot_k_selection(tables_dir, figures_dir, formats, dpi):
    df = pd.read_csv(tables_dir / "clustering_k_selection.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=dpi)
    ax.errorbar(df["k"], df["mean_score"], yerr=df["std_score"], marker="o", color=PALETTE[0])
    best_k = df.loc[df["mean_score"].idxmax(), "k"]
    ax.axvline(best_k, color=PALETTE[3], linestyle="--", label=f"selected k*={int(best_k)}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inner-validation separation score\n" r"($-\log_{10}$ log-rank $p$)")
    ax.set_title("Survival-aware k selection")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best")
    savefig(fig, figures_dir / "clustering_k_selection", formats)


def plot_km(clinical_with_clusters, title, figures_dir, name, formats, dpi):
    from lifelines.statistics import multivariate_logrank_test
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=dpi)
    for i, cluster in enumerate(sorted(clinical_with_clusters["cluster"].unique())):
        sub = clinical_with_clusters[clinical_with_clusters["cluster"] == cluster]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["OS_MONTHS_NUM"], sub["OS_EVENT"], label=f"Cluster {cluster} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i % len(PALETTE)])
    lr_p = multivariate_logrank_test(clinical_with_clusters["OS_MONTHS_NUM"], clinical_with_clusters["cluster"],
                                      clinical_with_clusters["OS_EVENT"]).p_value
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall survival probability")
    ax.set_title(title)
    ax.text(0.02, 0.04, f"log-rank $p$ = {lr_p:.3g}", transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, figures_dir / name, formats)


def plot_pathway_heatmap(tables_dir, figures_dir, proc_dir, formats, dpi, top_n=5):
    diff = pd.read_csv(tables_dir / "cluster_biological_pathways.csv")
    top_feats = (diff.sort_values(["cluster", "abs_smd"], ascending=[True, False])
                 .groupby("cluster").head(top_n)["feature"].unique())

    mrna = pd.read_parquet(proc_dir / "view_mrna_pathway.parquet")
    cna = pd.read_parquet(proc_dir / "view_cna_pathway.parquet")
    combined = pd.concat([mrna, cna], axis=1)
    cl_train = pd.read_parquet(proc_dir / "cluster_labels_train.parquet")["cluster"]

    z = (combined[top_feats] - combined[top_feats].mean()) / combined[top_feats].std()
    z = z.loc[cl_train.index]
    cluster_means = z.groupby(cl_train).mean().T

    def clean_label(name, max_len=42):
        n = name.replace("CNA_PATH__", "CNA: ").replace("HALLMARK__", "H: ").replace("REACTOME__", "R: ") \
                .replace("KEGG__", "K: ").replace("ARM__", "Arm ").replace("_", " ")
        return n if len(n) <= max_len else n[:max_len - 1] + "…"

    cluster_means.index = [clean_label(i) for i in cluster_means.index]
    fig, ax = plt.subplots(figsize=(6.5, max(4, 0.3 * len(cluster_means))), dpi=dpi)
    sns.heatmap(cluster_means, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Mean z-score"},
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_title(f"Top {top_n} features per cluster by |SMD|")
    savefig(fig, figures_dir / "cluster_pathway_heatmap", formats)


def plot_alluvial(clinical_with_clusters, figures_dir, formats, dpi):
    df = clinical_with_clusters[["cluster", "CLAUDIN_SUBTYPE"]].copy()
    df["CLAUDIN_SUBTYPE"] = df["CLAUDIN_SUBTYPE"].replace("", "Unknown").fillna("Unknown")
    left_labels = sorted(df["cluster"].unique())
    right_labels = sorted(df["CLAUDIN_SUBTYPE"].unique())

    left_counts = df["cluster"].value_counts()
    right_counts = df["CLAUDIN_SUBTYPE"].value_counts()
    total = len(df)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    gap = 0.02
    left_y, right_y = {}, {}
    y = 0
    for lab in left_labels:
        h = left_counts[lab] / total
        left_y[lab] = (y, y + h)
        ax.add_patch(mpatches.Rectangle((0, y), 0.08, h, color=PALETTE[left_labels.index(lab) % len(PALETTE)]))
        ax.text(-0.02, y + h / 2, f"Cluster {lab}", ha="right", va="center", fontsize=8)
        y += h + gap
    y = 0
    for lab in right_labels:
        h = right_counts[lab] / total
        right_y[lab] = (y, y + h)
        ax.add_patch(mpatches.Rectangle((0.92, y), 0.08, h, color="lightgray"))
        ax.text(1.02, y + h / 2, lab, ha="left", va="center", fontsize=8)
        y += h + gap

    left_cursor = {lab: left_y[lab][0] for lab in left_labels}
    right_cursor = {lab: right_y[lab][0] for lab in right_labels}
    for lc in left_labels:
        for rc in right_labels:
            n = int(((df["cluster"] == lc) & (df["CLAUDIN_SUBTYPE"] == rc)).sum())
            if n == 0:
                continue
            h = n / total
            y0l, y0r = left_cursor[lc], right_cursor[rc]
            verts = [(0.08, y0l), (0.5, y0l), (0.5, y0r), (0.92, y0r),
                     (0.92, y0r + h), (0.5, y0r + h), (0.5, y0l + h), (0.08, y0l + h), (0.08, y0l)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            patch = mpatches.PathPatch(MplPath(verts, codes), alpha=0.35,
                                        color=PALETTE[left_labels.index(lc) % len(PALETTE)], lw=0)
            ax.add_patch(patch)
            left_cursor[lc] += h
            right_cursor[rc] += h

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, max(y, 1))
    ax.axis("off")
    ax.set_title("Cluster membership vs. PAM50-surrogate subtype (CLAUDIN_SUBTYPE)")
    savefig(fig, figures_dir / "alluvial_cluster_pam50", formats)


def plot_forest(tables_dir, figures_dir, formats, dpi):
    cox = pd.read_csv(tables_dir / "causal_cox_forest_plot.csv")
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(cox) + 1.5), dpi=dpi)
    y_pos = np.arange(len(cox))
    ax.errorbar(cox["hazard_ratio"], y_pos,
                xerr=[cox["hazard_ratio"] - cox["ci_lower"], cox["ci_upper"] - cox["hazard_ratio"]],
                fmt="o", color=PALETTE[0], capsize=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in cox["cluster"]])
    ax.set_xlabel("Chemotherapy hazard ratio (IPTW-weighted Cox, 95% CI)")
    ax.set_title("Cluster-specific chemotherapy effect (ER+, training cohort)")
    savefig(fig, figures_dir / "forest_plot_treatment_effect", formats)


def plot_baseline_comparison(tables_dir, figures_dir, formats, dpi):
    df = pd.read_csv(tables_dir / "baseline_comparison.csv").sort_values("c_index")
    short = {"SAMOC (this paper)": "SAMOC", "Standard MOFA + k-means (unsupervised k)": "Std. MOFA+k-means",
             "SNF (Wang et al. 2014)": "SNF", "NEMO-repro (Rappoport & Shamir 2019)": "NEMO-repro",
             "CIMLR-repro (Ramazzotti et al. 2018)": "CIMLR-repro",
             "iClusterPlus-repro (Shen et al. 2009/2012)": "iClusterPlus-repro"}
    labels = [short.get(m, m) for m in df["method"]]
    colors = [PALETTE[3] if m == "SAMOC (this paper)" else PALETTE[0] for m in df["method"]]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.barh(labels, df["c_index"], color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0.5, color="0.4", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("Test-set C-index")
    ax.set_title("SAMOC vs. alternative multi-omic clustering methods")
    ax.set_xlim(0.45, max(df["c_index"]) + 0.05)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(df["c_index"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    savefig(fig, figures_dir / "baseline_comparison", formats)


def plot_ablation(tables_dir, figures_dir, formats, dpi):
    df = pd.read_csv(tables_dir / "ablation_study.csv")
    short = {"Full SAMOC (pathway agg. + MOFA + survival-aware k)": "Full SAMOC",
             "No survival-aware clustering (unsupervised k)": "− survival-aware\nselection",
             "No MOFA (raw concatenated pathway features + survival-aware k)": "− MOFA",
             "No pathway aggregation (raw top-2000-variance genes + MOFA + survival-aware k)": "− pathway\naggregation"}
    labels = [short.get(v, v) for v in df["variant"]]
    colors = [PALETTE[3] if "Full" in v else PALETTE[1] for v in df["variant"]]
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=dpi)
    ax.bar(labels, df["c_index_test"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Test-set C-index")
    ax.set_title("Ablation: contribution of each SAMOC component")
    ax.set_ylim(0.45, 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(df["c_index_test"]):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    savefig(fig, figures_dir / "ablation_study", formats)


def build_table1(clinical_with_clusters, tables_dir):
    df = clinical_with_clusters.copy()
    df["AGE_AT_DIAGNOSIS"] = pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce")
    rows = []
    for cluster in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster]
        rows.append({
            "cluster": cluster, "n": len(sub),
            "pct_of_cohort": 100 * len(sub) / len(df),
            "median_age": sub["AGE_AT_DIAGNOSIS"].median() if "AGE_AT_DIAGNOSIS" in sub else np.nan,
            "pct_ER_positive": 100 * (sub["ER_IHC"] == "Positve").mean() if "ER_IHC" in sub else np.nan,
            "pct_HER2_gain": 100 * (sub["HER2_SNP6"] == "GAIN").mean() if "HER2_SNP6" in sub else np.nan,
            "pct_chemo_treated": 100 * (sub["CHEMOTHERAPY"] == "YES").mean() if "CHEMOTHERAPY" in sub else np.nan,
            "median_os_months": sub["OS_MONTHS_NUM"].median(),
            "pct_deceased": 100 * sub["OS_EVENT"].mean(),
            "dominant_PAM50_surrogate": sub["CLAUDIN_SUBTYPE"].mode().iloc[0] if not sub["CLAUDIN_SUBTYPE"].mode().empty else "NA",
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(tables_dir / "table1_cluster_characteristics.csv", index=False)
    return table1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    logger = get_logger("09_generate_figures_tables", resolve(cfg["paths"]["logs_dir"]))
    proc_dir = resolve(cfg["paths"]["processed_metabric_dir"])
    proc_tcga_dir = resolve(cfg["paths"]["processed_tcga_dir"])
    tables_dir = resolve(cfg["paths"]["tables_dir"])
    figures_dir = resolve(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    rcfg = cfg["reporting"]
    formats, dpi = rcfg["figure_formats"], rcfg["dpi"]

    clinical_with_clusters = pd.read_parquet(proc_dir / "clinical_with_clusters.parquet")
    test_ids = pd.read_parquet(proc_dir / "cluster_labels_test.parquet").index
    test_cwc = clinical_with_clusters.loc[test_ids]

    logger.info("Plotting MOFA variance explained ...")
    plot_mofa_variance(tables_dir, figures_dir, formats, dpi)

    logger.info("Plotting k-selection curve ...")
    plot_k_selection(tables_dir, figures_dir, formats, dpi)

    logger.info("Plotting Kaplan-Meier curves (held-out METABRIC test set) ...")
    plot_km(test_cwc, "Overall survival by cluster (METABRIC held-out test set)",
            figures_dir, "km_test_by_cluster", formats, dpi)

    tcga_path = proc_tcga_dir / "clinical_with_clusters.parquet"
    if tcga_path.exists():
        logger.info("Plotting Kaplan-Meier curves (TCGA external validation) ...")
        tcga_cwc = pd.read_parquet(tcga_path)
        plot_km(tcga_cwc, "Overall survival by cluster (TCGA-BRCA external validation)",
                figures_dir, "km_tcga_by_cluster", formats, dpi)
    else:
        logger.warning("TCGA processed clinical file not found; skipping TCGA KM plot")

    logger.info("Plotting cluster-characterising pathway heatmap ...")
    plot_pathway_heatmap(tables_dir, figures_dir, proc_dir, formats, dpi)

    logger.info("Plotting alluvial diagram (cluster vs PAM50 surrogate) ...")
    plot_alluvial(clinical_with_clusters, figures_dir, formats, dpi)

    logger.info("Plotting forest plot of cluster-specific treatment effects ...")
    plot_forest(tables_dir, figures_dir, formats, dpi)

    if (tables_dir / "baseline_comparison.csv").exists():
        logger.info("Plotting baseline comparison ...")
        plot_baseline_comparison(tables_dir, figures_dir, formats, dpi)
    if (tables_dir / "ablation_study.csv").exists():
        logger.info("Plotting ablation study ...")
        plot_ablation(tables_dir, figures_dir, formats, dpi)

    logger.info("Building Table 1 (cluster clinical/molecular characteristics) ...")
    build_table1(clinical_with_clusters, tables_dir)

    manifest = {
        "git_commit": get_git_commit(resolve(".")),
        "n_train": int((clinical_with_clusters.index.isin(
            pd.read_parquet(proc_dir / 'cluster_labels_train.parquet').index)).sum()),
        "n_test": len(test_cwc),
        "figures": sorted(p.name for p in figures_dir.glob("*")),
        "tables": sorted(p.name for p in tables_dir.glob("*")),
    }
    with open(tables_dir / "reproducibility_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Stage 09 complete. Figures written to %s, tables to %s", figures_dir, tables_dir)


if __name__ == "__main__":
    main()
