#!/usr/bin/env python3
"""Paper-build helper (not a numbered pipeline stage): renders selected
outputs/tables/*.csv into LaTeX table fragments under paper/tables/ for
\\input{} inclusion in the manuscript. Re-run any time after stage 09 to
refresh the manuscript's tables from the latest pipeline outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "outputs" / "tables"
PAPER_TABLES_DIR = ROOT / "paper" / "tables"
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def escape_tex(val):
    if isinstance(val, str):
        return val.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    return val


def write_tex(df: pd.DataFrame, name: str, float_format="%.3f", raw_cols=(), **kwargs):
    df = df.apply(lambda col: col.map(escape_tex) if (col.dtype == object and col.name not in raw_cols) else col)
    df.columns = [escape_tex(c) for c in df.columns]
    tex = df.to_latex(index=False, float_format=lambda x: float_format % x if pd.notna(x) else "--",
                       na_rep="--", **kwargs)
    (PAPER_TABLES_DIR / f"{name}.tex").write_text(tex)
    print(f"wrote {name}.tex")


def signed(x, digits):
    """Signed number in math mode, so a negative value gets a proper minus; zero has no sign."""
    s = f"{x:.{digits}f}"
    if float(s) == 0:
        return f"${abs(x):.{digits}f}$"
    return f"${x:+.{digits}f}$"


def plain(x, digits):
    """Unsigned format for values that should not show '+', with a proper minus when negative."""
    s = f"{x:.{digits}f}"
    if float(s) == 0:
        return f"{abs(x):.{digits}f}"
    return f"${s}$" if x < 0 else s


def p_text(p, digits=3):
    return "$<0.001$" if p < 0.001 else f"{p:.{digits}f}"


def tabular(spec, header, rows, footer=None):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *header, "\\midrule",
             *[" & ".join(map(str, r)) + " \\\\" for r in rows]]
    if footer:
        lines += ["\\midrule", *[" & ".join(map(str, r)) + " \\\\" for r in footer]]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_fragment(name, text):
    (PAPER_TABLES_DIR / f"{name}.tex").write_text(text)
    print(f"wrote {name}.tex")


def write_sensitivity_tables():
    from scipy import stats

    # full-procedure stability, k reselected in each subsample (stage 14)
    f = TABLES_DIR / "full_pipeline_stability.csv"
    if f.exists():
        d = pd.read_csv(f)
        k_ref = int(d.k_reference.iloc[0])
        rows = [(int(r.run), int(r.n_factors), int(r.k_selected), f"{r.ari_unconditional:.3f}",
                 f"{r.ari_conditional:.3f}") for r in d.itertuples()]
        foot = [("Mean", "", "", f"{d.ari_unconditional.mean():.3f}", f"{d.ari_conditional.mean():.3f}"),
                ("Range", "", "", f"{d.ari_unconditional.min():.3f}--{d.ari_unconditional.max():.3f}",
                 f"{d.ari_conditional.min():.3f}--{d.ari_conditional.max():.3f}")]
        write_fragment("full_pipeline_stability", tabular(
            "@{}ccccc@{}", [f"Run & MOFA factors & $k$ selected & ARI, $k$ reselected & ARI, $k$ fixed at {k_ref} \\\\"],
            rows, foot))

    # covariate balance under three weighting schemes (stage 15)
    fb, fc = TABLES_DIR / "causal_weighting_balance.csv", TABLES_DIR / "causal_weighting_comparison.csv"
    if fb.exists() and fc.exists():
        bal = pd.read_csv(fb).set_index("covariate")
        cmp_ = pd.read_csv(fc).set_index("scheme")
        names = {"AGE_AT_DIAGNOSIS": "Age at diagnosis", "GRADE": "Grade", "TUMOR_SIZE": "Tumour size",
                 "TUMOR_STAGE": "Tumour stage", "LYMPH_NODES_EXAMINED_POSITIVE": "Positive lymph nodes",
                 "HER2_SNP6_LOSS": "HER2 = loss", "HER2_SNP6_NEUTRAL": "HER2 = neutral",
                 "HER2_SNP6_UNDEF": "HER2 = undefined", "INFERRED_MENOPAUSAL_STATE_Pre": "Pre-menopausal",
                 "HORMONE_THERAPY_YES": "Hormone therapy", "RADIO_THERAPY_YES": "Radiotherapy"}
        cols = list(bal.columns)            # Unweighted, IPTW, Overlap, Matched
        schemes = cols[1:]
        rows = [(names.get(c, c), *[plain(bal.loc[c, s], 2) for s in cols]) for c in bal.index]
        n = len(bal)
        ab = bal.abs()
        foot = [("Mean $|$SMD$|$", *[f"{ab[s].mean():.2f}" for s in cols]),
                ("Number above 0.1", *[f"{int((ab[s] > 0.1).sum())} / {n}" for s in cols]),
                ("Patients retained", int(cmp_.loc[schemes[0], "n_used"]),
                 *[int(cmp_.loc[s, "n_used"]) for s in schemes]),
                ("Treated patients retained", int(cmp_.loc[schemes[0], "n_treated_used"]),
                 *[int(cmp_.loc[s, "n_treated_used"]) for s in schemes]),
                ("Interaction test $p$", "--", *[f"{cmp_.loc[s, 'lrt_p']:.2f}" for s in schemes])]
        write_fragment("causal_weighting", tabular(
            "@{}lrrrr@{}", ["Covariate & Unweighted & IPTW & Overlap & Matched \\\\"], rows, foot))

    # positive control (stage 16): paired t-test over replicates within each setting
    f = TABLES_DIR / "positive_control.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = []
        for sg, g in d.groupby("signal"):
            p = stats.ttest_rel(g.c_index_survival, g.c_index_silhouette).pvalue
            ksil = sorted(g.k_silhouette.unique())
            rows.append((f"{sg:.1f}", ", ".join(map(str, sorted(g.k_survival))), ", ".join(map(str, ksil)),
                         f"{g.c_index_survival.mean():.3f}", f"{g.c_index_silhouette.mean():.3f}",
                         signed(g.c_index_survival.mean() - g.c_index_silhouette.mean(), 3),
                         "$<0.001$" if p < 0.001 else f"{p:.2f}",
                         plain(g.ari_signal_survival.mean(), 2), plain(g.ari_signal_silhouette.mean(), 2)))
        write_fragment("positive_control", tabular(
            "@{}ccccccccc@{}",
            ["& \\multicolumn{2}{c}{$k$ selected} & \\multicolumn{3}{c}{Test-set C-index} & & "
             "\\multicolumn{2}{c}{ARI to signal groups} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){8-9}",
             "Signal & survival & silh. & survival & silh. & difference & $p$ & survival & silh. \\\\"], rows))

    # number of inner repeats (stage 17)
    f = TABLES_DIR / "inner_repeat_sensitivity.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [(int(r.run), int(r.n_factors), int(r.k_R5), int(r.k_R15), int(r.k_R50)) for r in d.itertuples()]
        foot = [("Distinct values", "", *[d[c].nunique() for c in ("k_R5", "k_R15", "k_R50")]),
                ("Mean winning margin", "", *[f"{d[c].mean():.2f}" for c in ("gap_R5", "gap_R15", "gap_R50")])]
        write_fragment("inner_repeat_sensitivity", tabular(
            "@{}ccccc@{}", ["Subsample & MOFA factors & $k$ at $R=5$ & $k$ at $R=15$ & $k$ at $R=50$ \\\\"],
            rows, foot))

    # gene-set draws (stage 18), against the original draw used in the paper
    f = TABLES_DIR / "geneset_seed_sensitivity.csv"
    if f.exists():
        d = pd.read_csv(f)
        abl = pd.read_csv(TABLES_DIR / "ablation_study.csv").iloc[0]
        n_fac = sum(c.startswith("Factor") for c in pd.read_csv(TABLES_DIR / "mofa_variance_explained.csv").columns)
        view = ROOT / "data" / "processed" / "metabric" / "view_mrna_pathway.parquet"
        n_mrna = pd.read_parquet(view).shape[1] if view.exists() else "--"
        rows = [("Original (paper)", int(d.n_gene_sets.iloc[0]), n_mrna, n_fac, int(abl.k),
                 f"{abl.c_index_test:.3f}", p_text(abl.logrank_p_test), "--")]
        rows += [(f"Draw {i + 1}", int(r.n_gene_sets), int(r.n_mrna_features), int(r.n_factors), int(r.k_selected),
                  f"{r.c_index_test:.3f}", p_text(r.logrank_p_test), f"{r.ari_vs_reference_train:.2f}")
                 for i, r in enumerate(d.itertuples())]
        write_fragment("geneset_seed_sensitivity", tabular(
            "@{}lccccccc@{}", ["Gene-set draw & Gene sets & mRNA features & Factors & $k$ & Test C-index & "
                               "Log-rank $p$ & ARI to original \\\\"], rows))

    # positive control with the search extended (stage 19)
    f = TABLES_DIR / "positive_control_k_range.csv"
    if f.exists():
        d = pd.read_csv(f)
        k_max = int(d.k_max.iloc[0])
        rows = [(f"{sg:.1f}", ", ".join(map(str, sorted(g.k_survival))),
                 f"{int((g.k_survival == k_max).sum())} / {len(g)}", f"{g.c_index_survival.mean():.3f}")
                for sg, g in d.groupby("signal")]
        write_fragment("positive_control_k_range", tabular(
            "@{}cccc@{}", [f"Signal & $k$ selected (search to {k_max}) & At the ceiling & Mean test C-index \\\\"],
            rows))

    # calibration of the k-sample log-rank test under the null (stage 21)
    f = TABLES_DIR / "logrank_null_calibration.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [(int(r.k), int(r.group_size), f"{r.mean_neg_log10_p:.3f}", f"{r.rejection_rate_005:.3f}")
                for r in d.itertuples()]
        write_fragment("logrank_null_calibration", tabular(
            "@{}cccc@{}", ["$k$ & Group size & Mean $-\\log_{10}p$ & Rejection rate at 0.05 \\\\"], rows,
            [("Valid test", "", f"{1 / np.log(10):.3f}", "0.050")]))


def main():
    model_labels = {"cluster_only": "Cluster only", "clinical_only": "Clinical only",
                     "pam50_only": "PAM50 (surrogate) only", "cluster_plus_clinical": "Cluster + clinical"}
    perf = pd.read_csv(TABLES_DIR / "survival_performance_test.csv")
    perf["model"] = perf["model"].map(model_labels).fillna(perf["model"])
    perf = perf.rename(columns={"model": "Model", "c_index": "C-index",
                                 "auc_60m": "AUC@60m", "auc_120m": "AUC@120m", "auc_180m": "AUC@180m"})
    write_tex(perf, "survival_performance_test")

    stab = pd.read_csv(TABLES_DIR / "cluster_stability_ari.csv")
    stab = stab.rename(columns={"run": "Run", "ari": "Adjusted Rand Index"})
    write_tex(stab, "cluster_stability_ari")

    forest = pd.read_csv(TABLES_DIR / "causal_cox_forest_plot.csv")
    forest = forest[["cluster", "hazard_ratio", "ci_lower", "ci_upper", "p_value", "e_value", "n_treated", "n_control"]]
    forest = forest.rename(columns={"cluster": "Cluster", "hazard_ratio": "HR", "ci_lower": "CI low",
                                     "ci_upper": "CI high", "p_value": "p", "e_value": "E-value",
                                     "n_treated": "N treated", "n_control": "N control"})
    write_tex(forest, "causal_cox_forest_plot")

    # out-of-fold vs in-sample labels: HR, LRT and causal-forest CATE side by side
    ins = pd.read_csv(TABLES_DIR / "causal_cox_forest_plot_insample_labels.csv")
    cate_oof = pd.read_csv(TABLES_DIR / "causal_forest_cate_by_cluster.csv")
    cate_ins = pd.read_csv(TABLES_DIR / "causal_forest_cate_by_cluster_insample_labels.csv")
    cmp_rows = []
    for _, r in pd.read_csv(TABLES_DIR / "causal_cox_forest_plot.csv").iterrows():
        c = int(r["cluster"])
        ri = ins[ins["cluster"] == c].iloc[0]
        co = cate_oof[cate_oof["cluster"] == c].iloc[0]
        ci = cate_ins[cate_ins["cluster"] == c].iloc[0]
        cmp_rows.append({
            "Cluster": c,
            "HR (out-of-fold)": f"{r['hazard_ratio']:.2f} ({r['ci_lower']:.2f}--{r['ci_upper']:.2f})",
            "HR (in-sample)": f"{ri['hazard_ratio']:.2f} ({ri['ci_lower']:.2f}--{ri['ci_upper']:.2f})",
            "CATE, months (out-of-fold)": f"{co['cate_rmst_months']:.1f} ({co['ci_lower']:.1f}, {co['ci_upper']:.1f})",
            "CATE, months (in-sample)": f"{ci['cate_rmst_months']:.1f} ({ci['ci_lower']:.1f}, {ci['ci_upper']:.1f})",
        })
    cmp = pd.DataFrame(cmp_rows)
    write_tex(cmp, "causal_oof_vs_insample")

    fold = pd.read_csv(TABLES_DIR / "causal_oof_fold_summary.csv")
    fold = fold[["fold", "n_fit", "n_held_out", "k_selected", "clusters_merged_on_matching"]]
    fold["clusters_merged_on_matching"] = fold["clusters_merged_on_matching"].map({True: "yes", False: "no"})
    fold = fold.rename(columns={"fold": "Fold", "n_fit": "N fit", "n_held_out": "N held out",
                                "k_selected": "k selected", "clusters_merged_on_matching": "Merged on matching?"})
    write_tex(fold, "causal_oof_fold_summary", float_format="%.0f")

    balance_labels = {
        "AGE_AT_DIAGNOSIS": "Age at diagnosis", "GRADE": "Grade", "TUMOR_SIZE": "Tumor size",
        "TUMOR_STAGE": "Tumor stage", "LYMPH_NODES_EXAMINED_POSITIVE": "Positive lymph nodes",
        "HER2_SNP6_LOSS": "HER2 SNP6 = loss", "HER2_SNP6_NEUTRAL": "HER2 SNP6 = neutral",
        "HER2_SNP6_UNDEF": "HER2 SNP6 = undefined", "INFERRED_MENOPAUSAL_STATE_Pre": "Menopausal state = pre",
        "HORMONE_THERAPY_YES": "Hormone therapy = yes", "RADIO_THERAPY_YES": "Radiotherapy = yes",
    }
    balance = pd.read_csv(TABLES_DIR / "causal_propensity_balance.csv")
    balance["covariate"] = balance["covariate"].map(balance_labels).fillna(balance["covariate"])
    balance = balance.rename(columns={"covariate": "Covariate", "smd_before_weighting": "SMD (before)",
                                       "smd_after_weighting": "SMD (after IPTW)"})
    write_tex(balance, "causal_propensity_balance")

    ksel = pd.read_csv(TABLES_DIR / "clustering_k_selection.csv")
    ksel = ksel[["k", "mean_logrank_p", "std_logrank_p", "mean_cindex_diff", "std_cindex_diff"]]
    ksel = ksel.rename(columns={"k": "k", "mean_logrank_p": "Log-rank score, mean", "std_logrank_p": "SD",
                                "mean_cindex_diff": "C-index, mean", "std_cindex_diff": "SD "})
    write_tex(ksel, "clustering_k_selection")

    table1 = pd.read_csv(TABLES_DIR / "table1_cluster_characteristics.csv")
    table1 = table1.rename(columns={
        "cluster": "Cluster", "n": "N", "pct_of_cohort": "% of cohort", "median_age": "Median age",
        "pct_ER_positive": "% ER+", "pct_HER2_gain": "% HER2 gain", "pct_chemo_treated": "% chemo",
        "median_os_months": "Median OS (mo)", "pct_deceased": "% deceased",
        "dominant_PAM50_surrogate": "Dominant PAM50-surrogate",
    })
    write_tex(table1, "table1_cluster_characteristics", float_format="%.1f")

    tcga_path = TABLES_DIR / "tcga_survival_validation.csv"
    if tcga_path.exists():
        tcga_surv = pd.read_csv(tcga_path)
        tcga_surv = tcga_surv.rename(columns={"n_patients": "N", "n_clusters": "N clusters",
                                               "logrank_p": "Log-rank p", "c_index_cluster_only": "C-index"})
        write_tex(tcga_surv, "tcga_survival_validation")
        tcga_prop = pd.read_csv(TABLES_DIR / "tcga_cluster_proportions_comparison.csv")
        tcga_prop = tcga_prop.rename(columns={tcga_prop.columns[0]: "Cluster",
                                               "metabric_train_proportion": "METABRIC train prop.",
                                               "tcga_proportion": "TCGA prop."})
        write_tex(tcga_prop, "tcga_cluster_proportions_comparison")
    else:
        print("TCGA tables not yet available; skipping")

    # ---- baseline comparison + significance tests -----------------------------------------
    baseline = pd.read_csv(TABLES_DIR / "baseline_comparison.csv")
    baseline = baseline[["method", "k", "c_index", "logrank_p", "is_reimplementation"]]
    baseline["is_reimplementation"] = baseline["is_reimplementation"].map({True: "yes", False: "no"})
    baseline = baseline.rename(columns={"method": "Method", "k": "k", "c_index": "C-index",
                                         "logrank_p": "Log-rank p", "is_reimplementation": "Python reimpl.?"})
    write_tex(baseline, "baseline_comparison")

    sig = pd.read_csv(TABLES_DIR / "baseline_significance_tests.csv")
    sig = sig.rename(columns={"comparison": "Comparison", "delta_c_index": "Delta C-index",
                               "ci_lower": "95% CI low", "ci_upper": "95% CI high", "p_value": "p (bootstrap)"})
    sig = sig.drop(columns=["n_bootstrap"])
    write_tex(sig, "baseline_significance_tests")

    # ---- ablation study ---------------------------------------------------------------------
    ablation = pd.read_csv(TABLES_DIR / "ablation_study.csv")
    ablation = ablation.rename(columns={"variant": "Variant", "k": "k",
                                         "c_index_test": "C-index (test)", "logrank_p_test": "Log-rank p (test)"})
    write_tex(ablation, "ablation_study")

    # ---- enrichment validation ---------------------------------------------------------------
    enrich = pd.read_csv(TABLES_DIR / "enrichment_validation.csv")
    mean_cols = [c for c in enrich.columns if c.startswith("cluster_") and c.endswith("_mean")]
    enrich_cols = ["term", "biological_axis"] + mean_cols + ["range_of_cluster_means_in_sd_units", "cluster_ranking_high_to_low"]
    enrich = enrich[enrich_cols].copy()
    enrich["term"] = (enrich["term"].str.replace("HALLMARK__", "", regex=False)
                      .str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True))
    enrich["biological_axis"] = enrich["biological_axis"].str.replace("_", " ")
    # ">" is not printable in OT1 text mode; typeset the ranking in math mode
    enrich["cluster_ranking_high_to_low"] = enrich["cluster_ranking_high_to_low"].map(
        lambda r: "$" + r.replace(">", " > ") + "$")
    enrich = enrich.rename(columns={"term": "Hallmark term", "biological_axis": "Axis",
                                     **{c: "C" + c.split("_")[1] for c in mean_cols},
                                     "range_of_cluster_means_in_sd_units": "Range (SD units)",
                                     "cluster_ranking_high_to_low": "Ranking (high to low)"})
    write_tex(enrich, "enrichment_validation", float_format="%.2f", raw_cols=("Ranking (high to low)",))

    # ---- cross-platform batch correction -----------------------------------------------------
    batch_mag = pd.read_csv(TABLES_DIR / "cross_platform_batch_effect_magnitude.csv")
    batch_mag = batch_mag.rename(columns={
        "view": "View", "n_features": "N features",
        "mean_abs_cohens_d_before": "Mean |d| (before)", "mean_abs_cohens_d_after": "Mean |d| (after ComBat)",
        "pct_features_|d|>0.5_before": "% |d|>0.5 (before)", "pct_features_|d|>0.5_after": "% |d|>0.5 (after)",
    })
    batch_mag = batch_mag[["View", "N features", "Mean |d| (before)", "Mean |d| (after ComBat)",
                            "% |d|>0.5 (before)", "% |d|>0.5 (after)"]]
    write_tex(batch_mag, "cross_platform_batch_effect_magnitude")

    correction = pd.read_csv(TABLES_DIR / "cross_platform_correction_comparison.csv")
    correction = correction.rename(columns={"metric": "Metric", "before_combat": "Before ComBat", "after_combat": "After ComBat"})
    write_tex(correction, "cross_platform_correction_comparison")

    # ---- sensitivity, control and weighting analyses (stages 14-19, 21) -----------------------
    write_sensitivity_tables()

    # ---- repeated outer splits (stage 20) -----------------------------------------------------
    if (TABLES_DIR / "repeated_splits_summary.csv").exists():
        summ = pd.read_csv(TABLES_DIR / "repeated_splits_summary.csv")
        comp = pd.read_csv(TABLES_DIR / "repeated_splits_comparisons.csv")
        comp["method"] = comp["comparison"].str.replace("Survival-guided k vs ", "", regex=False)
        rep = summ.merge(comp, on="method", how="left")
        labels = {"Survival-guided k (evaluated pipeline)": "Survival-guided $k$ (evaluated pipeline)",
                  "Silhouette-selected k": "Silhouette-selected $k$",
                  "Survival-guided k, no MOFA": "Survival-guided $k$, no MOFA",
                  "SNF": "SNF", "NEMO approximation": "NEMO (reimplementation)",
                  "CIMLR approximation": "CIMLR (reimplementation)",
                  "iClusterPlus approximation": "iClusterPlus (reimplementation)",
                  "PAM50 surrogate": "PAM50 surrogate", "Clinical variables": "Clinical variables",
                  "Clusters plus clinical": "Clusters plus clinical"}
        out = pd.DataFrame({
            "Method": rep["method"].map(labels),
            "Mean C-index": rep["mean_c_index"],
            "SD": rep["sd_c_index"],
            "Range": [f"{a:.3f}--{b:.3f}" for a, b in zip(rep.min_c_index, rep.max_c_index)],
            "Difference": ["--" if pd.isna(d) else f"${d:+.3f}$" for d in rep.mean_difference],
            "Splits": ["--" if pd.isna(n) else f"{int(b)}/{int(n)}"
                       for b, n in zip(rep.splits_survival_better, rep.n_splits)],
            "Corrected p": ["--" if pd.isna(p) else ("$<0.001$" if p < 0.001 else f"{p:.3f}")
                            for p in rep.p_corrected_resampled_t]})
        write_tex(out, "repeated_splits", raw_cols=("Method", "Difference", "Corrected p"))

    # ---- fixed-k control on the same splits (stage 22) ----------------------------------------
    if (TABLES_DIR / "fixed_k_summary.csv").exists():
        summ = pd.read_csv(TABLES_DIR / "fixed_k_summary.csv")
        comp = pd.read_csv(TABLES_DIR / "fixed_k_comparisons.csv")
        comp["quantity"] = comp["comparison"].str.replace(r"^survival-guided k \(2-8\) vs ", "", regex=True)
        fk = summ.merge(comp, on="quantity", how="left")
        fk = fk[~fk["quantity"].str.match(r"fixed k = (9|1[0-5])$")]
        # reference row first, then the fixed values, then the other rules, then the ceiling
        order = (["survival-guided k, searched 2-8"] + [f"fixed k = {k}" for k in range(2, 9)]
                 + ["k uniform on 2-8", "survival-guided k, searched 2-15",
                    "best k on held-out patients, 2-8 (oracle)"])
        fk = fk.set_index("quantity").loc[order].reset_index()

        def label(q):
            q = q.replace("fixed k = ", "Fixed $k=").replace("k uniform on 2-8", "$k$ drawn uniformly from 2--8")
            q = q.replace("survival-guided k, searched 2-8", "Survival-guided $k$, searched 2--8")
            q = q.replace("survival-guided k, searched 2-15", "Survival-guided $k$, searched 2--15")
            q = q.replace("best k on held-out patients, 2-8 (oracle)", "Best $k$ on held-out patients (oracle)")
            return q + "$" if q.startswith("Fixed $k=") else q

        out = pd.DataFrame({
            "Choice of $k$": fk["quantity"].map(label),
            "Mean C-index": fk["mean_c_index"],
            "SD": fk["sd_c_index"],
            "Difference": ["--" if pd.isna(d) else ("$0.000$" if abs(d) < 0.0005 else f"${d:+.3f}$")
                           for d in fk.mean_difference],
            "Splits": ["--" if pd.isna(n) else f"{int(b)}/{int(n)}"
                       for b, n in zip(fk.splits_survival_better, fk.n_splits)],
            "Corrected p": ["--" if pd.isna(p) else ("$<0.001$" if p < 0.001 else f"{p:.3f}")
                            for p in fk.p_corrected_resampled_t]})
        out.columns = ["Choice of $k$"] + list(out.columns[1:])
        (PAPER_TABLES_DIR / "fixed_k_control.tex").write_text(
            out.to_latex(index=False, float_format=lambda x: "%.3f" % x, na_rep="--"))
        print("wrote fixed_k_control.tex")


if __name__ == "__main__":
    main()
