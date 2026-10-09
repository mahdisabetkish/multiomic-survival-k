#!/usr/bin/env python3
"""Paper-build helper (not a numbered pipeline stage): renders the SAMOC
workflow overview as a publication-quality graphviz diagram."""
from pathlib import Path
import graphviz

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Helvetica"
g = graphviz.Digraph("samoc_workflow", format="pdf")
g.attr(rankdir="TB", splines="polyline", nodesep="0.30", ranksep="0.45", fontname=FONT)
g.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontsize="15",
       margin="0.14,0.08", penwidth="1.2")
g.attr("edge", fontname=FONT, fontsize="13", color="#555555", penwidth="1.1", arrowsize="0.7",
       labeldistance="1.4", labelangle="0")

COL_DATA = "#E8EEF7"
COL_STAGE = "#DCEEE1"
COL_EVAL = "#FBEEDC"
COL_CAUSAL = "#F6E1E1"
COL_EXT = "#EDE3F5"

g.node("raw", "METABRIC raw omics\n(mRNA, CNA, mutation, clinical)", fillcolor=COL_DATA, color="#5b7fae")
g.node("split", "Leakage-free stratified\ntrain / test split\n(before any fitting)", fillcolor=COL_STAGE, color="#3f8a54")
g.node("feat", "Feature aggregation\n(ssGSEA pathway scores,\narm and pathway CNA burden,\nrecurrent mutations)", fillcolor=COL_STAGE, color="#3f8a54")
g.node("mofa", "MOFA joint\nlatent factorisation\n(K=11 factors)", fillcolor=COL_STAGE, color="#3f8a54")
g.node("clust", "Choice of k by inner CV\non survival separation,\nthen k-means with k fixed\n(survival used only for k)", fillcolor=COL_STAGE, color="#3f8a54")
g.node("eval", "Evaluation on\nheld-out patients\n(C-index, AUC),\n10 repeated splits,\nsimulation, stability,\nbaselines, ablations", fillcolor=COL_EVAL, color="#c98a2e")
g.node("causal", "Treatment-effect\nheterogeneity\n(weighted Cox,\ncausal forest,\nE-values)", fillcolor=COL_CAUSAL, color="#b14f4f")
g.node("ext", "External\nvalidation on\nTCGA-BRCA,\nComBat\nharmonisation", fillcolor=COL_EXT, color="#7a4fa3")
g.node("out", "Clusters, forest plots,\nfigures & tables\n(outputs/)", fillcolor=COL_DATA, color="#5b7fae")

g.edge("raw", "split")
g.edge("split", "feat", label="  train + test\n(fit on train only)")
g.edge("feat", "mofa", label="  train-only\nstandardisation")
g.edge("mofa", "clust")
g.edge("clust", "eval")
g.edge("clust", "causal", label=" out-of-fold\n labels")
g.edge("mofa", "ext", label=" frozen\n model")
g.edge("clust", "ext", label=" frozen\n centroids")
g.edge("eval", "out")
g.edge("causal", "out")
g.edge("ext", "out")

g.attr(dpi="200")
g.format = "pdf"
g.render(str(OUT / "samoc_workflow"), cleanup=True)
g.format = "png"
g.render(str(OUT / "samoc_workflow"), cleanup=True)
print("wrote", OUT / "samoc_workflow.pdf", "and .png")
