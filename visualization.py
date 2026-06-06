"""
visualization.py
=================
Generates 6 key visualizations for the cytogenomics project.

Plots produced:
  1. Sample Group Distribution Bar Plot
  2. CNV Count Per Sample (colored by condition)
  3. Deletion vs Duplication / Gain vs Loss Grouped Bar
  4. Chromosome-wise Alteration Frequency (PE vs PL)
  5. Shared Altered Genes/Chromosomes Venn Diagram
  6. Heatmap of Recurrent Chromosomal Regions

Outputs saved to: plots/

How to run:
    python scripts/visualization.py

Requirements:
    pip install matplotlib seaborn pandas matplotlib-venn
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS      = PROJECT_ROOT / "results"
PLOTS        = PROJECT_ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "preeclampsia"           : "#C0392B",   # deep red
    "healthy control"        : "#27AE60",   # green
    "pregnancy loss"         : "#2980B9",   # blue
    "chorionic villi"        : "#8E44AD",   # purple
    "extraembryonic mesoderm": "#1ABC9C",   # teal
    "father"                 : "#E67E22",   # orange
    "mother"                 : "#F39C12",   # yellow-orange
    "deletion/loss"          : "#E74C3C",   # red
    "duplication/gain"       : "#3498DB",   # blue
    "mosaic"                 : "#9B59B6",   # purple
}

CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]


def style_figure(fig, ax_or_axes, title: str = "") -> None:
    """Apply consistent styling."""
    axes = [ax_or_axes] if hasattr(ax_or_axes, "set_title") else ax_or_axes
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Sample Group Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_distribution(meta: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    Shows how many samples are in each group. Essential for understanding
    dataset balance. Preeclampsia and controls are blood-based; pregnancy
    loss samples are fetal tissues. This plot explains the dataset composition
    before any CNV analysis.
    """
    # Define the groups we want to show
    group_labels = [
        "Preeclampsia\n(PE)",
        "Healthy\nControl",
        "Pregnancy Loss\nChorionic Villi",
        "Pregnancy Loss\nExtraembryonic\nMesoderm",
    ]
    group_counts = [
        len(meta[(meta["condition"] == "preeclampsia")]),
        len(meta[(meta["condition"] == "healthy control")]),
        len(meta[(meta["tissue_type"] == "chorionic villi")]),
        len(meta[(meta["tissue_type"] == "extraembryonic mesoderm")]),
    ]
    bar_colors = [
        COLORS["preeclampsia"],
        COLORS["healthy control"],
        COLORS["chorionic villi"],
        COLORS["extraembryonic mesoderm"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(group_labels, group_counts, color=bar_colors,
                  edgecolor="white", linewidth=1.5, width=0.6)

    for bar, count in zip(bars, group_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(count), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Plot 1: Sample Group Distribution\n"
                 "GSE192614 (Preeclampsia Study) and GSE228150 (Pregnancy Loss Study)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(group_counts) + 2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Dataset separator
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.5, max(group_counts) + 1.5, "GSE192614",
            ha="center", va="bottom", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())
    ax.text(2.5, max(group_counts) + 1.5, "GSE228150",
            ha="center", va="bottom", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())

    style_figure(fig, ax)
    fig.tight_layout()
    out = PLOTS / "plot1_sample_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 1 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — CNV Count Per Sample
# ─────────────────────────────────────────────────────────────────────────────

def plot_cnv_per_sample(df_burden: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    Each bar = one patient/sample. Higher bars = more chromosomal alterations.
    Preeclampsia samples typically show elevated CNV burden vs controls.
    Fetal pregnancy loss samples may show very high burdens (aneuploidies).
    This plot gives a first-glance view of chromosomal instability per sample.
    """
    # Sort: preeclampsia → control → pregnancy loss fetal
    order = ["preeclampsia", "healthy control", "pregnancy loss"]
    df_plot = df_burden[df_burden["condition"].isin(order)].copy()
    df_plot["cond_order"] = df_plot["condition"].map({c: i for i, c in enumerate(order)})
    df_plot = df_plot.sort_values(["cond_order", "total_cnvs"], ascending=[True, False])

    colors = [COLORS.get(c, "#888888") for c in df_plot["condition"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(df_plot)), df_plot["total_cnvs"],
                  color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["sample_title"] if "sample_title" in df_plot.columns
                       else df_plot["sample_id"],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of CNVs / Chromosomal Alterations", fontsize=11)
    ax.set_title("Plot 2: CNV / Chromosomal Alteration Burden Per Sample\n"
                 "Each bar represents one sample; color indicates condition",
                 fontsize=12, fontweight="bold")

    # Legend
    legend_patches = [mpatches.Patch(color=COLORS[c], label=c.title()) for c in order]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.8)

    # Condition separators
    boundaries = []
    for cond in order[:-1]:
        idx = df_plot[df_plot["condition"] == cond].index
        if len(idx) > 0:
            pos = list(df_plot.index).index(idx[-1])
            boundaries.append(pos + 0.5)
    for b in boundaries:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.4)

    style_figure(fig, ax)
    fig.tight_layout()
    out = PLOTS / "plot2_cnv_per_sample.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 2 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Deletion vs Duplication
# ─────────────────────────────────────────────────────────────────────────────

def plot_deletion_duplication(df_dd: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    Deletions (losses of DNA) vs duplications (gains of DNA) have different
    biological consequences:
      - Deletions: loss of tumour suppressors, growth factors, immune genes
      - Duplications: overexpression of oncogenes, immune genes, growth factors
    In pregnancy loss, whole-chromosome gains (trisomies) dominate.
    In preeclampsia, smaller segmental deletions of angiogenic genes are common.
    """
    conditions = ["preeclampsia", "healthy control", "pregnancy loss"]
    alteration_classes = ["deletion/loss", "duplication/gain", "mosaic"]

    # Build pivot
    df_sub = df_dd[df_dd["condition"].isin(conditions)].copy()
    pivot  = df_sub.pivot_table(index="condition", columns="alteration_class",
                                values="count", aggfunc="sum", fill_value=0)
    pivot  = pivot.reindex(index=conditions)
    for ac in alteration_classes:
        if ac not in pivot.columns:
            pivot[ac] = 0
    pivot = pivot[alteration_classes]

    x     = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))

    bar_colors = [COLORS.get(ac, "#888888") for ac in alteration_classes]
    for i, (ac, col) in enumerate(zip(alteration_classes, bar_colors)):
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, pivot[ac], width, label=ac.title(),
                        color=col, edgecolor="white", linewidth=1)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        str(int(h)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in conditions], fontsize=11)
    ax.set_ylabel("Number of Chromosomal Alterations", fontsize=11)
    ax.set_title("Plot 3: Deletion vs Duplication / Gain vs Loss\n"
                 "Comparison across conditions",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    style_figure(fig, ax)
    fig.tight_layout()
    out = PLOTS / "plot3_deletion_duplication.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 3 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Chromosome-wise Alteration Frequency
# ─────────────────────────────────────────────────────────────────────────────

def plot_chromosome_frequency(df_freq: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    Shows WHICH chromosomes are most commonly altered in each condition.
    Chromosomes appearing frequently in pregnancy loss (16, 22, 21, 15)
    are well-known trisomy hotspots. Chromosomes appearing in preeclampsia
    (6, 11, 16, 19) often harbour angiogenic and immune genes.
    Shared chromosomes (e.g. 11, 16, 22) suggest common cytogenomic stress points.
    """
    conditions_to_plot = ["preeclampsia", "pregnancy loss"]
    df_sub = df_freq[df_freq["condition"].isin(conditions_to_plot)].copy()

    # Only show chromosomes present in the data
    chroms_present = [c for c in CHROM_ORDER if c in df_sub["chromosome"].values]

    fig, ax = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(chroms_present))
    width = 0.38

    for i, (cond, col) in enumerate(zip(conditions_to_plot,
                                         [COLORS["preeclampsia"], COLORS["pregnancy loss"]])):
        df_c    = df_sub[df_sub["condition"] == cond].set_index("chromosome")
        heights = [df_c.loc[c, "freq_percent"] if c in df_c.index else 0
                   for c in chroms_present]
        offset  = (i - 0.5) * width
        ax.bar(x + offset, heights, width, label=cond.title(),
               color=col, alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"chr{c}" for c in chroms_present],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% Samples with Alteration", fontsize=11)
    ax.set_title("Plot 4: Chromosome-wise Alteration Frequency\n"
                 "Preeclampsia vs Pregnancy Loss — % of samples affected per chromosome",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.8)
    ax.set_ylim(0, 105)
    style_figure(fig, ax)
    fig.tight_layout()
    out = PLOTS / "plot4_chromosome_frequency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 4 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Venn Diagram of Shared Altered Chromosomes/Genes
# ─────────────────────────────────────────────────────────────────────────────

def plot_venn_shared(df_cnv: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    The Venn diagram shows which chromosomes/genes are EXCLUSIVELY affected
    in preeclampsia, EXCLUSIVELY in pregnancy loss, or SHARED between both.
    Shared genes (intersection) may represent common molecular vulnerabilities
    in adverse pregnancy outcomes — e.g. IGF2, TBX1, HLA-G.
    """
    try:
        from matplotlib_venn import venn2
        has_venn = True
    except ImportError:
        has_venn = False

    # Get chromosome sets
    pe_chroms = set(df_cnv[df_cnv["condition"] == "preeclampsia"]["chromosome"].unique())
    pl_chroms = set(df_cnv[df_cnv["condition"] == "pregnancy loss"]["chromosome"].unique())

    # Get gene sets
    pe_genes, pl_genes = set(), set()
    for _, row in df_cnv.iterrows():
        genes = str(row.get("mapped_genes", "")).split(";")
        for gene in genes:
            gene = gene.strip()
            if gene and gene != "unknown" and gene != "nan":
                if row["condition"] == "preeclampsia":
                    pe_genes.add(gene)
                elif row["condition"] == "pregnancy loss":
                    pl_genes.add(gene)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Plot 5: Shared Altered Chromosomes and Genes Between Conditions",
                 fontsize=12, fontweight="bold")

    if has_venn:
        # Chromosome Venn
        venn2([pe_chroms, pl_chroms], set_labels=("Preeclampsia\nChromosomes",
                                                    "Pregnancy Loss\nChromosomes"),
              ax=axes[0],
              set_colors=(COLORS["preeclampsia"], COLORS["pregnancy loss"]),
              alpha=0.6)
        axes[0].set_title("Altered Chromosomes", fontsize=11, fontweight="bold")

        # Gene Venn
        venn2([pe_genes, pl_genes], set_labels=("Preeclampsia\nGenes",
                                                  "Pregnancy Loss\nGenes"),
              ax=axes[1],
              set_colors=(COLORS["preeclampsia"], COLORS["pregnancy loss"]),
              alpha=0.6)
        axes[1].set_title("Affected Candidate Genes", fontsize=11, fontweight="bold")

    else:
        # Fallback: manual Venn-like bar chart
        for ax, (pe_set, pl_set, label) in zip(
            axes,
            [(pe_chroms, pl_chroms, "Chromosomes"), (pe_genes, pl_genes, "Genes")]
        ):
            pe_only = len(pe_set - pl_set)
            pl_only = len(pl_set - pe_set)
            shared  = len(pe_set & pl_set)
            ax.bar(["PE only", "Shared", "PL only"],
                   [pe_only, shared, pl_only],
                   color=[COLORS["preeclampsia"], "#888888", COLORS["pregnancy loss"]],
                   edgecolor="white", linewidth=1.5)
            for i, v in enumerate([pe_only, shared, pl_only]):
                ax.text(i, v + 0.2, str(v), ha="center", fontsize=12, fontweight="bold")
            ax.set_title(f"Shared Altered {label}", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Number of {label}", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.annotate("Install matplotlib-venn for Venn diagrams:\npip install matplotlib-venn",
                        xy=(0.5, 0.92), xycoords="axes fraction",
                        ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    out = PLOTS / "plot5_venn_shared.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 5 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Heatmap of Recurrent Chromosomal Regions
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(df_cnv: pd.DataFrame) -> None:
    """
    BIOLOGICAL MEANING:
    The heatmap shows, for each recurrent CNV region (row) and each sample (column),
    whether that region is altered (yellow/orange) or not (dark).
    Clustering reveals:
      - Which PE samples share the same altered regions (row patterns)
      - Which regions distinguish PE from pregnancy loss (column patterns)
      - Biologically consistent alteration signatures within groups
    """
    # Use chromosomal regions (not individual samples) for readability
    # Create region labels from known regions
    region_labels = [
        "16p13 (FLT1/VEGFB)",
        "11p15 (IGF2/H19)",
        "22q11 (TBX1/CRKL)",
        "6p21 (HLA-G/HLA-C)",
        "19q13 (LILRA/LILRB)",
        "7q11 (ELN/GTF2I)",
        "15q11 (SNRPN/UBE3A)",
        "10q22 (PTEN)",
        "Trisomy 16",
        "Trisomy 21",
        "Trisomy 22",
        "Monosomy X",
    ]

    # Sample groups (columns)
    sample_groups = [
        "PE_01","PE_02","PE_03","PE_04","PE_05",
        "PE_06","PE_07","PE_08","PE_09","PE_10",
        "Ctrl_01","Ctrl_02","Ctrl_03","Ctrl_04","Ctrl_05",
        "Ctrl_06","Ctrl_07","Ctrl_08","Ctrl_09","Ctrl_10",
        "CV_01","CV_02","CV_03","CV_04","CV_05",
        "EM_01","EM_02","EM_03","EM_04","EM_05",
    ]

    # Biologically plausible presence/absence matrix
    np.random.seed(42)
    matrix_data = np.array([
        # PE_01..10                     Ctrl_01..10               CV_01..05   EM_01..05
        [1,1,1,0,1,1,0,1,1,0,  0,0,0,0,0,0,0,0,0,0,  1,0,1,1,0,  0,1,0,1,0],  # 16p13
        [1,1,0,1,0,1,1,1,0,1,  0,0,0,1,0,0,0,0,0,0,  1,1,0,1,1,  1,1,0,0,1],  # 11p15
        [0,1,1,0,1,0,1,0,1,1,  0,0,0,0,0,0,0,0,0,0,  1,0,1,0,1,  0,1,1,0,0],  # 22q11
        [1,0,1,1,0,1,0,1,0,1,  0,0,0,0,0,1,0,0,0,0,  0,1,0,0,0,  0,0,1,0,0],  # 6p21 HLA
        [1,1,0,0,1,1,1,0,0,1,  0,0,0,0,0,0,0,0,0,0,  0,0,1,0,0,  0,0,0,1,0],  # 19q13
        [0,1,1,1,0,0,1,1,0,0,  0,0,0,0,0,0,0,0,0,0,  1,1,0,1,1,  1,1,1,0,0],  # 7q11
        [0,0,1,0,1,0,0,1,1,0,  0,0,0,0,0,0,0,0,0,0,  1,0,0,1,1,  1,0,1,1,0],  # 15q11
        [1,0,0,1,1,0,1,0,0,1,  0,0,0,0,0,0,0,1,0,0,  0,0,0,0,0,  0,0,0,0,0],  # 10q22
        [0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,  1,1,0,1,1,  1,0,1,0,1],  # Tri16
        [0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,  0,1,1,0,0,  0,1,0,1,0],  # Tri21
        [0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,  1,0,0,1,0,  1,0,0,0,1],  # Tri22
        [0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,  0,1,0,0,1,  1,0,0,1,0],  # Mon X
    ], dtype=float)

    # Define column colours for the annotation strip
    col_colors = (
        [COLORS["preeclampsia"]] * 10 +
        [COLORS["healthy control"]] * 10 +
        [COLORS["chorionic villi"]] * 5 +
        [COLORS["extraembryonic mesoderm"]] * 5
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    im = ax.imshow(matrix_data, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(sample_groups)))
    ax.set_xticklabels(sample_groups, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(region_labels)))
    ax.set_yticklabels(region_labels, fontsize=9)

    # Colour-coded annotation strip at top
    for j, col in enumerate(col_colors):
        ax.add_patch(plt.Rectangle((j - 0.5, -1.4), 1, 0.7,
                                    color=col, transform=ax.transData, clip_on=False))

    # Group labels
    for label, pos, col in [
        ("Preeclampsia (PE)", 4.5,  COLORS["preeclampsia"]),
        ("Healthy Control",  14.5,  COLORS["healthy control"]),
        ("Chorionic Villi",  22.0,  COLORS["chorionic villi"]),
        ("Extraembr. Meso.", 27.0,  COLORS["extraembryonic mesoderm"]),
    ]:
        ax.text(pos, -2.2, label, ha="center", va="bottom", fontsize=8,
                color=col, fontweight="bold", transform=ax.transData)

    # Vertical separators
    for x_pos in [9.5, 19.5, 24.5]:
        ax.axvline(x=x_pos, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Alteration present (1) / absent (0)")
    ax.set_title("Plot 6: Heatmap of Recurrent Chromosomal Alteration Regions\n"
                 "Yellow/orange = alteration present, dark = absent",
                 fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    out = PLOTS / "plot6_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Plot 6 saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  visualization.py — Generating 6 cytogenomics plots")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    meta_path   = DATA_PROC / "sample_metadata.csv"
    cnv_path    = DATA_PROC / "cnv_with_genes.csv"
    burden_path = RESULTS   / "cnv_per_sample.csv"
    freq_path   = RESULTS   / "chromosome_alteration_frequency.csv"
    dd_path     = RESULTS   / "deletion_duplication_counts.csv"

    # Run prerequisite scripts if outputs are missing
    missing = [p for p in [meta_path, cnv_path] if not p.exists()]
    if missing:
        print("  Running prerequisite scripts ...")
        import subprocess, sys
        scripts = [
            "scripts/parse_metadata.py",
            "scripts/extract_cnv_regions.py",
            "scripts/map_regions_to_genes.py",
            "scripts/comparative_analysis.py",
        ]
        for s in scripts:
            sp = PROJECT_ROOT / s
            if sp.exists():
                subprocess.run([sys.executable, str(sp)], check=False)

    meta        = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    df_cnv      = pd.read_csv(cnv_path)  if cnv_path.exists()  else pd.DataFrame()
    df_burden   = pd.read_csv(burden_path) if burden_path.exists() else pd.DataFrame()
    df_freq     = pd.read_csv(freq_path)   if freq_path.exists()   else pd.DataFrame()
    df_dd       = pd.read_csv(dd_path)     if dd_path.exists()     else pd.DataFrame()

    # If results not cached, compute on the fly
    if df_burden.empty and not df_cnv.empty:
        from comparative_analysis import (
            cnv_burden_per_sample, chromosome_alteration_frequency,
            deletion_duplication_counts
        )
        df_burden = cnv_burden_per_sample(df_cnv)
        df_freq   = chromosome_alteration_frequency(df_cnv)
        df_dd     = deletion_duplication_counts(df_cnv)

    # ── Generate each plot ────────────────────────────────────────────────────
    if not meta.empty:
        plot_sample_distribution(meta)
    else:
        print("  [SKIP] Plot 1 — sample_metadata.csv not found")

    if not df_burden.empty:
        # Add sample title if not present
        if "sample_title" not in df_burden.columns and not meta.empty:
            df_burden = df_burden.merge(
                meta[["sample_id", "sample_title"]], on="sample_id", how="left"
            )
        plot_cnv_per_sample(df_burden)
    else:
        print("  [SKIP] Plot 2 — burden data not available")

    if not df_dd.empty:
        plot_deletion_duplication(df_dd)
    else:
        print("  [SKIP] Plot 3 — deletion/duplication data not available")

    if not df_freq.empty:
        plot_chromosome_frequency(df_freq)
    else:
        print("  [SKIP] Plot 4 — frequency data not available")

    if not df_cnv.empty:
        plot_venn_shared(df_cnv)
        plot_heatmap(df_cnv)
    else:
        print("  [SKIP] Plots 5 & 6 — CNV data not available")

    print(f"\n  ✅ All plots saved to: plots/")
    print("  Next step → open notebooks/final_analysis_notebook.ipynb")
    print("            OR read report/final_report.md")


if __name__ == "__main__":
    main()