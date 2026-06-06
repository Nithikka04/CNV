"""
comparative_analysis.py
========================
Performs chromosomal-level comparative analysis between:
  - Preeclampsia vs healthy controls (GSE192614)
  - Pregnancy loss fetal tissue vs parental background (GSE228150)
  - Preeclampsia altered regions vs pregnancy loss altered regions (cross-dataset)

Outputs:
  - results/cnv_per_sample.csv
  - results/chromosome_alteration_frequency.csv
  - results/deletion_duplication_counts.csv
  - results/shared_chromosomes.csv
  - results/shared_genes.csv
  - results/condition_specific_regions.csv

How to run:
    python scripts/comparative_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS      = PROJECT_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CHROM_ORDER = {str(i): i for i in range(1, 23)}
CHROM_ORDER.update({"X": 23, "Y": 24})


def cnv_burden_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Count CNVs per sample, broken down by deletion/gain."""
    summary = (
        df.groupby(["sample_id", "condition", "dataset", "tissue_type"])
        .agg(
            total_cnvs   =("cnv_type", "count"),
            deletions    =("cnv_type", lambda x: (x.str.contains("deletion|loss", na=False)).sum()),
            duplications =("cnv_type", lambda x: (x.str.contains("duplication|gain", na=False)).sum()),
            mosaic       =("cnv_type", lambda x: (x.str.contains("mosaic", na=False)).sum()),
            total_bp     =("size_bp", "sum"),
            n_chromosomes=("chromosome", "nunique"),
        )
        .reset_index()
    )
    return summary


def chromosome_alteration_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each condition and chromosome, calculate the fraction of samples
    that carry at least one CNV on that chromosome.
    """
    # Only include primary disease conditions (not parental background)
    conditions_of_interest = ["preeclampsia", "healthy control", "pregnancy loss"]
    df_sub = df[df["condition"].isin(conditions_of_interest)].copy()

    results = []
    for condition in conditions_of_interest:
        df_cond  = df_sub[df_sub["condition"] == condition]
        n_samples = df_cond["sample_id"].nunique()
        if n_samples == 0:
            continue
        for chrom in list(map(str, range(1, 23))) + ["X", "Y"]:
            df_chrom     = df_cond[df_cond["chromosome"] == chrom]
            n_affected   = df_chrom["sample_id"].nunique()
            freq_percent = round(100 * n_affected / n_samples, 1)
            results.append({
                "condition"      : condition,
                "chromosome"     : chrom,
                "chrom_order"    : CHROM_ORDER.get(chrom, 99),
                "n_affected"     : n_affected,
                "n_total_samples": n_samples,
                "freq_percent"   : freq_percent,
            })

    df_result = pd.DataFrame(results).sort_values(["chrom_order", "condition"])
    return df_result


def deletion_duplication_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise deletion/duplication/mosaic counts per condition."""
    conditions = ["preeclampsia", "healthy control", "pregnancy loss"]
    df_sub = df[df["condition"].isin(conditions)].copy()

    def classify(cnv_type: str) -> str:
        t = str(cnv_type).lower()
        if "mosaic" in t:
            return "mosaic"
        if "deletion" in t or "loss" in t:
            return "deletion/loss"
        if "duplication" in t or "gain" in t:
            return "duplication/gain"
        return "other"

    df_sub["alteration_class"] = df_sub["cnv_type"].apply(classify)

    summary = (
        df_sub.groupby(["condition", "alteration_class"])
        .size()
        .reset_index(name="count")
    )
    return summary


def find_shared_chromosomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find chromosomes altered in both preeclampsia AND pregnancy loss.
    Reports frequency in each condition.
    """
    pe_chroms = set(
        df[df["condition"] == "preeclampsia"]["chromosome"].unique()
    )
    pl_chroms = set(
        df[df["condition"] == "pregnancy loss"]["chromosome"].unique()
    )
    ctrl_chroms = set(
        df[df["condition"] == "healthy control"]["chromosome"].unique()
    )

    shared = pe_chroms & pl_chroms
    records = []
    for chrom in sorted(shared, key=lambda x: CHROM_ORDER.get(x, 99)):
        records.append({
            "chromosome"           : chrom,
            "in_preeclampsia"      : chrom in pe_chroms,
            "in_pregnancy_loss"    : chrom in pl_chroms,
            "in_healthy_control"   : chrom in ctrl_chroms,
            "pe_only"              : chrom in pe_chroms - pl_chroms,
            "pl_only"              : chrom in pl_chroms - pe_chroms,
            "shared"               : chrom in shared,
        })
    # Also add condition-specific chromosomes
    for chrom in sorted((pe_chroms | pl_chroms) - shared, key=lambda x: CHROM_ORDER.get(x, 99)):
        records.append({
            "chromosome"         : chrom,
            "in_preeclampsia"    : chrom in pe_chroms,
            "in_pregnancy_loss"  : chrom in pl_chroms,
            "in_healthy_control" : chrom in ctrl_chroms,
            "pe_only"            : chrom in pe_chroms - pl_chroms,
            "pl_only"            : chrom in pl_chroms - pe_chroms,
            "shared"             : False,
        })
    return pd.DataFrame(records)


def find_shared_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Identify genes affected in both preeclampsia and pregnancy loss."""
    gene_records = []
    for _, row in df.iterrows():
        genes = str(row.get("mapped_genes", "")).split(";")
        for gene in genes:
            gene = gene.strip()
            if gene and gene != "unknown" and gene != "nan":
                gene_records.append({
                    "gene"       : gene,
                    "condition"  : row["condition"],
                    "dataset"    : row["dataset"],
                    "cnv_type"   : row["cnv_type"],
                    "chromosome" : row["chromosome"],
                })

    if not gene_records:
        return pd.DataFrame()

    df_genes = pd.DataFrame(gene_records)
    pe_genes = set(df_genes[df_genes["condition"] == "preeclampsia"]["gene"])
    pl_genes = set(df_genes[df_genes["condition"] == "pregnancy loss"]["gene"])
    shared   = pe_genes & pl_genes

    records = []
    for gene in sorted(shared):
        pe_info = df_genes[(df_genes["gene"] == gene) & (df_genes["condition"] == "preeclampsia")]
        pl_info = df_genes[(df_genes["gene"] == gene) & (df_genes["condition"] == "pregnancy loss")]
        records.append({
            "gene"          : gene,
            "chromosome"    : pe_info["chromosome"].iloc[0] if not pe_info.empty else "?",
            "pe_cnv_types"  : ";".join(sorted(pe_info["cnv_type"].unique())) if not pe_info.empty else "",
            "pl_cnv_types"  : ";".join(sorted(pl_info["cnv_type"].unique())) if not pl_info.empty else "",
            "shared"        : True,
        })
    return pd.DataFrame(records)


def condition_specific_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise condition-specific vs shared chromosomal alteration patterns."""
    df_pe = df[df["condition"] == "preeclampsia"]
    df_pl = df[df["condition"] == "pregnancy loss"]

    pe_chroms = set(df_pe["chromosome"].unique())
    pl_chroms = set(df_pl["chromosome"].unique())
    shared    = pe_chroms & pl_chroms

    records = [
        {"category": "PE-specific chromosomes",    "count": len(pe_chroms - pl_chroms), "chromosomes": str(sorted(pe_chroms - pl_chroms, key=lambda x: CHROM_ORDER.get(x,99)))},
        {"category": "PL-specific chromosomes",    "count": len(pl_chroms - pe_chroms), "chromosomes": str(sorted(pl_chroms - pe_chroms, key=lambda x: CHROM_ORDER.get(x,99)))},
        {"category": "Shared chromosomes",         "count": len(shared),                "chromosomes": str(sorted(shared,                key=lambda x: CHROM_ORDER.get(x,99)))},
        {"category": "Total PE alterations",       "count": len(df_pe),                 "chromosomes": ""},
        {"category": "Total PL alterations",       "count": len(df_pl),                 "chromosomes": ""},
        {"category": "PE deletion fraction (%)",   "count": round(100 * df_pe["cnv_type"].str.contains("deletion|loss",na=False).sum() / max(len(df_pe),1), 1), "chromosomes": ""},
        {"category": "PL deletion fraction (%)",   "count": round(100 * df_pl["cnv_type"].str.contains("deletion|loss",na=False).sum() / max(len(df_pl),1), 1), "chromosomes": ""},
    ]
    return pd.DataFrame(records)


def main():
    print("\n" + "=" * 60)
    print("  comparative_analysis.py — Chromosomal-level comparison")
    print("=" * 60)

    # ── Load enriched CNV data ────────────────────────────────────────────────
    cnv_path = DATA_PROC / "cnv_with_genes.csv"
    if not cnv_path.exists():
        # Fall back to basic CNV table
        cnv_path = DATA_PROC / "cnv_regions.csv"
    if not cnv_path.exists():
        print("  ERROR: No CNV data found. Run previous scripts first.")
        return

    df = pd.read_csv(cnv_path)
    print(f"  Loaded {len(df)} CNV entries from {cnv_path.name}")

    # ── 1. CNV burden per sample ──────────────────────────────────────────────
    df_burden = cnv_burden_per_sample(df)
    out1 = RESULTS / "cnv_per_sample.csv"
    df_burden.to_csv(out1, index=False)
    print(f"\n  [1] CNV burden per sample → {out1.relative_to(PROJECT_ROOT)}")
    print(df_burden[["sample_id","condition","total_cnvs","deletions","duplications","mosaic"]].to_string(index=False))

    # ── 2. Chromosome-wise alteration frequency ───────────────────────────────
    df_freq = chromosome_alteration_frequency(df)
    out2 = RESULTS / "chromosome_alteration_frequency.csv"
    df_freq.to_csv(out2, index=False)
    print(f"\n  [2] Chromosome alteration frequency → {out2.relative_to(PROJECT_ROOT)}")
    print("      Top 10 most-altered chromosomes:")
    top_chroms = df_freq.sort_values("n_affected", ascending=False).head(10)
    print(top_chroms[["condition","chromosome","n_affected","n_total_samples","freq_percent"]].to_string(index=False))

    # ── 3. Deletion vs duplication counts ────────────────────────────────────
    df_dd = deletion_duplication_counts(df)
    out3 = RESULTS / "deletion_duplication_counts.csv"
    df_dd.to_csv(out3, index=False)
    print(f"\n  [3] Deletion/duplication counts → {out3.relative_to(PROJECT_ROOT)}")
    print(df_dd.to_string(index=False))

    # ── 4. Shared chromosomes ────────────────────────────────────────────────
    df_shared_chroms = find_shared_chromosomes(df)
    out4 = RESULTS / "shared_chromosomes.csv"
    df_shared_chroms.to_csv(out4, index=False)
    shared_count = df_shared_chroms["shared"].sum()
    print(f"\n  [4] Shared chromosomes → {out4.relative_to(PROJECT_ROOT)}")
    print(f"      {shared_count} chromosomes altered in BOTH conditions")

    # ── 5. Shared genes ──────────────────────────────────────────────────────
    df_shared_genes = find_shared_genes(df)
    if not df_shared_genes.empty:
        out5 = RESULTS / "shared_genes.csv"
        df_shared_genes.to_csv(out5, index=False)
        print(f"\n  [5] Shared genes → {out5.relative_to(PROJECT_ROOT)}")
        print(f"      {len(df_shared_genes)} genes shared between PE and pregnancy loss:")
        print(df_shared_genes[["gene","chromosome","pe_cnv_types","pl_cnv_types"]].to_string(index=False))

    # ── 6. Condition-specific summary ────────────────────────────────────────
    df_spec = condition_specific_summary(df)
    out6 = RESULTS / "condition_specific_regions.csv"
    df_spec.to_csv(out6, index=False)
    print(f"\n  [6] Condition-specific summary → {out6.relative_to(PROJECT_ROOT)}")
    print(df_spec.to_string(index=False))

    print("\n  ✅ All comparative analysis results saved to results/")
    print("  Next step → run:  python scripts/visualization.py")


if __name__ == "__main__":
    main()