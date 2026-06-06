"""
map_regions_to_genes.py
========================
Maps chromosomal CNV regions to known genes using a curated gene annotation table.

Approach:
  1. Uses a built-in curated gene list covering key cytogenomic loci
     (sourced from NCBI Gene, Ensembl GRCh38, UCSC genome browser).
  2. For each CNV region, identifies overlapping genes by genomic coordinate intersection.
  3. Highlights candidate genes relevant to:
       - Placental development & trophoblast invasion
       - Angiogenesis & vascular function
       - Immune regulation (NK cell, HLA)
       - Fetal development & imprinting

Outputs:
  - data/processed/cnv_with_genes.csv
  - data/processed/gene_cnv_matrix.csv
  - data/processed/candidate_gene_summary.csv

How to run:
    python scripts/map_regions_to_genes.py

For a full gene annotation, students can download:
  UCSC knownGene: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/knownGene.txt.gz
  Ensembl GTF:    https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/
  NCBI RefSeq:    https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# CURATED GENE ANNOTATION TABLE (GRCh38 coordinates, key cytogenomic genes)
# Includes genes relevant to placentation, angiogenesis, immune function
# ─────────────────────────────────────────────────────────────────────────────

GENE_ANNOTATION = [
    # (gene_symbol, chrom, tx_start, tx_end, biotype, pathway_relevance)
    # Angiogenesis / vascular
    ("FLT1",    "13", 28003744,  28387741, "protein_coding", "angiogenesis"),
    ("VEGFB",   "11", 64033716,  64062165, "protein_coding", "angiogenesis"),
    ("VEGFC",   "4",  177723215, 177826858,"protein_coding", "angiogenesis"),
    ("KDR",     "4",  55084229,  55164414, "protein_coding", "angiogenesis"),
    ("ANGPT1",  "8",  108915614, 109017975,"protein_coding", "angiogenesis"),
    ("ANGPT2",  "8",  6294693,   6377073,  "protein_coding", "angiogenesis"),
    ("ENG",     "9",  127119482, 127199697,"protein_coding", "angiogenesis"),
    ("PLGF",    "14", 75188490,  75194476, "protein_coding", "angiogenesis"),
    ("NOS3",    "7",  150982340, 151013734,"protein_coding", "angiogenesis"),
    # Trophoblast invasion / placenta
    ("IGF2",    "11", 2154781,   2168694,  "protein_coding", "trophoblast_invasion"),
    ("H19",     "11", 1995025,   1997581,  "lncRNA",         "trophoblast_invasion"),
    ("PHLDA2",  "11", 2906765,   2912022,  "protein_coding", "trophoblast_invasion"),
    ("CDKN1C",  "11", 2907895,   2913908,  "protein_coding", "trophoblast_invasion"),
    ("KCNQ1",   "11", 2445150,   2850748,  "protein_coding", "trophoblast_invasion"),
    ("ELN",     "7",  73443670,  73544303, "protein_coding", "trophoblast_invasion"),
    ("LIMK1",   "7",  73111218,  73173869, "protein_coding", "trophoblast_invasion"),
    ("GTF2I",   "7",  74066604,  74277001, "protein_coding", "trophoblast_invasion"),
    ("MMP2",    "16", 55514614,  55583786, "protein_coding", "trophoblast_invasion"),
    ("MMP9",    "20", 46010108,  46019376, "protein_coding", "trophoblast_invasion"),
    # Immune regulation
    ("HLA-G",   "6",  29794756,  29798890, "protein_coding", "immune_regulation"),
    ("HLA-C",   "6",  31268749,  31272130, "protein_coding", "immune_regulation"),
    ("HLA-A",   "6",  29910248,  29913728, "protein_coding", "immune_regulation"),
    ("LILRA1",  "19", 54223696,  54241867, "protein_coding", "immune_regulation"),
    ("LILRB1",  "19", 54156174,  54201074, "protein_coding", "immune_regulation"),
    ("LILRB2",  "19", 54102665,  54151060, "protein_coding", "immune_regulation"),
    ("KIR2DL1", "19", 54602162,  54617765, "protein_coding", "immune_regulation"),
    ("C1QB",    "1",  22944831,  22948455, "protein_coding", "immune_regulation"),
    ("PDCD1",   "2",  241849902, 241858908,"protein_coding", "immune_regulation"),
    # Fetal development / imprinting
    ("SNRPN",   "15", 24870306,  25170728, "protein_coding", "fetal_development"),
    ("UBE3A",   "15", 25282444,  25591756, "protein_coding", "fetal_development"),
    ("GABRB3",  "15", 26684676,  27057437, "protein_coding", "fetal_development"),
    ("MECP2",   "X",  154021573, 154097717,"protein_coding", "fetal_development"),
    ("FMRP",    "X",  147911919, 148137650,"protein_coding", "fetal_development"),
    ("SOX2",    "3",  181711922, 181714436,"protein_coding", "fetal_development"),
    ("OCT4",    "6",  31097163,  31109480, "protein_coding", "fetal_development"),
    ("NANOG",   "12", 7787018,   7793376,  "protein_coding", "fetal_development"),
    # Cardiovascular / DiGeorge
    ("TBX1",    "22", 19744226,  19771898, "protein_coding", "cardiovascular"),
    ("CRKL",    "22", 21788115,  21882521, "protein_coding", "cardiovascular"),
    ("HIRA",    "22", 19073802,  19240168, "protein_coding", "cardiovascular"),
    ("SCN5A",   "3",  38551702,  38691864, "protein_coding", "cardiovascular"),
    # Tumour suppressors / cell cycle
    ("PTEN",    "10", 89692905,  89731687, "protein_coding", "tumour_suppressor"),
    ("BRCA1",   "17", 43044295,  43125483, "protein_coding", "tumour_suppressor"),
    ("TP53",    "17", 7661779,   7687538,  "protein_coding", "tumour_suppressor"),
    ("CDKN2A",  "9",  21967751,  21994445, "protein_coding", "tumour_suppressor"),
    # Williams syndrome / neurodevelopment
    ("RFC2",    "7",  73870765,  73941100, "protein_coding", "neurodevelopment"),
    ("CLIP2",   "7",  73710399,  73762813, "protein_coding", "neurodevelopment"),
    ("GTF2IRD1","7",  73884977,  74045041, "protein_coding", "neurodevelopment"),
    # Cri-du-chat / 5p
    ("TERT",    "5",  1253262,   1295184,  "protein_coding", "telomere_maintenance"),
    ("SEMA5A",  "5",  9313578,   9559516,  "protein_coding", "neurodevelopment"),
    # Wolf-Hirschhorn / 4p
    ("FGFRL1",  "4",  1027884,   1051337,  "protein_coding", "fetal_development"),
    ("WHSC1",   "4",  1871673,   2033354,  "protein_coding", "fetal_development"),
    ("MSX1",    "4",  4861413,   4868389,  "protein_coding", "fetal_development"),
    # Leukemia / growth
    ("PDGFRA",  "4",  54229076,  54298058, "protein_coding", "growth_factor"),
    ("KIT",     "4",  54658039,  54740109, "protein_coding", "growth_factor"),
    ("LEPR",    "1",  65421140,  65642811, "protein_coding", "metabolism"),
    ("SMN1",    "5",  70924941,  70953015, "protein_coding", "neurodevelopment"),
    ("SMN2",    "5",  70049523,  70077594, "protein_coding", "neurodevelopment"),
]


def overlaps(cnv_start: int, cnv_end: int, gene_start: int, gene_end: int) -> bool:
    """Check if a CNV region overlaps with a gene's coordinates."""
    return cnv_start < gene_end and cnv_end > gene_start


def map_cnv_to_genes(cnv_df: pd.DataFrame, gene_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each CNV region, find overlapping annotated genes.
    Returns an enriched CNV table with gene annotations added.
    """
    gene_df["chrom"] = gene_df["chrom"].astype(str)
    cnv_df["chromosome"] = cnv_df["chromosome"].astype(str)

    enriched_rows = []
    for _, cnv in cnv_df.iterrows():
        chrom = cnv["chromosome"]
        start = int(cnv["start"])
        end   = int(cnv["end"])

        # Find genes on the same chromosome
        chrom_genes = gene_df[gene_df["chrom"] == chrom]

        overlap_genes    = []
        pathway_hits     = []
        is_whole_chrom   = (start < 1_000_000 and end > CHROM_SIZES_MAP.get(chrom, 0) * 0.8)

        if is_whole_chrom:
            # Whole chromosome alteration → mark all genes on that chrom
            for _, g in chrom_genes.iterrows():
                overlap_genes.append(g["gene_symbol"])
                pathway_hits.append(g["pathway_relevance"])
        else:
            for _, g in chrom_genes.iterrows():
                if overlaps(start, end, int(g["tx_start"]), int(g["tx_end"])):
                    overlap_genes.append(g["gene_symbol"])
                    pathway_hits.append(g["pathway_relevance"])

        row = cnv.to_dict()
        row["mapped_genes"]     = ";".join(overlap_genes) if overlap_genes else row.get("genes", "")
        row["pathway_hits"]     = ";".join(sorted(set(pathway_hits)))
        row["n_mapped_genes"]   = len(overlap_genes)
        row["is_whole_chrom"]   = is_whole_chrom
        row["has_candidate_gene"] = len(overlap_genes) > 0
        enriched_rows.append(row)

    return pd.DataFrame(enriched_rows)


def get_candidate_gene_summary(cnv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-gene summary table showing which conditions each gene is
    affected in, and how frequently.
    """
    records = []
    for _, row in cnv_df.iterrows():
        genes = str(row.get("mapped_genes", "")).split(";")
        for gene in genes:
            gene = gene.strip()
            if not gene or gene == "unknown":
                continue
            records.append({
                "gene_symbol": gene,
                "dataset"    : row["dataset"],
                "condition"  : row["condition"],
                "sample_id"  : row["sample_id"],
                "cnv_type"   : row["cnv_type"],
                "chromosome" : row["chromosome"],
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["gene_symbol", "condition"])
        .agg(
            n_samples=("sample_id", "nunique"),
            cnv_types=("cnv_type", lambda x: ";".join(sorted(set(x)))),
            chromosomes=("chromosome", lambda x: ";".join(sorted(set(x)))),
        )
        .reset_index()
    )
    return summary


CHROM_SIZES_MAP = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}


def main():
    print("\n" + "=" * 60)
    print("  map_regions_to_genes.py — Mapping CNV regions to genes")
    print("=" * 60)

    cnv_path = DATA_PROC / "cnv_regions.csv"
    if not cnv_path.exists():
        print("  ERROR: cnv_regions.csv not found.")
        print("  Please run extract_cnv_regions.py first.")
        return

    df_cnv  = pd.read_csv(cnv_path)
    df_genes = pd.DataFrame(
        GENE_ANNOTATION,
        columns=["gene_symbol", "chrom", "tx_start", "tx_end", "biotype", "pathway_relevance"]
    )

    print(f"  Loaded {len(df_cnv)} CNV regions, {len(df_genes)} annotated genes")
    print("  Running coordinate overlap mapping ...")

    df_enriched = map_cnv_to_genes(df_cnv, df_genes)

    # Save enriched CNV table
    out1 = DATA_PROC / "cnv_with_genes.csv"
    df_enriched.to_csv(out1, index=False)
    print(f"\n  ✅ Saved: {out1.relative_to(PROJECT_ROOT)}")

    # ── Candidate gene summary ────────────────────────────────────────────────
    df_gene_summary = get_candidate_gene_summary(df_enriched)
    if not df_gene_summary.empty:
        out2 = DATA_PROC / "candidate_gene_summary.csv"
        df_gene_summary.to_csv(out2, index=False)
        print(f"  ✅ Saved: {out2.relative_to(PROJECT_ROOT)}")

    # ── Gene-sample matrix (presence/absence) ────────────────────────────────
    gene_list = []
    for _, row in df_enriched.iterrows():
        for gene in str(row.get("mapped_genes", "")).split(";"):
            gene = gene.strip()
            if gene and gene != "unknown":
                gene_list.append((gene, row["sample_id"], row["condition"]))

    if gene_list:
        df_gm = pd.DataFrame(gene_list, columns=["gene", "sample_id", "condition"])
        matrix = df_gm.pivot_table(index="gene", columns="sample_id",
                                    aggfunc="size", fill_value=0).clip(upper=1)
        out3 = DATA_PROC / "gene_cnv_matrix.csv"
        matrix.to_csv(out3)
        print(f"  ✅ Saved: {out3.relative_to(PROJECT_ROOT)}")
        print(f"     Matrix: {matrix.shape[0]} genes × {matrix.shape[1]} samples")

    # ── Top candidate genes ───────────────────────────────────────────────────
    print("\n  Top Candidate Genes (by number of affected samples):")
    if not df_gene_summary.empty:
        top = df_gene_summary.sort_values("n_samples", ascending=False).head(15)
        print(top[["gene_symbol", "condition", "n_samples", "cnv_types"]].to_string(index=False))

    # ── Shared genes between conditions ──────────────────────────────────────
    print("\n  Genes shared between Preeclampsia and Pregnancy Loss:")
    if not df_gene_summary.empty:
        pe_genes = set(df_gene_summary[df_gene_summary["condition"] == "preeclampsia"]["gene_symbol"])
        pl_genes = set(df_gene_summary[df_gene_summary["condition"] == "pregnancy loss"]["gene_symbol"])
        shared   = pe_genes & pl_genes
        print(f"    Preeclampsia genes   : {len(pe_genes)}")
        print(f"    Pregnancy loss genes : {len(pl_genes)}")
        print(f"    Shared genes         : {len(shared)}")
        if shared:
            print(f"    Shared gene list     : {', '.join(sorted(shared))}")

    print("\n  Next step → run:  python scripts/comparative_analysis.py")


if __name__ == "__main__":
    main()