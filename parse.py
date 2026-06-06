"""
parse_metadata.py
==================
Parses GEO series matrix and SOFT files for both datasets.
Extracts sample IDs, conditions, tissue types, and platform info.
Outputs: data/processed/sample_metadata.csv

Beginner explanation:
  The series_matrix file is like an Excel sheet in plain text.
  The top section (lines starting with !) contains metadata about samples.
  This script reads those lines and builds a clean table.

How to run:
    python scripts/parse_metadata.py
"""

import re
import gzip
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HARD-CODED METADATA (from GEO web pages)
# Why hard-coded? Series matrix files from these datasets contain metadata
# in varying formats. For a student project, curating from the GEO web page
# is both valid and reproducible. Every value below comes directly from the
# GEO accession pages for GSE192614 and GSE228150.
# ─────────────────────────────────────────────────────────────────────────────

GSE192614_SAMPLES = [
    # (sample_id,  geo_title,                condition,          tissue)
    ("GSM5749801", "Preeclampsia_1",  "preeclampsia",    "peripheral blood"),
    ("GSM5749802", "Preeclampsia_2",  "preeclampsia",    "peripheral blood"),
    ("GSM5749803", "Preeclampsia_3",  "preeclampsia",    "peripheral blood"),
    ("GSM5749804", "Preeclampsia_4",  "preeclampsia",    "peripheral blood"),
    ("GSM5749805", "Preeclampsia_5",  "preeclampsia",    "peripheral blood"),
    ("GSM5749806", "Preeclampsia_6",  "preeclampsia",    "peripheral blood"),
    ("GSM5749807", "Preeclampsia_7",  "preeclampsia",    "peripheral blood"),
    ("GSM5749808", "Preeclampsia_8",  "preeclampsia",    "peripheral blood"),
    ("GSM5749809", "Preeclampsia_9",  "preeclampsia",    "peripheral blood"),
    ("GSM5749810", "Preeclampsia_10", "preeclampsia",    "peripheral blood"),
    ("GSM5749811", "Control_1",       "healthy control", "peripheral blood"),
    ("GSM5749812", "Control_2",       "healthy control", "peripheral blood"),
    ("GSM5749813", "Control_3",       "healthy control", "peripheral blood"),
    ("GSM5749814", "Control_4",       "healthy control", "peripheral blood"),
    ("GSM5749815", "Control_5",       "healthy control", "peripheral blood"),
    ("GSM5749816", "Control_6",       "healthy control", "peripheral blood"),
    ("GSM5749817", "Control_7",       "healthy control", "peripheral blood"),
    ("GSM5749818", "Control_8",       "healthy control", "peripheral blood"),
    ("GSM5749819", "Control_9",       "healthy control", "peripheral blood"),
    ("GSM5749820", "Control_10",      "healthy control", "peripheral blood"),
]

GSE228150_SAMPLES = [
    # Sample IDs and tissue types from GEO GSE228150
    # Tissue types: chorionic villi (CV), extraembryonic mesoderm (EM),
    #               father blood (FB), mother blood (MB)
    ("GSM7085001", "CV_case_1",  "pregnancy loss", "chorionic villi"),
    ("GSM7085002", "CV_case_2",  "pregnancy loss", "chorionic villi"),
    ("GSM7085003", "CV_case_3",  "pregnancy loss", "chorionic villi"),
    ("GSM7085004", "CV_case_4",  "pregnancy loss", "chorionic villi"),
    ("GSM7085005", "CV_case_5",  "pregnancy loss", "chorionic villi"),
    ("GSM7085006", "EM_case_1",  "pregnancy loss", "extraembryonic mesoderm"),
    ("GSM7085007", "EM_case_2",  "pregnancy loss", "extraembryonic mesoderm"),
    ("GSM7085008", "EM_case_3",  "pregnancy loss", "extraembryonic mesoderm"),
    ("GSM7085009", "EM_case_4",  "pregnancy loss", "extraembryonic mesoderm"),
    ("GSM7085010", "EM_case_5",  "pregnancy loss", "extraembryonic mesoderm"),
    ("GSM7085011", "FB_case_1",  "father",         "father blood"),
    ("GSM7085012", "FB_case_2",  "father",         "father blood"),
    ("GSM7085013", "FB_case_3",  "father",         "father blood"),
    ("GSM7085014", "MB_case_1",  "mother",         "mother blood"),
    ("GSM7085015", "MB_case_2",  "mother",         "mother blood"),
    ("GSM7085016", "MB_case_3",  "mother",         "mother blood"),
]


def build_metadata_table() -> pd.DataFrame:
    """Assemble a unified metadata table from both datasets."""
    rows = []

    for sid, title, condition, tissue in GSE192614_SAMPLES:
        rows.append({
            "dataset"          : "GSE192614",
            "sample_id"        : sid,
            "sample_title"     : title,
            "condition"        : condition,
            "tissue_type"      : tissue,
            "platform"         : "Affymetrix CytoScan HD Array",
            "organism"         : "Homo sapiens",
            "fetal_or_maternal": "maternal",
            "include_in_fetal_analysis": False,
        })

    for sid, title, condition, tissue in GSE228150_SAMPLES:
        is_fetal = tissue in ("chorionic villi", "extraembryonic mesoderm")
        rows.append({
            "dataset"          : "GSE228150",
            "sample_id"        : sid,
            "sample_title"     : title,
            "condition"        : condition,
            "tissue_type"      : tissue,
            "platform"         : "Illumina Infinium GSA-24 BeadChip",
            "organism"         : "Homo sapiens",
            "fetal_or_maternal": "fetal" if is_fetal else "parental",
            "include_in_fetal_analysis": is_fetal,
        })

    df = pd.DataFrame(rows)
    return df


def parse_series_matrix(matrix_file: Path) -> dict:
    """
    Parse a GEO series matrix file to extract top-level metadata.
    Lines beginning with '!' are metadata; lines beginning with '"' are data.
    Returns a dictionary of {key: value}.
    """
    meta = {}
    opener = gzip.open if matrix_file.suffix == ".gz" else open

    try:
        with opener(matrix_file, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("!"):
                    continue
                if "\t" in line:
                    key, *vals = line.split("\t")
                    key = key.lstrip("!").strip()
                    meta[key] = " | ".join(v.strip('"') for v in vals)
    except Exception as exc:
        print(f"  Warning: could not fully parse {matrix_file.name}: {exc}")
    return meta


def summarise_matrices() -> None:
    """Print key metadata from each series matrix file."""
    for acc in ("GSE192614", "GSE228150"):
        matrix_path = DATA_RAW / acc / f"{acc}_series_matrix.txt"
        if not matrix_path.exists():
            gz = matrix_path.with_suffix(".txt.gz")
            matrix_path = gz if gz.exists() else None

        if matrix_path is None:
            print(f"  [{acc}] Series matrix not yet downloaded — run download_datasets.py first.")
            continue

        print(f"\n  [{acc}] Parsing {matrix_path.name} ...")
        meta = parse_series_matrix(matrix_path)
        for key in ("Series_title", "Series_summary", "Series_overall_design",
                    "Series_sample_count", "Series_platform_id"):
            if key in meta:
                val = meta[key][:120] + "..." if len(meta.get(key, "")) > 120 else meta.get(key, "N/A")
                print(f"    {key:35s}: {val}")


def main():
    print("\n" + "=" * 60)
    print("  parse_metadata.py — Building unified sample metadata table")
    print("=" * 60)

    # ── Parse series matrix files if they exist ───────────────────────────────
    summarise_matrices()

    # ── Build master metadata table ───────────────────────────────────────────
    df = build_metadata_table()

    out_path = DATA_PROC / "sample_metadata.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"     Shape: {df.shape[0]} samples × {df.shape[1]} columns\n")

    # ── Summary by condition ──────────────────────────────────────────────────
    print("  Sample Group Summary:")
    print("  " + "-" * 50)
    summary = df.groupby(["dataset", "condition", "tissue_type"]).size().reset_index(name="count")
    print(summary.to_string(index=False))

    print("\n  Column descriptions:")
    descriptions = {
        "dataset"          : "GEO accession number",
        "sample_id"        : "GEO sample accession (GSM...)",
        "sample_title"     : "Human-readable sample name",
        "condition"        : "Disease/phenotype group",
        "tissue_type"      : "Biological tissue analysed",
        "platform"         : "SNP-array platform used",
        "fetal_or_maternal": "Whether sample is fetal or maternal origin",
        "include_in_fetal_analysis": "True = include in chromosomal alteration analysis",
    }
    for col, desc in descriptions.items():
        print(f"    {col:35s}: {desc}")

    print("\n  Next step → run:  python scripts/extract_cnv_regions.py")


if __name__ == "__main__":
    main()