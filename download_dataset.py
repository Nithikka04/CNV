"""
download_datasets.py
=====================
Downloads GSE192614 (Preeclampsia) and GSE228150 (Pregnancy Loss) from NCBI GEO.

What this script does:
- Downloads series matrix files (contain sample metadata + any processed values)
- Downloads supplementary files (may contain CNV calls, processed tables)
- Saves everything into data/raw/GSE192614/ and data/raw/GSE228150/

How to run:
    python scripts/download_datasets.py

Requirements:
    pip install requests tqdm

IMPORTANT NOTE FOR STUDENTS:
Raw SNP-array files (.CEL for Affymetrix, .idat for Illumina) are very large
(hundreds of MB to several GB) and require vendor software to process:
  - Affymetrix CytoScan HD  → requires Chromosome Analysis Suite (ChAS)
  - Illumina GSA BeadChip   → requires GenomeStudio or BlueFuse Multi

This script downloads the PROCESSED files available on GEO, which are
suitable for a student-level cytogenomics project.
"""

import os
import gzip
import shutil
import requests
from pathlib import Path
from urllib.request import urlretrieve

# ── Project root (works regardless of where you run the script from) ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"

# ── GEO FTP base URLs ─────────────────────────────────────────────────────────
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

DATASETS = {
    "GSE192614": {
        "folder"    : "GSE192nnn",
        "accession" : "GSE192614",
        "condition" : "Preeclampsia",
        "platform"  : "Affymetrix CytoScan HD Array",
        "files": [
            # Series matrix  → sample metadata + any processed signal values
            "matrix/GSE192614_series_matrix.txt.gz",
            # Supplementary files — may include CNV summary tables
            # We list known/expected files; script will skip if 404
            "suppl/GSE192614_RAW.tar",
        ],
    },
    "GSE228150": {
        "folder"    : "GSE228nnn",
        "accession" : "GSE228150",
        "condition" : "Pregnancy Loss",
        "platform"  : "Illumina Infinium GSA-24 BeadChip",
        "files": [
            "matrix/GSE228150_series_matrix.txt.gz",
            "suppl/GSE228150_RAW.tar",
        ],
    },
}

# ── Soft files (full metadata in structured format) ───────────────────────────
SOFT_URL_TEMPLATE = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{acc}/soft/{acc}_family.soft.gz"
)


def download_file(url: str, dest: Path, skip_large_mb: int = 500) -> bool:
    """
    Download a single file from `url` to `dest`.
    Skips files larger than skip_large_mb (to avoid accidental multi-GB downloads).
    Returns True on success, False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  [SKIP]  Already downloaded: {dest.name}")
        return True

    print(f"  [GET]   {url}")
    try:
        # HEAD request to check size first
        head = requests.head(url, allow_redirects=True, timeout=30)
        if head.status_code == 404:
            print(f"  [404]   Not found — skipping.")
            return False

        content_length = int(head.headers.get("Content-Length", 0))
        size_mb = content_length / (1024 ** 2)
        if size_mb > skip_large_mb:
            print(
                f"  [LARGE] {size_mb:.0f} MB — skipping raw archive "
                f"(requires vendor software to process)."
            )
            return False

        # Download with streaming
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)

        print(f"  [OK]    Saved → {dest.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")
        return True

    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return False


def decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file and return the path to the decompressed file."""
    out_path = gz_path.with_suffix("")  # remove .gz
    if out_path.exists():
        print(f"  [SKIP]  Already decompressed: {out_path.name}")
        return out_path
    print(f"  [UNZIP] {gz_path.name} → {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def download_dataset(acc: str, info: dict) -> None:
    folder  = info["folder"]
    out_dir = DATA_RAW / acc

    print(f"\n{'='*60}")
    print(f"  Dataset  : {acc}")
    print(f"  Condition: {info['condition']}")
    print(f"  Platform : {info['platform']}")
    print(f"{'='*60}")

    # ── 1. Series matrix + supplementary files ───────────────────────────────
    for rel_path in info["files"]:
        url  = f"{GEO_FTP}/{folder}/{acc}/{rel_path}"
        dest = out_dir / Path(rel_path).name
        ok   = download_file(url, dest)
        # Decompress matrix files immediately
        if ok and dest.suffix == ".gz":
            decompress_gz(dest)

    # ── 2. SOFT family file (rich metadata) ──────────────────────────────────
    soft_url  = SOFT_URL_TEMPLATE.format(folder=folder, acc=acc)
    soft_dest = out_dir / f"{acc}_family.soft.gz"
    ok        = download_file(soft_url, soft_dest)
    if ok and soft_dest.exists():
        decompress_gz(soft_dest)

    print(f"\n  Files in {out_dir.relative_to(PROJECT_ROOT)}:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:60s}  {size_kb:8.1f} KB")


def main():
    print("\n" + "=" * 60)
    print("  Cytogenomics Project — GEO Dataset Downloader")
    print("  Datasets: GSE192614 (Preeclampsia) | GSE228150 (Pregnancy Loss)")
    print("=" * 60)

    for acc, info in DATASETS.items():
        download_dataset(acc, info)

    print("\n\n✅  Download phase complete.")
    print("   Next step → run:  python scripts/parse_metadata.py")
    print()
    print("FILE GUIDE FOR STUDENTS:")
    print("  *_series_matrix.txt  → Sample metadata + any processed values")
    print("                         USE THIS for metadata parsing.")
    print("  *_family.soft        → Full structured GEO metadata (SOFT format)")
    print("                         USE THIS for detailed sample characteristics.")
    print("  *_RAW.tar            → Raw CEL/IDAT files — very large,")
    print("                         need ChAS/GenomeStudio — NOT downloaded here.")


if __name__ == "__main__":
    main()