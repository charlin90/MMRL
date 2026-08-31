#!/usr/bin/env python3
"""Prepare mutation features required by MMRL.

Supported workflows
-------------------
1) VCF -> vcf2maf/VEP (+ optional AlphaMissense plugin) -> MMRL TSV
2) Existing MAF/MAF-like TSV -> MMRL TSV

The final output contains:
    Hugo_Symbol, SBS96, am_pathogenicity, Tumor_Sample_Barcode,
    AAchange, Relative_Position

This script intentionally works by MAF column names rather than fixed column numbers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
try:
    import pysam
except ImportError:  # allows --help to work before optional preprocessing deps are installed
    pysam = None
from tqdm import tqdm


MODEL_COLUMNS = [
    "Hugo_Symbol",
    "SBS96",
    "am_pathogenicity",
    "Tumor_Sample_Barcode",
    "AAchange",
    "Relative_Position",
]

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}


def log(msg: str) -> None:
    print(f"[MMRL-preprocess] {msg}", file=sys.stderr)


def run_command(cmd: list[str]) -> None:
    log("Running: " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def resolve_executable_or_file(value: str) -> str:
    p = Path(value).expanduser()
    if p.exists():
        return str(p.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise FileNotFoundError(f"Cannot find executable/file: {value}")


def validate_alphamissense_files(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"AlphaMissense file not found: {path}")
    # VEP AlphaMissense plugin uses tabix-indexed files.
    if not Path(str(path) + ".tbi").exists():
        log(
            "Warning: AlphaMissense tabix index (.tbi) was not found. "
            "The VEP AlphaMissense plugin normally requires a tabix-indexed file."
        )


def run_vcf2maf(args: argparse.Namespace, output_maf: Path) -> Path:
    vcf2maf = resolve_executable_or_file(args.vcf2maf)
    input_vcf = Path(args.input_vcf).expanduser().resolve()
    if not input_vcf.exists():
        raise FileNotFoundError(f"Input VCF not found: {input_vcf}")

    output_maf.parent.mkdir(parents=True, exist_ok=True)

    # Explicitly invoke Perl so vcf2maf.pl does not need executable permissions.
    cmd = [
        "perl", vcf2maf,
        "--input-vcf", str(input_vcf),
        "--output-maf", str(output_maf),
        "--ref-fasta", str(Path(args.ref_fasta).expanduser().resolve()),
        "--ncbi-build", args.ncbi_build,
        "--vep-forks", str(args.vep_forks),
        "--retain-ann", "am_pathogenicity,am_class",
    ]

    if args.tumor_id:
        cmd += ["--tumor-id", args.tumor_id]
    if args.normal_id:
        cmd += ["--normal-id", args.normal_id]
    if args.vcf_tumor_id:
        cmd += ["--vcf-tumor-id", args.vcf_tumor_id]
    if args.vcf_normal_id:
        cmd += ["--vcf-normal-id", args.vcf_normal_id]
    if args.vep_path:
        cmd += ["--vep-path", str(Path(args.vep_path).expanduser().resolve())]
    if args.vep_data:
        cmd += ["--vep-data", str(Path(args.vep_data).expanduser().resolve())]
    if args.cache_version:
        cmd += ["--cache-version", str(args.cache_version)]
    if args.tmp_dir:
        cmd += ["--tmp-dir", str(Path(args.tmp_dir).expanduser().resolve())]
    if args.verbose_vcf2maf:
        cmd += ["--verbose"]

    if args.alphamissense_file:
        am_file = Path(args.alphamissense_file).expanduser().resolve()
        validate_alphamissense_files(am_file)
        cmd += [
            "--vep-plugins",
            f"AlphaMissense,file={am_file},cols=all",
        ]
    else:
        log(
            "Warning: --alphamissense-file was not supplied. The final table will only "
            "be complete if am_pathogenicity is already present in the resulting MAF."
        )

    run_command(cmd)
    return output_maf


def read_maf(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mutation file not found: {path}")
    # vcf2maf writes '#version 2.4' before the header; comment='#' handles this.
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if df.empty:
        raise ValueError(f"No mutation rows found in: {path}")
    return df


def filter_missense_somatic(
    df: pd.DataFrame,
    no_somatic_filter: bool = False,
    somatic_column: Optional[str] = None,
    somatic_values: Iterable[str] = ("somatic",),
) -> Tuple[pd.DataFrame, Dict[str, int | str | bool]]:
    if "Variant_Classification" not in df.columns:
        raise ValueError("Input MAF must contain column 'Variant_Classification'.")

    stats: Dict[str, int | str | bool] = {"input_rows": int(len(df))}
    out = df[df["Variant_Classification"].astype(str).eq("Missense_Mutation")].copy()
    stats["missense_rows"] = int(len(out))

    if no_somatic_filter:
        stats["somatic_filter_applied"] = False
        return out, stats

    candidate = somatic_column
    if candidate is None and "Mutation_Status" in out.columns:
        candidate = "Mutation_Status"

    if candidate is None:
        stats["somatic_filter_applied"] = False
        stats["somatic_filter_reason"] = "no somatic-status column selected/found"
        return out, stats

    if candidate not in out.columns:
        raise ValueError(f"Somatic filter column not found: {candidate}")

    accepted = {str(v).strip().lower() for v in somatic_values}
    normalized = out[candidate].astype(str).str.strip().str.lower()
    recognized = normalized.isin(accepted)

    # Auto-detected Mutation_Status can be empty/unknown in some MAFs. In that case,
    # avoid accidentally dropping the entire dataset. An explicitly requested column
    # is always enforced.
    if somatic_column is None and recognized.sum() == 0:
        stats["somatic_filter_applied"] = False
        stats["somatic_filter_reason"] = (
            f"auto-detected {candidate}, but none of its values matched {sorted(accepted)}"
        )
        return out, stats

    out = out[recognized].copy()
    stats["somatic_filter_applied"] = True
    stats["somatic_filter_column"] = candidate
    stats["somatic_rows"] = int(len(out))
    return out, stats


def find_fasta_contig(fasta: pysam.FastaFile, chrom: str) -> Optional[str]:
    chrom = str(chrom)
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append("chr" + chrom)

    if chrom in {"M", "MT", "chrM", "chrMT"}:
        candidates.extend(["M", "MT", "chrM", "chrMT"])

    refs = set(fasta.references)
    for candidate in candidates:
        if candidate in refs:
            return candidate
    return None


def get_sbs96(
    row: pd.Series,
    fasta: pysam.FastaFile,
) -> Optional[str]:
    try:
        chrom = find_fasta_contig(fasta, str(row["Chromosome"]))
        if chrom is None:
            return None

        pos = int(row["Start_Position"])
        ref = str(row["Reference_Allele"]).upper()
        alt = str(row["Tumor_Seq_Allele2"]).upper()

        if len(ref) != 1 or len(alt) != 1:
            return None
        if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
            return None
        if "End_Position" in row and pd.notna(row["End_Position"]):
            if int(row["End_Position"]) != pos:
                return None
        if pos < 2:
            return None

        context = fasta.fetch(chrom, pos - 2, pos + 1).upper()
        if len(context) != 3 or context[1] != ref:
            return None
        if any(base not in "ACGT" for base in context):
            return None

        # COSMIC SBS96 convention: represent substitutions on the C/T strand.
        if ref in {"A", "G"}:
            ref = COMPLEMENT[ref]
            alt = COMPLEMENT[alt]
            context = "".join(COMPLEMENT[b] for b in reversed(context))

        return f"{context[0]}[{ref}>{alt}]{context[2]}"
    except (ValueError, TypeError, KeyError, IndexError, OSError):
        return None


def add_sbs96(df: pd.DataFrame, ref_fasta: Path) -> pd.DataFrame:
    if pysam is None:
        raise ImportError("pysam is required for SBS96 extraction. Install it with: pip install pysam")
    required = [
        "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot calculate SBS96; missing MAF columns: {missing}")

    if not ref_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {ref_fasta}")

    if not Path(str(ref_fasta) + ".fai").exists() and ref_fasta.suffix != ".gz":
        log(f"FASTA index not found; creating {ref_fasta}.fai")
        pysam.faidx(str(ref_fasta))

    fasta = pysam.FastaFile(str(ref_fasta))
    try:
        tqdm.pandas(desc="SBS96", unit="mut")
        out = df.copy()
        out["SBS96"] = out.progress_apply(get_sbs96, axis=1, fasta=fasta)
    finally:
        fasta.close()
    return out


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def load_transcript_lengths(protein_fasta: Path) -> Dict[str, int]:
    """Load transcript -> protein length from an Ensembl peptide FASTA.

    Headers are expected to contain a token like 'transcript:ENST...'. Version
    suffixes are stripped so ENST... and ENST....1 can be matched.
    """
    if not protein_fasta.exists():
        raise FileNotFoundError(f"Protein FASTA not found: {protein_fasta}")

    lengths: Dict[str, int] = {}
    current_tid: Optional[str] = None
    current_len = 0

    def flush() -> None:
        nonlocal current_tid, current_len
        if current_tid:
            lengths[current_tid] = current_len

    with open_text_auto(protein_fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                flush()
                current_tid = None
                current_len = 0
                for token in line.strip().split():
                    if token.startswith("transcript:"):
                        current_tid = token.split(":", 1)[1].split(".")[0]
                        break
            else:
                current_len += len(line.strip())
        flush()

    log(f"Loaded protein lengths for {len(lengths):,} transcripts")
    return lengths


def parse_protein_position(value) -> Tuple[Optional[float], Optional[float]]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    # Examples handled: 379/393, 379-380/393, 379, 379-380
    match = re.match(r"^(\d+)(?:-\d+)?(?:/(\d+))?$", text)
    if not match:
        return None, None
    pos = float(match.group(1))
    total = float(match.group(2)) if match.group(2) else None
    return pos, total


def normalize_hgvsp_to_one_letter(text: str) -> str:
    out = text
    for aa3, aa1 in AA3_TO_1.items():
        out = out.replace(aa3, aa1)
    return out


def extract_aa_change(hgvsp, amino_acids=None) -> Optional[str]:
    if pd.notna(hgvsp):
        text = normalize_hgvsp_to_one_letter(str(hgvsp))
        # Standard vcf2maf HGVSp_Short: p.R379C
        match = re.search(r"p\.([A-Z*])\d+([A-Z*])(?:$|[^A-Za-z])", text)
        if match:
            return f"{match.group(1)}>{match.group(2)}"

    # VEP Amino_acids is commonly R/C for a missense SNV.
    if amino_acids is not None and pd.notna(amino_acids):
        text = str(amino_acids).strip()
        match = re.match(r"^([A-Z*])/([A-Z*])$", text)
        if match:
            return f"{match.group(1)}>{match.group(2)}"
    return None


def add_protein_features(
    df: pd.DataFrame,
    protein_fasta: Optional[Path] = None,
) -> pd.DataFrame:
    if "Protein_position" not in df.columns:
        raise ValueError("Input MAF must contain 'Protein_position'.")

    transcript_lengths: Dict[str, int] = {}
    if protein_fasta:
        transcript_lengths = load_transcript_lengths(protein_fasta)

    out = df.copy()

    aa_series = out["Amino_acids"] if "Amino_acids" in out.columns else pd.Series(
        [None] * len(out), index=out.index
    )
    hgvsp_col = "HGVSp_Short" if "HGVSp_Short" in out.columns else "HGVSp"
    if hgvsp_col not in out.columns:
        raise ValueError("Input MAF must contain 'HGVSp_Short' or 'HGVSp'.")

    out["AAchange"] = [
        extract_aa_change(hgvsp, aa)
        for hgvsp, aa in zip(out[hgvsp_col], aa_series)
    ]

    rel_positions = []
    for _, row in out.iterrows():
        pos, total = parse_protein_position(row["Protein_position"])

        if total is None and pos is not None and transcript_lengths:
            tid = row.get("Transcript_ID")
            if pd.notna(tid):
                clean_tid = str(tid).split(".")[0]
                total = transcript_lengths.get(clean_tid)

        if pos is None or total is None or total <= 0 or pos <= 0:
            rel_positions.append(math.nan)
        else:
            rel_positions.append(pos / total)

    out["Relative_Position"] = rel_positions
    return out


def normalize_alphamissense_column(
    df: pd.DataFrame,
    requested_column: str,
) -> pd.DataFrame:
    out = df.copy()
    if requested_column in out.columns:
        if requested_column != "am_pathogenicity":
            out["am_pathogenicity"] = out[requested_column]
        return out

    aliases = [
        "am_pathogenicity",
        "Missense_pathogenicity",
        "AM_pathogenicity",
        "AlphaMissense_pathogenicity",
    ]
    for col in aliases:
        if col in out.columns:
            out["am_pathogenicity"] = out[col]
            log(f"Using AlphaMissense score column: {col}")
            return out

    raise ValueError(
        "AlphaMissense pathogenicity score column was not found. "
        "For VCF input, pass --alphamissense-file so vcf2maf/VEP can annotate it; "
        "for MAF input, use --alphamissense-column if your column has a custom name."
    )


def prepare_features(args: argparse.Namespace, maf_path: Path) -> Tuple[pd.DataFrame, dict]:
    df = read_maf(maf_path)
    df, qc = filter_missense_somatic(
        df,
        no_somatic_filter=args.no_somatic_filter,
        somatic_column=args.somatic_column,
        somatic_values=args.somatic_values,
    )

    ref_fasta = Path(args.ref_fasta).expanduser().resolve()
    df = add_sbs96(df, ref_fasta)
    qc["sbs96_non_missing"] = int(df["SBS96"].notna().sum())

    protein_fasta = (
        Path(args.protein_fasta).expanduser().resolve() if args.protein_fasta else None
    )
    df = add_protein_features(df, protein_fasta=protein_fasta)
    qc["aachange_non_missing"] = int(df["AAchange"].notna().sum())
    qc["relative_position_non_missing"] = int(df["Relative_Position"].notna().sum())

    df = normalize_alphamissense_column(df, args.alphamissense_column)
    df["am_pathogenicity"] = pd.to_numeric(df["am_pathogenicity"], errors="coerce")
    qc["alphamissense_non_missing"] = int(df["am_pathogenicity"].notna().sum())

    for col in ["Hugo_Symbol", "Tumor_Sample_Barcode"]:
        if col not in df.columns:
            raise ValueError(f"Input MAF must contain '{col}'.")

    qc["rows_before_complete_case_filter"] = int(len(df))
    qc["missing_by_model_column"] = {
        col: int(df[col].isna().sum()) for col in MODEL_COLUMNS
    }

    final = df.dropna(subset=MODEL_COLUMNS).copy()
    final = final[MODEL_COLUMNS]
    qc["final_rows"] = int(len(final))
    qc["final_samples"] = int(final["Tumor_Sample_Barcode"].nunique())

    if final.empty:
        raise ValueError(
            "No complete MMRL mutation rows remain after preprocessing. "
            "Check the QC report and AlphaMissense/SBS96/protein-position annotations."
        )

    return final, {"qc": qc, "annotated": df}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MMRL mutation-model input from a somatic VCF or an existing MAF. "
            "VCF input is first converted with vcf2maf/VEP."
        )
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-vcf", help="Somatic VCF/VCF.GZ input")
    src.add_argument("--input-maf", help="Existing MAF/TSV input")

    parser.add_argument("--output", required=True, help="Final MMRL input TSV")
    parser.add_argument(
        "--ref-fasta",
        required=True,
        help="Reference genome FASTA matching the variants (used by SBS96 and vcf2maf)",
    )
    parser.add_argument(
        "--annotated-output",
        help="Optional TSV containing all retained missense rows plus derived features",
    )
    parser.add_argument(
        "--qc-output",
        help="QC JSON path (default: <output>.qc.json)",
    )

    # vcf2maf / VEP options
    parser.add_argument("--vcf2maf", default="vcf2maf.pl", help="Path to vcf2maf.pl")
    parser.add_argument(
        "--ncbi-build",
        choices=["GRCh37", "GRCh38"],
        help="Genome build; required for --input-vcf",
    )
    parser.add_argument("--tumor-id", help="Tumor_Sample_Barcode written to MAF")
    parser.add_argument("--normal-id", help="Matched normal sample ID")
    parser.add_argument("--vcf-tumor-id", help="Tumor genotype-column sample ID in the VCF")
    parser.add_argument("--vcf-normal-id", help="Normal genotype-column sample ID in the VCF")
    parser.add_argument("--vep-path", help="Directory containing vep/variant_effect_predictor.pl")
    parser.add_argument("--vep-data", help="VEP cache/plugin directory")
    parser.add_argument("--cache-version", help="VEP cache version")
    parser.add_argument("--vep-forks", type=int, default=4)
    parser.add_argument("--tmp-dir", help="Temporary directory passed to vcf2maf")
    parser.add_argument(
        "--maf-output",
        help="Intermediate vcf2maf MAF path (default: alongside --output)",
    )
    parser.add_argument("--verbose-vcf2maf", action="store_true")

    # AlphaMissense
    parser.add_argument(
        "--alphamissense-file",
        help=(
            "Tabix-indexed AlphaMissense hg19/hg38 TSV.GZ for the VEP plugin. "
            "Recommended for --input-vcf."
        ),
    )
    parser.add_argument(
        "--alphamissense-column",
        default="am_pathogenicity",
        help="Pathogenicity-score column in an existing MAF (default: am_pathogenicity)",
    )

    # Filtering / protein position fallback
    parser.add_argument(
        "--no-somatic-filter",
        action="store_true",
        help="Do not filter on somatic status (missense filtering is still applied)",
    )
    parser.add_argument(
        "--somatic-column",
        help=(
            "Column used to identify somatic mutations. If omitted, Mutation_Status is "
            "auto-detected when usable."
        ),
    )
    parser.add_argument(
        "--somatic-values",
        nargs="+",
        default=["somatic"],
        help="Accepted somatic values, case-insensitive (default: somatic)",
    )
    parser.add_argument(
        "--protein-fasta",
        help=(
            "Optional Ensembl peptide FASTA(.gz) used only when Protein_position lacks "
            "the '/protein_length' denominator."
        ),
    )

    args = parser.parse_args()
    if args.input_vcf and not args.ncbi_build:
        parser.error("--ncbi-build is required with --input-vcf")
    if args.input_vcf and not args.tumor_id:
        parser.error("--tumor-id is required with --input-vcf so sample IDs are preserved correctly")
    if args.input_vcf and not args.alphamissense_file:
        parser.error("--alphamissense-file is required with --input-vcf to generate am_pathogenicity")
    if args.vep_forks < 1:
        parser.error("--vep-forks must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.input_vcf:
        if args.maf_output:
            maf_path = Path(args.maf_output).expanduser().resolve()
        else:
            maf_path = output.with_suffix(".vcf2maf.maf")
        run_vcf2maf(args, maf_path)
    else:
        maf_path = Path(args.input_maf).expanduser().resolve()

    final, payload = prepare_features(args, maf_path)
    final.to_csv(output, sep="\t", index=False)
    log(f"Final MMRL input: {output} ({len(final):,} rows)")

    if args.annotated_output:
        annotated_output = Path(args.annotated_output).expanduser().resolve()
        annotated_output.parent.mkdir(parents=True, exist_ok=True)
        payload["annotated"].to_csv(annotated_output, sep="\t", index=False)
        log(f"Annotated mutations: {annotated_output}")

    qc_output = (
        Path(args.qc_output).expanduser().resolve()
        if args.qc_output
        else Path(str(output) + ".qc.json")
    )
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    with open(qc_output, "w") as handle:
        json.dump(payload["qc"], handle, indent=2, ensure_ascii=False)
    log(f"QC report: {qc_output}")


if __name__ == "__main__":
    main()
