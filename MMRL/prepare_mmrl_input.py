#!/usr/bin/env python3
"""Prepare mutation features required by MMRL.

Supported workflows
-------------------
1) Single VCF -> vcf2maf/VEP -> direct AlphaMissense lookup -> MMRL TSV
2) VCF directory -> per-sample vcf2maf/VEP -> one shared AlphaMissense scan -> combined MMRL TSV
3) Existing MAF/MAF-like TSV -> optional direct AlphaMissense lookup -> MMRL TSV

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


def validate_alphamissense_file(path: Path) -> None:
    """Validate the AlphaMissense lookup table used by the direct Python annotator."""
    if not path.exists():
        raise FileNotFoundError(f"AlphaMissense file not found: {path}")


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
    lower = str(path).lower()
    if lower.endswith((".gz", ".bgz", ".bgzf")):
        return gzip.open(path, "rt")
    return open(path, "rt")


def normalize_alphamissense_chromosome(value) -> str:
    """Normalize chromosome names to the chr-prefixed convention used by AlphaMissense."""
    chrom = str(value).strip()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if chrom in {"M", "MT"}:
        chrom = "M"
    return "chr" + chrom


def generate_alphamissense_lookup_keys(df: pd.DataFrame) -> Tuple[set[str], pd.Series]:
    """Build CHROM:POS:REF:ALT lookup keys from MAF columns."""
    required = [
        "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot annotate AlphaMissense; missing MAF columns: {missing}"
        )

    chrom = df["Chromosome"].map(normalize_alphamissense_chromosome)
    pos = pd.to_numeric(df["Start_Position"], errors="coerce").astype("Int64")
    ref = df["Reference_Allele"].astype(str).str.upper()
    alt = df["Tumor_Seq_Allele2"].astype(str).str.upper()

    valid = pos.notna()
    keys = pd.Series(pd.NA, index=df.index, dtype="object")
    keys.loc[valid] = (
        chrom.loc[valid]
        + ":"
        + pos.loc[valid].astype(str)
        + ":"
        + ref.loc[valid]
        + ":"
        + alt.loc[valid]
    )
    return set(keys.dropna()), keys


def _find_alphamissense_header(handle) -> list[str]:
    """Find either '#CHROM ...' or 'CHROM ...' AlphaMissense header."""
    required = {
        "CHROM", "POS", "REF", "ALT", "am_pathogenicity", "am_class"
    }
    for line in handle:
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue

        candidate = stripped[1:] if stripped.startswith("#CHROM\t") else stripped
        fields = candidate.split("\t")
        if required.issubset(fields):
            return fields

    raise ValueError(
        "Could not find an AlphaMissense header containing "
        "CHROM, POS, REF, ALT, am_pathogenicity and am_class."
    )


def scan_alphamissense_database(
    keys_to_find: set[str],
    db_path: Path,
    chunk_size: int = 1_000_000,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Scan AlphaMissense once and return score/class maps for requested lookup keys."""
    validate_alphamissense_file(db_path)
    found_scores: Dict[str, object] = {}
    found_classes: Dict[str, object] = {}
    if not keys_to_find:
        return found_scores, found_classes

    usecols = ["CHROM", "POS", "REF", "ALT", "am_pathogenicity", "am_class"]
    with open_text_auto(db_path) as handle:
        header = _find_alphamissense_header(handle)
        reader = pd.read_csv(
            handle,
            sep="\t",
            names=header,
            header=None,
            comment="#",
            chunksize=chunk_size,
            usecols=usecols,
            dtype={
                "CHROM": str,
                "POS": "Int64",
                "REF": str,
                "ALT": str,
                "am_pathogenicity": str,
                "am_class": str,
            },
            low_memory=False,
        )

        pbar = tqdm(reader, desc="AlphaMissense", unit="chunk")
        for chunk in pbar:
            chunk = chunk.dropna(subset=["CHROM", "POS", "REF", "ALT"]).copy()
            if chunk.empty:
                continue

            chrom = chunk["CHROM"].map(normalize_alphamissense_chromosome)
            chunk["lookup_key"] = (
                chrom
                + ":"
                + chunk["POS"].astype(str)
                + ":"
                + chunk["REF"].astype(str).str.upper()
                + ":"
                + chunk["ALT"].astype(str).str.upper()
            )

            matches = chunk[chunk["lookup_key"].isin(keys_to_find)]
            if not matches.empty:
                for row in matches.itertuples(index=False):
                    key = row.lookup_key
                    found_scores[key] = row.am_pathogenicity
                    found_classes[key] = row.am_class

            pbar.set_postfix(found=len(found_scores))
            if len(found_scores) == len(keys_to_find):
                break
        pbar.close()

    log(
        "AlphaMissense direct lookup: "
        f"{len(found_scores):,}/{len(keys_to_find):,} unique variants matched"
    )
    return found_scores, found_classes


def apply_alphamissense_map(
    df: pd.DataFrame,
    found_scores: Dict[str, object],
    found_classes: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply a previously scanned AlphaMissense lookup map to one mutation table."""
    keys_to_find, lookup_series = generate_alphamissense_lookup_keys(df)
    out = df.copy()
    out["am_pathogenicity"] = lookup_series.map(found_scores)
    out["am_class"] = lookup_series.map(found_classes)
    matched_keys = set(lookup_series[out["am_pathogenicity"].notna()].dropna())
    stats = {
        "alphamissense_lookup_keys": int(len(keys_to_find)),
        "alphamissense_matched_keys": int(len(matched_keys)),
    }
    return out, stats


def annotate_alphamissense_direct(
    df: pd.DataFrame,
    db_path: Path,
    chunk_size: int = 1_000_000,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Annotate AlphaMissense by directly matching CHROM:POS:REF:ALT.

    This intentionally does not use the VEP AlphaMissense plugin. It supports
    plain TSV, gzip, bgzip (.bgz/.bgzf), and both '#CHROM' and 'CHROM' headers.
    """
    keys_to_find, _ = generate_alphamissense_lookup_keys(df)
    found_scores, found_classes = scan_alphamissense_database(
        keys_to_find,
        db_path=db_path,
        chunk_size=chunk_size,
    )
    return apply_alphamissense_map(df, found_scores, found_classes)

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

    # Handle values read by pandas as floats:
    #   379.0
    #   379.0/393.0
    # as well as standard VEP forms:
    #   379
    #   379-380
    #   379/393
    #   379-380/393
    match = re.match(
        r"^(\d+(?:\.0+)?)(?:-\d+(?:\.0+)?)?(?:/(\d+(?:\.0+)?))?$",
        text,
    )

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
        "Pass --alphamissense-file for direct CHROM:POS:REF:ALT annotation, "
        "or use --alphamissense-column when an existing MAF already contains the score."
    )


def prepare_features_before_alphamissense(
    args: argparse.Namespace,
    maf_path: Path,
) -> Tuple[pd.DataFrame, dict]:
    """Prepare missense/SBS96/protein features before AlphaMissense annotation."""
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
    return df, qc


def finalize_features(
    df: pd.DataFrame,
    qc: dict,
    allow_empty: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Finalize QC and complete-case filtering after AlphaMissense is available."""
    df = df.copy()
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

    if final.empty and not allow_empty:
        raise ValueError(
            "No complete MMRL mutation rows remain after preprocessing. "
            "Check the QC report and AlphaMissense/SBS96/protein-position annotations."
        )
    return final, {"qc": qc, "annotated": df}


def prepare_features(args: argparse.Namespace, maf_path: Path) -> Tuple[pd.DataFrame, dict]:
    df, qc = prepare_features_before_alphamissense(args, maf_path)

    if args.alphamissense_file:
        am_path = Path(args.alphamissense_file).expanduser().resolve()
        df, am_stats = annotate_alphamissense_direct(
            df,
            db_path=am_path,
            chunk_size=args.alphamissense_chunk_size,
        )
        qc.update(am_stats)
    else:
        df = normalize_alphamissense_column(df, args.alphamissense_column)

    return finalize_features(df, qc)


def vcf_basename(path: Path) -> str:
    """Return a stable filename stem for .vcf/.vcf.gz/.vcf.bgz/.vcf.bgzf."""
    name = path.name
    for suffix in (".vcf.bgzf", ".vcf.bgz", ".vcf.gz", ".vcf"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def is_vcf_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(x) for x in (".vcf", ".vcf.gz", ".vcf.bgz", ".vcf.bgzf"))


def discover_vcfs(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"VCF directory not found: {directory}")
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    vcfs = sorted(p.resolve() for p in iterator if p.is_file() and is_vcf_path(p))
    if not vcfs:
        raise FileNotFoundError(f"No .vcf/.vcf.gz files found under: {directory}")
    return vcfs


def read_vcf_sample_names(path: Path) -> list[str]:
    """Read genotype sample names directly from the #CHROM VCF header."""
    with open_text_auto(path) as handle:
        for line in handle:
            if line.startswith("#CHROM\t"):
                fields = line.rstrip("\r\n").split("\t")
                return fields[9:] if len(fields) > 9 else []
    raise ValueError(f"VCF #CHROM header not found: {path}")


def stage_plain_vcf(input_vcf: Path, sample_work_dir: Path) -> Path:
    """Stage a plain .vcf in the batch work directory; decompress when necessary."""
    sample_work_dir.mkdir(parents=True, exist_ok=True)
    staged = sample_work_dir / f"{vcf_basename(input_vcf)}.vcf"
    if staged.exists() or staged.is_symlink():
        staged.unlink()

    lower = input_vcf.name.lower()
    if lower.endswith((".vcf.gz", ".vcf.bgz", ".vcf.bgzf")):
        log(f"Decompressing for vcf2maf: {input_vcf.name}")
        with gzip.open(input_vcf, "rb") as src, open(staged, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        try:
            staged.symlink_to(input_vcf.resolve())
        except OSError:
            shutil.copy2(input_vcf, staged)

    stale_vep = staged.with_name(staged.name[:-4] + ".vep.vcf")
    if stale_vep.exists():
        stale_vep.unlink()
    return staged


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_vcf_directory(args: argparse.Namespace, output: Path) -> None:
    """Batch-process one tumor VCF per file and scan AlphaMissense only once."""
    input_dir = Path(args.input_vcf_dir).expanduser().resolve()
    vcfs = discover_vcfs(input_dir, recursive=args.recursive)
    work_dir = (
        Path(args.batch_work_dir).expanduser().resolve()
        if args.batch_work_dir
        else Path(str(output) + ".batch")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"Batch mode: discovered {len(vcfs):,} VCF file(s) under {input_dir}")
    log(f"Batch work directory: {work_dir}")

    prepared: list[dict] = []
    summary: list[dict] = []
    seen_barcodes: set[str] = set()

    for idx, input_vcf in enumerate(vcfs, start=1):
        file_key = vcf_basename(input_vcf)
        log(f"[{idx}/{len(vcfs)}] Processing {input_vcf.name}")
        record = {
            "input_vcf": str(input_vcf),
            "file_key": file_key,
            "status": "failed",
            "error": "",
        }
        try:
            samples = read_vcf_sample_names(input_vcf)
            if not samples:
                raise ValueError("VCF has no genotype sample column")

            vcf_normal_id = args.vcf_normal_id
            if vcf_normal_id and vcf_normal_id not in samples:
                raise ValueError(
                    f"--vcf-normal-id {vcf_normal_id!r} not found; VCF samples={samples}"
                )

            if args.vcf_tumor_id:
                if args.vcf_tumor_id not in samples:
                    raise ValueError(
                        f"--vcf-tumor-id {args.vcf_tumor_id!r} not found; VCF samples={samples}"
                    )
                vcf_tumor_id = args.vcf_tumor_id
            elif len(samples) == 1:
                vcf_tumor_id = samples[0]
            elif vcf_normal_id and len(samples) == 2:
                # Common tumor-normal layout: when the same normal column name is used
                # across files, infer the other genotype column as the tumor.
                vcf_tumor_id = next(s for s in samples if s != vcf_normal_id)
            else:
                raise ValueError(
                    "VCF contains multiple genotype samples. In directory mode, either provide "
                    "a common --vcf-tumor-id, provide --vcf-normal-id for a two-sample tumor-normal "
                    "VCF, or split to one tumor per VCF. "
                    f"Samples={samples}"
                )

            tumor_barcode = file_key if args.batch_sample_id_source == "filename" else vcf_tumor_id
            if tumor_barcode in seen_barcodes:
                raise ValueError(
                    f"Duplicate Tumor_Sample_Barcode {tumor_barcode!r}. "
                    "Use --batch-sample-id-source filename if VCF sample names are reused."
                )
            seen_barcodes.add(tumor_barcode)

            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_key) or f"sample_{idx}"
            safe_dir_name = f"{idx:04d}_{safe_label}"
            sample_dir = work_dir / safe_dir_name
            staged_vcf = stage_plain_vcf(input_vcf, sample_dir)
            maf_path = sample_dir / f"{safe_dir_name}.vcf2maf.maf"
            if maf_path.exists():
                maf_path.unlink()

            sample_args = argparse.Namespace(**vars(args))
            sample_args.input_vcf = str(staged_vcf)
            sample_args.tumor_id = tumor_barcode
            sample_args.vcf_tumor_id = vcf_tumor_id
            sample_args.normal_id = vcf_normal_id
            sample_args.vcf_normal_id = vcf_normal_id
            if args.tmp_dir is None:
                sample_args.tmp_dir = str(sample_dir / "tmp")
                Path(sample_args.tmp_dir).mkdir(parents=True, exist_ok=True)

            run_vcf2maf(sample_args, maf_path)
            df, qc = prepare_features_before_alphamissense(sample_args, maf_path)

            keys, _ = generate_alphamissense_lookup_keys(df)
            prepared.append({
                "input_vcf": input_vcf,
                "file_key": file_key,
                "sample_dir": sample_dir,
                "tumor_barcode": tumor_barcode,
                "vcf_tumor_id": vcf_tumor_id,
                "maf_path": maf_path,
                "df": df,
                "qc": qc,
                "lookup_keys": keys,
            })
            record.update({
                "status": "prepared",
                "tumor_sample_barcode": tumor_barcode,
                "vcf_tumor_id": vcf_tumor_id,
                "input_rows": qc.get("input_rows", 0),
                "missense_rows": qc.get("missense_rows", 0),
            })
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            log(f"FAILED {input_vcf.name}: {record['error']}")
            if args.fail_fast:
                raise
        summary.append(record)

    if not prepared:
        summary_path = (
            Path(args.batch_summary).expanduser().resolve()
            if args.batch_summary
            else Path(str(output) + ".batch_summary.tsv")
        )
        pd.DataFrame(summary).to_csv(summary_path, sep="\t", index=False)
        raise RuntimeError("No VCFs completed the vcf2maf/pre-AlphaMissense stage")

    all_keys: set[str] = set()
    for item in prepared:
        all_keys.update(item["lookup_keys"])
    log(
        f"Scanning AlphaMissense once for {len(all_keys):,} unique variants "
        f"across {len(prepared):,} prepared sample(s)"
    )
    am_path = Path(args.alphamissense_file).expanduser().resolve()
    found_scores, found_classes = scan_alphamissense_database(
        all_keys,
        db_path=am_path,
        chunk_size=args.alphamissense_chunk_size,
    )

    finals: list[pd.DataFrame] = []
    summary_by_input = {row["input_vcf"]: row for row in summary}
    successful_samples = 0

    for item in prepared:
        record = summary_by_input[str(item["input_vcf"])]
        try:
            annotated, am_stats = apply_alphamissense_map(
                item["df"], found_scores, found_classes
            )
            qc = item["qc"]
            qc.update(am_stats)
            final, payload = finalize_features(annotated, qc, allow_empty=True)

            annotated_path = item["sample_dir"] / f"{item['file_key']}.annotated.tsv"
            final_path = item["sample_dir"] / f"{item['file_key']}.mmrl.tsv"
            qc_path = item["sample_dir"] / f"{item['file_key']}.qc.json"
            payload["annotated"].to_csv(annotated_path, sep="\t", index=False)
            final.to_csv(final_path, sep="\t", index=False)
            write_json(qc_path, payload["qc"])

            record.update({
                "status": "ok" if not final.empty else "empty",
                "error": "" if not final.empty else "No complete MMRL rows after filtering",
                "alphamissense_matched_keys": qc.get("alphamissense_matched_keys", 0),
                "alphamissense_non_missing": qc.get("alphamissense_non_missing", 0),
                "final_rows": qc.get("final_rows", 0),
                "maf_output": str(item["maf_path"]),
                "annotated_output": str(annotated_path),
                "sample_output": str(final_path),
                "qc_output": str(qc_path),
            })
            if not final.empty:
                finals.append(final)
                successful_samples += 1
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            log(f"FAILED finalization {item['file_key']}: {record['error']}")
            if args.fail_fast:
                raise

    summary_path = (
        Path(args.batch_summary).expanduser().resolve()
        if args.batch_summary
        else Path(str(output) + ".batch_summary.tsv")
    )
    pd.DataFrame(summary).to_csv(summary_path, sep="\t", index=False)
    log(f"Batch summary: {summary_path}")

    if not finals:
        raise RuntimeError("Batch completed, but no sample produced complete MMRL rows")

    combined = pd.concat(finals, ignore_index=True)
    combined.to_csv(output, sep="\t", index=False)
    log(
        f"Combined MMRL input: {output} ({len(combined):,} rows, "
        f"{successful_samples:,} sample(s))"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MMRL mutation-model input from a somatic VCF or an existing MAF. "
            "VCF input is first converted with vcf2maf/VEP."
        )
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-vcf", help="Single somatic VCF/VCF.GZ input")
    src.add_argument(
        "--input-vcf-dir",
        help="Directory of VCF/VCF.GZ files; one tumor VCF per file by default",
    )
    src.add_argument("--input-maf", help="Existing MAF/TSV input")

    parser.add_argument(
        "--output",
        required=True,
        help="Final MMRL TSV; in directory mode this is the combined multi-sample table",
    )
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

    # VCF-directory batch options
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover VCF files under --input-vcf-dir",
    )
    parser.add_argument(
        "--batch-work-dir",
        help="Per-sample MAF/annotated/QC work directory (default: <output>.batch)",
    )
    parser.add_argument(
        "--batch-summary",
        help="Batch summary TSV (default: <output>.batch_summary.tsv)",
    )
    parser.add_argument(
        "--batch-sample-id-source",
        choices=["vcf", "filename"],
        default="vcf",
        help=(
            "Tumor_Sample_Barcode source in directory mode: VCF genotype sample name "
            "or VCF filename stem (default: vcf)"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop directory mode at the first failed VCF instead of continuing",
    )

    # AlphaMissense (direct lookup; not a VEP plugin)
    parser.add_argument(
        "--alphamissense-file",
        help=(
            "AlphaMissense hg19/hg38 TSV/TSV.GZ/BGZ. The script directly annotates "
            "CHROM:POS:REF:ALT after vcf2maf; no VEP AlphaMissense plugin is used."
        ),
    )
    parser.add_argument(
        "--alphamissense-column",
        default="am_pathogenicity",
        help=(
            "Pathogenicity-score column in an existing MAF when "
            "--alphamissense-file is not supplied (default: am_pathogenicity)"
        ),
    )
    parser.add_argument(
        "--alphamissense-chunk-size",
        type=int,
        default=1_000_000,
        help="Rows per AlphaMissense database chunk for direct lookup (default: 1000000)",
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
    vcf_mode = bool(args.input_vcf or args.input_vcf_dir)
    if vcf_mode and not args.ncbi_build:
        parser.error("--ncbi-build is required with --input-vcf/--input-vcf-dir")
    if args.input_vcf and not args.tumor_id:
        parser.error("--tumor-id is required with --input-vcf so sample IDs are preserved correctly")
    if vcf_mode and not args.alphamissense_file:
        parser.error("--alphamissense-file is required with VCF input to generate am_pathogenicity")
    if args.input_vcf_dir and args.tumor_id:
        parser.error(
            "--tumor-id is a single-file option. Directory mode automatically uses the VCF "
            "sample name, or use --batch-sample-id-source filename."
        )
    if args.input_vcf_dir and args.normal_id:
        parser.error(
            "--normal-id is a single-file option. In directory mode, use --vcf-normal-id "
            "only when the same normal genotype-column name is present in every VCF."
        )
    if args.input_vcf_dir and args.maf_output:
        parser.error("--maf-output is only valid for single --input-vcf mode")
    if args.input_vcf_dir and args.annotated_output:
        parser.error(
            "--annotated-output is only valid outside directory mode; per-sample annotated "
            "files are written automatically under --batch-work-dir."
        )
    if args.input_vcf_dir and args.qc_output:
        parser.error(
            "--qc-output is only valid outside directory mode; per-sample QC files and a "
            "batch summary are written automatically."
        )
    if args.vep_forks < 1:
        parser.error("--vep-forks must be >= 1")
    if args.alphamissense_chunk_size < 1:
        parser.error("--alphamissense-chunk-size must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.input_vcf_dir:
        run_vcf_directory(args, output)
        return

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
    write_json(qc_output, payload["qc"])
    log(f"QC report: {qc_output}")


if __name__ == "__main__":
    main()
