# Missense Mutation Embedding Extraction

<img src="C:\Users\41378\Desktop\个性化WES模型\202605结果整理\主要分析代码\github\mmrl.png" alt="mmrl" style="zoom:50%;" />

Prepare somatic missense-mutation features for **MMRL** and extract one embedding per tumor sample.

The preprocessing script is:

```text
MMRL/prepare_mmrl_input.py
```

Supported workflows:

```text
Single VCF         -> vcf2maf/VEP -> AlphaMissense lookup -> MMRL TSV
VCF directory      -> per-sample vcf2maf/VEP -> shared AlphaMissense scan -> combined MMRL TSV
MAF / MAF-like TSV -> preprocessing -> optional AlphaMissense lookup -> MMRL TSV
```

> MAF and tab-separated mutation tables both use `--input-maf`.

The final model input contains exactly:

```text
Hugo_Symbol
SBS96
am_pathogenicity
Tumor_Sample_Barcode
AAchange
Relative_Position
```

---

## 1. Installation

```bash
git clone https://github.com/charlin90/MMRL.git
cd MMRL

pip install numpy pandas torch scikit-learn joblib tqdm pysam
```

Check the preprocessing CLI:

```bash
python MMRL/prepare_mmrl_input.py --help
```

For embedding extraction, the checkpoint directory must contain:

```text
checkpoints/
├── best_model.pth
├── gene_encoder.pkl
└── mut_type_encoder.pkl
```

### VCF-only dependencies

VCF input is converted with `vcf2maf`, which runs Ensembl VEP. Install:

- Perl
- [vcf2maf](https://github.com/mskcc/vcf2maf)
- [Ensembl VEP](https://github.com/Ensembl/ensembl-vep)
- a VEP cache matching the genome build

Typical paths are passed as:

```text
--vcf2maf /path/to/vcf2maf/vcf2maf.pl
--vep-path /path/to/ensembl-vep
--vep-data /path/to/.vep
```

---

## 2. Reference data

### Reference genome FASTA

`--ref-fasta` is required for all workflows because SBS96 is calculated from the reference sequence.

Use the same assembly as the mutation coordinates:

```text
GRCh38 variants -> GRCh38 FASTA
GRCh37 variants -> GRCh37 FASTA
```

Pre-indexing is recommended:

```bash
samtools faidx reference.fa
```

For an uncompressed FASTA, the script can create a missing `.fai` automatically with `pysam`.

### AlphaMissense predictions

Use the genomic AlphaMissense file matching the same assembly:

```text
GRCh38 -> AlphaMissense_hg38.tsv.gz
GRCh37 -> AlphaMissense_hg19.tsv.gz
```

Official resources:

- https://github.com/google-deepmind/alphamissense

The script directly scans the table and matches variants by:

```text
CHROM:POS:REF:ALT
```

The AlphaMissense file must contain:

```text
CHROM
POS
REF
ALT
am_pathogenicity
am_class
```

For all workflows, `--alphamissense-file` is required. 

### Optional peptide FASTA

Normally `Protein_position` contains both the residue position and protein length, for example:

```text
379/393
```

The script calculates:

```text
Relative_Position = 379 / 393
```

If only `379` is available, provide:

```bash
--protein-fasta /path/to/ensembl.pep.all.fa.gz
```

and ensure `Transcript_ID` is present.

> Keep the mutation coordinates, reference FASTA, VEP cache, and AlphaMissense file on the same genome assembly.

For human, the Ensembl peptide FASTA can be downloaded with `wget`:

**GRCh38**

```bash
wget https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz
```

**GRCh37**

```bash
wget https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.pep.all.fa.gz
```

---

## 3. Existing MAF or MAF-like TSV

This is the simplest workflow when variants have already been annotated.

```bash
python MMRL/prepare_mmrl_input.py \
    --input-maf data/sample.maf \
    --output data/sample.mmrl.tsv \
    --ref-fasta /path/to/reference.fa \
    --alphamissense-file /path/to/AlphaMissense_hg38.tsv.gz \
    --protein-fasta /path/to/Homo_sapiens.GRCh38.pep.all.fa.gz \
    --qc-output sample.mmrl.qc.json
```

A tab-separated mutation file is run in the same way:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-maf data/mutations.tsv \
    --output data/sample.mmrl.tsv \
    --ref-fasta /path/to/reference.fa \
    --alphamissense-file /path/to/AlphaMissense_hg38.tsv.gz \
    --protein-fasta /path/to/Homo_sapiens.GRCh38.pep.all.fa.gz \
    --qc-output sample.mmrl.qc.json
```

### Required input columns

```text
Hugo_Symbol
Variant_Classification
Chromosome
Start_Position
Reference_Allele
Tumor_Seq_Allele2
Tumor_Sample_Barcode
Protein_position
HGVSp_Short or HGVSp
```

Optional columns:

```text
Mutation_Status   # somatic filtering
Amino_acids       # AAchange fallback
Transcript_ID     # protein-length fallback with --protein-fasta
End_Position      # checked when present
```

The script always retains only:

```text
Variant_Classification == Missense_Mutation
```

`Variant_Classification` is therefore required even if the input has already been restricted to missense variants.

### Somatic filtering

If `Mutation_Status` is present, the script recognizes `somatic` case-insensitively.

Skip somatic filtering:

```bash
--no-somatic-filter
```

Use a custom status column or accepted values:

```bash
--somatic-column Somatic_Status \
--somatic-values Somatic LikelySomatic
```

When `Mutation_Status` is auto-detected but none of its values match, somatic filtering is skipped rather than removing all rows.

## 4. Single VCF

Single-VCF mode requires `--ncbi-build`, `--tumor-id`, `--ref-fasta`, `--alphamissense-file`, and a working vcf2maf/VEP installation.

Example for GRCh38:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-vcf data/sample.vcf \
    --output data/sample.mmrl.tsv \
    --tumor-id SAMPLE_T \
    --normal-id SAMPLE_N \
    --vcf-tumor-id TN2005D0839 \
    --vcf-normal-id NN2005D0839 \
    --ncbi-build GRCh38 \
    --ref-fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --vcf2maf /path/to/vcf2maf/vcf2maf.pl \
    --vep-path /path/to/ensembl-vep \
    --vep-data /path/to/.vep \
    --alphamissense-file /path/to/AlphaMissense_hg38.tsv.gz \
    --annotated-output sample.annotated.tsv \
    --qc-output sample.mmrl.qc.json
```

For GRCh37, change the build, reference FASTA, VEP cache, and AlphaMissense file consistently.

---

## 5. Directory of VCFs

Use directory mode for a cohort:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-vcf-dir data/vcfs \
    --output data/cohort.mmrl.tsv \
    --ncbi-build GRCh38 \
    --ref-fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --vcf2maf /path/to/vcf2maf/vcf2maf.pl \
    --vep-path /path/to/ensembl-vep \
    --vep-data /path/to/.vep \
    --alphamissense-file /path/to/AlphaMissense_hg38.tsv.gz
```

Recognized files:

```text
*.vcf
*.vcf.gz
```

Compressed files are staged as plain VCFs before vcf2maf in directory mode.

Search subdirectories with:

```bash
--recursive
```

### Sample IDs

By default, `Tumor_Sample_Barcode` comes from the tumor genotype sample name.

For a two-sample tumor/normal VCF where the same normal-column name is used across files:

```bash
--vcf-normal-id NORMAL
```

The other genotype sample is inferred as the tumor.

If genotype sample names are reused across files, use filename stems instead:

```bash
--batch-sample-id-source filename
```

The batch workflow collects variants from successfully prepared samples and scans the AlphaMissense database only once.

Useful options:

```text
--recursive
--batch-work-dir DIR
--batch-summary FILE
--batch-sample-id-source {vcf,filename}
--fail-fast
```

Without `--fail-fast`, failed VCFs are recorded and the remaining files continue.

Default batch outputs:

```text
<output>                     # combined MMRL TSV
<output>.batch/              # per-sample intermediate/output files
<output>.batch_summary.tsv   # sample-level processing summary
```

---

## 6. Output and QC

Final TSV:

```text
Hugo_Symbol	SBS96	am_pathogenicity	Tumor_Sample_Barcode	AAchange	Relative_Position
TP53	A[T>C]A	0.98	sample_001	R>H	0.42
KRAS	G[C>T]G	0.91	sample_001	G>D	0.18
```

Rows missing any of the six model features are removed.

For single-file workflows, QC is written by default to:

```text
<output>.qc.json
```

To save the retained mutation table with derived annotations:

```bash
--annotated-output data/sample.annotated.tsv
```

The QC report includes counts for input rows, retained missense/somatic rows, non-missing derived features, AlphaMissense matches, final rows/samples, and missing values by model column.

Directory mode writes per-sample QC/annotated files under the batch work directory and a cohort summary to `<output>.batch_summary.tsv`.

---

## 7. Troubleshooting

**Many SBS96 values are missing**

Check that the FASTA assembly matches the variants, REF agrees with the reference genome, and the records are SNVs. Common chromosome naming differences such as `1` versus `chr1` are handled automatically.

**Many AlphaMissense scores are missing**

Check the genome build and `Chromosome`, `Start_Position`, `Reference_Allele`, and `Tumor_Seq_Allele2`. AlphaMissense lookup uses genomic `CHROM:POS:REF:ALT`; HGVSp/AAchange is not part of the lookup key.

**AAchange is missing**

The script first parses `HGVSp_Short`/`HGVSp` (for example `p.R379C -> R>C`) and falls back to `Amino_acids` such as `R/C`.

**Relative_Position is missing**

Prefer `Protein_position` in `position/length` form, for example `379/393`. If the denominator is absent, provide `--protein-fasta` and a matching `Transcript_ID`.

---

## 8. Extract embeddings

```bash
python MMRL/extract_embeddings.py \
    --input_file data/mutations.mmrl.tsv \
    --checkpoint_dir ./checkpoints \
    --output_file ./outputs/sample_embeddings.npy \
    --output_pids ./outputs/sample_ids.tsv
```

Optional arguments:

```text
--device cpu|cuda:0|npu:0
--batch_size 256
--num_workers 16
```

Outputs:

```text
sample_embeddings.npy
sample_ids.tsv
```

Mutations are grouped by `Tumor_Sample_Barcode`, producing one embedding per sample.
