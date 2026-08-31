# Missense Mutation Embedding Extraction

Extract tumor sample missense mutation embeddings using a pretrained `MMRL`.

<img src="mmrl.png" alt="mmrl" style="zoom: 50%;" />

## Installation

Clone the repository:

```bash
git clone https://github.com/charlin90/MMRL.git
cd MMRL
```

Install Python dependencies:

```bash
pip install numpy pandas torch scikit-learn joblib tqdm pysam
```

Make sure the checkpoint directory contains:

```text
checkpoints/
├── best_model.pth
├── gene_encoder.pkl
└── mut_type_encoder.pkl
```

## Input data preparation

MMRL uses mutation-level features derived from somatic missense SNVs. The preprocessing script accepts three mutually exclusive input formats:

```text
VCF ── vcf2maf + VEP ──┐
                       │
MAF ───────────────────┼─> missense/somatic filtering
                       │   -> SBS96
TSV ───────────────────┘   -> AAchange
                           -> Relative_Position
                           -> AlphaMissense score
                           -> MMRL input TSV
```

Use exactly one of `--input-vcf`, `--input-maf`, or `--input-tsv`. VCF input is converted with `vcf2maf`; MAF and TSV inputs skip that conversion and enter the feature-preparation steps directly.

The repository provides `MMRL/prepare_mmrl_input.py` as a single command-line entry point. It replaces dataset-specific `awk` column indices and the separate SBS96 / protein-length / amino-acid-change scripts with column-name-based processing.

### External prerequisites

For VCF input, install and configure:

- `vcf2maf` with VEP support. Use a recent version that supports `--vep-plugins` and `--retain-ann` (vcf2maf >= 1.6.21).
- Ensembl VEP and the VEP cache matching the genome build.
- Ensembl VEP `AlphaMissense` plugin.
- AlphaMissense precomputed data for the matching build (`hg19/GRCh37` or `hg38/GRCh38`).
- `samtools` and `tabix`.

The reference FASTA should have a `.fai` index:

```bash
samtools faidx Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

AlphaMissense data should be tabix indexed before the first VEP run. For example:

```bash
tabix -s 1 -b 2 -e 2 -f -S 1 AlphaMissense_hg38.tsv.gz
```

Use the hg19 AlphaMissense file for GRCh37 data.

### Prepare MMRL input directly from VCF

The input VCF should contain somatic mutation calls for one tumor sample (with an optional matched normal). For a GRCh38 tumor-normal VCF:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-vcf data/sample.vcf.gz \
    --output data/sample.mmrl.tsv \
    --tumor-id SAMPLE_T \
    --normal-id SAMPLE_N \
    --ncbi-build GRCh38 \
    --ref-fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --vcf2maf /path/to/vcf2maf.pl \
    --vep-path /path/to/vep/bin \
    --vep-data /path/to/.vep \
    --alphamissense-file /path/to/AlphaMissense_hg38.tsv.gz
```

If the VCF genotype columns are literally named `TUMOR` and `NORMAL`, but you want the output sample IDs to be `SAMPLE_T` and `SAMPLE_N`, add:

```bash
    --vcf-tumor-id TUMOR \
    --vcf-normal-id NORMAL
```

`vcf2maf` runs VEP with total protein length enabled, so its `Protein_position` is normally formatted like `379/393`. `prepare_mmrl_input.py` uses this directly to calculate:

```text
Relative_Position = protein_position / protein_length
```

Therefore, an additional peptide FASTA is normally not required for the VCF workflow.

### Prepare MMRL input from an existing MAF

An existing MAF can be processed without rerunning `vcf2maf`:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-maf data/sample.maf \
    --output data/sample.mmrl.tsv \
    --ref-fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

`--ref-fasta` is only required here when `SBS96` is not already present (or when `--recompute-sbs96` is requested).

The script filters `Variant_Classification == Missense_Mutation` by column name. If `Mutation_Status` is present and contains `Somatic`/`SOMATIC`, it is also used automatically. For a different somatic-status column:

```bash
    --somatic-column mutation_status \
    --somatic-values Somatic SOMATIC
```

If the MAF has already been restricted to somatic calls, somatic-status filtering can be disabled:

```bash
    --no-somatic-filter
```

If AlphaMissense scores are stored under another column name:

```bash
    --alphamissense-column YOUR_SCORE_COLUMN
```

For legacy MAF files where `Protein_position` contains only a residue position (for example `379`) rather than `379/393`, provide an Ensembl peptide FASTA as a fallback:

```bash
    --protein-fasta /path/to/Homo_sapiens.GRCh38.pep.all.fa.gz
```

The fallback matches `Transcript_ID` to the `transcript:` identifier in the Ensembl peptide FASTA and uses the corresponding protein sequence length.

### Prepare MMRL input from an existing TSV

A tab-separated mutation table can be supplied explicitly with `--input-tsv`:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-tsv data/mutations.tsv \
    --output data/mutations.mmrl.tsv \
    --ref-fasta /path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

If the TSV uses standard MAF-style column names, no additional mapping is needed. The normal preprocessing path uses columns such as:

```text
Hugo_Symbol
Variant_Classification
Mutation_Status                 # optional
Chromosome
Start_Position
End_Position                    # optional for SNVs
Reference_Allele
Tumor_Seq_Allele2
Tumor_Sample_Barcode
HGVSp_Short or HGVSp
Protein_position
Transcript_ID                   # needed only for protein-length fallback
am_pathogenicity
```

For cohort-specific TSV files with different column names, map them to canonical names with `--column-map CANONICAL=SOURCE`. For example:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-tsv data/cohort_mutations.tsv \
    --output data/cohort.mmrl.tsv \
    --ref-fasta /path/to/hg38.fa \
    --column-map \
        Hugo_Symbol=Gene \
        Variant_Classification=Consequence \
        Mutation_Status=Somatic_Status \
        Tumor_Sample_Barcode=Sample_ID \
        HGVSp_Short=Protein_Change \
        Protein_position=Protein_Position \
        am_pathogenicity=AlphaMissense_score
```

This makes preprocessing independent of dataset-specific column numbers such as `$10`, `$27`, `$41`, or `$46`.

If a TSV has already been restricted to missense mutations and does not contain `Variant_Classification`, use:

```bash
    --no-missense-filter
```

If it already contains an `SBS96` column, the script reuses it and `--ref-fasta` is not required for SBS96. To deliberately recalculate the context from the genome, use:

```bash
    --recompute-sbs96 --ref-fasta /path/to/reference.fa
```

Similarly, existing `AAchange` and `Relative_Position` columns are reused. Missing values can instead be derived from `HGVSp_Short`/`Amino_acids` and `Protein_position`.

A TSV that already contains exactly the six MMRL model features can also be supplied directly:

```bash
python MMRL/prepare_mmrl_input.py \
    --input-tsv data/already_prepared.tsv \
    --output data/validated.mmrl.tsv
```

In this case the script validates numeric fields, removes incomplete rows, records QC statistics, and does not require a reference FASTA.

### Preprocessing outputs and QC

The final TSV contains exactly the six features used by MMRL:

```text
Hugo_Symbol
SBS96
am_pathogenicity
Tumor_Sample_Barcode
AAchange
Relative_Position
```

A QC report is written automatically to:

```text
<output>.qc.json
```

It reports the number of input mutations, retained missense/somatic mutations, successful SBS96 annotations, AlphaMissense scores, AA changes, relative positions, final complete mutation rows, and final sample count.

To also save the full annotated mutation table before the final six-column selection:

```bash
    --annotated-output data/sample.annotated.tsv
```

Rows missing any of the six required model features are excluded from the final MMRL input. This is consistent with the embedding extraction code, which requires all six fields.

## Input File

If preprocessing has already been completed, the input file should be a tab-separated TSV file with the following columns:

```text
Hugo_Symbol
SBS96
am_pathogenicity
Tumor_Sample_Barcode
AAchange
Relative_Position
```

Example:

```text
Hugo_Symbol	SBS96	am_pathogenicity	Tumor_Sample_Barcode	AAchange	Relative_Position
TP53	A[T>C]A	0.98	sample_001	R>H	0.42
KRAS	G[C>T]G	0.91	sample_001	G>D	0.18
PIK3CA	A[T>G]G	0.87	sample_002	H>R	0.63
```

Mutations are grouped by `Tumor_Sample_Barcode` to generate one embedding for each sample.

## Usage

Run:

```bash
python MMRL/extract_embeddings.py \
    --input_file data/mutations.tsv \
    --checkpoint_dir ./checkpoints \
    --output_file ./outputs/sample_embeddings.npy \
    --output_pids ./outputs/sample_ids.tsv
```

For a specific device:

```bash
python MMRL/extract_embeddings.py \
    --input_file data/mutations.tsv \
    --checkpoint_dir ./checkpoints \
    --output_file ./outputs/sample_embeddings.npy \
    --output_pids ./outputs/sample_ids.tsv \
    --device npu:0
```

You can also use:

```bash
--device cpu
```

or:

```bash
--device cuda:0
```

Optional arguments:

```text
--batch_size     Inference batch size (default: 256)
--num_workers    Number of DataLoader workers (default: 16)
```

The script outputs:

```text
sample_embeddings.npy    # sample-level embeddings
sample_ids.tsv            # corresponding sample IDs
```
