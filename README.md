# Oncology Navigator

An end-to-end pipeline that takes a patient's genomic variant data and
returns ranked, currently recruiting clinical trials they may qualify for,
with plain-language explanations for each match.

Built for [your name], MS Data Science, Seattle University.

## Why this exists

Precision oncology trials are matched on biomarkers (BRCA1, EGFR, HER2,
BRAF, etc), but eligibility criteria are buried in dense unstructured text,
and trial data is scattered across a live, constantly-updating registry.
Oncology Navigator connects the two: interpret the variant, then find the
trials that variant unlocks.

## Pipeline

1. **Input** — gene/mutation form, raw variant text paste, or VCF upload
2. **Variant Interpretation** — live ClinVar lookup for pathogenicity + associated conditions
3. **Graph Matching** — GraphSAGE over a variant-drug-trial graph to rank live clinicaltrials.gov results, catching non-obvious matches
4. **Explanation** — LLM-generated plain-language "why this trial matches" summary
5. **Frontend** — Streamlit app deployed on Hugging Face Spaces

## Status

- [x] Project scaffolded
- [x] ClinVar client (`src/clinvar_client.py`) — live tested against NCBI E-utilities
- [x] ClinicalTrials.gov client (`src/trials_client.py`) — live tested against API v2
- [ ] Variant-drug-trial graph schema
- [ ] GraphSAGE ranking model
- [ ] LLM explanation layer (Qwen2.5)
- [ ] Streamlit frontend
- [ ] Deploy to Hugging Face Spaces

## Data sources

- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) via [E-utilities](https://www.ncbi.nlm.nih.gov/clinvar/docs/programmatic_access/) — free, no key required for light use
- [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) — free, no key required

## Setup

```bash
pip install -r requirements.txt
python src/clinvar_client.py    # quick manual test against live ClinVar
python src/trials_client.py     # quick manual test against live trials API
```
