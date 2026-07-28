# Oncology Navigator

An end-to-end pipeline that takes a patient's genomic variant data and returns ranked, currently recruiting clinical trials they may qualify for, with plain-language explanations for each match.

Built by Ruman Sidhu, MS Data Science, Seattle University.

## Why this exists

I did not start in medicine. I started as a writer, and somewhere along the way I became someone who is also obsessed with data. This project sits exactly at that intersection, the place where a messy, human problem needed someone willing to read closely and someone willing to build carefully, and it turned out I could be both.

Here is the problem, in plain terms. A patient's genetic test comes back with a list of mutations. Somewhere out there, hundreds of clinical trials are recruiting patients whose tumors carry exactly those mutations, trials that could mean access to a treatment not otherwise available. But the connection between "this is my mutation" and "this is my trial" is buried in dense, inconsistent, jargon-heavy text that even doctors sometimes have to dig through by hand. Nobody designed that gap on purpose. It is just what happens when a field moves fast and the infrastructure catches up slowly.

I wanted to build something that closes that gap, not with a chatbot that guesses, but with something that actually reads the eligibility criteria, actually looks up the variant, and actually reasons through why a match makes sense or doesn't. Oncology Navigator does that. It takes a variant, checks it against ClinVar's live archive of clinical significance, and searches currently recruiting trials on clinicaltrials.gov, then explains its reasoning the way I would want a friend in medicine to explain it to me, plainly, and without hiding the uncertainty.

The best tools get built by people willing to sit with a problem long enough to actually understand it, not just automate around it. That's the ethos behind this project, and behind most of the work I want to do.

## Pipeline

1. **Input** - gene/mutation form, raw variant text paste, or VCF upload
2. **Variant Interpretation** - live ClinVar lookup for pathogenicity + associated conditions
3. **Graph Matching** - GraphSAGE over a variant-drug-trial graph to rank live clinicaltrials.gov results, catching non-obvious matches
4. **Explanation** - LLM-generated plain-language "why this trial matches" summary
5. **Frontend** - Streamlit app deployed on Hugging Face Spaces

## Status

- [x] Project scaffolded
- [x] ClinVar client (`src/clinvar_client.py`) - live tested against NCBI E-utilities
- [x] ClinicalTrials.gov client (`src/trials_client.py`) - live tested against API v2
- [x] Variant-drug-trial graph schema (`docs/graph_schema.md`)
- [x] GraphSAGE ranking model (`src/graphsage_model.py`) - tested against live ClinVar + clinicaltrials.gov data
- [ ] LLM explanation layer (Qwen2.5)
- [ ] Streamlit frontend
- [ ] Deploy to Hugging Face Spaces

## Data sources

- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) via [E-utilities](https://www.ncbi.nlm.nih.gov/clinvar/docs/programmatic_access/) - free, no key required for light use
- [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) - free, no key required

## What the live test actually taught me

I ran the full pipeline against real, live BRAF variants and real, currently
recruiting melanoma trials testing vemurafenib, and both trials that came
back explicitly said "BRAF" in their eligibility text. At first that felt
like the graph hadn't proven its point. But sitting with it, I realized
that's actually the correct and expected result, not a miss.

Here's why. Trials built around a targeted therapy like vemurafenib almost
always spell out the biomarker in their inclusion criteria, because that's
how the drug got approved in the first place, on the condition that only
biomarker-positive patients receive it. Regulators require it, so trial
writers state it plainly. A well-designed precision oncology trial being
upfront about its biomarker isn't a flaw in my system, it's the field
working the way it's supposed to.

Where the graph actually earns its keep is messier territory: a
combination trial where only one drug out of several targets the variant,
a trial that names a drug class instead of the drug itself, or a biomarker
that's implied by an approved indication without ever being spelled out
in so many words. Direct text matching handles the clean, well-labeled
cases fine on its own. The graph is for the cases that aren't clean.

I'd rather write that down honestly than pretend my first live run found
some dramatic hidden connection it didn't. The mechanism is sound, and an
earlier synthetic test already confirmed the indirect variant-drug-trial
path fires correctly when a trial doesn't mention the gene by name. The
real lesson from today is a domain one, not a bug: sometimes the "boring"
result is the one worth understanding.

## Setup

```bash
pip install -r requirements.txt
python src/clinvar_client.py    # quick manual test against live ClinVar
python src/trials_client.py     # quick manual test against live trials API
```
