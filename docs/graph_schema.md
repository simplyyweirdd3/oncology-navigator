# Variant-Drug-Trial Graph Schema

This is the design for Stage 3 of Oncology Navigator, the graph that
GraphSAGE will run over to rank trial matches. Written up before touching
code so the graph construction logic has a clear target.

## Why a graph instead of direct matching?

Direct matching (variant text -> trial eligibility text) only catches
trials that explicitly name the variant. A graph catches trials that test
a *drug* known to target that variant, even when the trial's eligibility
text doesn't spell out the biomarker. Example: a patient with an EGFR
exon 19 deletion should match trials testing osimertinib, even if a
specific trial's text says "advanced NSCLC" without ever saying "EGFR."
The variant -> drug -> trial path is what makes that visible.

## Node types

### 1. Variant nodes
Source: `clinvar_client.py` output (`VariantRecord`)

| Field | Example |
|---|---|
| gene | EGFR |
| variation_name | NM_005228.5(EGFR):c.2235_2249del |
| classification | Pathogenic |
| associated_traits | ["Non-small cell lung cancer"] |

### 2. Drug nodes
Source: new lookup, built from the `interventions` field already returned by
`trials_client.py`, deduplicated across trials. Each drug node also gets
enriched with known variant-targeting relationships (a small manually
curated seed table to start, since there's no single clean free API for
"drug targets variant X" the way there is for ClinVar or clinicaltrials.gov).

| Field | Example |
|---|---|
| name | Osimertinib |
| target_variants | ["EGFR exon 19 deletion", "EGFR L858R", "EGFR T790M"] |
| drug_class | EGFR tyrosine kinase inhibitor |

### 3. Trial nodes
Source: `trials_client.py` output (`TrialRecord`)

| Field | Example |
|---|---|
| nct_id | NCT04487080 |
| conditions | ["Non-small cell lung cancer"] |
| interventions | ["Osimertinib"] |
| eligibility_criteria | (full text) |
| status | RECRUITING |

## Edge types

| Edge | Direction | Meaning | Source |
|---|---|---|---|
| `TARGETS` | Drug -> Variant | This drug is known to target this variant | seed table (curated) |
| `TESTS` | Trial -> Drug | This trial administers this drug | parsed from `interventions` |
| `MENTIONS` | Trial -> Variant | Trial eligibility text explicitly names this variant/gene | text match on `eligibility_criteria` |

Two paths reach a trial from a variant:
- **Direct**: Variant --MENTIONS(reverse)--> Trial (explicit text match, high confidence)
- **Indirect**: Variant --TARGETS(reverse)-- Drug --TESTS(reverse)--> Trial (the "non-obvious" match GraphSAGE surfaces)

## What GraphSAGE actually does here

GraphSAGE learns a numerical embedding (a vector) for every node by
aggregating information from its neighbors. Once trained, a patient's
variant node has an embedding, and every trial node has an embedding.
Ranking = sort trials by embedding similarity (cosine similarity) to the
patient's variant embedding. Trials reachable only through the indirect
drug path still end up with embeddings "close to" the variant, because
GraphSAGE aggregates through the drug node in between, that's the whole
point, it's why this beats keyword search.

## Seed data needed before this can run

The one piece with no free live API is the drug-target-variant seed table.
It currently covers 61 real, clinically documented drug-gene pairs across
34 genes (EGFR, BRAF, HER2, BRCA1/2, ALK, KRAS, MET, RET, ROS1, NTRK1-3,
FLT3, IDH1/2, JAK2, BCR-ABL1, KIT, PDGFRA, VHL, FGFR2/3, PIK3CA, ESR1,
NRAS, MAP2K1, ATM, PALB2, CHEK2, and the Lynch syndrome mismatch repair
genes), spanning lung, breast, blood, GI, and renal cancers. Every entry
is a real FDA-approved or clinically established targeted therapy
relationship, hand-checked rather than pulled from an automated source.
This is normal and expected for a student project, real precision oncology
platforms maintain similar curated knowledge bases (e.g. OncoKB) because
this information isn't standardized anywhere else for free.

## Next build steps (for next session)

1. Build `drug_variant_seed.csv` (~15-20 rows, hand-curated)
2. Write `graph_builder.py`: takes VariantRecord list + TrialRecord list +
   seed CSV, outputs a `torch_geometric` HeteroData graph object
3. Write a basic GraphSAGE model (2-layer, node embedding output)
4. Test: does a variant with no direct trial text match still surface a
   relevant trial through the drug path? This is the demo moment.
