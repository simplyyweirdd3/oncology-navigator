"""
End-to-end live demo for Oncology Navigator.

Runs the full pipeline against real, live data:
  1. Pull real pathogenic variants for a gene from ClinVar
  2. Pull real currently-recruiting trials for a related condition/drug
     from clinicaltrials.gov
  3. Build the variant-drug-trial graph
  4. Run GraphSAGE and rank the live trials

This is the actual proof-of-concept moment: a trial that never mentions
the gene by name in its eligibility text can still rank highly if it
uses a drug known to target that gene's variant, because the graph
connects them through the drug node.

Usage:
    python3 src/run_live_demo.py
"""

import sys
sys.path.insert(0, ".")

from src.clinvar_client import get_variants_for_gene
from src.trials_client import search_trials
from src.graph_builder import build_graph, load_seed_table
from src.graphsage_model import OncologyGraphSAGE, get_embeddings, rank_trials_for_variant
from torch_geometric.transforms import ToUndirected


def run_demo(gene: str, condition: str, drug_hint: str = None):
    print(f"\n{'='*60}")
    print(f"ONCOLOGY NAVIGATOR, LIVE DEMO for {gene}")
    print(f"{'='*60}\n")

    print(f"[1/4] Pulling live pathogenic {gene} variants from ClinVar...")
    variants = get_variants_for_gene(gene, limit=5)
    print(f"      Found {len(variants)} variant records.")
    for v in variants[:3]:
        print(f"        - {v.variation_name} ({v.classification})")

    print(f"\n[2/4] Pulling live recruiting trials for '{condition}'"
          + (f" + {drug_hint}" if drug_hint else "") + " from clinicaltrials.gov...")
    trials = search_trials(condition, intervention=drug_hint, page_size=8)
    print(f"      Found {len(trials)} recruiting trials.")
    for t in trials[:5]:
        mentions_gene = gene.lower() in t.eligibility_criteria.lower()
        flag = "mentions gene in text" if mentions_gene else "NO gene text match"
        print(f"        - {t.nct_id}: {t.title[:70]}... [{flag}]")

    print(f"\n[3/4] Building variant-drug-trial graph...")
    seed = load_seed_table("sample_data/drug_variant_seed.csv")
    graph, idx_maps = build_graph(variants, trials, seed)
    graph = ToUndirected()(graph)
    print(f"      variant nodes: {graph['variant'].num_nodes}, "
          f"drug nodes: {graph['drug'].num_nodes}, "
          f"trial nodes: {graph['trial'].num_nodes}")

    print(f"\n[4/4] Running GraphSAGE and ranking trials for {gene}...")
    model = OncologyGraphSAGE(hidden_channels=16, out_channels=8, metadata=graph.metadata())
    embeddings = get_embeddings(model, graph)
    ranking = rank_trials_for_variant(embeddings, idx_maps["variant"], idx_maps["trial"], gene)

    print(f"\n{'='*60}")
    print(f"RANKED TRIALS FOR {gene}")
    print(f"{'='*60}")
    nct_to_trial = {t.nct_id: t for t in trials}
    for nct_id, score in ranking:
        t = nct_to_trial[nct_id]
        mentions_gene = gene.lower() in t.eligibility_criteria.lower()
        tag = "[TEXT MATCH]" if mentions_gene else "[GRAPH-ONLY MATCH]"
        print(f"\n  {score:+.4f}  {nct_id}  {tag}")
        print(f"           {t.title[:80]}")
        print(f"           Interventions: {', '.join(t.interventions[:3])}")


if __name__ == "__main__":
    # BRAF is the clearest demo case: vemurafenib is a well-known
    # BRAF-targeting drug already in our seed table. Filtering trials by
    # this drug means the graph's indirect variant->drug->trial path has
    # something real to connect, so we can actually see whether a trial
    # ranks highly through that path even if its own text stays quiet
    # about "BRAF" specifically.
    run_demo(gene="BRAF", condition="melanoma", drug_hint="vemurafenib")
