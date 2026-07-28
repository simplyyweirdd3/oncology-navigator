"""
Graph builder for Oncology Navigator.

Assembles the variant-drug-trial heterogeneous graph described in
docs/graph_schema.md, from three sources:
  - VariantRecord objects (from clinvar_client.py)
  - TrialRecord objects (from trials_client.py)
  - the hand-curated drug-variant seed table (sample_data/drug_variant_seed.csv)

Output is a torch_geometric HeteroData object, ready for a GraphSAGE model.
"""

import re
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for loose text matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def load_seed_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_graph(variant_records: list, trial_records: list, seed_df: pd.DataFrame) -> tuple:
    """
    Build the heterogeneous graph.

    Returns (HeteroData graph, index_maps) where index_maps is a dict of
    dicts letting you go from a real-world id (gene name, drug name, NCT id)
    back to its integer node index in the graph, needed later to look up
    "which trial is ranked #1" after the model runs.
    """
    # --- Node indexing ---
    # Variant nodes: one per unique gene in our variant records (simplification
    # for v1; a future version could split by specific mutation instead of gene)
    variant_genes = sorted({v.gene for v in variant_records})
    variant_idx = {gene: i for i, gene in enumerate(variant_genes)}

    # Drug nodes: union of drugs from the seed table and from trial interventions
    seed_drugs = set(seed_df["drug_name"].str.strip())
    trial_drugs = set()
    for t in trial_records:
        trial_drugs.update(i.strip() for i in t.interventions if i.strip())
    all_drugs = sorted(seed_drugs | trial_drugs)
    drug_idx = {name: i for i, name in enumerate(all_drugs)}

    # Trial nodes
    trial_ids = [t.nct_id for t in trial_records]
    trial_idx = {nct: i for i, nct in enumerate(trial_ids)}

    # --- Edge construction ---

    # Drug --TARGETS--> Variant, from the seed table
    targets_src, targets_dst = [], []
    for _, row in seed_df.iterrows():
        gene = row["gene"].strip()
        drug = row["drug_name"].strip()
        if gene in variant_idx and drug in drug_idx:
            targets_src.append(drug_idx[drug])
            targets_dst.append(variant_idx[gene])

    # Trial --TESTS--> Drug, from trial interventions matched against known drugs
    tests_src, tests_dst = [], []
    for t in trial_records:
        t_i = trial_idx[t.nct_id]
        for intervention in t.interventions:
            name = intervention.strip()
            if name in drug_idx:
                tests_src.append(t_i)
                tests_dst.append(drug_idx[name])

    # Trial --MENTIONS--> Variant, loose text match of gene name in eligibility text
    mentions_src, mentions_dst = [], []
    for t in trial_records:
        t_i = trial_idx[t.nct_id]
        elig_norm = _normalize(t.eligibility_criteria)
        for gene, g_i in variant_idx.items():
            if _normalize(gene) in elig_norm:
                mentions_src.append(t_i)
                mentions_dst.append(g_i)

    # --- Assemble HeteroData ---
    data = HeteroData()

    # Placeholder node features: one-hot-ish identity for now (v1). A future
    # version can replace these with real embeddings (e.g. classification
    # status for variants, drug class one-hot for drugs, phase/status for trials).
    data["variant"].num_nodes = len(variant_idx)
    data["drug"].num_nodes = len(drug_idx)
    data["trial"].num_nodes = len(trial_idx)

    data["variant"].x = torch.eye(len(variant_idx)) if variant_idx else torch.zeros((0, 1))
    data["drug"].x = torch.eye(len(drug_idx)) if drug_idx else torch.zeros((0, 1))
    data["trial"].x = torch.eye(len(trial_idx)) if trial_idx else torch.zeros((0, 1))

    def edge_tensor(src, dst):
        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    data["drug", "targets", "variant"].edge_index = edge_tensor(targets_src, targets_dst)
    data["trial", "tests", "drug"].edge_index = edge_tensor(tests_src, tests_dst)
    data["trial", "mentions", "variant"].edge_index = edge_tensor(mentions_src, mentions_dst)

    index_maps = {
        "variant": variant_idx,
        "drug": drug_idx,
        "trial": trial_idx,
    }

    return data, index_maps


if __name__ == "__main__":
    # Smoke test with the saved samples we already have, confirms the
    # construction logic runs end to end before wiring in live API calls.
    import json
    import sys
    sys.path.insert(0, ".")
    from src.trials_client import _parse_study

    seed = load_seed_table("sample_data/drug_variant_seed.csv")

    with open("sample_data/trial_sample.json") as f:
        trial = _parse_study(json.load(f))

    # fabricate a couple of matching variant records for the smoke test
    class FakeVariant:
        def __init__(self, gene):
            self.gene = gene

    variants = [FakeVariant("BRAF"), FakeVariant("EGFR")]

    graph, idx_maps = build_graph(variants, [trial], seed)
    print(graph)
    print("Index maps:", idx_maps)
