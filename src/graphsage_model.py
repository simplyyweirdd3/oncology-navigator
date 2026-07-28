"""
GraphSAGE model for Oncology Navigator.

Takes the HeteroData graph built by graph_builder.py and learns node
embeddings, one vector per variant, drug, and trial node. Ranking a
patient's trials is then just cosine similarity between their variant's
embedding and every trial's embedding, sorted descending.

Why this works even with a small graph: GraphSAGE builds each node's
embedding by aggregating its neighbors' features. A trial connected to
a variant only through a drug (the "indirect" path from the schema doc)
still ends up with an embedding pulled toward that variant, because the
drug node sits between them and passes information both ways during
message passing. That's the whole mechanism behind catching non-obvious
matches.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.transforms import ToUndirected


class OncologyGraphSAGE(torch.nn.Module):
    """
    Two-layer heterogeneous GraphSAGE encoder.

    One SAGEConv per edge type per layer, combined with HeteroConv, which
    handles the bookkeeping of "variant nodes get updated from drug nodes
    via the targets edge, trial nodes get updated from drug nodes via the
    tests edge," etc, all in one forward pass.
    """

    def __init__(self, hidden_channels: int, out_channels: int, metadata):
        super().__init__()
        node_types, edge_types = metadata

        self.conv1 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types},
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), out_channels) for edge_type in edge_types},
            aggr="mean",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


def get_embeddings(model, data):
    """Run a forward pass, return a dict of {node_type: embedding tensor}."""
    model.eval()
    with torch.no_grad():
        return model(data.x_dict, data.edge_index_dict)


def rank_trials_for_variant(embeddings: dict, variant_idx: dict, trial_idx: dict,
                              gene: str) -> list:
    """
    Given node embeddings and a gene name, return trials ranked by cosine
    similarity to that variant's embedding, most relevant first.

    Returns a list of (nct_id, similarity_score) tuples.
    """
    if gene not in variant_idx:
        return []

    v_i = variant_idx[gene]
    variant_vec = embeddings["variant"][v_i].unsqueeze(0)  # shape (1, dim)
    trial_vecs = embeddings["trial"]  # shape (num_trials, dim)

    sims = F.cosine_similarity(variant_vec, trial_vecs)  # shape (num_trials,)

    id_to_nct = {i: nct for nct, i in trial_idx.items()}
    ranked = sorted(
        ((id_to_nct[i], sims[i].item()) for i in range(len(trial_idx))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked


if __name__ == "__main__":
    # Smoke test: reuse the same fabricated graph from graph_builder.py's
    # own test, confirms the model runs end to end and produces a ranking.
    import json
    import sys
    sys.path.insert(0, ".")
    from src.trials_client import _parse_study
    from src.graph_builder import build_graph, load_seed_table

    seed = load_seed_table("sample_data/drug_variant_seed.csv")

    with open("sample_data/trial_sample.json") as f:
        trial = _parse_study(json.load(f))

    class FakeVariant:
        def __init__(self, gene):
            self.gene = gene

    variants = [FakeVariant("BRAF"), FakeVariant("EGFR")]

    graph, idx_maps = build_graph(variants, [trial], seed)

    # Add reverse edges (variant->drug, drug->trial, etc) so every node
    # type receives messages and gets its embedding updated each layer.
    graph = ToUndirected()(graph)

    model = OncologyGraphSAGE(hidden_channels=16, out_channels=8, metadata=graph.metadata())
    embeddings = get_embeddings(model, graph)

    print("Embedding shapes:")
    for node_type, emb in embeddings.items():
        print(f"  {node_type}: {emb.shape}")

    print("\nRanking trials for BRAF variant:")
    ranking = rank_trials_for_variant(embeddings, idx_maps["variant"], idx_maps["trial"], "BRAF")
    for nct_id, score in ranking:
        print(f"  {nct_id}: similarity = {score:.4f}")
