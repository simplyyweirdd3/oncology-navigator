"""
Interactive graph visualization for Oncology Navigator.

Builds a small, readable subgraph for the current search: the searched
gene at the center, its confirmed drugs from the seed table, and the
top-ranked trials, connected either directly to the gene (text match) or
through a drug (graph-connected match). Weak-connection trials are left
out here, since they have no real edge to draw, the point of this view
is to make the actual connections visible, not just list everything.
"""

import networkx as nx
import plotly.graph_objects as go


def build_match_graph_figure(gene: str, ranking: list, nct_to_trial: dict,
                              known_drugs_for_gene: list, top_n: int = 6):
    """
    ranking: list of (nct_id, score) tuples, already sorted best-first.
    Returns a Plotly Figure, or None if there's nothing worth drawing
    (e.g. every top match is a weak connection with no real edge).
    """
    G = nx.Graph()
    G.add_node(gene, kind="gene")

    known_drugs_lower = {d.lower(): d for d in known_drugs_for_gene}
    added_any_trial = False

    for nct_id, score in ranking[:top_n]:
        t = nct_to_trial[nct_id]
        mentions_gene = gene.lower() in t.eligibility_criteria.lower()
        matched_drug = next(
            (i for i in t.interventions if i.strip().lower() in known_drugs_lower), None
        )

        if not mentions_gene and not matched_drug:
            continue  # weak connection, nothing real to draw

        trial_label = t.nct_id
        G.add_node(trial_label, kind="trial", title=t.title)
        added_any_trial = True

        if mentions_gene:
            G.add_edge(gene, trial_label, kind="direct")
        if matched_drug:
            drug_label = known_drugs_lower[matched_drug.strip().lower()]
            if drug_label not in G:
                G.add_node(drug_label, kind="drug")
            G.add_edge(gene, drug_label, kind="targets")
            G.add_edge(drug_label, trial_label, kind="tests")

    if not added_any_trial:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.9)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.4, color="#c9bfae"),
        hoverinfo="none", showlegend=False,
    )

    color_map = {"gene": "#0f6e75", "drug": "#8f5a24", "trial": "#7a2e2e"}
    size_map = {"gene": 34, "drug": 22, "trial": 22}

    node_traces = []
    for kind in ("gene", "drug", "trial"):
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        if not nodes:
            continue
        node_traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker=dict(size=size_map[kind], color=color_map[kind],
                        line=dict(width=2, color="#ffffff")),
            text=nodes,
            textposition="bottom center",
            textfont=dict(size=11, color="#2a1f1c"),
            hovertext=[G.nodes[n].get("title", n) for n in nodes],
            hoverinfo="text",
            name=kind.capitalize(),
            showlegend=True,
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#faf6ee",
        paper_bgcolor="#faf6ee",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
    )
    return fig
