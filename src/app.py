"""
Oncology Navigator - Streamlit frontend.

Ties together every stage built so far into one interactive app:
  1. Input: gene name (text paste and VCF upload can be added later)
  2. Variant Interpretation: live ClinVar lookup
  3. Graph Matching: GraphSAGE ranking over live clinicaltrials.gov results
  4. Explanation: plain-language reasoning for each match

Run with:
    streamlit run src/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from torch_geometric.transforms import ToUndirected

from src.clinvar_client import get_variants_for_gene, count_variants
from src.trials_client import search_trials, count_trials
from src.graph_builder import build_graph, load_seed_table
from src.graphsage_model import OncologyGraphSAGE, get_embeddings, rank_trials_for_variant
from src.explanation_generator import build_template_explanation
from src.graph_viz import build_match_graph_figure

st.set_page_config(page_title="Oncology Navigator", page_icon="🧬", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    div[data-testid="stSidebar"] {
        background: #f0e9d8; border-right: 1px solid #ddd0b8;
    }
    div[data-testid="stSidebar"] label { font-weight: 600 !important; letter-spacing: 0.01em; }
    div[data-testid="stSidebar"] input {
        border-radius: 8px !important; border: 1px solid #c9b896 !important;
    }
    div[data-testid="stSidebar"] button {
        background: #0f6e75 !important;
        color: #faf6ee !important; border: none !important; font-weight: 600 !important;
        border-radius: 8px !important; padding: 0.6rem 0 !important;
        box-shadow: 0 2px 8px rgba(15, 110, 117, 0.25) !important;
    }

    .hero {
        padding: 2.2rem 2.6rem; border-radius: 14px; margin-bottom: 1.6rem;
        background: #ffffff;
        border: 1px solid #e6dcc6; border-left: 5px solid #0f6e75;
    }
    .hero h1 {
        font-family: 'Source Serif 4', serif; color: #2a1f1c; font-weight: 600;
        font-size: 2.3rem; margin: 0 0 0.5rem 0; letter-spacing: -0.01em;
    }
    .hero p {
        color: #5a4d44; font-size: 1.05rem; margin: 0; max-width: 660px; line-height: 1.5;
    }

    h2, h3, h4 { font-family: 'Source Serif 4', serif !important; color: #2a1f1c !important; }

    .match-card {
        background: #ffffff; border-radius: 10px; padding: 1.5rem 1.7rem;
        margin-bottom: 1.1rem; border: 1px solid #e6dcc6;
        box-shadow: 0 1px 4px rgba(122, 90, 60, 0.08);
        transition: box-shadow 0.15s ease;
    }
    .match-card:hover {
        box-shadow: 0 4px 14px rgba(122, 90, 60, 0.15);
    }
    .match-card.direct { border-left: 4px solid #0f6e75; }
    .match-card.graph { border-left: 4px solid #b0763f; }
    .match-card.weak { border-left: 4px solid #c9bfae; }
    .match-card h4 {
        font-family: 'Source Serif 4', serif !important;
        color: #2a1f1c !important; font-size: 1.2rem !important; font-weight: 600 !important;
    }
    .match-card p { color: #4a3f38 !important; }

    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.74rem; font-weight: 700; margin-right: 8px;
        letter-spacing: 0.03em; text-transform: uppercase;
    }
    .badge.direct { background: #dcedee; color: #0f6e75; }
    .badge.graph { background: #f5ead9; color: #8f5a24; }
    .badge.weak { background: #ece6da; color: #857868; }

    div[data-testid="stMetric"] {
        background: #ffffff; border-radius: 10px; padding: 0.9rem 1rem;
        border: 1px solid #e6dcc6;
    }
    div[data-testid="stMetricValue"] {
        color: #0f6e75 !important; font-family: 'Source Serif 4', serif;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🧬 Oncology Navigator</h1>
    <p>Enter a gene to find currently recruiting clinical trials that may be
    relevant, ranked using live ClinVar and clinicaltrials.gov data,
    with a graph model that catches connections plain keyword search misses.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("About this tool, how it works and what it doesn't do"):
    st.markdown("""
    **Pipeline:** a gene is looked up live against ClinVar for known pathogenic
    variants, then matched against currently recruiting trials from
    clinicaltrials.gov. A graph connecting variants, drugs, and trials
    (via a hand-curated set of confirmed oncology drug-gene links) is built
    and scored with a GraphSAGE model, so trials can be found even when
    their eligibility text never spells out the gene by name.

    **Match types:**
    - Direct text match: the trial's own eligibility criteria name the gene
    - Graph-connected match: the trial tests a drug our knowledge base
      confirms targets this gene, even though the text doesn't say so
    - Weak connection: no confirmed link either way, shown for completeness

    **What this isn't:** medical advice, a diagnostic tool, or a guarantee of
    trial eligibility. It's a research aid meant to surface trials worth
    discussing with a physician, built on a small, honestly labeled
    drug-gene knowledge base rather than a comprehensive clinical database.
    """)

with st.sidebar:
    st.header("Search")
    gene = st.text_input("Gene symbol", value="BRAF", help="e.g. BRAF, EGFR, BRCA1")
    condition = st.text_input("Condition", value="melanoma", help="e.g. melanoma, breast cancer")
    drug_hint = st.text_input(
        "Drug (optional)", value="",
        help="Narrows trial search to a specific drug, e.g. vemurafenib"
    )
    run_button = st.button("Find trials", type="primary")

st.divider()

if run_button:
    if not gene or not condition:
        st.error("Please enter both a gene and a condition.")
    else:
        with st.spinner(f"Pulling live pathogenic {gene} variants from ClinVar..."):
            try:
                variants = get_variants_for_gene(gene, limit=5)
                total_variant_count = count_variants(gene)
            except Exception as e:
                st.error(f"Couldn't reach ClinVar: {e}")
                variants = []
                total_variant_count = 0

        if not variants:
            st.warning(f"No pathogenic variant records found for {gene} in ClinVar.")
        else:
            st.success(
                f"Found {total_variant_count} pathogenic variant record(s) for {gene} "
                f"in ClinVar (showing top {len(variants)})."
            )
            with st.expander("Variant details"):
                for v in variants:
                    st.write(f"**{v.variation_name}**, {v.classification}")

            with st.spinner(f"Pulling live recruiting trials for '{condition}'..."):
                try:
                    trials = search_trials(
                        condition,
                        intervention=drug_hint if drug_hint else None,
                        page_size=10,
                    )
                    total_trial_count = count_trials(
                        condition, intervention=drug_hint if drug_hint else None
                    )
                except Exception as e:
                    st.error(f"Couldn't reach clinicaltrials.gov: {e}")
                    trials = []
                    total_trial_count = 0

            if not trials:
                st.warning("No currently recruiting trials found for that search.")
            else:
                st.success(
                    f"Found {total_trial_count} recruiting trial(s) "
                    f"(showing top {len(trials)})."
                )

                with st.spinner("Building match graph and ranking..."):
                    seed = load_seed_table("sample_data/drug_variant_seed.csv")
                    graph, idx_maps = build_graph(variants, trials, seed)
                    graph = ToUndirected()(graph)
                    model = OncologyGraphSAGE(
                        hidden_channels=16, out_channels=8, metadata=graph.metadata()
                    )
                    embeddings = get_embeddings(model, graph)
                    ranking = rank_trials_for_variant(
                        embeddings, idx_maps["variant"], idx_maps["trial"], gene
                    )

                nct_to_trial = {t.nct_id: t for t in trials}
                variant_classification = variants[0].classification
                known_drugs_for_gene = seed[seed["gene"].str.upper() == gene.upper()]["drug_name"].tolist()

                # Pre-compute match types once for the summary bar
                match_types = []
                for nct_id, score in ranking:
                    t = nct_to_trial[nct_id]
                    mentions_gene = gene.lower() in t.eligibility_criteria.lower()
                    exp = build_template_explanation(
                        gene=gene, classification=variant_classification,
                        trial_title=t.title, interventions=t.interventions,
                        conditions=t.conditions, mentions_gene=mentions_gene,
                        similarity_score=score, known_drugs_for_gene=known_drugs_for_gene,
                    )
                    match_types.append(exp.match_type)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total variants in ClinVar", total_variant_count)
                m2.metric("Total recruiting trials", total_trial_count)
                m3.metric("Direct text matches (top 10)", match_types.count("direct"))
                m4.metric("Graph-connected (top 10)", match_types.count("graph"))

                fig = build_match_graph_figure(gene, ranking, nct_to_trial, known_drugs_for_gene)
                if fig is not None:
                    st.subheader("How these matches connect")
                    st.caption(
                        f"{gene} in teal, confirmed drugs in amber, matching trials in "
                        "burgundy. A trial linked through a drug, rather than straight "
                        "to the gene, is exactly the kind of connection a plain keyword "
                        "search would miss."
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"Ranked trials for {gene}")

                badge_label = {
                    "direct": "Direct match",
                    "graph": "Graph-connected",
                    "weak": "Weak connection",
                }

                for nct_id, score in ranking:
                    t = nct_to_trial[nct_id]
                    mentions_gene = gene.lower() in t.eligibility_criteria.lower()

                    explanation = build_template_explanation(
                        gene=gene,
                        classification=variant_classification,
                        trial_title=t.title,
                        interventions=t.interventions,
                        conditions=t.conditions,
                        mentions_gene=mentions_gene,
                        similarity_score=score,
                        known_drugs_for_gene=known_drugs_for_gene,
                    )
                    mtype = explanation.match_type
                    interventions_text = ", ".join(t.interventions[:3]) if t.interventions else "Not specified"
                    locations_text = "; ".join(t.locations[:2]) if t.locations else "See ClinicalTrials.gov"

                    st.markdown(f"""
                    <div class="match-card {mtype}">
                        <span class="badge {mtype}">{badge_label[mtype]}</span>
                        <span style="color:#6b7280; font-size:0.85rem;">{t.nct_id} · score {score:+.3f}</span>
                        <h4 style="margin:0.4rem 0 0.6rem 0;">{t.title}</h4>
                        <p style="color:#374151; line-height:1.5;">{explanation.reasoning}</p>
                        <p style="color:#6b7280; font-size:0.85rem; margin-bottom:2px;">
                            <b>Status:</b> {t.status} &nbsp;·&nbsp; <b>Interventions:</b> {interventions_text}
                        </p>
                        <p style="color:#6b7280; font-size:0.85rem;">
                            <b>Locations:</b> {locations_text}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                st.caption(
                    "This tool surfaces potentially relevant trials using public data. "
                    "It is not medical advice. Always discuss trial eligibility with a "
                    "treating physician."
                )
else:
    st.info("Enter a gene and condition in the sidebar, then click **Find trials**.")
