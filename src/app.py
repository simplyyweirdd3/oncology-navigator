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

from src.clinvar_client import get_variants_for_gene
from src.trials_client import search_trials
from src.graph_builder import build_graph, load_seed_table
from src.graphsage_model import OncologyGraphSAGE, get_embeddings, rank_trials_for_variant
from src.explanation_generator import build_template_explanation

st.set_page_config(page_title="Oncology Navigator", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #fafbfc 0%, #f0f4f8 100%); }
    div[data-testid="stSidebar"] { background: #1a2332; }
    div[data-testid="stSidebar"] * { color: #e8edf2 !important; }
    div[data-testid="stSidebar"] input {
        background: #2a3a4f !important; color: #ffffff !important;
        border: 1px solid #3d5170 !important;
    }
    div[data-testid="stSidebar"] button {
        background: #d94f4f !important; color: white !important;
        border: none !important; font-weight: 600 !important;
    }
    h1 { color: #1a2332; }
    .match-card {
        background: white; border-radius: 12px; padding: 1.4rem 1.6rem;
        margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #d1d5db;
    }
    .match-card.direct { border-left-color: #2f9e5e; }
    .match-card.graph { border-left-color: #4a7fd1; }
    .match-card.weak { border-left-color: #d1d5db; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 8px;
    }
    .badge.direct { background: #e3f6ea; color: #1e7d47; }
    .badge.graph { background: #e8eefc; color: #2f5bb7; }
    .badge.weak { background: #f0f1f3; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Oncology Navigator")
st.caption(
    "Enter a gene to find currently recruiting clinical trials that may be "
    "relevant, ranked using live ClinVar and clinicaltrials.gov data."
)

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
            except Exception as e:
                st.error(f"Couldn't reach ClinVar: {e}")
                variants = []

        if not variants:
            st.warning(f"No pathogenic variant records found for {gene} in ClinVar.")
        else:
            st.success(f"Found {len(variants)} variant record(s) for {gene}.")
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
                except Exception as e:
                    st.error(f"Couldn't reach clinicaltrials.gov: {e}")
                    trials = []

            if not trials:
                st.warning("No currently recruiting trials found for that search.")
            else:
                st.success(f"Found {len(trials)} recruiting trial(s).")

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
                m1.metric("Variants found", len(variants))
                m2.metric("Trials ranked", len(trials))
                m3.metric("Direct text matches", match_types.count("direct"))
                m4.metric("Graph-connected matches", match_types.count("graph"))

                st.subheader(f"Ranked trials for {gene}")

                badge_label = {
                    "direct": "🎯 Direct text match",
                    "graph": "🕸️ Graph-connected match",
                    "weak": "◌ Weak connection",
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
