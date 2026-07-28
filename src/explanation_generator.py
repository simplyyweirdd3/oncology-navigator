"""
Explanation generator for Oncology Navigator.

Takes a variant, a ranked trial, and its similarity score, and produces a
plain-language "why this trial matches" explanation.

Two modes:
  - Template mode (default, instant, no model download): builds a clear,
    honest explanation from the structured data we already have (gene,
    classification, trial conditions, interventions, whether the match
    came from direct text or the graph's drug connection).
  - LLM mode (Qwen2.5-1.5B-Instruct via transformers): rewrites that same
    structured information into more natural, conversational language.
    Requires downloading the model on first run (~3GB), which only
    happens once and needs a real internet connection, so this only runs
    properly outside this sandbox.

Template mode exists so the app always has something to show immediately,
and LLM mode is a genuine upgrade layered on top, not a replacement the
whole system depends on.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchExplanation:
    trial_nct_id: str
    headline: str          # one-line summary
    reasoning: str          # fuller paragraph
    match_type: str          # "direct" or "graph"


def build_template_explanation(gene: str, classification: Optional[str],
                                 trial_title: str, interventions: list,
                                 conditions: list, mentions_gene: bool,
                                 similarity_score: float) -> MatchExplanation:
    """
    The always-available, instant explanation. No model required.
    """
    drug_list = ", ".join(interventions[:3]) if interventions else "an unspecified treatment"
    condition_list = ", ".join(conditions[:2]) if conditions else "this condition"

    if mentions_gene:
        match_type = "direct"
        headline = f"This trial explicitly requires a {gene} mutation."
        reasoning = (
            f"The trial's own eligibility criteria mention {gene} directly, "
            f"which is the clearest kind of match. It's currently recruiting "
            f"patients with {condition_list} and is testing {drug_list}. "
            f"Because your variant is classified as {classification or 'clinically relevant'} "
            f"in ClinVar, this is a strong candidate to discuss with a treating physician."
        )
    else:
        match_type = "graph"
        headline = f"This trial doesn't mention {gene} by name, but its drug does."
        reasoning = (
            f"This trial's eligibility text never says '{gene}' outright, but it's "
            f"testing {drug_list}, a drug connected to {gene} in our knowledge base. "
            f"That's the kind of connection a plain keyword search would miss entirely. "
            f"It's recruiting for {condition_list}. Worth flagging to a physician even "
            f"though the trial text alone wouldn't have surfaced it."
        )

    return MatchExplanation(
        trial_nct_id="",  # filled in by caller
        headline=headline,
        reasoning=reasoning,
        match_type=match_type,
    )


def build_llm_prompt(gene: str, classification: Optional[str], trial_title: str,
                       eligibility_snippet: str, interventions: list,
                       mentions_gene: bool) -> str:
    """
    Builds the prompt for the real LLM pass (Qwen2.5-1.5B-Instruct).
    Kept separate from the actual model call so the prompt can be tested
    and refined independently of having the model downloaded.
    """
    connection_note = (
        f"The trial's eligibility text explicitly mentions {gene}."
        if mentions_gene else
        f"The trial's eligibility text does NOT mention {gene} by name, but it "
        f"tests a drug known to target {gene} mutations, based on a curated "
        f"drug-variant knowledge base."
    )

    return f"""You are explaining a clinical trial match to a patient in plain, warm, honest language. Do not invent facts not given below.

Patient's variant: {gene} ({classification or "clinical significance not specified"})
Trial: {trial_title}
Trial treats with: {", ".join(interventions[:3])}
Trial eligibility text (excerpt): {eligibility_snippet[:300]}
{connection_note}

Write a 2-3 sentence explanation of why this trial might be relevant to someone with this variant, in plain language a non-medical person could follow. Do not use the words "please" or "I hope this helps." Be direct and factual."""


def generate_llm_explanation(prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    """
    Real LLM call. Requires `transformers`, `torch`, and a working internet
    connection to download the model on first run. Lazy-imports so template
    mode never pays this cost if the LLM isn't being used.
    """
    from transformers import pipeline

    generator = pipeline("text-generation", model=model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    result = generator(messages, max_new_tokens=150, do_sample=True, temperature=0.7)
    return result[0]["generated_text"][-1]["content"].strip()


if __name__ == "__main__":
    # Smoke test: template mode only, instant, no model download needed.
    explanation = build_template_explanation(
        gene="BRAF",
        classification="Pathogenic",
        trial_title="A Phase I/II Trial of Vemurafenib and Metformin to Melanoma Patients",
        interventions=["Vemurafenib", "Metformin"],
        conditions=["Melanoma"],
        mentions_gene=True,
        similarity_score=0.0569,
    )
    print("HEADLINE:", explanation.headline)
    print("REASONING:", explanation.reasoning)
    print("MATCH TYPE:", explanation.match_type)

    print("\n--- Example LLM prompt (not run, just showing the format) ---\n")
    prompt = build_llm_prompt(
        gene="BRAF",
        classification="Pathogenic",
        trial_title="A Phase I/II Trial of Vemurafenib and Metformin to Melanoma Patients",
        eligibility_snippet="Histologic documentation of metastatic melanoma with BRAFV600 mutation.",
        interventions=["Vemurafenib", "Metformin"],
        mentions_gene=True,
    )
    print(prompt)
