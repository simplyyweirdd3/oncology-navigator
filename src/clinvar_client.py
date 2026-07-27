"""
ClinVar client for Oncology Navigator.

Stage 1 of the pipeline: takes a gene symbol (and optionally a specific
variant/mutation string) and returns structured pathogenicity data pulled
live from NCBI's ClinVar via E-utilities.

No API key required for light use (3 requests/second). If you register for
an NCBI API key later, pass it in to bump that to 10 req/sec, useful once
you're batch-processing many variants for the graph stage.

Docs: https://www.ncbi.nlm.nih.gov/clinvar/docs/programmatic_access/
"""

import time
import requests
from dataclasses import dataclass, field
from typing import Optional

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Cancer-related trait keywords used to flag whether a variant is
# oncologically relevant. This list will grow as we plug in COSMIC.
ONCOLOGY_KEYWORDS = [
    "cancer", "carcinoma", "tumor", "tumour", "neoplasm", "melanoma",
    "leukemia", "lymphoma", "sarcoma", "malignant", "oncogen",
]


@dataclass
class VariantRecord:
    """A single structured, cleaned-up variant record ready to feed
    into the trial-matching stage."""
    uid: str
    gene: str
    variation_name: str
    protein_change: Optional[str]
    classification: Optional[str]        # e.g. "Pathogenic", "Benign"
    review_status: Optional[str]
    associated_traits: list = field(default_factory=list)
    is_oncology_relevant: bool = False

    def to_dict(self):
        return {
            "uid": self.uid,
            "gene": self.gene,
            "variation_name": self.variation_name,
            "protein_change": self.protein_change,
            "classification": self.classification,
            "review_status": self.review_status,
            "associated_traits": self.associated_traits,
            "is_oncology_relevant": self.is_oncology_relevant,
        }


def _is_oncology_relevant(traits: list) -> bool:
    joined = " ".join(traits).lower()
    return any(keyword in joined for keyword in ONCOLOGY_KEYWORDS)


def search_variants(gene: str, classification: str = "pathogenic",
                     retmax: int = 20, api_key: Optional[str] = None) -> list:
    """
    Step 1 of the E-utilities dance: find ClinVar record IDs for a gene.

    Example: search_variants("BRCA1") -> list of uids like ["4856951", ...]
    """
    params = {
        "db": "clinvar",
        "term": f"{gene}[gene] AND {classification}[CLNSIG]",
        "retmode": "json",
        "retmax": retmax,
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_variant_summary(uid: str, api_key: Optional[str] = None) -> Optional[VariantRecord]:
    """
    Step 2: given a ClinVar uid, pull the full structured summary and
    parse it down into a clean VariantRecord.
    """
    params = {"db": "clinvar", "id": uid, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(ESUMMARY_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", {})
    real_uid = None
    for key in result.get("uids", []):
        real_uid = key
        break
    if not real_uid or real_uid not in result:
        return None

    record = result[real_uid]

    genes = record.get("genes", [])
    gene_symbol = genes[0]["symbol"] if genes else "UNKNOWN"

    germline = record.get("germline_classification", {})
    classification = germline.get("description") or None
    review_status = germline.get("review_status") or None
    traits = [t.get("trait_name", "") for t in germline.get("trait_set", [])]

    return VariantRecord(
        uid=real_uid,
        gene=gene_symbol,
        variation_name=record.get("title", ""),
        protein_change=record.get("protein_change") or None,
        classification=classification,
        review_status=review_status,
        associated_traits=traits,
        is_oncology_relevant=_is_oncology_relevant(traits),
    )


def get_variants_for_gene(gene: str, classification: str = "pathogenic",
                           limit: int = 10, api_key: Optional[str] = None,
                           rate_limit_delay: float = 0.34) -> list:
    """
    Convenience wrapper: search + fetch in one call, returns a list of
    VariantRecord objects ready for the frontend or the graph stage.

    rate_limit_delay defaults to ~3 requests/second (NCBI's no-key limit).
    """
    uids = search_variants(gene, classification, retmax=limit, api_key=api_key)
    records = []
    for uid in uids:
        record = fetch_variant_summary(uid, api_key=api_key)
        if record:
            records.append(record)
        time.sleep(rate_limit_delay)
    return records


if __name__ == "__main__":
    # Quick manual test: pull 3 pathogenic BRCA1 variants live.
    print("Searching ClinVar for pathogenic BRCA1 variants...")
    results = get_variants_for_gene("BRCA1", limit=3)
    for r in results:
        print(r.to_dict())
