"""
ClinicalTrials.gov client for Oncology Navigator.

Pulls live, currently recruiting oncology trials filtered by condition
and/or biomarker/intervention, and extracts the eligibility criteria text
that Stage 3 (the LLM explanation layer) will later parse.

Docs: https://clinicaltrials.gov/data-api/api
"""

import requests
from dataclasses import dataclass, field
from typing import Optional

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


@dataclass
class TrialRecord:
    nct_id: str
    title: str
    status: str
    phases: list
    conditions: list
    interventions: list
    eligibility_criteria: str
    minimum_age: Optional[str]
    sex: Optional[str]
    locations: list = field(default_factory=list)

    def to_dict(self):
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "status": self.status,
            "phases": self.phases,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "eligibility_criteria": self.eligibility_criteria,
            "minimum_age": self.minimum_age,
            "sex": self.sex,
            "locations": self.locations[:5],  # cap for display
        }


def _parse_study(study: dict) -> TrialRecord:
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status_mod = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions_mod = protocol.get("conditionsModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    locations_mod = protocol.get("contactsLocationsModule", {})

    interventions = [i.get("name", "") for i in arms.get("interventions", [])]
    locations = [
        f"{loc.get('facility', '')}, {loc.get('city', '')}, {loc.get('country', '')}"
        for loc in locations_mod.get("locations", [])
    ]

    return TrialRecord(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        status=status_mod.get("overallStatus", ""),
        phases=design.get("phases", []),
        conditions=conditions_mod.get("conditions", []),
        interventions=interventions,
        eligibility_criteria=eligibility.get("eligibilityCriteria", ""),
        minimum_age=eligibility.get("minimumAge"),
        sex=eligibility.get("sex"),
        locations=locations,
    )


def search_trials(condition: str, intervention: Optional[str] = None,
                   recruiting_only: bool = True, page_size: int = 10) -> list:
    """
    Search live trials by condition (e.g. "breast cancer") and optionally
    a biomarker/drug intervention (e.g. "EGFR" or "trastuzumab").

    Returns a list of TrialRecord objects with full eligibility text intact,
    ready for Stage 3 to parse against a patient's variant profile.
    """
    params = {
        "query.cond": condition,
        "pageSize": page_size,
        "format": "json",
    }
    if intervention:
        params["query.intr"] = intervention
    if recruiting_only:
        params["filter.overallStatus"] = "RECRUITING"

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    studies = data.get("studies", [])
    return [_parse_study(s) for s in studies]


if __name__ == "__main__":
    print("Searching live recruiting breast cancer + EGFR trials...")
    trials = search_trials("breast cancer", intervention="EGFR", page_size=3)
    for t in trials:
        print(t.nct_id, "-", t.title, "-", t.status)
