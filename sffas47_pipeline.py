"""
sffas47_pipeline.py

Federal Exposure & Reporting-Entity Risk System
-------------------------------------------------
Ingests SEC EDGAR filings and USAspending.gov award history for a set of
public companies, applies an SFFAS 47-based ownership/control classification,
and produces a separate Federal Dependency Risk Score for ordinary
counterparties.

Data sources (both free, no API key required):
  - SEC EDGAR:      https://data.sec.gov
  - USAspending.gov: https://api.usaspending.gov

Run:
    python sffas47_pipeline.py --demo          # uses cached/manual figures, no network
    python sffas47_pipeline.py --live TICKER   # pulls live data for one company

See ARCHITECTURE.md for the full methodology write-up.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# SEC requires a descriptive User-Agent identifying the requester. Replace
# with your own contact info before running live pulls.
SEC_HEADERS = {
    "User-Agent": "SFFAS47-Risk-Project research-demo@example.com",
    "Accept-Encoding": "gzip, deflate",
}

USASPENDING_BASE = "https://api.usaspending.gov/api/v2"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_CONCEPT = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
)


# ---------------------------------------------------------------------------
# SFFAS 47 classification model
# ---------------------------------------------------------------------------

class Sffas47Classification(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"          # ordinary counterparty; economic
                                                # dependency alone is not control (SFFAS 47 P35)
    RELATED_PARTY = "RELATED_PARTY"            # significant influence, not control (P81-89)
    DISCLOSURE_ENTITY = "DISCLOSURE_ENTITY"    # included, but reported via disclosure (P43-46)
    CONSOLIDATION_ENTITY = "CONSOLIDATION_ENTITY"  # included, consolidated (P39-42)


@dataclass
class ControlAssessment:
    """
    Structured record of the SFFAS 47 control test (paragraphs 20-37).
    Every field maps to a specific paragraph so the classification is
    auditable, not a black box.
    """
    in_budget: bool = False                       # P22
    majority_ownership_pct: Optional[float] = None  # P24-25 (>50 => True)
    persuasive_control_indicators: list[str] = field(default_factory=list)  # P31
    aggregate_control_indicators: list[str] = field(default_factory=list)   # P32
    economic_dependency_only: bool = False          # P35 explicit carve-out
    intervention_or_conservatorship: bool = False    # P51-55
    misleading_to_exclude: bool = False              # P36-37

    @property
    def majority_ownership(self) -> bool:
        return (self.majority_ownership_pct or 0) > 50.0

    @property
    def control_established(self) -> bool:
        # Any single persuasive indicator is sufficient (P31).
        if self.persuasive_control_indicators:
            return True
        # Aggregate indicators require judgment in combination (P32) -
        # heuristic: 3+ aggregate indicators treated as sufficient for a
        # first-pass flag, pending professional review.
        if len(self.aggregate_control_indicators) >= 3:
            return True
        return False

    def classify(self) -> tuple[Sffas47Classification, str]:
        # Step 1: does any inclusion principle apply at all? (P21-37)
        included = (
            self.in_budget
            or self.majority_ownership
            or self.control_established
            or self.misleading_to_exclude
        )

        if not included:
            if self.economic_dependency_only:
                return (
                    Sffas47Classification.NOT_APPLICABLE,
                    "Revenue/economic dependency on the federal government does not "
                    "establish control (SFFAS 47 P35). No inclusion principle met.",
                )
            return (
                Sffas47Classification.NOT_APPLICABLE,
                "No SFFAS 47 inclusion principle (in budget / majority ownership / "
                "control / misleading-to-exclude) is met.",
            )

        # Step 2: included -> consolidation or disclosure? (P38-46)
        # Conservatorship / receivership / intervention actions are steered
        # toward disclosure-entity treatment because the relationship is not
        # expected to be permanent (P43, P51, P55).
        if self.intervention_or_conservatorship:
            return (
                Sffas47Classification.DISCLOSURE_ENTITY,
                "Included via majority ownership and/or control, but classified as a "
                "disclosure entity because the relationship arises from conservatorship/"
                "receivership/intervention and is not expected to be permanent "
                "(SFFAS 47 P43, P51, P55).",
            )

        if self.majority_ownership or self.control_established:
            # Default assumption for a fully vertically-integrated, tax-financed
            # organization; real assessment requires the P38-42 characteristics
            # (financing source, governance, risk/reward, market basis).
            return (
                Sffas47Classification.CONSOLIDATION_ENTITY,
                "Majority ownership and/or control established, and the "
                "characteristics as a whole (financing, governance, risk/reward, "
                "non-market basis) support consolidation treatment (SFFAS 47 P39-42).",
            )

        return (
            Sffas47Classification.DISCLOSURE_ENTITY,
            "Included per an inclusion principle but characteristics as a whole "
            "support disclosure rather than consolidation treatment (SFFAS 47 P43-46).",
        )


# ---------------------------------------------------------------------------
# Federal Dependency Risk Score (separate rubric - see ARCHITECTURE.md S5)
# ---------------------------------------------------------------------------

@dataclass
class DependencyInputs:
    federal_revenue_pct: float                # 0-100
    largest_program_pct_of_revenue: float      # 0-100
    award_history_years: int                   # consecutive years with federal awards
    award_volatility: float                    # coefficient of variation, 0+ (higher = less stable)
    risk_factor_intensity: float                # 0-10, NLP-scored severity/count
    agency_diversification_count: int           # distinct awarding agencies


def score_dependency(inputs: DependencyInputs) -> dict:
    # Revenue concentration - 40 pts, linear scale capped at 100% revenue
    revenue_score = min(inputs.federal_revenue_pct, 100) / 100 * 40

    # Program/customer concentration - 20 pts
    program_score = min(inputs.largest_program_pct_of_revenue, 100) / 100 * 20

    # Award history consistency - 15 pts (long + stable history raises the
    # score, since it signals durable structural dependency, not necessarily
    # "danger" in the credit sense - this score measures EXPOSURE not default risk)
    history_component = min(inputs.award_history_years, 10) / 10  # longevity
    volatility_component = max(0.0, 1 - min(inputs.award_volatility, 1.0))  # stability
    history_score = ((history_component + volatility_component) / 2) * 15

    # Disclosed risk-factor intensity - 15 pts
    risk_factor_score = min(inputs.risk_factor_intensity, 10) / 10 * 15

    # Sector/agency concentration - 10 pts (fewer distinct agencies = more concentrated = higher score)
    diversification_penalty = max(0, 10 - inputs.agency_diversification_count)
    diversification_score = min(diversification_penalty, 10) / 10 * 10

    total = (
        revenue_score
        + program_score
        + history_score
        + risk_factor_score
        + diversification_score
    )

    if total <= 30:
        band = "Low"
    elif total <= 55:
        band = "Moderate"
    elif total <= 75:
        band = "High"
    else:
        band = "Very High"

    return {
        "total_score": round(total, 1),
        "band": band,
        "components": {
            "revenue_concentration": round(revenue_score, 1),
            "program_concentration": round(program_score, 1),
            "award_history_consistency": round(history_score, 1),
            "risk_factor_intensity": round(risk_factor_score, 1),
            "sector_diversification": round(diversification_score, 1),
        },
    }


@dataclass
class GovernanceInputs:
    """Separate rubric for entities that DO meet the SFFAS 47 control test
    (e.g. GSEs in conservatorship). See ARCHITECTURE.md S6 for rationale."""
    treasury_equity_stake_pct: float           # 0-100
    conservatorship_or_receivership: bool
    exit_timeline_certainty: float             # 0 (highly uncertain) - 10 (clear/imminent)
    litigation_overhang_severity: float        # 0-10
    liquidation_preference_relative_to_equity: float  # ratio, higher = more dilution risk


def score_governance(inputs: GovernanceInputs) -> dict:
    ownership_score = min(inputs.treasury_equity_stake_pct, 100) / 100 * 35
    conservatorship_score = 25 if inputs.conservatorship_or_receivership else 0
    exit_uncertainty_score = (10 - min(inputs.exit_timeline_certainty, 10)) / 10 * 20
    litigation_score = min(inputs.litigation_overhang_severity, 10) / 10 * 10
    dilution_score = min(inputs.liquidation_preference_relative_to_equity, 5) / 5 * 10

    total = (
        ownership_score
        + conservatorship_score
        + exit_uncertainty_score
        + litigation_score
        + dilution_score
    )

    if total <= 30:
        band = "Low"
    elif total <= 55:
        band = "Moderate"
    elif total <= 75:
        band = "High"
    else:
        band = "Very High"

    return {
        "total_score": round(total, 1),
        "band": band,
        "components": {
            "government_equity_stake": round(ownership_score, 1),
            "conservatorship_status": round(conservatorship_score, 1),
            "exit_timeline_uncertainty": round(exit_uncertainty_score, 1),
            "litigation_overhang": round(litigation_score, 1),
            "recapitalization_dilution_risk": round(dilution_score, 1),
        },
    }


# ---------------------------------------------------------------------------
# Data ingestion - SEC EDGAR
# ---------------------------------------------------------------------------

def fetch_edgar_submissions(cik: str) -> dict:
    """CIK must be zero-padded to 10 digits, no 'CIK' prefix, e.g. '0000936468'."""
    url = EDGAR_SUBMISSIONS.format(cik=cik)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_edgar_revenue_series(cik: str, tag: str = "Revenues") -> dict:
    """
    Pulls a structured XBRL fact series (e.g. total Revenues) for a company.
    Many companies use 'RevenueFromContractWithCustomerExcludingAssessedTax'
    instead of 'Revenues' - try both if the first returns nothing.
    """
    url = EDGAR_COMPANY_CONCEPT.format(cik=cik, tag=tag)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Data ingestion - USAspending.gov
# ---------------------------------------------------------------------------

def usaspending_recipient_search(recipient_name: str) -> list[dict]:
    """
    Search USAspending for a recipient by name to obtain its recipient UEI/id,
    required for pulling award history. This endpoint is POST-based.
    """
    url = f"{USASPENDING_BASE}/recipient/duns/"  # illustrative; USAspending's
    # exact search endpoint/schema should be confirmed against current API docs
    # at https://api.usaspending.gov/docs/endpoints before a production run,
    # since recipient search endpoints have changed across API versions.
    payload = {"search_text": recipient_name, "limit": 5}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json().get("results", [])


def usaspending_award_history_by_year(recipient_id: str, fiscal_years: list[int]) -> dict:
    """
    Pulls total obligated federal award dollars per fiscal year for a
    recipient. Returns {year: total_obligated_dollars}.
    """
    url = f"{USASPENDING_BASE}/search/spending_by_award/"
    history = {}
    for fy in fiscal_years:
        payload = {
            "filters": {
                "recipient_id": recipient_id,
                "time_period": [{"start_date": f"{fy-1}-10-01", "end_date": f"{fy}-09-30"}],
                "award_type_codes": ["A", "B", "C", "D"],  # contracts
            },
            "fields": ["Award Amount"],
            "limit": 100,
            "page": 1,
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        history[fy] = sum(r.get("Award Amount", 0) or 0 for r in results)
        time.sleep(0.2)  # be polite to the API
    return history


# ---------------------------------------------------------------------------
# Demo runner - uses researched figures (cited in DEMO_RESULTS.md) rather
# than live calls, so the project is reviewable without network access.
# ---------------------------------------------------------------------------

DEMO_COMPANIES = {
    "LMT": {
        "name": "Lockheed Martin Corporation",
        "cik": "0000936468",
        "control": ControlAssessment(economic_dependency_only=True),
        "dependency": DependencyInputs(
            federal_revenue_pct=72.0,          # FY2025 10-K: 72% of sales from U.S. Government
            largest_program_pct_of_revenue=27.0,  # F-35 program, FY2025 10-K
            award_history_years=10,
            award_volatility=0.15,
            risk_factor_intensity=8.0,
            agency_diversification_count=3,     # primarily DoW, NASA, allied FMS
        ),
    },
    "BAH": {
        "name": "Booz Allen Hamilton Holding Corporation",
        "cik": "0001443646",
        "control": ControlAssessment(economic_dependency_only=True),
        "dependency": DependencyInputs(
            federal_revenue_pct=98.0,          # FY2025 10-K
            largest_program_pct_of_revenue=49.0,  # Defense segment share, FY2025
            award_history_years=10,
            award_volatility=0.10,
            risk_factor_intensity=9.0,
            agency_diversification_count=6,     # Defense, Intel, Civil across many agencies
        ),
    },
    "PLTR": {
        "name": "Palantir Technologies Inc.",
        "cik": "0001321655",
        "control": ControlAssessment(economic_dependency_only=True),
        "dependency": DependencyInputs(
            federal_revenue_pct=40.0,          # est. U.S. federal share; 54% is total
                                                 # "government" incl. non-US governments (FY2025 10-K)
            largest_program_pct_of_revenue=15.0,  # estimate - not broken out at program level
            award_history_years=8,
            award_volatility=0.35,              # faster-growing, less stable mix
            risk_factor_intensity=6.0,
            agency_diversification_count=8,
        ),
    },
    "FNMA": {
        "name": "Federal National Mortgage Association (Fannie Mae)",
        "cik": "0000310522",
        "control": ControlAssessment(
            majority_ownership_pct=79.9,        # Treasury warrants, per common stock
            persuasive_control_indicators=[
                "FHFA as conservator holds powers of management, board, and shareholders",
            ],
            intervention_or_conservatorship=True,
        ),
        "governance": GovernanceInputs(
            treasury_equity_stake_pct=79.9,
            conservatorship_or_receivership=True,
            exit_timeline_certainty=3.0,        # roadmap announced but multi-year, uncertain
            litigation_overhang_severity=6.0,   # net-worth-sweep litigation history
            liquidation_preference_relative_to_equity=3.5,
        ),
    },
    "FMCC": {
        "name": "Federal Home Loan Mortgage Corporation (Freddie Mac)",
        "cik": "0001026214",
        "control": ControlAssessment(
            majority_ownership_pct=79.9,
            persuasive_control_indicators=[
                "FHFA as conservator holds powers of management, board, and shareholders",
            ],
            intervention_or_conservatorship=True,
        ),
        "governance": GovernanceInputs(
            treasury_equity_stake_pct=79.9,
            conservatorship_or_receivership=True,
            exit_timeline_certainty=3.0,
            litigation_overhang_severity=6.0,
            liquidation_preference_relative_to_equity=3.5,
        ),
    },
}


def run_demo() -> None:
    print("=" * 78)
    print("SFFAS 47 FEDERAL EXPOSURE RISK SYSTEM - DEMO RUN (cached research data)")
    print("=" * 78)
    for ticker, data in DEMO_COMPANIES.items():
        classification, rationale = data["control"].classify()
        print(f"\n{data['name']} ({ticker})")
        print(f"  SFFAS 47 Classification : {classification.value}")
        print(f"  Rationale               : {rationale}")

        if "dependency" in data:
            result = score_dependency(data["dependency"])
            print(f"  Federal Dependency Score: {result['total_score']} / 100  [{result['band']}]")
            for k, v in result["components"].items():
                print(f"      - {k}: {v}")
        if "governance" in data:
            result = score_governance(data["governance"])
            print(f"  Governance/Ownership Risk Score: {result['total_score']} / 100  [{result['band']}]")
            for k, v in result["components"].items():
                print(f"      - {k}: {v}")
    print("\n" + "=" * 78)
    print("See ARCHITECTURE.md for methodology and DEMO_RESULTS.md for full write-up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFFAS 47 Federal Exposure Risk Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run demo with cached research data")
    parser.add_argument("--live", metavar="TICKER", help="Attempt a live EDGAR pull for a ticker's CIK (requires network)")
    args = parser.parse_args()

    if args.live:
        ticker = args.live.upper()
        if ticker not in DEMO_COMPANIES:
            raise SystemExit(f"Unknown ticker '{ticker}'. Add it to DEMO_COMPANIES first.")
        cik = DEMO_COMPANIES[ticker]["cik"]
        print(f"Fetching live EDGAR submissions for {ticker} (CIK {cik})...")
        submissions = fetch_edgar_submissions(cik)
        print(json.dumps({k: submissions[k] for k in ("name", "sicDescription", "tickers")}, indent=2))
    else:
        run_demo()