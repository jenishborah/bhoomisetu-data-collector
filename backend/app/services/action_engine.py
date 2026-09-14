"""Rule-based, measurable next-best actions for the prototype."""

from __future__ import annotations

from typing import Any


def _number(snapshot: dict[str, Any], field: str, default: float = 0) -> float:
    try:
        return float(snapshot.get(field, default) or default)
    except (TypeError, ValueError):
        return default


def _action(
    action: str,
    category: str,
    owner: str,
    priority: str,
    trigger: str,
    reason: str,
) -> dict[str, str]:
    return {
        "action": action,
        "category": category,
        "owner": owner,
        "priority": priority,
        "trigger": trigger,
        "reason": reason,
        "status": "Open",
    }


def get_recommended_actions(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    """Produce transparent recommendations based on present measurable fields.

    Rules are advisory workflow prompts. They do not imply an intervention will
    guarantee a particular risk reduction or project outcome.
    """
    recommendations: list[dict[str, str]] = []
    approvals = _number(snapshot, "pending_approvals_count")
    file_days = _number(snapshot, "file_pending_days")
    docs = _number(snapshot, "docs_complete_pct", 100)
    disbursal = _number(snapshot, "disbursal_pct", 100)
    awarded = _number(snapshot, "compensation_awarded_cr")
    cases = _number(snapshot, "court_cases_count")
    rnr = _number(snapshot, "rnr_progress_pct", 100)
    grievances = _number(snapshot, "grievances_30d")
    redressal_days = _number(snapshot, "grievance_redressal_days")
    days_in_stage = _number(snapshot, "days_in_current_stage")

    if approvals >= 4:
        recommendations.append(_action(
            "Review pending approvals and assign a responsible department.",
            "Administrative", "Project Authority", "HIGH", f"pending_approvals_count = {approvals:g}",
            "Multiple approvals remain pending in the latest project snapshot.",
        ))
    if file_days >= 60:
        recommendations.append(_action(
            "Escalate the long-pending administrative file.", "Administrative", "Project Authority", "HIGH",
            f"file_pending_days = {file_days:g}",
            "The administrative file has remained pending for an extended observed period.",
        ))
    if docs < 80:
        recommendations.append(_action(
            "Complete missing acquisition documentation.", "Administrative", "Project Authority", "HIGH" if docs < 60 else "MEDIUM",
            f"docs_complete_pct = {docs:g}%", "Documentation completeness is below the prototype monitoring threshold.",
        ))
    if awarded > 0 and disbursal < 70:
        recommendations.append(_action(
            "Prioritize compensation disbursement for eligible cases.", "Compensation", "Compensation Cell", "HIGH" if disbursal < 40 else "MEDIUM",
            f"disbursal_pct = {disbursal:g}%", "A material share of awarded compensation remains undisbursed.",
        ))
    if cases >= 2:
        recommendations.append(_action(
            "Review pending legal/title disputes with the legal cell.", "Legal", "Legal Cell", "HIGH" if cases >= 5 else "MEDIUM",
            f"court_cases_count = {cases:g}", "Open legal disputes are recorded in the latest project snapshot.",
        ))
    if rnr < 60:
        recommendations.append(_action(
            "Prioritize rehabilitation and resettlement actions.", "Rehabilitation & Resettlement", "R&R Cell", "HIGH" if rnr < 35 else "MEDIUM",
            f"rnr_progress_pct = {rnr:g}%", "R&R implementation progress is below the prototype monitoring threshold.",
        ))
    if grievances >= 5 or redressal_days >= 30:
        trigger = f"grievances_30d = {grievances:g}; grievance_redressal_days = {redressal_days:g}"
        recommendations.append(_action(
            "Review unresolved grievances and strengthen grievance redressal.", "Community / Grievance", "Grievance Redressal Cell",
            "HIGH" if grievances >= 10 or redressal_days >= 60 else "MEDIUM", trigger,
            "The latest snapshot indicates elevated grievance volume or extended resolution time.",
        ))
    if days_in_stage >= 180:
        recommendations.append(_action(
            "Conduct a stage-review meeting and record the next transition dependency.", "Administrative", "Project Authority", "MEDIUM",
            f"days_in_current_stage = {days_in_stage:g}", "The project has an extended observed duration in its current stage.",
        ))

    if not recommendations:
        recommendations.append(_action(
            "Maintain routine milestone monitoring and update the project record.", "Administrative", "Project Authority", "LOW",
            "No rule threshold exceeded", "No current deterministic escalation threshold was triggered by the latest snapshot.",
        ))

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(recommendations, key=lambda item: (order[item["priority"]], item["category"], item["action"]))
