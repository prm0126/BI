"""Healthcare-operations recommendations.

Generates natural-language recommendations from a forecast + explanation.

The default engine is rule-based so it works offline. If `anthropic` is
installed and an API key is configured, `recommend_with_llm` can produce
richer narrative recommendations using Claude.
"""

from typing import List, Optional
import os
import numpy as np

from .types import ForecastResult, Recommendation


HEALTHCARE_PRESETS = {
    "admissions": {
        "unit": "patients/day",
        "high_action": "Pre-allocate beds, alert on-call physicians, and pre-stage triage staff.",
        "low_action": "Reduce overtime scheduling and consider elective procedure pull-forward.",
        "volatility_action": "Hold a 10-15% buffer staffing margin and run hourly capacity checks.",
    },
    "occupancy": {
        "unit": "%",
        "high_action": "Trigger discharge-coordination huddle and open swing beds.",
        "low_action": "Right-size nurse-to-bed ratios; reassign float pool to high-demand units.",
        "volatility_action": "Stabilize length-of-stay variance; review discharge bottlenecks.",
    },
    "wait_time": {
        "unit": "minutes",
        "high_action": "Activate fast-track lane and surge staffing protocol.",
        "low_action": "Maintain current scheduling; consider catch-up on backlogged appointments.",
        "volatility_action": "Audit triage variance and same-day cancellation rates.",
    },
    "staffing_gap": {
        "unit": "FTE",
        "high_action": "Open agency requisitions and escalate to nursing supervisor on-call.",
        "low_action": "Pause new agency requisitions; redirect float to training and audits.",
        "volatility_action": "Tighten 14-day rolling roster review and PTO approval thresholds.",
    },
    "revenue": {
        "unit": "$",
        "high_action": "Confirm billing cycle capacity and AR follow-up cadence.",
        "low_action": "Investigate denial patterns and outstanding claims; review payer mix.",
        "volatility_action": "Audit claim-submission lag and denial cohorts week over week.",
    },
    "no_show": {
        "unit": "%",
        "high_action": "Trigger reminder cascade (SMS + call) and overbook by forecast delta.",
        "low_action": "Maintain reminder protocol; reduce overbooking.",
        "volatility_action": "Segment no-show by payer and visit type; refine reminder timing.",
    },
}


def _detect_preset(target: str) -> Optional[str]:
    t = target.lower()
    if any(k in t for k in ["admit", "admission", "ed_visit", "arrival"]):
        return "admissions"
    if "occup" in t or "bed" in t:
        return "occupancy"
    if "wait" in t or "los_wait" in t:
        return "wait_time"
    if "staff" in t or "fte" in t or "nurse" in t:
        return "staffing_gap"
    if "rev" in t or "revenue" in t or "billing" in t or "collections" in t:
        return "revenue"
    if "no_show" in t or "noshow" in t or "cancel" in t:
        return "no_show"
    return None


def recommend_actions(
    result: ForecastResult,
    explanation: dict,
    threshold_pct: float = 10.0,
) -> List[Recommendation]:
    preset_key = _detect_preset(result.target) or ""
    preset = HEALTHCARE_PRESETS.get(preset_key, {})

    pct = explanation.get("expected_change_pct", 0.0)
    interval = explanation.get("interval_width", 0.0)
    hist_mean = float(np.mean(result.history_values)) if result.history_values else 0.0
    rmse = explanation.get("backtest", {}).get("rmse")
    mape = explanation.get("backtest", {}).get("mape_pct")

    recs: List[Recommendation] = []

    # Directional recommendation.
    direction = "rise" if pct > 0 else "fall"
    severity = "act" if abs(pct) >= threshold_pct else "watch" if abs(pct) >= threshold_pct / 2 else "info"
    action = (
        preset.get("high_action") if pct > 0 else preset.get("low_action")
    ) or "Review staffing, scheduling, and inventory plans against the forecast."

    recs.append(
        Recommendation(
            headline=f"{result.target} expected to {direction} ~{abs(pct):.1f}% over next {result.horizon} steps",
            severity=severity,
            rationale=(
                f"Model {result.model} backtest RMSE={rmse:.2f} "
                f"(MAPE={mape:.1f}%) over a held-out window. "
                f"Forecast mean shifts by {explanation.get('expected_change_abs', 0):.2f} vs. recent history "
                f"(mean={hist_mean:.2f})."
                if rmse is not None and not _isnan(rmse)
                else f"Forecast mean shifts by {explanation.get('expected_change_abs', 0):.2f} vs. recent history."
            ),
            action=action,
        )
    )

    # Volatility / uncertainty recommendation.
    if hist_mean and interval / max(abs(hist_mean), 1e-9) > 0.4:
        recs.append(
            Recommendation(
                headline="Forecast uncertainty is high",
                severity="watch",
                rationale=(
                    f"Average prediction-interval width is {interval:.2f}, "
                    f"~{interval / hist_mean * 100:.0f}% of historical mean."
                ),
                action=preset.get("volatility_action")
                or "Pad capacity plans with a contingency buffer until variance narrows.",
            )
        )

    # Calendar driver recommendation.
    drivers = explanation.get("drivers", {}).get("calendar_drivers", {})
    if drivers:
        top, val = max(drivers.items(), key=lambda kv: abs(kv[1]))
        if top == "weekday_effect" and val > 0.05:
            recs.append(
                Recommendation(
                    headline="Strong weekday pattern detected",
                    severity="info",
                    rationale=f"Weekday explains a notable share of variance (score={val:.2f}).",
                    action="Align rosters and clinic blocks to peak weekdays; flatten low-demand days with telehealth slots.",
                )
            )
        elif top == "month_effect" and val > 0.05:
            recs.append(
                Recommendation(
                    headline="Seasonal monthly pattern detected",
                    severity="info",
                    rationale=f"Monthly seasonality drives a notable share of variance (score={val:.2f}).",
                    action="Plan seasonal hiring and supply contracts ahead of recurring peaks.",
                )
            )

    # Data-quality recommendation.
    feats = result.metadata.get("series_features", {})
    if feats.get("missing_ratio", 0) > 0.1:
        recs.append(
            Recommendation(
                headline="Input data has gaps",
                severity="watch",
                rationale=f"{feats['missing_ratio'] * 100:.0f}% of input rows are missing.",
                action="Tighten EHR feed reliability and backfill gaps before re-running the model.",
            )
        )
    if feats.get("n_obs", 0) < 60:
        recs.append(
            Recommendation(
                headline="Short history reduces forecast confidence",
                severity="info",
                rationale=f"Only {feats.get('n_obs', 0)} observations available.",
                action="Extend history window and re-train once you have ≥3 months of daily data.",
            )
        )

    # Model-improvement recommendation.
    if mape is not None and not _isnan(mape) and mape > 20:
        recs.append(
            Recommendation(
                headline="Backtest accuracy is weak",
                severity="watch",
                rationale=f"Holdout MAPE={mape:.1f}% indicates the chosen model is struggling.",
                action=(
                    "Add exogenous drivers (clinic hours, public holidays, marketing pushes, "
                    "weather) and rerun the router so it considers richer feature sets."
                ),
            )
        )

    return recs


def recommend_with_llm(
    result: ForecastResult,
    explanation: dict,
    recs: List[Recommendation],
    context: str = "",
) -> Optional[str]:
    """Optional richer narrative via Claude. Returns None if SDK or key missing."""
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    client = anthropic.Anthropic(api_key=key)
    prompt = (
        "You are advising a hospital operations team. Given the forecast summary, "
        "data quality, and rule-based recommendations below, write a 4-6 sentence "
        "executive briefing that explains what to do this week and how to improve "
        "the model going forward.\n\n"
        f"Target: {result.target}\nModel: {result.model}\nHorizon: {result.horizon}\n"
        f"Explanation: {explanation}\nRule-based recs: {[r.to_dict() for r in recs]}\n"
        f"Context: {context}\n"
    )
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if hasattr(block, "text"))


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False
