"""bi-forecast CLI."""

import json
import sys
from pathlib import Path
import click
import pandas as pd

from .data import load_series
from .router import ModelRouter
from .explain import explain_forecast
from .recommend import recommend_actions, recommend_with_llm
from .whatif import what_if, WhatIfScenario


@click.group()
@click.version_option()
def cli():
    """Healthcare-ops forecasting engine: route → forecast → explain → recommend."""


@cli.command("forecast")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--date-col", default="ds", show_default=True)
@click.option("--target-col", default="y", show_default=True)
@click.option("--exog", "exog_cols", multiple=True, help="Exogenous column name (repeat).")
@click.option("--horizon", default=14, show_default=True, type=int, help="Forecast steps.")
@click.option("--allow", multiple=True, help="Limit router to these models.")
@click.option("--out", "out_path", type=click.Path(), default=None, help="Write JSON output here.")
@click.option("--csv-out", type=click.Path(), default=None, help="Write forecast CSV here.")
@click.option("--llm/--no-llm", default=False, help="Add a Claude narrative briefing (needs ANTHROPIC_API_KEY).")
def forecast_cmd(csv_path, date_col, target_col, exog_cols, horizon, allow, out_path, csv_out, llm):
    """Route to best base model, forecast, explain, and recommend improvements."""
    series, exog = load_series(csv_path, date_col=date_col, target_col=target_col, exog_cols=list(exog_cols) or None)

    router = ModelRouter(allow=list(allow) or None)
    result, decision, feats = router.forecast(series, horizon, target_name=target_col, exog=exog)
    explanation = explain_forecast(result, series=series)
    recs = recommend_actions(result, explanation)

    narrative = None
    if llm:
        narrative = recommend_with_llm(result, explanation, recs)

    _print_summary(result, decision, feats, explanation, recs, narrative)

    payload = {
        "forecast": json.loads(result.to_json()),
        "explanation": explanation,
        "decision": {
            "chosen": decision.chosen,
            "candidates": decision.candidates,
            "scores": decision.scores,
            "reason": decision.reason,
        },
        "recommendations": [r.to_dict() for r in recs],
        "narrative": narrative,
    }

    if out_path:
        Path(out_path).write_text(json.dumps(payload, indent=2, default=str))
        click.secho(f"\nJSON written to {out_path}", fg="green")
    if csv_out:
        result.to_frame().to_csv(csv_out, index=False)
        click.secho(f"Forecast CSV written to {csv_out}", fg="green")


@cli.command("route")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--date-col", default="ds", show_default=True)
@click.option("--target-col", default="y", show_default=True)
def route_cmd(csv_path, date_col, target_col):
    """Explain which base model the router would pick and why."""
    series, _ = load_series(csv_path, date_col=date_col, target_col=target_col)
    router = ModelRouter()
    decision, feats = router.route(series)
    click.echo(f"Chosen model:  {decision.chosen}")
    click.echo(f"Candidates:    {', '.join(decision.candidates)}")
    click.echo(f"Reason:        {decision.reason}")
    click.echo("Series features:")
    for k, v in feats.to_dict().items():
        click.echo(f"  - {k}: {v}")
    click.echo("Backtest scores:")
    for name, score in decision.scores.items():
        click.echo(f"  - {name}: rmse={score['rmse']:.3f}  mape={score['mape']:.2f}%")


@cli.command("whatif")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--date-col", default="ds", show_default=True)
@click.option("--target-col", default="y", show_default=True)
@click.option("--horizon", default=14, show_default=True, type=int)
@click.option(
    "--scenario",
    "scenarios",
    multiple=True,
    help="name=NAME,multiplier=1.1,delta=0,on_last_n=30 (repeatable).",
)
def whatif_cmd(csv_path, date_col, target_col, horizon, scenarios):
    """Run what-if scenarios: each is a multiplier/delta applied to history."""
    series, _ = load_series(csv_path, date_col=date_col, target_col=target_col)
    parsed = [_parse_scenario(s) for s in scenarios] or [
        WhatIfScenario(name="surge_10pct", multiplier=1.10, on_last_n=30),
        WhatIfScenario(name="drop_10pct", multiplier=0.90, on_last_n=30),
    ]

    results = what_if(series, horizon, parsed, target_name=target_col)
    base_mean = sum(results["baseline"].forecast_values) / horizon
    click.echo(f"Baseline forecast mean ({results['baseline'].model}): {base_mean:.2f}")
    for name, r in results.items():
        if name == "baseline":
            continue
        mean = sum(r.forecast_values) / horizon
        diff = mean - base_mean
        click.echo(
            f"  {name:20s}  mean={mean:.2f}  Δ={diff:+.2f} ({diff / base_mean * 100:+.1f}%)  model={r.model}"
        )


def _parse_scenario(spec: str) -> WhatIfScenario:
    parts = dict(p.split("=", 1) for p in spec.split(","))
    return WhatIfScenario(
        name=parts.get("name", "scenario"),
        multiplier=float(parts.get("multiplier", 1.0)),
        delta=float(parts.get("delta", 0.0)),
        on_last_n=int(parts["on_last_n"]) if "on_last_n" in parts else None,
    )


def _print_summary(result, decision, feats, explanation, recs, narrative):
    click.echo(click.style("\n=== Forecast ===", bold=True))
    click.echo(f"Target:   {result.target}")
    click.echo(f"Model:    {result.model}  (candidates: {', '.join(decision.candidates)})")
    click.echo(f"Horizon:  {result.horizon}")
    click.echo(
        f"Backtest: rmse={result.rmse_cv:.3f}  mape={result.mape_cv:.2f}%"
        if result.rmse_cv is not None and result.rmse_cv == result.rmse_cv
        else "Backtest: n/a (series too short)"
    )

    click.echo(click.style("\n=== Routing ===", bold=True))
    click.echo(decision.reason)
    for name, score in decision.scores.items():
        mark = "*" if name == decision.chosen else " "
        click.echo(f"  {mark} {name:15s} rmse={score['rmse']:.3f}  mape={score['mape']:.2f}%")

    click.echo(click.style("\n=== Explanation ===", bold=True))
    click.echo(
        f"Expected change vs recent history: {explanation['expected_change_abs']:+.2f} "
        f"({explanation['expected_change_pct']:+.1f}%)"
    )
    click.echo(f"Mean interval width: {explanation['interval_width']:.2f}")
    drivers = explanation.get("drivers", {})
    if drivers.get("model_features"):
        click.echo("Top model features:")
        for k, v in drivers["model_features"].items():
            click.echo(f"  - {k}: {v:.3f}")
    if drivers.get("calendar_drivers"):
        click.echo("Calendar drivers:")
        for k, v in drivers["calendar_drivers"].items():
            click.echo(f"  - {k}: {v:.3f}")

    click.echo(click.style("\n=== Recommendations ===", bold=True))
    badge = {"act": click.style("ACT  ", fg="red"), "watch": click.style("WATCH", fg="yellow"), "info": click.style("INFO ", fg="cyan")}
    for r in recs:
        click.echo(f"[{badge.get(r.severity, r.severity)}] {r.headline}")
        click.echo(f"        why:    {r.rationale}")
        click.echo(f"        action: {r.action}")

    if narrative:
        click.echo(click.style("\n=== Narrative ===", bold=True))
        click.echo(narrative)


if __name__ == "__main__":
    cli()
