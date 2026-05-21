"""Streamlit operations dashboard for the bi-forecast engine.

Run:
    streamlit run bi_forecast/dashboard/app.py
"""

import io
import pandas as pd
import streamlit as st

from bi_forecast.router import ModelRouter
from bi_forecast.explain import explain_forecast
from bi_forecast.recommend import recommend_actions
from bi_forecast.whatif import what_if, WhatIfScenario
from bi_forecast.noshow import NoShowClassifier, synthesize_appointments


st.set_page_config(page_title="Healthcare AI Ops", layout="wide")
st.title("Healthcare Ops AI — Forecasts, Drivers, Actions")

tab_forecast, tab_whatif, tab_noshow = st.tabs(["Forecast", "What-if", "No-show"])

# --------------- Forecast tab ---------------
with tab_forecast:
    st.subheader("Operational forecast")
    uploaded = st.file_uploader("Upload CSV with date + target column", type=["csv"], key="fcst_csv")
    date_col = st.text_input("Date column", value="ds")
    target_col = st.text_input("Target column", value="admissions")
    horizon = st.slider("Horizon (steps)", 1, 60, 14)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        series = df[target_col].astype(float)
        series.name = target_col

        router = ModelRouter()
        with st.spinner("Routing across base models…"):
            result, decision, _ = router.forecast(series, horizon, target_name=target_col)
        explanation = explain_forecast(result, series=series)
        recs = recommend_actions(result, explanation)

        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen model", result.model)
        c2.metric("Backtest MAPE", f"{result.mape_cv:.1f}%" if result.mape_cv == result.mape_cv else "n/a")
        c3.metric("Expected change", f"{explanation['expected_change_pct']:+.1f}%")

        chart_df = pd.concat([
            series.rename("history"),
            pd.Series(result.forecast_values, index=result.forecast_index, name="forecast"),
            pd.Series(result.upper, index=result.forecast_index, name="upper"),
            pd.Series(result.lower, index=result.forecast_index, name="lower"),
        ], axis=1)
        st.line_chart(chart_df)

        st.markdown("### Router decision")
        st.json({
            "chosen": decision.chosen,
            "candidates": decision.candidates,
            "scores": decision.scores,
            "reason": decision.reason,
        })

        st.markdown("### Recommendations")
        for r in recs:
            colour = {"act": "🔴", "watch": "🟡", "info": "🔵"}.get(r.severity, "·")
            st.markdown(f"**{colour} {r.headline}**  \n*why:* {r.rationale}  \n*action:* {r.action}")
    else:
        st.info("Upload a CSV to begin. Try `examples/admissions.csv` from the repo.")

# --------------- What-if tab ---------------
with tab_whatif:
    st.subheader("What-if scenarios")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="wi_csv")
    target_col2 = st.text_input("Target column", value="admissions", key="wi_target")
    horizon2 = st.slider("Horizon", 1, 30, 7, key="wi_horizon")
    mult_up = st.slider("Surge multiplier (last 30 days)", 1.0, 2.0, 1.2)
    mult_dn = st.slider("Drop multiplier (last 30 days)", 0.3, 1.0, 0.8)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df["ds"] = pd.to_datetime(df["ds"])
        series = df.sort_values("ds").set_index("ds")[target_col2].astype(float)
        scenarios = [
            WhatIfScenario(name="surge", multiplier=mult_up, on_last_n=30),
            WhatIfScenario(name="drop", multiplier=mult_dn, on_last_n=30),
        ]
        with st.spinner("Running scenarios…"):
            results = what_if(series, horizon2, scenarios, target_name=target_col2)
        chart_df = pd.DataFrame({
            name: pd.Series(r.forecast_values, index=r.forecast_index)
            for name, r in results.items()
        })
        st.line_chart(chart_df)
        st.dataframe(chart_df.describe().T)

# --------------- No-show tab ---------------
with tab_noshow:
    st.subheader("Patient-level no-show risk")
    if st.button("Train default model on synthetic data"):
        df = synthesize_appointments(n=2000)
        clf = NoShowClassifier()
        ev = clf.fit(df, target="no_show")
        st.session_state["clf"] = clf
        st.success(f"Trained — AUC={ev.auc:.3f}, n={ev.n_train}")
        st.write("Feature importance:", ev.feature_importance)

    uploaded = st.file_uploader("Upload scheduled appointments CSV", type=["csv"], key="ns_csv")
    if uploaded is not None and "clf" in st.session_state:
        df = pd.read_csv(uploaded)
        scores = st.session_state["clf"].predict(df)
        actions = st.session_state["clf"].recommend(scores)
        st.dataframe(pd.DataFrame(actions))
    elif uploaded is not None:
        st.warning("Train the model first.")
