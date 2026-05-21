"""Streamlit operations dashboard for the Verdan AI healthcare engine.

Talks to the running bi-forecast FastAPI service. Start the API first:

    bi-forecast serve --port 8000

Then launch this dashboard in another terminal:

    streamlit run bi_forecast/dashboard/app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st


# ----------------------------- API client -----------------------------

DEFAULT_API = os.environ.get("BI_API_URL", "http://127.0.0.1:8000")


class Api:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> Dict[str, Any]:
        return self._client.get("/health").json()

    def history(self, kind: str, limit: int = 50) -> Dict[str, Any]:
        return self._client.get(f"/history/{kind}", params={"limit": limit}).json()

    def forecast(self, target: str, series: List[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
        return self._client.post("/forecast", json={"target": target, "series": series, "horizon": horizon}).json()

    def whatif(self, target: str, series: List[Dict[str, Any]], horizon: int, scenarios: List[dict]) -> Dict[str, Any]:
        return self._client.post(
            "/whatif",
            json={"target": target, "series": series, "horizon": horizon, "scenarios": scenarios},
        ).json()

    def noshow_train(self, csv_bytes: bytes, target: str = "no_show", save_to: Optional[str] = None) -> Dict[str, Any]:
        files = {"file": ("train.csv", csv_bytes, "text/csv")}
        data = {"target": target}
        if save_to:
            data["save_to"] = save_to
        return self._client.post("/noshow/train", files=files, data=data).json()

    def noshow_score(self, appointments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._client.post("/noshow/score", json={"appointments": appointments}).json()

    def noshow_status(self) -> Dict[str, Any]:
        return self._client.get("/noshow/status").json()

    def engagement_preview(self, appointments: List[Dict[str, Any]], channel: str) -> Dict[str, Any]:
        return self._client.post(
            "/engagement/preview", json={"appointments": appointments, "channel": channel}
        ).json()

    def engagement_send(self, appointments: List[Dict[str, Any]], channel: str, skip_low: bool) -> Dict[str, Any]:
        return self._client.post(
            "/engagement/send",
            json={"appointments": appointments, "channel": channel, "skip_low_risk": skip_low},
        ).json()

    def hl7_post(self, message: str) -> Dict[str, Any]:
        return self._client.post("/hooks/hl7", json={"message": message}).json()

    def pipeline(self, csv_bytes: bytes, channel: str, send: bool, skip_low: bool, clinic: str) -> Dict[str, Any]:
        files = {"file": ("pipeline.csv", csv_bytes, "text/csv")}
        data = {
            "channel": channel,
            "send": str(send).lower(),
            "skip_low_risk": str(skip_low).lower(),
            "default_clinic": clinic,
        }
        return self._client.post("/pipeline/score-and-send", files=files, data=data).json()


@st.cache_resource(show_spinner=False)
def get_api(base_url: str) -> Api:
    return Api(base_url)


# ----------------------------- UI helpers -----------------------------

st.set_page_config(page_title="Verdan AI — Ops Console", layout="wide")

with st.sidebar:
    st.title("Verdan AI")
    st.caption("Healthcare operations console")
    base_url = st.text_input("API base URL", value=DEFAULT_API)
    api = get_api(base_url)
    try:
        h = api.health()
        st.success(f"API reachable\nDB: {h.get('db', '?')}")
    except Exception as e:
        st.error(f"API unreachable at {base_url}\n{e}")
        st.stop()
    st.divider()
    st.caption("Start the API with `bi-forecast serve --port 8000`")


tab_overview, tab_pipeline, tab_noshow, tab_forecast, tab_engage, tab_history, tab_hl7 = st.tabs(
    ["Overview", "Pipeline", "No-show ops", "Forecast", "Engagement", "History", "HL7 inbox"]
)


# ----------------------------- Pipeline (one-shot) -----------------------------

with tab_pipeline:
    st.subheader("One-call pipeline: score → send → persist")
    st.caption(
        "Upload an appointments CSV with at least `appointment_id`, `name`, `phone`, "
        "`appointment_time`. Optional no-show features (`age`, `prior_no_shows`, etc.) "
        "improve scoring accuracy. See `examples/pipeline_appointments.csv`."
    )
    pf = st.file_uploader("Appointments CSV", type=["csv"], key="pipe_csv")
    c1, c2, c3 = st.columns(3)
    pchannel = c1.selectbox("Channel", ["console", "sms", "whatsapp"], index=0, key="pipe_ch")
    psend = c2.toggle("Actually send (else dry-run)", value=False)
    pskip = c3.toggle("Skip low-risk", value=True)
    pclinic = st.text_input("Default clinic name", value="Verdan Care")

    if pf is not None and st.button("Run pipeline", type="primary"):
        with st.spinner("Scoring → dispatching → persisting…"):
            res = api.pipeline(pf.getvalue(), pchannel, psend, pskip, pclinic)

        summary = res.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input rows", summary.get("n_input", 0))
        m2.metric("Scored", summary.get("n_scored", 0))
        m3.metric("Dispatched", summary.get("n_sent", 0))
        m4.metric("Mode", summary.get("mode", "?"))

        if summary.get("by_risk_band"):
            st.bar_chart(pd.Series(summary["by_risk_band"]).rename("count"))

        rows = res.get("results", [])
        if rows:
            df = pd.DataFrame(rows)
            st.markdown("##### Per-appointment results")
            st.dataframe(
                df[["appointment_id", "risk_band", "probability", "channel", "to",
                    "status", "template_key"]],
                hide_index=True, use_container_width=True,
            )
            with st.expander("Show rendered message bodies"):
                for r in rows:
                    badge = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(r["risk_band"], "·")
                    st.markdown(f"{badge} **{r['appointment_id']}** → `{r['to']}` ({r['status']})")
                    st.code(r["rendered_body"])


# ----------------------------- Overview -----------------------------

with tab_overview:
    st.subheader("Today at a glance")
    try:
        preds = api.history("predictions", limit=500)
        delivs = api.history("deliveries", limit=500)
        hl7s = api.history("hl7", limit=500)
        models = api.history("models", limit=20)
    except Exception as e:
        st.error(f"Could not load history: {e}")
        preds = delivs = hl7s = models = {"n": 0, "items": []}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions stored", preds["n"])
    c2.metric("Reminders sent / dry-run", delivs["n"])
    c3.metric("HL7 messages received", hl7s["n"])
    c4.metric("Trained model versions", models["n"])

    if preds["items"]:
        df = pd.DataFrame(preds["items"])
        df["created_at"] = pd.to_datetime(df["created_at"])
        st.markdown("##### Risk band distribution (recent)")
        st.bar_chart(df["risk_band"].value_counts())

        st.markdown("##### Last 10 predictions")
        st.dataframe(df.head(10)[["appointment_id", "probability", "risk_band", "created_at"]], hide_index=True)
    else:
        st.info("No predictions yet. Score some appointments in the *No-show ops* tab.")


# ----------------------------- No-show ops -----------------------------

with tab_noshow:
    st.subheader("Train + score + send reminders")
    try:
        status = api.noshow_status()
    except Exception as e:
        st.warning(f"Could not fetch model status: {e}")
        status = {"loaded": False, "meta": {}}
    cstat1, cstat2 = st.columns(2)
    cstat1.metric("Model loaded", "yes" if status.get("loaded") else "no")
    cstat2.caption(f"Source: {status.get('meta', {}).get('source', '—')}")

    st.markdown("##### 1. Train the no-show model")
    train_file = st.file_uploader(
        "Past appointments CSV (must include `no_show` column)",
        type=["csv"], key="train_csv",
    )
    save_to = st.text_input("Save model to path (optional)", value="models/noshow_v1.joblib")
    if st.button("Train model", disabled=train_file is None):
        with st.spinner("Training…"):
            res = api.noshow_train(train_file.getvalue(), save_to=save_to or None)
        st.success(f"Trained — AUC={res.get('auc'):.3f}" if res.get("auc") else "Trained")
        if res.get("feature_importance"):
            st.write("Top features", res["feature_importance"])

    st.divider()
    st.markdown("##### 2. Score upcoming appointments")
    score_file = st.file_uploader("Upcoming appointments CSV (no `no_show` column needed)", type=["csv"], key="score_csv")
    if st.button("Score appointments", disabled=score_file is None):
        appts_df = pd.read_csv(score_file)
        with st.spinner("Scoring…"):
            res = api.noshow_score(appts_df.to_dict(orient="records"))
        st.session_state["last_scored"] = res.get("results", [])
        st.success(f"Scored {res.get('n', 0)} appointments")

    last = st.session_state.get("last_scored")
    if last:
        df = pd.DataFrame(last)
        st.dataframe(df, hide_index=True)

        st.divider()
        st.markdown("##### 3. Send reminders for these scores")
        st.caption("Needs an appointments CSV with `name`, `phone`, `appointment_time` etc. — upload below.")
        contact_file = st.file_uploader("Appointment contacts CSV", type=["csv"], key="contact_csv")
        channel = st.selectbox("Channel", ["console", "sms", "whatsapp"], index=0)
        skip_low = st.checkbox("Skip low-risk", value=True)
        if contact_file is not None and st.button("Send reminders"):
            contacts = pd.read_csv(contact_file)
            score_lookup = {r["appointment_id"]: r["risk_band"] for r in last}
            contacts["risk_band"] = contacts["appointment_id"].map(score_lookup).fillna("low")
            appts = contacts.to_dict(orient="records")
            with st.spinner("Dispatching…"):
                res = api.engagement_send(appts, channel=channel, skip_low=skip_low)
            st.success(f"Dispatched {res.get('n', 0)} messages")
            st.dataframe(pd.DataFrame(res.get("results", [])), hide_index=True)


# ----------------------------- Forecast -----------------------------

with tab_forecast:
    st.subheader("Operational forecast")
    f = st.file_uploader("Time-series CSV", type=["csv"], key="fcst_csv")
    date_col = st.text_input("Date column", value="ds")
    target_col = st.text_input("Target column", value="admissions")
    horizon = st.slider("Horizon", 1, 60, 14)

    if f is not None and st.button("Run forecast"):
        df = pd.read_csv(f)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        series = [{"ds": d.isoformat(), "y": float(v)} for d, v in zip(df[date_col], df[target_col])]
        with st.spinner("Routing & forecasting…"):
            res = api.forecast(target_col, series, horizon)

        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen model", res.get("model", "?"))
        exp = res.get("explanation", {})
        c2.metric("Expected change", f"{exp.get('expected_change_pct', 0):+.1f}%")
        bt = exp.get("backtest", {})
        c3.metric("Backtest MAPE", f"{bt.get('mape_pct', 0):.1f}%" if bt.get("mape_pct") is not None else "n/a")

        fc_df = pd.DataFrame(res.get("forecast", []))
        if not fc_df.empty:
            fc_df["ds"] = pd.to_datetime(fc_df["ds"])
            chart = pd.concat([
                df.rename(columns={date_col: "ds", target_col: "history"})[["ds", "history"]].set_index("ds"),
                fc_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]],
            ], axis=1)
            st.line_chart(chart)

        st.markdown("##### Router decision")
        st.json({"candidates": res.get("candidates"), "scores": res.get("scores")})

        st.markdown("##### Recommendations")
        for r in res.get("recommendations", []):
            badge = {"act": "🔴", "watch": "🟡", "info": "🔵"}.get(r.get("severity"), "·")
            st.markdown(f"**{badge} {r.get('headline')}**  \n*why:* {r.get('rationale')}  \n*action:* {r.get('action')}")


# ----------------------------- Engagement -----------------------------

with tab_engage:
    st.subheader("Preview message templates")
    f = st.file_uploader("Appointment contacts CSV", type=["csv"], key="engage_csv")
    channel = st.selectbox("Channel", ["sms", "whatsapp", "console"], key="engage_channel")
    if f is not None and st.button("Preview rendered messages"):
        df = pd.read_csv(f)
        appts = df.to_dict(orient="records")
        res = api.engagement_preview(appts, channel=channel)
        for r in res.get("results", []):
            st.markdown(f"**{r['risk_band'].upper()}** → `{r['to']}`  ({r['template_key']})")
            st.code(r["rendered_body"])


# ----------------------------- History -----------------------------

with tab_history:
    st.subheader("Persisted history")
    sub = st.radio("Show", ["Predictions", "Deliveries", "Models"], horizontal=True)
    limit = st.slider("Rows", 10, 500, 100)
    kind_map = {"Predictions": "predictions", "Deliveries": "deliveries", "Models": "models"}
    data = api.history(kind_map[sub], limit=limit)
    items = data.get("items", [])
    if not items:
        st.info("No rows yet.")
    else:
        st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)


# ----------------------------- HL7 inbox -----------------------------

with tab_hl7:
    st.subheader("HL7 inbox & test bench")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("##### Submit an HL7 v2 message")
        default = (
            "MSH|^~\\&|HIS|HOSP|AI_ENGINE|HOSP|20260521090000||SIU^S12|MSG00001|P|2.5\r"
            "SCH|APPT-2001||||||||30^MIN|^^^20260525101500^20260525104500|||SCHEDULED\r"
            "PID|1||MRN12345^^^HOSP^MR||Doe^Jane^A||19850314|F\r"
            "PV1|1|O|CARDIO^^OPD-3||||1234^Smith^John^Dr|||||||||||VIP"
        )
        hl7_text = st.text_area("Paste HL7 v2 message", value=default, height=180)
        if st.button("POST to /hooks/hl7"):
            res = api.hl7_post(hl7_text)
            st.json(res)

    with col_r:
        st.markdown("##### Recent inbox")
        inbox = api.history("hl7", limit=20)
        items = inbox.get("items", [])
        if items:
            df = pd.DataFrame([
                {"id": x["id"], "type": x["message_type"], "received_at": x["received_at"]}
                for x in items
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
            with st.expander("Latest parsed payload"):
                st.json(items[0]["parsed"])
        else:
            st.info("No HL7 traffic yet.")
