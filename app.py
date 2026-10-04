"""
TrueStream: AI-assisted quality checks for citizen stream science.
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from truestream.ai_review import ERRORS, GEMINI_MODEL, MODEL, review
from truestream.evaluate import evaluate, false_alarms
from truestream.generate_data import generate
from truestream.rules import run_rules

st.set_page_config(page_title="TrueStream", page_icon="💧", layout="wide")

STATUS_COLORS = {  # used on the map
    "clean": "#4E7D3A",
    "needs review": "#D08C17",
    "confirmed issue": "#A8382B",
    "dismissed": "#3C6E8F",
}

# ---------------- sidebar: data + settings ----------------
st.sidebar.title("💧 TrueStream")
st.sidebar.caption("AI-assisted quality checks for citizen stream science")

source = st.sidebar.radio("Data", ["Demo data with planted errors", "Upload CSV"])
uploaded = st.sidebar.file_uploader("Observations CSV", type="csv") if source == "Upload CSV" else None

def _load_secret(name: str) -> None:
    """Copy a key from .streamlit/secrets.toml into the environment, if present."""
    if os.getenv(name):
        return
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file configured
        value = None
    if value:
        os.environ[name] = value


for _name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
    _load_secret(_name)

AI_OPTIONS = {
    "Gemini (real AI, free tier)": ("gemini", "GEMINI_API_KEY"),
    "Claude (real AI)": ("claude", "ANTHROPIC_API_KEY"),
    "Offline demo (keyword rules, not AI)": ("offline", None),
}
default_ai = next((i for i, (_, k) in enumerate(AI_OPTIONS.values()) if k and os.getenv(k)), 2)
ai_choice = st.sidebar.radio("AI review", list(AI_OPTIONS), index=default_ai,
                             help="Real AI needs an API key in .streamlit/secrets.toml.")
mode, key_name = AI_OPTIONS[ai_choice]
if key_name and not os.getenv(key_name):
    st.sidebar.warning(f"No {key_name} found. Add it to .streamlit/secrets.toml and restart the app.")

run = st.sidebar.button("Run quality checks", type="primary", width="stretch")


def load_data() -> pd.DataFrame | None:
    if source == "Upload CSV":
        if uploaded is None:
            return None
        df = pd.read_csv(uploaded)
    else:
        df = generate()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "planted_error" not in df.columns:
        df["planted_error"] = False
        df["planted_error_type"] = ""
    return df


if run:
    df = load_data()
    if df is None:
        st.sidebar.error("Upload a CSV first.")
    else:
        with st.spinner("Running validation rules…"):
            rule_issues = run_rules(df)
        bar = st.progress(0.0, text="AI is reviewing observations…")
        ai_issues = review(df, rule_issues, mode=mode,
                           progress=lambda p: bar.progress(p, text="AI is reviewing observations…"))
        bar.empty()
        if ERRORS:
            st.error(f"{len(ERRORS)} AI batch(es) failed and were skipped (run again to retry). "
                     f"First error: {ERRORS[0]}")
        st.session_state.update(df=df, issues=pd.concat([rule_issues, ai_issues], ignore_index=True),
                                mode=mode, decisions={})

if "df" not in st.session_state:
    st.title("TrueStream")
    st.write("Citizen scientists collect valuable stream data, but mistakes slip in. TrueStream "
             "checks every observation with validation rules and AI, explains each problem in plain "
             "language, and leaves the final decision to a human reviewer.")
    st.info("Choose a data source in the sidebar, then select **Run quality checks**.")
    st.stop()

df: pd.DataFrame = st.session_state.df
issues: pd.DataFrame = st.session_state.issues
decisions: dict = st.session_state.decisions
flagged_ids = list(dict.fromkeys(issues.observation_id))


def status_of(obs_id: str) -> str:
    if obs_id in decisions:
        return decisions[obs_id]["decision"]
    return "needs review" if obs_id in flagged_ids else "clean"


df["status"] = df.observation_id.map(status_of)

# ---------------- header metrics ----------------
st.title("TrueStream")
if st.session_state.mode == "offline":
    st.warning("Offline demo mode: the AI layer is a keyword heuristic, not real AI. "
               "Switch to Gemini or Claude for real results.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", len(df))
c2.metric("Flagged for review", len(flagged_ids))
c3.metric("Reviewed by human", len(decisions))
c4.metric("Still to review", (df.status == "needs review").sum())

tab_review, tab_map, tab_report, tab_acc = st.tabs(
    ["Review queue", "Map", "Data quality report", "Accuracy test"])

# ---------------- review queue (human-in-the-loop) ----------------
with tab_review:
    show = st.radio("Show", ["Needs review", "Reviewed", "All flagged"], horizontal=True)
    ids = [i for i in flagged_ids
           if (show == "Needs review" and i not in decisions)
           or (show == "Reviewed" and i in decisions)
           or show == "All flagged"]
    if not ids:
        st.success("Nothing here. All flagged observations have been reviewed.")
    for obs_id in ids[:60]:
        rec = df[df.observation_id == obs_id].iloc[0]
        rec_issues = issues[issues.observation_id == obs_id]
        label = f"{obs_id} · {rec.get('site_name', '')} · {len(rec_issues)} issue(s)"
        if obs_id in decisions:
            label += f" · {decisions[obs_id]['decision']}"
        with st.expander(label):
            left, right = st.columns([3, 2])
            with left:
                for _, iss in rec_issues.iterrows():
                    who = "🤖 AI" if iss.layer == "ai" else "📏 Rule"
                    st.markdown(f"**{who} · {iss.rule.replace('_', ' ')}** — {iss.message}")
                    if iss.volunteer_question:
                        st.caption(f"Ask the volunteer: “{iss.volunteer_question}”")
            with right:
                cols = [c for c in ["timestamp", "water_temp_c", "ph", "water_clarity", "odor", "flow",
                                    "trash_level", "insects_count", "health_rating", "notes"] if c in df]
                st.dataframe(rec[cols].astype(str).rename("value"), width="stretch")
            comment = st.text_input("Reviewer comment (optional)", key=f"c_{obs_id}")
            b1, b2 = st.columns(2)
            if b1.button("Confirm issue", key=f"y_{obs_id}", width="stretch"):
                decisions[obs_id] = {"decision": "confirmed issue", "comment": comment,
                                     "reviewed_at": datetime.now().isoformat(timespec="seconds")}
                st.rerun()
            if b2.button("Dismiss — record is fine", key=f"n_{obs_id}", width="stretch"):
                decisions[obs_id] = {"decision": "dismissed", "comment": comment,
                                     "reviewed_at": datetime.now().isoformat(timespec="seconds")}
                st.rerun()

    st.divider()
    log = pd.DataFrame([{"observation_id": k, **v} for k, v in decisions.items()])
    d1, d2 = st.columns(2)
    d1.download_button("Download review log (CSV)", log.to_csv(index=False), "review_log.csv",
                       disabled=log.empty, width="stretch")
    trusted = df[df.status.isin(["clean", "dismissed"])].drop(
        columns=["planted_error", "planted_error_type", "status"], errors="ignore")
    d2.download_button("Download trusted dataset (CSV)", trusted.to_csv(index=False),
                       "trusted_observations.csv", width="stretch")

# ---------------- map ----------------
with tab_map:
    m = df.dropna(subset=["latitude", "longitude"]).copy()
    m = m[m.latitude.between(-90, 90) & m.longitude.between(-180, 180)]
    m["color"] = m.status.map(STATUS_COLORS)
    st.map(m, latitude="latitude", longitude="longitude", color="color", size=25)
    st.caption(" · ".join(f"<span style='color:{c}'>●</span> {s}" for s, c in STATUS_COLORS.items()),
               unsafe_allow_html=True)

# ---------------- data quality report ----------------
with tab_report:
    st.subheader("Most common problems")
    counts = issues.rule.str.replace("_", " ").value_counts()
    st.bar_chart(counts, horizontal=True)
    a, b = st.columns(2)
    with a:
        st.subheader("By site")
        by_site = (issues.merge(df[["observation_id", "site_name"]], on="observation_id")
                   .groupby("site_name").observation_id.nunique().sort_values(ascending=False))
        st.bar_chart(by_site)
    with b:
        st.subheader("Volunteers who may need guidance")
        by_vol = (issues.merge(df[["observation_id", "volunteer_id"]], on="observation_id")
                  .groupby("volunteer_id").observation_id.nunique().sort_values(ascending=False).head(8))
        st.bar_chart(by_vol)
    if not counts.empty:
        top = counts.index[0]
        st.info(f"Most frequent problem: **{top}** ({counts.iloc[0]} cases). If one problem dominates, "
                "the app question behind it may be confusing and worth redesigning (supports Track 1).")

# ---------------- accuracy test ----------------
with tab_acc:
    if not df.planted_error.astype(bool).any():
        st.info("Accuracy testing needs data with known (planted) errors. Use the demo data.")
    else:
        summary, by_type = evaluate(df, issues)
        st.write("We planted known errors in the data, like seeding defects in software testing, "
                 "and measured how many TrueStream catches and how many false alarms it raises.")
        best = summary.loc["Rules + AI (TrueStream)"]
        x, y, z = st.columns(3)
        x.metric("Planted errors caught", f"{int(best.caught)} / {int(best.caught + best.missed)}")
        y.metric("False alarms", int(best.false_alarms))
        z.metric("Precision", f"{best.precision:.0%}")
        st.dataframe(summary, width="stretch")
        st.dataframe(by_type, width="stretch", hide_index=True)
        fa = false_alarms(df, issues)
        st.subheader(f"False alarms ({len(fa)})")
        if fa.empty:
            st.success("No false alarms: every flagged record really had a planted error.")
        else:
            st.caption("Valid records that were flagged anyway. Use these to improve the rules and AI prompt.")
            st.dataframe(fa, width="stretch", hide_index=True)
        if st.session_state.mode != "offline":
            model = GEMINI_MODEL if st.session_state.mode == "gemini" else MODEL
            st.caption(f"AI model: {model}")
