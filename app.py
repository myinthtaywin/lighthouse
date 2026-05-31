"""Streamlit UI for Lighthouse.

A thin presentation layer over the Orchestrator. It imports and renders only —
it does NOT modify orchestrator.py, the agents, state.py, or config.py.

The goal is to make the multi-agent orchestration VISIBLE: an animated pipeline
timeline that surfaces every agent step and BOTH self-correcting loops
(clarification + refinement), not just an input box and a final answer.

Run locally for the live demo:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import weave

from config import WEAVE_PROJECT
from orchestrator import run_orchestrator

# --- Page config (must be the first Streamlit call) ------------------------
st.set_page_config(page_title="Lighthouse", page_icon="🪧", layout="wide")

_SAMPLES_PATH = Path(__file__).resolve().parent / "data" / "intake_samples.json"
_CUSTOM = "— custom —"
_SAMPLE_5_ANSWER = "We serve Greater Boston, mainly Cambridge and Somerville"


# --- Weave init (guarded so Streamlit reruns don't re-init) ----------------
@st.cache_resource
def init_weave():
    return weave.init(WEAVE_PROJECT)


def _weave_project_url(client) -> str:
    entity = getattr(client, "entity", None)
    project = getattr(client, "project", None)
    if entity and project:
        return f"https://wandb.ai/{entity}/{project}/weave"
    return "https://wandb.ai/"


@st.cache_data
def load_samples() -> list[dict]:
    return json.loads(_SAMPLES_PATH.read_text())


# --- Initialize -------------------------------------------------------------
weave_client = init_weave()
weave_url = _weave_project_url(weave_client)
samples = load_samples()

# Map a friendly dropdown label -> sample dict.
_label_to_sample = {f"{s['id']}  ({s['label']})": s for s in samples}
_options = [_CUSTOM] + list(_label_to_sample.keys())

# Seed session state defaults (default to "— custom —" with an empty box, so
# the user can type their own description right away).
if "case_text" not in st.session_state:
    st.session_state.sample_choice = _CUSTOM
    st.session_state.case_text = ""
    st.session_state.clar_answer = ""


def _on_sample_change() -> None:
    choice = st.session_state.sample_choice
    if choice == _CUSTOM:
        st.session_state.case_text = ""
        st.session_state.clar_answer = ""
        return
    sample = _label_to_sample[choice]
    st.session_state.case_text = sample["text"]
    st.session_state.clar_answer = (
        _SAMPLE_5_ANSWER if sample["id"] == "sample_5" else ""
    )


# --- Header -----------------------------------------------------------------
st.title("🪧 Lighthouse — Multi-Agent Nonprofit Resource Matching")
st.markdown(
    "_Describe your nonprofit in plain language; Lighthouse triages it, finds "
    "the right resources, and vets the results before returning them._"
)
st.caption(
    "Four agents collaborate: an **Orchestrator** routes the case between an "
    "**Intake** agent (structures the request), a **Matcher** agent (ranks "
    "resources), and a **Critic** agent (quality-checks the matches) — with "
    "two self-correcting loops: clarification and refinement."
)

# --- Input ------------------------------------------------------------------
st.subheader("1 · Describe the nonprofit")

left, right = st.columns([2, 1])
with left:
    st.selectbox(
        "Load an example",
        options=_options,
        key="sample_choice",
        on_change=_on_sample_change,
    )
    st.text_area(
        "Nonprofit description",
        key="case_text",
        height=180,
    )
    if st.session_state.sample_choice == _CUSTOM:
        st.caption(
            "Paste your own nonprofit description (100–150 words works well): "
            "what you do, who you serve, your size/location, and what you're "
            "looking for."
        )
with right:
    st.text_input(
        "If the system asks a clarifying question, answer with:",
        key="clar_answer",
        help=(
            "Pre-supplied so the demo runs hands-free. Leave blank to be "
            "prompted only in a terminal run."
        ),
    )
    st.markdown("&nbsp;")
    run_clicked = st.button("🚀 Run Lighthouse", type="primary", use_container_width=True)


# --- Helpers for rendering --------------------------------------------------
def _render_timeline(trace_log: list[str]) -> None:
    """Animate the routing trace, highlighting loop events."""
    st.subheader("2 · Pipeline (live routing trace)")
    icons = {
        "Intake": "📝",
        "Matcher": "🔎",
        "Critic": "🧪",
        "Orchestrator": "🧭",
    }
    for line in trace_log:
        if line.startswith("Clarification:"):
            st.warning(f"❓ **Clarification loop** — {line[len('Clarification:'):].strip()}")
        elif line.startswith("Refinement:"):
            st.warning(f"🔁 **Refinement loop** — {line[len('Refinement:'):].strip()}")
        elif line.startswith("Critic:") and "broaden_and_retry" in line:
            st.info(f"🧪 {line}")
        elif line.startswith("Orchestrator:"):
            st.success(f"🧭 {line}")
        else:
            prefix = line.split(":", 1)[0]
            st.info(f"{icons.get(prefix, '•')} {line}")
        time.sleep(0.4)


def _render_understood_as(state) -> None:
    packet = state.final_packet or {}
    u = packet.get("understood_as", {})
    conf = packet.get("confidence")

    st.subheader("3 · Understood as")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**Summary:** {u.get('summary') or '—'}")
        st.markdown(
            f"**Need types:** {', '.join(u.get('need_types') or []) or '—'}  \n"
            f"**Focus areas:** {', '.join(u.get('focus_areas') or []) or '—'}  \n"
            f"**Populations:** {', '.join(u.get('populations_served') or []) or '—'}  \n"
            f"**Geography:** {u.get('geography') or '—'}  \n"
            f"**Org size:** {u.get('org_size') or '—'}"
        )
    with c2:
        if isinstance(conf, (int, float)):
            emoji = "🟢" if conf >= 0.6 else "🔴"
            st.metric("Intake confidence", f"{emoji} {conf:.2f}")
        else:
            st.metric("Intake confidence", "n/a")


def _render_clarifications(state) -> None:
    if not state.clarifications:
        return
    st.subheader("🙋 Human-in-the-loop clarification")
    for qa in state.clarifications:
        st.markdown(f"> **Q:** {qa.get('question', '')}")
        st.markdown(f"> **A:** {qa.get('answer', '')}")


def _render_vetting(state) -> None:
    verdict = state.critic_verdict or {}
    packet = state.final_packet or {}

    st.subheader("4 · Critic vetting")
    c1, c2 = st.columns([1, 3])
    with c1:
        q = verdict.get("quality_score")
        st.metric("Quality score", q if q is not None else "n/a")
        decision = verdict.get("decision", "—")
        if decision == "accept":
            st.success(f"Decision: {decision}")
        else:
            st.warning(f"Decision: {decision}")
    with c2:
        if packet.get("broadened"):
            st.markdown(
                "🔁 **Matches broadened after Critic review** — the first match "
                "set was judged insufficient, so the Matcher re-ran with relaxed "
                "constraints."
            )
        reasons = verdict.get("reasons") or []
        if reasons:
            st.markdown("**Critic reasoning:**")
            for r in reasons:
                st.markdown(f"- {r}")


def _render_resources(state) -> None:
    packet = state.final_packet or {}
    resources = packet.get("recommended_resources", [])
    st.subheader("5 · Recommended resources")
    if not resources:
        st.info("No resources were recommended.")
        return
    for i, r in enumerate(resources, start=1):
        with st.container(border=True):
            top = st.columns([6, 1])
            with top[0]:
                st.markdown(f"#### {i}. {r.get('name') or '—'}")
                st.caption(f"Type: {r.get('type') or '—'}")
            with top[1]:
                score = r.get("match_score")
                st.metric("Match", score if score is not None else "—")
            st.markdown(f"**Why it fits:** {r.get('why_it_fits') or '—'}")
            concerns = r.get("eligibility_concerns") or []
            if concerns:
                st.markdown(
                    "**Eligibility concerns:** " + "; ".join(concerns)
                )
            link = r.get("link")
            if link:
                st.markdown(f"🔗 [{link}]({link})")


# --- Run --------------------------------------------------------------------
if run_clicked:
    text = (st.session_state.case_text or "").strip()
    answer = (st.session_state.clar_answer or "").strip()

    if not text:
        st.warning("Enter a description or pick an example first.")
        st.stop()

    try:
        with st.spinner("Running the multi-agent pipeline… (this calls live models)"):
            state = run_orchestrator(text, answers=[answer] if answer else None)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        st.error(f"Something went wrong while running Lighthouse:\n\n{exc}")
        st.stop()

    _render_timeline(state.trace_log)
    _render_understood_as(state)
    _render_clarifications(state)
    _render_vetting(state)
    _render_resources(state)

    st.divider()
    st.markdown(
        f"🔍 **Full trace in W&B Weave** — every agent call "
        f"(Intake, clarifying-question, Matcher, Critic) was traced and nested "
        f"under a single `run_orchestrator` parent. "
        f"[Open the Weave project →]({weave_url})"
    )
else:
    st.info("Pick an example or paste a description, then click **Run Lighthouse**.")
