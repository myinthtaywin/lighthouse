"""Critic agent.

Quality-checks the Matcher's output against the profile and decides whether the
match set is good enough to ship or should be sent back for a broadened retry.
This is what makes Lighthouse self-correcting: the verdict drives the
Matcher<->Critic refinement loop in the Orchestrator.

Uses REASONING_MODEL (this is a judgment call, not extraction). It judges on the
PROFILE + MATCHES only — the match set already carries scores, reasoning, and
eligibility_concerns, so there is no need to reload the full catalog. That keeps
the Critic cheap.
"""

from __future__ import annotations

import json

import weave
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, REASONING_MODEL
from state import CaseState

client = Anthropic(api_key=ANTHROPIC_API_KEY)


def _build_system_prompt(already_broadened: bool) -> str:
    base = """You are the Critic agent for Lighthouse, a system that matches nonprofits to resources, funding, and partner organizations.

You are a critical reviewer. Given a nonprofit's PROFILE and the Matcher's ranked MATCHES, decide whether the match set is good enough to ship to the user, or should be sent back to the Matcher for a broadened retry.

Assess three things:
- STRENGTH: Are there at least 2-3 matches with strong scores (>= 70) that the org is plausibly eligible for?
- COVERAGE: Do the matches actually address the profile's need_types? (e.g. if they need funding AND partners, are BOTH represented in the set?)
- ELIGIBILITY: Are the top matches clean, or are they riddled with disqualifying eligibility_concerns? A "disqualifying" concern is a hard gate the org clearly fails (wrong population, unreachable geography, wrong size band) — not a soft caveat like "competitive" or "confirm budget".

DECISION RULES:
- decision = "broaden_and_retry" if ANY of:
    * top scores are weak (all matches < 65), OR
    * fewer than 2 viable matches remain after setting aside ones with disqualifying concerns, OR
    * a stated need_type is completely uncovered by the set.
- decision = "accept" otherwise.
"""

    if already_broadened:
        second_pass = """
IMPORTANT — THIS IS THE SECOND PASS (already_broadened = true):
The Matcher has already broadened once. Do NOT request another retry. You MUST set decision = "accept".
Accept the best available set and record any residual gaps honestly in "reasons". The Orchestrator caps retries at one; do not fight that.
"""
    else:
        second_pass = """
This is the FIRST pass (already_broadened = false). Apply the decision rules above normally.
"""

    output = """
Return ONLY valid JSON (no preamble, no markdown fences):
{
  "decision": "accept" | "broaden_and_retry",
  "quality_score": integer 0-100,
  "reasons": ["short strings explaining the verdict"],
  "broaden_notes": "specific, actionable guidance for the Matcher's retry — which constraints to relax and why; empty string if decision is accept"
}

broaden_notes is fed directly into the Matcher's retry, so make it CONCRETE and actionable. Good example:
"Geography gated out strong national grants; widen from Cambridge to national. Youth-arts focus too narrow; include general youth-development funders."
If decision is "accept", broaden_notes MUST be an empty string."""

    return base + second_pass + output


def _build_user_prompt(profile: dict, matches: list[dict]) -> str:
    return (
        "PROFILE:\n"
        + json.dumps(profile, indent=2)
        + "\n\nMATCHES (ranked):\n"
        + json.dumps(matches, indent=2)
        + "\n\nReturn your verdict as JSON now."
    )


def _strip_fences(text: str) -> str:
    """Remove accidental ```json ... ``` fences and surrounding whitespace."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _call_model(system: str, user: str) -> str:
    response = client.messages.create(
        model=REASONING_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


_FAIL_SAFE = {
    "decision": "accept",
    "quality_score": 50,
    "reasons": ["critic_parse_error"],
    "broaden_notes": "",
}


def _coerce_verdict(verdict: dict, already_broadened: bool) -> dict:
    """Defensive normalization of the model's JSON."""
    decision = verdict.get("decision")
    if decision not in ("accept", "broaden_and_retry"):
        decision = "accept"
    # Enforce the no-second-retry contract even if the model ignores it.
    if already_broadened:
        decision = "accept"

    try:
        quality = int(verdict.get("quality_score", 50))
    except (TypeError, ValueError):
        quality = 50

    reasons = verdict.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    notes = verdict.get("broaden_notes", "") or ""
    if decision == "accept":
        notes = ""

    return {
        "decision": decision,
        "quality_score": quality,
        "reasons": reasons,
        "broaden_notes": notes,
    }


@weave.op()
def run_critic(
    profile: dict,
    matches: list[dict],
    already_broadened: bool = False,
) -> dict:
    """Judge the match set and decide accept vs. broaden_and_retry.

    Args:
        profile: the structured profile from Intake.
        matches: the Matcher's ranked match list.
        already_broadened: True on the second pass; forces decision = "accept"
            so the system can never loop more than once.

    Returns:
        {"decision", "quality_score", "reasons", "broaden_notes"}
    """
    system = _build_system_prompt(already_broadened)
    user = _build_user_prompt(profile, matches)

    raw = _call_model(system, user)
    cleaned = _strip_fences(raw)

    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError:
        repair_user = (
            "The following was supposed to be a single valid JSON object but "
            "could not be parsed. Return ONLY the corrected, valid JSON object "
            "with no preamble and no markdown fences:\n\n" + raw
        )
        repaired = _strip_fences(_call_model(system, repair_user))
        try:
            verdict = json.loads(repaired)
        except json.JSONDecodeError:
            # Fail safe = accept, so a parsing bug never traps us in a loop.
            return dict(_FAIL_SAFE)

    if not isinstance(verdict, dict):
        return dict(_FAIL_SAFE)

    return _coerce_verdict(verdict, already_broadened)


def apply_critic(state: CaseState, verdict: dict) -> CaseState:
    """Write the verdict into state and append a human-readable trace line."""
    state.critic_verdict = verdict
    decision = verdict.get("decision", "accept")
    if decision == "accept":
        line = f"Critic: accept (quality {verdict.get('quality_score')})"
    else:
        notes = verdict.get("broaden_notes", "") or "(no notes)"
        line = f"Critic: broaden_and_retry — {notes}"
    state.trace_log.append(line)
    return state


if __name__ == "__main__":
    import json as _json
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from agents.intake import run_intake
    from agents.matcher import run_matcher

    samples_path = (
        Path(__file__).resolve().parent.parent / "data" / "intake_samples.json"
    )
    samples = _json.loads(samples_path.read_text())

    console = Console()
    table = Table(title="Critic — first-pass verdicts (broaden=False)")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("label")
    table.add_column("top_score", justify="right")
    table.add_column("decision")
    table.add_column("quality", justify="right")
    table.add_column("first_reason")

    for s in samples:
        intake = run_intake(s["text"])
        match_result = run_matcher(intake["profile"], broaden=False)
        matches = match_result.get("matches", [])
        top_score = matches[0]["match_score"] if matches else 0

        verdict = run_critic(intake["profile"], matches, already_broadened=False)
        decision = verdict["decision"]
        dec_style = "green" if decision == "accept" else "yellow"
        first_reason = (verdict.get("reasons") or ["-"])[0]

        table.add_row(
            s["id"],
            s["label"],
            str(top_score),
            f"[{dec_style}]{decision}[/{dec_style}]",
            str(verdict.get("quality_score")),
            first_reason,
        )

    console.print(table)
