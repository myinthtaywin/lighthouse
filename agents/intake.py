"""Intake agent.

Turns a nonprofit's free-text description into a structured profile and, just
as importantly, reports how confident it is and what critical information is
missing. That confidence / missing_fields judgment is what lets the
Orchestrator decide whether to ask a clarifying question, so honest
self-assessment matters more here than perfect extraction.

Uses FAST_MODEL: intake is fast triage. The reasoning model is deliberately
reserved for the Matcher.
"""

from __future__ import annotations

import json
import os

import weave
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, FAST_MODEL
from state import CaseState

# --- Controlled vocabularies ------------------------------------------------
# These MUST line up with the values used in data/resources.json so the Matcher
# can compare profile <-> resource like-for-like. "other" is the escape hatch
# the model may use only when nothing else fits.
FOCUS_AREAS = [
    "housing",
    "food_security",
    "homelessness",
    "workforce",
    "youth",
    "education",
    "immigrant_services",
    "mental_health",
    "reentry",
    "domestic_violence",
    "disability",
    "environment",
    "arts",
    "other",
]

POPULATIONS = [
    "families",
    "veterans",
    "seniors",
    "youth",
    "immigrants",
    "unhoused",
    "disabled",
    "general",
]

# Geography: a specific stated city/town (e.g. "Cambridge"), otherwise one of
# these regional buckets.
GEOGRAPHY_BUCKETS = ["Greater Boston", "Massachusetts", "New England", "national"]

# Org size bands, keyed to annual operating budget.
ORG_SIZE = ["early_stage", "mid_size", "established"]  # <$250k / $250k-$2M / >$2M

# What the nonprofit is actually seeking (can be multiple).
NEED_TYPES = [
    "funding",
    "partners",
    "in_kind",
    "volunteers",
    "capacity_building",
    "fiscal_sponsorship",
]

client = Anthropic(api_key=ANTHROPIC_API_KEY)


def _build_system_prompt() -> str:
    return f"""You are the Intake agent for Lighthouse, a system that matches nonprofits to resources, funding, and partner organizations.

Your job: read a nonprofit's plain-language description and extract a STRUCTURED PROFILE, while honestly reporting your confidence and what critical information is missing.

You MUST map onto these controlled vocabularies. Use the closest matching value(s); use "other" (for focus_areas) only when nothing fits.

FOCUS_AREAS: {", ".join(FOCUS_AREAS)}
POPULATIONS: {", ".join(POPULATIONS)}
NEED_TYPES: {", ".join(NEED_TYPES)}
ORG_SIZE: early_stage (annual budget <$250k), mid_size ($250k-$2M), established (>$2M)

NEED_TYPES hints (infer the implied need even when it is not named explicitly):
- "funding", "grant money", "we need money to..." -> funding
- "looking for partners / referrals / another org to work with" -> partners
- "donated goods / equipment / laptops / food / space" -> in_kind
- "more hands / people to help / mentors / tutors" -> volunteers
- "coaching / training / strategic planning / help running the org" -> capacity_building
- A group with NO 501(c)(3) / not tax-exempt / "we don't have our own nonprofit status yet"
  almost always implies a fiscal_sponsorship need (so they can receive grants) -> add fiscal_sponsorship.

GEOGRAPHY rules:
- If a specific city/town is stated (e.g. "Cambridge", "Lynn", "Worcester"), use that exact city/town name.
- Otherwise, if a broader region is clear, use one of: {", ".join(GEOGRAPHY_BUCKETS)}.
- If geography genuinely cannot be determined, you will flag it (see RULES).

Extract EXACTLY this JSON object (no extra keys):
{{
  "need_types": [list from NEED_TYPES],
  "focus_areas": [list from FOCUS_AREAS],
  "populations_served": [list from POPULATIONS],
  "geography": "a city/town or one of the regional buckets",
  "org_size": "early_stage | mid_size | established, or null if not inferable",
  "assets": [short strings: what they ALREADY have - staff, space, existing programs, partners],
  "urgency": "low | medium | high",
  "summary": "one-sentence restatement of their situation",
  "confidence": 0.0-1.0,
  "missing_fields": [field names that could NOT be determined and matter for matching]
}}

RULES:
- If geography cannot be determined, put "geography" in missing_fields and lower confidence.
- If need_types cannot be determined (you genuinely can't tell what they actually want), put "need_types" in missing_fields and lower confidence.
- confidence MUST be < 0.6 whenever any critical field (geography or need_types) is missing.
- confidence reflects whether the profile is accurate AND complete enough to match on - not just how much text you were given.
- Output ONLY valid JSON. No preamble, no explanation, no markdown code fences."""


def _build_user_prompt(
    raw_input: str, clarifications: list[dict] | None
) -> str:
    parts = [f"Nonprofit description:\n\"\"\"\n{raw_input}\n\"\"\""]
    if clarifications:
        qa_lines = "\n".join(
            f"- Q: {c.get('question', '')}\n  A: {c.get('answer', '')}"
            for c in clarifications
        )
        parts.append(
            "The following clarifying questions were already answered. "
            "Fold these answers into your profile and raise confidence "
            "accordingly:\n" + qa_lines
        )
    parts.append("Return the profile JSON now.")
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    """Remove accidental ```json ... ``` fences and surrounding whitespace."""
    t = text.strip()
    if t.startswith("```"):
        # drop the first fence line (``` or ```json) and any trailing fence
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _call_model(system: str, user: str) -> str:
    response = client.messages.create(
        model=FAST_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _coerce_result(profile: dict) -> dict:
    """Split the model's flat JSON into the {profile, confidence, missing_fields}
    return shape, with light defensive defaulting."""
    confidence = profile.pop("confidence", 0.0)
    missing_fields = profile.pop("missing_fields", [])
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not isinstance(missing_fields, list):
        missing_fields = [str(missing_fields)]
    return {
        "profile": profile,
        "confidence": confidence,
        "missing_fields": missing_fields,
    }


@weave.op()
def run_intake(
    raw_input: str, clarifications: list[dict] | None = None
) -> dict:
    """Extract a structured profile from free text.

    Args:
        raw_input: the nonprofit's plain-language description.
        clarifications: optional [{"question","answer"}] from a prior round.
            When present they are folded into the prompt so a re-run produces a
            better, higher-confidence profile (the Orchestrator's clarification
            loop hook).

    Returns:
        {"profile": {...}, "confidence": float, "missing_fields": [...]}
    """
    system = _build_system_prompt()
    user = _build_user_prompt(raw_input, clarifications)

    raw = _call_model(system, user)
    cleaned = _strip_fences(raw)

    try:
        profile = json.loads(cleaned)
    except json.JSONDecodeError:
        # ONE repair attempt: ask the model to return only valid JSON.
        repair_user = (
            "The following was supposed to be a single valid JSON object but "
            "could not be parsed. Return ONLY the corrected, valid JSON object "
            "with no preamble and no markdown fences:\n\n" + raw
        )
        repaired = _strip_fences(_call_model(system, repair_user))
        try:
            profile = json.loads(repaired)
        except json.JSONDecodeError:
            return {
                "profile": {},
                "confidence": 0.0,
                "missing_fields": ["parse_error"],
            }

    if not isinstance(profile, dict):
        return {
            "profile": {},
            "confidence": 0.0,
            "missing_fields": ["parse_error"],
        }

    return _coerce_result(profile)


def apply_intake(state: CaseState, result: dict) -> CaseState:
    """Write intake results into the shared state and log a trace line."""
    state.profile = result.get("profile")
    state.confidence = result.get("confidence")
    state.missing_fields = result.get("missing_fields", [])
    state.trace_log.append(
        f"Intake: confidence {state.confidence}, missing {state.missing_fields}"
    )
    return state


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    here = os.path.dirname(__file__)
    samples_path = os.path.join(here, "..", "data", "intake_samples.json")
    with open(samples_path) as f:
        samples = json.load(f)

    console = Console()
    table = Table(title="Intake agent — sample extraction")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("label")
    table.add_column("conf", justify="right")
    table.add_column("missing_fields", style="red")
    table.add_column("need_types")
    table.add_column("focus_areas")
    table.add_column("geography")

    for s in samples:
        result = run_intake(s["text"])
        prof = result["profile"]
        conf = result["confidence"]
        conf_style = "green" if conf >= 0.6 else "yellow"
        table.add_row(
            s["id"],
            s["label"],
            f"[{conf_style}]{conf:.2f}[/{conf_style}]",
            ", ".join(result["missing_fields"]) or "-",
            ", ".join(prof.get("need_types", []) or []) or "-",
            ", ".join(prof.get("focus_areas", []) or []) or "-",
            str(prof.get("geography") or "-"),
        )

    console.print(table)
