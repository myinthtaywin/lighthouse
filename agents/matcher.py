"""Matcher agent.

Takes the structured profile from Intake and ranks the resources in
data/resources.json by fit, returning a reasoned justification and a match
score for each. This is the core reasoning agent, so it earns REASONING_MODEL.

At 30 resources we load the FULL catalog and reason over it in-context. No
embeddings / vector store: in-context reasoning over the whole set is the right
design at this scale and keeps the justifications grounded in real entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import weave
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, REASONING_MODEL
from state import CaseState

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Load the full catalog once, relative to this file so cwd doesn't matter.
_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "resources.json"
RESOURCES = json.loads(_CATALOG_PATH.read_text())
_VALID_IDS = {r["id"] for r in RESOURCES}

TOP_N = 6


def _build_system_prompt(broaden: bool) -> str:
    base = """You are the Matcher agent for Lighthouse, a system that matches nonprofits to resources, funding, and partner organizations.

You are given (1) a structured PROFILE of a nonprofit and (2) the full CATALOG of available resources. Rank the resources by how well they fit THIS profile, and justify each.

VOCABULARY ALIGNMENT — the profile and catalog share fields. Map across them:
- profile.need_types -> resource.type:
    funding -> "grant" OR "government_funding"
    partners -> "partner_org"
    in_kind -> "in_kind"
    volunteers -> "volunteer_network"
    capacity_building -> "capacity_building"
    fiscal_sponsorship -> "fiscal_sponsor"
- profile.focus_areas <-> resource.focus_areas (a resource with focus "any" fits any focus)
- profile.populations_served <-> resource.populations_served ("general" serves everyone)
- profile.geography vs resource.geography — treat these as NESTED scopes, not exact strings:
    a specific city/town is inside "Greater Boston" is inside "Massachusetts" is inside "New England" is inside "national".
    A resource is geographically reachable if its scope CONTAINS the profile's location
    (e.g. a "national" or "Massachusetts" resource is reachable by a Cambridge nonprofit;
    a "Greater Boston" resource is NOT reachable by a nonprofit in western Massachusetts).
- profile.org_size vs resource.org_size_fit — "any" fits everyone; otherwise prefer matching bands.

SCORING:
- match_score is an integer 0-100 reflecting overall fit for THIS specific profile.
- Reward alignment on need_type, focus_areas, population, geography scope, and size fit.
- A resource that aligns on need + focus + is geographically reachable + size-appropriate should score high (80+).
- Broad resources (focus "any", geography "national") are genuine fallbacks but should
  generally score below a well-targeted resource that aligns specifically.
"""

    if broaden:
        mode = """
MODE: BROADEN. The first pass was judged insufficient, so RELAX soft constraints:
- Widen geography outward (city -> Greater Boston -> Massachusetts -> New England -> national) and include resources at wider scopes.
- Include adjacent / related focus areas, not just exact matches.
- Be more generous on org_size fit.
Still exclude resources with a HARD eligibility gate the profile clearly fails (e.g. a veterans-only resource for a non-veteran population). Use the broaden notes below to address WHY the first pass fell short.
"""
    else:
        mode = """
MODE: STANDARD. Apply eligibility sensibly. A resource with a hard population, geography,
or size gate that the profile clearly fails should rank low or be excluded entirely.
"""

    output = f"""
Return ONLY valid JSON (no preamble, no markdown fences): an object with one key "matches",
whose value is an array. Each item:
{{
  "resource_id": "must exactly match an id in the catalog",
  "name": "resource name",
  "type": "resource type",
  "match_score": integer 0-100,
  "reasoning": "one or two sentences on WHY it fits THIS profile specifically",
  "eligibility_concerns": ["short strings; anything that might disqualify them", "empty list if clean"]
}}
Return the top {TOP_N} resources by match_score, sorted descending. Output ONLY the JSON object."""

    return base + mode + output


def _build_user_prompt(
    profile: dict, broaden: bool, broaden_notes: str | None
) -> str:
    parts = [
        "PROFILE:\n" + json.dumps(profile, indent=2),
        "CATALOG:\n" + json.dumps(RESOURCES, indent=2),
    ]
    if broaden and broaden_notes:
        parts.append(
            "BROADEN NOTES (the Critic's reasons the first pass was "
            "insufficient — address these):\n" + broaden_notes
        )
    parts.append(f"Return the top {TOP_N} matches as JSON now.")
    return "\n\n".join(parts)


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
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _extract_matches(parsed) -> list:
    """Accept either {"matches": [...]} or a bare array."""
    if isinstance(parsed, dict):
        return parsed.get("matches", [])
    if isinstance(parsed, list):
        return parsed
    return []


def _clean_matches(matches: list) -> list:
    """Drop hallucinated ids, keep valid items, sort desc, cap at TOP_N."""
    cleaned = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        rid = m.get("resource_id")
        if rid not in _VALID_IDS:
            continue  # drop hallucinated / unknown ids
        try:
            m["match_score"] = int(m.get("match_score", 0))
        except (TypeError, ValueError):
            m["match_score"] = 0
        if not isinstance(m.get("eligibility_concerns"), list):
            m["eligibility_concerns"] = []
        cleaned.append(m)
    cleaned.sort(key=lambda x: x["match_score"], reverse=True)
    return cleaned[:TOP_N]


@weave.op()
def run_matcher(
    profile: dict,
    broaden: bool = False,
    broaden_notes: str | None = None,
) -> dict:
    """Rank catalog resources against a profile.

    Args:
        profile: structured profile produced by the Intake agent.
        broaden: when True, relax soft constraints (geography/focus/size) for a
            more generous retry.
        broaden_notes: the Critic's reasons the first pass was insufficient,
            folded into the prompt so the retry is informed. (Matcher<->Critic
            refinement loop hook.)

    Returns:
        {"matches": [...up to top 6...], "broadened": broaden}
    """
    system = _build_system_prompt(broaden)
    user = _build_user_prompt(profile, broaden, broaden_notes)

    raw = _call_model(system, user)
    cleaned_text = _strip_fences(raw)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        repair_user = (
            "The following was supposed to be a single valid JSON object with a "
            "\"matches\" array but could not be parsed. Return ONLY the "
            "corrected, valid JSON with no preamble and no markdown fences:\n\n"
            + raw
        )
        repaired = _strip_fences(_call_model(system, repair_user))
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return {"matches": [], "error": "parse_error"}

    matches = _clean_matches(_extract_matches(parsed))
    return {"matches": matches, "broadened": broaden}


def apply_matches(state: CaseState, result: dict) -> CaseState:
    """Write matches into the shared state, bump iterations, log a trace line."""
    matches = result.get("matches", [])
    state.matches = matches
    state.iterations += 1
    top_score = matches[0]["match_score"] if matches else 0
    state.trace_log.append(
        f"Matcher: {len(matches)} matches, top score {top_score}, "
        f"broadened={result.get('broadened', False)}"
    )
    return state


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    from agents.intake import run_intake

    samples_path = (
        Path(__file__).resolve().parent.parent / "data" / "intake_samples.json"
    )
    samples = {s["id"]: s for s in json.loads(samples_path.read_text())}

    console = Console()

    for sid in ("sample_1", "sample_2"):
        sample = samples[sid]
        intake = run_intake(sample["text"])
        profile = intake["profile"]

        console.rule(f"[bold]{sid} ({sample['label']})")
        console.print(
            f"[dim]Profile summary:[/dim] {profile.get('summary', '-')}"
        )
        console.print(
            f"[dim]need_types:[/dim] {profile.get('need_types')}  "
            f"[dim]focus:[/dim] {profile.get('focus_areas')}  "
            f"[dim]geo:[/dim] {profile.get('geography')}  "
            f"[dim]size:[/dim] {profile.get('org_size')}  "
            f"[dim]intake conf:[/dim] {intake['confidence']:.2f}"
        )

        result = run_matcher(profile)

        table = Table(title=f"Top {TOP_N} matches — {sid}")
        table.add_column("resource_id", style="cyan", no_wrap=True)
        table.add_column("name")
        table.add_column("type")
        table.add_column("score", justify="right")
        table.add_column("eligibility_concerns", style="yellow")

        for m in result["matches"]:
            table.add_row(
                m.get("resource_id", "-"),
                m.get("name", "-"),
                m.get("type", "-"),
                str(m.get("match_score", "-")),
                "; ".join(m.get("eligibility_concerns", [])) or "-",
            )
        console.print(table)
        console.print()
