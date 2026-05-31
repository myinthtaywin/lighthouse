"""Lighthouse entry point — Orchestrator.

Routes a single case through the agent pipeline and assembles a structured
resource packet, rendered as a clean CLI output.

Flow: Intake -> clarification loop (one round) -> Matcher -> Critic +
refinement loop (one retry) -> assembled packet.

Two self-correcting loops:
- Clarification: if Intake flags a missing critical field or low confidence,
  ask ONE clarifying question and re-run Intake with the answer folded in.
- Refinement: if the Critic judges the match set insufficient, re-run the
  Matcher in broaden mode ONCE, then the Critic accepts the best available set.

Weave is initialized once at the top of main() so the whole orchestrated run
shows up as a SINGLE parent trace (run_orchestrator) with every agent call
(run_intake, generate_clarifying_question, run_matcher, run_critic) nested
underneath — a key demo asset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import weave
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.critic import apply_critic, run_critic
from agents.intake import apply_intake, run_intake
from agents.matcher import apply_matches, run_matcher
from config import ANTHROPIC_API_KEY, FAST_MODEL, WEAVE_PROJECT
from state import CaseState

# Resolve data paths relative to this file so cwd doesn't matter.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_RESOURCES_PATH = _DATA_DIR / "resources.json"
_SAMPLES_PATH = _DATA_DIR / "intake_samples.json"

# Index the catalog by id so we can resolve resource_id -> link/name.
_RESOURCES_BY_ID = {r["id"]: r for r in json.loads(_RESOURCES_PATH.read_text())}

TOP_RESOURCES = 5

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Fields whose absence is worth a clarifying question before matching.
CRITICAL_FIELDS = {"geography", "need_types"}

# Templated fallbacks if the model call for a clarifying question fails.
_FALLBACK_QUESTIONS = {
    "geography": "What city, region, or area does your organization primarily serve?",
    "need_types": (
        "What kind of support are you mainly looking for — funding, partner "
        "organizations, volunteers, or something else?"
    ),
}
_GENERIC_FALLBACK_QUESTION = (
    "Could you tell us a bit more about your organization and what you need?"
)

# Used only when a clarifying question is raised in a non-interactive run with
# no pre-supplied answer — keeps us from crashing and avoids inventing facts.
_CANNED_ANSWER = "No additional information was provided."


@weave.op()
def generate_clarifying_question(profile: dict, missing_field: str) -> str:
    """Produce ONE short, specific question to fill a missing field.

    Uses FAST_MODEL. Falls back to a templated question (never crashes).
    """
    system = (
        "You are the Intake assistant for Lighthouse, a nonprofit resource "
        "matcher. Generate exactly ONE short, friendly, specific question that "
        "would fill in the missing piece of information described. Return ONLY "
        "the question text — no preamble, no quotes, no markdown."
    )
    user = (
        f"Here is what we understand so far about the nonprofit:\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f"The missing/unclear field we need to resolve is: '{missing_field}'.\n"
        f"Ask one question to clarify it."
    )
    try:
        response = client.messages.create(
            model=FAST_MODEL,
            max_tokens=100,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        question = response.content[0].text.strip()
        if question:
            return question
    except Exception:
        pass
    return _FALLBACK_QUESTIONS.get(missing_field, _GENERIC_FALLBACK_QUESTION)


def _obtain_answer(question: str, answers_queue: list[str]) -> tuple[str, str]:
    """Source an answer to a clarifying question.

    Order of preference: a pre-supplied answer (consumed in order), then an
    interactive prompt (only on a TTY), then a safe canned fallback.

    Returns (answer, source) where source is one of: pre-supplied, interactive,
    auto-filled.
    """
    if answers_queue:
        return answers_queue.pop(0), "pre-supplied"
    if sys.stdin.isatty():
        return input(f"\n  {question}\n  > ").strip(), "interactive"
    return _CANNED_ANSWER, "auto-filled"


@weave.op()
def run_orchestrator(raw_input: str, answers: list[str] | None = None) -> CaseState:
    """Run one case end-to-end and return the populated state.

    Order: Intake -> clarification loop (one round) -> Matcher ->
    Critic + refinement loop (one retry) -> assemble packet.

    Args:
        raw_input: the nonprofit's plain-language description.
        answers: optional pre-supplied clarification answers, consumed in order
            when a clarifying question is raised.
    """
    state = CaseState(raw_input=raw_input)
    answers_queue = list(answers or [])
    broadened = False  # set True if a refinement retry actually happened

    # --- 1. Intake ----------------------------------------------------------
    intake_result = run_intake(raw_input)
    state = apply_intake(state, intake_result)

    # --- 1b. CLARIFICATION LOOP (one round only) ----------------------------
    missing_critical = [
        f for f in (state.missing_fields or []) if f in CRITICAL_FIELDS
    ]
    low_confidence = state.confidence is not None and state.confidence < 0.6
    if missing_critical or low_confidence:
        # Pick the first flagged critical field, else default to need_types
        # (the case where low confidence fired without a specific flag).
        field = missing_critical[0] if missing_critical else "need_types"
        question = generate_clarifying_question(state.profile or {}, field)
        answer, source = _obtain_answer(question, answers_queue)
        if source == "auto-filled":
            state.trace_log.append(
                "Clarification: no answer supplied; used canned fallback"
            )
        state.clarifications.append({"question": question, "answer": answer})

        # Re-run intake WITH the clarification folded in.
        intake_result = run_intake(
            state.raw_input, clarifications=state.clarifications
        )
        state = apply_intake(state, intake_result)
        state.trace_log.append(
            f"Clarification: asked '{question}' -> re-ran intake, "
            f"confidence now {state.confidence}"
        )
    # Proceed regardless of the new confidence — one round is the cap.

    # --- 2. Matcher (first pass) -------------------------------------------
    match_result = run_matcher(state.profile)
    state = apply_matches(state, match_result)

    # --- 2b. CRITIC + REFINEMENT LOOP (exactly one retry) ------------------
    verdict = run_critic(state.profile, state.matches, already_broadened=False)
    state = apply_critic(state, verdict)

    if verdict.get("decision") == "broaden_and_retry":
        broadened = True
        state.trace_log.append(
            "Refinement: Critic requested broaden_and_retry -> "
            "re-running Matcher (broadened)"
        )
        match_result = run_matcher(
            state.profile,
            broaden=True,
            broaden_notes=verdict.get("broaden_notes", ""),
        )
        state = apply_matches(state, match_result)
        # Re-run Critic with the guard: already_broadened forces accept.
        verdict = run_critic(
            state.profile, state.matches, already_broadened=True
        )
        state = apply_critic(state, verdict)

    # `verdict` is now the FINAL verdict; state.matches the final set.

    # --- 3. Assemble final packet ------------------------------------------
    profile = state.profile or {}
    final_verdict = state.critic_verdict or {}
    recommended = []
    for m in state.matches[:TOP_RESOURCES]:
        resource = _RESOURCES_BY_ID.get(m.get("resource_id"), {})
        recommended.append(
            {
                "name": m.get("name") or resource.get("name"),
                "type": m.get("type") or resource.get("type"),
                "match_score": m.get("match_score"),
                "why_it_fits": m.get("reasoning"),
                "eligibility_concerns": m.get("eligibility_concerns", []),
                "link": resource.get("link"),
            }
        )

    state.final_packet = {
        "understood_as": {
            "summary": profile.get("summary"),
            "need_types": profile.get("need_types", []),
            "focus_areas": profile.get("focus_areas", []),
            "populations_served": profile.get("populations_served", []),
            "geography": profile.get("geography"),
            "org_size": profile.get("org_size"),
        },
        "confidence": state.confidence,
        "clarifications_asked": state.clarifications,
        "recommended_resources": recommended,
        "iterations": state.iterations,
        "broadened": broadened,
        "vetting": {
            "quality_score": final_verdict.get("quality_score"),
            "notes": final_verdict.get("reasons", []),
        },
    }

    state.trace_log.append(
        f"Orchestrator: packet assembled with {len(recommended)} resources"
    )
    return state


# --- CLI rendering ----------------------------------------------------------
def _confidence_style(confidence: float | None) -> str:
    if confidence is None:
        return "white"
    if confidence >= 0.75:
        return "green"
    if confidence >= 0.6:
        return "yellow"
    return "red"


def render(state: CaseState) -> None:
    console = Console()
    packet = state.final_packet or {}
    understood = packet.get("understood_as", {})
    conf = packet.get("confidence")
    conf_style = _confidence_style(conf)
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"

    # Vetting summary (from the final Critic verdict).
    vetting = packet.get("vetting", {})
    quality = vetting.get("quality_score")
    quality_str = str(quality) if quality is not None else "n/a"
    verdict = state.critic_verdict or {}
    decision = verdict.get("decision", "-")
    broadened_str = " (broadened)" if packet.get("broadened") else ""

    # --- Understood As panel ---
    lines = [
        f"[bold]Summary:[/bold] {understood.get('summary') or '-'}",
        "",
        f"[bold]Need types:[/bold] {', '.join(understood.get('need_types') or []) or '-'}",
        f"[bold]Focus areas:[/bold] {', '.join(understood.get('focus_areas') or []) or '-'}",
        f"[bold]Populations:[/bold] {', '.join(understood.get('populations_served') or []) or '-'}",
        f"[bold]Geography:[/bold] {understood.get('geography') or '-'}",
        f"[bold]Org size:[/bold] {understood.get('org_size') or '-'}",
        "",
        f"[bold]Confidence:[/bold] [{conf_style}]{conf_str}[/{conf_style}]",
        f"[bold]Vetting:[/bold] quality {quality_str} ({decision}){broadened_str}",
    ]
    console.print(
        Panel("\n".join(lines), title="Understood As", border_style="cyan")
    )

    # --- Recommended Resources table ---
    table = Table(title="Recommended Resources")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("name", style="bold")
    table.add_column("type")
    table.add_column("score", justify="right")
    table.add_column("why_it_fits")
    table.add_column("eligibility_concerns", style="yellow")
    table.add_column("link", style="blue")

    for i, r in enumerate(packet.get("recommended_resources", []), start=1):
        table.add_row(
            str(i),
            str(r.get("name") or "-"),
            str(r.get("type") or "-"),
            str(r.get("match_score") if r.get("match_score") is not None else "-"),
            str(r.get("why_it_fits") or "-"),
            "; ".join(r.get("eligibility_concerns") or []) or "-",
            str(r.get("link") or "-"),
        )
    console.print(table)

    # --- Routing Trace panel ---
    trace = "\n".join(f"• {line}" for line in state.trace_log) or "(no trace)"
    console.print(
        Panel(trace, title="Routing Trace", border_style="magenta")
    )


# --- CLI entry point --------------------------------------------------------
def _resolve_input(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    sample_id = args.sample or "sample_1"
    samples = {s["id"]: s for s in json.loads(_SAMPLES_PATH.read_text())}
    if sample_id not in samples:
        raise SystemExit(
            f"Unknown sample id '{sample_id}'. Available: "
            f"{', '.join(samples.keys())}"
        )
    return samples[sample_id]["text"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lighthouse — multi-agent nonprofit resource matching."
    )
    parser.add_argument(
        "--sample",
        help="Load a sample's text from data/intake_samples.json (e.g. sample_1).",
    )
    parser.add_argument(
        "--text",
        help="Use a raw free-text description directly (overrides --sample).",
    )
    parser.add_argument(
        "--answer",
        action="append",
        dest="answers",
        help=(
            "Pre-supplied answer to a clarifying question (repeatable; "
            "consumed in order)."
        ),
    )
    args = parser.parse_args()

    # Initialize Weave ONCE, before any agent runs, so the whole orchestrated
    # run is one parent trace with the agent calls nested underneath.
    weave_client = weave.init(WEAVE_PROJECT)

    text = _resolve_input(args)
    state = run_orchestrator(text, answers=args.answers)
    render(state)

    entity = getattr(weave_client, "entity", None)
    project = getattr(weave_client, "project", None)
    if entity and project:
        print(f"\nWeave trace: https://wandb.ai/{entity}/{project}/weave")


if __name__ == "__main__":
    main()
