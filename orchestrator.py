"""Lighthouse entry point — Orchestrator.

Routes a single case through the agent pipeline and assembles a structured
resource packet, rendered as a clean CLI output.

This is the skeleton: Intake -> Matcher -> assembled packet, with NO loops yet.
The clarification loop and the Critic + refinement loop are slotted in later
(Prompt 6) at the clearly marked placeholders, without refactoring this shape.

Weave is initialized once at the top of main() so the whole orchestrated run
shows up as a SINGLE parent trace (run_orchestrator) with the agent calls
(run_intake, run_matcher) nested underneath — a key demo asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import weave
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.intake import apply_intake, run_intake
from agents.matcher import apply_matches, run_matcher
from config import WEAVE_PROJECT
from state import CaseState

# Resolve data paths relative to this file so cwd doesn't matter.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_RESOURCES_PATH = _DATA_DIR / "resources.json"
_SAMPLES_PATH = _DATA_DIR / "intake_samples.json"

# Index the catalog by id so we can resolve resource_id -> link/name.
_RESOURCES_BY_ID = {r["id"]: r for r in json.loads(_RESOURCES_PATH.read_text())}

TOP_RESOURCES = 5


@weave.op()
def run_orchestrator(raw_input: str) -> CaseState:
    """Run one case end-to-end and return the populated state.

    Order: Intake -> (clarification loop later) -> Matcher ->
    (Critic + refinement loop later) -> assemble packet.
    """
    state = CaseState(raw_input=raw_input)

    # --- 1. Intake ----------------------------------------------------------
    intake_result = run_intake(raw_input)
    state = apply_intake(state, intake_result)

    # CLARIFICATION LOOP GOES HERE (Prompt 6)
    # For now we proceed regardless of confidence / missing_fields.

    # --- 2. Matcher ---------------------------------------------------------
    match_result = run_matcher(state.profile)
    state = apply_matches(state, match_result)

    # CRITIC + REFINEMENT LOOP GOES HERE (Prompt 6)

    # --- 3. Assemble final packet ------------------------------------------
    profile = state.profile or {}
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
        "clarifications_asked": state.clarifications,  # empty for now
        "recommended_resources": recommended,
        "iterations": state.iterations,
        "broadened": bool(match_result.get("broadened", False)),
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
    args = parser.parse_args()

    # Initialize Weave ONCE, before any agent runs, so the whole orchestrated
    # run is one parent trace with the agent calls nested underneath.
    weave_client = weave.init(WEAVE_PROJECT)

    text = _resolve_input(args)
    state = run_orchestrator(text)
    render(state)

    entity = getattr(weave_client, "entity", None)
    project = getattr(weave_client, "project", None)
    if entity and project:
        print(f"\nWeave trace: https://wandb.ai/{entity}/{project}/weave")


if __name__ == "__main__":
    main()
