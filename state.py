"""Shared state model passed between agents during a single case run."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseState(BaseModel):
    """The evolving state of one nonprofit's resource-matching case.

    Agents read from and write to this object as the Orchestrator routes the
    case through Intake -> Matcher -> Critic (with possible retries).
    """

    # Raw, plain-language description provided by the nonprofit.
    raw_input: str

    # Structured profile extracted by the Intake agent.
    profile: dict | None = None
    confidence: float | None = None
    missing_fields: list[str] = Field(default_factory=list)

    # Clarifying Q&A gathered when intake confidence is low.
    # Each item: {"question": str, "answer": str}
    clarifications: list[dict] = Field(default_factory=list)

    # Candidate resources/funding/partner orgs found by the Matcher agent.
    matches: list[dict] = Field(default_factory=list)

    # Critic agent's evaluation of the matches.
    # e.g. {"decision": "accept" | "broaden_and_retry", "reasons": [...]}
    critic_verdict: dict | None = None

    # Number of match/critic iterations performed so far.
    iterations: int = 0

    # Final structured resource packet returned to the user.
    final_packet: dict | None = None

    # Human-readable routing decisions, surfaced in the CLI.
    trace_log: list[str] = Field(default_factory=list)
