"""Central configuration for Lighthouse.

Loads environment variables from a local .env file (via python-dotenv) and
exposes the model names, Weave project, and API keys used across the system.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# --- Models -----------------------------------------------------------------
# Used by the Orchestrator/Matcher/Critic for non-trivial reasoning.
REASONING_MODEL = "claude-sonnet-4-6"
# Used for cheap, fast calls (e.g. intake triage, smoke test).
FAST_MODEL = "claude-haiku-4-5-20251001"

# --- Observability ----------------------------------------------------------
WEAVE_PROJECT = "lighthouse"

# --- Secrets ----------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
        "Anthropic API key (get one at https://console.anthropic.com/)."
    )
