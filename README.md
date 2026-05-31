# Lighthouse — multi-agent nonprofit resource matching

Lighthouse is a multi-agent AI system that helps nonprofits find the resources
they need. A nonprofit describes its situation in plain language; the system
triages the input, matches it to relevant resources, funding, and partner
organizations, and returns a structured resource packet. An Orchestrator agent
routes the case between specialized agents, and every step is observable via
[W&B Weave](https://wandb.ai/site/weave).

Built for **Multi-Agent Orchestration Build Day** (AGI House, Cambridge MA).

## Architecture

Lighthouse coordinates four agents:

- **Orchestrator** — entry point; routes the case between the other agents and
  decides when to retry or finish.
- **Intake** — parses the nonprofit's plain-language description into a
  structured profile, scores confidence, and flags missing fields.
- **Matcher** — searches available resources/funding/partner orgs and proposes
  candidate matches.
- **Critic** — evaluates the matches and either accepts them or asks the
  Matcher to broaden the search and retry.

Shared state flows between agents via the `CaseState` model (`state.py`), and
all agent calls are traced in Weave for observability.

## Setup

Requires Python 3.10+ (Weave's minimum).

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env              # then edit .env and fill in your keys
#   ANTHROPIC_API_KEY  — https://console.anthropic.com/
#   WANDB_API_KEY      — https://wandb.ai/authorize

# 4. Verify everything works (Anthropic + Weave tracing)
python smoke_test_weave.py
```

The smoke test makes one trivial model call and prints a Weave trace URL. Open
it and confirm you see the trace before building further.

## Demo

_Demo video coming soon._

## License

MIT — see [LICENSE](LICENSE).
