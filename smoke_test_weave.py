"""Smoke test: verifies the Anthropic API key, the W&B/Weave key, and that
Weave tracing fires.

Run after filling in .env:

    python smoke_test_weave.py

Expected: a short model reply printed to the console, plus a Weave trace URL.
Open the URL (or your Weave dashboard) and confirm the call was traced.
"""

import weave
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, FAST_MODEL, WEAVE_PROJECT

client = Anthropic(api_key=ANTHROPIC_API_KEY)


@weave.op()
def say_hello() -> str:
    """Make one trivial Anthropic call so Weave records a trace."""
    response = client.messages.create(
        model=FAST_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": "say hello"}],
    )
    return response.content[0].text


def main() -> None:
    # weave.init returns a client whose URL points at this project's traces.
    weave_client = weave.init(WEAVE_PROJECT)

    reply = say_hello()
    print("\nModel reply:")
    print(reply)

    print("\nWeave project URL:")
    print(weave_client.ui_url)
    print(
        "\nOpen the URL above and confirm you see a trace for `say_hello`."
    )


if __name__ == "__main__":
    main()
