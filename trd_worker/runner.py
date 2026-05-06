"""
Job runner — Phase 2 STUB.

This does NOT run real inference. It sleeps for a plausible duration and
returns a canned response so we can prove the register → poll → submit
lifecycle end-to-end.

Phase 3 swaps this out for real llama.cpp / vllm execution.

Returns: (result_text, duration_ms, tokens_generated)
"""

from __future__ import annotations
import random
import time


# Canned templates that loosely resemble what the agents will eventually produce
_HTML_STUB = """<section class="hero">
  <h1>{title}</h1>
  <p>{tagline}</p>
  <a href="#contact" class="btn">Get Started</a>
</section>"""

_TITLES = [
    "Your Vision, Beautifully Built",
    "Find Your Flow",
    "Crafted For You",
    "Where Quality Meets Care",
    "Modern Solutions, Trusted Service",
]
_TAGLINES = [
    "Professional. Local. Trusted by hundreds.",
    "Where every detail matters.",
    "Simple, fast, made for you.",
    "Excellence is our standard.",
    "Bringing your ideas to life.",
]


def run_job_stub(
    prompt: str, model_name: str, max_tokens: int, timeout_sec: int
) -> tuple[str, int, int]:
    """
    Simulate inference. Sleeps 2-8 seconds, returns a canned HTML snippet.

    Returns:
        (result_text, duration_ms, tokens_generated)
    """
    start = time.monotonic()

    # Plausible-feeling delay — varied to make stats look real
    sleep_sec = random.uniform(2.0, 8.0)
    # Cap by timeout if it's tight
    sleep_sec = min(sleep_sec, max(1.0, timeout_sec - 2))
    time.sleep(sleep_sec)

    # Generate canned response loosely based on prompt length
    title = random.choice(_TITLES)
    tagline = random.choice(_TAGLINES)
    result = _HTML_STUB.format(title=title, tagline=tagline)

    # Fake token count — roughly len/4 chars per token, capped at max_tokens
    tokens = min(max_tokens, max(20, len(result) // 4))

    duration_ms = int((time.monotonic() - start) * 1000)
    return result, duration_ms, tokens
