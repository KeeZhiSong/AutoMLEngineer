"""
llm.py -- per-role model routing, provider-agnostic.

WHY. Five agent roles were all running gpt-4o, which over-pays for some and
under-powers the one that is actually failing. Measured:

  gpt-4o-mini   could not write a correct grouped gradient at all -- array-shape
                crashes on every attempt.
  gpt-4o        names the right problem in 11 of 16 scored cycles, and 0 of 16
                implementations cleared the accept margin.

So the bottleneck is narrow: writing correct vectorised numpy for grouped
ranking objectives. That is a verifiable coding task, not a reasoning-about-the-
problem task, and it deserves the strongest model while the Reflector -- whose
keep/revert verdict is ARITHMETIC and whose only job is to write prose -- does
not.

Roles, and what they actually need:

    analyst     pick 3 tools, report numbers            cheap
    classifier  name the problem from measurements      mid
    inventor    propose interventions, may pull papers  strong reasoning
    coder       write correct numpy                     STRONGEST  <- bottleneck
    reflector   explain a decision already made         cheap

Set per role via env, falling back to the run's --model:

    export AMRA_MODEL_CODER=o3
    export AMRA_MODEL_ANALYST=gpt-4o-mini

Anthropic models route automatically when the id starts with "claude-", provided
ANTHROPIC_API_KEY is set and the SDK is installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("amra.llm")

ROLES = ("analyst", "classifier", "inventor", "coder", "reflector")

# Sensible defaults if nothing is configured: spend on the coder, save elsewhere.
DEFAULT_ROUTING = {
    "analyst": "gpt-4o-mini",
    "classifier": "gpt-4o",
    "inventor": "gpt-4o",
    "coder": "gpt-4o",
    "reflector": "gpt-4o-mini",
}


@dataclass
class Usage:
    """Per-role token accounting. Feasibility is 15% of the score, so this is
    reported, not merely collected."""
    by_role: dict = field(default_factory=dict)

    def add(self, role: str, tokens: int) -> None:
        self.by_role[role] = self.by_role.get(role, 0) + int(tokens)

    def total(self) -> int:
        return sum(self.by_role.values())

    def report(self) -> str:
        if not self.by_role:
            return "(no LLM usage recorded)"
        rows = sorted(self.by_role.items(), key=lambda kv: -kv[1])
        w = max(len(r) for r, _ in rows)
        return "\n".join(f"  {r:<{w}}  {t:>9,}" for r, t in rows) + \
               f"\n  {'TOTAL':<{w}}  {self.total():>9,}"


USAGE = Usage()


def model_for(role: str, fallback: str) -> str:
    """Resolve the model id for a role: env override, then default, then the
    run-level --model."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    env = os.environ.get(f"AMRA_MODEL_{role.upper()}")
    if env:
        return env
    if os.environ.get("AMRA_MODEL_UNIFORM"):
        return os.environ["AMRA_MODEL_UNIFORM"]
    return DEFAULT_ROUTING.get(role) or fallback


def routing_table(fallback: str) -> str:
    return "\n".join(f"  {r:<11} {model_for(r, fallback)}" for r in ROLES)


def complete(role: str, prompt: str, *, fallback_model: str,
             max_tokens: int = 1000, json_mode: bool = False) -> tuple[str, int]:
    """One completion for one role. Returns (text, tokens_used).

    Provider is chosen from the model id so a role can be pointed at any vendor
    without touching call sites.
    """
    model = model_for(role, fallback_model)

    if model.startswith("claude-"):
        text, tokens = _anthropic(model, prompt, max_tokens, json_mode)
    else:
        text, tokens = _openai(model, prompt, max_tokens, json_mode)

    USAGE.add(role, tokens)
    return text, tokens


def _openai(model: str, prompt: str, max_tokens: int, json_mode: bool):
    from openai import OpenAI
    kw = {"model": model, "max_tokens": max_tokens,
          "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    # Reasoning models reject max_tokens/temperature and use their own budget.
    if model.startswith(("o1", "o3", "o4")):
        kw.pop("max_tokens", None)
        kw["max_completion_tokens"] = max_tokens
    r = OpenAI().chat.completions.create(**kw)
    return (r.choices[0].message.content or ""), \
           (r.usage.prompt_tokens + r.usage.completion_tokens)


def _anthropic(model: str, prompt: str, max_tokens: int, json_mode: bool):
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            f"model {model!r} needs the anthropic SDK: pip install anthropic") from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(f"model {model!r} needs ANTHROPIC_API_KEY")
    content = prompt
    if json_mode:
        # Anthropic has no json_object mode; ask explicitly instead.
        content += "\n\nRespond with ONLY a single valid JSON object, no prose."
    r = Anthropic().messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if json_mode:
        # strip a markdown fence if the model added one
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1].rsplit("```", 1)[0]
            text = t.strip()
    return text, (r.usage.input_tokens + r.usage.output_tokens)
