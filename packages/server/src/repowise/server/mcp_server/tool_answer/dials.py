"""Per-repo deployment dials for get_answer.

``config.py`` holds get_answer's constants: repo-agnostic properties of the
retrieval and synthesis design. Two of those constants are better read as
defaults, because the cost they price differs by deployment: the page-excerpt
size and the synthesis output budget are billed spend against a hosted API
(5 hits x 1500 chars measures at +24% of the tool's own synthesis spend) but
prefill seconds against a local model. This module resolves their per-repo
values — process env first, then ``.repowise/config.yaml``, then the default,
the same precedence every other repo setting resolves with — and builds the
synthesis prompts whose word target must track the resolved budget.
"""

import logging
import os
from pathlib import Path

from repowise.server.mcp_server.tool_answer.config import (
    _GATED_EXCERPT_CHARS,
    _SYNTHESIS_MAX_TOKENS,
    _SYSTEM_PROMPT_TEMPLATE,
    _USER_TEMPLATE,
)

_log = logging.getLogger("repowise.mcp.answer")

_ANSWER_EXCERPT_CHARS_ENV = "REPOWISE_ANSWER_EXCERPT_CHARS"
_ANSWER_EXCERPT_CHARS_CONFIG_KEY = "answer_excerpt_chars"
_ANSWER_EXCERPT_CHARS_BOUNDS = (200, 20_000)
_ANSWER_MAX_TOKENS_ENV = "REPOWISE_ANSWER_MAX_TOKENS"
_ANSWER_MAX_TOKENS_CONFIG_KEY = "answer_max_tokens"
_ANSWER_MAX_TOKENS_BOUNDS = (256, 8_192)

# The word target scales with the token budget — raising max_tokens without
# moving the instruction just buys silent headroom — but sublinearly capped:
# past ~1200 words a grounded answer stops being an answer and becomes a
# report the caller did not ask for.
_ANSWER_WORDS_LOW = 150
_ANSWER_WORDS_HIGH_DEFAULT = 400
_ANSWER_WORDS_HIGH_CEILING = 1_200


def _config_yaml_int(repo_path: Path | str | None, key: str) -> int | None:
    """An int-valued key from ``<repo>/.repowise/config.yaml``, or None."""
    if repo_path is None:
        return None
    config_path = Path(str(repo_path)) / ".repowise" / "config.yaml"
    try:
        if config_path.is_file():
            import yaml

            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                raw = data.get(key)
                if raw is not None and not isinstance(raw, bool):
                    return int(raw)
    except Exception:
        _log.debug("Failed to read %s from %s", key, config_path, exc_info=True)
    return None


def _resolve_dial(
    env_name: str,
    config_key: str,
    default: int,
    bounds: tuple[int, int],
    repo_path: Path | str | None,
) -> int:
    raw_env = os.environ.get(env_name, "").strip()
    value: int | None = None
    if raw_env:
        try:
            value = int(raw_env)
        except ValueError:
            _log.warning("%s=%r is not an integer; ignoring", env_name, raw_env)
    if value is None:
        value = _config_yaml_int(repo_path, config_key)
    if value is None:
        return default
    low, high = bounds
    return max(low, min(high, value))


def answer_excerpt_chars(repo_path: Path | str | None = None) -> int:
    """How many chars of page content each top hit contributes.

    The excerpt fetch and the prompt formatter must both read this one
    resolution: an excerpt fetched at one size and truncated at another
    silently discards content the database round-trip paid for.
    """
    return _resolve_dial(
        _ANSWER_EXCERPT_CHARS_ENV,
        _ANSWER_EXCERPT_CHARS_CONFIG_KEY,
        _GATED_EXCERPT_CHARS,
        _ANSWER_EXCERPT_CHARS_BOUNDS,
        repo_path,
    )


def answer_max_tokens(repo_path: Path | str | None = None) -> int:
    """The synthesis output budget for one get_answer call."""
    return _resolve_dial(
        _ANSWER_MAX_TOKENS_ENV,
        _ANSWER_MAX_TOKENS_CONFIG_KEY,
        _SYNTHESIS_MAX_TOKENS,
        _ANSWER_MAX_TOKENS_BOUNDS,
        repo_path,
    )


def _answer_words_high(max_tokens: int) -> int:
    scaled = round(_ANSWER_WORDS_HIGH_DEFAULT * max_tokens / _SYNTHESIS_MAX_TOKENS)
    return max(_ANSWER_WORDS_HIGH_DEFAULT, min(_ANSWER_WORDS_HIGH_CEILING, scaled))


def synthesis_system_prompt(max_tokens: int = _SYNTHESIS_MAX_TOKENS) -> str:
    """The synthesis system prompt, word target matched to the token budget."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        words_low=_ANSWER_WORDS_LOW, words_high=_answer_words_high(max_tokens)
    )


def synthesis_user_prompt(
    question: str, n: int, context: str, max_tokens: int = _SYNTHESIS_MAX_TOKENS
) -> str:
    """The synthesis user prompt, word target matched to the token budget."""
    return _USER_TEMPLATE.format(
        question=question,
        n=n,
        context=context,
        words_low=_ANSWER_WORDS_LOW,
        words_high=_answer_words_high(max_tokens),
    )
