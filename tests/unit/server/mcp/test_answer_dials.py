"""Resolution of the get_answer deployment dials.

The excerpt size and the synthesis budget resolve per repo: process env,
then ``config.yaml`` (``answer_excerpt_chars`` / ``answer_max_tokens``),
then the defaults in ``config.py`` — clamped, because a typo'd value must
neither starve the prompt nor run unbounded. The synthesis word target
tracks the resolved budget: raising max_tokens without moving the
instruction would only buy silent headroom.
"""

import pytest

from repowise.server.mcp_server.tool_answer import dials
from repowise.server.mcp_server.tool_answer.config import (
    _GATED_EXCERPT_CHARS,
    _SYNTHESIS_MAX_TOKENS,
    _SYNTHESIS_TEMPERATURE,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("REPOWISE_ANSWER_EXCERPT_CHARS", raising=False)
    monkeypatch.delenv("REPOWISE_ANSWER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("REPOWISE_ANSWER_TEMPERATURE", raising=False)
    monkeypatch.delenv("REPOWISE_ANSWER_TOP_P", raising=False)
    monkeypatch.delenv("REPOWISE_ANSWER_TOP_K", raising=False)


def _write_config(tmp_path, **keys):
    import yaml

    d = tmp_path / ".repowise"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(keys), encoding="utf-8")
    return tmp_path


def test_defaults_are_the_config_constants():
    assert dials.answer_excerpt_chars(None) == _GATED_EXCERPT_CHARS
    assert dials.answer_max_tokens(None) == _SYNTHESIS_MAX_TOKENS
    assert dials.answer_sampling_parameters(None).configured() == {
        "temperature": _SYNTHESIS_TEMPERATURE
    }


def test_env_overrides_and_clamps(monkeypatch):
    monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "4500")
    assert dials.answer_excerpt_chars(None) == 4500
    monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "50")
    assert dials.answer_excerpt_chars(None) == 200  # floor
    monkeypatch.setenv("REPOWISE_ANSWER_MAX_TOKENS", "99999")
    assert dials.answer_max_tokens(None) == 8192  # ceiling


def test_garbage_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "plenty")
    assert dials.answer_excerpt_chars(None) == _GATED_EXCERPT_CHARS


def test_config_yaml_keys_are_honored(tmp_path):
    repo = _write_config(
        tmp_path,
        answer_excerpt_chars=3000,
        answer_max_tokens=2048,
        answer_temperature=1.0,
        answer_top_p=0.95,
        answer_top_k=64,
    )
    assert dials.answer_excerpt_chars(repo) == 3000
    assert dials.answer_max_tokens(repo) == 2048
    assert dials.answer_sampling_parameters(repo).configured() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }


def test_env_outranks_config_yaml(tmp_path, monkeypatch):
    repo = _write_config(tmp_path, answer_excerpt_chars=3000, answer_temperature=0.4)
    monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "6000")
    monkeypatch.setenv("REPOWISE_ANSWER_TEMPERATURE", "0.8")
    assert dials.answer_excerpt_chars(repo) == 6000
    assert dials.answer_sampling_parameters(repo).temperature == 0.8


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [
        pytest.param("REPOWISE_ANSWER_TEMPERATURE", "-1", id="temperature"),
        pytest.param("REPOWISE_ANSWER_TOP_P", "1.1", id="top-p"),
        pytest.param("REPOWISE_ANSWER_TOP_K", "2.5", id="top-k"),
    ],
)
def test_invalid_answer_sampling_is_rejected_exactly(monkeypatch, env_name, bad_value):
    monkeypatch.setenv(env_name, bad_value)

    with pytest.raises(ValueError):
        dials.answer_sampling_parameters(None)


def test_word_target_tracks_the_budget():
    assert "150–400 words" in dials.synthesis_system_prompt(_SYNTHESIS_MAX_TOKENS)
    assert "150–800 words" in dials.synthesis_system_prompt(2048)
    # Ceiling: past ~1200 words an answer becomes a report.
    assert "150–1200 words" in dials.synthesis_system_prompt(8192)
    # A budget below the default never shrinks the target below its floor.
    assert "150–400 words" in dials.synthesis_system_prompt(256)


def test_user_prompt_carries_the_scaled_target():
    prompt = dials.synthesis_user_prompt("q", 5, "ctx", max_tokens=2048)
    assert "150–800 words" in prompt
    assert "ctx" in prompt and "q" in prompt
