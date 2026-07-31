"""Resolution of the get_context token-budget dial.

The budget resolves per call — process env, then the repo's config.yaml
(``context_token_budget``), then ``TOKEN_BUDGET`` — and whatever it resolves
to stays clamped under the live MCP host cap, so a raised deployment budget
can never trip the host's reject-with-isError path.
"""

import pytest

from repowise.server.mcp_server import _state
from repowise.server.mcp_server._budget import budgeter


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("REPOWISE_CONTEXT_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("MAX_MCP_OUTPUT_TOKENS", raising=False)
    monkeypatch.setattr(_state, "_repo_path", None, raising=False)


def test_default_budget_unchanged():
    assert budgeter.configured_token_budget() == budgeter.TOKEN_BUDGET
    assert budgeter.effective_char_budget() == budgeter.CHAR_BUDGET


def test_env_raises_the_budget(monkeypatch):
    monkeypatch.setenv("REPOWISE_CONTEXT_TOKEN_BUDGET", "12000")
    assert budgeter.configured_token_budget() == 12000
    assert budgeter.effective_char_budget() == 12000 * budgeter.CHARS_PER_TOKEN


def test_config_yaml_key_is_honored(monkeypatch, tmp_path):
    import yaml

    d = tmp_path / ".repowise"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"context_token_budget": 10000}))
    monkeypatch.setattr(_state, "_repo_path", str(tmp_path), raising=False)
    assert budgeter.configured_token_budget() == 10000


def test_env_outranks_config(monkeypatch, tmp_path):
    import yaml

    d = tmp_path / ".repowise"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"context_token_budget": 10000}))
    monkeypatch.setattr(_state, "_repo_path", str(tmp_path), raising=False)
    monkeypatch.setenv("REPOWISE_CONTEXT_TOKEN_BUDGET", "14000")
    assert budgeter.configured_token_budget() == 14000


def test_bounds_clamp_a_typo(monkeypatch):
    monkeypatch.setenv("REPOWISE_CONTEXT_TOKEN_BUDGET", "999999")
    assert budgeter.configured_token_budget() == 25000
    monkeypatch.setenv("REPOWISE_CONTEXT_TOKEN_BUDGET", "3")
    assert budgeter.configured_token_budget() == 1000


def test_host_cap_still_clamps_a_raised_budget(monkeypatch):
    """A raised budget must never outrun the host's reject line."""
    monkeypatch.setenv("REPOWISE_CONTEXT_TOKEN_BUDGET", "25000")
    monkeypatch.setenv("MAX_MCP_OUTPUT_TOKENS", "10000")
    host_ceiling = int(10000 * budgeter.HOST_CAP_BUDGET_FRACTION) * budgeter.CHARS_PER_TOKEN
    assert budgeter.effective_char_budget() == host_ceiling
