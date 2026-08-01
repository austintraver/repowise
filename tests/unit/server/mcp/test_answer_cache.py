"""Answer-cache lifecycle: upsert on re-synthesis, commit/TTL invalidation.

Regression coverage for the silent-failure mode where the cache write was a
plain INSERT under ``suppress(Exception)``: every bypass-and-resynthesize
round (hedged rows, schema bumps) violated ``uq_answer_cache_q`` and failed
silently, so a bad cached row was permanent. The write is now a
delete-then-insert upsert, and reads invalidate on indexed-commit change or
hard TTL.
"""

from __future__ import annotations

import json as _json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from repowise.core.persistence.models import AnswerCache, Repository

QUESTION = "how does the auth service work"

# Top score >= 3.0 with an absolute gap >= 0.5 → dominant → synthesis runs.
_HITS = [
    {"page_id": "file_page:src/auth/service.py", "score": 5.0},
    {"page_id": "file_page:src/auth/middleware.py", "score": 4.0},
]


def _patch_retrieval(monkeypatch, answer_mod):
    async def _fake_retrieve(question, ctx):
        return [dict(h) for h in _HITS]

    async def _fake_hydrate(hits, ctx, *, scope=None):
        for h in hits:
            h["target_path"] = h["page_id"].removeprefix("file_page:")
            h["title"] = h["target_path"]
            h["summary"] = ""
            h["snippet"] = ""
            h["page_type"] = "file_page"
        return hits

    monkeypatch.setattr(answer_mod, "_hybrid_retrieve", _fake_retrieve)
    monkeypatch.setattr(answer_mod, "_hydrate_hits", _fake_hydrate)


class _Provider:
    provider_name = "mock"
    model_name = "mock-1"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    async def generate(self, **kwargs):
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return SimpleNamespace(content=self.content)


def _patch_provider(monkeypatch, answer_mod, provider: _Provider) -> None:
    monkeypatch.setattr(answer_mod, "_resolve_provider_for_answer", lambda _p: provider)


async def _cache_rows(factory) -> list[AnswerCache]:
    from repowise.core.persistence.database import get_session

    async with get_session(factory) as session:
        res = await session.execute(select(AnswerCache))
        return list(res.scalars().all())


@pytest.mark.asyncio
async def test_hedged_row_is_upgraded_on_resynthesis(setup_mcp, factory, monkeypatch):
    """Hedged cached answer → bypass → re-synthesis → row UPGRADED, not orphaned."""
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)

    hedged = _Provider("The provided excerpts do not contain the implementation details.")
    _patch_provider(monkeypatch, answer_mod, hedged)
    first = await get_answer(QUESTION)
    assert first["confidence"] == "low"

    rows = await _cache_rows(factory)
    assert len(rows) == 1
    assert "do not contain" in _json.loads(rows[0].payload_json)["answer"]

    direct = _Provider("Auth flows through src/auth/service.py via AuthService.check().")
    _patch_provider(monkeypatch, answer_mod, direct)
    second = await get_answer(QUESTION)
    assert direct.calls == 1, "hedged cache entry must be bypassed, not returned"
    assert "AuthService.check" in second["answer"]

    rows = await _cache_rows(factory)
    assert len(rows) == 1, "upsert must replace the hedged row, not duplicate or fail"
    upgraded = _json.loads(rows[0].payload_json)
    assert "AuthService.check" in upgraded["answer"], "row must carry the upgraded answer"

    # Third call: the upgraded row is a normal cache hit — no synthesis.
    third = await get_answer(QUESTION)
    assert direct.calls == 1, "upgraded row must serve from cache"
    assert third["_meta"].get("cached") is True
    assert "_indexed_commit" not in third


@pytest.mark.asyncio
async def test_cached_empty_answer_row_is_bypassed(setup_mcp, factory, session, monkeypatch):
    """A legacy cached gated payload (empty answer) never serves from cache.

    Older versions cached the gated best_guesses payload, pinning a retrieval
    miss until TTL and hiding every later improvement to the miss path. The
    write side no longer stores empty answers; the read side must retire the
    rows that predate that fix.
    """
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer
    from repowise.server.mcp_server.tool_answer.answer import _hash_question
    from repowise.server.mcp_server.tool_answer.config import _ANSWER_SCHEMA_VERSION

    _patch_retrieval(monkeypatch, answer_mod)

    res = await session.execute(select(Repository))
    repo = res.scalars().first()
    # Legacy row: current schema version (so the schema gate passes) but the
    # old empty-answer gated shape.
    session.add(AnswerCache(
        repository_id=repo.id,
        question_hash=_hash_question(QUESTION),
        question=QUESTION,
        payload_json=_json.dumps({
            "answer": "",
            "confidence": "low",
            "fallback_targets": ["src/auth/service.py"],
            "_schema_version": _ANSWER_SCHEMA_VERSION,
        }),
        provider_name="mock",
        model_name="mock-1",
    ))
    await session.commit()

    direct = _Provider("Auth flows through src/auth/service.py via AuthService.check().")
    _patch_provider(monkeypatch, answer_mod, direct)
    result = await get_answer(QUESTION)
    assert direct.calls == 1, "empty-answer cache row must be bypassed"
    assert "AuthService.check" in result["answer"]


@pytest.mark.asyncio
async def test_cache_bypassed_when_indexed_commit_changes(setup_mcp, factory, session, monkeypatch):
    """A row stamped at commit A is bypassed once the repo is indexed at B."""
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)

    repo = (await session.execute(select(Repository))).scalars().first()
    repo.head_commit = "a" * 40
    await session.commit()

    v1 = _Provider("Answer synthesised at commit A (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, v1)
    await get_answer(QUESTION)
    assert v1.calls == 1

    # Same commit → cache hit, no new synthesis.
    await get_answer(QUESTION)
    assert v1.calls == 1

    repo.head_commit = "b" * 40
    await session.commit()

    v2 = _Provider("Answer synthesised at commit B (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, v2)
    result = await get_answer(QUESTION)
    assert v2.calls == 1, "commit change must bypass the cached row"
    assert "commit B" in result["answer"]

    rows = await _cache_rows(factory)
    assert len(rows) == 1
    assert _json.loads(rows[0].payload_json)["_indexed_commit"] == "b" * 40


@pytest.mark.parametrize(
    ("env_name", "first_value", "second_value"),
    [
        pytest.param("REPOWISE_ANSWER_TEMPERATURE", "0.2", "0.8", id="temperature"),
        pytest.param("REPOWISE_ANSWER_TOP_P", "0.7", "0.9", id="top-p"),
        pytest.param("REPOWISE_ANSWER_TOP_K", "32", "64", id="top-k"),
    ],
)
@pytest.mark.asyncio
async def test_each_answer_sampling_field_bypasses_cache(
    setup_mcp,
    factory,
    monkeypatch,
    env_name,
    first_value,
    second_value,
):
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)
    monkeypatch.setenv(env_name, first_value)
    first_provider = _Provider("First sampling answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, first_provider)
    await get_answer(QUESTION)

    monkeypatch.setenv(env_name, second_value)
    second_provider = _Provider("Changed sampling answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, second_provider)
    result = await get_answer(QUESTION)

    assert second_provider.calls == 1
    assert "Changed sampling answer" in result["answer"]


@pytest.mark.asyncio
async def test_get_answer_forwards_resolved_sampling_to_synthesis(
    setup_mcp,
    monkeypatch,
):
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)
    monkeypatch.setenv("REPOWISE_ANSWER_TEMPERATURE", "1.0")
    monkeypatch.setenv("REPOWISE_ANSWER_TOP_P", "0.95")
    monkeypatch.setenv("REPOWISE_ANSWER_TOP_K", "64")
    provider = _Provider("Configured sampling answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, provider)

    await get_answer(QUESTION)

    assert provider.last_kwargs["sampling"].configured() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }


@pytest.mark.parametrize(
    "changed_input",
    ["provider", "model", "scope", "excerpt_chars", "max_tokens", "prompt_version"],
)
@pytest.mark.asyncio
async def test_each_other_synthesis_input_bypasses_cache(
    setup_mcp,
    factory,
    monkeypatch,
    changed_input,
):
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)
    first_scope = "src/auth" if changed_input == "scope" else None
    if changed_input == "excerpt_chars":
        monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "1500")
    if changed_input == "max_tokens":
        monkeypatch.setenv("REPOWISE_ANSWER_MAX_TOKENS", "1024")

    first_provider = _Provider("Original identity answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, first_provider)
    await get_answer(QUESTION, scope=first_scope)

    second_scope = first_scope
    second_provider = _Provider("Changed identity answer (src/auth/service.py).")
    if changed_input == "provider":
        second_provider.provider_name = "other"
    elif changed_input == "model":
        second_provider.model_name = "mock-2"
    elif changed_input == "scope":
        second_scope = "src/other"
    elif changed_input == "excerpt_chars":
        monkeypatch.setenv("REPOWISE_ANSWER_EXCERPT_CHARS", "3000")
    elif changed_input == "max_tokens":
        monkeypatch.setenv("REPOWISE_ANSWER_MAX_TOKENS", "2048")
    elif changed_input == "prompt_version":
        monkeypatch.setattr(
            answer_mod,
            "_SYNTHESIS_PROMPT_VERSION",
            answer_mod._SYNTHESIS_PROMPT_VERSION + 1,
        )
    _patch_provider(monkeypatch, answer_mod, second_provider)

    result = await get_answer(QUESTION, scope=second_scope)

    assert second_provider.calls == 1
    assert "Changed identity answer" in result["answer"]


@pytest.mark.asyncio
async def test_wiki_content_revision_bypasses_cache(
    setup_mcp,
    factory,
    session,
    monkeypatch,
):
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.core.persistence.models import Page
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)
    original_provider = _Provider("Original wiki answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, original_provider)
    await get_answer(QUESTION)

    repo = (await session.execute(select(Repository))).scalars().first()
    now = datetime.now(UTC)
    session.add(
        Page(
            id="module_page:new",
            repository_id=repo.id,
            page_type="module_page",
            title="New",
            content="New wiki content",
            summary="New wiki content",
            target_path="new",
            source_hash="a" * 64,
            model_name="mock-1",
            provider_name="mock",
            created_at=now,
            updated_at=now,
        )
    )
    await session.commit()

    changed_provider = _Provider("Changed wiki answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, changed_provider)
    result = await get_answer(QUESTION)

    assert changed_provider.calls == 1
    assert "Changed wiki answer" in result["answer"]


@pytest.mark.asyncio
async def test_cache_row_past_ttl_is_bypassed(setup_mcp, factory, session, monkeypatch):
    """Rows older than the hard TTL re-synthesise even without commit metadata."""
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)

    v1 = _Provider("Original answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, v1)
    await get_answer(QUESTION)

    rows = await _cache_rows(factory)
    assert len(rows) == 1
    from repowise.core.persistence.database import get_session

    async with get_session(factory) as s:
        row = (await s.execute(select(AnswerCache))).scalars().one()
        row.created_at = datetime.now(UTC) - timedelta(days=30)
        await s.commit()

    v2 = _Provider("Fresh answer (src/auth/service.py).")
    _patch_provider(monkeypatch, answer_mod, v2)
    result = await get_answer(QUESTION)
    assert v2.calls == 1, "expired row must be bypassed"
    assert "Fresh answer" in result["answer"]


@pytest.mark.asyncio
async def test_cache_write_failure_logs_instead_of_silencing(setup_mcp, monkeypatch, caplog):
    """A failing cache write must not block the response — and must be logged."""
    import repowise.server.mcp_server.tool_answer.answer as answer_mod
    from repowise.server.mcp_server import get_answer

    _patch_retrieval(monkeypatch, answer_mod)
    _patch_provider(monkeypatch, answer_mod, _Provider("Answer text (src/auth/service.py)."))

    real_dumps = answer_mod._json.dumps

    def _boom(obj, *a, **k):
        if isinstance(obj, dict) and "_schema_version" in obj:
            raise RuntimeError("simulated serialization failure")
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(answer_mod._json, "dumps", _boom)

    with caplog.at_level("WARNING", logger="repowise.mcp.answer"):
        result = await get_answer(QUESTION)

    assert result["answer"], "response must survive a cache-write failure"
    assert any("cache write failed" in r.message for r in caplog.records)
