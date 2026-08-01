"""Batched vector-store operations: pgvector executemany + IN-filtered
summary lookups (pgvector, LanceDB)."""

from __future__ import annotations

import pytest

from repowise.core.persistence.vector_store._base import VectorStore
from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore
from repowise.core.persistence.vector_store.lancedb_store import (
    _SNIPPET_LEN,
    _page_ids_in_filter,
    _paths_in_filter,
)
from repowise.core.persistence.vector_store.pgvector_store import (
    PgVectorStore,
    _summary_payload,
)
from repowise.core.providers.embedding.base import MockEmbedder

# ---------------------------------------------------------------------------
# Fake async-SQLAlchemy session plumbing
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Records every execute() call; serves canned rows."""

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.executed: list[tuple[str, object]] = []
        self.commits = 0
        self._rows = rows or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return _FakeResult(self._rows)

    async def commit(self):
        self.commits += 1


def _factory(session: _FakeSession):
    def make() -> _FakeSession:
        return session

    return make


# ---------------------------------------------------------------------------
# pgvector embed_batch — one executemany, not N round-trips
# ---------------------------------------------------------------------------


async def test_pg_embed_batch_single_executemany() -> None:
    session = _FakeSession()
    store = PgVectorStore(_factory(session), MockEmbedder())

    await store.embed_batch([("p1", "alpha", {}), ("p2", "beta", {}), ("p3", "gamma", {})])

    assert len(session.executed) == 1  # one statement, list-of-params
    stmt, params = session.executed[0]
    assert "UPDATE wiki_pages SET embedding" in stmt
    assert isinstance(params, list) and len(params) == 3
    assert [p["pid"] for p in params] == ["p1", "p2", "p3"]
    assert all(p["emb"].startswith("[") for p in params)
    assert session.commits == 1


async def test_pg_embed_batch_empty_is_noop() -> None:
    session = _FakeSession()
    store = PgVectorStore(_factory(session), MockEmbedder())
    await store.embed_batch([])
    assert session.executed == []


# ---------------------------------------------------------------------------
# pgvector get_page_summaries_by_paths — one IN query
# ---------------------------------------------------------------------------


async def test_pg_batch_summaries_single_query_first_row_wins() -> None:
    rows = [
        ("a.py", "Summary of a", {"exports": ["A"]}),
        ("a.py", "duplicate row ignored", {}),
        ("b.py", "Summary of b", '{"exports": ["B1", "B2"]}'),
        ("c.py", "", {}),  # empty summary dropped
    ]
    session = _FakeSession(rows=rows)
    store = PgVectorStore(_factory(session), MockEmbedder())

    out = await store.get_page_summaries_by_paths(["a.py", "b.py", "c.py"])

    assert len(session.executed) == 1
    stmt, params = session.executed[0]
    assert "target_path IN" in stmt
    assert params == {"paths": ["a.py", "b.py", "c.py"]}
    assert out == {
        "a.py": {"summary": "Summary of a", "key_exports": ["A"]},
        "b.py": {"summary": "Summary of b", "key_exports": ["B1", "B2"]},
    }


async def test_pg_batch_summaries_empty_paths() -> None:
    session = _FakeSession()
    store = PgVectorStore(_factory(session), MockEmbedder())
    assert await store.get_page_summaries_by_paths([]) == {}
    assert session.executed == []


def test_summary_payload_matches_single_path_parsing() -> None:
    assert _summary_payload("x" * 600, {"exports": ["E"]}) == {
        "summary": "x" * 500,
        "key_exports": ["E"],
    }
    assert _summary_payload(None, "not json") == {"summary": "", "key_exports": []}


async def test_pg_summary_reads_apply_the_requested_width() -> None:
    content = "x" * 1200
    single_session = _FakeSession(rows=[(content, {})])
    single_store = PgVectorStore(_factory(single_session), MockEmbedder())
    batch_session = _FakeSession(rows=[("a.py", content, {})])
    batch_store = PgVectorStore(_factory(batch_session), MockEmbedder())

    single = await single_store.get_page_summary_by_path("a.py", max_chars=733)
    batch = await batch_store.get_page_summaries_by_paths(["a.py"], max_chars=733)

    assert len(single["summary"]) == 733
    assert len(batch["a.py"]["summary"]) == 733


async def test_base_batch_forwards_the_requested_width() -> None:
    from types import SimpleNamespace

    received_widths: list[int | None] = []

    async def read_summary(path: str, max_chars: int | None = None) -> dict:
        received_widths.append(max_chars)
        return {"summary": f"{path}:{'x' * (max_chars or 0)}", "key_exports": []}

    store = SimpleNamespace(get_page_summary_by_path=read_summary)
    found = await VectorStore.get_page_summaries_by_paths(
        store,
        ["a.py", "b.py"],
        max_chars=733,
    )

    assert received_widths == [733, 733]
    assert all(len(payload["summary"].split(":", 1)[1]) == 733 for payload in found.values())


# ---------------------------------------------------------------------------
# LanceDB IN-filter escaping + optional round-trip
# ---------------------------------------------------------------------------


def test_lancedb_paths_filter_escapes_quotes() -> None:
    flt = _paths_in_filter(["a.py", "weird'name.py"])
    assert flt == "target_path IN ('a.py', 'weird''name.py')"


@pytest.mark.asyncio
async def test_lancedb_batch_summaries_roundtrip(tmp_path) -> None:
    pytest.importorskip("lancedb")
    from repowise.core.persistence.vector_store import LanceDBVectorStore

    store = LanceDBVectorStore(str(tmp_path / "lance"), MockEmbedder())
    try:
        await store.embed_batch(
            [
                ("p1", "Summary text for module a", {"target_path": "a.py"}),
                ("p2", "Summary text for module b", {"target_path": "b.py"}),
                ("p3", "Summary text for module c", {"target_path": "c.py"}),
            ]
        )
        batch = await store.get_page_summaries_by_paths(["a.py", "c.py", "missing.py"])
        singles = {p: await store.get_page_summary_by_path(p) for p in ("a.py", "c.py")}
        assert set(batch) == {"a.py", "c.py"}
        for p in ("a.py", "c.py"):
            assert batch[p]["summary"] == singles[p]["summary"]
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("dial", [200, 600])
async def test_lancedb_summaries_honor_the_requested_width(tmp_path, dial) -> None:
    """Reads honor the caller's width."""
    pytest.importorskip("lancedb")
    from repowise.core.persistence.vector_store import LanceDBVectorStore

    store = LanceDBVectorStore(str(tmp_path / "lance"), MockEmbedder())
    try:
        await store.embed_batch(
            [("p1", "x" * 3000, {"target_path": "a.py", "content": "x" * 3000})]
        )

        batch = await store.get_page_summaries_by_paths(["a.py"], max_chars=dial)
        single = await store.get_page_summary_by_path("a.py", max_chars=dial)
        assert len(batch["a.py"]["summary"]) == dial
        assert len(single["summary"]) == dial

        # No max_chars: excerpt callers keep the search-snippet width.
        assert len((await store.get_page_summary_by_path("a.py"))["summary"]) == _SNIPPET_LEN
    finally:
        await store.close()


def test_every_backend_accepts_the_summary_read_width() -> None:
    """Generation passes max_chars= to whatever backend is configured.

    A backend that kept the older two-argument signature raises TypeError
    inside the prefetch, which swallows it at debug level — so the symptom is
    silently empty dependency summaries rather than a failure.
    """
    import inspect

    from repowise.core.persistence.vector_store import (
        InMemoryVectorStore,
        LanceDBVectorStore,
        PgVectorStore,
        VectorStore,
    )

    backends = [InMemoryVectorStore, LanceDBVectorStore, PgVectorStore, VectorStore]
    for cls in backends:
        for name in ("get_page_summary_by_path", "get_page_summaries_by_paths"):
            params = inspect.signature(getattr(cls, name)).parameters
            assert "max_chars" in params, f"{cls.__name__}.{name} cannot take a read width"


# ---------------------------------------------------------------------------
# delete_many — bulk removal of swept page embeddings
# ---------------------------------------------------------------------------


async def test_in_memory_delete_many_removes_only_listed_ids() -> None:
    store = InMemoryVectorStore(MockEmbedder())
    await store.embed_batch([("p1", "a", {}), ("p2", "b", {}), ("p3", "c", {})])

    await store.delete_many(["p1", "p3"])

    assert await store.list_page_ids() == {"p2"}


async def test_in_memory_delete_many_empty_is_noop() -> None:
    store = InMemoryVectorStore(MockEmbedder())
    await store.embed_batch([("p1", "a", {})])
    await store.delete_many([])
    assert await store.list_page_ids() == {"p1"}


async def test_pg_delete_many_single_update_with_in_params() -> None:
    session = _FakeSession()
    store = PgVectorStore(_factory(session), MockEmbedder())

    await store.delete_many(["p1", "p2", "p3"])

    assert len(session.executed) == 1
    stmt, params = session.executed[0]
    assert "UPDATE wiki_pages SET embedding = NULL" in stmt
    assert "id IN" in stmt
    assert params == {"ids": ["p1", "p2", "p3"]}
    assert session.commits == 1


async def test_pg_delete_many_empty_is_noop() -> None:
    session = _FakeSession()
    store = PgVectorStore(_factory(session), MockEmbedder())
    await store.delete_many([])
    assert session.executed == []
    assert session.commits == 0


def test_lancedb_page_ids_filter_escapes_quotes() -> None:
    flt = _page_ids_in_filter(["module_page:c-1", "weird'id"])
    assert flt == "page_id IN ('module_page:c-1', 'weird''id')"


@pytest.mark.asyncio
async def test_lancedb_delete_many_roundtrip(tmp_path) -> None:
    pytest.importorskip("lancedb")
    from repowise.core.persistence.vector_store import LanceDBVectorStore

    store = LanceDBVectorStore(str(tmp_path / "lance"), MockEmbedder())
    try:
        await store.embed_batch(
            [
                ("p1", "module a", {"target_path": "a.py"}),
                ("p2", "module b", {"target_path": "b.py"}),
                ("p3", "module c", {"target_path": "c.py"}),
            ]
        )
        await store.delete_many(["p1", "p3"])
        assert await store.list_page_ids() == {"p2"}
        # empty list is a no-op
        await store.delete_many([])
        assert await store.list_page_ids() == {"p2"}
    finally:
        await store.close()
