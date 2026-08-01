"""The vector arm's evidence snippet, and the field it is cut from.

The full-text arm holds the page content at search time and can window it. The
vector arm cannot: LanceDB stores a fixed prefix of the content at index time
and that is all a search has to work with. So the stored prefix has to be wide
enough for a window to be cut out of it.

Summary readers use the same column. An omitted width keeps the historical
200-character excerpt, while an explicit width is served up to LanceDB's
stored-prefix ceiling.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from repowise.core.persistence.search import _SNIPPET_LEN
from repowise.core.persistence.vector_store.lancedb_store import (
    STORED_SNIPPET_CHARS,
    LanceDBVectorStore,
)
from repowise.core.providers.embedding.base import MockEmbedder

_OPENER = "## Overview\n\nThis module is part of the indexing pipeline. " + "Filler. " * 60
_MATCH = "The walker refuses to descend into a nested git checkout."


@pytest.fixture
async def store(tmp_path):
    st = LanceDBVectorStore(db_path=str(tmp_path / "lancedb"), embedder=MockEmbedder())
    yield st


def _page(content: str) -> dict:
    return {
        "title": "File: walker.py",
        "page_type": "file_page",
        "target_path": "src/walker.py",
        "content": content,
    }


class TestStoredWidth:
    def test_the_stored_prefix_is_wider_than_the_snippet_it_serves(self):
        """A window can only be cut from text the store actually kept."""
        assert STORED_SNIPPET_CHARS > _SNIPPET_LEN

    async def test_a_long_page_is_stored_up_to_the_cap(self, store):
        content = _OPENER + _MATCH + " tail " * 1000

        await store.embed_and_upsert("p1", content, _page(content))
        rows = await store._table.query().select(["content_snippet"]).to_list()

        assert len(rows[0]["content_snippet"]) == STORED_SNIPPET_CHARS

    async def test_a_page_shorter_than_the_cap_is_stored_whole(self, store):
        content = _OPENER + _MATCH
        assert len(content) < STORED_SNIPPET_CHARS

        await store.embed_and_upsert("p1", content, _page(content))
        rows = await store._table.query().select(["content_snippet"]).to_list()

        assert rows[0]["content_snippet"] == content


class TestSearchSnippet:
    async def test_a_query_carrying_search_centres_the_snippet(self, store):
        content = _OPENER + _MATCH
        await store.embed_and_upsert("p1", content, _page(content))

        results = await store.search("nested git checkout", limit=5)

        assert "nested git checkout" in results[0].snippet
        assert len(results[0].snippet) <= _SNIPPET_LEN

    async def test_no_match_falls_back_to_the_opener(self, store):
        content = _OPENER + _MATCH
        await store.embed_and_upsert("p1", content, _page(content))

        results = await store.search("xylophone tuning", limit=5)

        assert results[0].snippet.startswith("## Overview")

    async def test_searching_by_raw_vector_serves_the_opener_at_the_old_size(self, store):
        """That path has no query text, so there is nothing to centre on.

        It must not start returning the whole stored prefix instead — this is
        what the answer pipeline calls, and the snippet reaches a prompt.
        """
        content = _OPENER + _MATCH
        await store.embed_and_upsert("p1", content, _page(content))
        vector = (await store._embedder.embed(["anything"]))[0]

        results = await store.search_by_vector([float(v) for v in vector], limit=5)

        assert len(results[0].snippet) <= _SNIPPET_LEN
        assert results[0].snippet.startswith("## Overview")


class TestPromptSummaryWidths:
    """Summary reads distinguish the fallback width from an explicit request."""

    async def test_summary_by_path_stays_at_the_old_width(self, store):
        content = _OPENER + _MATCH
        await store.embed_and_upsert("p1", content, _page(content))

        found = await store.get_page_summary_by_path("src/walker.py")

        assert len(found["summary"]) <= _SNIPPET_LEN

    async def test_summaries_by_paths_stay_at_the_old_width(self, store):
        content = _OPENER + _MATCH
        await store.embed_and_upsert("p1", content, _page(content))

        found = await store.get_page_summaries_by_paths(["src/walker.py"])

        assert len(found["src/walker.py"]["summary"]) <= _SNIPPET_LEN

    async def test_a_read_above_the_stored_width_warns_once(self, store):
        content = "x" * (STORED_SNIPPET_CHARS + 100)
        await store.embed_batch(
            [
                ("p1", content, _page(content)),
                (
                    "p2",
                    content,
                    {
                        **_page(content),
                        "target_path": "src/other.py",
                    },
                ),
            ]
        )

        with capture_logs() as logs:
            found = await store.get_page_summaries_by_paths(
                ["src/walker.py", "src/other.py"],
                max_chars=STORED_SNIPPET_CHARS + 500,
            )

        assert {len(payload["summary"]) for payload in found.values()} == {
            STORED_SNIPPET_CHARS
        }
        warnings = [
            event
            for event in logs
            if event.get("event") == "vector_store.summary_read_exceeds_stored_width"
        ]
        assert len(warnings) == 1
