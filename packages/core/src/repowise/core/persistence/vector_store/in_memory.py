"""Pure-Python in-memory vector store (tests + small-scale dev)."""

from __future__ import annotations

from repowise.core.providers.embedding.base import Embedder

from ..search import SearchResult
from ._base import (
    FALLBACK_SUMMARY_CHARS,
    VectorStore,
    cosine_similarity,
    iter_embed_chunks,
)

__all__ = ["InMemoryVectorStore"]


def summary_text(meta: dict, max_chars: int | None) -> str:
    """The summary a lookup returns, cut once to the caller's width.

    Cutting the fallback to a fixed width first and the caller's width second
    would cap any request wider than the fixed one.
    """
    text = meta.get("summary") or str(meta.get("content", ""))
    return text[: max_chars or FALLBACK_SUMMARY_CHARS]


class InMemoryVectorStore(VectorStore):
    """Cosine-similarity vector store backed by a plain Python dict.

    Suitable for unit tests and small-scale development use.
    No external dependencies beyond the Embedder.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        # page_id → (vector, metadata)
        self._store: dict[str, tuple[list[float], dict]] = {}

    async def embed_and_upsert(self, page_id: str, text: str, metadata: dict) -> None:
        vectors = await self._embedder.embed([text])
        self._store[page_id] = (vectors[0], dict(metadata))

    async def embed_batch(self, items: list[tuple[str, str, dict]]) -> None:
        # Chunked like the persistent stores: this store is also the runtime
        # fallback when LanceDB isn't installed, so it can sit in front of a
        # real embedder with real request limits.
        if not items:
            return
        for chunk, texts in iter_embed_chunks(items):
            vectors = await self._embedder.embed(texts)
            for (page_id, _text, metadata), vector in zip(chunk, vectors, strict=True):
                self._store[page_id] = (vector, dict(metadata))

    def _search_by_vector(self, q_vec: list[float], limit: int) -> list[SearchResult]:
        scored: list[tuple[float, str, dict]] = []
        for pid, (vec, meta) in self._store.items():
            score = cosine_similarity(q_vec, vec)
            scored.append((score, pid, meta))

        scored.sort(key=lambda t: t[0], reverse=True)

        results = []
        for score, pid, meta in scored[:limit]:
            content = meta.get("content", "")
            results.append(
                SearchResult(
                    page_id=pid,
                    title=str(meta.get("title", "")),
                    page_type=str(meta.get("page_type", "")),
                    target_path=str(meta.get("target_path", "")),
                    score=score,
                    snippet=str(content)[:200].rstrip(),
                    search_type="vector",
                )
            )
        return results

    async def upsert_page_texts(self, items: list[tuple[str, str, dict]]) -> bool:
        """Store rows with a zero placeholder vector, no embedder call.

        Summary lookups and ``list_page_ids`` read only the metadata, so the
        rows behave identically to embedded ones; ``cosine_similarity``
        scores a zero vector 0.0, so unembedded rows rank last in search
        until the end-of-run :meth:`embed_batch` replaces the placeholder.
        """
        for page_id, _text, metadata in items:
            self._store[page_id] = ([0.0] * self._embedder.dimensions, dict(metadata))
        return True

    async def upsert_vectors(self, items: list[tuple[str, list[float], dict]]) -> bool:
        for page_id, vector, metadata in items:
            self._store[page_id] = (list(vector), dict(metadata))
        return True

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if not self._store:
            return []
        q_vecs = await self._embedder.embed([query])
        return self._search_by_vector(q_vecs[0], limit)

    async def search_by_vector(self, vector: list[float], limit: int = 10) -> list[SearchResult]:
        if not self._store:
            return []
        return self._search_by_vector(vector, limit)

    async def search_many(self, queries: list[str], limit: int = 10) -> list[list[SearchResult]]:
        """One embedder call for all queries, then local scoring per query."""
        if not queries:
            return []
        if not self._store:
            return [[] for _ in queries]
        q_vecs = await self._embedder.embed(list(queries))
        return [self._search_by_vector(q_vec, limit) for q_vec in q_vecs]

    async def delete(self, page_id: str) -> None:
        self._store.pop(page_id, None)

    async def delete_many(self, page_ids: list[str]) -> None:
        for page_id in page_ids:
            self._store.pop(page_id, None)

    async def close(self) -> None:
        self._store.clear()

    async def list_page_ids(self) -> set[str]:
        return set(self._store.keys())

    async def get_page_summary_by_path(
        self, path: str, max_chars: int | None = None
    ) -> dict | None:
        """Return {'summary': str, 'key_exports': list[str]} for a previously-indexed page, or None.

        Reads 'summary' from metadata if present (set by the generation
        pipeline), else 'content'. Either is cut to *max_chars*, or to
        :data:`FALLBACK_SUMMARY_CHARS` when the caller states no width.
        'key_exports' reads the 'exports' metadata field if present, else [].
        """
        for _pid, (_, meta) in self._store.items():
            if meta.get("target_path") == path:
                summary = summary_text(meta, max_chars)
                key_exports = meta.get("exports") or []
                return {"summary": summary, "key_exports": list(key_exports)}
        return None

    async def get_page_summaries_by_paths(
        self, paths: list[str], max_chars: int | None = None
    ) -> dict[str, dict]:
        """Single-pass scan over the store — avoids N full scans when
        many paths are queried at once.
        """
        if not paths:
            return {}
        wanted = set(paths)
        out: dict[str, dict] = {}
        for _pid, (_, meta) in self._store.items():
            tp = meta.get("target_path")
            if tp in wanted and tp not in out:
                summary = summary_text(meta, max_chars)
                out[str(tp)] = {
                    "summary": summary,
                    "key_exports": list(meta.get("exports") or []),
                }
                if len(out) == len(wanted):
                    break
        return out

    def __len__(self) -> int:
        return len(self._store)
