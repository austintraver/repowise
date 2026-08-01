"""The generation-side deployment dials: token_budget and dependency_summary_chars.

Both parse from config.yaml exactly the way ``max_tokens`` does, and both
default to the historical constants. ``dependency_summary_chars`` is the
assembler's consumption cap. Shared intermediates are sized by their
consumers: the in-run summary reservoir holds
max(dial, GUIDED_TOUR_SUMMARY_CHARS). Each persistence backend decides how much
page content it can retain and serve.
"""

from datetime import UTC, datetime

import pytest

from repowise.core.generation.models import GenerationConfig
from repowise.core.generation.page_generator.helpers import overview_summary
from repowise.core.generation.page_generator.orchestrate import _embed_item


def test_config_keys_parse_like_max_tokens():
    config = GenerationConfig.from_repo_config(
        {"max_tokens": 16384, "token_budget": 96000, "dependency_summary_chars": 600}
    )
    assert config.token_budget == 96000
    assert config.dependency_summary_chars == 600


def test_defaults_are_the_historical_constants():
    config = GenerationConfig.from_repo_config({"max_tokens": 16384})
    assert config.token_budget == 48000
    assert config.dependency_summary_chars == 200


@pytest.mark.parametrize("bad", [0, -5, True, "plenty", 3.5])
@pytest.mark.parametrize("key", ["token_budget", "dependency_summary_chars"])
def test_non_positive_or_non_integer_values_are_rejected(key, bad):
    with pytest.raises(ValueError, match=key):
        GenerationConfig.from_repo_config({"max_tokens": 16384, key: bad})


@pytest.mark.parametrize("bad", [0, -5, True, "plenty", 3.5])
def test_the_dial_is_validated_on_direct_construction_too(bad):
    """The CLI and the pipeline build configs without going through config.yaml.

    Every bad value here reaches a consumer as a slice bound rather than an
    error, so it has to be rejected at construction: -5 trims the end off each
    summary, True means one character, and 3.5 raises far from its cause.
    """
    with pytest.raises(ValueError, match="dependency_summary_chars"):
        GenerationConfig(dependency_summary_chars=bad)


def test_overview_summary_fills_the_requested_width():
    body = "intro\n## Overview\n" + "x" * 3000 + "\n## Next\nrest"
    assert len(overview_summary(body, 1200)) == 1200
    # Without a following heading, the scan needs room to discard leading
    # whitespace and still fill the caller's requested width.
    open_ended_body = "intro\n## Overview\n\n" + "x" * 3000
    assert len(overview_summary(open_ended_body, 600)) == 600
    # No ## Overview section: plain prefix at the cap.
    assert len(overview_summary("y" * 3000, 900)) == 900


def _page(content: str):
    from repowise.core.generation.models import GeneratedPage

    now = datetime.now(UTC).isoformat()
    return GeneratedPage(
        page_id="module_page:pkg",
        page_type="module_page",
        title="Pkg",
        content=content,
        source_hash="deadbeef",
        model_name="mock",
        provider_name="mock",
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
        generation_level=4,
        target_path="pkg",
        created_at=now,
        updated_at=now,
    )


def test_embed_item_leaves_content_width_to_the_backend():
    """Generation must not impose LanceDB's storage ceiling on every store."""
    content = "z" * 5000
    page = _page(content)
    config = GenerationConfig(dependency_summary_chars=600)

    page_id, text, metadata = _embed_item(page, config.summary_reservoir_chars)

    assert page_id == page.page_id
    assert text == content
    assert metadata["content"] == content


async def test_a_narrow_dial_does_not_starve_the_guided_tour():
    """The reservoir the tour actually reads is written at the reservoir width.

    The tour reads ``run.completed_page_summaries``, filled by ``run_level``.
    ``_embed_item``'s ``summary`` key is a different writer and LanceDB drops
    it, so asserting there would not protect this.
    """
    import asyncio
    from types import SimpleNamespace

    from repowise.core.generation.models import GUIDED_TOUR_SUMMARY_CHARS
    from repowise.core.generation.page_generator.orchestrate import _GenerationRun

    page = _page("z" * 5000)

    async def written():
        return page

    run = SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        job_system=None,
        job_id=None,
        on_page_done=None,
        on_page_ready=None,
        vector_store=None,
        completed_page_summaries={},
        # A dial below the tour's width: a writer using the bare dial starves it.
        config=GenerationConfig(dependency_summary_chars=100),
    )
    await _GenerationRun.run_level(run, [(page.page_id, written())], level=4)

    captured = run.completed_page_summaries[page.target_path]
    assert len(captured) == GUIDED_TOUR_SUMMARY_CHARS


@pytest.mark.asyncio
@pytest.mark.parametrize("dial", [200, 600])
async def test_dependency_summaries_are_read_at_the_configured_width(dial):
    """Reads scale with the reservoir width, not with a fixed constant."""
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore

    class _Embedder:
        dimensions = 2

        async def embed(self, texts):
            return [[0.0, 0.0] for _ in texts]

    store = InMemoryVectorStore(_Embedder())
    await store.upsert_page_texts(
        [("file_page:pkg/mod.py", "text", {"target_path": "pkg/mod.py", "summary": "w" * 3000})]
    )

    batch = await store.get_page_summaries_by_paths(["pkg/mod.py"], max_chars=dial)
    assert len(batch["pkg/mod.py"]["summary"]) == dial

    single = await store.get_page_summary_by_path("pkg/mod.py", max_chars=dial)
    assert len(single["summary"]) == dial


def _file_ctx(path: str, pagerank: float = 1.0):
    """A FilePageContext carrying only what module-page assembly reads."""
    from repowise.core.generation.context.contexts import FilePageContext

    return FilePageContext(
        file_path=path,
        language="python",
        docstring=None,
        symbols=[],
        imports=[],
        exports=[],
        file_source_snippet="",
        pagerank_score=pagerank,
        betweenness_score=0.0,
        community_id=0,
        dependents=[],
        dependencies=[],
        is_api_contract=False,
        is_entry_point=False,
        is_test=False,
        parse_errors=[],
        estimated_tokens=0,
    )


@pytest.mark.parametrize("dial", [100, 200, 600])
def test_the_dial_cuts_what_a_module_page_prompt_receives(dial):
    """The dial's only production consumer, asserted end to end.

    ``assemble_module_page`` is where the number becomes prompt text: each key
    file's summary is cut to it. Without this, both cuts can be deleted and the
    suite stays green.
    """
    import networkx as nx

    from repowise.core.generation.context.assembler import ContextAssembler

    assembler = ContextAssembler(GenerationConfig(dependency_summary_chars=dial))
    contexts = [_file_ctx("pkg/a.py"), _file_ctx("pkg/b.py", 0.5)]
    ctx = assembler.assemble_module_page(
        title="Pkg",
        language="python",
        file_contexts=contexts,
        graph=nx.DiGraph(),
        page_summaries={"pkg/a.py": "x" * 3000, "pkg/b.py": "y" * 3000},
    )

    summaries = [kf["summary"] for kf in ctx.key_files]
    assert summaries, "no key files reached the prompt"
    assert all(len(s) == dial for s in summaries)


@pytest.mark.parametrize("dial", [100, 600])
async def test_the_prefetch_fills_the_reservoir_at_its_width(dial):
    """The read this change exists to fix, asserted at the width it must use.

    Dial 100 is below GUIDED_TOUR_SUMMARY_CHARS, so a prefetch reading at the
    bare dial — or at the old hardcoded snippet width — differs from the
    reservoir and fails here.
    """
    from types import SimpleNamespace

    import networkx as nx

    from repowise.core.generation.page_generator.levels import (
        _prefetch_dependency_summaries,
    )
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore

    class _Embedder:
        dimensions = 2

        async def embed(self, texts):
            return [[0.0, 0.0] for _ in texts]

    store = InMemoryVectorStore(_Embedder())
    await store.upsert_page_texts(
        [("file_page:pkg/dep.py", "t", {"target_path": "pkg/dep.py", "summary": "s" * 3000})]
    )

    graph = nx.DiGraph()
    graph.add_edge("pkg/a.py", "pkg/dep.py")
    config = GenerationConfig(dependency_summary_chars=dial)
    run = SimpleNamespace(
        vector_store=store,
        graph=graph,
        completed_page_summaries={},
        config=config,
        code_files=[SimpleNamespace(file_info=SimpleNamespace(path="pkg/a.py"))],
    )

    await _prefetch_dependency_summaries(run)

    assert run.completed_page_summaries, "the prefetch reached no dependency"
    got = run.completed_page_summaries["pkg/dep.py"]
    assert len(got) == config.summary_reservoir_chars
