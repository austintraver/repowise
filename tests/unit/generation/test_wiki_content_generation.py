"""Generation tests for repository-wide content evidence and validation."""

from dataclasses import replace
from types import SimpleNamespace

from repowise.core.generation.context_assembler import ContextAssembler
from repowise.core.generation.page_generator import PageGenerator
from repowise.core.providers.llm.base import GeneratedResponse
from repowise.core.providers.llm.mock import MockProvider


class OverviewGraphEvidence:
    def community_info(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id=1,
                label="Readers",
                size=4,
                cohesion=0.8,
            )
        ]

    def execution_flows(self) -> SimpleNamespace:
        return SimpleNamespace(
            flows=[
                SimpleNamespace(
                    entry_point="src/demo/server.py::main",
                    score=0.9,
                    trace=["src/demo/server.py::main", "src/demo/core.py::run"],
                )
            ]
        )


async def test_repo_overview_prompt_includes_declared_purpose(
    sample_config,
    sample_repo_structure,
) -> None:
    provider = MockProvider()
    generator = PageGenerator(
        provider,
        ContextAssembler(sample_config),
        sample_config,
    )
    source_map = {
        "README.md": (
            b"# Demo\n\n"
            b"Demo finds earlier coding sessions and replays the original conversation.\n\n"
            b"## Installation\n\nInstall it with uv."
        )
    }

    await generator.generate_repo_overview(
        sample_repo_structure,
        pagerank={},
        sccs=[],
        community={},
        repo_name="demo",
        source_map=source_map,
    )

    prompt = str(provider.calls[0]["user_prompt"])
    assert "Declared purpose evidence" in prompt
    assert "`README.md`" in prompt
    assert "Demo finds earlier coding sessions and replays the original conversation." in prompt


async def test_repo_overview_retains_supplied_structural_evidence(
    sample_config,
    sample_repo_structure,
) -> None:
    response = GeneratedResponse(
        content=(
            "## Project Summary\n\nDemo processes sessions.\n\n"
            "## Technology Stack\n\nPython.\n\n"
            "## Entry Points\n\n`python_pkg/calculator.py`\n\n"
            "## Architecture\n\nThe package contains a calculator."
        ),
        input_tokens=10,
        output_tokens=20,
    )
    provider = MockProvider(responses=[response])
    generator = PageGenerator(
        provider,
        ContextAssembler(sample_config),
        sample_config,
    )

    page = await generator.generate_repo_overview(
        sample_repo_structure,
        pagerank={"python_pkg/calculator.py": 0.9},
        sccs=[],
        community={},
        repo_name="demo",
        graph_builder=OverviewGraphEvidence(),
        external_systems=[{"name": "sqlite", "category": "database", "ecosystem": "local"}],
        decision_records=[
            {
                "title": "Keep sessions local",
                "decision": "Store session data locally.",
                "rationale": "Avoid remote dependencies.",
            }
        ],
        git_meta_map={
            "python_pkg/calculator.py": {
                "file_path": "python_pkg/calculator.py",
                "commit_count_90d": 5,
                "is_hotspot": True,
                "is_stable": False,
                "first_commit_at": "2024-01-01",
                "age_days": 400,
            }
        },
    )

    assert "## Primary Execution Flows" in page.content
    assert "## Architectural Clusters" in page.content
    assert "## Most Central Files" in page.content
    assert "## External Systems" in page.content
    assert "## Recorded Decisions" in page.content
    assert "## Codebase health signals" in page.content
    assert "`src/demo/server.py::main`" in page.content
    assert "`python_pkg/calculator.py` (0.9000)" in page.content
    assert "`sqlite`" in page.content
    assert "Keep sessions local" in page.content
    assert "## Packages" not in page.content
    assert "## Languages" not in page.content
    assert page.content.count("## Project Summary") == 1
    assert page.content.count("## Entry Points") == 1

    stale_prior = replace(page, content=response.content)
    replay_provider = MockProvider()
    replay_generator = PageGenerator(
        replay_provider,
        ContextAssembler(sample_config),
        sample_config,
        prior_pages={page.page_id: stale_prior},
    )
    replayed_page = await replay_generator.generate_repo_overview(
        sample_repo_structure,
        pagerank={"python_pkg/calculator.py": 0.9},
        sccs=[],
        community={},
        repo_name="demo",
        graph_builder=OverviewGraphEvidence(),
        external_systems=[{"name": "sqlite", "category": "database", "ecosystem": "local"}],
        decision_records=[
            {
                "title": "Keep sessions local",
                "decision": "Store session data locally.",
                "rationale": "Avoid remote dependencies.",
            }
        ],
        git_meta_map={
            "python_pkg/calculator.py": {
                "file_path": "python_pkg/calculator.py",
                "commit_count_90d": 5,
                "is_hotspot": True,
                "is_stable": False,
                "first_commit_at": "2024-01-01",
                "age_days": 400,
            }
        },
    )

    assert replay_provider.call_count == 0
    assert "## Most Central Files" in replayed_page.content
    assert "reused_from_prior_run" not in replayed_page.metadata


async def test_repo_overview_does_not_duplicate_equivalent_structural_sections(
    sample_config,
    sample_repo_structure,
) -> None:
    response = GeneratedResponse(
        content=(
            "## Project Summary\n\nDemo processes sessions.\n\n"
            "## Technology Stack\n\nPython.\n\n"
            "## Entry Points\n\n`python_pkg/calculator.py`\n\n"
            "## Architecture\n\nThe package contains a calculator.\n\n"
            "## Top Files by PageRank\n\n`python_pkg/calculator.py`\n\n"
            "## Architectural Communities\n\nThe Readers community owns parsing.\n\n"
            "## Key Architectural Decisions\n\nKeep sessions local."
        ),
        input_tokens=10,
        output_tokens=20,
    )
    provider = MockProvider(responses=[response])
    generator = PageGenerator(
        provider,
        ContextAssembler(sample_config),
        sample_config,
    )

    page = await generator.generate_repo_overview(
        sample_repo_structure,
        pagerank={"python_pkg/calculator.py": 0.9},
        sccs=[],
        community={},
        repo_name="demo",
        graph_builder=OverviewGraphEvidence(),
        decision_records=[
            {
                "title": "Keep sessions local",
                "decision": "Store session data locally.",
                "rationale": "Avoid remote dependencies.",
            }
        ],
    )

    assert "## Top Files by PageRank" in page.content
    assert "## Architectural Communities" in page.content
    assert "## Key Architectural Decisions" in page.content
    assert "## Most Central Files" not in page.content
    assert "## Architectural Clusters" not in page.content
    assert "## Recorded Decisions" not in page.content
