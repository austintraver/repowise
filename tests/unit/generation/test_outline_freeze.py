"""Freezing the concept outline holds page instructions constant across runs.

The live namer is tested in test_concept_naming_wiring; these tests cover the
two halves the freeze adds. A named run must leave a donor artifact behind,
and a run configured with ``frozen_outline`` must replay that artifact —
titles, scopes, sections, order — without asking its own model, failing
loudly when the concept partition no longer matches the map.
"""

import json

import pytest

from repowise.core.generation import GenerationConfig
from repowise.core.generation.page_generator.outline_freeze import (
    NAMING_ARTIFACT_NAME,
    apply_frozen_naming,
    load_frozen_naming,
)
from repowise.core.generation.selection.selector import ModuleGroup
from repowise.core.pipeline import run_pipeline

from .test_concept_naming_wiring import (
    NamingProvider,
    _module_pages,
    _naming_payload_for,
    _run,
    _run_deterministic,
    _write_repo,
)


def _artifact_path(repo):
    return repo / ".repowise" / NAMING_ARTIFACT_NAME


async def _frozen_run(repo, provider):
    return await run_pipeline(
        repo,
        generate_docs=True,
        llm_client=provider,
        concurrency=1,
        generation_config=GenerationConfig(
            max_concurrency=1,
            frozen_outline=f".repowise/{NAMING_ARTIFACT_NAME}",
        ),
    )


# ---------------------------------------------------------------------------
# The donor artifact
# ---------------------------------------------------------------------------


async def test_a_named_run_leaves_a_donor_artifact(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    titles = {f"g{i:02d}": f"Title {i}" for i in range(1, 21)}

    await _run(repo, NamingProvider(_naming_payload_for(titles)))

    payload = json.loads(_artifact_path(repo).read_text(encoding="utf-8"))
    assert payload["naming_mode"] == "llm"
    assert payload["groups"], "artifact carries no groups"
    entries = list(payload["groups"].values())
    assert any(e["title"].startswith("Title ") for e in entries)
    assert all("scope" in e and "section" in e and "order" in e for e in entries)


async def test_a_keyless_run_writes_no_artifact(tmp_path):
    repo = _write_repo(tmp_path / "repo")

    await _run_deterministic(repo)

    assert not _artifact_path(repo).exists()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def test_a_frozen_run_replays_the_donor_and_never_asks_its_model(tmp_path):
    """The whole feature: run 2 writes its pages under run 1's instructions."""
    repo = _write_repo(tmp_path / "repo")

    donor = NamingProvider(_naming_payload_for({f"g{i:02d}": f"Alpha {i}" for i in range(1, 21)}))
    first = await _run(repo, donor)
    donor_titles = {p.page_id: p.title for p in _module_pages(first)}

    replayer = NamingProvider(
        _naming_payload_for({f"g{i:02d}": f"Omega {i}" for i in range(1, 21)})
    )
    second = await _frozen_run(repo, replayer)

    assert replayer.naming_calls == 0, "a frozen run paid for a naming call"
    frozen_titles = {p.page_id: p.title for p in _module_pages(second)}
    assert frozen_titles == donor_titles
    assert not any(t.startswith("Omega") for t in frozen_titles.values())


async def test_a_frozen_run_fails_loudly_when_the_partition_drifted(tmp_path):
    repo = _write_repo(tmp_path / "repo")
    await _run(
        repo, NamingProvider(_naming_payload_for({f"g{i:02d}": f"Alpha {i}" for i in range(1, 21)}))
    )

    artifact = _artifact_path(repo)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    dropped_key = next(iter(payload["groups"]))
    del payload["groups"][dropped_key]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover the current concept partition"):
        await _frozen_run(repo, NamingProvider(_naming_payload_for({"g01": "Unused"})))


async def test_a_frozen_run_fails_loudly_on_a_missing_map(tmp_path):
    repo = _write_repo(tmp_path / "repo")

    with pytest.raises(ValueError, match="does not exist"):
        await _frozen_run(repo, NamingProvider(_naming_payload_for({"g01": "Unused"})))


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_from_repo_config_parses_frozen_outline():
    config = GenerationConfig.from_repo_config({"frozen_outline": " maps/naming.json "})
    assert config.frozen_outline == "maps/naming.json"


def test_from_repo_config_defaults_frozen_outline_to_none():
    assert GenerationConfig.from_repo_config({}).frozen_outline is None


def test_from_repo_config_rejects_a_non_string_frozen_outline():
    for bad in (7, True, "", "   ", ["x"]):
        with pytest.raises(ValueError, match="frozen_outline"):
            GenerationConfig.from_repo_config({"frozen_outline": bad})


def test_an_explicit_override_beats_the_config_key():
    config = GenerationConfig.from_repo_config(
        {"frozen_outline": "from-config.json"}, frozen_outline=None
    )
    assert config.frozen_outline is None


# ---------------------------------------------------------------------------
# The pure pieces
# ---------------------------------------------------------------------------


def _group(key: str, title: str = "Plain") -> ModuleGroup:
    return ModuleGroup(
        key=f"src/{key}",
        display=title,
        language="python",
        file_paths=(f"src/{key}/a.py",),
        structural_key=f"concept-{key}",
    )


def test_apply_frozen_naming_renames_every_field():
    frozen = {
        "concept-a": {"title": "Ingest", "section": "Core", "order": 2, "scope": "Covers ingest."}
    }
    (renamed,) = apply_frozen_naming([_group("a")], frozen)
    assert (renamed.display, renamed.section, renamed.order, renamed.scope) == (
        "Ingest",
        "Core",
        2,
        "Covers ingest.",
    )


def test_apply_frozen_naming_names_the_missing_groups_in_its_error():
    with pytest.raises(ValueError, match="concept-b"):
        apply_frozen_naming([_group("a"), _group("b")], {"concept-a": {"title": "Ingest"}})


def test_load_frozen_naming_rejects_malformed_files(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'groups' mapping"):
        load_frozen_naming(empty)

    untitled = tmp_path / "untitled.json"
    untitled.write_text(json.dumps({"groups": {"concept-a": {"scope": "x"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no string 'title'"):
        load_frozen_naming(untitled)
