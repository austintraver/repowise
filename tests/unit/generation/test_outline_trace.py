"""Outline traces preserve enough evidence to explain and replay naming."""

import json
from pathlib import Path

import pytest

from repowise.core.generation.concept_tree.grouping import ConceptGroup
from repowise.core.generation.concept_tree.naming import (
    build_group_summaries,
    parse_json_object_result,
)
from repowise.core.generation.concept_tree.planner import PlannerInputs, name_groups
from repowise.core.generation.concept_tree.trace import (
    OutlineReplayProvider,
    OutlineTraceRecorder,
    load_outline_trace,
    outline_responses,
    text_sha256,
)
from repowise.core.providers.llm.base import GeneratedResponse, SamplingParameters
from repowise.core.providers.llm.mock import MockProvider


def trace_groups() -> list[ConceptGroup]:
    """Return two groups so one can succeed initially and one during repair."""
    return [
        ConceptGroup(
            members=["src/api/handler.py", "src/api/routes.py"],
            dirs=["src/api"],
            target_path="src/api",
        ),
        ConceptGroup(
            members=["src/store/models.py", "src/store/repository.py"],
            dirs=["src/store"],
            target_path="src/store",
        ),
    ]


def trace_responses() -> list[GeneratedResponse]:
    """Return an initial omission followed by a successful repair."""
    initial = {
        "sections": [{"title": "Core", "groups": ["g01", "g02"]}],
        "names": {
            "g01": {
                "title": "Request Routing",
                "scope": "Covers request routing but not storage behavior.",
            }
        },
    }
    repair = {
        "names": {
            "g02": {
                "title": "Persistent Record Storage",
                "scope": "Covers record persistence but not request routing.",
            }
        }
    }
    return [
        GeneratedResponse(
            content=json.dumps(initial),
            input_tokens=101,
            output_tokens=41,
            cached_tokens=7,
            stop_reason="end_turn",
            provider_stop_reason="stop",
            provider_response={
                "message": {
                    "content": json.dumps(initial),
                    "thinking": "Group the routing and persistence modules.",
                },
                "done_reason": "stop",
            },
        ),
        GeneratedResponse(
            content=json.dumps(repair),
            input_tokens=53,
            output_tokens=19,
            stop_reason="end_turn",
            provider_stop_reason="stop",
        ),
    ]


def planner_inputs(groups: list[ConceptGroup]) -> PlannerInputs:
    return PlannerInputs(
        repo_name="Recollection",
        production_files=[member for group in groups for member in group.members],
        summaries={
            "src/store": "repository.py: Persists transcript records and search indexes."
        },
    )


async def test_trace_records_calls_decisions_and_replays_without_ollama(
    tmp_path: Path,
) -> None:
    groups = trace_groups()
    inputs = planner_inputs(groups)
    responses = trace_responses()
    provider = MockProvider(model="gemma4:26b", responses=responses)
    trace_path = tmp_path / "outline.json"
    sampling = SamplingParameters(temperature=0.3, top_p=0.95, top_k=64)

    outline, _ = await name_groups(
        groups,
        inputs,
        provider=provider,
        reasoning="off",
        sampling=sampling,
        trace_path=trace_path,
        job_id="job-123",
    )

    artifact = load_outline_trace(trace_path)
    assert artifact["job_id"] == "job-123"
    assert artifact["repo_name"] == "Recollection"
    assert artifact["provider"] == "mock"
    assert artifact["model"] == "gemma4:26b"
    assert [call["request_id"] for call in artifact["calls"]] == [
        "job-123:outline:initial",
        "job-123:outline:repair",
    ]
    assert [call["request_id"] for call in provider.calls] == [
        "job-123:outline:initial",
        "job-123:outline:repair",
    ]
    assert [call["raw_response"] for call in artifact["calls"]] == [
        response.content for response in responses
    ]
    assert artifact["calls"][0]["raw_provider_response"] == responses[0].provider_response
    assert artifact["calls"][0]["sampling"] == {
        "temperature": 0.3,
        "top_k": 64,
        "top_p": 0.95,
    }
    assert artifact["calls"][0]["reasoning"] == "off"
    assert artifact["calls"][0]["provider_options"] == {}
    assert artifact["calls"][0]["input_tokens"] == 101
    assert artifact["calls"][0]["cached_tokens"] == 7
    assert artifact["calls"][0]["stop_reason"] == "end_turn"
    assert artifact["calls"][0]["response_sha256"] == text_sha256(responses[0].content)
    assert artifact["calls"][0]["parse_outcome"] == "parsed"
    assert "Recollection" in artifact["calls"][1]["user_prompt"]
    assert "Persists transcript records" in artifact["calls"][1]["user_prompt"]

    by_id = {group["group_id"]: group for group in artifact["groups"]}
    assert by_id["g01"]["initial"]["status"] == "accepted"
    assert by_id["g01"]["repair"]["status"] == "not_requested"
    assert by_id["g01"]["final"]["source"] == "initial"
    assert by_id["g02"]["initial"] == {
        "candidate_title": None,
        "reason": "missing_name",
        "status": "missing",
    }
    assert by_id["g02"]["repair"]["status"] == "accepted"
    assert by_id["g02"]["final"]["source"] == "repair"

    replay_provider = OutlineReplayProvider(trace_path)
    replay_trace = tmp_path / "replay.json"
    replay_outline, _ = await name_groups(
        groups,
        inputs,
        provider=replay_provider,
        reasoning="off",
        sampling=sampling,
        trace_path=replay_trace,
        job_id="job-123",
    )
    replay_provider.assert_complete()
    expected_titles = {page.structural_key: page.title for page in outline.pages}
    replayed_titles = {page.structural_key: page.title for page in replay_outline.pages}
    assert replayed_titles == expected_titles

    divergent_provider = OutlineReplayProvider(trace_path)
    await name_groups(
        list(reversed(groups)),
        inputs,
        provider=divergent_provider,
        reasoning="off",
        sampling=sampling,
        job_id="job-123",
    )
    with pytest.raises(AssertionError, match="differs"):
        divergent_provider.assert_complete()


async def test_trace_write_failure_does_not_discard_outline(tmp_path: Path) -> None:
    groups = trace_groups()
    blocked_parent = tmp_path / "regular-file"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    outline, report = await name_groups(
        groups,
        planner_inputs(groups),
        provider=MockProvider(responses=trace_responses()),
        trace_path=blocked_parent / "outline.json",
        job_id="job-123",
    )

    assert report.coverage == 1
    assert len(outline.pages) == 2
    assert outline.naming_mode == "llm"


def test_failed_rewrite_preserves_the_previous_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "outline.json"
    recorder = OutlineTraceRecorder(
        trace_path,
        job_id="job-123",
        repo_name="recollection",
        provider="mock",
        model="mock",
    )
    recorder.write()
    original_payload = load_outline_trace(trace_path)
    original_write_text = Path.write_text

    def fail_after_partial_write(
        target: Path,
        data: str,
        *args,
        **kwargs,
    ) -> int:
        if target.name == ".outline.json.tmp":
            original_write_text(target, "{", encoding="utf-8")
            raise OSError("simulated partial write")
        return original_write_text(target, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_after_partial_write)
    recorder.artifact.repo_name = "changed"
    recorder.write()

    assert load_outline_trace(trace_path) == original_payload
    assert not (tmp_path / ".outline.json.tmp").exists()


async def test_repair_updates_naming_mode_when_initial_response_names_nothing() -> None:
    groups = trace_groups()
    repair = {
        "names": {
            "g01": {
                "title": "Request Routing",
                "scope": "Covers request routing but not storage behavior.",
            },
            "g02": {
                "title": "Persistent Record Storage",
                "scope": "Covers record persistence but not request routing.",
            },
        }
    }
    provider = MockProvider(
        responses=[
            GeneratedResponse(content='{"names": {}}', input_tokens=1, output_tokens=1),
            GeneratedResponse(content=json.dumps(repair), input_tokens=1, output_tokens=1),
        ]
    )

    outline, _ = await name_groups(
        groups,
        planner_inputs(groups),
        provider=provider,
    )

    assert all(page.named_by_model for page in outline.pages)
    assert outline.naming_mode == "llm"


def test_json_parse_outcomes_distinguish_direct_recovery_and_failure() -> None:
    assert parse_json_object_result('{"names": {}}').outcome == "parsed"
    assert parse_json_object_result('preface {"names": {}} trailing').outcome == "recovered"
    assert parse_json_object_result("[]").outcome == "not_object"
    assert parse_json_object_result("not json").outcome == "invalid_json"
    assert parse_json_object_result("").outcome == "empty"


def test_group_summaries_are_bounded_and_cover_multiple_directories() -> None:
    group = ConceptGroup(
        members=[
            "src/api/handler.py",
            "src/api/routes.py",
            "src/store/models.py",
            "src/store/repository.py",
        ],
        dirs=["src/api", "src/store"],
        target_path="src/api",
    )
    docstrings = {
        "src/api/handler.py": "Accepts requests and dispatches handlers.",
        "src/api/routes.py": "Defines the public route table.",
        "src/store/models.py": "Defines persisted transcript records.",
        "src/store/repository.py": "Reads and writes the search index.",
    }

    summary = build_group_summaries([group], docstrings, max_chars=120)["src/api"]

    assert len(summary) <= 120
    assert "handler.py" in summary
    assert "models.py" in summary


def test_replay_rejects_non_text_and_reordered_calls(tmp_path: Path) -> None:
    trace_path = tmp_path / "outline.json"
    malformed = {
        "version": 1,
        "job_id": "job-123",
        "repo_name": "recollection",
        "provider": "mock",
        "model": "mock",
        "groups": [],
        "calls": [
            {
                "stage": "repair",
                "raw_response": {"names": {}},
            }
        ],
    }
    trace_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ValueError, match="out of order"):
        outline_responses(trace_path)

    malformed["calls"][0]["stage"] = "initial"
    trace_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="no text response"):
        outline_responses(trace_path)

    malformed["calls"][0]["raw_response"] = '{"names": {}}'
    malformed["calls"][0]["raw_provider_response"] = []
    trace_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid raw_provider_response"):
        outline_responses(trace_path)
