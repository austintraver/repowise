"""Tests for incremental `repowise generate --page-timings` records."""

import csv
from dataclasses import dataclass, field

import pytest

from repowise.cli.commands.generate_cmd.page_timings import PageTimingRecorder


@dataclass
class ExamplePage:
    page_id: str = "module_page:src"
    page_type: str = "module_page"
    title: str = "Source"
    target_path: str = "src"
    provider_name: str = "ollama"
    model_name: str = "gemma4:e2b-mlx-bf16-128k"
    input_tokens: int = 1200
    output_tokens: int = 450
    cached_tokens: int = 0
    metadata: dict[str, str] = field(
        default_factory=lambda: {
            "stop_reason": "complete",
            "provider_stop_reason": "stop",
        }
    )


def test_recorder_writes_page_metadata_incrementally(tmp_path) -> None:
    output_path = tmp_path / "page-timings.csv"
    recorder = PageTimingRecorder(output_path)

    recorder(ExamplePage())

    with output_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["page_id"] == "module_page:src"
    assert rows[0]["provider_name"] == "ollama"
    assert rows[0]["input_tokens"] == "1200"
    assert rows[0]["output_tokens"] == "450"
    assert rows[0]["stop_reason"] == "complete"
    assert float(rows[0]["elapsed_seconds"]) >= 0


def test_recorder_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output_path = tmp_path / "page-timings.csv"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        PageTimingRecorder(output_path)
