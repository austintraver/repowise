"""Incremental page timing records for supervised generation runs."""

import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PageTimingRecorder:
    """Append one CSV row when each model-written page becomes available."""

    FIELD_NAMES = (
        "completed_at",
        "elapsed_seconds",
        "page_id",
        "page_type",
        "title",
        "target_path",
        "provider_name",
        "model_name",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "stop_reason",
        "provider_stop_reason",
    )

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path.expanduser().resolve()
        self.started_at = time.monotonic()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite page timing data: {self.output_path}"
            )
        with self.output_path.open("x", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.FIELD_NAMES).writeheader()

    def __call__(self, page: Any) -> None:
        """Record the page at the generator's existing ready-page callback seam."""
        metadata = getattr(page, "metadata", None) or {}
        row = {
            "completed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": f"{time.monotonic() - self.started_at:.6f}",
            "page_id": page.page_id,
            "page_type": page.page_type,
            "title": page.title,
            "target_path": page.target_path,
            "provider_name": page.provider_name,
            "model_name": page.model_name,
            "input_tokens": page.input_tokens,
            "output_tokens": page.output_tokens,
            "cached_tokens": page.cached_tokens,
            "stop_reason": metadata.get("stop_reason", ""),
            "provider_stop_reason": metadata.get("provider_stop_reason", ""),
        }
        with self.output_path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=self.FIELD_NAMES).writerow(row)
