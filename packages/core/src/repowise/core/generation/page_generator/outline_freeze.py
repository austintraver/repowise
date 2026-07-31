"""Persist and replay the concept-naming outcome across runs.

Concept naming is the one model judgement that rewrites a module page's
instructions: the namer's title opens the page prompt and its scope sentence
is injected as a blockquote, so two runs over an identical corpus still hand
the model different briefs for every concept page. That is fine for normal
use and fatal for a controlled model comparison, where a page's instruction
must be held constant while the writer varies.

Two halves, both keyed by ``structural_key`` (the member-hash identity of a
group, stable across runs of an unchanged partition):

- every run that names its groups writes the *effective* outcome — title,
  section, order, scope for every selected module group, model-named or
  deterministic — to ``.repowise/concept-naming.json``, so any completed run
  can donate its naming;
- a run whose config sets ``frozen_outline: <path>`` replays such a file
  instead of calling the model.

The replay path fails loudly where live naming degrades quietly. Live naming
falls back to deterministic titles on any failure because a worse-named wiki
beats a broken run; a frozen run exists to hold an experimental control, and
silently un-freezing the control is exactly the failure the flag guards
against. A group missing from the map means the partition drifted from the
one the map was made for, and the run stops and says so.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..selection.selector import ModuleGroup

NAMING_ARTIFACT_NAME = "concept-naming.json"


def naming_payload(
    module_groups: "Sequence[ModuleGroup]",
    naming_mode: str,
    model_name: "str | None",
) -> dict:
    """The persisted form of a run's effective concept naming."""
    return {
        "version": 1,
        "naming_mode": naming_mode,
        "model": model_name,
        "groups": {
            group.structural_key: {
                "target_path": group.key,
                "title": group.display,
                "section": group.section,
                "order": group.order,
                "scope": group.scope,
            }
            for group in module_groups
        },
    }


def write_concept_naming(path: Path, payload: dict) -> None:
    """Write *payload* as pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_naming(path: Path) -> dict[str, dict]:
    """Load and validate a persisted naming map; return its groups mapping."""
    if not path.is_file():
        raise ValueError(f"frozen_outline file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"frozen_outline file is unreadable: {path}: {exc}") from exc
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"frozen_outline file has no 'groups' mapping: {path}")
    for key, entry in groups.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("title"), str):
            raise ValueError(f"frozen_outline entry {key!r} has no string 'title': {path}")
    return groups


def apply_frozen_naming(
    module_groups: "Sequence[ModuleGroup]",
    frozen: dict[str, dict],
) -> "list[ModuleGroup]":
    """Return *module_groups* renamed from *frozen*, strict on coverage.

    Every selected group must have an entry; a missing one means the concept
    partition no longer matches the partition the map was made for, and the
    frozen comparison the caller wanted is impossible. Entries for groups
    that no longer exist are ignored: they say a group disappeared, which the
    surviving groups' coverage check already makes visible.
    """
    missing = [g.structural_key for g in module_groups if g.structural_key not in frozen]
    if missing:
        raise ValueError(
            "frozen_outline does not cover the current concept partition; "
            f"missing structural keys: {', '.join(sorted(missing))}. "
            "The partition has drifted from the run that donated the map."
        )
    renamed = []
    for group in module_groups:
        entry = frozen[group.structural_key]
        renamed.append(
            replace(
                group,
                display=entry["title"],
                section=str(entry.get("section", "")),
                order=int(entry.get("order", 0)),
                scope=str(entry.get("scope", "")),
            )
        )
    return renamed
