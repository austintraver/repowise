"""Tests for post-generation KG enrichment (Phase 11)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from repowise.core.generation import onboarding
from repowise.core.generation.kg_enrichment import enrich_tour_with_wiki_links
from repowise.core.generation.onboarding import slots as slots_module

SLOTS_PATH = Path(slots_module.__file__).resolve()
# The installed product tree, so "is anything reading this?" is asked of the
# packages rather than of the tests that assert on them.
PACKAGES_ROOT = Path(onboarding.__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakePage:
    page_id: str
    target_path: str
    page_type: str = "file_page"


# ---------------------------------------------------------------------------
# enrich_tour_with_wiki_links
# ---------------------------------------------------------------------------


class TestEnrichTourWithWikiLinks:
    def test_adds_wiki_page_id_for_a_target_path(self):
        tour = [{"order": 1, "title": "Entry", "target_path": "src/main.py"}]
        pages = [FakePage(page_id="file_page:src/main.py", target_path="src/main.py")]

        enriched_tour = enrich_tour_with_wiki_links(tour, pages)

        assert enriched_tour[0]["wikiPageIds"] == ["file_page:src/main.py"]

    def test_normalizes_the_overview_stop_to_the_overview_page(self):
        tour = [
            {
                "order": 1,
                "title": "README.md",
                "page_type": "repo_overview",
                "target_path": "README.md",
            }
        ]
        pages = [
            FakePage(
                page_id="repo_overview:example",
                target_path="example",
                page_type="repo_overview",
            )
        ]

        enriched_tour = enrich_tour_with_wiki_links(tour, pages)

        assert enriched_tour[0]["target_path"] == "example"
        assert enriched_tour[0]["wikiPageIds"] == ["repo_overview:example"]

    def test_omits_steps_without_a_materialized_page(self):
        tour = [{"order": 1, "title": "Entry", "target_path": "src/missing.py"}]

        assert enrich_tour_with_wiki_links(tour, []) == []

    def test_keeps_links_for_every_materialized_file_in_a_step(self):
        tour = [
            {
                "order": 1,
                "title": "Entry",
                "nodeIds": ["file:src/main.py", "file:src/utils.py"],
            }
        ]
        pages = [
            FakePage(page_id="file_page:src/main.py", target_path="src/main.py"),
            FakePage(page_id="file_page:src/utils.py", target_path="src/utils.py"),
        ]

        enriched_tour = enrich_tour_with_wiki_links(tour, pages)

        assert enriched_tour[0]["wikiPageIds"] == [
            "file_page:src/main.py",
            "file_page:src/utils.py",
        ]


# ---------------------------------------------------------------------------
# Onboarding slot tables
# ---------------------------------------------------------------------------


class TestOnboardingTablesAreRead:
    """Every lookup table in ``slots.py`` must have a reader in the product.

    A table that declares behaviour nothing implements is worse than no table:
    it reads as a live feature to anyone deciding what onboarding already does.
    This is derived from the source rather than a fixed list, so a new unread
    table fails here instead of quietly joining the file.

    Tests asserting a table's shape cannot catch this — they pass whether or
    not anything consults it, which is how one such table survived for months.
    """

    def _tables(self) -> dict[str, ast.AST]:
        """Module-level collection constants — the shape a lookup table takes."""
        tree = ast.parse(SLOTS_PATH.read_text())
        found: dict[str, ast.AST] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Dict | ast.Tuple | ast.List):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    found[target.id] = value
        return found

    def _readers(self, name: str) -> list[Path]:
        return [
            path
            for path in PACKAGES_ROOT.rglob("*.py")
            if path != SLOTS_PATH and name in path.read_text()
        ]

    def test_every_table_has_a_reader(self):
        tables = self._tables()

        # Anti-vacuous: a parse that found nothing would pass the loop below
        # without checking anything at all.
        assert len(tables) >= 3, f"Parsed too few tables from {SLOTS_PATH.name}: {sorted(tables)}"
        assert "ONBOARDING_ORDER" in tables

        unread = [name for name in tables if not self._readers(name)]
        assert not unread, (
            f"Declared in {SLOTS_PATH.name} and read by nothing in the product: "
            f"{sorted(unread)}. Wire it up or delete it."
        )
