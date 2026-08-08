"""Post-generation KG enrichment — retain linked tour steps with wiki page IDs.

After all wiki pages are generated, this module cross-references KG tour
steps with the generated page list, writes ``wikiPageIds`` for resolvable
steps, and omits stops without a materialized page. The enriched KG JSON is
written back to disk so the frontend and MCP tools can navigate safely.

No LLM call — pure dict lookup.  Runs after ``interlinking`` in the
post-generation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def enrich_tour_with_wiki_links(
    kg_json_path: Path,
    generated_pages: list[Any],
) -> int:
    """Keep tour steps with pages and add their ``wikiPageIds``.

    Returns the number of tour steps that gained at least one wiki link.
    """
    try:
        kg = json.loads(kg_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("kg_enrichment.load_failed", path=str(kg_json_path), error=str(exc))
        return 0

    tour = kg.get("tour", [])
    if not tour:
        return 0

    page_id_map: dict[str, str] = {}
    overview_page: Any | None = None
    for page in generated_pages:
        tp = getattr(page, "target_path", None)
        pid = getattr(page, "page_id", None)
        if tp and pid:
            page_id_map[tp] = pid
        if getattr(page, "page_type", None) == "repo_overview":
            overview_page = page

    enriched_tour: list[dict[str, Any]] = []
    for step in tour:
        wiki_ids: list[str] = []
        target_path = step.get("target_path")
        if step.get("page_type") == "repo_overview" and overview_page is not None:
            target_path = getattr(overview_page, "target_path", None)
            page_id = getattr(overview_page, "page_id", None)
            if isinstance(target_path, str):
                step["target_path"] = target_path
            if isinstance(page_id, str):
                wiki_ids.append(page_id)
        elif isinstance(target_path, str):
            page_id = page_id_map.get(target_path)
            if page_id:
                wiki_ids.append(page_id)
        for nid in step.get("nodeIds", []):
            if nid.startswith("file:"):
                path = nid[5:]
                page_id = page_id_map.get(path)
                if page_id and page_id not in wiki_ids:
                    wiki_ids.append(page_id)
        if wiki_ids:
            step["wikiPageIds"] = wiki_ids
            enriched_tour.append(step)

    kg["tour"] = enriched_tour

    try:
        kg_json_path.write_text(json.dumps(kg, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("kg_enrichment.write_failed", path=str(kg_json_path), error=str(exc))
        return 0

    log.info(
        "kg_enrichment.tour_wiki_links",
        total_steps=len(tour),
        steps_with_links=len(enriched_tour),
    )
    return len(enriched_tour)
