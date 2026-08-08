"""Post-generation knowledge-graph enrichment."""

from typing import Any


def enrich_tour_with_wiki_links(
    tour: list[dict[str, Any]],
    generated_pages: list[Any],
) -> list[dict[str, Any]]:
    """Return tour steps that point at generated pages, with their page IDs.

    This operates on the pipeline's in-memory graph result. Persistence writes
    that result afterwards, so a fresh initialization and a later update use
    the same graph data rather than attempting to repair an already-written
    artifact.
    """
    page_id_by_target_path = {
        page.target_path: page.page_id
        for page in generated_pages
        if getattr(page, "target_path", None) and getattr(page, "page_id", None)
    }
    overview_page = next(
        (
            page
            for page in generated_pages
            if getattr(page, "page_type", None) == "repo_overview"
        ),
        None,
    )
    enriched_tour: list[dict[str, Any]] = []

    for source_step in tour:
        step = dict(source_step)
        page_ids: list[str] = []
        if step.get("page_type") == "repo_overview" and overview_page is not None:
            step["target_path"] = overview_page.target_path
            page_ids.append(overview_page.page_id)
        else:
            target_path = step.get("target_path")
            if isinstance(target_path, str) and target_path in page_id_by_target_path:
                page_ids.append(page_id_by_target_path[target_path])
            for node_id in step.get("nodeIds", []):
                if not node_id.startswith("file:"):
                    continue
                page_id = page_id_by_target_path.get(node_id[5:])
                if page_id and page_id not in page_ids:
                    page_ids.append(page_id)
        if page_ids:
            step["wikiPageIds"] = page_ids
            enriched_tour.append(step)

    return enriched_tour
