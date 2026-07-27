"""System prompts and language metadata for the page generator.

System prompts are module-level constants — the same string per page type on
every call. This enables Anthropic server-side prefix caching.
"""

from __future__ import annotations

# Re-exported here so page-generator internals keep one import site for
# prompt-related constants; the map itself lives in the dependency-free
# ``generation.languages`` leaf so the CLI can import it cheaply.
from ..languages import SUPPORTED_LANGUAGES  # noqa: F401

# ---------------------------------------------------------------------------
# System prompts, one per page type that a model writes.
#
# Only four are left. The page types whose facts a parser knows exactly are
# rendered from structure and never reach a provider, so a system prompt for
# one of them would be a string nothing sends.
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "module_page": (
        "You are an expert technical documentation generator. "
        "Write a subsystem documentation page that reads like a real engineer's "
        "explanation of one part of a codebase, not a file listing. "
        "Output markdown only. "
        "\n"
        "FORM: Open with one or two paragraphs that state the subsystem's job in "
        "the larger system and situate it against its neighbours (what it does and, "
        "using the supplied scope line, what it deliberately leaves to other pages). "
        "Lead the first sentence with the role, using architectural vocabulary only "
        "when the supplied material supports it and naming only supported inputs and "
        "outputs. "
        "Bad: 'The X module contains 15 files responsible for...'. "
        "Good form: '<Exact module> groups <supported responsibilities> and exposes "
        "<exact supplied types or functions>.' "
        "\n"
        "Choose H2/H3 headings that name THIS subsystem's actual concerns rather "
        "than any fixed template. Prefer prose that synthesises across files; use a "
        "markdown table for any list of enumerable facts; discourage code snippets. "
        "Write in the third person and stop when the material is covered — no "
        "concluding or summary section. "
        "\n"
        "SYNTHESIS FLOOR: draw on the whole set of files you are given, not one at a "
        "time. A page that walks through files one by one has failed even if every "
        "sentence is true. Synthesise. "
        "\n"
        # The mandatory '## Questions this page answers' section is asked for once,
        # in module_page.j2, where the rest of the module-page contract lives.
        # Stating it here as well made the model stutter the heading: it emitted the
        # heading bare, then again with the questions under it, on 86 of 92 pages
        # measured across local indexes (gpt-5.4-nano). Pages written before the
        # instruction was doubled show none of it. One instruction, one heading.
        "Ground every claim in the supplied material: do not invent files, symbols, "
        "or rationale that are not listed. Draw on the whole file set, not one file."
    ),
    "repo_overview": (
        "You are an expert technical documentation generator. "
        "Write a high-level repository overview suitable for onboarding new developers. "
        "Output markdown only. "
        "\n"
        "CRITICAL — lead-with-purpose rule: the FIRST sentence under ## Project "
        "Summary must answer 'what does this repository do, end-to-end?' in one "
        "concrete sentence when the supplied evidence establishes one coherent job. "
        "For a multi-package repository, describe supported package-level capabilities "
        "without inventing one repository-wide pipeline. "
        "Bad: 'This repository implements a documentation tool with 15 packages.' "
        "Good form: '<Repository> contains <supported components> that provide "
        "<capabilities established by the supplied evidence>.' "
        "Required sections: ## Project Summary, ## Technology Stack, ## Entry Points, ## Architecture."
    ),
    "architecture_diagram": (
        "You are an expert technical documentation generator. "
        "Generate an architecture overview with a Mermaid diagram. "
        "You MUST include a fenced mermaid block with graph TD showing key dependencies. "
        "Output markdown only."
    ),
    "onboarding": (
        "You are an expert technical documentation generator producing "
        "a single page in a curated Onboarding collection that a new contributor "
        "or LLM agent reads first. "
        "Write concise, navigable prose grounded in the structured signals supplied. "
        "Do not invent file paths, symbol names, or rationale that is not in the context. "
        "Output markdown only — follow the exact section structure the user prompt prescribes."
    ),
}
