"""Small repository excerpts shared by high-level generation pages."""

import re
from dataclasses import dataclass

README_FILENAMES = ("README.md", "readme.md", "README.MD", "README", "Readme.md")
DECLARED_PURPOSE_CHARS = 800
MARKDOWN_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
PRESERVED_OVERVIEW_HEADINGS = {
    "primary execution flows": frozenset({"primary execution flows"}),
    "architectural clusters": frozenset(
        {"architectural clusters", "architectural communities"}
    ),
    "most central files": frozenset({"most central files", "top files by pagerank"}),
    "external systems": frozenset({"external systems"}),
    "recorded decisions": frozenset(
        {"recorded decisions", "key architectural decisions"}
    ),
    "codebase health signals": frozenset({"codebase health signals"}),
}


@dataclass(frozen=True)
class SourceExcerpt:
    path: str
    text: str


def find_root_readme(source_map: dict[str, bytes]) -> tuple[str, bytes] | None:
    """Return the first repository root README and its path."""
    for name in README_FILENAMES:
        data = source_map.get(name)
        if data:
            return name, data
    for source_path, data in source_map.items():
        if "/" not in source_path and source_path.lower() in {
            "readme",
            "readme.md",
            "readme.txt",
        }:
            return source_path, data
    return None


def extract_declared_purpose(source_map: dict[str, bytes]) -> SourceExcerpt | None:
    """Return the README's first prose paragraph within a fixed prompt budget."""
    readme = find_root_readme(source_map)
    if readme is None:
        return None

    source_path, source_bytes = readme
    text = source_bytes.decode("utf-8", errors="replace")
    for raw_paragraph in text.split("\n\n"):
        lines = [
            line.strip()
            for line in raw_paragraph.splitlines()
            if line.strip()
            and not line.lstrip().startswith(("#", "[!", "![", "<"))
        ]
        paragraph = " ".join(lines)
        if paragraph:
            return SourceExcerpt(
                path=source_path,
                text=paragraph[:DECLARED_PURPOSE_CHARS],
            )
    return None


def merge_missing_overview_sections(
    generated_content: str,
    structural_content: str,
) -> str:
    """Append factual Overview sections that the generated response omitted."""
    generated_headings = {
        match.group(1).strip().casefold()
        for match in MARKDOWN_H2.finditer(generated_content)
    }
    structural_body = structural_content.split("\n---\n", 1)[0].rstrip()
    matches = list(MARKDOWN_H2.finditer(structural_body))
    missing_sections: list[str] = []

    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        normalized_heading = heading.casefold()
        equivalent_headings = PRESERVED_OVERVIEW_HEADINGS.get(normalized_heading)
        if (
            equivalent_headings is None
            or not equivalent_headings.isdisjoint(generated_headings)
        ):
            continue
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(
            structural_body
        )
        missing_sections.append(structural_body[match.start() : section_end].strip())

    if not missing_sections:
        return generated_content
    return generated_content.rstrip() + "\n\n" + "\n\n".join(missing_sections) + "\n"
