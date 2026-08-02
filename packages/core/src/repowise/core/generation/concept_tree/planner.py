"""Plan the outline: group, name, validate, repair, validate again.

Two passes rather than one, because a single call cannot hold coverage and
allocation at the same time — pushing on structure made a planner drop
directories and invent paths at nine times the rate. Here the two jobs are
separated at the source, so the second pass has almost nothing to do: grouping
already guarantees coverage, and repair only ever revisits names.

The repair pass sees only what failed. Handing back the whole tree invites the
model to rewrite the parts that were fine, which is how a "fix these four
titles" request turns into a different outline.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from repowise.core.providers.llm.base import GeneratedResponse, SamplingParameters

from .grouping import ConceptGroup, GroupingParams, group_files, params_for
from .models import ConceptOutline, ConceptPage, ConceptSection, OutlineReport
from .naming import (
    NAMING_INSTRUCTIONS,
    SYSTEM_PROMPT,
    NamedGroup,
    _humanise,
    build_group_areas,
    build_payload,
    decode_response,
    deterministic_scope,
    deterministic_title,
    disambiguate_titles,
    ground_scopes,
    has_complete_area_receipt,
    parse_json_object_result,
)
from .trace import (
    FinalTitleTrace,
    OutlineCallTrace,
    OutlineGroupTrace,
    OutlineTraceRecorder,
    TitleAttemptTrace,
)
from .validation import validate_outline
from .vocabulary import bind_terms, extract_terms

logger = structlog.get_logger(__name__)

COST_OPERATION = "outline_planning"

MIN_SECTIONS = 5
MAX_SECTIONS = 11


def section_count_bounds(group_count: int) -> tuple[int, int]:
    """Return section bounds that avoid singleton sections when possible."""
    if group_count < 1:
        return 0, 0

    maximum_without_singletons = max(1, group_count // 2)
    suggested_minimum = min(MIN_SECTIONS, max(2, group_count // 8))
    suggested_maximum = max(
        suggested_minimum + 1,
        min(MAX_SECTIONS, group_count // 3 or 2),
    )
    maximum = min(suggested_maximum, maximum_without_singletons)
    minimum = min(suggested_minimum, maximum)
    return minimum, maximum


@dataclass
class PlannerInputs:
    """Everything the planner needs, already resolved by the caller.

    Deliberately plain data. The planner is then testable without a pipeline,
    a database or a provider, which matters because its interesting properties
    are about determinism and those need to be asserted cheaply and often.
    """

    repo_name: str
    #: Production source files, tests already excluded (D8).
    production_files: list[str]
    repo_root: Path | None = None
    #: File path -> curated layer id.
    layer_of_file: dict[str, str] = field(default_factory=dict)
    #: Layer id -> display name, for section fallbacks and payload hints.
    layer_labels: dict[str, str] = field(default_factory=dict)
    #: Directory -> module summary written from the code.
    summaries: dict[str, str] = field(default_factory=dict)
    entry_points: set[str] = field(default_factory=set)
    #: Test files, kept only so the validator can prove none leaked in.
    test_files: set[str] = field(default_factory=set)


def _reasoning_kwargs(reasoning: str | None) -> dict[str, str]:
    """Pass the run's reasoning setting through, or nothing when unset.

    A user who turned thinking off did so for the whole run, and a naming call
    that quietly re-enables it bills them for what they declined. Omitted
    rather than defaulted when unset so a provider that does not take the
    keyword is not handed one.
    """
    return {"reasoning": reasoning} if reasoning else {}


@contextlib.contextmanager
def _billed_as(provider: Any, operation: str) -> Iterator[None]:
    """Label this step's spend on the Costs page; no-op without a tracker."""
    tracker = getattr(provider, "_cost_tracker", None)
    if tracker is not None and hasattr(tracker, "record_as"):
        with tracker.record_as(operation):
            yield
    else:
        yield


def _build_outline(named: list[NamedGroup]) -> ConceptOutline:
    """Assemble sections from named groups, preserving the namer's order."""
    outline = ConceptOutline()
    by_section: dict[str, ConceptSection] = {}
    for entry in named:
        section = by_section.get(entry.section)
        if section is None:
            section = ConceptSection(title=entry.section)
            by_section[entry.section] = section
            outline.sections.append(section)
        section.pages.append(
            ConceptPage(
                title=entry.title,
                scope=entry.scope,
                group=entry.group,
                named_by_model=not entry.fallback,
            )
        )
    outline.number_sections()
    return outline


def _merge_thin_sections(outline: ConceptOutline) -> None:
    """Fold a section holding a single page into its neighbour.

    A heading over one page is not a grouping, it is a heading pretending to
    be one, and it makes a table of contents look padded. The page keeps its
    own title, so nothing is lost by moving it.
    """
    if len(outline.sections) <= 1:
        return
    changed = True
    while changed and len(outline.sections) > 1:
        changed = False
        for i, section in enumerate(outline.sections):
            if len(section.pages) != 1:
                continue
            target = i - 1 if i > 0 else i + 1
            outline.sections[target].pages.extend(section.pages)
            del outline.sections[i]
            changed = True
            break
    outline.number_sections()


def _disambiguate_titles(named: list[NamedGroup]) -> None:
    """Make path-derived titles unique by qualifying them with their parent.

    Two groups carved out of one directory tree — one holding a directory's own
    files, one holding its subdirectories — produce the same name from the same
    path segments. Rather than numbering them, which tells a reader nothing,
    the later one borrows a segment from further up its own path.
    """
    seen: dict[str, int] = {}
    for entry in sorted(named, key=lambda n: (n.title.lower(), n.group.target_path)):
        key = entry.title.lower()
        if key not in seen:
            seen[key] = 1
            continue
        segments = [s for s in entry.group.target_path.split("/") if s]
        for extra in reversed(segments[:-1]):
            candidate = f"{deterministic_title(entry.group)} {_humanise(extra)}".strip()
            candidate = " ".join(dict.fromkeys(candidate.split()))
            if candidate.lower() not in seen and len(candidate.split()) <= 7:
                entry.title = candidate
                break
        else:
            seen[key] += 1
            entry.title = f"{entry.title} {seen[key]}"
        seen[entry.title.lower()] = 1


def name_deterministically(groups: list[ConceptGroup], inputs: PlannerInputs) -> ConceptOutline:
    """Title and section already-computed *groups* from paths and layers only.

    Split out from :func:`plan_deterministic` so a caller that has already
    partitioned the files can name that exact partition rather than asking for
    a second one. Two callers each running the grouper is two places that have
    to agree about page identity, which is the arrangement D2 exists to
    prevent.
    """
    named = [
        NamedGroup(
            group=g,
            title=deterministic_title(g, inputs.layer_labels.get(g.dominant_layer, "")),
            scope=deterministic_scope(g),
            section=inputs.layer_labels.get(g.dominant_layer) or "Reference",
            order=i,
            fallback=True,
        )
        for i, g in enumerate(groups)
    ]
    _disambiguate_titles(named)
    outline = _build_outline(named)
    _merge_thin_sections(outline)
    outline.naming_mode = "deterministic"
    return outline


def plan_deterministic(
    inputs: PlannerInputs, *, params: GroupingParams | None = None
) -> tuple[ConceptOutline, list[ConceptGroup]]:
    """The keyless outline: real structure, names derived from paths and layers.

    Per D5 this is the same tree the LLM path produces — identical grouping,
    identical identities — with plainer titles. Adding a key upgrades the
    prose, never the shape, so a user who adds one later does not get their
    wiki re-partitioned underneath them.

    That guarantee only holds if both paths partition with the same bounds, so
    *params* is threaded rather than re-derived. It was not, once: the caller's
    bounds were dropped here and the two paths produced different page counts
    for the same repository.
    """
    groups = group_files(inputs.production_files, layer_of_file=inputs.layer_of_file, params=params)
    return name_deterministically(groups, inputs), groups


_REPAIR_INSTRUCTIONS = """\
These page titles for `{repo}` need fixing. Nothing else \
about the outline is changing — do not restructure it, and do not rename any \
page that is not listed here.

Return a corrected title AND a scope sentence for exactly these group ids.

The title is {min_words} to {max_words} words, names the capability rather than \
the directory or the layer, and is unique across the whole outline.

The scope is a full English sentence saying what the page covers and what it \
deliberately does not. It is never a path: do not echo the `target` value back.

Each group carries an `areas` list. Return an `areas` object keyed by every \
area id in that list, with one non-empty phrase saying how that area contributes \
to the title. Do not omit an area even when the group spans several directories.

Titles already in use elsewhere in the outline (do not reuse any of these):
{taken}

GROUPS TO RENAME:
{failures}

Return ONLY this JSON:
{{"names": {{"g07": {{"title": "...", "scope": "...", \
"areas": {{"a01": "...", "a02": "..."}}}}, ...}}}}
"""


def _ground_pages(outline: ConceptOutline) -> list[str]:
    """Strip unsupported citations from every page's scope. Returns the tokens."""
    from ..onboarding.grounding import check_grounding

    ungrounded: list[str] = []
    for page in outline.pages:
        cleaned, bad = check_grounding(page.scope, page.members)
        page.scope = cleaned
        ungrounded.extend(bad)
    return ungrounded


def _usable_scope(scope: str, target_path: str) -> bool:
    """Whether a returned scope is a sentence rather than an echoed path.

    Asked for a title and a scope for a group described by its directory, a
    model will sometimes hand the directory back in the scope field. That
    reads as a broken page rather than a terse one, so it is rejected and the
    deterministic sentence stands.
    """
    text = scope.strip()
    if len(text.split()) < 4:
        return False
    if text.rstrip("/") == target_path.rstrip("/"):
        return False
    # A lone path has no spaces before its first separator run; a sentence does.
    return "/" not in text.split(" ", 1)[0] or " " in text.strip()


def _force_unique_titles(outline: ConceptOutline) -> set[str]:
    """Qualify shared titles and return the structural keys that changed."""
    pages = outline.pages
    unique = disambiguate_titles([(p.title, p.target_path) for p in pages])
    changed: set[str] = set()
    for page, title in zip(pages, unique, strict=True):
        if title != page.title:
            page.title = title
            changed.add(page.structural_key)
    if changed:
        logger.info("concept_outline_titles_disambiguated", changed=len(changed))
    return changed


def _repair_targets(outline: ConceptOutline, report: OutlineReport) -> set[str]:
    """The page titles worth a second call. Everything else is left alone.

    Three kinds qualify: titles the model repeated, titles that are just the
    directory respelled, and groups the model never named at all. The last is
    the common one on a large repository — a request to name seventy groups
    reliably loses a few near the end of the list — and it is also the
    cheapest to fix, because the second call carries only the stragglers.
    """
    bad = set(report.duplicate_titles)
    targets = {p.title for p in outline.pages if p.title.lower() in bad}
    targets |= set(report.bare_directory_titles)
    targets |= {t.rsplit(" (", 1)[0] for t in report.bad_length_titles}
    targets |= {p.title for p in outline.pages if not p.named_by_model}
    return targets


def build_trace_groups(
    index: dict[str, ConceptGroup],
    named: list[NamedGroup],
    outline: ConceptOutline,
    *,
    layer_labels: dict[str, str],
    repair_outcomes: dict[str, TitleAttemptTrace] | None = None,
    disambiguated_keys: set[str] | None = None,
) -> list[OutlineGroupTrace]:
    """Describe how every structural group reached its current title."""
    named_by_key = {entry.group.structural_key: entry for entry in named}
    page_by_key = {page.structural_key: page for page in outline.pages}
    repairs = repair_outcomes or {}
    disambiguated = disambiguated_keys or set()
    records: list[OutlineGroupTrace] = []
    for group_id, group in index.items():
        entry = named_by_key[group.structural_key]
        page = page_by_key[group.structural_key]
        if entry.fallback:
            initial_status = "missing" if entry.rejection_reason == "missing_name" else "rejected"
        else:
            initial_status = "accepted"
        initial = TitleAttemptTrace(
            candidate_title=entry.candidate_title or None,
            status=initial_status,
            reason=entry.rejection_reason,
        )
        repair_attempt = repairs.get(
            group_id,
            TitleAttemptTrace(candidate_title=None, status="not_requested"),
        )
        if group.structural_key in disambiguated:
            final_source = "disambiguation"
        elif repair_attempt.status == "accepted":
            final_source = "repair"
        elif entry.fallback:
            final_source = "fallback"
        else:
            final_source = "initial"
        records.append(
            OutlineGroupTrace(
                group_id=group_id,
                structural_key=group.structural_key,
                target_path=group.target_path,
                fallback_title=deterministic_title(
                    group,
                    layer_labels.get(group.dominant_layer, ""),
                ),
                initial=initial,
                repair=repair_attempt,
                final=FinalTitleTrace(
                    title=page.title,
                    source=final_source,
                    disambiguated=group.structural_key in disambiguated,
                ),
            )
        )
    return records


async def plan_outline(
    inputs: PlannerInputs,
    *,
    provider: Any | None = None,
    deterministic: bool = False,
    params: GroupingParams | None = None,
    repair: bool = True,
    reasoning: str | None = None,
    sampling: SamplingParameters | None = None,
    trace_path: Path | None = None,
    job_id: str | None = None,
) -> tuple[ConceptOutline, OutlineReport]:
    """Group *inputs* and produce a validated outline over that grouping."""
    groups = group_files(inputs.production_files, layer_of_file=inputs.layer_of_file, params=params)
    return await name_groups(
        groups,
        inputs,
        provider=provider,
        deterministic=deterministic,
        params=params,
        repair=repair,
        reasoning=reasoning,
        sampling=sampling,
        trace_path=trace_path,
        job_id=job_id,
    )


async def name_groups(
    groups: list[ConceptGroup],
    inputs: PlannerInputs,
    *,
    provider: Any | None = None,
    deterministic: bool = False,
    params: GroupingParams | None = None,
    repair: bool = True,
    reasoning: str | None = None,
    sampling: SamplingParameters | None = None,
    trace_path: Path | None = None,
    job_id: str | None = None,
) -> tuple[ConceptOutline, OutlineReport]:
    """Name and section an already-computed partition, then validate it.

    One call names every group. The model receives opaque ids and returns a
    title, a scope and a section per id; it never decides membership, so a
    response that is late, partial, malformed or full of invented ids costs
    the wiki its titles and never its structure or its coverage.

    Falls back to :func:`name_deterministically` whenever there is no provider,
    the run is in deterministic mode, or the model's response cannot be used.
    """
    all_files = set(inputs.production_files)
    resolved = params or params_for(len(all_files))
    resolved_sampling = sampling or SamplingParameters(temperature=0.2)

    if deterministic or provider is None:
        outline = name_deterministically(groups, inputs)
        report = validate_outline(
            outline,
            all_files=all_files,
            test_files=inputs.test_files,
            max_files_per_page=resolved.max_files,
        )
        return outline, report

    trace_recorder: OutlineTraceRecorder | None = None
    if trace_path is not None:
        if not job_id:
            raise ValueError("job_id is required when outline tracing is enabled")
        trace_recorder = OutlineTraceRecorder(
            trace_path,
            job_id=job_id,
            repo_name=inputs.repo_name,
            provider=str(getattr(provider, "provider_name", "")),
            model=str(getattr(provider, "model_name", "")),
        )

    payload, index = build_payload(
        groups,
        layer_labels=inputs.layer_labels,
        summaries=inputs.summaries,
        entry_points=inputs.entry_points,
    )
    payload["repo"] = inputs.repo_name

    terms: list[str] = []
    bound: dict[str, str] = {}
    if inputs.repo_root is not None:
        terms = extract_terms(inputs.repo_root)
        bound = bind_terms(terms, index)
        for gid, term in bound.items():
            for entry in payload["groups"]:
                if entry["id"] == gid:
                    entry["suggested_name"] = term
                    break
    logger.info(
        "concept_outline_vocabulary",
        terms_found=len(terms),
        terms_bound=len(bound),
    )

    sections_lo, sections_hi = section_count_bounds(len(groups))
    instructions = NAMING_INSTRUCTIONS.format(
        repo=inputs.repo_name,
        min_words=2,
        max_words=7,
        min_sections=sections_lo,
        max_sections=sections_hi,
    )
    body = json.dumps(payload, separators=(",", ":"))
    initial_prompt = instructions + body
    initial_request_id = f"{job_id}:outline:initial" if job_id else None

    data: dict[str, Any] = {}
    initial_response: GeneratedResponse | None = None
    initial_error: BaseException | None = None
    initial_parse_outcome = "not_attempted"
    initial_started_at = datetime.now(UTC).isoformat()
    initial_started_clock = time.monotonic()
    try:
        with _billed_as(provider, COST_OPERATION):
            initial_response = await provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=initial_prompt,
                max_tokens=16000,
                sampling=resolved_sampling,
                request_id=initial_request_id,
                **_reasoning_kwargs(reasoning),
            )
        parsed = parse_json_object_result(initial_response.content or "")
        data = parsed.value
        initial_parse_outcome = parsed.outcome
    except Exception as exc:
        initial_error = exc
        logger.warning("concept_outline_naming_failed", error=str(exc))
    finally:
        if trace_recorder is not None:
            trace_recorder.append_call(
                OutlineCallTrace.from_result(
                    stage="initial",
                    request_id=initial_request_id,
                    started_at=initial_started_at,
                    duration_seconds=time.monotonic() - initial_started_clock,
                    provider=str(getattr(provider, "provider_name", "")),
                    model=str(getattr(provider, "model_name", "")),
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=initial_prompt,
                    max_tokens=16000,
                    sampling=resolved_sampling.configured(),
                    reasoning=reasoning or "auto",
                    response=initial_response,
                    parse_outcome=initial_parse_outcome,
                    error=initial_error,
                )
            )

    # Decoding sits inside the guard too. It used to sit outside, which meant a
    # response that parsed as JSON but held the wrong shape reached the decoder
    # and could take the run down — the one failure mode this design exists to
    # rule out. Decoding is defensive in its own right; this is the second line.
    try:
        named, invented_ids = decode_response(data, index, layer_labels=inputs.layer_labels)
    except Exception as exc:
        logger.warning("concept_outline_decode_failed", error=str(exc))
        named, invented_ids = decode_response({}, index, layer_labels=inputs.layer_labels)
    ungrounded = ground_scopes(named)
    if invented_ids:
        logger.warning("concept_outline_invented_group_ids", ids=invented_ids[:10])

    outline = _build_outline(named)
    _merge_thin_sections(outline)
    outline.vocabulary = {bound[gid]: index[gid].target_path for gid in bound}

    report = validate_outline(
        outline,
        all_files=all_files,
        test_files=inputs.test_files,
        max_files_per_page=resolved.max_files,
    )
    report.invented_paths = sorted(set(report.invented_paths) | set(ungrounded))
    if trace_recorder is not None:
        trace_recorder.set_groups(
            build_trace_groups(index, named, outline, layer_labels=inputs.layer_labels)
        )

    repair_outcomes: dict[str, TitleAttemptTrace] = {}
    if repair:
        targets = _repair_targets(outline, report)
        if targets:
            repair_outcomes = await _repair_titles(
                outline,
                index,
                targets,
                provider=provider,
                repo_name=inputs.repo_name,
                evidence_by_gid={
                    str(entry["id"]): dict(entry)
                    for entry in payload["groups"]
                    if isinstance(entry, dict) and "id" in entry
                },
                reasoning=reasoning,
                sampling=resolved_sampling,
                trace_recorder=trace_recorder,
                job_id=job_id,
            )
            # Repair writes new prose, so it has to face the same citation
            # check the first pass did. Grounding after the last write rather
            # than after the first is the difference between checking the page
            # and checking a draft of it.
            ungrounded.extend(_ground_pages(outline))
            report = validate_outline(
                outline,
                all_files=all_files,
                test_files=inputs.test_files,
                max_files_per_page=resolved.max_files,
            )
            report.invented_paths = sorted(set(report.invented_paths) | set(ungrounded))

    # Last line of defence on titles. Repair is a second model call and can
    # decline, run out of ids, or hand back a name already in use, so a
    # duplicate can still reach here. Identity is structural, so nothing is
    # lost when two titles collide, but a reader sees two identical rows and
    # cannot tell which is which. Qualifying them by path is the same rule the
    # deterministic path already applies, and it cannot fail.
    disambiguated_keys = _force_unique_titles(outline)
    if disambiguated_keys:
        report = validate_outline(
            outline,
            all_files=all_files,
            test_files=inputs.test_files,
            max_files_per_page=resolved.max_files,
        )
        report.invented_paths = sorted(set(report.invented_paths) | set(ungrounded))

    if trace_recorder is not None:
        trace_recorder.set_groups(
            build_trace_groups(
                index,
                named,
                outline,
                layer_labels=inputs.layer_labels,
                repair_outcomes=repair_outcomes,
                disambiguated_keys=disambiguated_keys,
            )
        )

    outline.naming_mode = (
        "llm" if any(page.named_by_model for page in outline.pages) else "deterministic"
    )

    logger.info(
        "concept_outline_planned",
        pages=report.page_count,
        sections=report.section_count,
        coverage=round(report.coverage, 4),
        invented=len(report.invented_paths),
        naming_mode=outline.naming_mode,
        # How many titles the model actually decided. ``naming_mode`` only says
        # that at least one did, so a run where the response was unusable and
        # every page fell back to its path still reads as "llm". That happened
        # and looked like success; this is the number that shows it.
        named_by_model=sum(1 for p in outline.pages if p.named_by_model),
    )
    return outline, report


async def _repair_titles(
    outline: ConceptOutline,
    index: dict[str, ConceptGroup],
    targets: set[str],
    *,
    provider: Any,
    repo_name: str,
    evidence_by_gid: dict[str, dict[str, Any]],
    reasoning: str | None = None,
    sampling: SamplingParameters | None = None,
    trace_recorder: OutlineTraceRecorder | None = None,
    job_id: str | None = None,
) -> dict[str, TitleAttemptTrace]:
    """Re-ask for just the failing titles, then apply only what improved.

    A repair that made things worse is discarded: the replacement has to be
    non-empty, correctly sized and not already in use, or the original stands.
    That keeps a bad second call from being worse than no second call.
    """
    gid_of = {g.structural_key: gid for gid, g in index.items()}
    failing = [p for p in outline.pages if p.title in targets]
    if not failing:
        return {}
    taken = sorted({p.title for p in outline.pages if p.title not in targets})

    lines: list[str] = []
    for page in failing:
        gid = gid_of.get(page.structural_key, "")
        evidence = dict(evidence_by_gid.get(gid, {}))
        if not evidence:
            evidence = {
                "id": gid,
                "target": page.target_path,
                "files": len(page.members),
                "names": sorted(m.rsplit("/", 1)[-1] for m in page.members)[:6],
                "areas": build_group_areas(page.group),
            }
        evidence["current_title"] = page.title
        lines.append(
            json.dumps(
                evidence,
                separators=(",", ":"),
            )
        )

    prompt = _REPAIR_INSTRUCTIONS.format(
        repo=repo_name,
        min_words=2,
        max_words=7,
        taken="\n".join(f"- {t}" for t in taken) or "(none)",
        failures="\n".join(lines),
    )
    # Budgeted per failing group rather than flat. Repair was written for the
    # handful a long request loses near the end of the list, and a flat 2000
    # tokens covers that. It does not cover the case that actually matters: a
    # first call that comes back with no usable names at all, where repair is
    # asked to name the whole repository and truncates, so an outline that
    # could have been recovered ships with every title derived from a path.
    # Observed on this repository at 82 groups.
    budget = max(2000, min(16000, 200 + 150 * len(failing)))
    resolved_sampling = sampling or SamplingParameters(temperature=0.2)
    repair_request_id = f"{job_id}:outline:repair" if job_id else None
    repair_response: GeneratedResponse | None = None
    repair_error: BaseException | None = None
    repair_parse_outcome = "not_attempted"
    repair_started_at = datetime.now(UTC).isoformat()
    repair_started_clock = time.monotonic()
    data: dict[str, Any] = {}
    try:
        with _billed_as(provider, COST_OPERATION):
            repair_response = await provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=budget,
                sampling=resolved_sampling,
                request_id=repair_request_id,
                **_reasoning_kwargs(reasoning),
            )
        parsed = parse_json_object_result(repair_response.content or "")
        data = parsed.value
        repair_parse_outcome = parsed.outcome
    except Exception as exc:
        repair_error = exc
        logger.warning("concept_outline_repair_failed", error=str(exc))
    finally:
        if trace_recorder is not None:
            trace_recorder.append_call(
                OutlineCallTrace.from_result(
                    stage="repair",
                    request_id=repair_request_id,
                    started_at=repair_started_at,
                    duration_seconds=time.monotonic() - repair_started_clock,
                    provider=str(getattr(provider, "provider_name", "")),
                    model=str(getattr(provider, "model_name", "")),
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    max_tokens=budget,
                    sampling=resolved_sampling.configured(),
                    reasoning=reasoning or "auto",
                    response=repair_response,
                    parse_outcome=repair_parse_outcome,
                    error=repair_error,
                )
            )

    outcomes: dict[str, TitleAttemptTrace] = {}
    if repair_error is not None:
        for page in failing:
            gid = gid_of.get(page.structural_key, "")
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=None,
                status="error",
                reason=type(repair_error).__name__,
            )
        return outcomes

    names = data.get("names")
    if not isinstance(names, dict):
        for page in failing:
            gid = gid_of.get(page.structural_key, "")
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=None,
                status="missing",
                reason="missing_names_object",
            )
        return outcomes

    in_use = {p.title.lower() for p in outline.pages}
    fixed = 0
    for page in failing:
        gid = gid_of.get(page.structural_key, "")
        entry = names.get(gid)
        if not isinstance(entry, dict):
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=None,
                status="missing",
                reason="missing_name",
            )
            continue
        raw_title = entry.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else ""
        if not has_complete_area_receipt(entry, page.group):
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=title or None,
                status="rejected",
                reason="incomplete_area_receipt",
            )
            continue
        words = len(title.split())
        if not title:
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=None,
                status="rejected",
                reason="invalid_title",
            )
            continue
        if not 2 <= words <= 7:
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=title,
                status="rejected",
                reason="bad_title_length",
            )
            continue
        if title.lower() in in_use:
            outcomes[gid] = TitleAttemptTrace(
                candidate_title=title,
                status="rejected",
                reason="duplicate_title",
            )
            continue
        in_use.discard(page.title.lower())
        in_use.add(title.lower())
        page.title = title
        page.named_by_model = True
        scope = str(entry.get("scope") or "").strip()
        if _usable_scope(scope, page.target_path):
            page.scope = scope
        outcomes[gid] = TitleAttemptTrace(
            candidate_title=title,
            status="accepted",
        )
        fixed += 1
    logger.info("concept_outline_repaired", requested=len(failing), applied=fixed)
    return outcomes
