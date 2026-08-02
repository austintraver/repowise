"""Local evidence for diagnosing and replaying concept-outline naming.

Outline naming makes only one or two model calls, but those calls decide the
titles and scopes that every module-page prompt later receives. The effective
outline alone cannot explain a bad title: it loses the prompt, the raw reply,
fallback decisions, repair decisions, and final disambiguation.

This module persists that small diagnostic surface as JSON beneath the
generation job directory. It deliberately does not trace ordinary page
generation, whose prompts contain source code and scale with the wiki.
"""

import contextlib
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import structlog

from repowise.core.providers.llm.base import (
    BaseProvider,
    CacheHint,
    GeneratedResponse,
    SamplingParameters,
)
from repowise.core.reasoning import ReasoningMode

OutlineStage = Literal["initial", "repair"]
ParseOutcome = Literal[
    "not_attempted",
    "empty",
    "parsed",
    "recovered",
    "not_object",
    "invalid_json",
]
AttemptStatus = Literal["accepted", "rejected", "missing", "error", "not_requested"]
FinalTitleSource = Literal["initial", "fallback", "repair", "disambiguation"]

logger = structlog.get_logger(__name__)


def text_sha256(text: str) -> str:
    """Return the stable digest used to compare recorded prompt material."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutlineCallTrace:
    """One initial or repair request and the provider result it produced."""

    stage: OutlineStage
    request_id: str | None
    started_at: str
    duration_seconds: float
    provider: str
    model: str
    system_prompt: str
    user_prompt: str
    max_tokens: int
    sampling: dict[str, float | int]
    reasoning: str
    provider_options: dict[str, Any]
    raw_response: str | None
    raw_provider_response: dict[str, Any] | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    stop_reason: str | None
    provider_stop_reason: str | None
    parse_outcome: ParseOutcome
    error_type: str | None
    error_message: str | None
    prompt_sha256: str
    response_sha256: str | None
    system_prompt_chars: int
    user_prompt_chars: int
    response_chars: int | None

    @classmethod
    def from_result(
        cls,
        *,
        stage: OutlineStage,
        request_id: str | None,
        started_at: str,
        duration_seconds: float,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        sampling: dict[str, float | int],
        reasoning: str,
        provider_options: dict[str, Any],
        response: GeneratedResponse | None,
        parse_outcome: ParseOutcome,
        error: BaseException | None,
    ) -> Self:
        """Build a call record without retaining provider-specific usage data."""
        raw_response = response.content if response is not None else None
        prompt_material = json.dumps(
            [system_prompt, user_prompt],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return cls(
            stage=stage,
            request_id=request_id,
            started_at=started_at,
            duration_seconds=duration_seconds,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            sampling=dict(sampling),
            reasoning=reasoning,
            provider_options=dict(provider_options),
            raw_response=raw_response,
            raw_provider_response=(
                dict(response.provider_response)
                if response is not None and response.provider_response is not None
                else None
            ),
            input_tokens=response.input_tokens if response is not None else None,
            output_tokens=response.output_tokens if response is not None else None,
            cached_tokens=response.cached_tokens if response is not None else None,
            stop_reason=response.stop_reason if response is not None else None,
            provider_stop_reason=(
                response.provider_stop_reason if response is not None else None
            ),
            parse_outcome=parse_outcome,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
            prompt_sha256=text_sha256(prompt_material),
            response_sha256=text_sha256(raw_response) if raw_response is not None else None,
            system_prompt_chars=len(system_prompt),
            user_prompt_chars=len(user_prompt),
            response_chars=len(raw_response) if raw_response is not None else None,
        )


@dataclass(frozen=True)
class TitleAttemptTrace:
    """One model proposal and whether Repowise accepted it."""

    candidate_title: str | None
    status: AttemptStatus
    reason: str | None = None


@dataclass(frozen=True)
class FinalTitleTrace:
    """The title that survived fallback, repair, and disambiguation."""

    title: str
    source: FinalTitleSource
    disambiguated: bool


@dataclass(frozen=True)
class OutlineGroupTrace:
    """Naming provenance for one immutable concept group."""

    group_id: str
    structural_key: str
    target_path: str
    fallback_title: str
    initial: TitleAttemptTrace
    repair: TitleAttemptTrace
    final: FinalTitleTrace


@dataclass
class OutlineTraceArtifact:
    """The complete local record for one generation job's outline."""

    version: int
    job_id: str
    repo_name: str
    provider: str
    model: str
    calls: list[OutlineCallTrace] = field(default_factory=list)
    groups: list[OutlineGroupTrace] = field(default_factory=list)


class OutlineTraceRecorder:
    """Write the trace after every call and provenance update."""

    def __init__(
        self,
        path: Path,
        *,
        job_id: str,
        repo_name: str,
        provider: str,
        model: str,
    ) -> None:
        self.path = path
        self.artifact = OutlineTraceArtifact(
            version=1,
            job_id=job_id,
            repo_name=repo_name,
            provider=provider,
            model=model,
        )

    def append_call(self, call: OutlineCallTrace) -> None:
        """Persist one completed or failed provider call immediately."""
        self.artifact.calls.append(call)
        self.write()

    def set_groups(self, groups: list[OutlineGroupTrace]) -> None:
        """Replace the per-group view with its latest known provenance."""
        self.artifact.groups = list(groups)
        self.write()

    def write(self) -> None:
        """Persist the current artifact as readable local JSON."""
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                json.dumps(asdict(self.artifact), indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        except OSError as error:
            logger.warning(
                "concept_outline_trace_write_failed",
                path=str(self.path),
                error=str(error),
            )
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def load_outline_trace(path: Path) -> dict[str, Any]:
    """Load a trace and reject files that cannot support deterministic replay."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"outline trace is unreadable: {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"outline trace has an unsupported format: {path}")
    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise ValueError(f"outline trace has no calls list: {path}")
    if not isinstance(payload.get("groups"), list):
        raise ValueError(f"outline trace has no groups list: {path}")
    if not isinstance(payload.get("job_id"), str) or not payload["job_id"]:
        raise ValueError(f"outline trace has no job id: {path}")
    return payload


def trace_token_value(
    call: dict[str, Any],
    field_name: str,
    *,
    position: int,
    path: Path,
) -> int:
    """Return one recorded token count after rejecting malformed values."""
    value = call.get(field_name)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"outline trace call {position} has invalid {field_name}: {path}"
        )
    return value


def outline_responses(path: Path) -> list[GeneratedResponse]:
    """Extract provider-neutral responses from a valid outline trace.

    Use :class:`OutlineReplayProvider` when replay correctness matters. This
    lower-level helper is useful for inspecting or transforming responses, but
    a plain ``MockProvider`` does not verify requests or response consumption.
    """
    payload = load_outline_trace(path)
    calls = payload["calls"]
    if not 1 <= len(calls) <= 2:
        raise ValueError(f"outline trace must contain one or two calls: {path}")
    expected_stages = ["initial"] if len(calls) == 1 else ["initial", "repair"]
    actual_stages = [
        call.get("stage") if isinstance(call, dict) else None
        for call in calls
    ]
    if actual_stages != expected_stages:
        raise ValueError(
            f"outline trace calls are out of order: expected {expected_stages!r}, "
            f"received {actual_stages!r}: {path}"
        )

    responses: list[GeneratedResponse] = []
    for position, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            raise ValueError(f"outline trace call {position} is not an object: {path}")
        raw_response = call.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError(
                f"outline trace call {position} has no text response and cannot be replayed: {path}"
            )

        stop_reason = call.get("stop_reason")
        provider_stop_reason = call.get("provider_stop_reason")
        raw_provider_response = call.get("raw_provider_response")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ValueError(
                f"outline trace call {position} has invalid stop_reason: {path}"
            )
        if provider_stop_reason is not None and not isinstance(provider_stop_reason, str):
            raise ValueError(
                f"outline trace call {position} has invalid provider_stop_reason: {path}"
            )
        if raw_provider_response is not None and not isinstance(
            raw_provider_response,
            dict,
        ):
            raise ValueError(
                f"outline trace call {position} has invalid raw_provider_response: {path}"
            )
        responses.append(
            GeneratedResponse(
                content=raw_response,
                input_tokens=trace_token_value(
                    call,
                    "input_tokens",
                    position=position,
                    path=path,
                ),
                output_tokens=trace_token_value(
                    call,
                    "output_tokens",
                    position=position,
                    path=path,
                ),
                cached_tokens=trace_token_value(
                    call,
                    "cached_tokens",
                    position=position,
                    path=path,
                ),
                stop_reason=stop_reason,
                provider_stop_reason=provider_stop_reason,
                provider_response=raw_provider_response,
            )
        )
    return responses


class OutlineReplayProvider(BaseProvider):
    """Replay an outline trace while verifying every recorded request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.artifact = load_outline_trace(path)
        self.recorded_calls = self.artifact["calls"]
        self.responses = outline_responses(path)
        self.position = 0
        self.errors: list[str] = []

    @property
    def provider_name(self) -> str:
        return "outline-replay"

    @property
    def model_name(self) -> str:
        return str(self.artifact.get("model") or "recorded-model")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        sampling: SamplingParameters = SamplingParameters(),  # noqa: B008
        request_id: str | None = None,
        reasoning: ReasoningMode = "auto",
        cache_hints: tuple[CacheHint, ...] = (),
    ) -> GeneratedResponse:
        """Return the next response only when the request matches the trace."""
        if self.position >= len(self.recorded_calls):
            message = f"outline replay received an unrecorded call {self.position + 1}"
            self.errors.append(message)
            raise ValueError(message)

        call = self.recorded_calls[self.position]
        response = self.responses[self.position]
        self.position += 1
        if not isinstance(call, dict):
            message = f"outline replay call {self.position} is not an object"
            self.errors.append(message)
            raise ValueError(message)

        comparisons = {
            "system_prompt": (system_prompt, call.get("system_prompt")),
            "user_prompt": (user_prompt, call.get("user_prompt")),
            "max_tokens": (max_tokens, call.get("max_tokens")),
            "sampling": (sampling.configured(), call.get("sampling")),
            "request_id": (request_id, call.get("request_id")),
            "reasoning": (reasoning, call.get("reasoning")),
        }
        mismatches = [
            field_name
            for field_name, (actual, recorded) in comparisons.items()
            if actual != recorded
        ]
        if cache_hints:
            mismatches.append("cache_hints")
        if mismatches:
            message = (
                f"outline replay call {self.position} differs in: "
                f"{', '.join(mismatches)}"
            )
            self.errors.append(message)
            raise ValueError(message)
        return response

    def assert_complete(self) -> None:
        """Raise when replay requests diverged or did not consume the trace."""
        if self.errors:
            raise AssertionError("; ".join(self.errors))
        if self.position != len(self.recorded_calls):
            raise AssertionError(
                f"outline replay consumed {self.position} of "
                f"{len(self.recorded_calls)} recorded calls"
            )
