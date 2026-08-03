# /// script
# requires-python = ">=3.11"
# ///
"""Direct agents to the repository's Graphify extraction profile for this host."""

import argparse
import json
import re
import shlex
import sys
from collections.abc import Iterator, Sequence
from pathlib import PurePath
from typing import Any, Literal


SHELL_PUNCTUATION = ";&|\n"
ASSIGNMENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
COMMAND_WRAPPERS = {"command", "exec", "nohup"}
PROFILED_OPERATIONS = ("extract", "label", "cluster-only")
ProfiledOperation = Literal["extract", "label", "cluster-only"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("claude", "codex"), required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    command = command_from_stdin()
    if command is None:
        return 0
    operation = profiled_operation(command)
    if operation is None:
        return 0

    target: Literal["claude", "codex"] = arguments.target
    print(json.dumps(denial_for_target(target, operation), sort_keys=True))
    return 0


def command_from_stdin() -> str | None:
    """Extract a Bash command from hook JSON, making no decision on bad input."""

    try:
        payload: Any = json.load(sys.stdin)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("hook_event_name") != "PreToolUse":
        return None
    if payload.get("tool_name") != "Bash":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command")
    return command if isinstance(command, str) and command else None


def profiled_operation(command: str) -> ProfiledOperation | None:
    for invocation in command_invocations(command):
        operation = operation_from_invocation(invocation)
        if operation is not None:
            return operation
    return None


def command_invocations(command: str) -> Iterator[tuple[str, ...]]:
    """Yield argv from actual shell command positions, excluding argument prose."""

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=SHELL_PUNCTUATION)
        lexer.whitespace_split = True
        lexer.whitespace = " \t\r"
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return

    segment: list[str] = []
    for token in (*tokens, ";"):
        if is_shell_separator(token):
            invocation = invocation_from_segment(segment)
            if invocation is not None:
                yield invocation
            segment = []
            continue
        segment.append(token)


def is_shell_separator(token: str) -> bool:
    return bool(token) and all(character in SHELL_PUNCTUATION for character in token)


def invocation_from_segment(segment: Sequence[str]) -> tuple[str, ...] | None:
    index = 0
    while index < len(segment) and ASSIGNMENT_PATTERN.fullmatch(segment[index]):
        index += 1

    if index < len(segment) and PurePath(segment[index]).name == "env":
        index += 1
        while index < len(segment):
            token = segment[index]
            if token == "--" or token.startswith("-"):
                index += 1
                continue
            if ASSIGNMENT_PATTERN.fullmatch(token):
                index += 1
                continue
            break

    while index < len(segment) and PurePath(segment[index]).name in COMMAND_WRAPPERS:
        index += 1
        while index < len(segment) and segment[index].startswith("-"):
            index += 1

    if index >= len(segment):
        return None
    return tuple(segment[index:])


def operation_from_invocation(
    invocation: Sequence[str],
) -> ProfiledOperation | None:
    executable = PurePath(invocation[0]).name
    arguments = tuple(invocation[1:])
    for operation in PROFILED_OPERATIONS:
        if executable == "graphify" and arguments[:1] == (operation,):
            return operation
        if executable in {"uv", "uvx"} and adjacent_arguments(
            arguments, "graphify", operation
        ):
            return operation
        if executable in {"python", "python3"} and adjacent_arguments(
            arguments, "-m", "graphify", operation
        ):
            return operation
    return None


def adjacent_arguments(arguments: Sequence[str], *expected: str) -> bool:
    width = len(expected)
    return any(
        tuple(arguments[index : index + width]) == expected
        for index in range(len(arguments) - width + 1)
    )


def denial_for_target(
    target: Literal["claude", "codex"],
    operation: ProfiledOperation,
) -> dict[str, dict[str, str]]:
    agent_name = "Claude" if target == "claude" else "Codex"
    reason = (
        f"{agent_name} must use this repository's Graphify profile for this host. "
        f"Run `uv run python scripts/graphify_profile.py {operation}` instead. "
        "Add `--dry-run` to inspect the selected host, model, environment, and "
        "command without invoking Graphify."
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


if __name__ == "__main__":
    raise SystemExit(main())
