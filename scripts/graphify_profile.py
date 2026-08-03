"""Run an repowise Graphify model profile for the current host.

Usage:
    uv run python scripts/graphify_profile.py extract
    uv run python scripts/graphify_profile.py label
    uv run python scripts/graphify_profile.py cluster-only
"""

import argparse
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Literal


ProfiledOperation = Literal["extract", "label", "cluster-only"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("extract", "label", "cluster-only"),
        help="Graphify operation that needs the host's model profile",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved environment and command without running Graphify",
    )
    return parser.parse_args()


def hoenn_profile(
    operation: ProfiledOperation,
) -> tuple[dict[str, str], list[str]]:
    environment = {
        "GRAPHIFY_API_TIMEOUT": "900",
        "GRAPHIFY_MAX_OUTPUT_TOKENS": "32768",
        "GRAPHIFY_NO_BACKUP": "1",
        "GRAPHIFY_OLLAMA_KEEP_ALIVE": "60m",
        "OLLAMA_API_KEY": "ollama",
        "OLLAMA_MODEL": "gemma4:31b-mxfp8-ctx64k",
    }
    command = [
        "graphify",
        operation,
        ".",
        "--backend",
        "ollama",
        "--model",
        "gemma4:31b-mxfp8-ctx64k",
    ]
    if operation == "extract":
        command.extend(["--mode", "deep", "--token-budget", "24000"])
    command.extend(["--max-concurrency", "1"])
    return environment, command


def resolve_profile(
    host_name: str,
    operation: ProfiledOperation,
) -> tuple[dict[str, str], list[str]]:
    if host_name == "hoenn":
        return hoenn_profile(operation)
    if host_name == "johto":
        raise SystemExit(
            "Johto's Graphify model and resource limits have not been measured yet."
        )
    raise SystemExit(f"No Graphify model profile is configured for host {host_name!r}.")


def print_resolution(
    host_name: str,
    operation: ProfiledOperation,
    environment: dict[str, str],
    command: list[str],
) -> None:
    print("project: repowise")
    print(f"host: {host_name}")
    print(f"operation: {operation}")
    print("environment:")
    for name, value in sorted(environment.items()):
        print(f"  {name}={value}")
    print(f"command: {shlex.join(command)}")


def main() -> int:
    arguments = parse_arguments()
    host_name = socket.gethostname().split(".", maxsplit=1)[0].casefold()
    operation: ProfiledOperation = arguments.operation
    environment_updates, command = resolve_profile(host_name, operation)
    print_resolution(host_name, operation, environment_updates, command)
    if arguments.dry_run:
        return 0

    executable = shutil.which(command[0])
    if executable is None:
        raise SystemExit("graphify is not installed or is not on PATH.")
    command[0] = executable

    environment = os.environ.copy()
    for ignored_name in (
        "GRAPHIFY_BACKEND",
        "GRAPHIFY_MAX_CONCURRENCY",
        "GRAPHIFY_TOKEN_BUDGET",
    ):
        environment.pop(ignored_name, None)
    environment.update(environment_updates)

    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command, cwd=project_root, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
