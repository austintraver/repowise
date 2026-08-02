"""Resolve the project name shown to documentation models."""

import subprocess
from pathlib import Path
from urllib.parse import urlparse


def repository_name_from_remote(remote: str) -> str | None:
    """Return the final repository component from a Git remote."""
    value = remote.strip().rstrip("/")
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme:
        remote_path = parsed.path
    elif ":" in value and not value.startswith("/"):
        remote_path = value.split(":", 1)[1]
    else:
        remote_path = value

    name = Path(remote_path.rstrip("/")).name.removesuffix(".git").strip()
    return name or None


def repository_generation_name(repo_path: Path) -> str:
    """Prefer the origin name for prompts and fall back to the checkout folder."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "config",
                "--get",
                "remote.origin.url",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return repo_path.name

    if result.returncode == 0:
        remote_name = repository_name_from_remote(result.stdout)
        if remote_name:
            return remote_name
    return repo_path.name
