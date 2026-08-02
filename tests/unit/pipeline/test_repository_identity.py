"""Repository identity comes from Git rather than a disposable clone folder."""

from pathlib import Path

import pytest

from repowise.core.pipeline.repository_identity import (
    repository_generation_name,
    repository_name_from_remote,
)


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://github.com/example/Recollection.git", "Recollection"),
        ("git@github.com:example/Recollection.git", "Recollection"),
        ("ssh://git@github.com/example/Recollection.git", "Recollection"),
        ("/sources/Recollection.git", "Recollection"),
        ("file:///sources/Recollection.git", "Recollection"),
    ],
)
def test_repository_name_from_common_remote_formats(
    remote: str,
    expected: str,
) -> None:
    assert repository_name_from_remote(remote) == expected


def test_generation_name_falls_back_to_checkout_folder(tmp_path: Path) -> None:
    checkout = tmp_path / "baseline"
    checkout.mkdir()

    assert repository_generation_name(checkout) == "baseline"
