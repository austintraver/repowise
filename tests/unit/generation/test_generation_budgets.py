"""The generation-side deployment dials: token_budget and dependency_summary_chars.

Both parse from config.yaml exactly the way ``max_tokens`` does, and both
default to the historical constants, so an unset config generates
byte-identical output. ``dependency_summary_chars`` drives the whole
dependency-context chain at fixed ratios: the consumption cap itself, the
in-run summary reservoir at 2x, and the embed-metadata reservoir at 3x.
"""

from datetime import UTC, datetime

import pytest

from repowise.core.generation.models import GenerationConfig
from repowise.core.generation.page_generator.helpers import overview_summary
from repowise.core.generation.page_generator.orchestrate import _embed_item


def test_config_keys_parse_like_max_tokens():
    config = GenerationConfig.from_repo_config(
        {"max_tokens": 16384, "token_budget": 96000, "dependency_summary_chars": 600}
    )
    assert config.token_budget == 96000
    assert config.dependency_summary_chars == 600


def test_defaults_are_the_historical_constants():
    config = GenerationConfig.from_repo_config({"max_tokens": 16384})
    assert config.token_budget == 48000
    assert config.dependency_summary_chars == 200


@pytest.mark.parametrize("bad", [0, -5, True, "plenty", 3.5])
@pytest.mark.parametrize("key", ["token_budget", "dependency_summary_chars"])
def test_non_positive_or_non_integer_values_are_rejected(key, bad):
    with pytest.raises(ValueError, match=key):
        GenerationConfig.from_repo_config({"max_tokens": 16384, key: bad})


def test_overview_summary_default_is_unchanged():
    body = "intro\n## Overview\n" + "x" * 3000 + "\n## Next\nrest"
    assert overview_summary(body) == overview_summary(body, 400)
    assert len(overview_summary(body)) == 400


def test_overview_summary_scales_with_the_dial():
    body = "intro\n## Overview\n" + "x" * 3000 + "\n## Next\nrest"
    assert len(overview_summary(body, 1200)) == 1200
    # No ## Overview section: plain prefix at the cap.
    assert len(overview_summary("y" * 3000, 900)) == 900


def _page(content: str):
    from repowise.core.generation.models import GeneratedPage

    now = datetime.now(UTC).isoformat()
    return GeneratedPage(
        page_id="module_page:pkg",
        page_type="module_page",
        title="Pkg",
        content=content,
        source_hash="deadbeef",
        model_name="mock",
        provider_name="mock",
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
        generation_level=4,
        target_path="pkg",
        created_at=now,
        updated_at=now,
    )


def test_embed_item_reservoirs_scale_at_fixed_ratios():
    content = "z" * 5000
    page = _page(content)

    _pid, _text, meta = _embed_item(page)  # default: 200 -> 600-char reservoir
    assert len(meta["content"]) == 600
    assert len(meta["summary"]) == 400

    _pid, _text, meta = _embed_item(page, 600)
    assert len(meta["content"]) == 1800
    assert len(meta["summary"]) == 1200
