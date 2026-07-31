"""Tests for the deterministic architecture-diagram mermaid builder."""

from __future__ import annotations

from repowise.core.generation.architecture_mermaid import (
    ArchitectureMermaidBuilder,
    build_layer_mermaid,
    build_overview_mermaid,
    embed_mermaid,
)
from repowise.core.generation.kg_context import KnowledgeGraphContext


def _kg() -> dict:
    """Synthetic KG: two layers (UI, API), three modules, weighted imports.

    ui/a -> api/a has 15 file-pair imports (a strong, kept edge); ui/b -> api/a
    has 3 (below the overview floor, dropped); ui/a -> ui/b has 1 (an intra-UI
    edge, shown only in the UI layer diagram).
    """

    def files(mod: str, n: int) -> list[str]:
        return [f"{mod}/f{i}.py" for i in range(n)]

    ui_a, ui_b, api_a = files("ui/a", 15), files("ui/b", 3), files("api/a", 15)
    nodes = [{"id": f"file:{f}", "type": "file", "filePath": f} for f in ui_a + ui_b + api_a]

    edges: list[dict] = []
    for s, t in zip(ui_a, api_a, strict=True):  # 15 cross-layer imports ui/a -> api/a
        edges.append({"source": f"file:{s}", "target": f"file:{t}", "type": "imports"})
    for s, t in zip(ui_b, api_a[:3], strict=True):  # 3 cross-layer imports ui/b -> api/a
        edges.append({"source": f"file:{s}", "target": f"file:{t}", "type": "imports"})
    edges.append({"source": f"file:{ui_a[0]}", "target": f"file:{ui_b[0]}", "type": "imports"})

    return {
        "project": {"name": "t"},
        "nodes": nodes,
        "edges": edges,
        "layers": [
            {
                "id": "layer:ui",
                "name": "UI",
                "description": "",
                "nodeIds": [f"file:{f}" for f in ui_a + ui_b],
            },
            {
                "id": "layer:api",
                "name": "API",
                "description": "",
                "nodeIds": [f"file:{f}" for f in api_a],
            },
        ],
        "modules": [
            {
                "id": "module:ui-a",
                "name": "ui/a",
                "path": "ui/a",
                "layerId": "layer:ui",
                "nodeIds": [f"file:{f}" for f in ui_a],
            },
            {
                "id": "module:ui-b",
                "name": "ui/b",
                "path": "ui/b",
                "layerId": "layer:ui",
                "nodeIds": [f"file:{f}" for f in ui_b],
            },
            {
                "id": "module:api-a",
                "name": "api/a",
                "path": "api/a",
                "layerId": "layer:api",
                "nodeIds": [f"file:{f}" for f in api_a],
            },
        ],
    }


def _ctx(kg: dict | None = None) -> KnowledgeGraphContext:
    return KnowledgeGraphContext(None, data=kg if kg is not None else _kg())


class TestOverview:
    def test_shape_and_subgraphs(self):
        ov = build_overview_mermaid(_ctx())
        assert ov is not None
        assert ov.startswith("flowchart LR")
        # one subgraph per layer, module labels present
        assert ov.count("subgraph") == 2
        assert "ui/a" in ov and "api/a" in ov

    def test_edge_floor_keeps_strong_drops_weak(self):
        ov = build_overview_mermaid(_ctx())
        # only the count-15 cross edge clears the floor; the count-3 and count-1
        # edges are dropped
        assert '|"15"|' in ov
        assert '|"3"|' not in ov
        assert ov.count("-->") == 1
        assert ov.count("-.->") == 0

    def test_no_inline_style_theme_safe(self):
        ov = build_overview_mermaid(_ctx())
        assert "#" not in ov  # no raw hex — the renderer owns the palette
        assert "classDef" not in ov and "style " not in ov


def _kg_strong_edge_below_size_cut() -> dict:
    """Synthetic KG shaped like the corpus that shipped an edgeless overview.

    Layer ``app`` holds two large uncoupled modules (``big-a``, ``big-b``) and a
    smaller ``coupled`` module (size rank 3, below the per-layer cut of 2) that
    carries the only strong dependency: 15 imports into ``svc``'s ``service``.
    A size-only node cut selects big-a/big-b/service and can then draw no edge.
    """

    def files(mod: str, n: int) -> list[str]:
        return [f"{mod}/f{i}.py" for i in range(n)]

    big_a, big_b = files("app/big-a", 20), files("app/big-b", 18)
    coupled, service = files("app/coupled", 12), files("svc/service", 15)
    all_files = big_a + big_b + coupled + service
    nodes = [{"id": f"file:{f}", "type": "file", "filePath": f} for f in all_files]
    edges = [
        {"source": f"file:{s}", "target": f"file:{t}", "type": "imports"}
        for s, t in zip(coupled, service[: len(coupled)], strict=True)
    ] + [
        {"source": f"file:{s}", "target": f"file:{t}", "type": "imports"}
        for s, t in zip(coupled[:3], service[3:6], strict=True)
    ]  # 12 + 3 = 15 imports coupled -> service, nothing else cross-module

    def module(mid: str, path: str, layer: str, file_list: list[str]) -> dict:
        return {
            "id": f"module:{mid}",
            "name": path,
            "path": path,
            "layerId": f"layer:{layer}",
            "nodeIds": [f"file:{f}" for f in file_list],
        }

    return {
        "project": {"name": "t"},
        "nodes": nodes,
        "edges": edges,
        "layers": [
            {
                "id": "layer:app",
                "name": "App",
                "description": "",
                "nodeIds": [f"file:{f}" for f in big_a + big_b + coupled],
            },
            {
                "id": "layer:svc",
                "name": "Svc",
                "description": "",
                "nodeIds": [f"file:{f}" for f in service],
            },
        ],
        "modules": [
            module("big-a", "app/big-a", "app", big_a),
            module("big-b", "app/big-b", "app", big_b),
            module("coupled", "app/coupled", "app", coupled),
            module("service", "svc/service", "svc", service),
        ],
    }


class TestOverviewEdgeNodeCoherence:
    """The node cut must never silently starve the edge cut (edgeless hero)."""

    def test_strong_edge_endpoint_below_size_cut_is_pulled_in(self):
        ov = build_overview_mermaid(_ctx(_kg_strong_edge_below_size_cut()))
        assert ov is not None
        # the dependency the map exists to show is drawn...
        assert '|"15"|' in ov
        assert ov.count("-->") == 1
        # ...its below-the-cut endpoint becomes a node, inside its own layer
        app_subgraph = ov.split("subgraph layer_app")[1].split("end")[0]
        assert "app/coupled" in app_subgraph
        # and the size-selected anatomy is still there
        assert "app/big-a" in ov and "app/big-b" in ov and "svc/service" in ov

    def test_pulled_in_output_stays_theme_safe(self):
        ov = build_overview_mermaid(_ctx(_kg_strong_edge_below_size_cut()))
        assert "#" not in ov
        assert "classDef" not in ov and "style " not in ov

    def test_overview_with_no_strong_edges_returns_none(self):
        kg = _kg_strong_edge_below_size_cut()
        # keep only 3 of the 15 imports: below the overview floor
        kg["edges"] = kg["edges"][:3]
        assert build_overview_mermaid(_ctx(kg)) is None

    def test_overview_with_no_edges_at_all_returns_none(self):
        kg = _kg_strong_edge_below_size_cut()
        kg["edges"] = []
        assert build_overview_mermaid(_ctx(kg)) is None


class TestLayer:
    def test_shape_modules_and_intra_edges(self):
        c = _ctx()
        ui = c.get_layers()[0]
        lm = build_layer_mermaid(c, ui)
        assert lm is not None
        assert lm.startswith("flowchart TD")
        assert '"UI layer"' in lm
        assert "ui/a" in lm and "ui/b" in lm
        assert '|"1"|' in lm  # intra-layer ui/a -> ui/b

    def test_boundary_layers_as_stadium_nodes(self):
        c = _ctx()
        ui = c.get_layers()[0]
        lm = build_layer_mermaid(c, ui)
        # UI depends on API with 15 + 3 = 18 imports, drawn as a stadium node
        assert "API" in lm
        assert '(["' in lm  # stadium shape, not color, marks a boundary layer
        assert '|"18"|' in lm

    def test_no_inline_style_theme_safe(self):
        c = _ctx()
        lm = build_layer_mermaid(c, c.get_layers()[0])
        assert "#" not in lm
        assert "classDef" not in lm and "style " not in lm


class TestFallbacks:
    def test_unavailable_kg_returns_none(self):
        c = KnowledgeGraphContext(None)
        assert not c.available
        assert build_overview_mermaid(c) is None
        assert build_layer_mermaid(c, {"id": "layer:x", "name": "X"}) is None

    def test_no_modules_returns_none(self):
        kg = _kg()
        kg.pop("modules")
        c = _ctx(kg)
        assert c.available
        assert build_overview_mermaid(c) is None
        assert build_layer_mermaid(c, c.get_layers()[0]) is None

    def test_builder_reused_for_overview_and_layers(self):
        b = ArchitectureMermaidBuilder(_ctx())
        assert b.overview() is not None
        assert b.layer(_ctx().get_layers()[0]) is not None


class TestEmbed:
    def test_append_when_no_block(self):
        out = embed_mermaid("Prose.\n", "flowchart LR\nA-->B", heading="## Architecture")
        assert "## Architecture" in out
        assert out.count("```mermaid") == 1
        assert "A-->B" in out

    def test_append_is_idempotent(self):
        c1 = embed_mermaid("Prose.\n", "flowchart LR\nA-->B", heading="## Architecture")
        c2 = embed_mermaid(c1, "flowchart LR\nA-->B", heading="## Architecture")
        assert c1 == c2
        assert c2.count("```mermaid") == 1

    def test_replaces_existing_block_and_keeps_surrounding_prose(self):
        llm = "Intro\n```mermaid\ngraph TD\nX-->Y\n```\nOutro text"
        out = embed_mermaid(llm, "flowchart LR\nA-->B", heading="## X")
        assert out.count("```mermaid") == 1
        assert "A-->B" in out and "X-->Y" not in out
        assert "Intro" in out and "Outro text" in out

    def test_replace_is_idempotent(self):
        llm = "Intro\n```mermaid\ngraph TD\nX-->Y\n```\nOutro"
        r1 = embed_mermaid(llm, "flowchart LR\nA-->B", heading="## X")
        r2 = embed_mermaid(r1, "flowchart LR\nA-->B", heading="## X")
        assert r1 == r2

    def test_empty_mermaid_is_noop(self):
        assert embed_mermaid("abc", "", heading="## X") == "abc"
