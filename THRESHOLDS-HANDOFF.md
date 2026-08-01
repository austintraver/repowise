# repowise thresholds — verified facts and handoff

Written 2026-08-01 during work on branch `codex/scale-summary-read`.
Every claim is marked **[V]** verified by executing something in this session,
or **[I]** inferred / measured by a subagent and not re-derived. Treat **[I]**
as a lead, not a fact.

---

## 1. The one-paragraph mechanism

repowise is a funnel. Each stage's thresholds decide what survives into the
next, and **the units change at every joint**: selection speaks in counts and
percentiles, prompt assembly in tokens (converted from characters by
`len(text) // 4`), storage in characters, retrieval in ranks and scores,
serving in tokens again. Every defect found in this session sat at one of those
joints — a character cap read by code expecting a different character cap, a
prompt width bound by a multiplier to a storage width, a token budget that
never reaches the page type it appears to govern.

---

## 2. Threshold map

### Which pages exist — `generation/selection/`
| Value | Default | Effect |
|---|---|---|
| `top_symbol_percentile` | 0.10 | Fraction of public symbols, PageRank-first, getting a spotlight page. **[V]** |
| `max_file_pages` | unset | Unset = size policy (ceiling ~4500); 0 = one page per file; N = hard cap. **[V]** |
| `file_page_min_symbols` | 1 | Floor for a file to earn a page. **[V]** |
| `_MIN_STOPS` (guided_tour) | 2 | Below this the guided tour page is not emitted at all. **[V]** |
| `coverage_pct` | 0.20 | Carried by the server's ranked-generate request into selection config, but selection reportedly never reads it. **[I]** — verify before trusting. |
| `max_pages_pct` | 0.20 | Retained as an internal config field and reportedly written beside `coverage_pct`, but read by no production code. **[I]** — verify before trusting. |

### What the model is told — `generation/context/assembler.py`
- **`token_budget` (48000) does not bound a model prompt.** It applies to
  `file_page`, `api_contract`, and `infra_page`; all three are structural page
  types with no model path. `assemble_module_page` contains no budget logic at
  all — no `token_budget`, no `_trim_to_budget`, no `items_within_budget`.
  **[V]** (re-derived from the three assembler methods, `structural.py`, and
  their `pertype.py` call paths.)
  The `ContextAssembler` class docstring claims every assembly method stays
  under `config.token_budget` and names a nonexistent `_assemble_with_budget`
  method. **That docstring is false.** **[V]**
- File-page budget is a strict priority waterfall **[V]**: 800 tokens reserved
  for knowledge-graph context → path + 5 → public symbol signatures while
  budget remains → private-documented → private-undocumented → imports
  (all-or-nothing) → **source code gets the remainder**.
- If source exceeds both the remainder and `large_file_source_pct × budget`, it
  is replaced by a synthesized structural outline rather than truncated. **[V]**
- Module pages are bounded only by fixed counts **[V]**: top 10 files by
  PageRank, 5 heritage entries per file, 10 key classes, each file summary cut
  to `dependency_summary_chars`.
- `estimate_tokens(text) = len(text) // 4`. **[V]** This divisor is the only
  bridge between every character cap and every token budget.

### What is stored and indexed — `persistence/`
| Value | Where | Effect |
|---|---|---|
| `STORED_SNIPPET_CHARS` = 2000 | `vector_store/lancedb_store.py` | LanceDB's stored-prefix width, which controls how deep into a page a vector hit can point and how wide a dependency-summary read it can serve. Pgvector and the in-memory store hold whole pages. **[V]** |
| `_extract_summary` default 320 | `page_generator/helpers.py` | Produces `Page.summary`, which is **full-text indexed in both backends** (`PAGE_FTS_COLUMNS`, `PG_FTS_EXPRESSION`) — it decides whether a page is *findable*. **[V]** |
| `EMBED_TEXT_MAX_CHARS` = 30000 | `vector_store/_base.py` | The text that gets vectorized is truncated here while the full page is still stored, so a very long page is searchable only by its first 30k chars. **[V]** |
| `EMBED_BATCH_MAX_ITEMS` = 16 | same | Items per embedder request. **[V]** |
| `GUIDED_TOUR_SUMMARY_CHARS` = 240 | `generation/models.py` | Blurb under each guided-tour stop. **[V]** |
| `summary_reservoir_chars` | property on `GenerationConfig` | `max(dial, 240)` — the shared in-run buffer both consumers cut from. **[V]** |

### What a query returns and in what order — `persistence/search.py`, `mcp_server/_answer_pipeline.py`
| Value | Effect |
|---|---|
| `_SNIPPET_LEN` = 200 | Evidence-window width. The coverage re-ranker reads the snippet, so this feeds **ranking**, not only display. **[V]** |
| `_PREFIX_MIN_CHARS` = 4, `_MIN_KEPT_TERMS` = 3 | Which query words count as meaningful enough to centre a window on. **[V]** |
| `_RETRIEVAL_FETCH_LIMIT` = 15 | Candidates pulled per retrieval mode. **[V]** |
| `_RRF_K` = 60 | Reciprocal-rank-fusion constant merging vector + full-text rankings. **[V]** |
| `_RRF_SCORE_SCALE` = 180.0 | Rescales fused scores into BM25's numeric range so downstream confidence gates still fire. Pure rescale — never changes ordering. **[V]** |
| `_PAGERANK_BIAS_MAX` = 0.3 | Centrality breaks ties via a [1.0, 1.3] multiplier; cannot outrank a strong text match. **[V]** |
| `_GRAPH_EXPAND_TOP_N`/`MAX_NEW`/`DAMPING` = 2/3/0.7 | Pull neighbours of strong hits in at 70% confidence, capped so a hub file can't flood the set. **[V]** |

### What an MCP agent receives
`context_token_budget` 8000 (clamped 1000–25000), `answer_max_tokens` 1024
(clamped 256–8192, word target scales 150–400), `answer_excerpt_chars` 1500,
`_EMBED_TIMEOUT_S` 8.0. **[V]** that these exist and their clamps are
documented; **[I]** that CONFIG.md's numbers match the code — not re-verified.

### What repowise claims it saved you
`mcp_server/_savings/counterfactual.py` holds ~15 token floors
(`SEARCH_FLOOR_PER_HIT` 400, `OVERVIEW_FLOOR` 1200, `RISK_PER_TARGET` 300, …)
that estimate what reading files directly would have cost, producing a
user-visible savings figure. **[V]** they exist. **Nobody has audited them** —
this is the least-examined threshold surface in the codebase.

---

## 3. Defects found and fixed this session

All **[V]** — each was reproduced before fixing and re-checked after.

1. **`dependency_summary_chars` could never exceed 200.** The store writes
   `content_snippet` at 3× the dial, but both read paths cut to a hardcoded
   `_SNIPPET_LEN = 200` first. The two numbers are equal at the default, so the
   chain looked consistent and the dial could only ever shrink summaries.
   Measured proof: every one of 1127 stored snippets in a real run is exactly
   600 chars, and the reader discarded two-thirds of each.
2. **`PgVectorStore` never received the `max_chars` parameter.** The prefetch
   passes it as a keyword → `TypeError` → caught at `except Exception` and
   logged at **debug**. Symptom on that backend: silently zero dependency
   summaries in every prompt, exit 0.
3. **`InMemoryVectorStore` cut to 500 before applying `max_chars`**, capping any
   wider request at 500.
4. **The dial accepted `True`, `2.5`, `"200"`, `-5` via direct construction**
   (the CLI builds configs that way at two sites). `[:True]` is `[:1]` — every
   dependency summary would have been one character.
5. **Constructing a `GenerationConfig` cost 0.69s** and pulled SQLAlchemy into
   the process, because validation imported a persistence constant.
6. **A 2000-char ceiling was enforced on all backends** though only LanceDB has
   it — pgvector reads an unbounded `Text` column, the in-memory store holds
   whole pages.
7. **Three production behaviors had no mutation-sensitive test.** Pgvector
   could drop both `max_chars` forwards and fall back to 500, the base batch
   implementation could omit the width when fanning out to single reads, and
   `overview_summary` could shrink its no-following-heading scan window from
   `4 * max_chars` to `max_chars`. Each mutation now turns its focused test
   red at 500 vs 733, `None` vs 733, and 598 vs 600 respectively.

### Design changes made
- LanceDB writes a flat `STORED_SNIPPET_CHARS`; the `3×` and `2×` ratios are gone.
- Reads take the caller's width; the reservoir is `max(dial, 240)`, derived from
  consumers rather than a multiplier.
- The LanceDB ceiling now lives on LanceDB (`served_width()` warns when a read
  outruns the stored prefix) instead of hard-failing every backend's config.

---

## 4. Known-dead code (verified, deliberately left in place)

- **`FilePageContext.dependency_summaries`** — written at `assembler.py:237`,
  read by nothing. No Python reader, no reference in any of ~30 `.j2`
  templates. **[V]** It is the only reservoir consumer that copies text
  *uncut*, so it looks like a third consumer and would wrongly justify widening
  the reservoir.
- **`_embed_item`'s `summary` metadata key** — LanceDB's `_row` never copies it
  and pgvector discards metadata entirely, so it reaches only the in-memory
  store, whose `persists_across_runs` is False. **[V]**
- **The three fallback read widths** (LanceDB 200, pgvector 500, in-memory 500)
  sit on a branch no production caller reaches: exactly two callers exist and
  the live one always passes `max_chars`. **[V]** Making `max_chars` required
  would delete all three.

---

## 5. Open items

1. `_savings/counterfactual.py` has never been audited.
2. `docs/architecture/deep-dives.md:643` described LanceDB's stored
   `content_snippet` as 200 chars and has been corrected to 2000. **[V]**
   `docs/reference/COMPUTED_GLOSSARY.md:265` describes the served search
   snippet, whose `_SNIPPET_LEN` remains 200, so that row is not stale. **[V]**
3. **[I]** Mixed-width store rows bias the coverage re-ranker: a 2000-wide row
   scores strictly higher than the same page at 600 on ~16.7% of query-page
   pairs, gains up to +0.333. Rows are only rewritten when a page changes, so
   mixed widths are the steady state, not a transition.
4. **[I]** A dependency summary is Overview prose when produced in-run and raw
   markdown boilerplate when read back from the store — reported different on
   400/400 real pages. A warm-store run and a cold-store run are therefore not
   the same experiment.
5. Sampling configuration for documentation and answers is designed but not
   implemented. The experiment policy is settled; the policy and defaults for
   an eventual upstream pull request remain open. See section 7. **[V]**

---

## 6. For whoever picks this up: the failure pattern to avoid

I answered "what do these thresholds do" three times and was wrong or
incomplete each time. The pattern was the same every time, and it is worth
naming because it is cheap to avoid:

**I verified the thing I was looking at, then made a claim about a neighbouring
thing.**

- Claimed both bakeoff runs had "zero retries" — I grepped the client log,
  where retries are structurally invisible. The evidence was in ollama's server
  log: one request ran 599.9s, was cancelled, and ~600s of GPU work was thrown
  away.
- Claimed a test "had teeth" after mutating `_embed_item` — which is the path
  LanceDB drops. The guided tour reads a different writer, and that one had no
  test.
- Claimed the fix was "provably a no-op at the default" — the arithmetic was
  right and the mechanism was wrong.
- Explained the dial's failure as a key mismatch between directory paths and
  file paths. Wrong: the prefetch explicitly loads file-keyed summaries. The
  real cause was a hardcoded constant equal to the default.

Three practices that would have caught all of them:

1. **Inventory before explaining.** One `rg` for module-level numeric constants
   across `generation/`, `persistence/`, `mcp_server/` returns ~40 thresholds
   across seven stages. I described four for two rounds because they were the
   ones in my diff.
2. **Mutate, don't assert.** Break the production line and confirm a test goes
   red. A hand-mutation audit found nine surviving mutations in tests I had
   just written and believed in — including the read path this branch is named
   for, which could be reverted with the suite still green.
3. **Read whole files rather than grepping.** The worst bug (pgvector) is
   invisible to `grep max_chars`, which shows only the files that already have
   it. It is obvious reading the four backends side by side.

---

## 7. Sampling configuration design — not implemented

This section records the design for adding configurable `temperature`,
`top_p`, and `top_k`. No production code, branch reference, seal, or model run
was changed while reaching it. **[V]**

### Scope and settled experiment policy

- Documentation generation and `get_answer` need independent settings. The
  documentation settings govern page prose, concept outline naming and repair,
  and knowledge graph layer and tour prose. The answer settings govern only
  answer synthesis. Austin selected this scope over implementations limited to
  documentation or to the bakeoff. **[V]**
- The documentation names are `temperature`, `top_p`, and `top_k`. The answer
  names are `answer_temperature`, `answer_top_p`, and `answer_top_k`. **[V]**
- Generation 3 must explicitly pin `temperature: 1.0`, `top_p: 0.95`, and
  `top_k: 64`. Ollama's official Gemma 4 listings call this the standardized
  sampling configuration for best performance, and the same values are present
  in the local artifacts. See the
  [26b-mxfp8 listing](https://ollama.com/library/gemma4%3A26b-mxfp8). **[V]**
- The bakeoff contract is exact or error: every configured sampling value must
  reach the model unchanged or Repowise must reject the request before
  generation. It must not clamp, omit, replace, or retry without a configured
  value. **[V]**
- The eventual upstream policy is deliberately unresolved. Austin expects that
  best effort with a precise warning may make more sense when a provider will
  alter or ignore an explicitly configured value, but asked that this remain a
  pending decision until the feature is prepared for a Repowise pull request.
  The upstream defaults are also unresolved: either all three values remain
  unset unless configured, or the existing implicit documentation and answer
  temperatures remain `0.3` and `0.2`. The bakeoff does not depend on that
  choice because it pins all three documentation values. **[V]**

### Why this is not three fields in one dataclass

- `GenerationConfig` currently exposes only `temperature`, with a default of
  `0.3`; `BaseProvider.generate` also accepts only `temperature`. **[V]**
- Concept outline naming and repair bypass that configuration with
  `temperature=0.2`. Knowledge graph layer naming and tour generation use
  fixed `temperature=0.3`. Answer synthesis uses a separate fixed
  `_SYNTHESIS_TEMPERATURE=0.2`. **[V]**
- `OllamaProvider.generate` currently sends requests through Ollama's
  OpenAI compatible chat completions endpoint. Ollama documents
  `temperature` and `top_p` on that endpoint but not `top_k`; its native
  `/api/chat` request accepts runtime generation options, and Ollama documents
  `top_k` as one of those generation parameters. Real per-request `top_k`
  control therefore requires moving generation that does not stream to the
  native chat endpoint. See Ollama's
  [compatibility fields](https://docs.ollama.com/api/openai-compatibility),
  [native chat request](https://docs.ollama.com/api/chat), and
  [generation parameters](https://docs.ollama.com/modelfile). **[V]**
- Support varies by provider and model. For example, Gemini allows `top_k` only
  on models whose metadata says it applies, while current Claude families can
  reject all three sampling parameters. A common interface cannot honestly
  promise universal support without checking the selected provider and model.
  See the
  [Gemini generation configuration](https://ai.google.dev/api/generate-content)
  and
  [Claude Messages restrictions](https://platform.claude.com/docs/en/build-with-claude/working-with-messages).
  **[V]**

The provider interface should carry one immutable `SamplingParameters` value
with optional `temperature`, `top_p`, and `top_k` fields. `None` means that the
field was not configured and should be omitted. Each provider must validate the
configured values for its selected model and reasoning mode before sending the
request. The experiment branch rejects an unsupported value;
the pending upstream policy may later replace that rejection with an explicit
warning. **[V]**

### Required production changes

1. Add the documentation fields to `GenerationConfig`, parse and validate
   them, and include them in job snapshots. Add the three independent answer
   settings to the answer dial resolver. **[V]**
2. Extend `BaseProvider.generate` and every concrete provider. Providers that
   can transmit a field must send it unchanged; providers that cannot must
   report that fact rather than silently accepting it. `MockProvider` must
   record all three so tests can observe forwarding. **[V]**
3. Move `OllamaProvider.generate` to native `/api/chat`, mapping `max_tokens`
   to `num_predict` and the three sampling values into `options`. Preserve the
   existing retry, token accounting, reasoning, timeout, and cost recording
   behavior. **[V]**
4. Forward the documentation settings through page generation,
   concept outline naming and repair, and knowledge graph enrichment. Forward
   the independent answer settings into synthesis. **[V]**
5. Replace page reuse's identity based only on the prompt with one canonical
   generation request fingerprint shared by persistent reuse and the in-memory
   cache. It must include provider, model, system and user prompts, maximum
   output tokens, all three sampling values, reasoning, and the existing source
   salt. The current persistent check compares only model plus a hash of the
   user prompt and salt. **[V]**
6. Resolve answer generation settings before the answer cache lookup. Its
   identity must include provider, model, scope, excerpt width, maximum output
   tokens, all three answer sampling values, prompt/schema version, and a wiki
   content revision. The current table is uniquely keyed by repository and
   normalized question hash, and the read path does not use its stored provider
   or model to decide reuse. **[V]**
7. Add the baseline configuration digest and candidate protocol digest to the
   bakeoff seal. `doctor` must compare the seal with the harness, restored
   configuration, model artifact, and frozen outline. Each run must record its
   requested sampling values and the actual outbound fields. **[V]**

The implementation belongs on a separate feature branch, proposed as
`codex/configure-sampling`, based on `codex/scale-summary-read`. After it passes
its mutation checks, merge it into `codex/runtime-bakeoff-3` and reseal there.
There is no reason to create another runtime branch: generation 3 has no frozen
outline and, as of this check, only the 26b run that creates the outline exists.
`codex/runtime-bakeoff-2` remains untouched at `09d6de80`. **[V]**

### Generation 3 consequence and proof requirements

The existing generation 3 output from the 26b run cannot supply the frozen
outline under the new seal because it did not run with the explicit
`1.0 / 0.95 / 64` configuration. All six candidates must run again after
resealing. No other generation 3 candidate has completed, so this is the least
expensive point to make the correction. **[V]**

Passing tests alone are not sufficient. Before the branch is accepted:

1. Mutating each of the three documentation forwarding lines must turn a
   focused test red.
2. Mutating each sampling field in the page fingerprint must make a persistent
   reuse test red.
3. Mutating each native Ollama option must make a test that captures the request
   turn red.
4. Mutating each answer forwarding line and each answer cache identity field
   must make its focused test red.
5. Mutating the sealed sampling configuration or its digest must make the
   bakeoff `doctor` test red.
6. One short live request against the installed Ollama must capture
   `temperature=1.0`, `top_p=0.95`, and `top_k=64` at the daemon boundary.
   Comparing generated prose cannot prove that a sampling field was honored.
   **[V]**
