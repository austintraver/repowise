# Gemma 4 Context Length Under Ollama and Repowise

Checked 2026-08-02 against the installed Ollama 0.31.2 runtime, the current
`codex/fix-generation-bounds` branch, retained bakeoff artifacts, Google
documentation, and Ollama documentation and source.

## Conclusion

The idea that Google or Ollama chose 32K because Gemma 4 26B should not be run
at 256K is based on two false premises:

1. Ollama advertises the ordinary pullable `gemma4:26b-mxfp8` tag as an MLX
   model with a 256K context window. There is no need for a separate `256k`
   weight tag because the model's trained capacity and the runtime allocation
   are separate settings.
2. Ollama did not choose 32K on Hoenn. The local LaunchAgent explicitly sets
   `OLLAMA_CONTEXT_LENGTH=32768`. Ollama's automatic policy would choose 256K
   for a host with at least 48 GiB; Hoenn has 64 GiB of unified memory.

The opposite conclusion, that 256K must therefore be Repowise's best working
context, also does not follow. It is the model's supported ceiling, not a
workload optimum. Across 482 completed page records retained by the bakeoff,
the largest input was 4,820 tokens and the largest combined input and output
was 11,861 tokens. Raising the ceiling does not supply the model with more
repository evidence.

For the 26B MLX model, retaining a 256K ceiling is reasonable when full model
capacity is desired. Ollama's MLX cache starts empty and grows in 256-token
chunks, and the selected context is described in source as a recommended
limit. No source mechanism was found by which the higher ceiling changes the
position encoding of a short prompt. This is an inference from source, not a
short-prompt quality comparison executed on Hoenn.

The same conclusion does not apply to GGUF candidates. The llama.cpp path
allocates its global KV cache according to the configured context. The
bakeoff's allocator logs measured 2,720 MiB for the 26B model at 262,144
tokens versus 85 MiB at 8,192, and 10,880 MiB for the 31B model at 262,144
versus 2,720 MiB at 65,536. A full context window is therefore a materially
different serving condition for GGUF even when requests are short.

## What Google Claims

Google lists the 26B A4B model's context length as 256K. The published
configuration uses `max_position_embeddings=262144`.

Google also measured long-context behavior rather than merely publishing a
capacity number:

- RULER accuracy for 26B A4B was 97.3 at 32K and 89.8 at 128K.
- MTOB full-book tests used approximately 256K. The 26B result declined from
  50.0 to 48.9 in one translation direction, where higher is better, and from
  45.0 to 42.7 in the other. One decline was slight and the other was clearer.

These results show that 256K is supported and evaluated. They do not show that
quality remains constant as context grows or that every application should
allocate the maximum.

Sources:

- [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Google Gemma 4 26B configuration](https://huggingface.co/google/gemma-4-26B-A4B/blob/main/config.json)
- [Gemma 4 Technical Report, Table 9](https://arxiv.org/html/2607.02770v2)

## What Ollama Distributes and Selects

Ollama's tag registry lists `gemma4:26b-mxfp8` as 28 GB, MLX, and 256K. The
local base tag does not pin `num_ctx`; its model parameters pin temperature,
top-p, and top-k. The local derived profile adds
`PARAMETER num_ctx 262144`.

Ollama 0.31.2 applies context configuration in this order:

1. Request option `num_ctx`
2. Model or Modelfile option `num_ctx`
3. `OLLAMA_CONTEXT_LENGTH`
4. Automatic host tier

Its automatic tiers are 4K below 24 GiB, 32K from 24 GiB to below 48 GiB, and
256K at or above 48 GiB. The dedicated context documentation also says larger
contexts require more memory and recommends at least 64K for agents, coding
tools, and web search. These are host and workload policies, not model quality
recommendations.

The precedence was also executed on Hoenn with the installed
`gemma4:12b-it-q4_K_M` artifact. With the LaunchAgent's 32K environment and no
request override, Ollama served 32,768. Requests carrying `num_ctx=65536` and
`num_ctx=262144` served 65,536 and 262,144 respectively. A separate temporary
Ollama server with no context environment selected 262,144 automatically on
the same 64 GiB host. The corresponding GGUF global KV allocations were 272
MiB, 544 MiB, and 2,176 MiB, confirming both request precedence and the
memory consequence on this backend.

An explicit request, model option, or environment setting opts out of the
scheduler's automatic context reduction after a load failure. Pinning 256K is
therefore deterministic but less forgiving than allowing Ollama to choose and
reduce an automatic value.

Sources:

- [Ollama Gemma 4 tags](https://ollama.com/library/gemma4/tags)
- [Ollama context length documentation](https://docs.ollama.com/context-length)
- [Ollama 0.31.2 option precedence](https://github.com/ollama/ollama/blob/v0.31.2/server/routes.go)
- [Ollama 0.31.2 scheduler fallback](https://github.com/ollama/ollama/blob/v0.31.2/server/sched.go)
- [Ollama 0.31.2 MLX context handling](https://github.com/ollama/ollama/blob/v0.31.2/x/mlxrunner/client.go)
- [Ollama 0.31.2 MLX KV cache](https://github.com/ollama/ollama/blob/v0.31.2/x/mlxrunner/cache/kvcache.go)

## Repowise's Current Contract

Repowise now accepts an optional context window in the provider section:

```yaml
ollama:
  num_ctx: 262144
```

The native `/api/chat` generation request sends that value alongside
`num_predict` and the configured sampling values. Omitting it preserves
Ollama's model, environment, or automatic default. The value is included in
Repowise's page reuse fingerprint and outline trace, so changing context under
the same model tag cannot silently reuse pages generated with the earlier
setting.

The generation defaults separately allow:

- `token_budget=48000` for assembled input context
- `max_tokens=16384` for generated output

Their sum is 64,384 before system prompt, template, and token estimation
overhead. A 32K runtime window cannot honor that theoretical envelope, and a
64K window does not leave credible margin. The defaults are not a reliable
basis for selecting a larger exact number yet: at least one module page
assembly path does not enforce `token_budget`, while the measured bakeoff
requests are much smaller than the configured ceiling.

Repowise deliberately does not turn the setting into a universal constant.
Prompt and output bounds still need a separate contract if they are ever to
guarantee that an assembled request fits, while an unset context continues to
defer to the model and server.

## Bakeoff Implications

- Existing 256K runs do not need to be discarded for output-quality ranking.
  Their prompts did not approach the limit, and all candidates used their
  declared serving profile.
- Memory comparisons must distinguish MLX from GGUF. A 256K ceiling is cheap
  for a short MLX request because its cache grows with use; it reserves a much
  larger global KV cache on the tested GGUF path.
- A future workload comparison can use a context sized to the actual corpus.
  A separate deployment comparison can evaluate each model at its advertised
  maximum. Combining those questions obscures what is being measured.
- The comment for `31b_gguf_q4km_ctx64k` in the current bakeoff configuration
  says 64K and 256K do not differ in footprint. That is contradicted by the
  later allocator measurements and should not be treated as current evidence.

## Unknowns

- A 2026-07-03 session recommended 32K as a way to reduce the context
  footprint and cold-load cost of a 27 GB GGUF answer model. The LaunchAgent
  was created about two weeks later, but no record was found that proves the
  recommendation caused the setting. The surviving evidence supports treating
  32K as a general host guardrail, not a Gemma 4 requirement.
- The exact maximum comfortable context for the 26B MLX model on Hoenn while
  macOS and other applications are active has not been measured.
- No executed comparison establishes that 32K, 64K, 128K, or 256K changes
  short-prompt output quality for the MLX model. Source inspection only shows
  that a plausible mechanism was not found.
