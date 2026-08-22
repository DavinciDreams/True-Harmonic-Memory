# Benchmark: holographic engine vs. normal RAG

Compares this repo's holographic memory (`src/lib/holo`) against two
conventional retrieval baselines — brute-force cosine and an approximate
HNSW index — on a real retrieval task, on speed and quality.

All three methods share the exact same embedding (`textToSphereVector`, this
repo's own hash→hypersphere encoder), so the comparison isolates *retrieval
mechanism* (holographic engine vs. brute-force vs. approximate graph search)
rather than embedding quality.

**Dataset:** any [BEIR](https://github.com/beir-cellar/beir) dataset — part
of the [MTEB](https://github.com/embeddings-benchmark/mteb) retrieval
benchmark suite — used unmodified, evaluated with the standard metrics
(nDCG@10, Recall@100, MRR@10). Default is
[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (~3.6K
medical/nutrition docs, ~3.2K queries); [SCIDOCS](https://allenai.org/data/scidocs)
(~25.7K docs, ~7x bigger) is used as a scale test — see the root
[README](../README.md#bigger-scale-scidocs) for why HNSW's behavior
specifically depends on testing at more than one scale.

## Run it

```bash
pnpm bench:fetch [dataset]              # default: nfcorpus. Also works: scidocs, fiqa, ...
pnpm bench                              # content-only queries: full test-split query set
pnpm bench -- --dataset scidocs         # same, on a different fetched dataset
pnpm bench -- --limit 50                # quick smoke run over the first 50 judged queries
pnpm bench -- --ann                     # opt the holographic engine into its batch-built ANN
                                         # index instead of an exact scan — see the root
                                         # README's "Optional: ANN index" section
pnpm bench -- --ann --ann-m 12 --ann-ef 60   # override ann.ts's M/efConstruction defaults
pnpm bench:context                       # context-bound queries (NFCorpus-specific, see below)
pnpm bench:temporal                      # native 24-tick phase vs distributed six-plane rotor
pnpm bench:fetch:glove                   # optional: ~171MB, adds a 4th "real embedding" row to pnpm bench
```

Any BEIR dataset name works with `bench:fetch` (same public mirror for all
of them) — see `bench/fetch-data.sh` for a few more size references.
`bench/context.ts` only runs against NFCorpus: its synthetic topic tags
(see below) are keyword-matched against NFCorpus's specific vocabulary.
`bench/glove.ts` is an optional 4th row for `pnpm bench` — real GloVe word
embeddings, averaged per document, included purely as a reference point for
"how much does embedding quality matter" (this project's core engine stays
"no embeddings, no LLM" as advertised; this baseline never touches
`src/lib/holo`). Auto-skipped with a note if you haven't run
`bench:fetch:glove`. See the root
[README](../README.md#real-embeddings-for-comparison) for results — the
outcome is more interesting than "real embeddings obviously win".

NFCorpus queries carry no context, so `pnpm bench` only exercises this
engine's content-only retrieval path (full-vector rerank, see
`src/lib/holo/engine.ts`). `bench/context.ts` covers the other path: it
synthesizes topic tags via keyword match (cancer, diabetes, cholesterol,
...) shared between docs and queries, then compares context-bound retrieval
— which scores via HRR circular convolution (`src/lib/holo/hrr.ts`), not the
Clifford blade binding it used to — against both a content-only baseline on
the same store and a same-context brute-force ceiling. See the root
[README](../README.md#context-bound-queries) for results and the full
history of how that binding scheme changed.

## What it measures

| Method | Index cost | Query cost | Notes |
|---|---|---|---|
| Brute-force cosine | none | O(N) per query | Exact top-k under the shared embedding — the quality ceiling for that embedding. |
| HNSW (approximate) | O(N log N)-ish graph build | O(log N)-ish per query | From-scratch implementation in `hnsw.ts`, no native deps. Its gap from brute-force is the approximation cost; its relative speed advantage grows with corpus size — see the SCIDOCS results in the root README for where that starts to show. (An earlier version of `searchLayer()` let its working set grow unbounded instead of staying capped at `ef`, which made index-build scale far worse than `O(N log N)` in practice — fixed; see the root README for the before/after.) |
| Holographic engine | `HoloStore.addBulk` per doc (+ optional `buildContentIndex()` if `--ann`) | O(N) exact scan by default, or O(log N)-ish with `--ann` | Content-only queries are reranked against the full un-compressed embedding (see `engine.ts`'s `retrieve()`), so quality tracks brute-force closely — ties it exactly on both NFCorpus and SCIDOCS with the default exact scan. That exactness is real but not free: the engine is the slowest of the three non-GloVe methods per query at SCIDOCS scale, a genuine `O(N)` structural gap (several real per-query constant-factor fixes landed along the way — see the root README's SCIDOCS section — but none of them change the asymptotics). `--ann` (`src/lib/holo/ann.ts`) trades that exactness for real speed, same trade the HNSW baseline makes; see the root README's "Optional: ANN index" section for the measured trade-off (it doesn't beat the HNSW baseline outright, but it's a real, working, opt-in dial). |

`HoloStore.addBulk` (as opposed to `addMemory`) bypasses the interactive
demo's recent-ring eviction, dedup/reinforcement, and decay — those model
working memory and would otherwise evict most of a multi-thousand-doc corpus
by design. Bulk-loading indexes it as a permanent store instead, which is
the fair comparison against the other two methods.

## Temporal rotor ablation

`pnpm bench:temporal` is self-contained and does not ingest arbitrary
external documents. It stores 72 byte-identical versions of each of eight
semantic families at consecutive logical ticks. Content similarity can select
the family but cannot select the version, so the benchmark isolates temporal
addressing and makes the native single-plane rotor's exact 24-tick recurrence
visible. It compares no temporal signal, the native Gaussian phase window,
a six-plane incommensurate HAM-style correlation over Alex's native field,
and the same correlation with every field contribution geometrically rotated
across all six planes. This separates ranking value from field-construction
cost. Exact ticks are the oracle; the rotor remains a fuzzy secondary signal.
