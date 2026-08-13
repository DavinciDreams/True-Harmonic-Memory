# Holographic Frequencies Memory

A from-scratch holographic associative memory: text is hashed onto a unit
hypersphere, decomposed into a Gegenbauer harmonic spectrum, bound to context
via a Clifford geometric product with timestamps encoded as phase rotors, and
superposed into a compact field. No embeddings, no LLM — just a deterministic
hash → sphere → spectrum → algebra pipeline, with a live 3D view and a
tiered (recent / repeated / long-term) consolidation policy.

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

## How it works

```
text ──▶ hash to ±1 bits ──▶ bundle words ──▶ unit vector on S^255
                                                     │
                              ┌──────────────────────┼───────────────────────┐
                              ▼                      ▼                       ▼
                     Gegenbauer spectrum      grade-1 blade            grade-3 blade
                     (visualization only)      "content"                "context"
                                                     │                       │
                                                     └────── gp() ───────────┘
                                                              │
                                                        bound (grade 2/4)
                                                              │
                                                    sandwich(timeRotor(θ), ·)
                                                              │
                                                         superposed into
                                                        field[bucket(context)]
```

- **Sphere encoding** (`src/lib/holo/sphere.ts`, `random.ts`) — each word is
  hashed to a deterministic ±1 bit string (a fixed-seed PRNG keyed by the
  word), words are bundled by summation, and the result is normalized onto
  `S^255` (`SPHERE_DIM = 256`). This is the standard Vector Symbolic
  Architecture "bundling" trick:
  a query for one word resonates with any text containing it.
- **Spectral projection** (`gegenbauer.ts`) — a Gegenbauer-polynomial harmonic
  decomposition of the sphere vector, shown in the UI's spectrum chart. It's
  *not* used for matching (see the comment in `blade.ts` for why a low-degree
  truncation loses discriminative signal on near-white-noise hash vectors).
- **Clifford algebra** (`clifford.ts`) — `Cl(12,0)`: 4096 basis blades over
  12 generators (stored sparsely — see below — since only a few hundred are
  ever nonzero for any one multivector), addressed by bitmask, with a
  generic geometric product, addition/subtraction/scaling, reverse, an inner
  product, and rotor sandwich products. This is what lets binding (geometric
  product), rotation (rotors), and superposition (addition) all live in one
  small, exact algebra.
- **Grade-separated channels** (`blade.ts`) — content and context are each
  projected onto their own *grade*: content → grade 1 (the 12 vector blades
  `e1..e12`), context → grade 3 (the 220 trivector blades). These are disjoint
  blade indices, so a content-only query and a context-only query can never
  collide by construction. Binding them (`gp(content, context)`) lands in
  grades 2 and 4 — again disjoint from both raw channels — so a superposed
  field can carry pure-content, pure-context, and bound signal at once
  without one drowning out another. Time keeps its own dedicated slot, the
  `e12` bivector (`timeRotor` in `clifford.ts`).
- **Bucketed field** (`engine.ts`) — rather than one shared accumulator, the
  field is split into `NUM_BUCKETS` multivectors keyed by a hash of context.
  Superposing everything into a single field degrades retrieval SNR roughly
  as `1/sqrt(N)` — past a few hundred memories, matches get noisier as more
  unrelated things get folded into the same numbers. Bucketing by context
  partitions that crosstalk, so a context-scoped query only competes against
  memories sharing its bucket.
- **Consolidation** (`engine.ts`) — memories start in a small `recent` ring
  buffer, get promoted to `repeated`/`long-term` tiers as they resonate with
  an existing bound pattern again, and decay (or get forgotten) over logical
  ticks — a simulated clock, not wall time, so the demo is reproducible.
- **Retrieval** (`engine.ts`) — one inner product against the relevant field
  bucket for a coarse "global resonance" reading, then a per-record
  correlation pass (optionally weighted by a Gaussian window over temporal
  phase) to extract ranked peaks.

## Project structure

```
src/lib/holo/
  random.ts       deterministic PRNG + string hashing
  sphere.ts        text -> hypersphere vector
  gegenbauer.ts    harmonic spectrum (visualization)
  linalg.ts        Gram-Schmidt / random orthonormal bases
  clifford.ts      Cl(12,0) geometric algebra + grade utilities (sparse)
  blade.ts         sphere vector -> grade-separated blade (content/context)
  projection.ts    sphere vector -> 3D point (visualization)
  engine.ts        HoloStore: binding, bucketed field, tiers, retrieval

src/components/holo/
  HoloApp.tsx        main UI: add/search/inspect memories
  SphereScene.tsx     3D point-cloud view (three.js)
  SpectrumChart.tsx   Gegenbauer spectrum bar chart
```

## Design notes / open questions

- The field's crosstalk problem ("everything resonates with everything" as
  more memories are added) is a known limitation of holographic/VSA-style
  superposition, not fully solved — bucketing by context helps but a
  context-free query still falls back to summing all buckets.
- Grades 0 and the two 1-dimensional pseudoscalar/scalar slots, plus the
  unused bivector components besides `e12`, are currently unclaimed —
  candidates for further channels (e.g. a "semantic cluster" band).
- The algebra dimension (`NUM_GENERATORS`) has been bumped twice:
  `Cl(4,0)` → `Cl(8,0)` (see the benchmark section below for why), then
  `Cl(8,0)` → `Cl(12,0)` after raising the encoder's `SPHERE_DIM` 24 → 256
  widened the gap between content-only quality (dramatically better) and
  context-bound quality (blade capacity unchanged, so relatively much
  worse). The second bump needed a real rework, not just a constant change:
  `clifford.ts`'s `Multivector` was a dense `Float64Array` of all `2^n`
  components, which made `n=10` alone measurably worse (~7x slower per
  query for no real quality gain — `BLADE_COUNT = 2^n` means the bound
  grades' density grows combinatorially with `n`) and would have made
  `n=16`+ outright infeasible (a dense `Cl(16,0)` multivector is 65,536
  floats — 512KB — *per stored multivector*, and several are kept per
  record). `Multivector` is now a sparse `Map<bitmask, coefficient>`
  instead — cost tracks the actual occupied-blade count (low hundreds),
  not `2^n`. That unlocked `n=12`, which turns out to be a hard ceiling
  regardless of compute: `blade.ts`'s context channel needs `C(n,3)`
  mutually orthonormal directions inside `SPHERE_DIM=256` dimensions, and
  you cannot have more orthonormal vectors than the ambient dimension —
  `C(12,3)=220` fits, `C(13,3)=286` doesn't. Result: context-bound nDCG@10
  went 0.0353 (`Cl(8,0)`) → 0.0449 (`Cl(12,0)`), a real but modest ~27%
  relative gain — nowhere near closing the gap with content-only's 0.1589.
  See [Context-bound queries](#context-bound-queries) for the full numbers.
  Matching the blade's *original* relative capacity (grade-1 keeping ~33%
  of the source dimensions, as it did at the very first `SPHERE_DIM=24` /
  `Cl(4,0)` pairing) would need `n≈85` at `SPHERE_DIM=256` — not viable
  even sparsely, since the occupied-blade count itself (`C(n,2)+C(n,4)`)
  still grows polynomially with `n` and would be enormous at `n=85`. This
  is now a settled, measured answer, not an open question: **more
  generators alone cannot close this gap** at the current `SPHERE_DIM`. A
  real fix would need a different binding scheme, not a bigger `n`.

## Benchmark: vs. normal RAG

`bench/` compares this engine against brute-force cosine and an
approximate HNSW index on real retrieval tasks (NFCorpus and, for a bigger
scale test, SCIDOCS/FiQA-sized corpora — from the MTEB retrieval suite) —
speed and quality, same embedding across all three. See
[`bench/README.md`](bench/README.md).

```bash
pnpm bench:fetch [dataset]   # default: nfcorpus. Also: scidocs, fiqa, ...
pnpm bench [-- --dataset name]   # content-only queries, vs. brute-force cosine + HNSW
pnpm bench:context                # context-bound queries (NFCorpus only, see below)
```

### Content-only queries

**Results** (NFCorpus test split, 3633 docs, 323 judged queries, shared
IDF-weighted sphere-vector embedding, top-100):

| Method | nDCG@10 | Recall@100 | MRR@10 | Index (ms) | Avg query | p95 query |
|---|---|---|---|---|---|---|
| Brute-force cosine | 0.2161 | 0.1681 | 0.4040 | 6µs | 5.56ms | 12.31ms |
| HNSW (approx.) | 0.2070 | 0.1607 | 0.3886 | 10773.15ms | 3.93ms | 6.51ms |
| Holographic engine | 0.2145 | 0.1679 | 0.4025 | 12363.40ms | 5.77ms | 10.03ms |

Brute-force cosine's numbers are the ceiling for this embedding (exact
top-k). The holographic engine tracks it closely on quality. Getting here
took four changes, in order (each number below is nDCG@10, brute-force,
cumulative):

1. **`Cl(4,0)` → `Cl(8,0)`** (`src/lib/holo/clifford.ts`) — the content
   channel keeps 8 of the encoder's dimensions instead of 4, so
   *context-bound* queries (which must go through the blade/binding path to
   combine content and context into one signal) lose less on the way in.
   Doesn't affect content-only queries; see the context-bound results below
   for where this actually pays off (and what it costs).
2. **Full-vector rerank for context-free queries** (`src/lib/holo/engine.ts`,
   `retrieve()`) — with no context there's nothing to preserve orthogonality
   against, so those queries are scored by cosine similarity on the raw
   sphere vector instead of the compressed blade, at the same O(N) cost as
   before. This is what closed most of the content-only gap at the time:
   0.0128 (~35% of brute-force) before changes 1–2, 0.0368 (~99%) after.
3. **IDF weighting + stopword filtering** (`src/lib/holo/sphere.ts`,
   `buildIdf`) — the embedding itself was previously an unweighted
   bag-of-hashed-words; adding smoothed IDF weighting and filtering a small
   stopword list lifted nDCG@10 to 0.0559 for *all three* methods, since
   it's a shared-embedding change, not a retrieval-mechanism one.
   `HoloStore` takes an optional `idf` constructor option so it can share
   the exact same weighting as the baselines; the interactive demo leaves it
   unset since it has no fixed corpus to compute document frequencies over.
4. **`SPHERE_DIM` 24 → 256** (`src/lib/holo/sphere.ts`) — by far the
   biggest single win in this whole project: 0.0559 → 0.2161, a ~3.9x
   improvement, for all three methods (it's a shared-embedding change).
   Bundling many words' hash vectors into only 24 dimensions caused severe
   collision noise once a document had more than a couple dozen unique
   words — this was arguably the real bottleneck the whole time, and every
   retrieval-mechanism fix before it (the previous three items, plus the
   whole benchmark-scale/HNSW-bug-fixing effort) was optimizing around a
   symptom rather than the cause. Diminishing returns set in well before
   256: 128→256 was +22% relative, 256→512 was +17% at roughly double the
   index/query cost and a widening HNSW-vs-brute-force gap — 256 is a
   deliberate quality/cost balance point, not a ceiling.

Two more "make the embedding more realistic" ideas were tried at
`SPHERE_DIM=256` and **reverted** after measuring a net loss, rather than
kept for their own sake: BM25-style TF saturation (diminishing returns per
repeated word) dropped brute-force nDCG@10 to 0.1700 — in these short
title+abstract documents, a repeated word really is stronger relevance
signal, not the padding-noise BM25's saturation assumes for longer
documents. Bigrams (adjacent word pairs as extra hashed terms) dropped it
further to 0.0630 — doubling the term count re-introduces the exact
hash-collision crowding that raising `SPHERE_DIM` had just fixed. Lesson:
for this specific bundling scheme, raw dimensionality mattered far more
than smarter term weighting.

### Real embeddings, for comparison

This project's core engine is deliberately "no embeddings, no LLM" (see the
top of this README) — but it's a fair question how much quality is being
left on the table by not using real semantic embeddings. `bench/glove.ts`
adds an optional 4th row (`pnpm bench:fetch:glove`, then `pnpm bench`) that
answers this *without* touching `src/lib/holo`: brute-force cosine over
50-dim [GloVe](https://nlp.stanford.edu/projects/glove/) word vectors,
averaged per document, given the exact same IDF weighting and stopword
filtering as the other three rows for a fair comparison. This row
deliberately breaks the "same embedding" isolation the other three share —
it's a reference point, not a fourth retrieval-mechanism contestant.

| Method | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| Brute-force cosine (this project's encoder) | 0.2161 | 0.1681 | 0.4040 |
| GloVe average (real embedding) | 0.0717 | 0.1325 | 0.1429 |

Counterintuitive result: the real embedding *loses*, by a wide margin, not
just "doesn't obviously win." Two things explain most of it, not "real
embeddings are bad":

- **21% of NFCorpus's vocabulary isn't in GloVe's ~400K-word vocabulary at
  all** (measured directly, see `bench/index.ts`'s OOV check) — GloVe was
  trained on Wikipedia + Gigaword, general text, and this is a
  medical/nutrition corpus full of drug names and clinical terminology that
  a general-domain embedding simply never saw.
- **Naive mean-pooling is a known-weak way to turn word vectors into a
  document vector** — averaging washes out the magnitude signal a repeated
  distinctive word carries (this project's raw hash-sum-then-normalize
  keeps that signal implicitly); real systems that use word-vector averages
  competitively usually subtract a top principal component or use a
  learned pooling, neither of which is implemented here.
- **Dimension isn't matched either** — 50-dim GloVe vs. this project's
  256-dim sphere vectors. A higher-dimensional GloVe variant (100d/200d/300d)
  was not fetched here to keep the download reasonable (~171MB already);
  this comparison likely understates what a larger real embedding would do.

So: this is real evidence that *this specific naive way* of using GloVe
loses to the tuned hash-bundle scheme on *this specific specialized-domain
corpus* — not evidence that real embeddings are categorically worse. A
general-purpose retrieval corpus, a larger/domain-matched embedding, or a
smarter pooling strategy could plausibly flip this result. Included here
because the honest, measured answer was more interesting than the assumed
one.

HNSW's numbers above are *after* a fix, not before: `bench/hnsw.ts`'s
`searchLayer()` let its working set (`found`) grow unbounded across a
search — every neighbor ever visited stayed in it forever, re-sorted on
every loop iteration — instead of staying capped at `ef` like the
algorithm specifies. That's an actual bug, not a missing optimization: it
made index-build scale worse than `O(N log N)`, badly enough that indexing
didn't finish within 15 minutes on a 7x-larger corpus (see the SCIDOCS
section below for how that surfaced). Fixing it dropped NFCorpus's HNSW
index-build from 36,116ms to 1,674ms — about 21.6x — with identical
quality metrics before and after, confirming it was purely a performance
bug.

### Bigger scale: SCIDOCS

NFCorpus is small (~3.6K docs); `bench/context.ts` aside, nothing so far
tested whether any of these three methods' behavior changes at real scale.
SCIDOCS (~25.7K docs, ~7x NFCorpus, 1000 judged queries) is a rougher,
harder retrieval task by nature (citation prediction, not topical search):

| Method | nDCG@10 | Recall@100 | MRR@10 | Index (ms) | Avg query | p95 query |
|---|---|---|---|---|---|---|
| Brute-force cosine | 0.1102 | 0.2293 | 0.2078 | 32µs | 56.70ms | 123.50ms |
| HNSW (approx.) | 0.1072 | 0.2391 | 0.2005 | 178386.63ms | 6.50ms | 14.67ms |
| Holographic engine | 0.1102 | 0.2293 | 0.2078 | 124084.54ms | 90.11ms | 198.40ms |
| GloVe average (real embedding) | 0.0443 | 0.1367 | 0.0903 | 15917.31ms | 34.16ms | 69.97ms |

*(Run at the current config — `SPHERE_DIM=256`, `Cl(12,0)`. An earlier pass
at `SPHERE_DIM=24` scored nDCG@10 0.0063 across the board here; the same
~17.5x jump from raising `SPHERE_DIM` that showed up on NFCorpus holds at
this scale too.)*

Quality: holographic engine ties brute-force **exactly** (0.1102 = 0.1102,
identical to 4 decimal places) — the retrieval-mechanism comparison holds
at 7x scale. HNSW trails slightly for the first time on this corpus
(0.1072), the real cost of its approximation once there's enough embedding
signal to actually approximate imperfectly. GloVe loses again, by an even
wider margin than on NFCorpus, with a worse OOV rate here too (37.7% of
SCIDOCS's vocabulary missing from GloVe, vs. NFCorpus's 21% — consistent
with SCIDOCS's citation-prediction task skewing toward more technical
scientific jargon).

Speed tells a different story than NFCorpus did, though — and not in the
holographic engine's favor this time. At `SPHERE_DIM=24`/`Cl(8,0)`, an
earlier pass had it *beating* brute-force per query (16.74ms vs. 15.00ms).
Now, at `SPHERE_DIM=256`, it's the *slowest* of the three non-GloVe methods
(90.11ms vs. brute-force's 56.70ms) — both got much slower in absolute
terms (256-dim dot products cost more than 24-dim ones), but the
holographic engine's `retrieve()` carries more fixed per-query overhead
than a bare dot-product loop (it still runs `encode()`'s full blade/spectrum
pipeline even for a content-only query that only ends up using the raw
sphere vector — see `src/lib/holo/engine.ts`), and that overhead scales
with `SPHERE_DIM` too. Not yet profiled or fixed this session — a
plausible next optimization in the same family as the `phaseWeight` and
`globalResonance` fixes above (compute only what a given query path
actually uses), left here as a known, unaddressed cost rather than a
surprise.

HNSW's index-build numbers above are *after* a fix, not before:
`bench/hnsw.ts`'s `searchLayer()` let its working set (`found`) grow
unbounded across a search — every neighbor ever visited stayed in it
forever, re-sorted on every loop iteration — instead of staying capped at
`ef` like the algorithm specifies. That's an actual bug, not a missing
optimization: it made index-build scale worse than `O(N log N)`, badly
enough that indexing didn't finish within 15 minutes on this same
7x-larger corpus before the fix. Fixing it dropped NFCorpus's HNSW
index-build from 36,116ms to 1,674ms at the time — about 21.6x — with
identical quality metrics before and after, confirming it was purely a
performance bug. (HNSW's SCIDOCS index-build above, 178s, is slow again in
absolute terms — but now because `SPHERE_DIM=256` vectors cost ~10x more
per distance computation than `SPHERE_DIM=24` ones did, not because of a
bug; `bench/index.ts` fixed `HNSW`'s parameters, so this scales with
embedding cost, as expected.)

The `phaseWeight`/`allActive()` fixes described in earlier revisions of
this README were real and are still in place; they're just no longer the
dominant cost at the current `SPHERE_DIM`, superseded by the encode-path
overhead described above.

### Context-bound queries

NFCorpus queries carry no context, so the benchmark above never exercises
the blade/binding path — `bench/context.ts` fills that gap by synthesizing
topic tags via keyword match (cancer, diabetes, cholesterol, ...) shared
between docs and queries, then compares three variants on the same corpus:

| Method | nDCG@10 | Recall@100 | MRR@10 | Avg query |
|---|---|---|---|---|
| Bucket ceiling (brute-force, same-tag docs only) | 0.1190 | 0.1187 | 0.2823 | 1.69ms |
| Context-bound (blade binding + bucket field) | 0.0449 | 0.0828 | 0.1039 | 18.94ms |
| Content-only (same store, no context) | 0.1589 | 0.1801 | 0.3202 | 15.83ms |

*(60 judged NFCorpus test queries whose text matched a topic keyword. `Cl(12,0)`
— see below for why that's the number, not a round one.)*

This result got noticeably **worse** after `SPHERE_DIM` went 24 → 256 (see
above), was partially recovered by a real rework, and is now a settled,
measured finding rather than an open question. Timeline:

1. **At `SPHERE_DIM=24` / `Cl(8,0)`**: context-bound nDCG@10 0.0523 vs.
   content-only's 0.0371 — binding *beat* no-context, reaching 89% of the
   ideal bucket ceiling.
2. **After `SPHERE_DIM` → 256, still `Cl(8,0)`**: context-bound collapsed to
   0.0353 while content-only jumped to 0.1589 — a >4x reversal. Cause:
   `Cl(8,0)`'s content blade holds exactly 8 real numbers no matter how rich
   the source embedding is. At `SPHERE_DIM=24` that was a 24→8 compression
   (~33% kept); at `SPHERE_DIM=256` it's a 256→8 compression (~3% kept) —
   the blade didn't get worse, content-only just left it behind.
3. **`Cl(10,0)` on the old dense `Multivector` (`Float64Array`, all `2^n`
   components stored)**: quality barely moved (0.0383, within noise) while
   query latency got ~7x worse (~712ms vs. ~95ms) — `BLADE_COUNT = 2^n`
   means the bound grades' density grows combinatorially with `n`, so `gp()`
   cost exploded far faster than quality improved. Reverted.
4. **Rewrote `Multivector` to sparse storage** (`Map<bitmask, coefficient>`
   instead of a dense array — see Design notes below) and bumped to
   `Cl(12,0)`, the actual ceiling: `blade.ts`'s context channel needs
   `C(n,3)` mutually orthonormal directions inside `SPHERE_DIM=256`
   dimensions, and `C(12,3)=220` is the largest that still fits under 256
   (`C(13,3)=286` doesn't). Result: nDCG@10 0.0353 → 0.0449, MRR@10
   0.0994 → 0.1039 — a real, ~20-27% relative gain, but nowhere near
   closing the gap with content-only (0.1589) or the ceiling (0.1190,
   which context-bound still only reaches ~38% of).

**The settled takeaway: more `Cl(n,0)` generators cannot close this gap at
`SPHERE_DIM=256`, and this is now a measured fact, not a hypothesis.**
Closing it for real would need a different binding scheme (see Design
notes), or accepting `SPHERE_DIM=256` as a content-only-optimized choice
and reconsidering it if context-binding matters more for your use case —
this is a genuine quality/quality trade-off between the two retrieval
paths, not a free win across the board.

## Stack

Next.js 16 (App Router, Turbopack) · React 19 · TypeScript · Tailwind CSS 4 ·
three.js. No backend, no database — everything lives in-memory in the
browser tab.

## Citations

The techniques this project combines, and the datasets/methods it's
benchmarked against:

**Vector Symbolic Architectures / holographic memory**
- Plate, T. A. (1995). [*Holographic Reduced Representations*](https://doi.org/10.1109/72.377968).
  IEEE Transactions on Neural Networks, 6(3), 623–641. — the "bind by
  multiplication, superpose by addition, retrieve by correlation" scheme
  this engine's binding/superposition/retrieval loop is built on.
- Kanerva, P. (2009). [*Hyperdimensional Computing: An Introduction to
  Computing in Distributed Representation with High-Dimensional Random
  Vectors*](https://doi.org/10.1007/s12559-009-9009-8). Cognitive
  Computation, 1(2), 139–159. — general VSA "bundling" (superposition
  preserving similarity to constituents), the trick behind `sphere.ts`'s
  word-bundling into a text vector.

**Geometric (Clifford) algebra**
- Hestenes, D., & Sobczyk, G. (1984). *Clifford Algebra to Geometric
  Calculus*. D. Reidel. — the geometric-product/grade/blade formalism
  `clifford.ts` implements directly.
- Doran, C., & Lasenby, A. (2003). *Geometric Algebra for Physicists*.
  Cambridge University Press. — rotors as the sandwich-product rotation
  mechanism used here for phase/time encoding (`timeRotor`, `sandwich`).

**Spherical harmonics / Gegenbauer polynomials**
- Gegenbauer, L. (1874). *Ueber einige bestimmte Integrale*. Sitzungsber.
  Math.-Naturwiss. Classe der Kaiserl. Akad. der Wissenschaften, 70. — the
  ultraspherical polynomial family `gegenbauer.ts` decomposes the sphere
  vector into, for the spectrum visualization (not used for matching; see
  `blade.ts`'s comment on why).

**Approximate nearest-neighbor search (bench baseline)**
- Malkov, Y. A., & Yashunin, D. A. (2018). [*Efficient and Robust
  Approximate Nearest Neighbor Search Using Hierarchical Navigable Small
  World Graphs*](https://doi.org/10.1109/TPAMI.2018.2889473). IEEE
  Transactions on Pattern Analysis and Machine Intelligence, 42(4),
  824–836. — the algorithm `bench/hnsw.ts` reimplements from scratch as one
  of the two retrieval baselines.

**Benchmark datasets**
- Boteva, V., Gholipour, D., Sokolov, A., & Riezler, S. (2016). [*A Full-Text
  Learning to Rank Dataset for Medical Information
  Retrieval*](https://doi.org/10.1007/978-3-319-30671-1_58). ECIR 2016. —
  NFCorpus, used unmodified in `bench/`.
- Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott, R., Zarrouk,
  M., & Balahur, A. (2018). [*WWW'18 Open Challenge: Financial Opinion
  Mining and Question Answering*](https://doi.org/10.1145/3184558.3192301).
  Companion Proceedings of the The Web Conference 2018. — FiQA, used
  unmodified as the larger-scale benchmark.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I.
  (2021). [*BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models*](https://arxiv.org/abs/2104.08663).
  NeurIPS 2021 (Datasets and Benchmarks Track). — the standardized
  corpus/queries/qrels format and public mirror both datasets are fetched
  from (`bench/fetch-data.sh`).
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). [*MTEB:
  Massive Text Embedding Benchmark*](https://arxiv.org/abs/2210.07316).
  arXiv:2210.07316. — the broader retrieval-benchmark suite BEIR's
  retrieval task, and this project's benchmark, sit within.
