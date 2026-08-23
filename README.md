# Holographic Frequencies Memory

A from-scratch holographic associative memory: text is hashed onto a unit
hypersphere, decomposed into a Gegenbauer harmonic spectrum, bound to context
via HRR circular convolution (Plate 1995), with a Clifford-algebra side
channel — geometric product, rotor phase encoding, grade-separated blades —
still driving the superposed field and its resonance stat. No embeddings, no
LLM — just a deterministic hash → sphere → spectrum → binding pipeline, with
a live 3D view and a tiered (recent / repeated / long-term) consolidation
policy.

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

## How it works

```
text ──▶ hash to ±1 bits ──▶ bundle words ──▶ unit vector on S^383 (content, context)
                                                     │
                              ┌──────────────────────┼───────────────────────────┐
                              ▼                      ▼                           ▼
                     Gegenbauer spectrum   circularConvolve(content, context)   grade-1/3 blades
                     (visualization only)      = boundHrr, full SPHERE_DIM      ──── gp() ────▶ bound (grade 2/4)
                                                          │                                       │
                                              what retrieve() actually scores          sandwich(timeRotor(θ), ·)
                                              context-bound candidates against                    │
                                              (see Design notes)                        superposed into field[bucket]
                                                                                        (globalResonance stat only)
```

- **Sphere encoding** (`src/lib/holo/sphere.ts`, `random.ts`) — each word is
  hashed to a deterministic ±1 bit string (a fixed-seed PRNG keyed by the
  word), words are bundled by summation, and the result is normalized onto
  `S^383` (`SPHERE_DIM = 384`, matching production/standard embedding
  widths). This is the standard Vector Symbolic Architecture "bundling"
  trick: a query for one word resonates with any text containing it.
- **Spectral projection** (`gegenbauer.ts`) — a Gegenbauer-polynomial harmonic
  decomposition of the sphere vector, shown in the UI's spectrum chart. It's
  *not* used for matching (see the comment in `blade.ts` for why a low-degree
  truncation loses discriminative signal on near-white-noise hash vectors).
- **HRR binding** (`hrr.ts`) — content and context are bound by circular
  convolution at full `SPHERE_DIM`, no dimensionality reduction: this is
  what `retrieve()` actually scores context-bound candidates against
  (`MemoryRecord.boundHrr`). Replaced an earlier Clifford-blade-compressed
  binding scheme after measuring a ~2x context-bound quality loss from that
  compression — see [Design notes](#design-notes--open-questions) for the
  full story and why it wasn't a simple tuning fix.
- **Clifford algebra** (`clifford.ts`) — `Cl(12,0)`: 4096 basis blades over
  12 generators (stored sparsely — see below — since only a few hundred are
  ever nonzero for any one multivector), addressed by bitmask, with a
  generic geometric product, addition/subtraction/scaling, reverse, an inner
  product, and rotor sandwich products. No longer what content-bound
  retrieval is scored against (see HRR binding above), but still real and
  still used: it drives the superposed field, the rotor-based time-phase
  encoding, and the `globalResonance` stat — a deliberate hybrid, not
  vestigial code (see Design notes).
- **Grade-separated channels** (`blade.ts`) — content and context are each
  projected onto their own *grade*: content → grade 1 (the 12 vector blades
  `e1..e12`), context → grade 3 (the 220 trivector blades). These are disjoint
  blade indices, so a content-only query and a context-only query can never
  collide by construction. Binding them (`gp(content, context)`) lands in
  grades 2 and 4 — again disjoint from both raw channels — so a superposed
  field can carry pure-content, pure-context, and bound signal at once
  without one drowning out another (this is the field/`globalResonance`
  path above, not the retrieval-scoring path — see HRR binding). Time keeps
  its own dedicated slot, the `e12` bivector (`timeRotor` in `clifford.ts`).
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
  clifford.ts      Cl(12,0) geometric algebra + grade utilities (sparse) —
                   drives the field/globalResonance stat, not retrieval scoring
  blade.ts         sphere vector -> grade-separated blade (content/context)
  hrr.ts           HRR circular convolution binding — what context-bound
                   retrieval is actually scored against
  projection.ts    sphere vector -> 3D point (visualization)
  engine.ts        HoloStore: binding, bucketed field, tiers, retrieval
  ann.ts           optional ANN index for content-only retrieval (opt-in, see Benchmark)
  trueHarmonic.ts  separate reversible byte/QPSK/temporal-harmonic experiment

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
- `SPHERE_DIM` was later raised again, 256 → 384 (matching a standard
  production embedding width), which reopened the `Cl(n,0)` capacity
  ceiling above — `C(14,3)=364` now fits under 384, where `C(13,3)=286`
  didn't fit under 256. `NUM_GENERATORS` was bumped 12 → 14 to use that
  headroom, and reverted after measuring the real cost at corpus scale: the
  quality gain was noise-level (context-bound nDCG@10 0.0595 → 0.0601,
  ~1%), but indexing SCIDOCS (~25.7K docs) at `Cl(14,0)` OOM'd Node's
  default ~4GB heap (needed 12GB+ to complete) — each stored record's
  bound-family blades (grades 2+4) went from `C(12,2)+C(12,4)=561` slots to
  `C(14,2)+C(14,4)=1092`, almost double, for a gain within measurement
  noise. `NUM_GENERATORS` is back to 12; see the
  [context-bound queries](#context-bound-queries) section for the numbers.
- Two real bugs surfaced while investigating that OOM, both fixed and both
  independent of the generator-count question above:
  - **`gp()` never pruned near-zero coefficients** (`clifford.ts`).
    Algebraic cancellations rarely land on exactly 0 in floating point —
    they leave ~1e-17-scale residue, which the sparse `Map` then retained
    as a "real" entry forever. `sandwich()` (used for every stored record's
    time-rotor binding) chains two `gp()` calls, and its *intermediate*
    result isn't grade-pure even though the final rotation is (a standard
    Clifford-algebra theorem), so that intermediate noise propagated
    through. Measured on one record at `Cl(14,0)`: the `rotated` blade held
    1588 entries, 496 of them (~31%) below `1e-9` in magnitude, the largest
    of those ~3.5e-18 — pure noise, but each one still cost a `Map` slot
    and got iterated by every later `add()`/`innerProduct()`/`magnitude()`
    call on that record. Fixed by pruning `gp()`'s output below an
    `EPSILON = 1e-9` threshold; quality is unaffected (verified identical
    nDCG@10 before/after on NFCorpus), since it only removes noise.
  - **`MemoryRecord.bound` was a dead field.** It was stored on every
    record but only ever read in one place, `findResonant()` — which
    recomputed `normalize(r.bound)` from scratch for every active record on
    every single `addMemory()` call, when `r.boundNorm` already caches
    exactly that value from construction. Fixed `findResonant()` to use
    `r.boundNorm` directly, which made `bound` unused, so it was dropped
    from `MemoryRecord` entirely — one whole extra Multivector's worth of
    memory (up to `C(n,2)+C(n,4)` entries) saved per stored record.
  - **Same pattern, found again after the HRR integration below:**
    `MemoryRecord.contentBlade`/`contextBlade` were stored on every record
    but never read anywhere after construction — they only ever needed to
    exist as local variables in `buildRecord()` to compute `bound` for the
    Clifford field. Dropped from `MemoryRecord`; `contextBlade` alone held
    up to 220 entries at `Cl(12,0)`, real memory saved per record across a
    bulk-loaded corpus for fields nothing read.
- **The generator-count dead end pointed at the real fix: a different
  binding scheme.** Every `Cl(n,0)` tuning pass above improved context-bound
  quality by 20-27% at best, capped by the blade's own capacity — the
  compression was the bottleneck, not the specific `n`. `src/lib/holo/hrr.ts`
  implements HRR (Plate 1995) circular convolution as an alternative:
  `content.sphereVec (*) context.sphereVec` at full `SPHERE_DIM`, no
  dimensionality reduction at all. Prototyped first as a `bench/context.ts`
  comparison row (not touching the live engine), which measured nDCG@10
  0.0595 → 0.1053 — confirmed real, then integrated into `HoloStore` as a
  **hybrid**: `MemoryRecord.boundHrr` (the HRR bind) is what `retrieve()`
  actually scores context-bound candidates against now; the Clifford side
  (`boundNorm`/`rotated`, `gp()`/`sandwich()`/`timeRotor()`) is kept, not
  deleted, for the field superposition and `globalResonance` stat — a
  deliberate choice to keep "Clifford geometric product" true for that
  subsystem rather than a from-scratch rewrite. Live
  result matched the prototype closely (0.1071 vs. 0.1053) — see
  [Context-bound queries](#context-bound-queries) for the full timeline and
  numbers. `blade.ts`'s grade-separation is still real, still documented,
  still exercised (by the field/stat path) — it's just no longer what
  determines context-bound retrieval quality.
- **Two more waste-on-every-record fixes surfaced auditing `buildRecord()`
  after the HRR integration**, both quality-neutral (verified identical
  nDCG@10 before/after):
  - `boundHrr` (the HRR bind, `O(SPHERE_DIM^2)` circular convolution) was
    computed unconditionally, even for records added with no context at
    all — including every record in `bench/index.ts`'s content-only
    benchmarks (`addBulk(text)`, no context argument), whose `retrieve()`
    calls never read `boundHrr` in the first place. Now skipped (a zero
    vector) when context is empty — and a zero vector is *more* correct
    there too, not just cheaper (see the field's own doc comment in
    `engine.ts` for why).
  - `buildRecord()` was calling the *full* `encodeContext()` (spectrum +
    point3d + blade) for the context channel, but only ever read
    `.blade`/`.sphereVec` from the result — `MemoryRecord` has no
    `contextSpectrum`/`contextPoint3d` field to store the rest into, so it
    was computed and discarded on every single record, forever. Query-side
    encoding already had this `full`/lean split (see the `encodeQuery`
    section above); `buildRecord()` was the one storage-side call site that
    still used the full encoder unnecessarily. Now uses `encodeContextQuery`.
- **Tried closing more of the remaining context-bound-vs-ceiling gap
  (HRR reaches ~81% of ceiling) with Hadamard (element-wise) binding
  instead of circular convolution — rejected, but for an instructive
  reason.** A `bench/context.ts` prototype scored it *exactly* at the
  ceiling (nDCG@10, Recall@100, and MRR@10 all identical to 4 decimal
  places), which looked like a clean win until it turned out to be a
  mathematical artifact of the benchmark's synthetic tags being single
  words: a single-word `textToSphereVector` output has every component at
  the same magnitude (only signs vary), and element-wise-multiplying both
  sides of a cosine comparison by such a vector is a signed permutation —
  it cancels out in cosine similarity *exactly*. So the "Hadamard-bound"
  score was silently plain content-only cosine, not a real content+context
  signal, and it hit the ceiling because that's mathematically how the
  ceiling itself is computed for this benchmark. Confirmed directly:
  `cosine(hadamard(q,c), hadamard(d,c)) === cosine(q,d)` bit-for-bit for a
  single-word `c`, genuinely different (and untested) for a real
  multi-word one. `hadamard()` is kept in `src/lib/holo/hrr.ts`, unused, as
  a documented dead end — see its doc comment for the full derivation.
  Not integrated into `HoloStore`.

## Experiment: reversible wire + temporal harmonics

`src/lib/holo/trueHarmonic.ts` explores a separate, full-fidelity memory
substrate. It does **not** replace or silently alter the hash → sphere →
Clifford engine above. Instead, literal bytes are framed with length and
CRC32, optionally protected with Hamming(7,4), Gray-QPSK modulated, and bound
directly to a timestamp rotor. In a single wire field:

```
F[k] = Σᵢ FFT(Wᵢ)[k] exp(-2π i k tᵢ/N)
```

This makes two retrieval directions mechanical rather than metaphorical:

- time → content: reverse the rotor/circular shift, demodulate the QPSK wire,
  and validate the exact bytes with CRC32;
- content → time: conjugate-multiply the field spectrum by an exact or partial
  probe spectrum, then IFFT once to expose matching offsets.

`DirectRotorWireField` writes non-overlapping frames locally in the time
domain and batches their mathematically equivalent rotor bindings into one
FFT per query-ready snapshot. `SeparableHarmonicWireField` retains the more
expensive independent time/content matrix as an exact-overlap control, not as
the default architecture.

Run the deterministic checks, controlled basis benchmark, and same-corpus
comparison with:

```bash
pnpm test:true-harmonic
pnpm bench:true-harmonic
pnpm bench:true-harmonic:nfcorpus -- --queries 60
```

On all 3,633 NFCorpus documents (5.79MB of literal UTF-8), 90 × 262,144-symbol
fields took 1.14s to encode/place and 0.87s to prepare spectra: 2.01s
query-ready versus 7.32s for `HoloStore` indexing in the same process. Thirty-
two sampled time-addressed reads were 100% CRC-valid. A 64-byte prefix probe
ranked its source first. The cost is global search: exhaustive byte-wave
correlation averaged 1.00s/query versus 2.14ms for `HoloStore` on 60 evenly
spaced judged queries.

| Method (60-query sample) | nDCG@10 | Recall@100 | MRR@10 | Avg query |
|---|---:|---:|---:|---:|
| Direct rotor, exhaustive | 0.1788 | **0.1823** | 0.2934 | 1002.5ms |
| `HoloStore` | **0.2508** | 0.1797 | **0.4432** | **2.14ms** |

Two acceleration controls were rejected rather than promoted. One exact
33,554,432-symbol global harmonic superposition took 3.82s to become query-
ready and about 5.4s/query—roughly 5× slower than the smaller FFTs. A lossy
128-band magnitude/rotor router cut search to about 100ms, but missed the
known prefix shard and fell to 0.0599 nDCG@10. A useful router must retain
phase/locality or use another query-conditioned coarse representation.

The next bake-off keeps complex phase and tests 1,024-coefficient localized
projections per shard: uniform sparse Fourier samples, a leading DPSS/Slepian
taper, periodic needlet-style bands, cycle-graph diffusion-wavelet bands, and
Gabor windows. Each reduced inverse FFT routes to candidate shards; the exact
full-resolution FFT still scores records inside those shards. The router adds
only 5.6–8.4MiB for the promoted variants. Use `--router-power`,
`--route-shards`, and `--router-kinds` to reproduce other points, for example:

```bash
pnpm bench:true-harmonic:nfcorpus -- --queries 60 --router-power 10 \
  --route-shards 32 --router-kinds needlet,diffusion
```

On the established 60-query sample, diffusion was the strongest natural-query
router while the needlet-style bank was the only one to retain the known
64-byte prefix reliably. Reserving eight of 32 candidates for needlets and
filling the rest from diffusion produced the best current combined result:

| Method | Prefix | nDCG@10 | Recall@100 | MRR@10 | Avg query |
|---|:---:|---:|---:|---:|---:|
| Direct rotor, exhaustive | #1 | **0.1788** | **0.1823** | 0.2934 | 983.4ms |
| Needlet-style, 32/90 shards | #1 | 0.1137 | 0.0815 | 0.2146 | 384.2ms |
| Diffusion, 32/90 shards | miss | 0.1535 | 0.1145 | **0.3251** | **375.8ms** |
| Needlet + diffusion, 8 + 24 | #1 | 0.1492 | 0.1180 | 0.3054 | 397.2ms |
| `HoloStore` | n/a | 0.2508 | 0.1797 | 0.4432 | 2.68ms |

The hybrid is about 2.5× faster than exhaustive wire correlation, retains 83%
of its nDCG and 65% of its Recall@100, slightly improves MRR, and preserves the
sharp prefix case. That makes it the best surviving wire-router configuration,
not a replacement for `HoloStore`: semantic retrieval remains two orders of
magnitude faster. The divergence between needlet and diffusion results is also
evidence that interference is structured—different localized frames expose or
suppress different signals—rather than uniformly meaningless noise.

The separable control retains the earlier capacity result: at 128 random
32-byte records a complete basis recovered 100% of timestamps and CRC-valid
payloads; a same-coefficient partial 4096-tick basis kept timestamp search at
100% while CRC recovery fell to 0%. Packet-level FEC does not create missing
independent time modes.

This experiment adds exact preservation and field-native temporal lookup that
the semantic engine does not attempt. Conversely, it supplies no learned or
hash-bundle semantics: natural-language similarity remains the existing
engine's job. A hybrid should therefore compare or compose the two result
lanes, not claim that either one is a drop-in improvement to the other.

## Benchmark: vs. normal RAG

`bench/` compares this engine against brute-force cosine and an
approximate HNSW index on real retrieval tasks (NFCorpus and, for a bigger
scale test, SCIDOCS/FiQA-sized corpora — from the MTEB retrieval suite) —
speed and quality, same embedding across all three. See
[`bench/README.md`](bench/README.md).

```bash
pnpm bench:fetch [dataset]   # default: nfcorpus. Also: scidocs, fiqa, ...
pnpm bench [-- --dataset name] [--ann]   # content-only queries, vs. brute-force cosine + HNSW
                                          # (--ann: opt the holographic engine into its own
                                          #  batch-built ANN index instead of an exact scan —
                                          #  see "Optional: ANN index" below)
pnpm bench:context                # context-bound queries (NFCorpus only, see below)
pnpm bench:temporal               # native single-plane vs distributed temporal rotor
```

### Distributed temporal rotor experiment

The opt-in `distributed` addressing mode maps time onto the six independent
commuting bivector planes available in `Cl(12,0)`. Its incommensurate bands
span roughly 1.4 logical hours to 6.5 years. Ranking uses the rotor as a 5%
additive tie-break, not as an ordering oracle; exact logical ticks remain
authoritative. `rotateWithDistributedTimeRotor()` also supports literal
geometric bind/unbind of field contributions so its cost can be measured
separately from the ranking signal.

Representative local result (`pnpm bench:temporal`, 576 memories and 576
queries; three-build median index time):

| Method | Exact version @1 | Semantic family @1 | MRR | 24-tick alias | Index | Avg query |
|---|---:|---:|---:|---:|---:|---:|
| Content only | 1.4% | 100.0% | 0.0675 | 2.8% | 730.50ms | 370µs |
| Native single-plane | 33.3% | 100.0% | 0.6111 | 66.7% | 730.50ms | 329µs |
| Distributed score only | 100.0% | 100.0% | 1.0000 | 0.0% | 730.50ms | 652µs |
| Distributed geometric field | 100.0% | 100.0% | 1.0000 | 0.0% | 1429.20ms | 611µs |

The distributed code had no correlation at or above 0.99 for any nonzero
delta through 100,000 ticks; the worst was 0.984292 at 73,756 ticks. This is
still a finite phase code, so recurrence is delayed rather than eliminated.
The controlled result supports the distributed correlation as a useful
secondary address: it breaks identical-content temporal ties without changing
the semantic family. It does **not** support rotating every field contribution
by default: that added no peak-ranking quality here and roughly doubled index
cost. A 25-query NFCorpus smoke rerun retained the native content metrics
exactly (nDCG@10 0.2168, Recall@100 0.1006, MRR@10 0.5240), as expected for
an opt-in temporal path. This synthetic ablation is intentionally narrower
than a claim about arbitrary temporal reasoning.

### Content-only queries

**Results** (NFCorpus test split, 3633 docs, 323 judged queries, shared
IDF-weighted sphere-vector embedding, top-100, `SPHERE_DIM=384`):

| Method | nDCG@10 | Recall@100 | MRR@10 | Index (ms) | Avg query | p95 query |
|---|---|---|---|---|---|---|
| Brute-force cosine | 0.2361 | 0.1858 | 0.4188 | 3µs | 2.72ms | 3.55ms |
| HNSW (approx.) | 0.2281 | 0.1663 | 0.4061 | 5438.28ms | 2.59ms | 3.68ms |
| Holographic engine | 0.2348 | 0.1856 | 0.4172 | 7717.06ms | 4.50ms | 6.11ms |

Brute-force cosine's numbers are the ceiling for this embedding (exact
top-k). The holographic engine tracks it closely on quality. Getting here
took five changes, in order (each number below is nDCG@10, brute-force,
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
   index/query cost and a widening HNSW-vs-brute-force gap.
5. **`SPHERE_DIM` 256 → 384** (`src/lib/holo/sphere.ts`) — matches a
   standard production embedding width rather than the point the 128/256/512
   sweep above happened to land on: 0.2161 → 0.2361, a further ~9%
   improvement, again for all three methods. This also reopened the
   `Cl(n,0)` capacity ceiling discussed in
   [Design notes](#design-notes--open-questions) — tried `NUM_GENERATORS=14`
   to use the extra headroom, measured a noise-level quality gain against a
   real memory cost at corpus scale, and reverted to 12. Two real bugs
   (`gp()` not pruning near-zero coefficients, and `MemoryRecord` carrying a
   fully dead `bound` field) were found and fixed in the course of that
   investigation — see Design notes for both; neither changes quality,
   both reduce memory and redundant compute.

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
| Brute-force cosine (this project's encoder) | 0.2361 | 0.1858 | 0.4188 |
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
  384-dim sphere vectors. A higher-dimensional GloVe variant (100d/200d/300d)
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
| Brute-force cosine | 0.1189 | 0.2646 | 0.2211 | 5µs | 36.55ms | 52.94ms |
| HNSW (approx.) | 0.1143 | 0.2641 | 0.2122 | 108810.28ms | 4.76ms | 7.86ms |
| Holographic engine | 0.1189 | 0.2646 | 0.2211 | 87137.52ms | 52.82ms | 72.65ms |
| GloVe average (real embedding) | 0.0443 | 0.1367 | 0.0903 | 9307.99ms | 17.38ms | 24.64ms |

*(Run at the current config — `SPHERE_DIM=384`, `Cl(12,0)`, HRR context
binding (see Design notes). An earlier pass at `SPHERE_DIM=24` scored
nDCG@10 0.0063 across the board here; the same kind of jump from raising
`SPHERE_DIM` that showed up on NFCorpus holds at this scale too.)*

**A caveat this table itself demonstrates**: absolute wall-clock ms here are
noisy enough on a shared/loaded machine that they shouldn't be read too
precisely run-to-run — `bench/hnsw.ts`'s index-build, completely unrelated
to any change in this revision, measured 65.7s in one full-suite run and
108.8s in another (this one), same unchanged algorithm. That's a larger
swing than several of the "real" fixes described elsewhere in this README
moved the needle by. Treat single-run deltas under roughly 2x as noise, not
signal; the nDCG/Recall/MRR quality columns are exact and reproducible,
unlike the timing columns.

Quality: holographic engine ties brute-force **exactly** again (0.1189 =
0.1189, identical to 4 decimal places) — the retrieval-mechanism comparison
holds at 7x scale. HNSW trails slightly (0.1143), the real cost of its
approximation once there's enough embedding signal to actually approximate
imperfectly. GloVe loses again, by an even wider margin than on NFCorpus,
with a worse OOV rate here too (37.7% of SCIDOCS's vocabulary missing from
GloVe, vs. NFCorpus's 21% — consistent with SCIDOCS's citation-prediction
task skewing toward more technical scientific jargon).

Speed: still the slowest of the three non-GloVe methods per query, same
shape as before — a real, structural gap, not fixed by any of several real
constant-factor fixes along the way:
- `retrieve()`'s query-side `encode()` used to unconditionally run the full
  store-time pipeline (`spectralProject`, `projectTo3D`) even though a
  content-only query only ever reads `.sphereVec`/`.blade` from the result —
  `encodeChannel()` now takes a `full` flag, and `retrieve()` calls the lean
  `encodeQuery`/`encodeContextQuery` variants that skip both
  (`src/lib/holo/engine.ts`).
- `buildRecord()` was unconditionally computing `boundHrr` (the HRR bind,
  `O(SPHERE_DIM^2)` circular convolution) for *every* indexed record,
  including ones added with no context at all — like every record in this
  very benchmark (`addBulk(text)`, no context argument). `retrieve()`'s
  content-only path never reads `boundHrr`, so this was pure waste; now
  skipped (a zero vector) whenever a record's context is empty.

Quality is unaffected by either — neither field was ever part of
content-only matching. What's left is `retrieve()`'s genuine O(N) linear
candidate scan (`peaks: candidates.map(...)`, same file) — same asymptotic
order as brute-force cosine, just a larger constant factor per candidate
(the flat-array dot product itself is comparable, but per-record overhead
beyond it adds up). An opt-in ANN index now exists for exactly this gap
(`--ann`, see below) — not automatic, since it trades away the exact-tie
result this section's whole comparison rests on.

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
performance bug. (HNSW's SCIDOCS index-build above, ~66s, is slow again in
absolute terms — but because higher-`SPHERE_DIM` vectors cost more per
distance computation than `SPHERE_DIM=24` ones did, not because of a bug;
`bench/index.ts` fixed `HNSW`'s parameters, so this scales with embedding
cost, as expected.)

The `phaseWeight`/`allActive()` fixes described in earlier revisions of
this README were real and are still in place; so is the query-side
`encode()` fix described above. What's left is the linear scan itself, not
fixed overhead stacked on top of it.

### Optional: ANN index (`--ann`)

The linear scan above ties brute-force cosine's quality exactly, but is
still `O(N)` per query with real overhead beyond the dot product (see
above) — an HNSW-style approximate index (`src/lib/holo/ann.ts`) is
available as an opt-in alternative for the content-only path
(`HoloStore.buildContentIndex()`, `pnpm bench --ann`), same trade the
`HNSW (approx.)` baseline row already makes. **Off by default** — `retrieve()`
never reaches for it on its own, since giving up exactness is a real cost,
not a free win.

Getting a working version took two attempts. The first inserted into the
index incrementally, inside the same loop that does the per-record
Clifford-algebra work (`addBulk()`) — measured on SCIDOCS (25.7K docs), that
made *indexing* 7.6x slower (50.8s → 384.6s) for only a ~10% query-time
improvement and a real quality loss, worse on every axis than the plain
scan it was meant to speed up. Root cause: sharing a heap with a second,
unrelated, allocation-heavy workload cost 2.4x versus running the identical
HNSW insert loop standalone (141s → 334s of pure insert time) — not
anything intrinsic to HNSW. The fix was building the graph as one clean
batch pass *after* bulk-loading finishes, instead of interleaved per
record, plus a leaner default graph (`M`, `efConstruction` lower than the
HNSW baseline's own). Three configs measured on SCIDOCS after that fix:

| Config | nDCG@10 | % of ceiling | Index time | Avg query |
|---|---|---|---|---|
| Exact scan (default) | 0.1189 | 100% | 50.8s | 52.21ms |
| `M=8, efConstruction=40` | 0.1057 | 88.9% | 61.8s | 11.60ms |
| **`M=12, efConstruction=60` (ANN default)** | **0.1163** | **97.8%** | 163.8s | 21.09ms |
| HNSW baseline (`M=16, efConstruction=100`) | 0.1143 | 96.1% | 62.7s | 5.28ms |

`M=12/efConstruction=60` edges out HNSW's own quality (97.8% vs 96.1% of
ceiling) while still being faster to build and query than the heavier
`M=16/efConstruction=100` config — but it doesn't dominate the HNSW
baseline outright: HNSW still builds and queries faster. This is a real
speed/quality dial across all three configs, not a solved trade-off with
one clear winner — reasonable if you want this store's own index, faster
than its exact scan, without pulling in a separate ANN dependency, but not
a claim that it beats a dedicated HNSW index at its own game.

### Context-bound queries

NFCorpus queries carry no context, so the benchmark above never exercises
the context-binding path — `bench/context.ts` fills that gap by synthesizing
topic tags via keyword match (cancer, diabetes, cholesterol, ...) shared
between docs and queries, then compares variants on the same corpus:

| Method | nDCG@10 | Recall@100 | MRR@10 | Avg query |
|---|---|---|---|---|
| Bucket ceiling (brute-force, same-tag docs only) | 0.1321 | 0.1185 | 0.3170 | 418µs |
| Context-bound (`HoloStore.retrieve`, HRR scoring) | **0.1071** | 0.1079 | 0.2317 | 2.53ms |
| Content-only (same store, no context) | 0.1646 | 0.1970 | 0.3284 | 3.82ms |

*(60 judged NFCorpus test queries whose text matched a topic keyword.
`SPHERE_DIM=384`.)*

This result got noticeably **worse** after `SPHERE_DIM` went 24 → 256, was
partially recovered by tuning `Cl(n,0)`'s generator count, and — after
tuning was proven to be a dead end — was fixed for real by changing the
binding scheme, exactly as this README used to say would be required.
Timeline:

1. **At `SPHERE_DIM=24` / `Cl(8,0)`**: context-bound nDCG@10 0.0523 vs.
   content-only's 0.0371 — binding *beat* no-context, reaching 89% of the
   ideal bucket ceiling.
2. **After `SPHERE_DIM` → 256, still `Cl(8,0)`**: context-bound collapsed to
   0.0353 while content-only jumped to 0.1589 — a >4x reversal. Cause:
   `Cl(8,0)`'s content blade holds exactly 8 real numbers no matter how rich
   the source embedding is. At `SPHERE_DIM=24` that was a 24→8 compression
   (~33% kept); at `SPHERE_DIM=256` it's a 256→8 compression (~3% kept) —
   the blade didn't get worse, content-only just left it behind.
3. **`Cl(10,0)` on the old dense `Multivector`**: quality barely moved
   (0.0383, within noise) while query latency got ~7x worse. Reverted.
4. **Sparse `Multivector` + `Cl(12,0)`**, the (then) capacity ceiling under
   `SPHERE_DIM=256`: nDCG@10 0.0353 → 0.0449 — real but nowhere near closing
   the gap with content-only (0.1589) or the ceiling (0.1190, ~38% reached).
5. **`SPHERE_DIM` 256 → 384** reopened that ceiling (`Cl(14,0)` now fit),
   tried, and reverted — the isolated generator-count contribution was
   ~1% (noise-level), and it OOM'd a 25.7K-doc corpus at the default Node
   heap (see Design notes). Settled at `NUM_GENERATORS=12`, nDCG@10 0.0595.
6. **Switched the binding operator itself**: HRR (Plate 1995) circular
   convolution at full `SPHERE_DIM=384`, no grade compression, replacing
   the `Cl(12,0)` grade-3 blade for *scoring* (Clifford is still used for
   the field/`globalResonance` stat — see Design notes for why this is a
   deliberate hybrid, not a full rewrite). Result: **nDCG@10 0.0595 →
   0.1071, +80% relative** — 81% of the bucket ceiling, up from 45%, and
   *faster* per query too (flat-array dot products instead of sparse
   `Map`-based Clifford ops). A prototype of this
   (`src/lib/holo/hrr.ts`, also shown by `bench/context.ts`'s `HRR-bound`
   row, same-tag restricted rather than hash-bucketed) measured 0.1053 —
   the live integration reproduces it closely.

**The earlier "settled" takeaway needed a correction, not a restatement: it
was right that more `Cl(n,0)` generators couldn't close the gap, but wrong
to leave it there — a different binding scheme genuinely could, and does.**
Context-bound retrieval still trails content-only (0.1071 vs. 0.1646) on
this corpus, so scoping to context still isn't a free win here — but the
remaining gap is now a real, much smaller one (compare 81% vs. 45% of
ceiling), not an architectural ceiling from lossy compression.

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
  circular convolution, superpose by addition, retrieve by correlation"
  scheme this engine's binding/superposition/retrieval loop is built on.
  Directly implemented, not just an inspiration: `src/lib/holo/hrr.ts`'s
  `circularConvolve` is this paper's binding operator, and it's what
  context-bound retrieval is actually scored against — see the root
  README's Design notes for how that replaced an earlier Clifford-blade
  binding scheme.
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
