// The holographic memory engine: ties bit->sphere mapping, Gegenbauer
// spectral projection, HRR binding (hrr.ts), and Clifford-algebra
// superposition/retrieval into one stateful store, plus a 3-tier
// (recent / repeated / long-term) consolidation policy.
//
// Time is a logical tick counter (not wall-clock) so the demo is
// reproducible and doesn't depend on how fast someone clicks around.

import {
  Multivector,
  add,
  gp,
  innerProduct,
  magnitude,
  mv,
  normalize,
  reverse,
  sandwich,
  scale,
  sub,
  timeRotor,
} from "./clifford";
import { sphereToContentBlade, sphereToContextBlade } from "./blade";
import { spectralProject } from "./gegenbauer";
import { projectTo3D } from "./projection";
import { hashString } from "./random";
import { IdfLookup, sphereDot, textToSphereVector } from "./sphere";
import { ANNIndex } from "./ann";
import { circularConvolve, dot as hrrDot, normalize as hrrNormalize } from "./hrr";
import {
  distributedRotorCorrelation,
  rotateWithDistributedTimeRotor,
} from "./temporalRotor";

export type Tier = "recent" | "repeated" | "long-term";
export type TemporalAddressing = "single" | "distributed";

export interface MemoryRecord {
  id: string;
  text: string;
  context: string;
  createdAtTick: number;
  lastSeenTick: number;
  repeatCount: number;
  tier: Tier;
  weight: number;
  rotorAngle: number;
  bucket: number; // which field bucket (hashed from context) this record's contribution lives in
  spectrum: number[];
  // content.blade/ctx.blade (the Clifford grade-1/grade-3 projections)
  // aren't stored on the record: they're only ever needed as local
  // variables in buildRecord() to compute `bound` for the Clifford field —
  // nothing reads a stored contentBlade/contextBlade after construction
  // (same dead-field pattern the `bound` field itself was fixed for — see
  // git history / README Design notes). At Cl(12,0), contextBlade alone
  // holds up to 220 entries; across a 25.7K-doc corpus that's real memory
  // saved for a field nothing ever read.
  //
  // `bound` (content (x) context via gp(), before the time rotor) is the
  // same story — not stored either, same reasoning.
  boundNorm: Multivector; // normalize(bound), cached — see retrieve()
  rotated: Multivector; // bound after the time rotor is applied
  // HRR (Plate 1995) binding: normalize(circularConvolve(content.sphereVec,
  // context.sphereVec)) — full SPHERE_DIM, no grade compression. This is
  // what retrieve() actually scores context-bound candidates against now
  // (see the README's "Context-bound queries" section for the measured
  // ~2x nDCG@10 win over boundNorm's grade-3-blade-compressed Clifford
  // product). boundNorm/rotated/the Clifford field above are kept — not
  // dead code — for the coarse globalResonance stat and time-phase
  // superposition; see contribution()/retrieve()'s wantGlobalResonance
  // branch. A hybrid, deliberately: full replacement would have been
  // simpler code, but this keeps "Clifford geometric product" true for the
  // time-phase/field subsystem while fixing the actual quality bottleneck
  // (per-record context-bound scoring), which is what was asked for.
  boundHrr: Float64Array;
  point3d: [number, number, number];
  // Full SPHERE_DIM sphere vector for the content channel, kept alongside
  // the compressed grade-1 blade. The blade is what the field/binding path
  // uses (that's the whole "orthogonal channels" point of blade.ts), but a
  // grade-1 blade only keeps NUM_GENERATORS of SPHERE_DIM dimensions — for
  // a context-free query there's no binding to preserve orthogonality
  // against, so retrieve() reranks by full-vector cosine here instead,
  // which recovers most of the quality that the blade compression costs.
  contentVector: Float64Array;
}

export interface Peak {
  record: MemoryRecord;
  similarity: number;
  phaseWeight: number;
  score: number;
}

export interface RetrievalResult {
  globalResonance: number;
  correlation: Multivector;
  peaks: Peak[];
}

const RECENT_CAPACITY = 7;
const RESONANCE_THRESHOLD = 0.88; // content+context similarity treated as "the same memory recurring"
const RECENT_DECAY = 0.82;
const REPEATED_DECAY = 0.94;
const PROMOTE_TO_REPEATED_AT = 1; // repeatCount >= 1 (seen twice total)
const PROMOTE_TO_LONG_TERM_AT = 3; // repeatCount >= 3 (seen four times total)
const ROTOR_PERIOD = 24;
const DEFAULT_SIGMA = 1.1; // radians, width of the Gaussian phase window
const FORGET_EPSILON = 0.05;

// The field is split into buckets keyed by context, instead of one shared
// accumulator. Superposing every memory ever added into a single field means
// retrieval SNR degrades roughly as 1/sqrt(N): past a few hundred items
// "everything resonates with everything" a bit more than it should.
// Bucketing by context partitions that crosstalk — a context-scoped query
// only has to compete against the memories that share its bucket, not the
// whole store. It's an orthogonal-channel trick at the storage level, the
// same way grade separation (see blade.ts) is one at the algebra level.
const NUM_BUCKETS = 64;

function bucketOf(context: string): number {
  return hashString(context || "∅") % NUM_BUCKETS;
}

function phaseOf(tick: number): number {
  return ((tick % ROTOR_PERIOD) / ROTOR_PERIOD) * 2 * Math.PI;
}

function angularDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function gaussian(d: number, sigma: number): number {
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

interface Encoded {
  spectrum: number[];
  blade: Multivector;
  point3d: [number, number, number];
  sphereVec: Float64Array;
}

/** Just what retrieve() needs to score candidates — see encodeChannel's `full` param. */
interface QueryEncoded {
  blade: Multivector;
  sphereVec: Float64Array;
}

// `full` gates spectrum/point3d: both are UI-only (spectrum chart, 3D scene)
// and neither is read anywhere on the query path (retrieve() only uses
// .blade/.sphereVec below) — computing them per query was pure waste,
// stacked on top of retrieve()'s already-O(N) candidate scan on every
// search. buildRecord() (storage path) still wants both, so it passes
// full=true; retrieve() passes full=false. See the README's SCIDOCS section
// for where this was originally flagged as an unaddressed cost.
function encodeChannel(
  text: string,
  channel: "content" | "context",
  idf: IdfLookup | undefined,
  full: true
): Encoded;
function encodeChannel(
  text: string,
  channel: "content" | "context",
  idf: IdfLookup | undefined,
  full: false
): QueryEncoded;
function encodeChannel(
  text: string,
  channel: "content" | "context",
  idf: IdfLookup | undefined,
  full: boolean
): Encoded | QueryEncoded {
  const sphereVec = textToSphereVector(text, undefined, idf);
  // blade: what's actually used for binding/matching. Content and context
  // are projected onto disjoint grades (see blade.ts) so they're orthogonal
  // channels rather than two signals sharing the same SPHERE_DIM numbers.
  const blade =
    channel === "content" ? sphereToContentBlade(sphereVec) : sphereToContextBlade(sphereVec);
  if (!full) return { blade, sphereVec };
  // spectrum: the Gegenbauer harmonic decomposition, kept for the spectrum
  // chart / conceptual fidelity to "decompose into harmonic coefficients".
  const spectrum = Array.from(spectralProject(sphereVec));
  const point3d = projectTo3D(sphereVec);
  return { spectrum, blade, point3d, sphereVec };
}

/**
 * Encode text on the content channel (grade-1 blade), including the
 * spectrum/point3d fields the interactive UI displays. `idf` is optional —
 * the interactive demo has no fixed corpus to compute document frequencies
 * over, so it's left unweighted (every word counts equally) there. Callers
 * that do have a corpus up front (see HoloStore's constructor, and
 * bench/index.ts) should supply one via buildIdf for higher-quality
 * bundling — see sphere.ts for why.
 */
export function encode(text: string, idf?: IdfLookup): Encoded {
  return encodeChannel(text, "content", idf, true);
}

/** Encode text on the context channel (grade-3 blade) — see blade.ts. */
export function encodeContext(text: string, idf?: IdfLookup): Encoded {
  return encodeChannel(text, "context", idf, true);
}

/** Lean content-channel encode for retrieve()'s query side — see encodeChannel's `full` param. */
function encodeQuery(text: string, idf?: IdfLookup): QueryEncoded {
  return encodeChannel(text, "content", idf, false);
}

/** Lean context-channel encode for retrieve()'s query side — see encodeChannel's `full` param. */
function encodeContextQuery(text: string, idf?: IdfLookup): QueryEncoded {
  return encodeChannel(text, "context", idf, false);
}

export class HoloStore {
  clock = 0;
  // Random per-instance tag, not a module-level counter: immune to Fast
  // Refresh resetting module state out from under a live store instance.
  private idTag = Math.random().toString(36).slice(2, 8);
  private nextId = 0;
  // Optional IDF weighting (see sphere.ts's buildIdf) shared by every
  // encode/encodeContext call this store makes, so a corpus-aware caller
  // (e.g. bench/index.ts) gets exactly the same term weighting the
  // brute-force/HNSW baselines use — otherwise "same embedding across all
  // three" would stop being true the moment one of them adds IDF.
  private idf?: IdfLookup;
  private temporalAddressing: TemporalAddressing;

  constructor(opts: { idf?: IdfLookup; temporalAddressing?: TemporalAddressing } = {}) {
    this.idf = opts.idf;
    this.temporalAddressing = opts.temporalAddressing ?? "single";
  }
  // One multivector per context bucket instead of a single shared field —
  // see NUM_BUCKETS above.
  field: Multivector[] = Array.from({ length: NUM_BUCKETS }, () => mv());
  recent: MemoryRecord[] = [];
  repeated: MemoryRecord[] = [];
  longTerm: MemoryRecord[] = [];
  // O(1) id -> record lookup, maintained alongside the tier arrays — used
  // by removeMemory() (used to be an allActive().find(), O(active)) and by
  // the ANN path below to resolve search hits back to a MemoryRecord.
  private recordsById: Map<string, MemoryRecord> = new Map();
  // Content-vector ANN index — opt-in, not automatic (contrast with the
  // first attempt at this; see ann.ts's header for why). null until
  // buildContentIndex() is called; retrieve() falls back to its exact scan
  // whenever it's null, so calling this is purely a speed lever, never
  // required for correctness.
  private ann: ANNIndex<string> | null = null;

  /**
   * Batch-build the content-vector ANN index from every currently-active
   * record, for retrieve()'s content-only path. Call this once after
   * bulk-loading a corpus (e.g. after a run of addBulk() calls) — not
   * automatic, and not kept in sync with later addMemory()/removeMemory()
   * calls; call it again if the store's contents change enough to matter.
   * The interactive demo never calls this (its stores stay tiny enough that
   * the exact scan is already fast — see the README's Design notes for the
   * measured cost of building this index at all).
   */
  buildContentIndex(opts?: { M?: number; efConstruction?: number; seed?: number }) {
    const active = this.allActive();
    this.ann = ANNIndex.build(
      active.map((r): [string, Float64Array] => [r.id, r.contentVector]),
      opts
    );
  }
  // allActive() used to rebuild this (array-spread + Set dedup) from
  // scratch on every call — free at the interactive demo's scale (a few
  // dozen records at most), but it's also what retrieve() calls on every
  // single query, so at bulk-corpus scale (thousands of records, thousands
  // of queries) that rebuild cost compounds and shows up directly in query
  // latency — see the SCIDOCS benchmark in the README, where it made this
  // the slowest of the three compared methods per-query. Cached here and
  // invalidated by every mutation to recent/repeated/longTerm (see
  // invalidateActive()); retrieve() never mutates, so a bulk-loaded,
  // read-only store now builds this exactly once, not once per query.
  private _activeCache: MemoryRecord[] | null = null;

  private invalidateActive() {
    this._activeCache = null;
  }

  private allActive(): MemoryRecord[] {
    if (this._activeCache) return this._activeCache;
    // De-duped defensively by id: each tier array should be disjoint, but
    // this guarantees it even if a bug ever lets an id appear twice.
    const seen = new Set<string>();
    const out: MemoryRecord[] = [];
    for (const r of [...this.recent, ...this.repeated, ...this.longTerm]) {
      if (seen.has(r.id)) continue;
      seen.add(r.id);
      out.push(r);
    }
    this._activeCache = out;
    return out;
  }

  private contribution(r: MemoryRecord): Multivector {
    return scale(r.rotated, r.weight);
  }

  private rotateAt(bound: Multivector, tick: number): Multivector {
    return this.temporalAddressing === "distributed"
      ? rotateWithDistributedTimeRotor(tick, bound)
      : sandwich(timeRotor(phaseOf(tick)), bound);
  }

  private setWeight(r: MemoryRecord, newWeight: number) {
    const old = this.contribution(r);
    r.weight = newWeight;
    const next = this.contribution(r);
    this.field[r.bucket] = add(sub(this.field[r.bucket], old), next);
  }

  private removeFromTierArrays(id: string) {
    this.recent = this.recent.filter((r) => r.id !== id);
    this.repeated = this.repeated.filter((r) => r.id !== id);
    this.longTerm = this.longTerm.filter((r) => r.id !== id);
    this.recordsById.delete(id);
    this.invalidateActive();
  }

  private forget(r: MemoryRecord) {
    this.field[r.bucket] = sub(this.field[r.bucket], this.contribution(r));
    this.removeFromTierArrays(r.id);
  }

  /** Explicitly delete a memory: pull its contribution out of the field and drop it. */
  removeMemory(id: string) {
    const r = this.recordsById.get(id);
    if (!r) return;
    this.forget(r);
  }

  /** Shared record construction for addMemory/addBulk: bind, fold in the current temporal phase. */
  private buildRecord(
    text: string,
    context: string,
    tier: Tier,
    weight: number,
    repeatCount: number
  ): { record: MemoryRecord; bound: Multivector } {
    const content = encode(text, this.idf);
    // Lean (encodeContextQuery, not encodeContext): only ctx.blade/.sphereVec
    // are ever used below — MemoryRecord has no contextSpectrum/contextPoint3d
    // field to store the full version's spectrum/point3d into, so computing
    // them here was pure waste on every single record, forever (the same
    // full-vs-lean split retrieve()'s query side already gets — this was the
    // one storage-side spot still calling the full encoder unnecessarily).
    const ctx = encodeContextQuery(context || "∅", this.idf); // empty-context placeholder
    const bound = gp(content.blade, ctx.blade);
    // HRR bind: circular convolution at full SPHERE_DIM, no grade
    // compression — see MemoryRecord.boundHrr's doc comment for why this,
    // not `bound` above, is what retrieve() actually scores against.
    //
    // Skipped (a zero vector instead) for records with no real context:
    // retrieve()'s content-only path (hasContext=false) never reads
    // boundHrr at all, and a context-bound query only reaches an
    // empty-context record's boundHrr in the rare case its hash bucket
    // collides with bucketOf("∅") — where circularConvolve(content, "∅")
    // would be semantically meaningless noise anyway, not a real signal; a
    // zero vector (cosine similarity 0) is *more* correct there, not just
    // cheaper. `circularConvolve` is O(SPHERE_DIM^2); for a corpus indexed
    // with addBulk(text) and no context (e.g. bench/index.ts's entire
    // content-only benchmark — every "Holographic engine" index-time
    // number in the README came from a run like this), computing it
    // unconditionally was pure waste on every single record.
    const boundHrr = context.trim()
      ? hrrNormalize(circularConvolve(content.sphereVec, ctx.sphereVec))
      : new Float64Array(content.sphereVec.length);
    const bucket = bucketOf(context);
    const angle = phaseOf(this.clock);
    const rotated = this.rotateAt(bound, this.clock);
    const record: MemoryRecord = {
      id: `m${this.idTag}-${this.nextId++}`,
      text,
      context,
      createdAtTick: this.clock,
      lastSeenTick: this.clock,
      repeatCount,
      tier,
      weight,
      rotorAngle: angle,
      bucket,
      spectrum: content.spectrum,
      boundNorm: normalize(bound),
      rotated,
      boundHrr,
      point3d: content.point3d,
      contentVector: content.sphereVec,
    };
    return { record, bound };
  }

  /** Bind text+context, fold in the current temporal phase, and add to the field. */
  addMemory(text: string, context = ""): MemoryRecord {
    const { record, bound } = this.buildRecord(text, context, "recent", 1, 0);
    const existing = this.findResonant(bound);
    if (existing) return this.reinforce(existing, bound, record.boundHrr);

    this.field[record.bucket] = add(this.field[record.bucket], this.contribution(record));
    this.recent.push(record);
    this.recordsById.set(record.id, record);
    this.invalidateActive();
    if (this.recent.length > RECENT_CAPACITY) {
      const oldest = this.recent[0];
      this.forget(oldest);
    }
    this.clock++;
    return record;
  }

  /**
   * Bulk-load a memory straight into the long-term tier, bypassing the
   * recent-ring eviction, dedup/reinforcement, and decay policy that
   * addMemory applies. Those are a *working-memory* simulation, appropriate
   * for the interactive demo but wrong for indexing a whole corpus (they'd
   * silently evict all but the last few items, and the O(N) resonance scan
   * in addMemory would make bulk loads O(N^2)). Used by the benchmark
   * harness to index a full retrieval corpus as a permanent store.
   */
  addBulk(text: string, context = ""): MemoryRecord {
    const { record } = this.buildRecord(text, context, "long-term", 1, PROMOTE_TO_LONG_TERM_AT);
    this.field[record.bucket] = add(this.field[record.bucket], this.contribution(record));
    this.longTerm.push(record);
    this.recordsById.set(record.id, record);
    this.invalidateActive();
    this.clock++;
    return record;
  }

  private findResonant(bound: Multivector): MemoryRecord | null {
    let best: MemoryRecord | null = null;
    let bestScore = -Infinity;
    const nb = normalize(bound);
    for (const r of this.allActive()) {
      // r.boundNorm is exactly normalize(r.bound), cached at construction —
      // recomputing it here (as this used to) was a redundant O(nnz) pass
      // per candidate, every single addMemory() call.
      const score = innerProduct(nb, r.boundNorm);
      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
    }
    return bestScore >= RESONANCE_THRESHOLD ? best : null;
  }

  private reinforce(r: MemoryRecord, bound: Multivector, boundHrr: Float64Array): MemoryRecord {
    // Pull the old contribution out, re-bind at the current phase, and put
    // an updated (reinforced) contribution back in.
    this.field[r.bucket] = sub(this.field[r.bucket], this.contribution(r));

    r.repeatCount += 1;
    r.lastSeenTick = this.clock;
    r.rotorAngle = phaseOf(this.clock);
    r.boundNorm = normalize(bound);
    r.rotated = this.rotateAt(bound, this.clock);
    r.boundHrr = boundHrr;

    if (r.repeatCount >= PROMOTE_TO_LONG_TERM_AT) {
      r.tier = "long-term";
      r.weight = 3;
      this.removeFromTierArrays(r.id); // also invalidates the active-record cache
      this.longTerm.push(r);
      this.invalidateActive();
    } else if (r.repeatCount >= PROMOTE_TO_REPEATED_AT) {
      r.tier = "repeated";
      r.weight = 1 + 0.6 * r.repeatCount;
      if (!this.repeated.includes(r)) {
        this.recent = this.recent.filter((x) => x.id !== r.id);
        this.repeated.push(r);
        this.invalidateActive();
      }
    } else {
      r.weight = 1;
    }

    this.field[r.bucket] = add(this.field[r.bucket], this.contribution(r));
    this.clock++;
    return r;
  }

  /** Advance logical time: recent/repeated memories decay and may be forgotten. */
  tick() {
    this.clock++;
    for (const r of [...this.recent]) {
      this.setWeight(r, r.weight * RECENT_DECAY);
      if (r.weight < FORGET_EPSILON) this.forget(r);
    }
    for (const r of [...this.repeated]) {
      this.setWeight(r, r.weight * REPEATED_DECAY);
      if (r.weight < FORGET_EPSILON) this.forget(r);
    }
    // long-term memories do not decay: consolidation is permanent.
  }

  retrieve(
    queryText: string,
    queryContext = "",
    opts: {
      sigma?: number;
      topK?: number;
      centerTick?: number;
      temporal?: boolean;
      temporalAddressing?: TemporalAddressing;
      rotorWeight?: number;
      globalResonance?: boolean;
    } = {}
  ): RetrievalResult {
    const sigma = opts.sigma ?? DEFAULT_SIGMA;
    const topK = opts.topK ?? 5;
    const centerTick = opts.centerTick ?? this.clock;
    // globalResonance/correlation are a coarse field-level reading the
    // interactive UI displays as a stat — real, but unconditionally
    // recomputing them (a full-field add() reduce plus a gp() against it)
    // on every retrieve() call costs real time once field buckets get large
    // (see the README's Cl(12,0) section), for callers (like bench/) that
    // never read them. Opt-out, not opt-in, so the interactive demo's
    // existing behavior doesn't change by default.
    const wantGlobalResonance = opts.globalResonance ?? true;
    // Phase-windowed (temporal) ranking is opt-in. By default the window
    // always centers on "now" (this.clock), which — with no explicit
    // center — would silently bias every search toward whatever was added
    // most recently, regardless of actual content relevance. Content
    // similarity should win a plain search; the Gaussian phase window is a
    // deliberate extra filter for "what happened around this time".
    const temporal = opts.temporal ?? false;
    const temporalAddressing = opts.temporalAddressing ?? this.temporalAddressing;
    const rotorWeight = Math.max(0, Math.min(opts.rotorWeight ?? 0.05, 0.2));

    // Lean query-side encode (encodeQuery/encodeContextQuery): retrieve()
    // never reads .spectrum or .point3d, so it skips computing them — see
    // encodeChannel's `full` param.
    const content = encodeQuery(queryText, this.idf);
    const queryVec = content.sphereVec;
    const hasContext = !!queryContext.trim();
    // Only bind context into the probe if the caller actually supplied one —
    // binding scrambles content into a different representation, so a
    // context-free query must be compared against stored *content* alone,
    // not against the bound (content x context) pattern, or it would almost
    // never resonate.
    const ctxQuery = hasContext ? encodeContextQuery(queryContext, this.idf) : null;
    // Kept for the Clifford field/globalResonance stat below — see
    // MemoryRecord.boundHrr's doc comment for why actual peak scoring
    // (further down) uses the HRR bind (queryBoundHrr) instead.
    const qBlade = hasContext ? gp(content.blade, ctxQuery!.blade) : content.blade;
    // HRR bind (see hrr.ts): what context-bound peak scoring actually uses.
    const queryBoundHrr = hasContext
      ? hrrNormalize(circularConvolve(queryVec, ctxQuery!.sphereVec))
      : null;

    // Hermitian-style inner product against the field. With a context, we
    // can go straight to that context's bucket — the query only has to
    // resonate against the memories that share its bucket, not the whole
    // store, which is most of the point of bucketing. Without a context we
    // don't know which bucket to prefer, so fall back to the sum of all
    // buckets (equivalent to the old single-field behavior). Skippable (see
    // wantGlobalResonance above) since the sum-of-all-buckets case is a
    // full-field add() reduce, and both cases end in a gp() against
    // whatever that field turns out to be — real cost once field buckets
    // are large, for a value plenty of callers never read.
    let correlation: Multivector = mv();
    let globalResonance = 0;
    if (wantGlobalResonance) {
      const targetField = hasContext
        ? this.field[bucketOf(queryContext)]
        : this.field.reduce((acc, f) => add(acc, f), mv());
      // A temporal probe must be bound at the requested center before
      // correlation. Multiplication by its reverse is the unbind operation:
      // contributions stored under the same rotor collapse back toward the
      // query, while different temporal addresses remain rotated away.
      const fieldProbe = temporal ? this.rotateAt(qBlade, centerTick) : qBlade;
      correlation = gp(targetField, reverse(fieldProbe));
      const fieldMag = magnitude(targetField) || 1;
      const qMag = magnitude(fieldProbe) || 1;
      globalResonance = (correlation.get(0) ?? 0) / (fieldMag * qMag);
    }

    // Cleanup / peak extraction: correlate the query against each stored
    // pattern (bound if we have a context to match against, content-only
    // otherwise), weighted by a Gaussian window over temporal phase.
    //
    // With a context, restrict candidates to records sharing the query's
    // bucket — this used to only be true of targetField/globalResonance
    // above (an informational scalar, not part of the ranked output); peaks
    // themselves were scored against every active record regardless of
    // context, which defeated the stated point of bucketing entirely for
    // the results that actually get returned. Filtering here both matches
    // that intent and cuts scoring cost roughly NUM_BUCKETS-fold.
    //
    // Content-only, with buildContentIndex() called: candidates come from
    // the ANN index (ann.ts) instead of a full scan. A first attempt at
    // this inserted into the index incrementally inside addBulk(), which
    // measured 2.4x slower to build than the identical algorithm run as a
    // clean, un-interleaved pass — see ann.ts's header for the numbers and
    // why this version is a caller-triggered batch build instead. Approximate,
    // by construction — the price for whatever `buildContentIndex()`'s
    // caller decided the speedup was worth; retrieve() itself never chooses
    // this over the exact scan on its own. Over-fetches (annEf) past `topK`
    // since the downstream score also folds in `weight` and, when `temporal`
    // is requested, a phase window that can reorder within the fetched set
    // but can't surface a candidate the index search didn't return at all.
    const annEf = Math.max(topK * 4, 100);
    const candidates = hasContext
      ? this.allActive().filter((r) => r.bucket === bucketOf(queryContext))
      : this.ann
        ? this.ann
            .search(queryVec, annEf, annEf)
            .map((id) => this.recordsById.get(id))
            .filter((r): r is MemoryRecord => !!r)
        : this.allActive();
    const centerAngle = phaseOf(centerTick);
    // phaseWeight (a Math.exp call via gaussian(), plus angularDiff) only
    // affects score/ranking when `temporal` is true — the default (false)
    // multiplies it by 1, i.e. discards it, for every candidate except the
    // handful that survive into the final topK. Computing it for all of
    // `candidates` regardless was pure waste at bulk-corpus scale (thousands
    // of discarded candidates paying for a Math.exp each) — this is what
    // actually dominated the holographic engine's per-query cost at SCIDOCS
    // scale, not allActive() (see the cache above, which is a real but
    // smaller win). Deferred here to only the records that make it through
    // ranking; still computed for every candidate up front when `temporal`
    // is true, since then it's part of what ranking has to sort by.
    const peaks: Peak[] = candidates.map((record) => {
      // Context-bound queries score via the HRR bind (see hrr.ts and
      // MemoryRecord.boundHrr) — full SPHERE_DIM circular convolution, no
      // grade compression, unlike the Clifford blade path this replaced for
      // scoring (see the README's "Context-bound queries" section for the
      // measured ~2x nDCG@10 difference). A context-free query has nothing
      // to bind against, so it scores by full-vector cosine on the raw
      // sphere vector instead, same as before.
      const similarity = hasContext
        ? hrrDot(record.boundHrr, queryBoundHrr!)
        : sphereDot(queryVec, record.contentVector);
      const phaseWeight = temporal
        ? temporalAddressing === "distributed"
          ? distributedRotorCorrelation(record.createdAtTick, centerTick).coherence
          : gaussian(angularDiff(record.rotorAngle, centerAngle), sigma)
        : 1;
      const semanticScore = similarity * Math.sqrt(record.weight);
      // Preserve Alex's multiplicative single-plane behavior exactly.  The
      // distributed rotor follows HAM's safer policy: a small additive
      // secondary signal, so time can break semantic ties without erasing a
      // materially better content match.
      const score = !temporal
        ? semanticScore
        : temporalAddressing === "distributed"
          ? (1 - rotorWeight) * semanticScore + rotorWeight * phaseWeight
          : semanticScore * phaseWeight;
      return { record, similarity, phaseWeight, score };
    });
    peaks.sort((a, b) => b.score - a.score);
    const top = peaks.slice(0, topK).filter((p) => p.score > 0.01);
    if (!temporal) {
      // Now fill in the real phaseWeight for just the survivors, so the
      // returned Peak.phaseWeight is always accurate (e.g. for the UI's
      // "phase wt" display) even though it played no part in ranking here.
      for (const p of top) {
        p.phaseWeight = temporalAddressing === "distributed"
          ? distributedRotorCorrelation(p.record.createdAtTick, centerTick).coherence
          : gaussian(angularDiff(p.record.rotorAngle, centerAngle), sigma);
      }
    }

    return { globalResonance, correlation, peaks: top };
  }

  fieldMagnitude(): number {
    return magnitude(this.field.reduce((acc, f) => add(acc, f), mv()));
  }
}

export const EXAMPLE_MEMORIES: Array<[string, string]> = [
  ["Morning coffee ritual before standup", "routine"],
  ["Reviewed the Q3 roadmap deck", "work"],
  ["Walked along the river at sunset", "personal"],
  ["Debugged the retrieval race condition", "work"],
  ["Called mom about the weekend trip", "personal"],
  ["Read a paper on hyperspherical harmonics", "learning"],
  ["Team lunch after the release went out", "work"],
  ["Morning coffee ritual before standup", "routine"],
  ["Practiced scales on the guitar", "hobby"],
  ["Fixed a flaky test in the CI pipeline", "work"],
  ["Grabbed ramen with an old friend", "personal"],
  ["Morning coffee ritual before standup", "routine"],
  ["Watched a documentary about deep sea creatures", "learning"],
  ["Argued about tabs vs spaces in code review", "work"],
  ["Went for a 5k run before breakfast", "routine"],
  ["Repotted the tomato seedlings", "hobby"],
  ["Morning coffee ritual before standup", "routine"],
  ["Paid the electricity bill", "chores"],
  ["Pair-programmed the Clifford algebra module", "work"],
  ["Fell asleep reading a sci-fi novel", "personal"],
  ["Morning coffee ritual before standup", "routine"],
  ["Sketched a new UI concept on the whiteboard", "work"],
  ["Meditated for ten minutes before bed", "routine"],
];

export function seedMemories(store: HoloStore) {
  for (const [text, ctx] of EXAMPLE_MEMORIES) {
    store.addMemory(text, ctx);
  }
}

// Deterministic template combinatorics for generating a large volume of
// distinct memories, used to stress-test add/retrieve speed at scale. Each
// field bucket is a fixed-size (BLADE_COUNT-number) multivector no matter
// how many memories have ever been added, and the tracked-record cleanup set
// is bounded by tier capacities/promotion — so both add and retrieve should
// stay fast even at N=500+, which is the point being demonstrated.
const BULK_VERBS = [
  "Reviewed",
  "Debugged",
  "Refactored",
  "Deployed",
  "Tested",
  "Documented",
  "Sketched",
  "Discussed",
  "Optimized",
  "Migrated",
];
const BULK_SUBJECTS = [
  "the auth service",
  "the retrieval pipeline",
  "the onboarding flow",
  "the billing module",
  "the search index",
  "the notification system",
  "the dashboard",
  "the caching layer",
  "the export tool",
  "the settings page",
];
const BULK_CONTEXTS = ["work", "personal", "learning", "routine", "hobby", "chores"];

export function generateBulkMemories(count: number, offset = 0): Array<[string, string]> {
  const out: Array<[string, string]> = [];
  for (let i = 0; i < count; i++) {
    const n = offset + i;
    const verb = BULK_VERBS[n % BULK_VERBS.length];
    const subject = BULK_SUBJECTS[Math.floor(n / BULK_VERBS.length) % BULK_SUBJECTS.length];
    const ctx = BULK_CONTEXTS[n % BULK_CONTEXTS.length];
    out.push([`${verb} ${subject} #${n}`, ctx]);
  }
  return out;
}
