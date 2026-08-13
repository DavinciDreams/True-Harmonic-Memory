// The holographic memory engine: ties bit->sphere mapping, Gegenbauer
// spectral projection, and Clifford-algebra binding/superposition/retrieval
// into one stateful store, plus a 3-tier (recent / repeated / long-term)
// consolidation policy.
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

export type Tier = "recent" | "repeated" | "long-term";

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
  contentBlade: Multivector;
  contextBlade: Multivector;
  bound: Multivector; // content (x) context, before the time rotor
  boundNorm: Multivector; // normalize(bound), cached — see retrieve()
  rotated: Multivector; // bound after the time rotor is applied
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

function encodeChannel(
  text: string,
  channel: "content" | "context",
  idf?: IdfLookup
): Encoded {
  const sphereVec = textToSphereVector(text, undefined, idf);
  // spectrum: the Gegenbauer harmonic decomposition, kept for the spectrum
  // chart / conceptual fidelity to "decompose into harmonic coefficients".
  const spectrum = Array.from(spectralProject(sphereVec));
  // blade: what's actually used for binding/matching. Content and context
  // are projected onto disjoint grades (see blade.ts) so they're orthogonal
  // channels rather than two signals sharing the same 256 numbers.
  const blade =
    channel === "content" ? sphereToContentBlade(sphereVec) : sphereToContextBlade(sphereVec);
  const point3d = projectTo3D(sphereVec);
  return { spectrum, blade, point3d, sphereVec };
}

/**
 * Encode text on the content channel (grade-1 blade). `idf` is optional —
 * the interactive demo has no fixed corpus to compute document frequencies
 * over, so it's left unweighted (every word counts equally) there. Callers
 * that do have a corpus up front (see HoloStore's constructor, and
 * bench/index.ts) should supply one via buildIdf for higher-quality
 * bundling — see sphere.ts for why.
 */
export function encode(text: string, idf?: IdfLookup): Encoded {
  return encodeChannel(text, "content", idf);
}

/** Encode text on the context channel (grade-3 blade) — see blade.ts. */
export function encodeContext(text: string, idf?: IdfLookup): Encoded {
  return encodeChannel(text, "context", idf);
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

  constructor(opts: { idf?: IdfLookup } = {}) {
    this.idf = opts.idf;
  }
  // One multivector per context bucket instead of a single shared field —
  // see NUM_BUCKETS above.
  field: Multivector[] = Array.from({ length: NUM_BUCKETS }, () => mv());
  recent: MemoryRecord[] = [];
  repeated: MemoryRecord[] = [];
  longTerm: MemoryRecord[] = [];
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
    this.invalidateActive();
  }

  private forget(r: MemoryRecord) {
    this.field[r.bucket] = sub(this.field[r.bucket], this.contribution(r));
    this.removeFromTierArrays(r.id);
  }

  /** Explicitly delete a memory: pull its contribution out of the field and drop it. */
  removeMemory(id: string) {
    const r = this.allActive().find((x) => x.id === id);
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
    const ctx = encodeContext(context || "∅", this.idf); // empty-context placeholder
    const bound = gp(content.blade, ctx.blade);
    const bucket = bucketOf(context);
    const angle = phaseOf(this.clock);
    const rotated = sandwich(timeRotor(angle), bound);
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
      contentBlade: content.blade,
      contextBlade: ctx.blade,
      bound,
      boundNorm: normalize(bound),
      rotated,
      point3d: content.point3d,
      contentVector: content.sphereVec,
    };
    return { record, bound };
  }

  /** Bind text+context, fold in the current temporal phase, and add to the field. */
  addMemory(text: string, context = ""): MemoryRecord {
    const { record, bound } = this.buildRecord(text, context, "recent", 1, 0);
    const existing = this.findResonant(bound);
    if (existing) return this.reinforce(existing, bound);

    this.field[record.bucket] = add(this.field[record.bucket], this.contribution(record));
    this.recent.push(record);
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
    this.invalidateActive();
    this.clock++;
    return record;
  }

  private findResonant(bound: Multivector): MemoryRecord | null {
    let best: MemoryRecord | null = null;
    let bestScore = -Infinity;
    const nb = normalize(bound);
    for (const r of this.allActive()) {
      const score = innerProduct(nb, normalize(r.bound));
      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
    }
    return bestScore >= RESONANCE_THRESHOLD ? best : null;
  }

  private reinforce(r: MemoryRecord, bound: Multivector): MemoryRecord {
    // Pull the old contribution out, re-bind at the current phase, and put
    // an updated (reinforced) contribution back in.
    this.field[r.bucket] = sub(this.field[r.bucket], this.contribution(r));

    r.repeatCount += 1;
    r.lastSeenTick = this.clock;
    r.rotorAngle = phaseOf(this.clock);
    r.bound = bound;
    r.boundNorm = normalize(bound);
    r.rotated = sandwich(timeRotor(r.rotorAngle), bound);

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

    const content = encode(queryText, this.idf);
    const queryVec = content.sphereVec;
    const hasContext = !!queryContext.trim();
    // Only bind context into the probe if the caller actually supplied one —
    // binding is a geometric product, which deliberately scrambles content
    // into a different subspace, so a context-free query must be compared
    // against stored *content* alone, not against the bound (content x
    // context) pattern, or it would almost never resonate.
    const qBlade = hasContext
      ? gp(content.blade, encodeContext(queryContext, this.idf).blade)
      : content.blade;

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
      correlation = gp(targetField, reverse(qBlade));
      const fieldMag = magnitude(targetField) || 1;
      const qMag = magnitude(qBlade) || 1;
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
    const candidates = hasContext
      ? this.allActive().filter((r) => r.bucket === bucketOf(queryContext))
      : this.allActive();
    const centerAngle = phaseOf(centerTick);
    // Hoisted out of the per-record loop below: normalize(qBlade) doesn't
    // depend on the record, so computing it once per query instead of once
    // per (query, record) pair turns an O(N) loop with an O(BLADE_COUNT)
    // normalize() nested inside it back into a real O(N) loop.
    const qBladeNorm = normalize(qBlade);
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
      // Context-bound queries must go through the blade/binding path — that's
      // the only way to score "this content in this context" as one signal.
      // A context-free query has nothing to preserve orthogonality against,
      // so score it by full-vector cosine on the un-compressed sphere vector
      // instead of the grade-1 blade: the blade only keeps NUM_GENERATORS of
      // SPHERE_DIM dimensions (see clifford.ts), and reranking against the
      // full vector recovers most of what that compression would otherwise
      // cost, at no extra asymptotic expense (this loop is already O(N)).
      const similarity = hasContext
        ? innerProduct(record.boundNorm, qBladeNorm)
        : sphereDot(queryVec, record.contentVector);
      const phaseWeight = temporal
        ? gaussian(angularDiff(record.rotorAngle, centerAngle), sigma)
        : 1;
      const score = similarity * phaseWeight * Math.sqrt(record.weight);
      return { record, similarity, phaseWeight, score };
    });
    peaks.sort((a, b) => b.score - a.score);
    const top = peaks.slice(0, topK).filter((p) => p.score > 0.01);
    if (!temporal) {
      // Now fill in the real phaseWeight for just the survivors, so the
      // returned Peak.phaseWeight is always accurate (e.g. for the UI's
      // "phase wt" display) even though it played no part in ranking here.
      for (const p of top) {
        p.phaseWeight = gaussian(angularDiff(p.record.rotorAngle, centerAngle), sigma);
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
