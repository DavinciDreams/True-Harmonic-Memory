// Approximate nearest-neighbor index (HNSW — Malkov & Yashunin 2016) for
// HoloStore's content-only retrieval path, built as an explicit batch pass
// (see HoloStore.buildContentIndex()) rather than incrementally per add.
//
// First attempt at this (see git history / README) inserted incrementally
// inside addBulk(), interleaved with the per-record Clifford-algebra work
// (blade projection, gp(), sandwich() — each allocating its own Maps).
// Measured cost on SCIDOCS (25.7K docs): 334s of ANN insert time alone,
// vs. 141s for the *identical* algorithm/params run standalone
// (bench/hnsw.ts, no other allocation happening alongside it) — a 2.4x
// penalty from sharing a heap with a second, unrelated, allocation-heavy
// workload, not anything intrinsic to HNSW. That version also showed the
// query-side win didn't need `efConstruction=100`-grade precision to hold
// up. This version batches construction into one dedicated pass (avoiding
// the interleaving penalty) and defaults to a leaner graph (lower
// M/efConstruction — coarser, faster to build, still returns useful
// candidates for the rerank that follows).
//
// No delete/tombstone support this time: a batch-built index has an
// explicit build step already, so a store whose contents changed enough to
// matter can just call buildContentIndex() again rather than needing live
// mutation support baked into the index itself — see that method's doc
// comment in engine.ts.
//
// M/efConstruction=12/60 (below) is the middle of three configs measured on
// SCIDOCS (25.7K docs), none of which beats a dedicated HNSW baseline
// outright — this is a real speed/quality dial, not a solved trade-off:
//   M=8,  efConstruction=40:  nDCG@10 88.9% of ceiling, 61.8s build,  11.60ms/query
//   M=12, efConstruction=60:  nDCG@10 97.8% of ceiling, 163.8s build, 21.09ms/query  <- default
//   M=16, efConstruction=100: nDCG@10 98.6% of ceiling, 384.6s build, 46.91ms/query (see git history — this
//                              was the first, interleaved-insert attempt, since redone as a batch build)
//   HNSW baseline (bench/hnsw.ts, same M/ef=16/100): nDCG@10 96.1% of ceiling, 62.7s build, 5.28ms/query
// 12/60 was picked for edging out HNSW's own quality (97.8% vs 96.1%) while
// still being faster to build and query than the 16/100 config — not
// because it dominates HNSW (it doesn't: HNSW still builds and queries
// faster). Reasonable default for "I want this store's own index, faster
// than its exact scan, without pulling in a separate ANN dependency" — not
// a claim that it beats a dedicated HNSW index at its own game.

export type Vector = Float64Array;

function dot(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Assuming unit-normalized vectors, cosine distance = 1 - dot. */
function dist(a: Vector, b: Vector): number {
  return 1 - dot(a, b);
}

interface Node<Id> {
  id: Id;
  vec: Vector;
  level: number;
  neighbors: number[][]; // neighbors[layer] = array of internal node indices
}

export class ANNIndex<Id> {
  private nodes: Node<Id>[] = [];
  private entryPoint = -1;
  private readonly M: number;
  private readonly maxM: number;
  private readonly maxM0: number;
  private readonly efConstruction: number;
  private readonly mL: number;
  private rngState: number;

  // Leaner defaults than bench/hnsw.ts's M=16/efConstruction=100 — see this
  // file's header comment for the measured trade-off table these came from.
  constructor(opts: { M?: number; efConstruction?: number; seed?: number } = {}) {
    this.M = opts.M ?? 12;
    this.maxM = this.M;
    this.maxM0 = this.M * 2;
    this.efConstruction = opts.efConstruction ?? 60;
    this.mL = 1 / Math.log(this.M);
    this.rngState = (opts.seed ?? 1234) >>> 0;
  }

  private rand(): number {
    // xorshift32, deterministic so index construction is reproducible.
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return this.rngState / 4294967296;
  }

  private randomLevel(): number {
    return Math.floor(-Math.log(this.rand() || 1e-9) * this.mL);
  }

  private searchLayer(query: Vector, entryIds: number[], ef: number, layer: number): number[] {
    const visited = new Set<number>(entryIds);
    const candidates = entryIds.map((id) => ({ id, d: dist(query, this.nodes[id].vec) }));
    candidates.sort((a, b) => a.d - b.d);
    const found = candidates.slice();

    while (candidates.length > 0) {
      const c = candidates.shift()!;
      const worstFound = found[Math.min(found.length, ef) - 1];
      if (worstFound && c.d > worstFound.d && found.length >= ef) break;

      const newlyFound: { id: number; d: number }[] = [];
      for (const nId of this.nodes[c.id].neighbors[layer] ?? []) {
        if (visited.has(nId)) continue;
        visited.add(nId);
        const d = dist(query, this.nodes[nId].vec);
        const worst = found[Math.min(found.length, ef) - 1];
        if (found.length < ef || !worst || d < worst.d) {
          candidates.push({ id: nId, d });
          newlyFound.push({ id: nId, d });
        }
      }
      if (newlyFound.length > 0) {
        found.push(...newlyFound);
        found.sort((a, b) => a.d - b.d);
        if (found.length > ef) found.length = ef;
        candidates.sort((a, b) => a.d - b.d);
      }
    }
    return found.slice(0, ef).map((f) => f.id);
  }

  private insert(id: Id, vec: Vector): void {
    const idx = this.nodes.length;
    const level = this.randomLevel();
    const node: Node<Id> = { id, vec, level, neighbors: Array.from({ length: level + 1 }, () => []) };
    this.nodes.push(node);

    if (this.entryPoint === -1) {
      this.entryPoint = idx;
      return;
    }

    let ep = [this.entryPoint];
    const topLayer = this.nodes[this.entryPoint].level;
    for (let layer = topLayer; layer > level; layer--) {
      ep = this.searchLayer(vec, ep, 1, layer);
    }
    for (let layer = Math.min(level, topLayer); layer >= 0; layer--) {
      const candidates = this.searchLayer(vec, ep, this.efConstruction, layer);
      const maxNeighbors = layer === 0 ? this.maxM0 : this.maxM;
      const chosen = candidates.slice(0, this.M);
      node.neighbors[layer] = chosen;
      for (const nIdx of chosen) {
        const back = this.nodes[nIdx];
        back.neighbors[layer] = back.neighbors[layer] ?? [];
        back.neighbors[layer].push(idx);
        if (back.neighbors[layer].length > maxNeighbors) {
          back.neighbors[layer] = back.neighbors[layer]
            .map((nb) => ({ id: nb, d: dist(back.vec, this.nodes[nb].vec) }))
            .sort((a, b) => a.d - b.d)
            .slice(0, maxNeighbors)
            .map((x) => x.id);
        }
      }
      ep = candidates;
    }
    if (level > topLayer) this.entryPoint = idx;
  }

  /**
   * Build the whole index in one pass from (id, vector) pairs. The only way
   * to populate this index — see this file's header comment for why batch,
   * not incremental insert, is the point.
   */
  static build<Id>(
    entries: Array<[Id, Vector]>,
    opts: { M?: number; efConstruction?: number; seed?: number } = {}
  ): ANNIndex<Id> {
    const index = new ANNIndex<Id>(opts);
    for (const [id, vec] of entries) index.insert(id, vec);
    return index;
  }

  /** Up to `k` nearest ids, best (most similar) first. */
  search(query: Vector, k: number, ef = Math.max(k, 50)): Id[] {
    if (this.entryPoint === -1) return [];
    let ep = [this.entryPoint];
    const topLayer = this.nodes[this.entryPoint].level;
    for (let layer = topLayer; layer > 0; layer--) {
      ep = this.searchLayer(query, ep, 1, layer);
    }
    return this.searchLayer(query, ep, Math.max(ef, k), 0)
      .slice(0, k)
      .map((idx) => this.nodes[idx].id);
  }

  get size(): number {
    return this.nodes.length;
  }
}
