// Minimal from-scratch HNSW (Hierarchical Navigable Small World) index —
// no native deps, so it runs anywhere pnpm bench does. Implements the
// standard algorithm (Malkov & Yashunin 2016) at a scale appropriate for a
// few thousand vectors: multi-layer proximity graph, greedy descent from
// the top layer, beam search (efConstruction/efSearch) at each layer.
//
// This is deliberately simple over "fast": candidate sets are built with
// plain arrays + sort rather than binary heaps. Fine at NFCorpus scale
// (~3.6K vectors); a production HNSW would use heaps and neighbor-diversity
// heuristics for pruning.

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

interface Node {
  id: number;
  vec: Vector;
  level: number;
  neighbors: number[][]; // neighbors[layer] = array of node ids
}

export class HNSW {
  private nodes: Node[] = [];
  private entryPoint = -1;
  private readonly M: number;
  private readonly maxM: number;
  private readonly maxM0: number;
  private readonly efConstruction: number;
  private readonly mL: number;
  private rngState: number;

  constructor(opts: { M?: number; efConstruction?: number; seed?: number } = {}) {
    this.M = opts.M ?? 16;
    this.maxM = this.M;
    this.maxM0 = this.M * 2;
    this.efConstruction = opts.efConstruction ?? 100;
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
    // `found` (the algorithm's "W" set) must stay capped at `ef` throughout —
    // that's what keeps each search O(ef log ef) instead of O(visited log
    // visited) where "visited" grows with the graph's total size over time.
    // The previous version let this grow unbounded (every neighbor ever
    // touched stayed in `found` forever) and re-sorted the whole thing on
    // every loop iteration — O(N log N)-per-insert in principle, but the
    // constant grew with graph density, so real runs scaled far worse than
    // that in practice (confirmed: NFCorpus's ~3.6K-doc index took ~36s, but
    // a 7x-larger corpus didn't finish in 15x more time, let alone 7x).
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
        // Only a candidate for `found`/further expansion if it could plausibly
        // improve the current top-ef, or we don't have ef results yet.
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

  insert(vec: Vector): number {
    const id = this.nodes.length;
    const level = this.randomLevel();
    const node: Node = { id, vec, level, neighbors: Array.from({ length: level + 1 }, () => []) };
    this.nodes.push(node);

    if (this.entryPoint === -1) {
      this.entryPoint = id;
      return id;
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
      for (const nId of chosen) {
        const back = this.nodes[nId];
        back.neighbors[layer] = back.neighbors[layer] ?? [];
        back.neighbors[layer].push(id);
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
    if (level > topLayer) this.entryPoint = id;
    return id;
  }

  /** Returns up to k nearest node ids, best (most similar) first. */
  search(query: Vector, k: number, ef = Math.max(k, 50)): number[] {
    if (this.entryPoint === -1) return [];
    let ep = [this.entryPoint];
    const topLayer = this.nodes[this.entryPoint].level;
    for (let layer = topLayer; layer > 0; layer--) {
      ep = this.searchLayer(query, ep, 1, layer);
    }
    return this.searchLayer(query, ep, Math.max(ef, k), 0).slice(0, k);
  }

  get size(): number {
    return this.nodes.length;
  }
}
