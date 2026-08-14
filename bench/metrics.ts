// Standard IR retrieval metrics, computed the same way BEIR/MTEB report
// them: nDCG@10 (graded relevance, the primary MTEB retrieval metric),
// Recall@100, and MRR@10 (binary relevance: any qrel score > 0 counts).

import type { Qrels } from "./loadNfcorpus";

function dcg(gains: number[]): number {
  let s = 0;
  for (let i = 0; i < gains.length; i++) s += gains[i] / Math.log2(i + 2);
  return s;
}

/** rankedIds: this query's retrieved doc ids, best first. */
export function ndcgAtK(rankedIds: string[], rel: Map<string, number>, k: number): number {
  const gains = rankedIds.slice(0, k).map((id) => rel.get(id) ?? 0);
  const ideal = [...rel.values()].sort((a, b) => b - a).slice(0, k);
  const idealDcg = dcg(ideal);
  return idealDcg === 0 ? 0 : dcg(gains) / idealDcg;
}

export function recallAtK(rankedIds: string[], rel: Map<string, number>, k: number): number {
  const relevant = new Set([...rel.entries()].filter(([, s]) => s > 0).map(([id]) => id));
  if (relevant.size === 0) return 0;
  const hit = rankedIds.slice(0, k).filter((id) => relevant.has(id));
  return hit.length / relevant.size;
}

export function mrrAtK(rankedIds: string[], rel: Map<string, number>, k: number): number {
  for (let i = 0; i < Math.min(k, rankedIds.length); i++) {
    if ((rel.get(rankedIds[i]) ?? 0) > 0) return 1 / (i + 1);
  }
  return 0;
}

export interface QueryResult {
  queryId: string;
  rankedIds: string[]; // best first
  ms: number; // this query's search latency
}

export interface MetricSummary {
  ndcg10: number;
  recall100: number;
  mrr10: number;
  avgQueryMs: number;
  p95QueryMs: number;
  evaluated: number;
}

export function summarize(results: QueryResult[], qrels: Qrels): MetricSummary {
  let ndcgSum = 0,
    recallSum = 0,
    mrrSum = 0,
    n = 0;
  const timings: number[] = [];
  for (const r of results) {
    const rel = qrels.get(r.queryId);
    if (!rel) continue; // some queries have no judgments in this split
    ndcgSum += ndcgAtK(r.rankedIds, rel, 10);
    recallSum += recallAtK(r.rankedIds, rel, 100);
    mrrSum += mrrAtK(r.rankedIds, rel, 10);
    timings.push(r.ms);
    n++;
  }
  timings.sort((a, b) => a - b);
  const p95 = timings.length ? timings[Math.min(timings.length - 1, Math.floor(timings.length * 0.95))] : 0;
  const avg = timings.length ? timings.reduce((a, b) => a + b, 0) / timings.length : 0;
  return {
    ndcg10: n ? ndcgSum / n : 0,
    recall100: n ? recallSum / n : 0,
    mrr10: n ? mrrSum / n : 0,
    avgQueryMs: avg,
    p95QueryMs: p95,
    evaluated: n,
  };
}
