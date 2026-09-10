// Benchmark: this repo's holographic engine vs. two "normal RAG"
// baselines, all sharing the exact same embedding
// (textToSphereVector, this repo's own hash->hypersphere encoder) so the
// comparison isolates *retrieval mechanism* (holographic field vs.
// brute-force cosine vs. approximate HNSW) rather than embedding quality.
//
// Dataset: any BEIR dataset (MTEB retrieval suite), used unmodified — e.g.
// nfcorpus (~3.6K docs, default), fiqa (~57K docs). Fetch it first:
// pnpm bench:fetch [dataset]
//
// Run: pnpm bench [--dataset name] [--limit N] [--ann]
//   --dataset  which fetched BEIR dataset to use (default: nfcorpus)
//   --limit    cap on #queries evaluated, for a quick smoke run; omit to
//              evaluate the full test-split query set
//   --ann      opt the holographic engine's content-only path into its
//              batch-built ANN index (HoloStore.buildContentIndex(), see
//              src/lib/holo/ann.ts) instead of its default exact scan.
//              Off by default: the exact scan is what ties brute-force
//              cosine's quality exactly (this engine's headline result);
//              --ann trades that for speed at bulk-corpus scale, same
//              trade the HNSW baseline row already makes.

import { gloveAvailable, loadGlove, textToGloveVector } from "./glove";
import { HNSW, Vector } from "./hnsw";
import { assertDataPresent, loadCorpus, loadQrels, loadQueries } from "./loadNfcorpus";
import { MetricSummary, QueryResult, summarize } from "./metrics";
import { HoloStore } from "../src/lib/holo/engine";
import { buildIdf, textToSphereVector, SPHERE_DIM } from "../src/lib/holo/sphere";

const TOP_K = 100;

function now(): number {
  return performance.now();
}

function parseLimit(): number | undefined {
  const i = process.argv.indexOf("--limit");
  if (i === -1) return undefined;
  const n = Number(process.argv[i + 1]);
  return Number.isFinite(n) ? n : undefined;
}

function parseDataset(): string {
  const i = process.argv.indexOf("--dataset");
  return i === -1 ? "nfcorpus" : process.argv[i + 1];
}

function parseAnn(): boolean {
  return process.argv.includes("--ann");
}

// Override ann.ts's defaults, for tuning experiments without editing the
// module itself — e.g. `pnpm bench --ann --ann-m 12 --ann-ef 60`.
function parseIntFlag(flag: string): number | undefined {
  const i = process.argv.indexOf(flag);
  if (i === -1) return undefined;
  const n = Number(process.argv[i + 1]);
  return Number.isFinite(n) ? n : undefined;
}

function fmtMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`;
}

function printTable(rows: Array<{ name: string; indexMs: number; summary: MetricSummary }>) {
  type ColumnKey = "name" | "indexMs" | keyof MetricSummary;
  const cols: ReadonlyArray<readonly [string, ColumnKey]> = [
    ["Method", "name"],
    ["nDCG@10", "ndcg10"],
    ["Recall@100", "recall100"],
    ["MRR@10", "mrr10"],
    ["Index (ms)", "indexMs"],
    ["Avg query", "avgQueryMs"],
    ["p95 query", "p95QueryMs"],
  ];

  const fmt = (row: (typeof rows)[number], key: ColumnKey): string => {
    if (key === "name") return row.name;
    if (key === "indexMs") return fmtMs(row.indexMs);
    if (key === "avgQueryMs") return fmtMs(row.summary.avgQueryMs);
    if (key === "p95QueryMs") return fmtMs(row.summary.p95QueryMs);
    const v = row.summary[key];
    return v.toFixed(4);
  };

  const widths = cols.map(([label], i) =>
    Math.max(label.length, ...rows.map((r) => fmt(r, cols[i][1]).length))
  );
  const line = (cells: string[]) =>
    "  " + cells.map((c, i) => c.padEnd(widths[i])).join("  |  ");
  console.log(line(cols.map(([label]) => label)));
  console.log(line(widths.map((w) => "-".repeat(w))));
  for (const row of rows) console.log(line(cols.map(([, key]) => fmt(row, key))));
}

async function main() {
  const dataset = parseDataset();
  assertDataPresent(dataset);
  const limit = parseLimit();
  const useAnn = parseAnn();
  const annM = parseIntFlag("--ann-m");
  const annEfConstruction = parseIntFlag("--ann-ef");

  console.log(`Loading ${dataset}...`);
  const [corpus, allQueries, qrels] = await Promise.all([
    loadCorpus(dataset),
    loadQueries(dataset),
    loadQrels(dataset, "test"),
  ]);
  // Only evaluate queries that actually have judgments in the test split
  // (a dataset's queries.jsonl typically includes train/dev queries too).
  let queries = allQueries.filter((q) => qrels.has(q.id));
  if (limit) queries = queries.slice(0, limit);
  console.log(
    `${corpus.length} docs, ${queries.length} judged test queries` +
      (limit ? ` (--limit ${limit})` : "")
  );

  // ---- Shared embedding: this repo's own sphere-vector encoder ----------
  // IDF computed once over the corpus and shared by all three methods below
  // (brute-force, HNSW, and HoloStore's own encode calls) — otherwise this
  // benchmark would stop isolating retrieval mechanism the moment only one
  // of them got smarter term weighting.
  console.log("Encoding corpus to sphere vectors...");
  const idf = buildIdf(corpus.map((d) => d.text));
  const docVecs: Vector[] = new Array(corpus.length);
  for (let i = 0; i < corpus.length; i++)
    docVecs[i] = textToSphereVector(corpus[i].text, undefined, idf);
  const idOf = corpus.map((d) => d.id);

  const rows: Array<{ name: string; indexMs: number; summary: MetricSummary }> = [];

  // ---- Baseline A: brute-force cosine (exact top-k under this embedding) --
  {
    console.log("\n=== Brute-force cosine ===");
    const t0 = now();
    // No index to build — brute force has zero build cost, all cost is per-query.
    const indexMs = now() - t0;

    const results: QueryResult[] = queries.map((q) => {
      const qv = textToSphereVector(q.text, undefined, idf);
      const t = now();
      const scored = docVecs.map((v, i) => ({ i, s: dot(qv, v) }));
      scored.sort((a, b) => b.s - a.s);
      const rankedIds = scored.slice(0, TOP_K).map((x) => idOf[x.i]);
      const ms = now() - t;
      return { queryId: q.id, rankedIds, ms };
    });
    rows.push({ name: "Brute-force cosine", indexMs, summary: summarize(results, qrels) });
  }

  // ---- Baseline B: HNSW over the same sphere vectors (approximate ANN) --
  {
    console.log("=== HNSW (approximate) ===");
    const index = new HNSW({ M: 16, efConstruction: 100, seed: 42 });
    const t0 = now();
    for (const v of docVecs) index.insert(v);
    const indexMs = now() - t0;

    const results: QueryResult[] = queries.map((q) => {
      const qv = textToSphereVector(q.text, undefined, idf);
      const t = now();
      const ids = index.search(qv, TOP_K, 100);
      const rankedIds = ids.map((i) => idOf[i]);
      const ms = now() - t;
      return { queryId: q.id, rankedIds, ms };
    });
    rows.push({ name: "HNSW (approx.)", indexMs, summary: summarize(results, qrels) });
  }

  // ---- This repo: holographic Clifford-algebra engine --------------------
  {
    console.log(`=== Holographic engine (this repo)${useAnn ? " [--ann]" : ""} ===`);
    const store = new HoloStore({ idf });
    const t0 = now();
    for (const d of corpus) store.addBulk(d.text);
    // Batch ANN build, if requested, is timed as part of index cost — same
    // convention as the HNSW baseline above (its indexMs is build-only too).
    if (useAnn) store.buildContentIndex({ M: annM, efConstruction: annEfConstruction });
    const indexMs = now() - t0;

    const results: QueryResult[] = queries.map((q) => {
      const t = now();
      const r = store.retrieve(q.text, "", { topK: TOP_K, globalResonance: false });
      const rankedIds = r.peaks.map((p) => idOfFromText(p.record, corpus));
      const ms = now() - t;
      return { queryId: q.id, rankedIds, ms };
    });
    rows.push({
      name: `Holographic engine${useAnn ? " [--ann]" : ""}`,
      indexMs,
      summary: summarize(results, qrels),
    });
  }

  // ---- Optional 4th row: real GloVe embeddings (reference, not isolated) --
  // Unlike the three rows above, this one deliberately does NOT share the
  // sphere-vector embedding — it's brute-force cosine over averaged GloVe
  // word vectors instead, included only to answer "how much does embedding
  // quality matter, versus retrieval mechanism". Skipped if GloVe hasn't
  // been fetched (pnpm bench:fetch:glove); this repo's core engine
  // (src/lib/holo) never uses real embeddings — see bench/glove.ts.
  if (gloveAvailable()) {
    console.log("=== GloVe average (real embedding, reference — not mechanism-isolated) ===");
    const t0 = now();
    const gloveVocab = await loadGlove();
    const gloveDocVecs = corpus.map((d) => textToGloveVector(d.text, gloveVocab, idf));
    const indexMs = now() - t0;

    // Transparency check: how much of the corpus's vocabulary GloVe's ~400K
    // general-domain words actually cover. A high OOV rate on a specialized
    // corpus (medical/nutrition jargon, drug names, ...) would explain a
    // weak GloVe score as "wrong reference vocabulary", not "real embeddings
    // don't help" — worth knowing which one it is before drawing conclusions.
    const allWords = new Set<string>();
    for (const d of corpus) for (const w of d.text.toLowerCase().match(/[a-z0-9]+/g) ?? []) allWords.add(w);
    let oov = 0;
    for (const w of allWords) if (!gloveVocab.has(w)) oov++;
    console.log(
      `    (corpus vocabulary: ${allWords.size} unique words, ` +
        `${((oov / allWords.size) * 100).toFixed(1)}% not found in GloVe)`
    );

    const results: QueryResult[] = queries.map((q) => {
      const qv = textToGloveVector(q.text, gloveVocab, idf);
      const t = now();
      const scored = gloveDocVecs.map((v, i) => ({ i, s: dot(qv, v) }));
      scored.sort((a, b) => b.s - a.s);
      const rankedIds = scored.slice(0, TOP_K).map((x) => idOf[x.i]);
      const ms = now() - t;
      return { queryId: q.id, rankedIds, ms };
    });
    rows.push({ name: "GloVe average (real embedding)", indexMs, summary: summarize(results, qrels) });
  } else {
    console.log(
      "(skipping GloVe reference row — run `pnpm bench:fetch:glove` to include it)"
    );
  }

  console.log(`\nResults (${dataset} test split, shared sphere-vector embedding, top-${TOP_K}):\n`);
  printTable(rows);
  console.log(
    "\nNotes:\n" +
      "  - Brute-force cosine's nDCG/Recall/MRR is the ceiling for this embedding: it's\n" +
      "    exact top-k, so HNSW and the holographic engine are both compared against it\n" +
      "    as well as against the real qrels.\n" +
      "  - This benchmark's queries carry no context, so the holographic engine's\n" +
      "    retrieve() reranks by full-vector cosine on the un-compressed sphere vector\n" +
      "    (see engine.ts) rather than any blade or bound representation — it should\n" +
      "    track brute-force closely here. A context-bound query instead scores via\n" +
      `    HRR circular convolution (src/lib/holo/hrr.ts, full SPHERE_DIM=${SPHERE_DIM}) — see\n` +
      `    pnpm bench:context for that path; not exercised by this benchmark.\n` +
      "  - HNSW's gap from brute-force (at equal embedding) is the approximation cost;\n" +
      "    its speed advantage should grow with corpus size beyond NFCorpus's ~3.6K docs.\n" +
      "  - The GloVe row (if present) is NOT under the same-embedding isolation the other\n" +
      "    three share — it's a different (real) embedding entirely, included only to show\n" +
      "    the gap between this project's hash-based encoder and a real one, at fixed\n" +
      "    retrieval mechanism (brute-force cosine for both)."
  );
}

function dot(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// MemoryRecord doesn't carry the original corpus doc id (it's a text/context
// store, not a keyed index) — recover it by matching text back to the
// corpus. Built once per store via closure-cached map for O(1) lookups.
let textToIdCache: Map<string, string> | null = null;
function idOfFromText(record: { text: string }, corpus: { id: string; text: string }[]): string {
  if (!textToIdCache) {
    textToIdCache = new Map(corpus.map((d) => [d.text, d.id]));
  }
  return textToIdCache.get(record.text) ?? "";
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
