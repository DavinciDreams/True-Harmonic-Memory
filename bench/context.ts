// Benchmark: does the holographic engine's context-bound retrieval path
// (HRR circular convolution binding, see src/lib/holo/hrr.ts and engine.ts —
// this used to be grade-3 blade binding via src/lib/holo/blade.ts, replaced
// after this same benchmark showed it losing ~2x nDCG@10 to the compression
// that scheme required; see the root README's Design notes) actually help,
// or hurt?
//
// bench/index.ts never exercises this path — NFCorpus queries carry no
// context, so every query there goes through the content-only rerank
// (full-vector cosine) instead of the HRR-binding path. This script
// synthesizes context: docs and queries are tagged by a shared set of topic
// keywords (crude but deterministic and content-derived, not random), which
// gives every query a "correct" context bucket to be scored against.
//
// Four variants are compared, all sharing the same corpus/queries/qrels
// and the same IDF-weighted embedding:
//   - context-bound   : store.retrieve(text, queryContext) (HRR bind + hash bucket)
//   - HRR-bound        : standalone prototype of the above, same-tag restricted
//                        instead of hash-bucketed — included to confirm the
//                        live integration reproduces the prototype's numbers
//   - content-only     : store.retrieve(text, "")            (same store, no context)
//   - bucket ceiling   : brute-force cosine, restricted to docs sharing the
//                        query's topic tag — the best any method could do if
//                        it perfectly knew and used the right bucket.
//
// Run: pnpm bench:context [-- --limit N]

import { assertDataPresent, loadCorpus, loadQrels, loadQueries } from "./loadNfcorpus";
import { MetricSummary, QueryResult, summarize } from "./metrics";
import { HoloStore } from "../src/lib/holo/engine";
import { buildIdf, textToSphereVector, SPHERE_DIM } from "../src/lib/holo/sphere";
import { NUM_GENERATORS } from "../src/lib/holo/clifford";
import { circularConvolve, normalize as hrrNormalize, dot as hrrDot } from "../src/lib/holo/hrr";

const TOP_K = 100;

// Order matters: first match wins, so put more specific terms first where
// they could otherwise be shadowed by a broader one.
const TOPIC_KEYWORDS = [
  "cancer", "diabetes", "cardiovascular", "heart", "cholesterol", "obesity",
  "vitamin", "protein", "arthritis", "alzheimer", "depression", "kidney",
  "bone", "sugar", "fat", "diet",
];

function tagOf(text: string): string {
  const lower = text.toLowerCase();
  for (const kw of TOPIC_KEYWORDS) if (lower.includes(kw)) return kw;
  return "general";
}

function now(): number {
  return performance.now();
}

function parseLimit(): number | undefined {
  const i = process.argv.indexOf("--limit");
  if (i === -1) return undefined;
  const n = Number(process.argv[i + 1]);
  return Number.isFinite(n) ? n : undefined;
}

function fmtMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(2)}ms`;
}

function printTable(rows: Array<{ name: string; summary: MetricSummary }>) {
  type ColumnKey = "name" | keyof MetricSummary;
  const cols: ReadonlyArray<readonly [string, ColumnKey]> = [
    ["Method", "name"],
    ["nDCG@10", "ndcg10"],
    ["Recall@100", "recall100"],
    ["MRR@10", "mrr10"],
    ["Avg query", "avgQueryMs"],
    ["Evaluated", "evaluated"],
  ];

  const fmt = (row: (typeof rows)[number], key: ColumnKey): string => {
    if (key === "name") return row.name;
    if (key === "avgQueryMs") return fmtMs(row.summary.avgQueryMs);
    if (key === "evaluated") return String(row.summary.evaluated);
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

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function main() {
  const dataset = "nfcorpus"; // synthetic topic tags below are NFCorpus-specific
  assertDataPresent(dataset);
  const limit = parseLimit();

  console.log(`Loading ${dataset}...`);
  const [corpus, allQueries, qrels] = await Promise.all([
    loadCorpus(dataset),
    loadQueries(dataset),
    loadQrels(dataset, "test"),
  ]);

  const docContext = corpus.map((d) => tagOf(d.text));
  const queryContext = allQueries.map((q) => tagOf(q.text));

  // Only queries that (a) have judgments in the test split, and (b) got a
  // real topic tag (not "general") — a "general"-tagged query has no
  // meaningful bucket to be scored against, by construction of this synthetic
  // eval, not a limitation of the engine.
  let queries = allQueries
    .map((q, i) => ({ q, tag: queryContext[i] }))
    .filter(({ q, tag }) => qrels.has(q.id) && tag !== "general");
  if (limit) queries = queries.slice(0, limit);

  const taggedDocs = docContext.filter((t) => t !== "general").length;
  console.log(
    `${corpus.length} docs (${taggedDocs} tagged, ${((taggedDocs / corpus.length) * 100).toFixed(1)}%), ` +
      `${queries.length} judged+tagged test queries` + (limit ? ` (--limit ${limit})` : "")
  );

  console.log("Encoding corpus (IDF-weighted, shared with bench/index.ts)...");
  const idf = buildIdf(corpus.map((d) => d.text));
  const docVecs = corpus.map((d) => textToSphereVector(d.text, undefined, idf));
  const idOf = corpus.map((d) => d.id);

  console.log("Indexing holographic engine (context-tagged)...");
  const store = new HoloStore({ idf });
  for (let i = 0; i < corpus.length; i++) store.addBulk(corpus[i].text, docContext[i]);

  // --- HRR prototype: same corpus/tags, different binding scheme only
  // (circular convolution at full SPHERE_DIM instead of grade-3 blade
  // compression) — see src/lib/holo/hrr.ts for why this is worth testing.
  console.log("Building HRR-bound vectors (circular convolution, same-tag restricted)...");
  const tagVecCache = new Map<string, Float64Array>();
  const tagVec = (tag: string): Float64Array => {
    let v = tagVecCache.get(tag);
    if (!v) {
      v = textToSphereVector(tag, undefined, idf);
      tagVecCache.set(tag, v);
    }
    return v;
  };
  const docBoundHrr = corpus.map((d, i) =>
    hrrNormalize(circularConvolve(docVecs[i], tagVec(docContext[i])))
  );
  const textToId = new Map(corpus.map((d, i) => [d.text, idOf[i]]));

  const contextBoundResults: QueryResult[] = [];
  const contentOnlyResults: QueryResult[] = [];
  const ceilingResults: QueryResult[] = [];
  const hrrResults: QueryResult[] = [];

  for (const { q, tag } of queries) {
    // --- context-bound: the actual feature under test ---
    {
      const t = now();
      const r = store.retrieve(q.text, tag, { topK: TOP_K, globalResonance: false });
      const ms = now() - t;
      contextBoundResults.push({
        queryId: q.id,
        rankedIds: r.peaks.map((p) => textToId.get(p.record.text) ?? ""),
        ms,
      });
    }
    // --- content-only: same store, no context, for comparison ---
    {
      const t = now();
      const r = store.retrieve(q.text, "", { topK: TOP_K, globalResonance: false });
      const ms = now() - t;
      contentOnlyResults.push({
        queryId: q.id,
        rankedIds: r.peaks.map((p) => textToId.get(p.record.text) ?? ""),
        ms,
      });
    }
    // --- bucket ceiling: brute-force cosine restricted to the same tag ---
    let qv: Float64Array;
    {
      qv = textToSphereVector(q.text, undefined, idf);
      const t = now();
      const scored: Array<{ i: number; s: number }> = [];
      for (let i = 0; i < corpus.length; i++) {
        if (docContext[i] !== tag) continue;
        scored.push({ i, s: dot(qv, docVecs[i]) });
      }
      scored.sort((a, b) => b.s - a.s);
      const ms = now() - t;
      ceilingResults.push({
        queryId: q.id,
        rankedIds: scored.slice(0, TOP_K).map((x) => idOf[x.i]),
        ms,
      });
    }
    // --- HRR-bound: circular convolution at full SPHERE_DIM, same-tag
    // restricted (apples-to-apples against 'context-bound' and 'ceiling'
    // above — same bucketing, only the binding math differs) ---
    {
      const t = now();
      const qBound = hrrNormalize(circularConvolve(qv, tagVec(tag)));
      const scored: Array<{ i: number; s: number }> = [];
      for (let i = 0; i < corpus.length; i++) {
        if (docContext[i] !== tag) continue;
        scored.push({ i, s: hrrDot(qBound, docBoundHrr[i]) });
      }
      scored.sort((a, b) => b.s - a.s);
      const ms = now() - t;
      hrrResults.push({
        queryId: q.id,
        rankedIds: scored.slice(0, TOP_K).map((x) => idOf[x.i]),
        ms,
      });
    }
  }

  console.log(`\nResults (NFCorpus test split, synthetic topic-tagged context, top-${TOP_K}):\n`);
  printTable([
    { name: "Bucket ceiling (brute-force, same-tag docs only)", summary: summarize(ceilingResults, qrels) },
    { name: "Context-bound (HoloStore.retrieve — HRR scoring)", summary: summarize(contextBoundResults, qrels) },
    { name: "HRR-bound (prototype, same-tag not hash-bucket)", summary: summarize(hrrResults, qrels) },
    { name: "Content-only (same store, no context)", summary: summarize(contentOnlyResults, qrels) },
  ]);

  console.log(
    "\nNotes:\n" +
      "  - Context tags are synthesized by keyword match (see TOPIC_KEYWORDS) since\n" +
      "    NFCorpus has no real category field — this is a stand-in for \"the user tagged\n" +
      "    this note with a context\", not a claim about NFCorpus's true topic structure.\n" +
      "  - 'Bucket ceiling' is the best any method could do while restricted to the\n" +
      "    query's own tag — it's lower than bench/index.ts's untagged brute-force\n" +
      "    ceiling because most of a query's true relevant docs don't share its\n" +
      "    keyword tag (a qrel judgment isn't the same as \"same topic keyword\").\n" +
      `  - 'Context-bound' now scores via HRR circular convolution at full\n` +
      `    SPHERE_DIM=${SPHERE_DIM} (src/lib/holo/hrr.ts), not the Cl(${NUM_GENERATORS},0) grade-3 blade\n` +
      `    compression that used to dominate this gap — Clifford (gp/sandwich/timeRotor)\n` +
      `    is still used for the field/globalResonance stat, but no longer for peak\n` +
      `    scoring. See the root README's Design notes for the before/after and why.\n` +
      `  - 'Context-bound' vs 'bucket ceiling' isolates what's left after that fix:\n` +
      `    hash-bucket restriction plus whatever headroom HRR itself doesn't recover.\n` +
      `  - 'Context-bound' vs 'content-only' answers the actual product question: does\n` +
      `    scoping retrieval to a matching context help this engine, on this corpus?\n` +
      `    Still no as of this run — content-only remains ahead — but the gap is much\n` +
      `    smaller now than with the old blade-scored path. See the root README's\n` +
      `    context-bound-queries section for the numbers.\n` +
      `  - 'HRR-bound' is the standalone prototype (src/lib/holo/hrr.ts) this fix came\n` +
      `    from, restricted by same human-readable tag rather than hashed bucket —\n` +
      `    included alongside 'Context-bound' to show the live integration reproduces\n` +
      `    the prototype's result (it does, closely).`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
