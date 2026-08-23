/** Compare the direct wire/rotor field and HoloStore on the same BEIR corpus. */

import { performance } from "node:perf_hooks";

import { HoloStore } from "../src/lib/holo/engine";
import {
  DirectRotorWireField,
  HarmonicShardRouter,
  LocalizedSpectralRouterKind,
  LocalizedSpectralShardRouter,
  PreparedWireProbe,
  symbolsPerByte,
  wireFrameSymbolLength,
} from "../src/lib/holo/trueHarmonic";
import { buildIdf } from "../src/lib/holo/sphere";
import { assertDataPresent, Doc, loadCorpus, loadQrels, loadQueries, Query } from "./loadNfcorpus";
import { QueryResult, summarize } from "./metrics";

const TOP_K = 100;
const textEncoder = new TextEncoder();

interface DirectRecord {
  id: string;
  bytes: Uint8Array;
  packetOffset: number;
  payloadStart: number;
  payloadSymbols: number;
}

interface DirectShard {
  field: DirectRotorWireField;
  records: DirectRecord[];
  nextOffset: number;
}

function intFlag(name: string, fallback: number): number {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Number(process.argv[index + 1]);
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  return value;
}

function stringFlag(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
}

function selectQueries(queries: Query[], count: number): Query[] {
  if (count >= queries.length) return queries;
  if (count <= 1) return queries.slice(0, count);
  return Array.from({ length: count }, (_, index) =>
    queries[Math.round(index * (queries.length - 1) / (count - 1))]
  );
}

function buildDirectIndex(
  corpus: Doc[],
  options: { fec: boolean; shardSamples: number; maxPayloadBytes: number },
): { shards: DirectShard[]; indexMs: number; totalSymbols: number } {
  const { fec, shardSamples, maxPayloadBytes } = options;
  const shards: DirectShard[] = [];
  const makeShard = (): DirectShard => {
    const shard = {
      field: new DirectRotorWireField({ horizonSamples: shardSamples, maxPayloadBytes, fec }),
      records: [],
      nextOffset: 0,
    };
    shards.push(shard);
    return shard;
  };
  let totalSymbols = 0;
  const started = performance.now();
  let shard = makeShard();
  for (const document of corpus) {
    const bytes = textEncoder.encode(document.text);
    const frameSymbols = wireFrameSymbolLength(bytes.length, fec);
    if (shard.nextOffset + frameSymbols > shardSamples) shard = makeShard();
    const packetOffset = shard.nextOffset;
    shard.field.add(bytes, packetOffset);
    const payloadStart = packetOffset + shard.field.headerSymbols;
    const payloadSymbols = bytes.length * symbolsPerByte(fec);
    shard.records.push({ id: document.id, bytes, packetOffset, payloadStart, payloadSymbols });
    shard.nextOffset += frameSymbols;
    totalSymbols += frameSymbols;
  }
  return { shards, indexMs: performance.now() - started, totalSymbols };
}

function prepareDirectIndex(shards: DirectShard[]): number {
  const started = performance.now();
  for (const shard of shards) shard.field.prepareSpectrum();
  return performance.now() - started;
}

function scoreShard(
  shard: DirectShard,
  probe: PreparedWireProbe,
  stride: number,
): Array<{ id: string; score: number }> {
  const correlation = shard.field.correlationPrepared(probe);
  return shard.records.map((record) => {
    const lastStart = record.payloadStart + record.payloadSymbols - probe.symbolCount;
    let score = 0;
    if (lastStart >= record.payloadStart) {
      for (let offset = record.payloadStart; offset <= lastStart; offset += stride) {
        score = Math.max(score, Math.hypot(correlation[2 * offset], correlation[2 * offset + 1]));
      }
    }
    return { id: record.id, score };
  });
}

function directQuery(shards: DirectShard[], text: string, fec: boolean): QueryResult {
  const bytes = textEncoder.encode(text);
  const started = performance.now();
  const probe = shards[0].field.prepareProbe(bytes);
  const scored = shards.flatMap((shard) => scoreShard(shard, probe, symbolsPerByte(fec)));
  scored.sort((left, right) => right.score - left.score);
  return {
    queryId: "",
    rankedIds: scored.slice(0, TOP_K).map((row) => row.id),
    ms: performance.now() - started,
  };
}

function buildRouter(shards: DirectShard[]): { router: HarmonicShardRouter; buildMs: number } {
  const size = 2 ** Math.ceil(Math.log2(shards.length));
  const router = new HarmonicShardRouter(size);
  const started = performance.now();
  shards.forEach((shard, address) => router.add(shard.field.spectrumSketch(size), address));
  return { router, buildMs: performance.now() - started };
}

function buildLocalizedRouter(
  shards: DirectShard[],
  kind: LocalizedSpectralRouterKind,
  projectionSize: number,
): { router: LocalizedSpectralShardRouter; buildMs: number } {
  const router = new LocalizedSpectralShardRouter({
    kind,
    horizonSamples: shards[0].field.horizonSamples,
    projectionSize,
  });
  const started = performance.now();
  shards.forEach((shard, address) => router.add(shard.field, address));
  return { router, buildMs: performance.now() - started };
}

function routedDirectQuery(
  shards: DirectShard[],
  router: HarmonicShardRouter,
  text: string,
  fec: boolean,
  routeShards: number,
): QueryResult & { routedShards: number[] } {
  const bytes = textEncoder.encode(text);
  const started = performance.now();
  const probe = shards[0].field.prepareProbe(bytes);
  const sketch = shards[0].field.probeSketch(probe, router.size);
  const routedShards = router.route(sketch, routeShards).map((hit) => hit.timeTick);
  const scored = routedShards.flatMap((address) =>
    scoreShard(shards[address], probe, symbolsPerByte(fec))
  );
  scored.sort((left, right) => right.score - left.score);
  return {
    queryId: "",
    rankedIds: scored.slice(0, TOP_K).map((row) => row.id),
    ms: performance.now() - started,
    routedShards,
  };
}

interface RoutedQueryResult extends QueryResult {
  routedShards: number[];
  routeMs: number;
}

function localizedRoutedDirectQuery(
  shards: DirectShard[],
  router: LocalizedSpectralShardRouter,
  text: string,
  fec: boolean,
  routeShards: number,
): RoutedQueryResult {
  const bytes = textEncoder.encode(text);
  const started = performance.now();
  const probe = shards[0].field.prepareProbe(bytes);
  const routeStarted = performance.now();
  const routedShards = router.route(shards[0].field, probe, routeShards).map((hit) => hit.timeTick);
  const routeMs = performance.now() - routeStarted;
  const scored = routedShards.flatMap((address) =>
    scoreShard(shards[address], probe, symbolsPerByte(fec))
  );
  scored.sort((left, right) => right.score - left.score);
  return {
    queryId: "",
    rankedIds: scored.slice(0, TOP_K).map((row) => row.id),
    ms: performance.now() - started,
    routedShards,
    routeMs,
  };
}

function hybridRoutedDirectQuery(
  shards: DirectShard[],
  localRouter: LocalizedSpectralShardRouter,
  diffuseRouter: LocalizedSpectralShardRouter,
  text: string,
  fec: boolean,
  routeShards: number,
  localQuota: number,
): RoutedQueryResult {
  const bytes = textEncoder.encode(text);
  const started = performance.now();
  const probe = shards[0].field.prepareProbe(bytes);
  const routeStarted = performance.now();
  const local = localRouter.route(shards[0].field, probe, Math.min(localQuota, routeShards));
  const diffuse = diffuseRouter.route(shards[0].field, probe, shards.length);
  const routedShards = local.map((hit) => hit.timeTick);
  for (const hit of diffuse) {
    if (routedShards.length >= routeShards) break;
    if (!routedShards.includes(hit.timeTick)) routedShards.push(hit.timeTick);
  }
  const routeMs = performance.now() - routeStarted;
  const scored = routedShards.flatMap((address) =>
    scoreShard(shards[address], probe, symbolsPerByte(fec))
  );
  scored.sort((left, right) => right.score - left.score);
  return {
    queryId: "",
    rankedIds: scored.slice(0, TOP_K).map((row) => row.id),
    ms: performance.now() - started,
    routedShards,
    routeMs,
  };
}

function routingSummary(
  results: RoutedQueryResult[],
  qrels: Map<string, Map<string, number>>,
  documentShards: Map<string, number>,
): { anyRelevantShardRate: number; meanRelevantDocumentCoverage: number; avgRouteMs: number } {
  let anyRelevant = 0;
  let coverage = 0;
  let evaluated = 0;
  for (const result of results) {
    const relevance = qrels.get(result.queryId);
    if (!relevance) continue;
    const relevantDocuments = [...relevance.entries()].filter(([, score]) => score > 0);
    if (!relevantDocuments.length) continue;
    const routed = new Set(result.routedShards);
    const covered = relevantDocuments.filter(([id]) => {
      const shard = documentShards.get(id);
      return shard !== undefined && routed.has(shard);
    }).length;
    anyRelevant += Number(covered > 0);
    coverage += covered / relevantDocuments.length;
    evaluated++;
  }
  return {
    anyRelevantShardRate: evaluated ? anyRelevant / evaluated : 0,
    meanRelevantDocumentCoverage: evaluated ? coverage / evaluated : 0,
    avgRouteMs: results.length
      ? results.reduce((total, result) => total + result.routeMs, 0) / results.length
      : 0,
  };
}

function exactReadCheck(shards: DirectShard[]): { samples: number; crcRecall: number; totalMs: number } {
  const records = shards.flatMap((shard, shardIndex) =>
    shard.records.map((record) => ({ shardIndex, record }))
  );
  const sampleCount = Math.min(32, records.length);
  const samples = sampleCount === 1
    ? records.slice(0, 1)
    : Array.from({ length: sampleCount }, (_, index) =>
      records[Math.round(index * (records.length - 1) / (sampleCount - 1))]
    );
  let correct = 0;
  const started = performance.now();
  for (const sample of samples) {
    const decoded = shards[sample.shardIndex].field.readPacket(sample.record.packetOffset);
    if (decoded.crcOk && Buffer.from(decoded.payload).equals(Buffer.from(sample.record.bytes))) correct++;
  }
  return { samples: samples.length, crcRecall: correct / samples.length, totalMs: performance.now() - started };
}

function holoBenchmark(corpus: Doc[], queries: Query[]): { indexMs: number; results: QueryResult[] } {
  const idf = buildIdf(corpus.map((document) => document.text));
  const store = new HoloStore({ idf });
  const textToId = new Map(corpus.map((document) => [document.text, document.id]));
  const started = performance.now();
  for (const document of corpus) store.addBulk(document.text);
  const indexMs = performance.now() - started;
  const results = queries.map((query) => {
    const queryStarted = performance.now();
    const retrieved = store.retrieve(query.text, "", { topK: TOP_K, globalResonance: false });
    return {
      queryId: query.id,
      rankedIds: retrieved.peaks.map((peak) => textToId.get(peak.record.text) ?? ""),
      ms: performance.now() - queryStarted,
    };
  });
  return { indexMs, results };
}

async function main() {
  const dataset = stringFlag("--dataset", "nfcorpus");
  const queryCount = intFlag("--queries", 5);
  const shardPower = intFlag("--shard-power", 18);
  const routeShards = intFlag("--route-shards", 8);
  const routerPower = intFlag("--router-power", 10);
  const routerKinds = stringFlag(
    "--router-kinds",
    "sparse-fourier,needlet,slepian,diffusion,gabor",
  ).split(",") as LocalizedSpectralRouterKind[];
  const fec = process.argv.includes("--fec");
  const globalField = process.argv.includes("--global");
  assertDataPresent(dataset);
  const [corpus, allQueries, qrels] = await Promise.all([
    loadCorpus(dataset),
    loadQueries(dataset),
    loadQrels(dataset, "test"),
  ]);
  const judged = allQueries.filter((query) => qrels.has(query.id));
  const queries = selectQueries(judged, queryCount);
  const documentByteArrays = corpus.map((document) => textEncoder.encode(document.text));
  const maxPayloadBytes = Math.max(...documentByteArrays.map((bytes) => bytes.length));
  const occupiedSymbols = documentByteArrays.reduce(
    (total, bytes) => total + wireFrameSymbolLength(bytes.length, fec),
    0,
  );
  const shardSamples = globalField
    ? 2 ** Math.ceil(Math.log2(occupiedSymbols))
    : 2 ** shardPower;
  if (wireFrameSymbolLength(maxPayloadBytes, fec) > shardSamples) {
    throw new Error("largest document does not fit; increase --shard-power");
  }

  const direct = buildDirectIndex(corpus, { fec, shardSamples, maxPayloadBytes });
  const spectrumMs = prepareDirectIndex(direct.shards);
  const harmonicRouter = buildRouter(direct.shards);
  const localizedRouters = routerKinds.map((kind) => ({
    kind,
    ...buildLocalizedRouter(direct.shards, kind, 2 ** routerPower),
  }));
  const directResults = queries.map((query) => ({
    ...directQuery(direct.shards, query.text, fec),
    queryId: query.id,
  }));
  const routedResults = queries.map((query) => ({
    ...routedDirectQuery(direct.shards, harmonicRouter.router, query.text, fec, routeShards),
    queryId: query.id,
  }));
  const exactRead = exactReadCheck(direct.shards);
  const prefixRecord = direct.shards[0].records[0];
  const prefixText = new TextDecoder().decode(prefixRecord.bytes.slice(0, Math.min(64, prefixRecord.bytes.length)));
  const prefixResult = directQuery(direct.shards, prefixText, fec);
  const prefixRank = prefixResult.rankedIds.indexOf(prefixRecord.id) + 1;
  const routedPrefix = routedDirectQuery(
    direct.shards,
    harmonicRouter.router,
    prefixText,
    fec,
    routeShards,
  );
  const routedPrefixRank = routedPrefix.rankedIds.indexOf(prefixRecord.id) + 1;

  const documentShards = new Map<string, number>();
  direct.shards.forEach((shard, shardIndex) => {
    for (const record of shard.records) documentShards.set(record.id, shardIndex);
  });
  const localizedResults = localizedRouters.map(({ kind, router, buildMs }) => {
    const results = queries.map((query) => ({
      ...localizedRoutedDirectQuery(direct.shards, router, query.text, fec, routeShards),
      queryId: query.id,
    }));
    const prefix = localizedRoutedDirectQuery(
      direct.shards,
      router,
      prefixText,
      fec,
      routeShards,
    );
    return {
      kind,
      projectionSize: router.projectionSize,
      buildMs,
      storageMiB: router.storageBytes / 2 ** 20,
      prefixProbe: {
        sourceShardIncluded: prefix.routedShards.includes(0),
        rank: prefix.rankedIds.indexOf(prefixRecord.id) + 1,
        routeMs: prefix.routeMs,
        queryMs: prefix.ms,
      },
      routing: routingSummary(results, qrels, documentShards),
      retrieval: summarize(results, qrels),
    };
  });
  const needletRouter = localizedRouters.find(({ kind }) => kind === "needlet")?.router;
  const diffusionRouter = localizedRouters.find(({ kind }) => kind === "diffusion")?.router;
  const hybridLocalQuota = Math.min(8, Math.max(1, Math.floor(routeShards / 4)));
  const hybridResult = needletRouter && diffusionRouter ? (() => {
    const results = queries.map((query) => ({
      ...hybridRoutedDirectQuery(
        direct.shards,
        needletRouter,
        diffusionRouter,
        query.text,
        fec,
        routeShards,
        hybridLocalQuota,
      ),
      queryId: query.id,
    }));
    const prefix = hybridRoutedDirectQuery(
      direct.shards,
      needletRouter,
      diffusionRouter,
      prefixText,
      fec,
      routeShards,
      hybridLocalQuota,
    );
    return {
      kind: "needlet+diffusion",
      localQuota: hybridLocalQuota,
      candidateShards: routeShards,
      prefixProbe: {
        sourceShardIncluded: prefix.routedShards.includes(0),
        rank: prefix.rankedIds.indexOf(prefixRecord.id) + 1,
        routeMs: prefix.routeMs,
        queryMs: prefix.ms,
      },
      routing: routingSummary(results, qrels, documentShards),
      retrieval: summarize(results, qrels),
    };
  })() : null;

  const holo = holoBenchmark(corpus, queries);
  const directStorageBytes = direct.shards.reduce((total, shard) => total + shard.field.storageBytes, 0);
  console.log(JSON.stringify({
    dataset,
    documents: corpus.length,
    documentBytes: corpus.reduce((total, document) => total + textEncoder.encode(document.text).length, 0),
    queries: queries.length,
    directRotor: {
      fec,
      layout: globalField ? "one_global_harmonic_superposition" : "sharded_rotor_fields",
      shardSamples,
      shards: direct.shards.length,
      occupiedSymbols: direct.totalSymbols,
      indexMs: direct.indexMs,
      spectrumPreparationMs: spectrumMs,
      harmonicRouterBuildMs: harmonicRouter.buildMs,
      queryReadyMs: direct.indexMs + spectrumMs,
      queryReadyWithRouterMs: direct.indexMs + spectrumMs + harmonicRouter.buildMs,
      storageMiB: directStorageBytes / 2 ** 20,
      exactRead,
      prefixProbe: {
        bytes: textEncoder.encode(prefixText).length,
        exhaustive: { rank: prefixRank, queryMs: prefixResult.ms },
        routed: {
          candidateShards: routeShards,
          sourceShardIncluded: routedPrefix.routedShards.includes(0),
          rank: routedPrefixRank,
          queryMs: routedPrefix.ms,
        },
      },
      naturalQueries: summarize(directResults, qrels),
      routedNaturalQueries: {
        candidateShards: routeShards,
        ...summarize(routedResults, qrels),
      },
      localizedRouters: localizedResults,
      localizedHybrid: hybridResult,
    },
    holoStore: {
      indexMs: holo.indexMs,
      naturalQueries: summarize(holo.results, qrels),
    },
    boundary: "Wire correlation measures literal byte structure; HoloStore measures semantic retrieval under its hash-sphere encoder.",
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
