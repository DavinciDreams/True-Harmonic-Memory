// Isolated temporal-address benchmark.
//
// Each semantic family has 72 byte-identical versions stored at consecutive
// logical ticks. Content similarity identifies the family but cannot identify
// the version; temporal addressing has to break that tie.

import { HoloStore, type TemporalAddressing } from "../src/lib/holo/engine";
import { distributedRotorCorrelation } from "../src/lib/holo/temporalRotor";

const VERSIONS = 72;
const FAMILIES = [
  "amber deployment ledger",
  "birch architecture decision",
  "cobalt incident summary",
  "dahlia research notebook",
  "ember release handoff",
  "fjord benchmark finding",
  "garnet product constraint",
  "hazel integration status",
];

interface Fixture {
  store: HoloStore;
  records: Array<{ text: string; tick: number }>;
  indexMs: number;
}

interface Summary {
  name: string;
  exactAt1: number;
  semanticFamilyAt1: number;
  mrr: number;
  meanAbsoluteTickError: number;
  nativePeriodAliasRate: number;
  avgQueryMs: number;
  p95QueryMs: number;
  indexMs: number;
}

function build(temporalAddressing: TemporalAddressing): Fixture {
  const store = new HoloStore({ temporalAddressing });
  const records: Array<{ text: string; tick: number }> = [];
  const started = performance.now();
  for (const text of FAMILIES) {
    for (let version = 0; version < VERSIONS; version++) {
      const record = store.addBulk(text);
      records.push({ text, tick: record.createdAtTick });
    }
  }
  return { store, records, indexMs: performance.now() - started };
}

function medianBuild(temporalAddressing: TemporalAddressing, repetitions = 3): Fixture {
  const fixtures = Array.from({ length: repetitions }, () => build(temporalAddressing));
  fixtures.sort((left, right) => left.indexMs - right.indexMs);
  return fixtures[Math.floor(fixtures.length / 2)];
}

function evaluate(
  name: string,
  fixture: Fixture,
  temporal: boolean,
  temporalAddressing: TemporalAddressing
): Summary {
  let exactAt1 = 0;
  let semanticFamilyAt1 = 0;
  let reciprocalRank = 0;
  let absoluteTickError = 0;
  let nativePeriodAliases = 0;
  const timings: number[] = [];

  for (const target of fixture.records) {
    const started = performance.now();
    const result = fixture.store.retrieve(target.text, "", {
      topK: VERSIONS,
      centerTick: target.tick,
      temporal,
      temporalAddressing,
      rotorWeight: 0.05,
      globalResonance: false,
    });
    timings.push(performance.now() - started);

    const first = result.peaks[0]?.record;
    if (!first) continue;
    if (first.text === target.text) semanticFamilyAt1++;
    const error = Math.abs(first.createdAtTick - target.tick);
    absoluteTickError += error;
    if (error === 0) exactAt1++;
    else if (error % 24 === 0) nativePeriodAliases++;

    const rank = result.peaks.findIndex((peak) => peak.record.createdAtTick === target.tick);
    if (rank >= 0) reciprocalRank += 1 / (rank + 1);
  }

  timings.sort((a, b) => a - b);
  const n = fixture.records.length;
  return {
    name,
    exactAt1: exactAt1 / n,
    semanticFamilyAt1: semanticFamilyAt1 / n,
    mrr: reciprocalRank / n,
    meanAbsoluteTickError: absoluteTickError / n,
    nativePeriodAliasRate: nativePeriodAliases / n,
    avgQueryMs: timings.reduce((sum, value) => sum + value, 0) / timings.length,
    p95QueryMs: timings[Math.min(timings.length - 1, Math.floor(timings.length * 0.95))],
    indexMs: fixture.indexMs,
  };
}

function percent(value: number): string {
  return `${(100 * value).toFixed(1)}%`;
}

function milliseconds(value: number): string {
  return value < 1 ? `${Math.round(value * 1000)}µs` : `${value.toFixed(2)}ms`;
}

function print(rows: Summary[]) {
  const header = [
    "Method", "Exact@1", "Family@1", "MRR", "Mean |tick error|",
    "24-tick alias", "Index", "Avg query", "p95 query",
  ];
  const body = rows.map((row) => [
    row.name,
    percent(row.exactAt1),
    percent(row.semanticFamilyAt1),
    row.mrr.toFixed(4),
    row.meanAbsoluteTickError.toFixed(2),
    percent(row.nativePeriodAliasRate),
    milliseconds(row.indexMs),
    milliseconds(row.avgQueryMs),
    milliseconds(row.p95QueryMs),
  ]);
  const widths = header.map((label, index) =>
    Math.max(label.length, ...body.map((row) => row[index].length))
  );
  const line = (values: string[]) =>
    "  " + values.map((value, index) => value.padEnd(widths[index])).join("  |  ");
  console.log(line(header));
  console.log(line(widths.map((width) => "-".repeat(width))));
  for (const row of body) console.log(line(row));
}

function collisionScan(limit: number) {
  let worst = { delta: 0, coherence: 0 };
  let nearCollisions = 0;
  for (let delta = 1; delta <= limit; delta++) {
    const coherence = distributedRotorCorrelation(0, delta).coherence;
    if (coherence > worst.coherence) worst = { delta, coherence };
    if (coherence >= 0.99) nearCollisions++;
  }
  return { limit, worst, nearCollisions };
}

const single = medianBuild("single");
const distributed = medianBuild("distributed");
const results = [
  evaluate("Content only", single, false, "single"),
  evaluate("Native single-plane", single, true, "single"),
  evaluate("Distributed score only", single, true, "distributed"),
  evaluate("Distributed geometric field", distributed, true, "distributed"),
];

console.log(
  `Temporal tie-break benchmark (${FAMILIES.length} semantic families × ${VERSIONS} versions, ` +
  `${FAMILIES.length * VERSIONS} memories and queries)\n`
);
print(results);

const scan = collisionScan(100_000);
console.log(
  `\nDistributed alias scan: ${scan.nearCollisions} coherence >= 0.99 in ` +
  `1..${scan.limit.toLocaleString()} ticks; worst was ${scan.worst.coherence.toFixed(6)} ` +
  `at Δ=${scan.worst.delta.toLocaleString()} ticks.`
);
console.log(
  "\nInterpretation: exact ticks are the answer key. The rotor is useful only when it " +
  "breaks a semantic tie without changing the winning semantic family."
);
