/** Capability benchmark: phase-biased retrieval versus rotor temporal reasoning. */

import { performance } from "node:perf_hooks";

import { HoloStore } from "../src/lib/holo/engine";
import {
  DirectRotorWireField,
  LocalizedSpectralShardRouter,
  symbolsPerByte,
  wireFrameSymbolLength,
} from "../src/lib/holo/trueHarmonic";

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const TICKS = 96;
const SLOT_SAMPLES = 512;
const HORIZON_SAMPLES = 65_536;
const EPOCH_TICKS = 24;
const EPOCH_HORIZON_SAMPLES = 16_384;
const routedEpochFlag = process.argv.indexOf("--routed-epochs");
const ROUTED_EPOCHS = routedEpochFlag < 0 ? 2 : Number(process.argv[routedEpochFlag + 1]);
if (!Number.isInteger(ROUTED_EPOCHS) || ROUTED_EPOCHS < 1 || ROUTED_EPOCHS > TICKS / EPOCH_TICKS) {
  throw new Error("--routed-epochs must be an integer in [1, 4]");
}
// Exact/prefix QPSK matches score ~1.0 in this fixture. The first run used
// 0.75 and admitted one unrelated short-cue collision at 0.7526, so the
// benchmark makes the evidence boundary explicit instead of treating every
// correlation peak as an occurrence.
const MATCH_THRESHOLD = 0.9;

interface EventRecord {
  id: string;
  tick: number;
  text: string;
  packetOffset: number;
  payloadStart: number;
  payloadSymbols: number;
}

interface EvidencePoint {
  id: string;
  tick: number;
  score: number;
}

interface EpochShard {
  field: DirectRotorWireField;
  records: EventRecord[];
}

interface CapabilityResult {
  name: string;
  expected: unknown;
  rotor: unknown;
  rotorPass: boolean;
  holo: unknown;
  holoPass: boolean | null;
  holoSupport: "native" | "unsupported";
  evidence: EvidencePoint[];
  ms: number;
}

const scheduled = new Map<number, string>([
  [5, "kick model selected baseline"],
  [11, "checkpoint heartbeat"],
  [17, "qk fold experiment started"],
  [29, "qk fold experiment completed"],
  [35, "checkpoint heartbeat"],
  [40, "lohe delta compression proposed"],
  [46, "compression candidate selected"],
  [52, "lohe delta compression benchmarked"],
  [59, "checkpoint heartbeat"],
  [65, "deployment state staging"],
  [71, "deployment state production"],
  [77, "deployment state rollback"],
  [83, "checkpoint heartbeat"],
  [89, "gold star snapshot recorded"],
]);

function textAt(tick: number): string {
  return scheduled.get(tick) ?? `unrelated distractor event tick ${tick}`;
}

function evidenceFor(
  field: DirectRotorWireField,
  records: EventRecord[],
  cue: string,
): EvidencePoint[] {
  const probe = field.prepareProbe(encoder.encode(cue));
  const correlation = field.correlationPrepared(probe);
  return records.map((record) => {
    const lastStart = record.payloadStart + record.payloadSymbols - probe.symbolCount;
    let score = 0;
    for (let offset = record.payloadStart; offset <= lastStart; offset++) {
      score = Math.max(score, Math.hypot(correlation[2 * offset], correlation[2 * offset + 1]));
    }
    return { id: record.id, tick: record.tick, score };
  }).sort((left, right) => right.score - left.score);
}

function occurrences(evidence: EvidencePoint[]): EvidencePoint[] {
  return evidence
    .filter((point) => point.score >= MATCH_THRESHOLD)
    .sort((left, right) => left.tick - right.tick);
}

function pointerText(field: DirectRotorWireField, records: EventRecord[], tick: number): string {
  const record = records.find((candidate) => candidate.tick === tick);
  if (!record) throw new Error(`no event at tick ${tick}`);
  const packet = field.readPacket(record.packetOffset);
  if (!packet.crcOk) throw new Error(`CRC failed at tick ${tick}`);
  return decoder.decode(packet.payload);
}

function orderedPair(
  left: EvidencePoint[],
  right: EvidencePoint[],
): { leftTick: number; rightTick: number; lag: number } | null {
  let best: { leftTick: number; rightTick: number; lag: number; score: number } | null = null;
  for (const before of occurrences(left)) {
    for (const after of occurrences(right)) {
      if (before.tick >= after.tick) continue;
      const score = before.score * after.score;
      if (!best || score > best.score) {
        best = { leftTick: before.tick, rightTick: after.tick, lag: after.tick - before.tick, score };
      }
    }
  }
  return best && { leftTick: best.leftTick, rightTick: best.rightTick, lag: best.lag };
}

function unsupported(): { value: string; pass: null; support: "unsupported" } {
  return { value: "unsupported", pass: null, support: "unsupported" };
}

function main() {
  const allTexts = Array.from({ length: TICKS }, (_, tick) => textAt(tick));
  const maxPayloadBytes = Math.max(...allTexts.map((text) => encoder.encode(text).length));
  if (wireFrameSymbolLength(maxPayloadBytes, false) > SLOT_SAMPLES) {
    throw new Error("slot is too small for the longest temporal event");
  }

  const field = new DirectRotorWireField({
    horizonSamples: HORIZON_SAMPLES,
    maxPayloadBytes,
    fec: false,
  });
  const epochShards: EpochShard[] = Array.from(
    { length: Math.ceil(TICKS / EPOCH_TICKS) },
    () => ({
      field: new DirectRotorWireField({
        horizonSamples: EPOCH_HORIZON_SAMPLES,
        maxPayloadBytes,
        fec: false,
      }),
      records: [],
    }),
  );
  const holo = new HoloStore();
  const records: EventRecord[] = [];
  for (let tick = 0; tick < TICKS; tick++) {
    const text = allTexts[tick];
    const bytes = encoder.encode(text);
    const packetOffset = tick * SLOT_SAMPLES;
    field.add(bytes, packetOffset);
    records.push({
      id: `event-${tick}`,
      tick,
      text,
      packetOffset,
      payloadStart: packetOffset + field.headerSymbols,
      payloadSymbols: bytes.length * symbolsPerByte(false),
    });
    const epoch = Math.floor(tick / EPOCH_TICKS);
    const epochOffset = (tick % EPOCH_TICKS) * SLOT_SAMPLES;
    epochShards[epoch].field.add(bytes, epochOffset);
    epochShards[epoch].records.push({
      id: `event-${tick}`,
      tick,
      text,
      packetOffset: epochOffset,
      payloadStart: epochOffset + field.headerSymbols,
      payloadSymbols: bytes.length * symbolsPerByte(false),
    });
    const holoRecord = holo.addBulk(text);
    if (holoRecord.createdAtTick !== tick) throw new Error("HoloStore clock diverged from fixture");
  }
  const preparationStarted = performance.now();
  field.prepareSpectrum();
  for (const shard of epochShards) shard.field.prepareSpectrum();
  const preparationMs = performance.now() - preparationStarted;

  const needletRouter = new LocalizedSpectralShardRouter({
    kind: "needlet",
    horizonSamples: EPOCH_HORIZON_SAMPLES,
    projectionSize: 256,
  });
  const diffusionRouter = new LocalizedSpectralShardRouter({
    kind: "diffusion",
    horizonSamples: EPOCH_HORIZON_SAMPLES,
    projectionSize: 256,
  });
  epochShards.forEach((shard, address) => {
    needletRouter.add(shard.field, address);
    diffusionRouter.add(shard.field, address);
  });

  const evidenceCache = new Map<string, EvidencePoint[]>();
  const evidenceLatencies: number[] = [];
  const evidence = (cue: string): EvidencePoint[] => {
    const cached = evidenceCache.get(cue);
    if (cached) return cached;
    const started = performance.now();
    const found = evidenceFor(field, records, cue);
    evidenceLatencies.push(performance.now() - started);
    evidenceCache.set(cue, found);
    return found;
  };
  const routedEvidenceLatencies: number[] = [];
  const routedEpochSelections = new Map<string, number[]>();
  const routedEvidenceCache = new Map<string, EvidencePoint[]>();
  const routedEvidence = (cue: string, forcedEpoch?: number): EvidencePoint[] => {
    const cacheKey = `${cue}|${forcedEpoch ?? "route"}`;
    const cached = routedEvidenceCache.get(cacheKey);
    if (cached) return cached;
    const started = performance.now();
    const probe = epochShards[0].field.prepareProbe(encoder.encode(cue));
    let selected: number[];
    if (forcedEpoch !== undefined) {
      selected = [forcedEpoch];
    } else {
      const local = needletRouter.route(epochShards[0].field, probe, 1).map((hit) => hit.timeTick);
      const diffuse = diffusionRouter.route(
        epochShards[0].field,
        probe,
        epochShards.length,
      ).map((hit) => hit.timeTick);
      selected = local;
      for (const epoch of diffuse) {
        if (selected.length >= ROUTED_EPOCHS) break;
        if (!selected.includes(epoch)) selected.push(epoch);
      }
    }
    const found = selected.flatMap((epoch) =>
      evidenceFor(epochShards[epoch].field, epochShards[epoch].records, cue)
    ).sort((left, right) => right.score - left.score);
    routedEvidenceLatencies.push(performance.now() - started);
    routedEpochSelections.set(cacheKey, selected);
    routedEvidenceCache.set(cacheKey, found);
    return found;
  };

  const capabilities: CapabilityResult[] = [];
  const addCapability = (
    name: string,
    expected: unknown,
    cueEvidence: EvidencePoint[],
    rotorValue: () => unknown,
    holoResult: { value: unknown; pass: boolean | null; support: "native" | "unsupported" },
  ) => {
    const started = performance.now();
    const rotor = rotorValue();
    capabilities.push({
      name,
      expected,
      rotor,
      rotorPass: JSON.stringify(rotor) === JSON.stringify(expected),
      holo: holoResult.value,
      holoPass: holoResult.pass,
      holoSupport: holoResult.support,
      evidence: occurrences(cueEvidence),
      ms: performance.now() - started,
    });
  };

  const deploymentEvidence = evidence("deployment state");
  const nearHolo = holo.retrieve("deployment state", "", {
    temporal: true,
    centerTick: 71,
    sigma: 0.35,
    topK: 1,
    globalResonance: false,
  }).peaks[0]?.record.createdAtTick ?? null;
  addCapability(
    "near tick",
    71,
    deploymentEvidence,
    () => occurrences(deploymentEvidence)
      .sort((left, right) => Math.abs(left.tick - 71) - Math.abs(right.tick - 71))[0]?.tick ?? null,
    { value: nearHolo, pass: nearHolo === 71, support: "native" },
  );

  const heartbeatEvidence = evidence("checkpoint heartbeat");
  const aliasHolo = holo.retrieve("checkpoint heartbeat", "", {
    temporal: true,
    centerTick: 83,
    sigma: 0.2,
    topK: 1,
    globalResonance: false,
  }).peaks[0]?.record.createdAtTick ?? null;
  addCapability(
    "distinguish epochs with identical rotor phase",
    83,
    heartbeatEvidence,
    () => occurrences(heartbeatEvidence)
      .sort((left, right) => Math.abs(left.tick - 83) - Math.abs(right.tick - 83))[0]?.tick ?? null,
    { value: aliasHolo, pass: aliasHolo === 83, support: "native" },
  );

  const startEvidence = evidence("qk fold experiment started");
  const completeEvidence = evidence("qk fold experiment completed");
  const qkPair = () => orderedPair(startEvidence, completeEvidence);
  addCapability("before", true, startEvidence, () => qkPair() !== null, unsupported());
  addCapability("elapsed ticks", 12, completeEvidence, () => qkPair()?.lag ?? null, unsupported());

  const proposedEvidence = evidence("lohe delta compression proposed");
  const benchmarkedEvidence = evidence("lohe delta compression benchmarked");
  const candidateEvidence = evidence("compression candidate selected");
  const lohePair = orderedPair(proposedEvidence, benchmarkedEvidence);
  addCapability(
    "event between two anchors",
    46,
    candidateEvidence,
    () => occurrences(candidateEvidence).find((point) =>
      !!lohePair && point.tick > lohePair.leftTick && point.tick < lohePair.rightTick
    )?.tick ?? null,
    unsupported(),
  );

  addCapability(
    "ordered sequence",
    [65, 71, 77],
    deploymentEvidence,
    () => occurrences(deploymentEvidence).map((point) => point.tick),
    unsupported(),
  );
  addCapability(
    "recurrence",
    { ticks: [11, 35, 59, 83], gaps: [24, 24, 24] },
    heartbeatEvidence,
    () => {
      const ticks = occurrences(heartbeatEvidence).map((point) => point.tick);
      return { ticks, gaps: ticks.slice(1).map((tick, index) => tick - ticks[index]) };
    },
    unsupported(),
  );
  addCapability(
    "state at tick",
    "deployment state production",
    deploymentEvidence,
    () => {
      const state = occurrences(deploymentEvidence)
        .filter((point) => point.tick <= 74)
        .sort((left, right) => right.tick - left.tick)[0];
      return state ? pointerText(field, records, state.tick) : null;
    },
    unsupported(),
  );
  addCapability(
    "latest declared superseding state",
    "deployment state rollback",
    deploymentEvidence,
    () => {
      const state = occurrences(deploymentEvidence).at(-1);
      return state ? pointerText(field, records, state.tick) : null;
    },
    unsupported(),
  );

  const crcSamples = [5, 17, 29, 40, 52, 65, 71, 77, 83, 89];
  const crcValid = crcSamples.filter((tick) =>
    field.readPacket(tick * SLOT_SAMPLES).crcOk
  ).length;
  const nativeHolo = capabilities.filter((result) => result.holoSupport === "native");
  const sortedEvidenceLatencies = evidenceLatencies.slice().sort((left, right) => left - right);
  const routedDeploymentNear = routedEvidence("deployment state", Math.floor(71 / EPOCH_TICKS));
  const routedHeartbeatNear = routedEvidence("checkpoint heartbeat", Math.floor(83 / EPOCH_TICKS));
  const routedStart = routedEvidence("qk fold experiment started");
  const routedComplete = routedEvidence("qk fold experiment completed");
  const routedQkPair = orderedPair(routedStart, routedComplete);
  const routedProposed = routedEvidence("lohe delta compression proposed");
  const routedBenchmarked = routedEvidence("lohe delta compression benchmarked");
  const routedCandidate = routedEvidence("compression candidate selected");
  const routedLohePair = orderedPair(routedProposed, routedBenchmarked);
  const routedDeployment = routedEvidence("deployment state");
  const routedHeartbeat = routedEvidence("checkpoint heartbeat");
  const routedChecks = [
    {
      name: "near tick",
      expected: 71,
      actual: occurrences(routedDeploymentNear)
        .sort((left, right) => Math.abs(left.tick - 71) - Math.abs(right.tick - 71))[0]?.tick ?? null,
      selectedEpochs: routedEpochSelections.get("deployment state|2") ?? [],
    },
    {
      name: "distinguish epochs with identical rotor phase",
      expected: 83,
      actual: occurrences(routedHeartbeatNear)
        .sort((left, right) => Math.abs(left.tick - 83) - Math.abs(right.tick - 83))[0]?.tick ?? null,
      selectedEpochs: routedEpochSelections.get("checkpoint heartbeat|3") ?? [],
    },
    {
      name: "before",
      expected: true,
      actual: routedQkPair !== null,
      selectedEpochs: [
        ...(routedEpochSelections.get("qk fold experiment started|route") ?? []),
        ...(routedEpochSelections.get("qk fold experiment completed|route") ?? []),
      ],
    },
    {
      name: "elapsed ticks",
      expected: 12,
      actual: routedQkPair?.lag ?? null,
      selectedEpochs: [],
    },
    {
      name: "event between two anchors",
      expected: 46,
      actual: occurrences(routedCandidate).find((point) =>
        !!routedLohePair && point.tick > routedLohePair.leftTick && point.tick < routedLohePair.rightTick
      )?.tick ?? null,
      selectedEpochs: [
        ...(routedEpochSelections.get("lohe delta compression proposed|route") ?? []),
        ...(routedEpochSelections.get("lohe delta compression benchmarked|route") ?? []),
        ...(routedEpochSelections.get("compression candidate selected|route") ?? []),
      ],
    },
    {
      name: "ordered sequence",
      expected: [65, 71, 77],
      actual: occurrences(routedDeployment).map((point) => point.tick),
      selectedEpochs: routedEpochSelections.get("deployment state|route") ?? [],
    },
    {
      name: "recurrence",
      expected: { ticks: [11, 35, 59, 83], gaps: [24, 24, 24] },
      actual: (() => {
        const ticks = occurrences(routedHeartbeat).map((point) => point.tick);
        return { ticks, gaps: ticks.slice(1).map((tick, index) => tick - ticks[index]) };
      })(),
      selectedEpochs: routedEpochSelections.get("checkpoint heartbeat|route") ?? [],
    },
    {
      name: "state at tick",
      expected: "deployment state production",
      actual: (() => {
        const state = occurrences(routedDeployment)
          .filter((point) => point.tick <= 74)
          .sort((left, right) => right.tick - left.tick)[0];
        return state ? pointerText(field, records, state.tick) : null;
      })(),
      selectedEpochs: routedEpochSelections.get("deployment state|route") ?? [],
    },
    {
      name: "latest declared superseding state",
      expected: "deployment state rollback",
      actual: (() => {
        const state = occurrences(routedDeployment).at(-1);
        return state ? pointerText(field, records, state.tick) : null;
      })(),
      selectedEpochs: routedEpochSelections.get("deployment state|route") ?? [],
    },
  ].map((result) => ({
    ...result,
    pass: JSON.stringify(result.actual) === JSON.stringify(result.expected),
  }));
  const sortedRoutedLatencies = routedEvidenceLatencies.slice().sort((left, right) => left - right);
  const rotorPassed = capabilities.filter((result) => result.rotorPass).length;
  const routedPassed = routedChecks.filter((result) => result.pass).length;
  if (rotorPassed !== capabilities.length || crcValid !== crcSamples.length) {
    throw new Error("faithful rotor temporal control failed");
  }
  console.log(JSON.stringify({
    fixture: {
      ticks: TICKS,
      rotorPeriodInHoloStore: 24,
      rotorSlotSamples: SLOT_SAMPLES,
      horizonSamples: HORIZON_SAMPLES,
      distractors: TICKS - scheduled.size,
      occurrenceThreshold: MATCH_THRESHOLD,
      spectrumPreparationMs: preparationMs,
      contentToTimeEvidence: {
        queries: evidenceLatencies.length,
        totalMs: evidenceLatencies.reduce((total, value) => total + value, 0),
        avgMs: evidenceLatencies.reduce((total, value) => total + value, 0) / evidenceLatencies.length,
        p95Ms: sortedEvidenceLatencies[
          Math.min(sortedEvidenceLatencies.length - 1, Math.floor(sortedEvidenceLatencies.length * 0.95))
        ],
      },
      routedContentToTimeEvidence: {
        epochShards: epochShards.length,
        candidateEpochs: ROUTED_EPOCHS,
        queries: routedEvidenceLatencies.length,
        totalMs: routedEvidenceLatencies.reduce((total, value) => total + value, 0),
        avgMs: routedEvidenceLatencies.reduce((total, value) => total + value, 0) / routedEvidenceLatencies.length,
        p95Ms: sortedRoutedLatencies[
          Math.min(sortedRoutedLatencies.length - 1, Math.floor(sortedRoutedLatencies.length * 0.95))
        ],
        routerStorageMiB: (needletRouter.storageBytes + diffusionRouter.storageBytes) / 2 ** 20,
      },
      fieldStorageMiB: field.storageBytes / 2 ** 20,
      sampledCrcRecovery: { correct: crcValid, total: crcSamples.length },
    },
    capabilities,
    routedCapabilities: routedChecks,
    summary: {
      rotorPassed,
      rotorTotal: capabilities.length,
      routedRotorPassed: routedPassed,
      routedRotorTotal: routedChecks.length,
      holoNativePassed: nativeHolo.filter((result) => result.holoPass).length,
      holoNativeTotal: nativeHolo.length,
      holoUnsupported: capabilities.filter((result) => result.holoSupport === "unsupported").length,
      boundary: "Rotor reasoning operates over content-to-time evidence and exact pointers. The deployment events are a declared shared-state lineage; temporal order alone does not infer causality or supersession.",
    },
  }, null, 2));
}

main();
