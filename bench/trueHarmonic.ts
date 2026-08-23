import { performance } from "node:perf_hooks";

import { SeparableHarmonicWireField } from "../src/lib/holo/trueHarmonic";
import { mulberry32 } from "../src/lib/holo/random";

function payloads(count: number, width: number, seed: number): Uint8Array[] {
  const random = mulberry32(seed);
  return Array.from({ length: count }, () =>
    Uint8Array.from({ length: width }, () => Math.floor(random() * 256))
  );
}

function trial(options: { count: number; fec: boolean; completeBasis: boolean; seed: number }) {
  const { count, fec, completeBasis, seed } = options;
  const maxPayloadBytes = 32;
  const nTimeBands = fec ? 128 : 256;
  const horizonTicks = completeBasis ? nTimeBands : 4096;
  const field = new SeparableHarmonicWireField({
    horizonTicks,
    nTimeBands,
    maxPayloadBytes,
    fec,
    seed,
  });
  const values = payloads(count, maxPayloadBytes, seed + 1);
  const ticks = Array.from({ length: count }, (_, index) => (index * 31 + 7) % horizonTicks);
  const started = performance.now();
  values.forEach((payload, index) => field.add(payload, ticks[index]));
  const indexMs = performance.now() - started;
  let searchCorrect = 0;
  let decodeCorrect = 0;
  values.forEach((payload, index) => {
    if (field.search(payload, { topK: 1 })[0].timeTick === ticks[index]) searchCorrect++;
    const decoded = field.readPacket(ticks[index]);
    if (decoded.crcOk && Buffer.from(decoded.payload).equals(Buffer.from(payload))) decodeCorrect++;
  });
  return {
    count,
    fec,
    temporalBasis: completeBasis ? "complete_dft" : "partial_fourier",
    horizonTicks,
    timeBands: nTimeBands,
    wireCoordinates: field.maximumFrameSymbols,
    coefficientCount: field.coefficientCount,
    fieldKiB: field.storageBytes / 1024,
    searchTop1: searchCorrect / count,
    crcByteRecovery: decodeCorrect / count,
    indexMs,
  };
}

const reports = [];
for (const count of [32, 64, 128]) {
  for (const fec of [false, true]) {
    for (const completeBasis of [true, false]) {
      reports.push(trial({ count, fec, completeBasis, seed: 20260822 }));
    }
  }
}
console.log(JSON.stringify({
  experiment: "reversible wire/FEC/temporal-harmonic field",
  reports,
  boundary: "Complete temporal modes provide exact separation; partial modes are an approximate search surface.",
}, null, 2));
