import assert from "node:assert/strict";

import {
  DirectRotorWireField,
  HarmonicShardRouter,
  LocalizedSpectralRouterKind,
  LocalizedSpectralShardRouter,
  SeparableHarmonicWireField,
  hamming74Decode,
  hamming74Encode,
  wireDecodeFrame,
  wireEncodeFrame,
} from "../src/lib/holo/trueHarmonic";
import { mulberry32 } from "../src/lib/holo/random";

const encoder = new TextEncoder();

for (let nibble = 0; nibble < 16; nibble++) {
  const encoded = hamming74Encode(nibble);
  assert.deepEqual(hamming74Decode(encoded), { nibble, corrected: false });
  for (let bit = 0; bit < 7; bit++) {
    const damaged = encoded.slice();
    damaged[bit] ^= 1;
    assert.deepEqual(hamming74Decode(damaged), { nibble, corrected: true });
  }
}

const everyByte = Uint8Array.from({ length: 256 }, (_, index) => index);
const framed = wireDecodeFrame(wireEncodeFrame(everyByte), true, everyByte.length);
assert.equal(framed.crcOk, true);
assert.deepEqual(framed.payload, everyByte);

const direct = new DirectRotorWireField({ horizonSamples: 4096, maxPayloadBytes: 48, fec: true });
const directPackets = [
  [encoder.encode("shared-prefix first direct packet"), 113],
  [encoder.encode("shared-prefix second direct packet"), 1703],
] as const;
for (const [payload, offset] of directPackets) direct.add(payload, offset);
for (const [payload, offset] of directPackets) {
  const decoded = direct.readPacket(offset);
  assert.equal(decoded.crcOk, true);
  assert.deepEqual(decoded.payload, payload);
}
direct.prepareSpectrum();
for (const [payload, offset] of directPackets) {
  assert.equal(direct.search(payload, 1)[0].timeTick, offset);
}
const directPrefixHits = new Set(
  direct.search(encoder.encode("shared-prefix"), 2).map((hit) => hit.timeTick),
);
assert.deepEqual(directPrefixHits, new Set(directPackets.map(([, offset]) => offset)));
const unrelated = new DirectRotorWireField({ horizonSamples: 4096, maxPayloadBytes: 48, fec: true });
unrelated.add(encoder.encode("completely unrelated control bytes"), 509);
unrelated.prepareSpectrum();
const router = new HarmonicShardRouter(8);
router.add(direct.spectrumSketch(8), 1);
router.add(unrelated.spectrumSketch(8), 3);
const prefixProbe = direct.prepareProbe(encoder.encode("shared-prefix"));
assert.equal(router.route(direct.probeSketch(prefixProbe, 8), 1)[0].timeTick, 1);
const localizedKinds: LocalizedSpectralRouterKind[] = [
  "sparse-fourier", "needlet", "slepian", "diffusion", "gabor",
];
for (const kind of localizedKinds) {
  const localized = new LocalizedSpectralShardRouter({
    kind,
    horizonSamples: 4096,
    projectionSize: 256,
  });
  localized.add(direct, 1);
  localized.add(unrelated, 3);
  assert.equal(localized.route(direct, prefixProbe, 1)[0].timeTick, 1, `${kind} prefix route`);
  assert.ok(localized.storageBytes > 0);
}
for (const [payload, offset] of directPackets) direct.subtract(payload, offset);
for (const [, offset] of directPackets) {
  assert.ok(direct.readWave(offset, 64).every((value) => Math.abs(value) < 1e-10));
}

const exact = new SeparableHarmonicWireField({
  horizonTicks: 64,
  nTimeBands: 64,
  maxPayloadBytes: 48,
  fec: true,
});
const packets = [
  [encoder.encode("the rotor is the timestamp"), 7],
  [encoder.encode("literal bytes become a QPSK wave"), 23],
  [encoder.encode("content can retrieve time"), 51],
] as const;
for (const [payload, tick] of packets) exact.add(payload, tick);
for (const [payload, tick] of packets) {
  const decoded = exact.readPacket(tick);
  assert.equal(decoded.crcOk, true);
  assert.deepEqual(decoded.payload, payload);
  assert.equal(exact.search(payload, { topK: 1 })[0].timeTick, tick);
}

const storageBytes = exact.storageBytes;
for (const [payload, tick] of packets) exact.subtract(payload, tick);
assert.equal(exact.storageBytes, storageBytes);
for (let tick = 0; tick < 64; tick++) {
  assert.ok(exact.readWave(tick).every((value) => Math.abs(value) < 1e-10));
}

const partial = new SeparableHarmonicWireField({
  horizonTicks: 256,
  nTimeBands: 64,
  maxPayloadBytes: 32,
  fec: false,
  seed: 20260822,
});
const partialPackets: Array<{ payload: Uint8Array; tick: number }> = [];
const random = mulberry32(917249);
for (let index = 0; index < 12; index++) {
  const payload = Uint8Array.from({ length: 24 }, () => Math.floor(random() * 256));
  const tick = 5 + index * 17;
  partialPackets.push({ payload, tick });
  partial.add(payload, tick);
}
for (const { payload, tick } of partialPackets) {
  assert.equal(partial.search(payload, { topK: 1 })[0].timeTick, tick);
}

const structured = new SeparableHarmonicWireField({
  horizonTicks: 256,
  nTimeBands: 32,
  maxPayloadBytes: 32,
  fec: false,
  seed: 20260822,
});
const shared = encoder.encode("project-alpha");
for (let index = 0; index < 12; index++) {
  structured.add(encoder.encode(`project-alpha packet ${index.toString().padStart(2, "0")}`), 5 + index * 17);
}
const expected = new Set(Array.from({ length: 12 }, (_, index) => 5 + index * 17));
const found = new Set(structured.search(shared, { topK: 12 }).map((hit) => hit.timeTick));
assert.ok([...found].filter((tick) => expected.has(tick)).length < expected.size);

console.log("true harmonic tests passed");
