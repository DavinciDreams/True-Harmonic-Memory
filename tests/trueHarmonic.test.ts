import assert from "node:assert/strict";

import {
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
