/**
 * Reversible wire/FEC/temporal-harmonic memory experiment.
 *
 * This is deliberately separate from the hash -> sphere -> Clifford engine.
 * Literal bytes are framed, optionally Hamming(7,4)-coded, Gray-QPSK modulated,
 * and bound to time on an independent Fourier axis:
 *
 *   F[h, d] = sum_i exp(-2 pi i k_h t_i / T) W_i[d]
 *
 * A complete temporal basis separates unique ticks exactly. A partial basis is
 * an approximate search surface with global temporal sidelobes.
 */

import { mulberry32 } from "./random";

const HEADER_BYTES = 8;
const SQRT_HALF = 1 / Math.sqrt(2);

export interface WireDecodeResult {
  payload: Uint8Array;
  declaredLength: number | null;
  lengthValid: boolean;
  crcOk: boolean;
  correctedCodewords: number;
  meanDecisionMargin: number;
}

export interface HarmonicSearchHit {
  timeTick: number;
  score: number;
}

export interface PreparedWireProbe {
  spectrum: Float64Array;
  symbolCount: number;
}

function assertBinary(values: Uint8Array): void {
  for (const value of values) {
    if (value > 1) throw new Error("bits must contain only zero or one");
  }
}

function bits(value: number, width: number): Uint8Array {
  const output = new Uint8Array(width);
  for (let index = 0; index < width; index++) {
    output[index] = (value >>> (width - index - 1)) & 1;
  }
  return output;
}

function fromBits(values: Uint8Array): number {
  let output = 0;
  for (const bit of values) output = (output << 1) | bit;
  return output;
}

export function hamming74Encode(nibble: number): Uint8Array {
  if (!Number.isInteger(nibble) || nibble < 0 || nibble > 0xf) {
    throw new Error("nibble must be an integer in [0, 15]");
  }
  const [d0, d1, d2, d3] = bits(nibble, 4);
  const code = new Uint8Array(7);
  code[2] = d0;
  code[4] = d1;
  code[5] = d2;
  code[6] = d3;
  code[0] = code[2] ^ code[4] ^ code[6];
  code[1] = code[2] ^ code[5] ^ code[6];
  code[3] = code[4] ^ code[5] ^ code[6];
  return code;
}

export function hamming74Decode(codeword: Uint8Array): { nibble: number; corrected: boolean } {
  if (codeword.length !== 7) throw new Error("codeword must contain seven bits");
  assertBinary(codeword);
  const code = codeword.slice();
  const s1 = code[0] ^ code[2] ^ code[4] ^ code[6];
  const s2 = code[1] ^ code[2] ^ code[5] ^ code[6];
  const s4 = code[3] ^ code[4] ^ code[5] ^ code[6];
  const syndrome = s1 | (s2 << 1) | (s4 << 2);
  if (syndrome) code[syndrome - 1] ^= 1;
  return {
    nibble: fromBits(new Uint8Array([code[2], code[4], code[5], code[6]])),
    corrected: syndrome !== 0,
  };
}

/** Interleaved [real, imag, ...] unit Gray-QPSK symbols. */
export function qpskModulate(input: Uint8Array): Float64Array {
  if (input.length % 2 !== 0) throw new Error("QPSK requires an even number of bits");
  assertBinary(input);
  const output = new Float64Array(input.length);
  for (let bit = 0; bit < input.length; bit += 2) {
    output[bit] = input[bit + 1] === 0 ? SQRT_HALF : -SQRT_HALF;
    output[bit + 1] = input[bit] === 0 ? SQRT_HALF : -SQRT_HALF;
  }
  return output;
}

function qpskDemodulate(symbols: Float64Array): { bits: Uint8Array; meanMargin: number } {
  if (symbols.length % 2 !== 0) throw new Error("complex symbols must be interleaved");
  const output = new Uint8Array(symbols.length);
  let margin = 0;
  for (let symbol = 0; symbol < symbols.length / 2; symbol++) {
    const real = symbols[2 * symbol];
    const imag = symbols[2 * symbol + 1];
    output[2 * symbol] = imag < 0 ? 1 : 0;
    output[2 * symbol + 1] = real < 0 ? 1 : 0;
    margin += Math.min(Math.abs(real), Math.abs(imag));
  }
  return { bits: output, meanMargin: symbols.length ? margin / (symbols.length / 2) : 0 };
}

export function symbolsPerByte(fec: boolean): number {
  return fec ? 7 : 4;
}

export function wireFrameSymbolLength(payloadBytes: number, fec = true): number {
  if (!Number.isInteger(payloadBytes) || payloadBytes < 0) {
    throw new Error("payloadBytes must be a non-negative integer");
  }
  return (HEADER_BYTES + payloadBytes) * symbolsPerByte(fec);
}

export function wireEncodePayload(data: Uint8Array, fec = true): Float64Array {
  const encoded = new Uint8Array(data.length * symbolsPerByte(fec) * 2);
  let offset = 0;
  for (const value of data) {
    if (fec) {
      encoded.set(hamming74Encode(value >>> 4), offset);
      encoded.set(hamming74Encode(value & 0xf), offset + 7);
      offset += 14;
    } else {
      encoded.set(bits(value, 8), offset);
      offset += 8;
    }
  }
  return qpskModulate(encoded);
}

const CRC_TABLE = new Uint32Array(256);
for (let byte = 0; byte < CRC_TABLE.length; byte++) {
  let value = byte;
  for (let bit = 0; bit < 8; bit++) {
    value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  CRC_TABLE[byte] = value >>> 0;
}

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of data) crc = CRC_TABLE[(crc ^ value) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

export function wireEncodeFrame(data: Uint8Array, fec = true): Float64Array {
  const frame = new Uint8Array(HEADER_BYTES + data.length);
  const view = new DataView(frame.buffer);
  view.setUint32(0, data.length, false);
  view.setUint32(4, crc32(data), false);
  frame.set(data, HEADER_BYTES);
  return wireEncodePayload(frame, fec);
}

function decodeWireBytes(symbols: Float64Array, fec: boolean): {
  bytes: Uint8Array;
  corrected: number;
  meanMargin: number;
} {
  const completeBytes = Math.floor(symbols.length / 2 / symbolsPerByte(fec));
  const usable = symbols.slice(0, completeBytes * symbolsPerByte(fec) * 2);
  const demodulated = qpskDemodulate(usable);
  const output = new Uint8Array(completeBytes);
  let corrected = 0;
  if (fec) {
    for (let index = 0; index < completeBytes; index++) {
      const start = index * 14;
      const high = hamming74Decode(demodulated.bits.slice(start, start + 7));
      const low = hamming74Decode(demodulated.bits.slice(start + 7, start + 14));
      output[index] = (high.nibble << 4) | low.nibble;
      corrected += Number(high.corrected) + Number(low.corrected);
    }
  } else {
    for (let index = 0; index < completeBytes; index++) {
      output[index] = fromBits(demodulated.bits.slice(index * 8, index * 8 + 8));
    }
  }
  return { bytes: output, corrected, meanMargin: demodulated.meanMargin };
}

export function wireDecodeFrame(
  symbols: Float64Array,
  fec = true,
  maxPayloadBytes?: number,
): WireDecodeResult {
  const decoded = decodeWireBytes(symbols, fec);
  if (decoded.bytes.length < HEADER_BYTES) {
    return {
      payload: new Uint8Array(), declaredLength: null, lengthValid: false, crcOk: false,
      correctedCodewords: decoded.corrected, meanDecisionMargin: decoded.meanMargin,
    };
  }
  const view = new DataView(decoded.bytes.buffer, decoded.bytes.byteOffset, decoded.bytes.byteLength);
  const declaredLength = view.getUint32(0, false);
  const expectedCrc = view.getUint32(4, false);
  const available = decoded.bytes.length - HEADER_BYTES;
  const lengthValid = declaredLength <= available && (
    maxPayloadBytes === undefined || declaredLength <= maxPayloadBytes
  );
  const payload = lengthValid
    ? decoded.bytes.slice(HEADER_BYTES, HEADER_BYTES + declaredLength)
    : new Uint8Array();
  return {
    payload,
    declaredLength,
    lengthValid,
    crcOk: lengthValid && crc32(payload) === expectedCrc,
    correctedCodewords: decoded.corrected,
    meanDecisionMargin: decoded.meanMargin,
  };
}

function fftInPlace(output: Float64Array, inverse = false): void {
  const length = output.length / 2;
  if (!Number.isInteger(length) || length < 1 || (length & (length - 1)) !== 0) {
    throw new Error("FFT length must be a positive power of two");
  }
  for (let index = 1, reversed = 0; index < length; index++) {
    let bit = length >>> 1;
    for (; reversed & bit; bit >>>= 1) reversed ^= bit;
    reversed ^= bit;
    if (index < reversed) {
      const real = output[2 * index];
      const imag = output[2 * index + 1];
      output[2 * index] = output[2 * reversed];
      output[2 * index + 1] = output[2 * reversed + 1];
      output[2 * reversed] = real;
      output[2 * reversed + 1] = imag;
    }
  }
  for (let width = 2; width <= length; width <<= 1) {
    const angle = (inverse ? 2 : -2) * Math.PI / width;
    const stepReal = Math.cos(angle);
    const stepImag = Math.sin(angle);
    for (let start = 0; start < length; start += width) {
      let twiddleReal = 1;
      let twiddleImag = 0;
      for (let index = 0; index < width / 2; index++) {
        const even = start + index;
        const odd = even + width / 2;
        const oddReal = output[2 * odd];
        const oddImag = output[2 * odd + 1];
        const productReal = twiddleReal * oddReal - twiddleImag * oddImag;
        const productImag = twiddleReal * oddImag + twiddleImag * oddReal;
        const evenReal = output[2 * even];
        const evenImag = output[2 * even + 1];
        output[2 * even] = evenReal + productReal;
        output[2 * even + 1] = evenImag + productImag;
        output[2 * odd] = evenReal - productReal;
        output[2 * odd + 1] = evenImag - productImag;
        const nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
        twiddleImag = twiddleReal * stepImag + twiddleImag * stepReal;
        twiddleReal = nextReal;
      }
    }
  }
  if (inverse) {
    for (let index = 0; index < output.length; index++) output[index] /= length;
  }
}

function fft(data: Float64Array, inverse = false): Float64Array {
  const output = data.slice();
  fftInPlace(output, inverse);
  return output;
}

function normalizedMagnitudeSketch(spectrum: Float64Array, bands: number): Float64Array {
  const frequencies = spectrum.length / 2;
  if (!Number.isInteger(bands) || bands < 1 || bands > frequencies) {
    throw new Error("sketch bands must be in [1, spectrum frequencies]");
  }
  const sketch = new Float64Array(bands);
  for (let frequency = 0; frequency < frequencies; frequency++) {
    const band = Math.min(bands - 1, Math.floor(frequency * bands / frequencies));
    const real = spectrum[2 * frequency];
    const imag = spectrum[2 * frequency + 1];
    sketch[band] += real * real + imag * imag;
  }
  let norm = 0;
  for (let band = 0; band < bands; band++) {
    sketch[band] = Math.sqrt(sketch[band]);
    norm += sketch[band] * sketch[band];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) for (let band = 0; band < bands; band++) sketch[band] /= norm;
  return sketch;
}

/**
 * Lossy magnitude/rotor routing control over superposed shard spectra.
 * The NFCorpus benchmark currently rejects it for promotion: it is fast but
 * discards enough phase/locality to miss even a known 64-byte prefix shard.
 */
export class HarmonicShardRouter {
  readonly size: number;
  private readonly field: Float64Array;
  private entries = 0;

  constructor(size: number) {
    if (!Number.isInteger(size) || size < 1 || (size & (size - 1)) !== 0) {
      throw new Error("router size must be a positive power of two");
    }
    this.size = size;
    this.field = new Float64Array(size * 2);
  }

  add(sketch: Float64Array, address = this.entries): void {
    if (sketch.length !== this.size || !Number.isInteger(address) || address < 0 || address >= this.size) {
      throw new Error("sketch and address must fit the router");
    }
    for (let band = 0; band < this.size; band++) {
      const angle = -2 * Math.PI * band * address / this.size;
      this.field[2 * band] += sketch[band] * Math.cos(angle);
      this.field[2 * band + 1] += sketch[band] * Math.sin(angle);
    }
    this.entries = Math.max(this.entries, address + 1);
  }

  route(querySketch: Float64Array, topK = 5): HarmonicSearchHit[] {
    if (querySketch.length !== this.size || !Number.isInteger(topK) || topK < 1) {
      throw new Error("query sketch must fit and topK must be positive");
    }
    const correlation = new Float64Array(this.field.length);
    for (let band = 0; band < this.size; band++) {
      correlation[2 * band] = this.field[2 * band] * querySketch[band];
      correlation[2 * band + 1] = this.field[2 * band + 1] * querySketch[band];
    }
    fftInPlace(correlation, true);
    const candidates = Array.from({ length: this.entries }, (_, address) => ({
      timeTick: address,
      score: correlation[2 * address],
    }));
    candidates.sort((left, right) => right.score - left.score);
    return candidates.slice(0, Math.min(topK, candidates.length));
  }
}

/**
 * One-axis rotor field. A timestamp is a circular shift of the QPSK wire;
 * in frequency space the same operation is multiplication by a phase rotor.
 * Writes stay local in the time-domain field, and prepareSpectrum() batches
 * the equivalent rotor bindings into one FFT for content-to-time search.
 */
export class DirectRotorWireField {
  readonly horizonSamples: number;
  readonly maxPayloadBytes: number;
  readonly fec: boolean;
  private readonly timeField: Float64Array;
  private fieldSpectrum: Float64Array | null = null;

  constructor(options: { horizonSamples: number; maxPayloadBytes: number; fec?: boolean }) {
    const { horizonSamples, maxPayloadBytes, fec = true } = options;
    if (
      !Number.isInteger(horizonSamples) || horizonSamples < 1 ||
      (horizonSamples & (horizonSamples - 1)) !== 0
    ) {
      throw new Error("horizonSamples must be a positive power of two");
    }
    if (!Number.isInteger(maxPayloadBytes) || maxPayloadBytes < 1) {
      throw new Error("maxPayloadBytes must be positive");
    }
    if (wireFrameSymbolLength(maxPayloadBytes, fec) > horizonSamples) {
      throw new Error("maximum frame must fit inside the field horizon");
    }
    this.horizonSamples = horizonSamples;
    this.maxPayloadBytes = maxPayloadBytes;
    this.fec = fec;
    this.timeField = new Float64Array(horizonSamples * 2);
  }

  get headerSymbols(): number {
    return HEADER_BYTES * symbolsPerByte(this.fec);
  }

  get storageBytes(): number {
    return this.timeField.byteLength + (this.fieldSpectrum?.byteLength ?? 0);
  }

  add(data: Uint8Array, offset: number, weight = 1): void {
    if (data.length > this.maxPayloadBytes) throw new Error("payload exceeds configured maximum");
    if (!Number.isInteger(offset)) throw new Error("offset must be an integer symbol coordinate");
    if (!Number.isFinite(weight)) throw new Error("weight must be finite");
    const frame = wireEncodeFrame(data, this.fec);
    const start = ((offset % this.horizonSamples) + this.horizonSamples) % this.horizonSamples;
    for (let symbol = 0; symbol < frame.length / 2; symbol++) {
      const target = (start + symbol) % this.horizonSamples;
      this.timeField[2 * target] += weight * frame[2 * symbol];
      this.timeField[2 * target + 1] += weight * frame[2 * symbol + 1];
    }
    this.fieldSpectrum = null;
  }

  subtract(data: Uint8Array, offset: number, weight = 1): void {
    this.add(data, offset, -weight);
  }

  readWave(offset: number, symbolCount = wireFrameSymbolLength(this.maxPayloadBytes, this.fec)): Float64Array {
    if (!Number.isInteger(offset) || !Number.isInteger(symbolCount) || symbolCount < 1) {
      throw new Error("offset and symbolCount must be valid integers");
    }
    const output = new Float64Array(symbolCount * 2);
    const start = ((offset % this.horizonSamples) + this.horizonSamples) % this.horizonSamples;
    for (let symbol = 0; symbol < symbolCount; symbol++) {
      const source = (start + symbol) % this.horizonSamples;
      output[2 * symbol] = this.timeField[2 * source];
      output[2 * symbol + 1] = this.timeField[2 * source + 1];
    }
    return output;
  }

  readPacket(offset: number): WireDecodeResult {
    return wireDecodeFrame(
      this.readWave(offset),
      this.fec,
      this.maxPayloadBytes,
    );
  }

  prepareSpectrum(): void {
    this.fieldSpectrum = fft(this.timeField);
  }

  prepareProbe(probe: Uint8Array): PreparedWireProbe {
    if (probe.length === 0) throw new Error("probe cannot be empty");
    const wire = wireEncodePayload(probe, this.fec);
    if (wire.length / 2 > this.horizonSamples) throw new Error("probe exceeds field horizon");
    const spectrum = new Float64Array(this.horizonSamples * 2);
    spectrum.set(wire);
    fftInPlace(spectrum);
    return { spectrum, symbolCount: wire.length / 2 };
  }

  spectrumSketch(bands: number): Float64Array {
    if (!this.fieldSpectrum) throw new Error("call prepareSpectrum() after the last write");
    return normalizedMagnitudeSketch(this.fieldSpectrum, bands);
  }

  probeSketch(probe: PreparedWireProbe, bands: number): Float64Array {
    if (probe.spectrum.length !== this.horizonSamples * 2) {
      throw new Error("probe was prepared for a different field horizon");
    }
    return normalizedMagnitudeSketch(probe.spectrum, bands);
  }

  correlationPrepared(probe: PreparedWireProbe): Float64Array {
    if (!this.fieldSpectrum) throw new Error("call prepareSpectrum() after the last write");
    if (probe.spectrum.length !== this.fieldSpectrum.length || probe.symbolCount < 1) {
      throw new Error("probe was prepared for a different field horizon");
    }
    const product = new Float64Array(this.fieldSpectrum.length);
    for (let frequency = 0; frequency < this.horizonSamples; frequency++) {
      const fieldReal = this.fieldSpectrum[2 * frequency];
      const fieldImag = this.fieldSpectrum[2 * frequency + 1];
      const probeReal = probe.spectrum[2 * frequency];
      const probeImag = probe.spectrum[2 * frequency + 1];
      product[2 * frequency] = fieldReal * probeReal + fieldImag * probeImag;
      product[2 * frequency + 1] = fieldImag * probeReal - fieldReal * probeImag;
    }
    fftInPlace(product, true);
    for (let index = 0; index < product.length; index++) product[index] /= probe.symbolCount;
    return product;
  }

  correlation(probe: Uint8Array): Float64Array {
    return this.correlationPrepared(this.prepareProbe(probe));
  }

  search(probe: Uint8Array, topK = 5): HarmonicSearchHit[] {
    if (!Number.isInteger(topK) || topK < 1) throw new Error("topK must be positive");
    const correlation = this.correlation(probe);
    const scores = new Float64Array(this.horizonSamples);
    for (let offset = 0; offset < this.horizonSamples; offset++) {
      scores[offset] = Math.hypot(correlation[2 * offset], correlation[2 * offset + 1]);
    }
    const hits: HarmonicSearchHit[] = [];
    const suppression = Math.max(1, Math.floor(wireEncodePayload(probe, this.fec).length / 4));
    for (let rank = 0; rank < Math.min(topK, this.horizonSamples); rank++) {
      let bestOffset = 0;
      let bestScore = -Infinity;
      for (let offset = 0; offset < scores.length; offset++) {
        if (scores[offset] > bestScore) {
          bestOffset = offset;
          bestScore = scores[offset];
        }
      }
      if (!Number.isFinite(bestScore)) break;
      hits.push({ timeTick: (bestOffset - this.headerSymbols + this.horizonSamples) % this.horizonSamples, score: bestScore });
      for (let delta = -suppression; delta <= suppression; delta++) {
        scores[(bestOffset + delta + this.horizonSamples) % this.horizonSamples] = -Infinity;
      }
    }
    return hits;
  }
}

function chooseFrequencyIndices(horizon: number, bands: number, seed: number): Uint32Array {
  if (bands === horizon) return Uint32Array.from({ length: horizon }, (_, index) => index);
  const values = Array.from({ length: horizon }, (_, index) => index);
  const random = mulberry32(seed);
  for (let index = values.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [values[index], values[swap]] = [values[swap], values[index]];
  }
  return Uint32Array.from(values.slice(0, bands).sort((left, right) => left - right));
}

export class SeparableHarmonicWireField {
  readonly horizonTicks: number;
  readonly nTimeBands: number;
  readonly maxPayloadBytes: number;
  readonly fec: boolean;
  readonly frequencyIndices: Uint32Array;
  private readonly field: Float64Array;

  constructor(options: {
    horizonTicks: number;
    nTimeBands: number;
    maxPayloadBytes: number;
    fec?: boolean;
    seed?: number;
  }) {
    const { horizonTicks, nTimeBands, maxPayloadBytes, fec = true, seed = 0 } = options;
    if (!Number.isInteger(horizonTicks) || horizonTicks < 1 || (horizonTicks & (horizonTicks - 1)) !== 0) {
      throw new Error("horizonTicks must be a positive power of two");
    }
    if (!Number.isInteger(nTimeBands) || nTimeBands < 1 || nTimeBands > horizonTicks) {
      throw new Error("nTimeBands must be in [1, horizonTicks]");
    }
    if (!Number.isInteger(maxPayloadBytes) || maxPayloadBytes < 1) {
      throw new Error("maxPayloadBytes must be positive");
    }
    this.horizonTicks = horizonTicks;
    this.nTimeBands = nTimeBands;
    this.maxPayloadBytes = maxPayloadBytes;
    this.fec = fec;
    this.frequencyIndices = chooseFrequencyIndices(horizonTicks, nTimeBands, seed);
    this.field = new Float64Array(this.coefficientCount * 2);
  }

  get maximumFrameSymbols(): number {
    return (HEADER_BYTES + this.maxPayloadBytes) * symbolsPerByte(this.fec);
  }

  get headerSymbols(): number {
    return HEADER_BYTES * symbolsPerByte(this.fec);
  }

  get coefficientCount(): number {
    return this.nTimeBands * this.maximumFrameSymbols;
  }

  get storageBytes(): number {
    return this.field.byteLength;
  }

  private rotor(timeTick: number): Float64Array {
    if (!Number.isInteger(timeTick)) throw new Error("timeTick must be an integer");
    const tick = ((timeTick % this.horizonTicks) + this.horizonTicks) % this.horizonTicks;
    const output = new Float64Array(this.nTimeBands * 2);
    for (let band = 0; band < this.nTimeBands; band++) {
      const angle = -2 * Math.PI * this.frequencyIndices[band] * tick / this.horizonTicks;
      output[2 * band] = Math.cos(angle);
      output[2 * band + 1] = Math.sin(angle);
    }
    return output;
  }

  private paddedFrame(data: Uint8Array): Float64Array {
    if (data.length > this.maxPayloadBytes) throw new Error("payload exceeds configured maximum");
    const frame = wireEncodeFrame(data, this.fec);
    const output = new Float64Array(this.maximumFrameSymbols * 2);
    output.set(frame);
    return output;
  }

  private update(data: Uint8Array, timeTick: number, weight: number): void {
    if (!Number.isFinite(weight)) throw new Error("weight must be finite");
    const rotor = this.rotor(timeTick);
    const frame = this.paddedFrame(data);
    for (let band = 0; band < this.nTimeBands; band++) {
      const rotorReal = rotor[2 * band];
      const rotorImag = rotor[2 * band + 1];
      for (let coordinate = 0; coordinate < this.maximumFrameSymbols; coordinate++) {
        const wireReal = frame[2 * coordinate];
        const wireImag = frame[2 * coordinate + 1];
        const fieldIndex = 2 * (band * this.maximumFrameSymbols + coordinate);
        this.field[fieldIndex] += weight * (rotorReal * wireReal - rotorImag * wireImag);
        this.field[fieldIndex + 1] += weight * (rotorReal * wireImag + rotorImag * wireReal);
      }
    }
  }

  add(data: Uint8Array, timeTick: number, weight = 1): void {
    this.update(data, timeTick, weight);
  }

  subtract(data: Uint8Array, timeTick: number, weight = 1): void {
    this.update(data, timeTick, -weight);
  }

  readWave(timeTick: number): Float64Array {
    const rotor = this.rotor(timeTick);
    const output = new Float64Array(this.maximumFrameSymbols * 2);
    for (let coordinate = 0; coordinate < this.maximumFrameSymbols; coordinate++) {
      let real = 0;
      let imag = 0;
      for (let band = 0; band < this.nTimeBands; band++) {
        const rotorReal = rotor[2 * band];
        const rotorImag = rotor[2 * band + 1];
        const fieldIndex = 2 * (band * this.maximumFrameSymbols + coordinate);
        const fieldReal = this.field[fieldIndex];
        const fieldImag = this.field[fieldIndex + 1];
        real += rotorReal * fieldReal + rotorImag * fieldImag;
        imag += rotorReal * fieldImag - rotorImag * fieldReal;
      }
      output[2 * coordinate] = real / this.nTimeBands;
      output[2 * coordinate + 1] = imag / this.nTimeBands;
    }
    return output;
  }

  readPacket(timeTick: number): WireDecodeResult {
    return wireDecodeFrame(this.readWave(timeTick), this.fec, this.maxPayloadBytes);
  }

  correlation(probe: Uint8Array, payloadOffsetBytes = 0): Float64Array {
    if (probe.length === 0 || !Number.isInteger(payloadOffsetBytes) || payloadOffsetBytes < 0) {
      throw new Error("probe must be non-empty and payloadOffsetBytes non-negative");
    }
    const query = wireEncodePayload(probe, this.fec);
    const querySymbols = query.length / 2;
    const start = this.headerSymbols + payloadOffsetBytes * symbolsPerByte(this.fec);
    if (start + querySymbols > this.maximumFrameSymbols) {
      throw new Error("probe exceeds the configured wire frame");
    }
    const spectrum = new Float64Array(this.horizonTicks * 2);
    for (let band = 0; band < this.nTimeBands; band++) {
      let real = 0;
      let imag = 0;
      for (let coordinate = 0; coordinate < querySymbols; coordinate++) {
        const fieldIndex = 2 * (band * this.maximumFrameSymbols + start + coordinate);
        const queryReal = query[2 * coordinate];
        const queryImag = query[2 * coordinate + 1];
        const fieldReal = this.field[fieldIndex];
        const fieldImag = this.field[fieldIndex + 1];
        real += fieldReal * queryReal + fieldImag * queryImag;
        imag += fieldImag * queryReal - fieldReal * queryImag;
      }
      const frequency = this.frequencyIndices[band];
      spectrum[2 * frequency] = real / querySymbols;
      spectrum[2 * frequency + 1] = imag / querySymbols;
    }
    const correlated = fft(spectrum, true);
    const scale = this.horizonTicks / this.nTimeBands;
    for (let index = 0; index < correlated.length; index++) correlated[index] *= scale;
    return correlated;
  }

  search(
    probe: Uint8Array,
    options: { topK?: number; payloadOffsetBytes?: number; suppressionTicks?: number } = {},
  ): HarmonicSearchHit[] {
    const { topK = 5, payloadOffsetBytes = 0, suppressionTicks = 0 } = options;
    if (!Number.isInteger(topK) || topK < 1 || !Number.isInteger(suppressionTicks) || suppressionTicks < 0) {
      throw new Error("topK must be positive and suppressionTicks non-negative");
    }
    const correlation = this.correlation(probe, payloadOffsetBytes);
    const scores = new Float64Array(this.horizonTicks);
    for (let tick = 0; tick < this.horizonTicks; tick++) {
      scores[tick] = Math.hypot(correlation[2 * tick], correlation[2 * tick + 1]);
    }
    const hits: HarmonicSearchHit[] = [];
    for (let rank = 0; rank < Math.min(topK, this.horizonTicks); rank++) {
      let bestTick = 0;
      let bestScore = -Infinity;
      for (let tick = 0; tick < scores.length; tick++) {
        if (scores[tick] > bestScore) {
          bestTick = tick;
          bestScore = scores[tick];
        }
      }
      if (!Number.isFinite(bestScore)) break;
      hits.push({ timeTick: bestTick, score: bestScore });
      for (let delta = -suppressionTicks; delta <= suppressionTicks; delta++) {
        scores[(bestTick + delta + this.horizonTicks) % this.horizonTicks] = -Infinity;
      }
    }
    return hits;
  }
}
