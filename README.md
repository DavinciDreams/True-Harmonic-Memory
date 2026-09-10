# True Harmonic Memory

True Harmonic Memory stores the payload itself as a wave. UTF-8 bytes become
bits, bit pairs become Gray-coded QPSK symbols, and the resulting ordered
complex wire is bound directly to time with a phase rotor. The representation
is literal and reversible: successful reads reconstruct the original bytes
and validate their frame CRC.

The central implementation is
[`src/lib/holo/trueHarmonic.ts`](src/lib/holo/trueHarmonic.ts). It supports
reversible payload recovery, content-to-time correlation, time-to-content
reads, optional Hamming(7,4) error correction, CRC verification, sharded
fields, and localized spectral routing experiments.

## Representation

```text
UTF-8 bytes
    │
    ▼
length + CRC32 frame
    │
    ├── optional Hamming(7,4) FEC
    ▼
ordered bits ── pair bits ──▶ Gray QPSK ──▶ complex wire W
                                                    │
                                      bind to time with a rotor
                                                    │
                                                    ▼
                                      superposed harmonic field F
```

The QPSK carrier mapping preserves every encoded bit pair directly:

| Bits | Complex symbol |
|---|---|
| `00` | `(+1 + i) / √2` |
| `01` | `(-1 + i) / √2` |
| `11` | `(-1 - i) / √2` |
| `10` | `(+1 - i) / √2` |

For a one-axis field, placing wire `Wᵢ` at time coordinate `tᵢ` is a circular
shift in the wire domain and the equivalent phase multiplication in the
frequency domain:

```text
F[k] = Σᵢ FFT(Wᵢ)[k] exp(-2π i k tᵢ / N)
```

That duality gives the memory two mechanical retrieval directions:

- **Time → content:** undo the shift, demodulate the QPSK symbols, decode
  optional FEC, and accept the payload only when its declared length and CRC
  validate.
- **Content → time:** encode an exact payload or byte prefix as a wire,
  conjugate-multiply its spectrum with the field spectrum, and use one IFFT
  to expose matching time coordinates.

`DirectRotorWireField` is the primary representation. It writes frames into a
local time-domain field and prepares the equivalent rotor-bound spectrum in
one FFT. `SeparableHarmonicWireField` is the controlled temporal-basis version
used to measure exact and partial harmonic bases.

## Quick start

```bash
pnpm install
pnpm test:true-harmonic
pnpm bench:true-harmonic
pnpm bench:true-harmonic:temporal
pnpm bench:true-harmonic:nfcorpus -- --queries 60
```

A minimal round trip:

```ts
import { DirectRotorWireField } from "./src/lib/holo/trueHarmonic";

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const payload = encoder.encode("literal bytes become a QPSK wave");

const field = new DirectRotorWireField({
  horizonSamples: 4096,
  maxPayloadBytes: 64,
  fec: true,
});

field.add(payload, 1024);
field.prepareSpectrum();

const hit = field.search(payload, 1)[0];
const recovered = field.readPacket(hit.timeTick);

console.log(hit.timeTick);                 // 1024
console.log(recovered.crcOk);              // true
console.log(decoder.decode(recovered.payload));
```

## Benchmarks

The repository includes the controlled benchmarks developed with the true
harmonic implementation. Timings below are representative local
CPU/TypeScript measurements; the accuracy and recovery checks are the
important invariants.

### Reversible QPSK field

[`bench/trueHarmonic.ts`](bench/trueHarmonic.ts) varies record count, FEC, and
temporal basis. With a complete DFT basis, all measured loads recovered both
the correct timestamp and the exact CRC-valid payload.

| Records | Encoding | Time-search top-1 | CRC-valid byte recovery | Field |
|---:|---|---:|---:|---:|
| 32 | direct QPSK | 100% | 100% | 640 KiB |
| 64 | direct QPSK | 100% | 100% | 640 KiB |
| 128 | direct QPSK | 100% | 100% | 640 KiB |
| 32 | Hamming(7,4) + QPSK | 100% | 100% | 560 KiB |
| 64 | Hamming(7,4) + QPSK | 100% | 100% | 560 KiB |
| 128 | Hamming(7,4) + QPSK | 100% | 100% | 560 KiB |

The field sizes differ because the controlled FEC runs use 128 complete time
modes and the direct-QPSK runs use 256; they are not equal-capacity storage
comparisons.

### Full-corpus preservation

[`bench/trueHarmonicNfcorpus.ts`](bench/trueHarmonicNfcorpus.ts) places all
3,633 NFCorpus documents—5.79 MB of literal UTF-8—into 90 sharded rotor fields.

| Measurement | Result |
|---|---:|
| Encode and place all documents | 1.14 s |
| Prepare all field spectra | 0.87 s |
| Total query-ready time | 2.01 s |
| Sampled exact reads | 32 / 32 CRC-valid |
| Known 64-byte prefix probe | source ranked #1 |
| Full field storage | 720 MiB |

This benchmark exercises literal preservation and byte-wave correlation. It
does not treat a byte prefix as learned semantic similarity.

### Temporal reasoning from harmonic evidence

[`bench/trueHarmonicTemporal.ts`](bench/trueHarmonicTemporal.ts) builds a
96-tick fixture with 82 distractors, phase-identical recurring heartbeats,
interval anchors, ordered events, and an explicitly declared deployment-state
lineage.

The faithful field passed all nine tested capabilities:

1. near-time lookup;
2. distinguishing events at identical 24-tick phases;
3. before/after;
4. elapsed ticks;
5. an event between two anchors;
6. ordered sequence recovery;
7. recurrence and gap recovery;
8. state at a requested time; and
9. latest declared superseding state.

| Temporal evidence mode | Coverage | Capabilities | Avg evidence query |
|---|---:|---:|---:|
| One faithful non-wrapping field | complete | 9 / 9 | 6.2 ms |
| Four exact epoch FFTs | complete | 9 / 9 | 4.76 ms |

Ten sampled payloads also recovered byte-for-byte with valid CRCs. The result
shows that relational operations can run over field-derived content/time
evidence while retaining pointers to reversible payloads.

## Correctness boundary

- A complete temporal basis provides exact separation up to its declared
  capacity. A partial basis is an approximate search surface, not lossless
  temporal compression.
- CRC32 detects whether the decoded frame is intact. Hamming(7,4) can correct
  single-bit codeword errors, but it cannot manufacture temporal modes that
  were not stored.
- Content-to-time matching is correlation over literal QPSK byte waves. A
  separate semantic layer may be composed with it, but is not part of the
  direct representation.
- Time order alone is not causality or supersession. The temporal benchmark
  answers state-lineage questions only where that lineage is explicitly
  declared.
- Rotors live in finite numerical representations, so claims are bounded by
  the measured field horizon and precision.

## Project map

```text
src/lib/holo/trueHarmonic.ts       direct QPSK wire, FEC, CRC, FFT fields,
                                   search, sharding, spectral routers
tests/trueHarmonic.test.ts         modulation, correction, recovery, search,
                                   routing, subtraction, and basis checks
bench/trueHarmonic.ts              complete/partial temporal-basis control
bench/trueHarmonicNfcorpus.ts      full-corpus rotor-field benchmark
bench/trueHarmonicTemporal.ts      temporal capability benchmark
```

The repository also retains the earlier visualization and comparison
benchmarks for reproducibility, but they are not the True Harmonic
representation described above.

## Provenance

This repository preserves the harmonic-memory research line from
[`DavinciDreams/Holographic-Frequencies-Memory`](https://github.com/DavinciDreams/Holographic-Frequencies-Memory)
without its later NFT layer. Its history retains the original non-NFT baseline
by `alextitonis` and the separately authored `DavinciDreams` temporal-rotor,
reversible-wire, spectral-routing, NFCorpus, and temporal-reasoning commits.

## References

- Plate, T. A. (1995). [Holographic Reduced
  Representations](https://doi.org/10.1109/72.377968). *IEEE Transactions on
  Neural Networks*, 6(3), 623–641.
- Hamming, R. W. (1950). [Error Detecting and Error Correcting
  Codes](https://doi.org/10.1002/j.1538-7305.1950.tb00463.x). *Bell System
  Technical Journal*, 29(2), 147–160.
