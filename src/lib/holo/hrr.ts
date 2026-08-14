// Holographic Reduced Representations (Plate 1995) binding — an alternative
// to blade.ts's grade-compressed Clifford binding, prototyped here to test
// whether the context-bound quality gap (see README's "Context-bound
// queries") is a property of *this specific* binding scheme (grade-3 blade
// compression, capped at C(NUM_GENERATORS,3) directions) or of context
// binding in general.
//
// Circular convolution binds two same-length vectors into one of the same
// length — no dimensionality reduction, unlike sphereToContextBlade's
// projection down to C(n,3) blade components. That's the whole hypothesis:
// blade.ts's binding is lossy *by construction* (it must compress
// SPHERE_DIM down to fit inside the algebra's grade-3 subspace); circular
// convolution binds at full SPHERE_DIM instead, so if the gap shrinks here,
// the compression — not "binding content to context" in general — was the
// bottleneck.

/**
 * a (*) b : circular convolution, O(n^2) direct form (fine at SPHERE_DIM
 * scale; an FFT-based O(n log n) version would be the move for much larger
 * n, not implemented here since this is a measurement prototype).
 */
export function circularConvolve(a: Float64Array, b: Float64Array): Float64Array {
  const n = a.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      s += a[i] * b[(((k - i) % n) + n) % n];
    }
    out[k] = s;
  }
  return out;
}

/**
 * a ⊙ b : element-wise (Hadamard) product — a cheaper (`O(n)`), diagonal
 * alternative to circular convolution, tried as a way to close more of the
 * remaining context-bound-vs-ceiling gap (HRR reaches ~81% of ceiling, see
 * the README's Context-bound queries section). Tested via a bench/context.ts
 * prototype and **rejected** — not because it scored badly, but because the
 * good score was a mathematical artifact, not real binding:
 * `bench/context.ts`'s topic tags are single words, and a single-word
 * `textToSphereVector` output has every component at exactly the same
 * magnitude (only signs vary — confirmed directly: component² variance
 * ~2.4e-17 relative to its own mean, vs. ~1e-5 for a real multi-word
 * context). Element-wise-multiplying both sides of a cosine comparison by
 * such a vector is a signed permutation, which cancels out in cosine
 * similarity *exactly* — so "Hadamard-bound" was silently computing plain
 * content-only cosine, not "content in this context", and hit the ceiling
 * because it's mathematically identical to how the ceiling itself is
 * computed. Confirmed directly: `cosine(hadamard(q,c), hadamard(d,c))
 * === cosine(q,d)` bit-for-bit when `c` is single-word, but genuinely
 * different (and untested for quality) when `c` is a real bundled
 * multi-word context. Kept here, unused, as a documented dead end rather
 * than deleted — re-testing this would need a benchmark with genuinely
 * multi-word contexts, not a fix to this function itself. Unlike circular
 * convolution, this also has no clean unbind operation (no elementwise
 * "divide" that's numerically well-behaved for near-zero components) —
 * moot for this engine either way, since it only ever correlates two bound
 * vectors directly and never unbinds.
 */
export function hadamard(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

export function normalize(v: Float64Array): Float64Array {
  let mag = 0;
  for (let i = 0; i < v.length; i++) mag += v[i] * v[i];
  mag = Math.sqrt(mag) || 1;
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / mag;
  return out;
}

export function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
