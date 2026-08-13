// Spectral projection via Gegenbauer polynomials.
//
// Gegenbauer polynomials C_l^lambda(x) are the hyperspherical generalization
// of Fourier / Chebyshev bases: on S^(n-1) they are the reproducing kernels
// for degree-l spherical harmonics when lambda = (n-2)/2. We use them here to
// decompose a sphere wave into a harmonic spectrum: a small set of
// coefficients (one per "frequency band" l) that summarize the wave's shape.

import { SPHERE_DIM } from "./sphere";

export const SPECTRUM_BANDS = 16;
/** Standard choice tying Gegenbauer order to ambient sphere dimension. */
export const GEGENBAUER_LAMBDA = (SPHERE_DIM - 2) / 2;

/** C_l^lambda(x) via the standard three-term recurrence. */
export function gegenbauer(l: number, lambda: number, x: number): number {
  if (l === 0) return 1;
  if (l === 1) return 2 * lambda * x;
  let cPrev2 = 1; // C_0
  let cPrev1 = 2 * lambda * x; // C_1
  let c = cPrev1;
  for (let k = 2; k <= l; k++) {
    c =
      (2 * (k - 1 + lambda) * x * cPrev1 - (k - 2 + 2 * lambda) * cPrev2) / k;
    cPrev2 = cPrev1;
    cPrev1 = c;
  }
  return c;
}

/**
 * C_l^lambda(x) normalized by its own endpoint value C_l^lambda(1), so it's
 * bounded to [-1, 1] for every degree. This matters: for lambda this large
 * (order SPHERE_DIM/2), the *raw* polynomial explodes near x = +-1 — e.g.
 * C_15^11(1) is in the billions — which would let the two sample points
 * nearest the sphere-vector's ends numerically swamp every other
 * component's contribution to the sum below. Normalizing keeps every
 * sample point's contribution comparable, so the projection actually
 * reflects the whole vector instead of ~2 of its 24 components.
 */
function gegenbauerNormalized(l: number, lambda: number, x: number): number {
  const end = gegenbauer(l, lambda, 1);
  return end === 0 ? 0 : gegenbauer(l, lambda, x) / end;
}

/**
 * Project a sphere vector onto the first `bands` Gegenbauer harmonics.
 * We treat the vector's components as samples of a signal over equally
 * spaced angles on [-1, 1] (x_k = cos(pi * k / N)) and take the discrete
 * Gegenbauer-weighted inner product with each polynomial degree. This is a
 * Fourier-like transform: low l = coarse/slow structure, high l = fine
 * structure, and the coefficient magnitudes form the memory's spectrum.
 */
export function spectralProject(
  v: Float64Array,
  bands = SPECTRUM_BANDS,
  lambda = GEGENBAUER_LAMBDA
): Float64Array {
  const n = v.length;
  const coeffs = new Float64Array(bands);
  for (let l = 0; l < bands; l++) {
    let sum = 0;
    for (let k = 0; k < n; k++) {
      const x = Math.cos((Math.PI * k) / (n - 1 || 1));
      sum += v[k] * gegenbauerNormalized(l, lambda, x);
    }
    coeffs[l] = sum / n;
  }
  return coeffs;
}
