// Small shared linear-algebra helpers: Gram-Schmidt orthonormalization and
// building a fixed random orthonormal basis from a seed.

import { mulberry32 } from "./random";

export function gramSchmidt(rows: Float64Array[]): Float64Array[] {
  const out: Float64Array[] = [];
  for (const row of rows) {
    const v = row.slice();
    for (const u of out) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) dot += v[i] * u[i];
      for (let i = 0; i < v.length; i++) v[i] -= dot * u[i];
    }
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    out.push(v);
  }
  return out;
}

/** `rows` orthonormal vectors of length `dim`, deterministic from `seed`. */
export function randomOrthonormalBasis(rows: number, dim: number, seed: number): Float64Array[] {
  const rng = mulberry32(seed);
  const raw: Float64Array[] = [];
  for (let r = 0; r < rows; r++) {
    const row = new Float64Array(dim);
    for (let i = 0; i < dim; i++) row[i] = rng() * 2 - 1;
    raw.push(row);
  }
  return gramSchmidt(raw);
}
