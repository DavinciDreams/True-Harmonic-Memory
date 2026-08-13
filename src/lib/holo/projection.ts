// Fixed random projection SPHERE_DIM -> 3, orthonormalized once at module
// load via a deterministic seed. Used only for visualization: it lets every
// 24-dim sphere wave map to a stable, reproducible point in 3D so the scene
// doesn't jitter between renders.

import { randomOrthonormalBasis } from "./linalg";
import { SPHERE_DIM } from "./sphere";

const PROJECTION = randomOrthonormalBasis(3, SPHERE_DIM, 0xc0ffee);

/** Project a SPHERE_DIM unit vector down to a 3D point (unit-ish length). */
export function projectTo3D(v: Float64Array): [number, number, number] {
  let x = 0,
    y = 0,
    z = 0;
  for (let i = 0; i < v.length; i++) {
    x += PROJECTION[0][i] * v[i];
    y += PROJECTION[1][i] * v[i];
    z += PROJECTION[2][i] * v[i];
  }
  const len = Math.sqrt(x * x + y * y + z * z) || 1;
  return [x / len, y / len, z / len];
}
