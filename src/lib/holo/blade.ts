// Compress a sphere wave down into a Clifford-algebra multivector, for
// binding — but instead of one 16-dim projection shared by everything,
// content and context each get their own *grade* (disjoint blade indices),
// so they're orthogonal channels by construction rather than two signals
// mixed into the same 16 numbers.
//
// NOT what context-bound retrieval is scored against anymore: that
// compression (down to C(NUM_GENERATORS,3) directions) measurably cost
// ~2x nDCG@10 versus HRR circular convolution at full SPHERE_DIM (see
// hrr.ts and the root README's Design notes / Context-bound queries
// sections), which is what engine.ts's retrieve() actually correlates
// context-bound candidates against now. This module's grade-separated
// blades are still real and still used — they drive engine.ts's Clifford
// field and its globalResonance stat — just not the thing that determines
// context-bound retrieval quality.
//
// Content -> grade 1 (the 12 vector blades e1..e12): the "what".
// Context -> grade 3 (the 220 trivector blades e123..e10-11-12): the
// "when/where tagged", chosen specifically because it's algebraically
// disjoint from grade 1 — a content-only query and a context-only query can
// never collide, since they simply don't share any nonzero components.
// Binding (gp(content, context)) then lands in grades |1-3|=2 and 1+3=4 —
// again disjoint from both raw channels — so a superposed field can hold
// pure-content, pure-context, and bound-content-x-context signal
// simultaneously without one drowning out another. Time keeps its existing
// dedicated slot: the e1^e2 bivector (see clifford.ts timeRotor), one of 66
// grade-2 components, so bound content-x-context signal only partially
// overlaps it rather than colliding head-on.
//
// 12 generators, up from an original 4 then 8 — see clifford.ts for the
// full history, including why 14 was tried (SPHERE_DIM=384 made it fit
// capacity-wise) and reverted (real-corpus memory cost wasn't worth a
// noise-level quality gain).
//
// Each channel still uses a fixed random orthonormal projection (not a
// low-degree Gegenbauer truncation) for the same reason as before: word-hash
// vectors are close to white noise, so a directional-bias-free JL-style
// projection preserves relative angles far better than keeping only the
// lowest harmonics.

import { Multivector, gradeIndices, mv, normalize } from "./clifford";
import { randomOrthonormalBasis } from "./linalg";
import { SPHERE_DIM } from "./sphere";

const CONTENT_GRADE = 1;
const CONTEXT_GRADE = 3;

const CONTENT_INDICES = gradeIndices(CONTENT_GRADE);
const CONTEXT_INDICES = gradeIndices(CONTEXT_GRADE);

const CONTENT_BASIS = randomOrthonormalBasis(CONTENT_INDICES.length, SPHERE_DIM, 0xc0111e17);
const CONTEXT_BASIS = randomOrthonormalBasis(CONTEXT_INDICES.length, SPHERE_DIM, 0xc012e17);

function projectOntoGrade(
  v: Float64Array,
  basis: Float64Array[],
  indices: number[]
): Multivector {
  const entries: Array<[number, number]> = [];
  for (let row = 0; row < indices.length; row++) {
    const basisRow = basis[row];
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * basisRow[i];
    entries.push([indices[row], s]);
  }
  return normalize(mv(entries));
}

/** Sphere wave -> pure grade-1 (vector) blade. The "content" channel. */
export function sphereToContentBlade(v: Float64Array): Multivector {
  return projectOntoGrade(v, CONTENT_BASIS, CONTENT_INDICES);
}

/** Sphere wave -> pure grade-3 (trivector) blade. The "context" channel. */
export function sphereToContextBlade(v: Float64Array): Multivector {
  return projectOntoGrade(v, CONTEXT_BASIS, CONTEXT_INDICES);
}
