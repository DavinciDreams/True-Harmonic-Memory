// Clifford (geometric) algebra Cl(12,0): basis blades built from 12
// orthonormal generators e1..e12, all squaring to +1.
//
// Blades are addressed by bitmask over generators {e1=bit0, e2=bit1, ...,
// e12=bit11}, which is the standard trick for implementing a geometric
// product generically: the product of two blades is (sign) * (xor of
// bitmasks), where the sign is the parity of transpositions needed to sort
// the concatenated generator list (metric here is Euclidean, so repeated
// generators contribute +1, no extra sign). This is what lets us rotate
// (rotors) and superpose (addition) memories in one small, exact algebra —
// geometric product (`gp()`) is still used for that (see engine.ts's
// Clifford field / globalResonance stat), but no longer for the binding
// context-bound retrieval is actually scored against; see hrr.ts.
//
// SPARSE representation (Map<bitmask, coefficient>), not a dense
// Float64Array of all 2^n components: a dense array was the representation
// through Cl(4,0)/Cl(8,0), but it hard-caps how large n can practically be
// on two axes at once — memory (2^n floats *per multivector*, and every
// stored record keeps several) and compute (gp()'s nested loop had to sweep
// all 2^n indices even to skip zeros). The blades actually used here
// (grades 1, 3, and whatever a product of those lands in) number in the
// hundreds at n=12, not 2^12=4096 — sparse storage means both memory and
// gp() cost track that real occupied-blade count instead of the full
// power-of-two, letting n grow well past where dense storage would already
// have been infeasible.
//
// n=12: a capacity ceiling from blade.ts's basis construction (the context
// channel needs C(n,3) mutually orthonormal directions in SPHERE_DIM
// dimensions, and you cannot have more than SPHERE_DIM orthonormal vectors
// in a SPHERE_DIM-dimensional space), but also a *practical* one, measured
// directly rather than assumed: with SPHERE_DIM raised to 384 (see
// sphere.ts), C(13,3)=286 and C(14,3)=364 both fit under 384, so n=14 was
// tried — reachable in principle, but at Cl(14,0) each stored record's
// bound-family blades (grades 2+4, C(14,2)+C(14,4)=1092 slots each) got
// large enough that indexing SCIDOCS (~25.7K docs) OOM'd Node's default
// ~4GB heap, even after the near-zero-coefficient pruning below and
// dropping MemoryRecord's dead `bound` field. It only needed 12GB+ to
// complete, for a context-bound nDCG@10 gain over n=12 of 0.0595→0.0601 —
// ~1%, within noise. Not worth the fragility; reverted to 12. Re-derive
// this number (and re-measure at real corpus scale, not just quality) if
// SPHERE_DIM changes again.

export const NUM_GENERATORS = 12;
export const BLADE_COUNT = 1 << NUM_GENERATORS; // 4096 — the full space size; never allocated directly (see above)

/** Sparse multivector: bitmask -> nonzero coefficient. Absent key = 0. */
export type Multivector = Map<number, number>;

export function mv(entries?: Iterable<readonly [number, number]>): Multivector {
  return entries ? new Map(entries) : new Map();
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    c += x & 1;
    x >>= 1;
  }
  return c;
}

/** Sign picked up by reordering generators when multiplying blades a * b. */
function productSign(a: number, b: number): number {
  let sum = 0;
  for (let i = 0; i < NUM_GENERATORS; i++) {
    if ((b >> i) & 1) {
      sum += popcount(a >> (i + 1));
    }
  }
  return sum % 2 === 0 ? 1 : -1;
}

// Below this magnitude, a coefficient is float noise, not signal — see
// prune()'s doc comment for why that noise accumulates here specifically.
const EPSILON = 1e-9;

/**
 * Coefficients that algebraically cancel to exactly 0 rarely land on exactly
 * 0 in floating point — gp() sums several av*bv terms per output index (see
 * below), so a cancellation like 0.7071...*0.7071... - 0.7071...*0.7071...
 * typically leaves a ~1e-17 residue instead of 0. sandwich() (two chained
 * gp() calls) is the worst offender: rotor conjugation is grade-preserving
 * in exact arithmetic (an established Clifford-algebra theorem — see
 * blade.ts's grade-disjointness comment for why that matters here), but the
 * *intermediate* gp(r, a) is not grade-pure, so the second gp() multiplies
 * against those extra grades and their cancellation-to-zero is exactly this
 * kind of near-zero float residue, not a true zero the Map ever drops.
 * Measured effect on a real record's `rotated` blade at Cl(14,0): 1588
 * stored entries, 496 of them (~31%) below 1e-9, largest of those ~3.5e-18 —
 * pure noise, no signal, but every one of them still costs a Map slot and
 * gets iterated by every later add()/innerProduct()/magnitude() call. Pruned
 * once here (gp()'s output) rather than at every downstream call site.
 */
function prune(m: Multivector): Multivector {
  for (const [i, v] of m) {
    if (Math.abs(v) < EPSILON) m.delete(i);
  }
  return m;
}

/** Geometric product of two multivectors — O(nnz(a) * nnz(b)), not O(2^n). */
export function gp(a: Multivector, b: Multivector): Multivector {
  const out = new Map<number, number>();
  for (const [ai, av] of a) {
    if (av === 0) continue;
    for (const [bi, bv] of b) {
      if (bv === 0) continue;
      const bits = ai ^ bi;
      const sign = productSign(ai, bi);
      out.set(bits, (out.get(bits) ?? 0) + sign * av * bv);
    }
  }
  return prune(out);
}

export function add(a: Multivector, b: Multivector): Multivector {
  const out = new Map(a);
  for (const [i, v] of b) out.set(i, (out.get(i) ?? 0) + v);
  return out;
}

export function sub(a: Multivector, b: Multivector): Multivector {
  const out = new Map(a);
  for (const [i, v] of b) out.set(i, (out.get(i) ?? 0) - v);
  return out;
}

export function scale(a: Multivector, s: number): Multivector {
  const out = new Map<number, number>();
  for (const [i, v] of a) out.set(i, v * s);
  return out;
}

/** Reverse (~A): grade-k blade picks up (-1)^(k(k-1)/2). Acts as conjugation. */
export function reverse(a: Multivector): Multivector {
  const out = new Map<number, number>();
  for (const [i, v] of a) {
    const k = popcount(i);
    const sign = (k * (k - 1)) / 2;
    out.set(i, sign % 2 === 0 ? v : -v);
  }
  return out;
}

/**
 * Scalar (grade-0) part of A * ~B: the Hermitian-style inner product.
 * Algebraically equals a plain component-wise dot product — gp(a,
 * reverse(b))[0] only gets contributions from index pairs (i, i) (any i≠j
 * pair XORs to something nonzero, i.e. not grade 0), and for those pairs
 * the geometric-product self-reordering sign productSign(i,i) is *always*
 * equal to reverse()'s sign(i) by construction (both come from the same
 * "how many transpositions to reorder i's generators against themselves"
 * count), so the two sign flips (one from reverse(), one from productSign)
 * cancel: sum_i a[i] * sign(i) * (sign(i) * b[i]) = sum_i a[i] * b[i].
 * Implemented directly as that dot product rather than a full gp() + index
 * lookup — cheaper (O(min(nnz(a), nnz(b))) instead of O(nnz(a) * nnz(b))),
 * and this is the single hottest call in retrieve()'s per-record scoring.
 */
export function innerProduct(a: Multivector, b: Multivector): number {
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let sum = 0;
  for (const [i, v] of small) {
    const bv = large.get(i);
    if (bv !== undefined) sum += v * bv;
  }
  return sum;
}

export function magnitude(a: Multivector): number {
  let s = 0;
  for (const v of a.values()) s += v * v;
  return Math.sqrt(s);
}

export function normalize(a: Multivector): Multivector {
  const m = magnitude(a) || 1;
  return scale(a, 1 / m);
}

/** Rotor cos(theta/2) + sin(theta/2) * e12, used to bind timestamps as phase. */
export function timeRotor(theta: number): Multivector {
  return mv([
    [0, Math.cos(theta / 2)], // scalar
    [3, Math.sin(theta / 2)], // e12 bivector (bits 0b0011 = 3)
  ]);
}

/** Sandwich product R A ~R: rotates A by the rotor R. */
export function sandwich(r: Multivector, a: Multivector): Multivector {
  return gp(gp(r, a), reverse(r));
}

/** Grade (0..NUM_GENERATORS) of a blade index: how many generators it's built from. */
export function gradeOf(bitmask: number): number {
  return popcount(bitmask);
}

/**
 * All blade indices belonging to a given grade, e.g. gradeIndices(1) =
 * [e1,e2,...]. Sweeps all 2^NUM_GENERATORS bitmasks once — fine as a
 * one-time cost at module load (blade.ts's basis construction), unlike the
 * old dense Multivector representation which paid a 2^n cost on every
 * single algebra operation.
 */
export function gradeIndices(grade: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < BLADE_COUNT; i++) {
    if (popcount(i) === grade) out.push(i);
  }
  return out;
}

/** Keep only the components of `a` at a given grade, zeroing everything else. */
export function gradeProject(a: Multivector, grade: number): Multivector {
  const out = new Map<number, number>();
  for (const [i, v] of a) {
    if (popcount(i) === grade) out.set(i, v);
  }
  return out;
}

/** Human-readable label for a single blade index, e.g. bladeLabel(3) = "e12". */
export function bladeLabel(bitmask: number): string {
  if (bitmask === 0) return "1";
  const gens: number[] = [];
  for (let b = 0; b < NUM_GENERATORS; b++) if (bitmask & (1 << b)) gens.push(b + 1);
  return "e" + gens.join("");
}
