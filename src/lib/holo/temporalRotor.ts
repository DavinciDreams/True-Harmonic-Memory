// Distributed temporal addressing adapted from HAM to this repository's
// Cl(12,0) geometry.  Six mutually orthogonal bivector planes are available
// without increasing NUM_GENERATORS: e12, e34, e56, e78, e9-10, e11-12.
// Their incommensurate periods span roughly 1.4 logical hours to 6.5 years.
//
// Exact ticks remain authoritative.  Every finite phase code eventually
// recurs, so correlation is a bounded fuzzy address, never an ordering proof.

import { gp, mv, reverse, sandwich, type Multivector } from "./clifford";

const PHI = (1 + Math.sqrt(5)) / 2;
const TWO_PI = 2 * Math.PI;

/** One logical engine tick represents one hour in the temporal experiment. */
export const TEMPORAL_SECONDS_PER_TICK = 3600;

/**
 * Six bands selected from HAM's eight-band distributed rotor.  Cl(12,0)
 * permits six independent commuting planes; retaining the shortest five and
 * longest band gives local resolution plus a multi-year anti-alias signal.
 */
export const DISTRIBUTED_TEMPORAL_PERIODS_TICKS = Object.freeze([
  Math.sqrt(2),
  6 * PHI,
  24 * Math.sqrt(3),
  7 * 24 * Math.sqrt(5),
  29.53059 * 24 * PHI,
  1461 * 24 * PHI,
]);

const TEMPORAL_PLANE_MASKS = Object.freeze([
  0b000000000011, // e12
  0b000000001100, // e34
  0b000000110000, // e56
  0b000011000000, // e78
  0b001100000000, // e9,10
  0b110000000000, // e11,12
]);

export interface DistributedRotorCorrelation {
  coherence: number;
  orientedPhase: number;
  real: number;
  imaginary: number;
}

function phaseAt(tick: number, period: number): number {
  const wrapped = ((tick % period) + period) % period;
  return TWO_PI * (wrapped / period);
}

function planeRotor(theta: number, planeMask: number): Multivector {
  return mv([
    [0, Math.cos(theta / 2)],
    [planeMask, Math.sin(theta / 2)],
  ]);
}

/** Product of six commuting plane rotors: the complete geometric time code. */
export function distributedTimeRotor(tick: number): Multivector {
  let rotor = mv([[0, 1]]);
  for (let i = 0; i < TEMPORAL_PLANE_MASKS.length; i++) {
    rotor = gp(
      rotor,
      planeRotor(
        phaseAt(tick, DISTRIBUTED_TEMPORAL_PERIODS_TICKS[i]),
        TEMPORAL_PLANE_MASKS[i]
      )
    );
  }
  return rotor;
}

/**
 * Bind time by applying each commuting plane sandwich separately.  This is
 * algebraically equivalent to one sandwich by distributedTimeRotor(tick),
 * but avoids materializing a 64-term rotor around every stored blade.
 */
export function rotateWithDistributedTimeRotor(
  tick: number,
  value: Multivector
): Multivector {
  let rotated = value;
  for (let i = 0; i < TEMPORAL_PLANE_MASKS.length; i++) {
    rotated = sandwich(
      planeRotor(
        phaseAt(tick, DISTRIBUTED_TEMPORAL_PERIODS_TICKS[i]),
        TEMPORAL_PLANE_MASKS[i]
      ),
      rotated
    );
  }
  return rotated;
}

/** Undo rotateWithDistributedTimeRotor exactly, modulo floating-point error. */
export function unrotateWithDistributedTimeRotor(
  tick: number,
  value: Multivector
): Multivector {
  let unrotated = value;
  for (let i = TEMPORAL_PLANE_MASKS.length - 1; i >= 0; i--) {
    const rotor = planeRotor(
      phaseAt(tick, DISTRIBUTED_TEMPORAL_PERIODS_TICKS[i]),
      TEMPORAL_PLANE_MASKS[i]
    );
    unrotated = sandwich(reverse(rotor), unrotated);
  }
  return unrotated;
}

/**
 * Observable bind/unbind correlation, equivalent to averaging the six
 * complex relative phases.  This matches HAM's secondary rotor diagnostic,
 * narrowed to the six planes the host Clifford algebra can represent.
 */
export function distributedRotorCorrelation(
  leftTick: number,
  rightTick: number
): DistributedRotorCorrelation {
  let real = 0;
  let imaginary = 0;
  for (const period of DISTRIBUTED_TEMPORAL_PERIODS_TICKS) {
    const delta = phaseAt(rightTick - leftTick, period);
    real += Math.cos(delta);
    imaginary += Math.sin(delta);
  }
  real /= DISTRIBUTED_TEMPORAL_PERIODS_TICKS.length;
  imaginary /= DISTRIBUTED_TEMPORAL_PERIODS_TICKS.length;
  return {
    coherence: Math.hypot(real, imaginary),
    orientedPhase: Math.atan2(imaginary, real),
    real,
    imaginary,
  };
}
