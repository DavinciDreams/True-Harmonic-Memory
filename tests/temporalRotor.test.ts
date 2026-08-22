import assert from "node:assert/strict";
import test from "node:test";

import { gp, magnitude, mv, normalize, type Multivector } from "../src/lib/holo/clifford";
import { HoloStore } from "../src/lib/holo/engine";
import {
  distributedRotorCorrelation,
  distributedTimeRotor,
  rotateWithDistributedTimeRotor,
  unrotateWithDistributedTimeRotor,
} from "../src/lib/holo/temporalRotor";

function maxCoefficientError(left: Multivector, right: Multivector): number {
  const keys = new Set([...left.keys(), ...right.keys()]);
  let error = 0;
  for (const key of keys) {
    error = Math.max(error, Math.abs((left.get(key) ?? 0) - (right.get(key) ?? 0)));
  }
  return error;
}

test("distributed rotor is identity-correlated and orientation reverses", () => {
  const identity = distributedRotorCorrelation(1234, 1234);
  const forward = distributedRotorCorrelation(1234, 1241);
  const backward = distributedRotorCorrelation(1241, 1234);

  assert.equal(identity.coherence, 1);
  assert.equal(identity.orientedPhase, 0);
  assert.ok(Math.abs(forward.coherence - backward.coherence) < 1e-12);
  assert.ok(Math.abs(forward.orientedPhase + backward.orientedPhase) < 1e-12);
});

test("distributed rotor breaks the native exact 24-tick alias", () => {
  const alias = distributedRotorCorrelation(0, 24);
  assert.ok(alias.coherence < 0.5, `24-tick coherence was ${alias.coherence}`);
  assert.ok(distributedRotorCorrelation(0, 0).coherence > alias.coherence);
});

test("six-plane geometric time binding can be unbound", () => {
  const value = normalize(mv([
    [5, 0.25],
    [18, -0.75],
    [96, 0.5],
    [513, 0.125],
  ]));
  const bound = rotateWithDistributedTimeRotor(731.25, value);
  const restored = unrotateWithDistributedTimeRotor(731.25, bound);

  assert.ok(maxCoefficientError(value, restored) < 1e-9);
  assert.ok(Math.abs(magnitude(bound) - magnitude(value)) < 1e-9);
});

test("materialized distributed rotor remains unit magnitude", () => {
  const rotor = distributedTimeRotor(9876.5);
  assert.ok(Math.abs(magnitude(rotor) - 1) < 1e-9);
  assert.ok(gp(rotor, rotor).size > 1);
});

test("HoloStore binds and unbinds the field probe at the requested time", () => {
  const store = new HoloStore({ temporalAddressing: "distributed" });
  const record = store.addBulk("deployed the temporal rotor", "release");
  const matching = store.retrieve("deployed the temporal rotor", "release", {
    temporal: true,
    temporalAddressing: "distributed",
    centerTick: record.createdAtTick,
    globalResonance: true,
  });
  const nativeAlias = store.retrieve("deployed the temporal rotor", "release", {
    temporal: true,
    temporalAddressing: "distributed",
    centerTick: record.createdAtTick + 24,
    globalResonance: true,
  });

  assert.ok(matching.globalResonance > nativeAlias.globalResonance);
  assert.ok(matching.globalResonance > 0.99);
});
