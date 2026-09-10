"use client";

import { useMemo, useState } from "react";
import {
  EXAMPLE_MEMORIES,
  HoloStore,
  MemoryRecord,
  RetrievalResult,
  encode,
  generateBulkMemories,
  seedMemories,
} from "@/lib/holo/engine";
import SphereScene, { ScenePoint } from "./SphereScene";
import SpectrumChart from "./SpectrumChart";

const TIER_LABEL: Record<MemoryRecord["tier"], string> = {
  recent: "Recent",
  repeated: "Repeated",
  "long-term": "Long-term",
};

const TIER_DOT: Record<MemoryRecord["tier"], string> = {
  recent: "bg-sky-400",
  repeated: "bg-amber-400",
  "long-term": "bg-violet-400",
};

// A handful of texts (not already in the seed set) offered as one-click
// "try this" chips, so testing retrieval doesn't require typing.
const QUICK_ADD_EXAMPLES = [
  "Presented the demo to the whole team",
  "Lost my keys again",
  "Late-night refactor of the binding logic",
  "Birthday dinner with the family",
  "Long flight, watched three movies",
];

export default function HoloApp() {
  const [store] = useState<HoloStore>(() => {
    const s = new HoloStore();
    seedMemories(s);
    return s;
  });

  const [, bump] = useState(0);
  const rerender = () => bump((v) => v + 1);

  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const [queryText, setQueryText] = useState("Morning coffee ritual");
  const [queryContext, setQueryContext] = useState("");
  const [sigma, setSigma] = useState(1.1);
  const [temporal, setTemporal] = useState(false);

  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [queryPoint, setQueryPoint] = useState<[number, number, number] | null>(null);
  const [inspected, setInspected] = useState<MemoryRecord | null>(null);
  const [searchMs, setSearchMs] = useState<number | null>(null);
  const [bulkAddMs, setBulkAddMs] = useState<{ count: number; ms: number } | null>(null);

  const allActive = [...store.recent, ...store.repeated, ...store.longTerm];

  const scenePoints: ScenePoint[] = useMemo(() => {
    const peakIds = new Map(result?.peaks.map((p) => [p.record.id, p]) ?? []);
    const maxScore = Math.max(...(result?.peaks.map((p) => p.score) ?? [1]), 1e-6);
    return allActive.map((r) => {
      const peak = peakIds.get(r.id);
      return {
        id: r.id,
        point3d: r.point3d,
        tier: r.tier,
        weight: r.weight,
        isPeak: !!peak,
        peakStrength: peak ? peak.score / maxScore : 0,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.clock, result, store.recent.length, store.repeated.length, store.longTerm.length]);

  function runRetrieve(qText: string, qContext: string, s: number) {
    const trimmed = qText.trim();
    if (!trimmed) {
      setResult(null);
      setQueryPoint(null);
      setSearchMs(null);
      return;
    }
    const start = performance.now();
    const r = store.retrieve(trimmed, qContext.trim(), { sigma: s, temporal });
    setSearchMs(performance.now() - start);
    setResult(r);
    setQueryPoint(encode(trimmed).point3d);
    if (r.peaks[0]) setInspected(r.peaks[0].record);
  }

  function addMemory(t: string, c: string) {
    if (!t.trim()) return;
    const r = store.addMemory(t.trim(), c.trim());
    setInspected(r);
    rerender();
  }

  function handleAdd() {
    addMemory(text, context);
    setText("");
    setContext("");
  }

  function handleRemove(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    store.removeMemory(id);
    if (inspected?.id === id) setInspected(null);
    rerender();
  }

  function handleTick() {
    store.tick();
    rerender();
  }

  function handleSeed() {
    seedMemories(store);
    rerender();
  }

  function handleBulkAdd(count: number) {
    const batch = generateBulkMemories(count, store.clock);
    const start = performance.now();
    for (const [t, c] of batch) store.addMemory(t, c);
    const ms = performance.now() - start;
    setBulkAddMs({ count, ms });
    rerender();
  }

  const fieldMagnitude = store.fieldMagnitude();
  const closest = result?.peaks[0] ?? null;

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-semibold tracking-tight">True Harmonic Memory</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Text is hashed onto a unit hypersphere, decomposed into a Gegenbauer harmonic
          spectrum, bound to context via a Clifford geometric product with timestamps encoded as
          phase rotors, and superposed into a single field. Retrieval takes one global inner
          product against that field, then peaks are extracted from a Gaussian phase window.{" "}
          <span className="text-slate-300">
            No embeddings, no LLM — just a deterministic hash → sphere → spectrum → algebra
            pipeline.
          </span>
        </p>
      </header>

      <div className="grid flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[300px_1fr_340px]">
        {/* Left: tiers */}
        <div className="flex flex-col gap-3 overflow-y-auto">
          <Panel title="Consolidation tiers">
            <div className="flex flex-col gap-3">
              {(["recent", "repeated", "long-term"] as const).map((tier) => {
                const list = allActive.filter((r) => r.tier === tier);
                return (
                  <div key={tier}>
                    <div className="mb-1 flex items-center gap-2 text-xs font-medium text-slate-400">
                      <span className={`h-2 w-2 rounded-full ${TIER_DOT[tier]}`} />
                      {TIER_LABEL[tier]} ({list.length})
                    </div>
                    <ul className="flex flex-col gap-1">
                      {list.map((r) => (
                        <li key={`${tier}-${r.id}`}>
                          <button
                            onClick={() => setInspected(r)}
                            className={`group relative w-full rounded-md border px-2 py-1.5 pr-6 text-left text-xs transition ${
                              inspected?.id === r.id
                                ? "border-sky-500 bg-sky-500/10"
                                : "border-slate-800 bg-slate-900 hover:border-slate-700"
                            }`}
                          >
                            <div className="truncate text-slate-200">{r.text}</div>
                            <div className="mt-0.5 flex items-center justify-between text-[10px] text-slate-500">
                              <span>{r.context || "no context"}</span>
                              <span>
                                w={r.weight.toFixed(2)} x{r.repeatCount + 1}
                              </span>
                            </div>
                            <span
                              role="button"
                              tabIndex={0}
                              onClick={(e) => handleRemove(r.id, e)}
                              title="Remove this memory"
                              className="absolute right-1 top-1 rounded px-1 text-slate-600 opacity-0 hover:bg-red-500/20 hover:text-red-300 group-hover:opacity-100"
                            >
                              ×
                            </span>
                          </button>
                        </li>
                      ))}
                      {list.length === 0 && (
                        <li className="text-[11px] text-slate-600">empty</li>
                      )}
                    </ul>
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel title="Add a memory">
            <div className="flex flex-col gap-2">
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="What happened..."
                className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1.5 text-sm outline-none focus:border-sky-500"
              />
              <input
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="context / tag (optional)"
                className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1.5 text-sm outline-none focus:border-sky-500"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="flex-1 rounded-md bg-sky-600 px-2 py-1.5 text-sm font-medium hover:bg-sky-500"
                >
                  Bind + superpose
                </button>
                <button
                  onClick={handleTick}
                  className="rounded-md border border-slate-700 px-2 py-1.5 text-sm hover:border-slate-500"
                  title="Advance logical time; recent/repeated memories decay"
                >
                  Tick
                </button>
              </div>
              <button
                onClick={handleSeed}
                className="rounded-md border border-slate-800 px-2 py-1.5 text-xs text-slate-400 hover:border-slate-600"
              >
                Load sample memories
              </button>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {QUICK_ADD_EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => addMemory(ex, "")}
                    title="Click to add this example memory"
                    className="rounded-full border border-slate-800 px-2 py-1 text-[10px] text-slate-400 hover:border-sky-600 hover:text-sky-300"
                  >
                    + {ex}
                  </button>
                ))}
              </div>
              <p className="mt-1 text-[10px] leading-snug text-slate-600">
                {EXAMPLE_MEMORIES.length} texts live in the seed set; add your own or click a
                chip above to test resonance and retrieval.
              </p>
            </div>
          </Panel>

          <Panel title="Stress test">
            <div className="flex flex-col gap-2">
              <div className="flex gap-2">
                <button
                  onClick={() => handleBulkAdd(500)}
                  className="flex-1 rounded-md border border-emerald-700/60 bg-emerald-500/10 px-2 py-1.5 text-xs font-medium text-emerald-200 hover:border-emerald-500"
                >
                  Add 500 memories
                </button>
                <button
                  onClick={() => handleBulkAdd(2000)}
                  className="rounded-md border border-slate-700 px-2 py-1.5 text-xs text-slate-300 hover:border-slate-500"
                >
                  +2000
                </button>
              </div>
              {bulkAddMs && (
                <p className="text-[10px] text-slate-500">
                  Bound + superposed {bulkAddMs.count} memories in{" "}
                  <span className="font-mono text-emerald-300">{bulkAddMs.ms.toFixed(1)} ms</span>{" "}
                  ({(bulkAddMs.ms / bulkAddMs.count).toFixed(3)} ms/item). The field stays a fixed
                  8-number multivector regardless of count — that&apos;s the point.
                </p>
              )}
            </div>
          </Panel>
        </div>

        {/* Center: 3D scene */}
        <div className="flex flex-col gap-2">
          <Panel title="Sphere field" className="relative flex-1 min-h-[420px] overflow-hidden p-0">
            <SphereScene points={scenePoints} fieldMagnitude={fieldMagnitude} queryPoint={queryPoint} />
            <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1 rounded-md border border-slate-800/80 bg-slate-950/70 px-2.5 py-2 text-[10px] text-slate-300 backdrop-blur-sm">
              <Legend color="bg-sky-400" label="Recent" />
              <Legend color="bg-amber-400" label="Repeated" />
              <Legend color="bg-violet-400" label="Long-term" />
              <Legend color="bg-yellow-300" label="Peak / query link" />
              <Legend color="bg-red-400" label="Query point" />
            </div>
            <div className="pointer-events-none absolute bottom-2 right-3 text-[10px] text-slate-500">
              drag to rotate · scroll to zoom
            </div>
          </Panel>
          <Panel title="Field state">
            <div className="flex items-center gap-4 text-sm">
              <Stat label="tick" value={store.clock} />
              <Stat label="|field|" value={fieldMagnitude.toFixed(3)} />
              <Stat label="items" value={allActive.length} />
              {result && (
                <Stat
                  label="global resonance"
                  value={result.globalResonance.toFixed(3)}
                  highlight
                />
              )}
            </div>
          </Panel>
        </div>

        {/* Right: query + inspector */}
        <div className="flex flex-col gap-3 overflow-y-auto">
          <Panel title="Search / retrieve">
            <div className="flex flex-col gap-2">
              <input
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runRetrieve(queryText, queryContext, sigma)}
                placeholder="search..."
                className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1.5 text-sm outline-none focus:border-sky-500"
              />
              <input
                value={queryContext}
                onChange={(e) => setQueryContext(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runRetrieve(queryText, queryContext, sigma)}
                placeholder="context (optional)"
                className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1.5 text-sm outline-none focus:border-sky-500"
              />
              <label className="flex items-center gap-2 text-xs text-slate-400">
                <input
                  type="checkbox"
                  checked={temporal}
                  onChange={(e) => setTemporal(e.target.checked)}
                />
                Weight by temporal phase window
              </label>
              <label
                className={`flex items-center gap-2 text-xs ${temporal ? "text-slate-400" : "text-slate-600"}`}
              >
                Gaussian window &sigma;
                <input
                  type="range"
                  min={0.2}
                  max={3}
                  step={0.05}
                  value={sigma}
                  disabled={!temporal}
                  onChange={(e) => setSigma(parseFloat(e.target.value))}
                  className="flex-1 disabled:opacity-40"
                />
                <span className="w-8 text-right">{sigma.toFixed(2)}</span>
              </label>
              <p className="text-[10px] leading-snug text-slate-600">
                Off: ranked purely by content resonance. On: also weighted by how close each
                memory&apos;s time-phase is to now — recent/matching-era memories are boosted,
                which can outrank a weaker keyword match (that&apos;s the tradeoff being demoed).
              </p>
              <button
                onClick={() => runRetrieve(queryText, queryContext, sigma)}
                className="rounded-md bg-violet-600 px-2 py-1.5 text-sm font-medium hover:bg-violet-500"
              >
                Search
              </button>
              {searchMs !== null && (
                <p className="text-[10px] text-slate-500">
                  Correlated against {allActive.length} tracked memories in{" "}
                  <span className="font-mono text-violet-300">{searchMs.toFixed(2)} ms</span>.
                </p>
              )}
            </div>
          </Panel>

          <Panel title="Closest match">
            {!closest && (
              <p className="text-xs text-slate-500">
                {queryText.trim() ? "No resonant peak found." : "Type a query above."}
              </p>
            )}
            {closest && (
              <button
                onClick={() => setInspected(closest.record)}
                className="w-full rounded-md border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-2 text-left"
              >
                <div className="text-sm text-emerald-100">{closest.record.text}</div>
                <div className="mt-1 flex items-center justify-between text-[10px] text-emerald-300/80">
                  <span>{closest.record.context || "no context"}</span>
                  <span className={`rounded px-1 ${TIER_DOT[closest.record.tier]} text-slate-950`}>
                    {TIER_LABEL[closest.record.tier]}
                  </span>
                </div>
                <div className="mt-1 flex justify-between text-[10px] text-emerald-300/70">
                  <span>similarity {closest.similarity.toFixed(3)}</span>
                  <span>score {closest.score.toFixed(3)}</span>
                </div>
              </button>
            )}
          </Panel>

          <Panel title="Other peaks">
            {result && result.peaks.length <= 1 && (
              <p className="text-xs text-slate-500">No other peaks above threshold.</p>
            )}
            <ul className="flex flex-col gap-1.5">
              {result?.peaks.slice(1).map((p) => (
                <li key={`peak-${p.record.id}`}>
                  <button
                    onClick={() => setInspected(p.record)}
                    className="w-full rounded-md border border-amber-500/30 bg-amber-500/5 px-2 py-1.5 text-left text-xs hover:border-amber-500/60"
                  >
                    <div className="truncate text-slate-200">{p.record.text}</div>
                    <div className="mt-0.5 flex justify-between text-[10px] text-amber-300/80">
                      <span>sim {p.similarity.toFixed(2)}</span>
                      <span>phase&nbsp;wt {p.phaseWeight.toFixed(2)}</span>
                      <span>score {p.score.toFixed(2)}</span>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>

          <Panel title={inspected ? `Spectrum · ${inspected.text.slice(0, 24)}` : "Spectrum"}>
            {inspected ? (
              <SpectrumChart spectrum={inspected.spectrum} />
            ) : (
              <p className="text-xs text-slate-500">Select a memory to inspect its Gegenbauer spectrum.</p>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Panel({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-slate-800 bg-slate-900/60 p-3 ${className}`}>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className={`font-mono text-sm ${highlight ? "text-violet-300" : "text-slate-200"}`}>
        {value}
      </div>
    </div>
  );
}
