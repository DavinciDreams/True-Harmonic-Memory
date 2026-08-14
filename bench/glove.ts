// GloVe (Pennington, Socher & Manning, 2014) word-vector loader + a simple
// averaged-vector text encoder — used ONLY as a 4th benchmark reference
// point, never wired into src/lib/holo. This project's core engine is
// explicitly "no embeddings, no LLM" (see the root README); this file
// exists purely to answer "how much does embedding quality matter, versus
// retrieval mechanism" as a reference data point, by deliberately breaking
// the "same embedding across methods" isolation the other three baselines
// share. Fetch the vectors first: pnpm bench:fetch:glove (~171MB).

import { createReadStream, existsSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { IdfLookup, STOPWORDS } from "../src/lib/holo/sphere";

export const GLOVE_DIM = 50;

const GLOVE_PATH = path.join(__dirname, "data", "glove", "glove.6B.50d.txt");

export function gloveAvailable(): boolean {
  return existsSync(GLOVE_PATH);
}

/** word -> 50-dim GloVe vector, ~400K-word vocabulary. */
export async function loadGlove(): Promise<Map<string, Float64Array>> {
  const vocab = new Map<string, Float64Array>();
  const rl = createInterface({ input: createReadStream(GLOVE_PATH), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    const parts = line.split(" ");
    const word = parts[0];
    const vec = new Float64Array(GLOVE_DIM);
    for (let i = 0; i < GLOVE_DIM; i++) vec[i] = Number(parts[i + 1]);
    vocab.set(word, vec);
  }
  return vocab;
}

// Same tokenization + stopword filtering as sphere.ts's encoder (imported
// STOPWORDS, not a separate list) — otherwise this baseline would be
// handicapped by dumping unfiltered function words into a plain average,
// which is a preprocessing-fairness bug, not evidence about embedding
// quality.
function tokenize(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z0-9]+/g);
  if (!words || words.length === 0) return [text || "∅"];
  const filtered = words.filter((w) => !STOPWORDS.has(w));
  return filtered.length > 0 ? filtered : words;
}

/**
 * Map text -> unit vector, as the IDF-weighted mean of its words' GloVe
 * vectors ("averaged word embeddings" — the simplest standard way to turn
 * word vectors into a document/sentence vector). Plain unweighted averaging
 * is a well-known weak baseline in real IR — it lets common words dilute
 * distinctive ones — so this applies the *same* IDF weighting
 * (src/lib/holo/sphere.ts's buildIdf) the other three benchmark rows get,
 * for a fair "embedding quality, other things equal" comparison. Words
 * outside GloVe's ~400K vocabulary are skipped (out-of-vocabulary) — worth
 * checking OOV rate on a specialized-vocabulary corpus (see the root
 * README) before trusting this as a general "real embeddings win" signal.
 */
export function textToGloveVector(
  text: string,
  vocab: Map<string, Float64Array>,
  idf?: IdfLookup
): Float64Array {
  const v = new Float64Array(GLOVE_DIM);
  let count = 0;
  for (const word of tokenize(text)) {
    const wv = vocab.get(word);
    if (!wv) continue;
    const weight = idf ? idf(word) : 1;
    for (let i = 0; i < GLOVE_DIM; i++) v[i] += wv[i] * weight;
    count++;
  }
  if (count > 0) for (let i = 0; i < GLOVE_DIM; i++) v[i] /= count;
  let normSq = 0;
  for (let i = 0; i < GLOVE_DIM; i++) normSq += v[i] * v[i];
  const norm = Math.sqrt(normSq) || 1;
  for (let i = 0; i < GLOVE_DIM; i++) v[i] /= norm;
  return v;
}
