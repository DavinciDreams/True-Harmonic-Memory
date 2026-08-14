// Bit-level -> sphere mapping.
//
// Each *word* is hashed into a deterministic +-1 bit string on the unit
// hypersphere S^(n-1). A piece of text is the *bundle* (superposition) of
// its words' sphere waves: sum the raw +-1 vectors, then normalize once.
// This is the standard Vector Symbolic Architecture "bundling" trick, and
// it's what makes retrieval by a single keyword ("Q3") resonate with a
// sentence that contains it ("Reviewed the Q3 roadmap deck") — a whole
// separate hash per string would have ~zero bit overlap with a substring.

import { textToBits } from "./random";

// 24 → 256: raising this was, by a wide margin, the single biggest quality
// win found across this whole project (see bench/ results in the README —
// brute-force nDCG@10 went 0.0559 → 0.2161 at this same change, dwarfing
// every retrieval-mechanism fix that came before it). Bundling many words'
// hash vectors into too few dimensions causes severe collision noise; more
// dimensions give words more room to stay distinguishable after summation.
// Diminishing returns set in well before 256 (128→256 was +22% relative,
// 256→512 was +17% at a much higher cost, and HNSW's approximation gap
// starts widening past this point too).
//
// 256 → 384: matches the production/standard embedding width (e.g.
// OpenAI/Cohere-class models commonly ship 384-dim variants) rather than
// the 256 the earlier 128/256/512 sweep above happened to land on. This also
// reopens clifford.ts's Cl(n,0) blade-capacity ceiling (C(n,3) ≤
// SPHERE_DIM) — n=14 fits capacity-wise now, but was tried and reverted;
// see clifford.ts for why NUM_GENERATORS is still 12. Content-only nDCG@10
// (bench/index.ts, NFCorpus): 0.2161 → 0.2361.
export const SPHERE_DIM = 384;

// A small closed-class stopword list. These carry ~no discriminative signal
// but show up in nearly every document, so bundling them in unweighted dilutes
// the words that actually distinguish one text from another. Filtered before
// bundling (not down-weighted) since there's no per-corpus statistic needed
// to know "the" is noise — unlike IDF below, this is a fixed, corpus-free win.
// Exported so bench/glove.ts's real-embedding reference baseline can apply
// the exact same filtering, rather than being handicapped by an unfair
// difference in preprocessing when compared against this encoder.
export const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
  "had", "he", "her", "him", "his", "i", "in", "into", "is", "it", "its", "of",
  "on", "or", "our", "she", "so", "that", "the", "their", "these", "this",
  "those", "to", "was", "we", "were", "will", "with", "you", "your",
]);

function tokenize(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z0-9]+/g);
  if (!words || words.length === 0) return [text || "∅"];
  const filtered = words.filter((w) => !STOPWORDS.has(w));
  // Don't discard everything if the text is nothing but stopwords/short queries.
  return filtered.length > 0 ? filtered : words;
}

/** word -> IDF weight, as returned by buildIdf. */
export type IdfLookup = (word: string) => number;

/**
 * Smoothed IDF over a corpus of texts (same smoothing as scikit-learn's
 * TfidfVectorizer: ln((N+1)/(df+1)) + 1, so every term gets a positive,
 * bounded weight even at df=N). Common words that appear in most documents
 * end up weighted near 1x; rare, distinctive words end up weighted several
 * times higher — this is what a plain hash-bundle embedding is missing
 * relative to real TF-IDF/BM25 style retrieval.
 */
export function buildIdf(texts: string[]): IdfLookup {
  const df = new Map<string, number>();
  for (const text of texts) {
    for (const w of new Set(tokenize(text))) df.set(w, (df.get(w) ?? 0) + 1);
  }
  const n = texts.length;
  return (word: string) => Math.log((n + 1) / ((df.get(word) ?? 0) + 1)) + 1;
}

// Two "more realistic IR" variants were tried here and measurably made
// things worse on NFCorpus, so they were reverted rather than kept for
// realism's own sake:
//   - BM25-style TF saturation (diminishing returns per repeated word)
//     dropped brute-force nDCG@10 from 0.2161 to 0.1700 at this same
//     SPHERE_DIM. In these short title+abstract documents, a word appearing
//     several times really is stronger relevance evidence, not noise to be
//     discounted the way BM25 assumes for longer, padding-prone documents.
//   - Bigrams (adjacent word pairs hashed as extra terms) dropped it further
//     to 0.0630. Adding bigrams roughly doubles the number of unique terms
//     competing to be bundled into the same fixed SPHERE_DIM, which
//     re-introduces exactly the hash-collision crowding that raising
//     SPHERE_DIM (see above) was fixing in the first place.
// Lesson: for this specific bundling scheme, raw dimensionality mattered far
// more than smarter term weighting — simpler beat "more realistic" here.

/** Map text -> unit vector on S^(SPHERE_DIM - 1), bundled from its words. */
export function textToSphereVector(text: string, dim = SPHERE_DIM, idf?: IdfLookup): Float64Array {
  const words = tokenize(text);
  const v = new Float64Array(dim);
  for (const word of words) {
    const bits = textToBits(word, dim);
    const weight = idf ? idf(word) : 1;
    for (let i = 0; i < dim; i++) v[i] += bits[i] * weight;
  }
  let normSq = 0;
  for (let i = 0; i < dim; i++) normSq += v[i] * v[i];
  const norm = Math.sqrt(normSq) || 1;
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

/** Cosine similarity between two points on the hypersphere. */
export function sphereDot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
