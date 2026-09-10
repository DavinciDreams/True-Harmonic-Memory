// Loader for BEIR retrieval tasks (part of the MTEB retrieval suite) —
// used unmodified. Despite the filename (kept for git history / import
// stability), this loads any BEIR dataset, not just NFCorpus — pass the
// dataset name (e.g. "nfcorpus", "fiqa", "scidocs") to each function.
//
// Fetch a dataset first with: pnpm bench:fetch [dataset] (see package.json;
// default dataset is nfcorpus).

import { createReadStream, existsSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";

export function dataDir(dataset: string): string {
  return path.join(__dirname, "data", dataset);
}

export interface Doc {
  id: string;
  text: string; // title + body, concatenated
}

export interface Query {
  id: string;
  text: string;
}

/** query-id -> corpus-id -> graded relevance score. */
export type Qrels = Map<string, Map<string, number>>;

async function readJsonl<T>(
  file: string,
  map: (obj: Record<string, unknown>) => T
): Promise<T[]> {
  const out: T[] = [];
  const rl = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    out.push(map(JSON.parse(line)));
  }
  return out;
}

export function assertDataPresent(dataset: string) {
  if (!existsSync(dataDir(dataset))) {
    throw new Error(
      `${dataset} data not found at ${dataDir(dataset)}.\n` +
        `Fetch it first: pnpm bench:fetch ${dataset} (downloads from the public BEIR dataset mirror).`
    );
  }
}

export async function loadCorpus(dataset: string): Promise<Doc[]> {
  return readJsonl(path.join(dataDir(dataset), "corpus.jsonl"), (o) => ({
    id: String(o._id),
    text: [o.title, o.text]
      .filter((part): part is string => typeof part === "string" && part.length > 0)
      .join(". "),
  }));
}

export async function loadQueries(dataset: string): Promise<Query[]> {
  return readJsonl(path.join(dataDir(dataset), "queries.jsonl"), (o) => ({
    id: String(o._id),
    text: String(o.text),
  }));
}

export async function loadQrels(
  dataset: string,
  split: "test" | "dev" | "train" = "test"
): Promise<Qrels> {
  const file = path.join(dataDir(dataset), "qrels", `${split}.tsv`);
  const qrels: Qrels = new Map();
  const rl = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
  let first = true;
  for await (const line of rl) {
    if (first) {
      first = false;
      continue; // header row: query-id  corpus-id  score
    }
    if (!line.trim()) continue;
    const [qid, did, scoreStr] = line.split("\t");
    const score = Number(scoreStr);
    if (!qrels.has(qid)) qrels.set(qid, new Map());
    qrels.get(qid)!.set(did, score);
  }
  return qrels;
}
