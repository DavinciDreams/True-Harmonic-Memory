#!/usr/bin/env bash
# Fetches GloVe 50-dim word vectors (Pennington, Socher & Manning, 2014) for
# the optional "real embedding" reference baseline in bench/index.ts — see
# bench/glove.ts. ~171MB, plain-text format (word followed by 50 floats per
# line).
#
# nlp.stanford.edu (the original source) was unreachable from the
# environment this was written in, so this pulls from a community re-upload
# on Hugging Face instead — file size matches the well-known original
# (171,350,079 bytes) closely enough to trust it, but swap this URL for
# https://nlp.stanford.edu/data/glove.6B.zip (extract glove.6B.50d.txt) if
# you'd rather verify against the canonical source.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/glove
if [ ! -f data/glove/glove.6B.50d.txt ]; then
  echo "Downloading GloVe 50d vectors (~171MB)..."
  curl -L --max-time 600 -o data/glove/glove.6B.50d.txt \
    "https://huggingface.co/datasets/antokun/glove.6B.50d/resolve/main/glove.6B.50d.txt"
  echo "Done: bench/data/glove/glove.6B.50d.txt"
else
  echo "bench/data/glove/glove.6B.50d.txt already present, skipping."
fi
