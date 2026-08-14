#!/usr/bin/env bash
# Fetches a BEIR retrieval task (part of the MTEB retrieval suite) into
# bench/data/<dataset>. Requires curl + unzip (both present in Git Bash on
# Windows, and on virtually any Linux/macOS shell).
#
# Usage: bash bench/fetch-data.sh [dataset]   (default: nfcorpus, ~2.4MB)
#
# Any BEIR dataset name works (same public mirror for all of them), e.g.:
#   nfcorpus   ~3.6K docs   ~2.4MB   (default)
#   scifact    ~5K docs     ~3MB
#   scidocs    ~25K docs    ~15MB
#   fiqa       ~57K docs    ~19MB
#   quora      ~523K docs   ~76MB
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
DATASET="${1:-nfcorpus}"
if [ ! -d "data/$DATASET" ]; then
  echo "Downloading $DATASET..."
  curl -sL --max-time 300 -o "data/$DATASET.zip" \
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$DATASET.zip"
  (cd data && unzip -o -q "$DATASET.zip" && rm "$DATASET.zip")
  echo "Done: bench/data/$DATASET"
else
  echo "bench/data/$DATASET already present, skipping."
fi
