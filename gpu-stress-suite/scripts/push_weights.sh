#!/usr/bin/env bash
# push_weights.sh — push finished quantized weights to S3.
# Run once on the node that quantized them. Idempotent: re-running only uploads what changed.
# Usage: ./push_weights.sh <local_weights_dir>
#   e.g. ./push_weights.sh cache/workload8_model
set -euo pipefail
 
SRC="$1"          # local dir containing the quantized weights
MODEL=$(basename "$SRC")        # model identifier, e.g. workload8_model
BUCKET="tensormachines-benchmark-weights"
 
[ -d "$SRC" ] || { echo "ERROR: $SRC not found"; exit 1; }
 
DEST="s3://$BUCKET/a100/cache/$MODEL/"
 
echo "=== Pushing $SRC -> $DEST ==="
for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  fname=$(basename "$f")
  echo "--- Uploading $fname ---"
  aws s3 cp "$f" "$DEST$fname" \
    --sse AES256 \
    --storage-class STANDARD_IA \
    --no-progress
  echo "--- Done $fname, pausing 60s ---"
  sleep 60
done
# aws s3 sync "$SRC" "$DEST" \
#   --sse AES256 \
#   --storage-class STANDARD_IA \
#   --exclude ".*" \
#   --no-progress \
#   --multipart-chunksize=64MB
echo "=== Done. Manifest: ==="
aws s3 ls "$DEST" --recursive --human-readable --summarize