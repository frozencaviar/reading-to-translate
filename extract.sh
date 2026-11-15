#!/usr/bin/env bash
set -euo pipefail
tables=$(cd "${1:?Usage: bash extract.sh /path/to/Tables}" && pwd)
cd "$(dirname "$0")"
out=attention-norm-yiyu-positive-dur
mkdir -p data "$out"
python generate_translog.py --critt_dir "$tables" --output data/sessions.json
for pair in token-segment segment-token sentence-sentence; do
    file="$out/yiyu-features-$pair.tsv"
    python src/nmt_attention.py --session_file data/sessions.json --source_table_dir "$tables" \
        --outputf "$file" --src_level "${pair%-*}" --tgt_level "${pair#*-}" \
        --bsz 8 --num_workers 0 --normalize --max
    if ! awk 'END {exit (NR < 2)}' "$file"; then
        echo "No feature rows in $file. Check the session IDs and table layout." >&2
        exit 1
    fi
done
