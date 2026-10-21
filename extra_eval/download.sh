#!/bin/bash
# Download Robin3D's released checkpoint, ScanNet Mask3D features and val annotations from
# the authors' Google Drive folder (only the files evaluation needs; see drive_files.tsv).
#   ROBIN3D_DATA=/path/to/robin3d bash extra_eval/download.sh
# Files that already exist are skipped; partial ones are resumed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f env.sh ] && source env.sh
DST="${ROBIN3D_DATA:?set ROBIN3D_DATA}"
while IFS=$'\t' read -r id path; do
    [ -z "$id" ] && continue
    if [ -s "$DST/$path" ]; then echo "have $path"; continue; fi
    mkdir -p "$DST/$(dirname "$path")"
    gdown --continue --retries 5 -O "$DST/$path" "$id" || { echo "FAILED $path"; fail=1; }
done < extra_eval/drive_files.tsv
# Drive sometimes refuses a file for a while ("too many accesses"); just run this again later.
[ "${fail:-0}" = 0 ] && echo DOWNLOAD_DONE || { echo DOWNLOAD_INCOMPLETE; exit 1; }
