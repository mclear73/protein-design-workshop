#!/bin/bash
# setup_msa_cache.sh — pre-warm the MSA cache for the three workshop targets.
#
# Runs once during instance setup. Hits the ColabFold MSA server three times
# (one per target) and populates ~/msa_cache/. Once this completes, the
# in-workshop ColabFold validation runs are MSA-server-free for these targets.
#
# Idempotent: skips targets whose cache already exists. Safe to re-run after a
# stop/start (cache lives on the persistent volume).
#
# Usage:
#   ./setup_msa_cache.sh
#
# CRITICAL: this script drives the prewarm through the SAME code path the
# Streamlit app uses at runtime — download_target() to fetch and truncate the
# PDB, extract_wt_sequence() to read the sequence from the resulting ATOM
# records. This guarantees the cache key matches what the app produces, so
# the cache hit actually happens at workshop time. Slicing UniProt directly
# (which an earlier version of this script did) doesn't match because PDB
# structures often resolve a slightly different range than what's annotated
# in UniProt.

set -euo pipefail

# Resolve SCRIPT_DIR FIRST, before any cd, so BASH_SOURCE[0] still points at
# the script's actual location rather than wherever we cd to next.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for required in msa_cache.py app.py; do
    if [[ ! -f "${SCRIPT_DIR}/${required}" ]]; then
        echo "ERROR: ${required} not found next to setup_msa_cache.sh" >&2
        echo "  Expected: ${SCRIPT_DIR}/${required}" >&2
        exit 1
    fi
done

CACHE_DIR="${HOME}/msa_cache"
mkdir -p "${CACHE_DIR}"

# Run all of this from SCRIPT_DIR so download_target writes its inputs/ next
# to app.py, just like Streamlit does at runtime.
cd "${SCRIPT_DIR}"

echo "Pre-warming MSA cache for the three workshop targets."
echo "Driving through the app's own download_target + extract_wt_sequence,"
echo "so the cache keys match what Streamlit produces at runtime."
echo ""

# -----------------------------------------------------------------------------
# Drive the prewarm via the app's own functions
# -----------------------------------------------------------------------------
# Run python from this directory so `import app` resolves (app.py is here).
# Streamlit isn't actually invoked — we just call its module-level functions.

python3 - <<'PYEOF'
import os
import sys
sys.path.insert(0, os.getcwd())

# We need to import app.py without triggering its Streamlit page setup. The
# st.set_page_config call at module load is fine — Streamlit lets it no-op
# when there's no script run context.
import app
import msa_cache

for team in app.TARGETS.keys():
    print(f"\n=== {team} ===")

    # 1. Download + truncate the target PDB exactly like the app does
    pdb_path, n_res = app.download_target(team)
    print(f"  PDB:        {pdb_path}")
    print(f"  Residues:   {n_res}")

    # 2. Extract the sequence the same way the app does
    chain = app.TARGETS[team]["chain"]
    seq = app.extract_wt_sequence(pdb_path, chain)
    print(f"  Sequence:   {len(seq)} aa")

    # 3. Compute the cache key — must match what the app will compute
    key = msa_cache.cache_key(team, seq)
    print(f"  Cache key:  {key}")

    # 4. Generate or skip
    if msa_cache.cache_exists(key):
        print(f"  Status:     already cached, skipping")
        continue

    print(f"  Status:     generating MSA via ColabFold server...")
    key = msa_cache.generate_msa(team, seq, source="prewarm")
    print(f"  a3m:        {msa_cache.cache_path(key)}")

print("\n=== Done ===")
PYEOF

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "Cache directory: ${CACHE_DIR}"
ls -lh "${CACHE_DIR}"/*.a3m 2>/dev/null | awk '{print "  " $5 "  " $NF}'
echo ""
echo "Workshop validation runs will now reuse these MSAs without hitting the server."
