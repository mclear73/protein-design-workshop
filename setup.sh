#!/bin/bash
# =============================================================================
# setup.sh — Install ColabDesign + ColabFold + AF2 + Streamlit workshop UI
# =============================================================================
# Target: AWS Deep Learning AMI (Ubuntu 20.04/22.04, PyTorch 2.x)
# Instance: g5.xlarge (A10G) recommended; g4dn.xlarge (T4) works but slower
# Expected runtime: 15-25 min on first install (env + first colabfold_batch
# call which downloads ~3.5 GB of AF2 weights + MSA cache prewarm)
#
# This script provisions one conda environment:
#
#   colabfold — ColabFold 1.5.5 + ColabDesign-from-master + Streamlit + py3Dmol
#               Used by the workshop UI in streamlit_ui/app.py, which uses
#               the ColabFold MSA-based AF2 pipeline (much higher pLDDT than
#               ColabDesign's single-sequence default).
#
# A note on dependencies:
# - This env runs jax 0.6+, which broke ColabDesign v1.1.3 (deprecated
#   jax.tree_map). Master has the fix. So we install ColabDesign from git's
#   main branch, NOT a pinned release.
# - Streamlit + py3Dmol are pip-installed alongside, so the UI can drive
#   ColabFold directly without env switching.
#
# The repo also contains two Jupyter notebooks (ProteinMPNN_Workshop_Teams,
# RFdiffusion_Industrial_Demo). They use a different env (SE3nv) that this
# script no longer installs — to run them, see the RFdiffusion repo's setup
# instructions or check this file's git history before commit <will-fill-in>.
#
# Usage:
#   ./setup.sh                    # full install (env + prewarm; ~15-25 min)
#   ./setup.sh --no-prewarm       # skip MSA cache prewarm (~10 min server hit)
#   ./setup.sh --check            # verify an existing install, no changes
# =============================================================================

set -e
set -o pipefail

# Argument parsing
DO_PREWARM=1
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-prewarm)     DO_PREWARM=0 ;;
        --check)          CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '2,/^# =====/p' "$0" | sed 's/^# //;s/^#$//'
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg (use --help)"
            exit 1
            ;;
    esac
done

# Log everything to a timestamped file for post-mortem debugging
LOG_FILE="$HOME/setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Resolve the repo root (= dir containing this script). Used to find
# streamlit_ui/ regardless of where the user cd'd before running setup.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Pinned versions (update these after testing a new combination end-to-end)
# -----------------------------------------------------------------------------
# ColabDesign master is required because pinned releases (v1.1.3 and earlier)
# use deprecated jax.tree_map which jax 0.6+ removed. If a future release
# stabilizes, pin to a release tag here.
COLABDESIGN_VERSION="main"
COLABFOLD_PIN="1.5.5"          # last tested version

# Name of the conda environment this script manages
ENV_CFOLD="colabfold"

# -----------------------------------------------------------------------------
# Locate conda (DLAMI typically has it at /opt/conda or ~/miniconda3)
# -----------------------------------------------------------------------------
locate_conda() {
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        CONDA_ROOT=/opt/conda
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_ROOT="$HOME/anaconda3"
    else
        echo "❌ Conda not found. Are you on a Deep Learning AMI?"
        echo "   Install Miniforge first: https://github.com/conda-forge/miniforge"
        exit 1
    fi
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    echo "✅ Conda: $CONDA_ROOT"
}

# -----------------------------------------------------------------------------
# Sanity check used by --check and at end of install.
# -----------------------------------------------------------------------------
run_sanity_check_colabfold() {
    echo ""
    echo "--- Sanity check: $ENV_CFOLD ---"
    python - <<'PY'
import sys, os, shutil, subprocess
ok = True

try:
    import torch
    print(f"  ✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False

try:
    import streamlit
    print(f"  ✅ streamlit {streamlit.__version__}")
except Exception as e:
    print(f"  ❌ streamlit: {e}"); ok = False

try:
    import py3Dmol
    print("  ✅ py3Dmol")
except Exception as e:
    print(f"  ❌ py3Dmol: {e}"); ok = False

try:
    import numpy
    print(f"  ✅ numpy {numpy.__version__}")
except Exception as e:
    print(f"  ❌ numpy: {e}"); ok = False

try:
    from colabdesign.mpnn import mk_mpnn_model  # noqa
    print("  ✅ colabdesign.mpnn (from master)")
except Exception as e:
    print(f"  ❌ colabdesign.mpnn: {e}"); ok = False

# colabfold_batch must be on PATH (we use it as a CLI subprocess from msa_cache.py)
if shutil.which("colabfold_batch"):
    print("  ✅ colabfold_batch on PATH")
else:
    print("  ❌ colabfold_batch not found on PATH")
    ok = False

# Streamlit UI files
ui_files = ["app.py", "msa_cache.py", "setup_msa_cache.sh"]
ui_dir = os.environ.get("WORKSHOP_UI_DIR", os.path.expanduser("~/protein-design-workshop/streamlit_ui"))
for f in ui_files:
    p = os.path.join(ui_dir, f)
    if os.path.isfile(p):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} not found at {p}")
        ok = False

# MSA cache directory should exist and ideally have entries
cache_dir = os.path.expanduser("~/msa_cache")
if os.path.isdir(cache_dir):
    a3m_files = [f for f in os.listdir(cache_dir) if f.endswith(".a3m")]
    if a3m_files:
        print(f"  ✅ MSA cache: {len(a3m_files)} entries in {cache_dir}")
    else:
        print(f"  ⚠️  MSA cache dir exists but is empty (run setup_msa_cache.sh)")
else:
    print(f"  ⚠️  MSA cache dir not found at {cache_dir} (run setup_msa_cache.sh)")

sys.exit(0 if ok else 1)
PY
}

# -----------------------------------------------------------------------------
# --check mode: just verify, then exit
# -----------------------------------------------------------------------------
if [ "$CHECK_ONLY" = "1" ]; then
    echo "=============================================="
    echo "  Verifying existing install (--check mode)"
    echo "=============================================="
    locate_conda

    overall_ok=1

    if conda env list | grep -qE "^${ENV_CFOLD}\s"; then
        conda activate "$ENV_CFOLD"
        export WORKSHOP_UI_DIR="$REPO_ROOT/streamlit_ui"
        if ! run_sanity_check_colabfold; then overall_ok=0; fi
        conda deactivate
    else
        echo "⚠️  $ENV_CFOLD environment does not exist (Streamlit UI won't work)"
        overall_ok=0
    fi

    echo ""
    if [ "$overall_ok" = "1" ]; then
        echo "✅ All checks passed. Instance is workshop-ready."
        exit 0
    else
        echo "❌ Some checks failed. Re-run setup.sh (without --check) to repair."
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "  Workshop install (Streamlit UI)"
echo "  Log: $LOG_FILE"
echo "=============================================="
echo ""
echo "Pinned versions:"
echo "  ColabDesign:  $COLABDESIGN_VERSION"
echo "  ColabFold:    $COLABFOLD_PIN"
echo ""

locate_conda

# -----------------------------------------------------------------------------
# Fix the DLAMI's broken channel configuration
# -----------------------------------------------------------------------------
# The Deep Learning AMI ships with an AWS-internal S3 channel
# (aws-ml-conda-ec2.s3.us-west-2.amazonaws.com) configured by default.
# This channel returns 403 Forbidden from public-subnet instances and breaks
# `conda env create`. We strip it and ensure conda-forge is present.
# -----------------------------------------------------------------------------
echo ""
echo "--- Cleaning up conda channels ---"

# Remove any aws-ml-conda channels (they commonly 403 from public subnets)
while conda config --show channels 2>/dev/null | grep -q "aws-ml-conda"; do
    broken=$(conda config --show channels | grep -m1 aws-ml-conda | sed 's/^[[:space:]]*-[[:space:]]*//')
    echo "  Removing broken channel: $broken"
    conda config --remove channels "$broken" 2>/dev/null || break
done

# Make sure conda-forge is in the list (required for SE3nv deps)
if ! conda config --show channels 2>/dev/null | grep -q "conda-forge"; then
    echo "  Adding conda-forge"
    conda config --add channels conda-forge
fi

# Flexible priority — required for SE3nv's cross-channel pinning
conda config --set channel_priority flexible

echo "  Final channel list:"
conda config --show channels | sed 's/^/    /'

# -----------------------------------------------------------------------------
# Pick the fastest available env solver (mamba > libmamba > conda)
# -----------------------------------------------------------------------------
echo ""
echo "--- Selecting fastest available package manager ---"

SOLVER_CMD=""
if command -v mamba &>/dev/null; then
    SOLVER_CMD="mamba"
    echo "✅ mamba already installed — using it"
elif conda config --set solver libmamba 2>/dev/null && \
     conda config --show solver 2>/dev/null | grep -q libmamba; then
    SOLVER_CMD="conda"
    echo "✅ libmamba solver enabled for conda (fast, no extra install)"
else
    echo "⏳ Installing mamba for fast dependency resolution..."
    conda install -n base -c conda-forge mamba -y -q
    SOLVER_CMD="mamba"
    echo "✅ mamba installed"
fi


# -----------------------------------------------------------------------------
# Create colabfold conda env from a heredoc'd yaml
# -----------------------------------------------------------------------------
# The env spec installs ColabFold via pip-from-git (with [alphafold] extras),
# plus jax[cuda12] and tensorflow as dependencies. This recipe is the one
# documented at https://github.com/sokrypton/ColabFold and on Isambard-AI's
# install guide (both verified 2026-05-02).
#
# We do NOT pin colabfold via conda's bioconda channel because that build is
# often behind master. Pinning happens in the pip line via the extras spec.
# -----------------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_CFOLD}\s"; then
    echo ""
    echo "--- Creating env $ENV_CFOLD with $SOLVER_CMD (typically 3-5 min) ---"

    CFOLD_YAML="$(mktemp /tmp/colabfold-env.XXXXXX.yml)"
    cat > "$CFOLD_YAML" <<EOF
name: $ENV_CFOLD
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.10
  - pip
  - mmseqs2
  - pip:
    - jax[cuda12]
    - tensorflow
    - "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold@v$COLABFOLD_PIN"
    - streamlit
    - py3Dmol
    - numpy
EOF
    $SOLVER_CMD env create -f "$CFOLD_YAML"
    rm -f "$CFOLD_YAML"
else
    echo "✅ Env $ENV_CFOLD already exists"
fi

conda activate "$ENV_CFOLD"
echo "✅ Activated: $ENV_CFOLD"

# -----------------------------------------------------------------------------
# Install ColabDesign from MASTER into the colabfold env
# -----------------------------------------------------------------------------
# The pinned v1.1.3 release uses the deprecated jax.tree_map API, which jax
# 0.6.x (installed by jax[cuda12] above) removed in favor of jax.tree.map.
# Master has the fix. So this env needs ColabDesign from git's main branch,
# not the same pin used in SE3nv.
# -----------------------------------------------------------------------------
if ! python -c "from colabdesign.mpnn import mk_mpnn_model" 2>/dev/null; then
    echo ""
    echo "--- Installing ColabDesign ($COLABDESIGN_VERSION) into $ENV_CFOLD ---"
    pip install --quiet "git+https://github.com/sokrypton/ColabDesign.git@$COLABDESIGN_VERSION"
fi

# -----------------------------------------------------------------------------
# Verify the streamlit_ui/ files are in place
# -----------------------------------------------------------------------------
# These should be committed to the repo. If we're missing any, fail loudly
# rather than silently — without them the prewarm step will fail anyway.
# -----------------------------------------------------------------------------
UI_DIR="$REPO_ROOT/streamlit_ui"
echo ""
echo "--- Checking streamlit_ui/ files ---"
missing_files=0
for f in app.py msa_cache.py setup_msa_cache.sh; do
    if [ -f "$UI_DIR/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f missing from $UI_DIR/"
        missing_files=1
    fi
done

if [ "$missing_files" = "1" ]; then
    echo ""
    echo "❌ Streamlit UI files are missing. Make sure your repo includes:"
    echo "     $UI_DIR/app.py"
    echo "     $UI_DIR/msa_cache.py"
    echo "     $UI_DIR/setup_msa_cache.sh"
    echo "   (Either commit them, or pull the latest from the repo.)"
    exit 1
fi

# Make setup_msa_cache.sh executable in case git didn't preserve the flag
chmod +x "$UI_DIR/setup_msa_cache.sh"

# -----------------------------------------------------------------------------
# Convenience launcher: ./start_app.sh
# -----------------------------------------------------------------------------
# Activates colabfold env and launches streamlit on 127.0.0.1:8501. Generated
# rather than committed because it has to bake in CONDA_ROOT, which may
# differ between AMIs.
# -----------------------------------------------------------------------------
START_APP="$REPO_ROOT/start_app.sh"
echo ""
echo "--- Generating start_app.sh launcher ---"
cat > "$START_APP" <<EOF
#!/bin/bash
# Generated by setup.sh — launches the Streamlit workshop UI.
#
# This file is generated per-instance (CONDA_ROOT and paths are baked in).
# Don't commit it to the repo — add 'start_app.sh' to .gitignore instead.
# Re-run setup.sh to regenerate after changing conda location.
set -e
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate $ENV_CFOLD
cd "$UI_DIR"
exec streamlit run app.py --server.port 8501 --server.address 127.0.0.1
EOF
chmod +x "$START_APP"
echo "  ✅ Wrote $START_APP"

# Best-effort: add to .gitignore if one exists in the repo
GITIGNORE="$REPO_ROOT/.gitignore"
if [ -f "$GITIGNORE" ] && ! grep -qE "^/?start_app\.sh\$" "$GITIGNORE"; then
    echo "start_app.sh" >> "$GITIGNORE"
    echo "  ✅ Added start_app.sh to .gitignore"
fi

# -----------------------------------------------------------------------------
# Prewarm the MSA cache (unless --no-prewarm)
# -----------------------------------------------------------------------------
# This invokes setup_msa_cache.sh which:
#   - Downloads each target's input PDB (PETase, ZAR1, CarRP)
#   - Extracts the WT sequence using app.py's own logic (so cache keys match
#     what the app generates at runtime)
#   - Calls colabfold_batch with --msa-only to get the MSA from the ColabFold
#     server, saving it to ~/msa_cache/<key>.a3m
# Idempotent: existing cache entries are skipped.
# -----------------------------------------------------------------------------
if [ "$DO_PREWARM" = "1" ]; then
    echo ""
    echo "--- Prewarming MSA cache (one-shot, ~10 min) ---"
    echo "    Skip with --no-prewarm if rate-limited or offline."

    if "$UI_DIR/setup_msa_cache.sh"; then
        echo "✅ MSA cache prewarm complete"
    else
        echo "⚠️  Prewarm failed (likely rate-limit or network). Cache will populate"
        echo "    on first use of each target in the app, or you can re-run:"
        echo "      $UI_DIR/setup_msa_cache.sh"
    fi
else
    echo ""
    echo "⏭  Skipping MSA cache prewarm (--no-prewarm). To run later:"
    echo "    $UI_DIR/setup_msa_cache.sh"
fi

# Streamlit stack done
conda deactivate

# =============================================================================
# Final sanity check
# =============================================================================

conda activate "$ENV_CFOLD"
export WORKSHOP_UI_DIR="$UI_DIR"
run_sanity_check_colabfold || true
conda deactivate

echo ""
echo "=============================================="
echo "  Install complete."
echo "  Log saved to: $LOG_FILE"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Launch the app:"
echo "       $REPO_ROOT/start_app.sh"
echo ""
echo "  2. From your laptop, open an SSH tunnel:"
echo "       ssh -L 8501:localhost:8501 -i <key.pem> ubuntu@<instance-ip>"
echo ""
echo "  3. Browse to http://localhost:8501"
echo ""
echo "  --- Verify a working install later ---"
echo ""
echo "       ./setup.sh --check"
echo ""
