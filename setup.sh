#!/bin/bash
# =============================================================================
# setup.sh — Install RFdiffusion + ColabDesign + AF2 + Streamlit/ColabFold UI
# =============================================================================
# Target: AWS Deep Learning AMI (Ubuntu 20.04/22.04, PyTorch 2.x)
# Instance: g5.xlarge (A10G) recommended; g4dn.xlarge (T4) works but slower
# Expected runtime: 25-40 min on first install (notebooks + streamlit + prewarm)
#
# This script provisions TWO independent conda environments:
#
#   1. SE3nv     — RFdiffusion + older ColabDesign + AF2-via-ColabDesign
#                  Used by the Jupyter notebooks (ProteinMPNN_Workshop_Teams,
#                  RFdiffusion_Industrial_Demo).
#
#   2. colabfold — ColabFold 1.5.5 + ColabDesign-from-master + Streamlit + py3Dmol
#                  Used by the Streamlit UI in streamlit_ui/app.py, which uses
#                  the ColabFold MSA-based AF2 pipeline (much higher pLDDT than
#                  ColabDesign's single-sequence default).
#
# Why two envs:
# - SE3nv is pinned to ColabDesign v1.1.3 because RFdiffusion's env constrains
#   jax to an older version where v1.1.3 works.
# - The colabfold env runs jax 0.6+, which broke ColabDesign v1.1.3 (deprecated
#   jax.tree_map). Master has the fix. So this env needs ColabDesign from git.
# - Streamlit + py3Dmol live in colabfold so the UI can use ColabFold directly.
#
# Usage:
#   ./setup.sh                    # full install (notebooks + streamlit + prewarm)
#   ./setup.sh --no-prewarm       # skip MSA cache prewarm (~10 min server hit)
#   ./setup.sh --check            # verify an existing install, no changes
#   ./setup.sh --streamlit-only   # only install/repair the colabfold env
# =============================================================================

set -e
set -o pipefail

# Argument parsing
DO_PREWARM=1
CHECK_ONLY=0
STREAMLIT_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-prewarm)     DO_PREWARM=0 ;;
        --check)          CHECK_ONLY=1 ;;
        --streamlit-only) STREAMLIT_ONLY=1 ;;
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
# RFdiffusion commit pinned to a known-good state. If you need to update,
# test the full pipeline against the new commit before changing this.
RFDIFFUSION_COMMIT="b44206a2a79f219bb1a649ea50603a284c225050"  # main as of Nov 2024
COLABDESIGN_VERSION_SE3NV="v1.1.3"     # works with SE3nv's older jax
COLABDESIGN_VERSION_CFOLD="main"        # master required for jax 0.6+ (v1.1.3 broken)
COLABFOLD_PIN="1.5.5"                   # last tested version

# Minimum file sizes (bytes) for partial-download detection.
# RFdiffusion checkpoints range from ~200 MB to ~700 MB; 100 MB is a safe floor.
MIN_WEIGHT_SIZE=100000000      # 100 MB
MIN_AF2_TAR_SIZE=3000000000    # 3 GB (tarball is ~3.5 GB)

# Names of conda environments this script manages
ENV_SE3NV="SE3nv"
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
# Sanity checks — one per env. Used by --check and at end of install.
# -----------------------------------------------------------------------------
run_sanity_check_se3nv() {
    echo ""
    echo "--- Sanity check: $ENV_SE3NV ---"
    python - <<'PY'
import sys, os
ok = True
try:
    import torch
    print(f"  ✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  CUDA not available — RFdiffusion/AF2 will not work")
        ok = False
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False

try:
    from colabdesign.mpnn import mk_mpnn_model
    from colabdesign.af import mk_af_model
    print("  ✅ colabdesign (MPNN + AF2)")
except Exception as e:
    print(f"  ❌ colabdesign: {e}"); ok = False

try:
    import py3Dmol
    print("  ✅ py3Dmol")
except Exception as e:
    print(f"  ❌ py3Dmol: {e}"); ok = False

# Check RFdiffusion weights with size validation
weights_dir = os.path.expanduser("~/RFdiffusion/models")
required_weights = ["Base_ckpt.pt", "Complex_base_ckpt.pt", "ActiveSite_ckpt.pt"]
for w in required_weights:
    p = os.path.join(weights_dir, w)
    if os.path.isfile(p) and os.path.getsize(p) > 100_000_000:
        print(f"  ✅ RFdiffusion weight: {w} ({os.path.getsize(p) // 1_000_000} MB)")
    else:
        print(f"  ❌ RFdiffusion weight missing or truncated: {w}")
        ok = False

# AF2 params — search recursively rather than hardcoding the doubled path
import glob
af2_params = glob.glob(os.path.expanduser("~/params/**/params_model_1.npz"), recursive=True)
if af2_params:
    print(f"  ✅ AlphaFold2 params: {af2_params[0]}")
else:
    print("  ❌ AF2 params missing")
    ok = False

sys.exit(0 if ok else 1)
PY
}

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

    # SE3nv
    if conda env list | grep -qE "^${ENV_SE3NV}\s"; then
        conda activate "$ENV_SE3NV"
        if ! run_sanity_check_se3nv; then overall_ok=0; fi
        conda deactivate
    else
        echo "⚠️  $ENV_SE3NV environment does not exist (notebooks won't work)"
        overall_ok=0
    fi

    # colabfold
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
echo "  Workshop install (notebooks + Streamlit UI)"
echo "  Log: $LOG_FILE"
echo "=============================================="
echo ""
echo "Pinned versions:"
echo "  RFdiffusion commit:           $RFDIFFUSION_COMMIT"
echo "  ColabDesign (SE3nv):          $COLABDESIGN_VERSION_SE3NV"
echo "  ColabDesign (colabfold env):  $COLABDESIGN_VERSION_CFOLD"
echo "  ColabFold:                    $COLABFOLD_PIN"
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

# =============================================================================
# PART 1 — Notebook stack (SE3nv env, RFdiffusion, AF2 params)
# =============================================================================
# Skipped if --streamlit-only.
# =============================================================================

if [ "$STREAMLIT_ONLY" = "0" ]; then

# -----------------------------------------------------------------------------
# Clone RFdiffusion (pinned to a specific commit)
# -----------------------------------------------------------------------------
cd "$HOME"
if [ ! -d "RFdiffusion" ]; then
    echo ""
    echo "--- Cloning RFdiffusion @ $RFDIFFUSION_COMMIT ---"
    git clone https://github.com/RosettaCommons/RFdiffusion.git
    cd RFdiffusion
    git checkout "$RFDIFFUSION_COMMIT"
    cd ..
else
    # If repo exists but is on a different commit, warn (don't auto-rewrite —
    # they may have local changes from a previous install).
    cd RFdiffusion
    current=$(git rev-parse HEAD)
    if [ "$current" != "$RFDIFFUSION_COMMIT" ]; then
        echo "⚠️  RFdiffusion is at $current, expected $RFDIFFUSION_COMMIT"
        echo "    Run 'cd ~/RFdiffusion && git checkout $RFDIFFUSION_COMMIT' to align."
    fi
    cd ..
fi

# -----------------------------------------------------------------------------
# Create SE3nv conda env using the fast solver
# -----------------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_SE3NV}\s"; then
    echo ""
    echo "--- Creating env $ENV_SE3NV with $SOLVER_CMD (typically 1-3 min) ---"
    cd "$HOME/RFdiffusion"
    if [ ! -f env/SE3nv.yml ]; then
        echo "❌ env/SE3nv.yml not found — RFdiffusion repo layout may have changed"
        exit 1
    fi
    $SOLVER_CMD env create -f env/SE3nv.yml
else
    echo "✅ Env $ENV_SE3NV already exists"
fi

conda activate "$ENV_SE3NV"
echo "✅ Activated: $ENV_SE3NV"

# -----------------------------------------------------------------------------
# Install SE3Transformer (RFdiffusion's custom module)
# -----------------------------------------------------------------------------
cd "$HOME/RFdiffusion/env/SE3Transformer"
if ! python -c "import se3_transformer" 2>/dev/null; then
    echo ""
    echo "--- Installing SE3Transformer ---"
    pip install --no-cache-dir -r requirements.txt
    python setup.py install
fi

# Install RFdiffusion itself as a package
cd "$HOME/RFdiffusion"
if ! python -c "import rfdiffusion" 2>/dev/null; then
    echo ""
    echo "--- Installing RFdiffusion package ---"
    pip install -e .
fi

# -----------------------------------------------------------------------------
# Download RFdiffusion model weights (~2 GB total)
# Uses HTTPS, validates file sizes, retries on failure.
# -----------------------------------------------------------------------------
mkdir -p "$HOME/RFdiffusion/models"
cd "$HOME/RFdiffusion/models"

# Use indexed array of "name|url" pairs for deterministic ordering.
WEIGHTS=(
    "Base_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"
    "Complex_base_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    "Complex_Fold_base_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt"
    "InpaintSeq_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt"
    "InpaintSeq_Fold_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt"
    "ActiveSite_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt"
    "Base_epoch8_ckpt.pt|https://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt"
)

echo ""
echo "--- Downloading RFdiffusion weights (~2 GB total) ---"
for entry in "${WEIGHTS[@]}"; do
    name="${entry%%|*}"
    url="${entry#*|}"

    # Re-download if file is missing OR smaller than the minimum size
    # (catches partial downloads from interrupted connections).
    if [ -f "$name" ]; then
        size=$(stat -c%s "$name")
        if [ "$size" -ge "$MIN_WEIGHT_SIZE" ]; then
            echo "  $name (already present, $((size / 1000000)) MB)"
            continue
        else
            echo "  $name (truncated at $((size / 1000000)) MB — re-downloading)"
            rm -f "$name"
        fi
    fi

    echo "  $name (downloading from $url)..."
    if ! wget --tries=3 --timeout=120 --show-progress "$url"; then
        # Try the http:// fallback if https fails (some IPD mirrors are flaky)
        echo "  HTTPS failed, trying HTTP..."
        http_url="${url/https:/http:}"
        wget --tries=3 --timeout=120 --show-progress "$http_url"
    fi

    # Verify the download succeeded with the right size
    if [ ! -f "$name" ] || [ "$(stat -c%s "$name")" -lt "$MIN_WEIGHT_SIZE" ]; then
        echo "❌ Failed to download $name (or file is truncated)"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Install ColabDesign (wraps ProteinMPNN and AF2) into SE3nv
# -----------------------------------------------------------------------------
if ! python -c "import colabdesign" 2>/dev/null; then
    echo ""
    echo "--- Installing ColabDesign $COLABDESIGN_VERSION_SE3NV (SE3nv) ---"
    pip install "git+https://github.com/sokrypton/ColabDesign.git@$COLABDESIGN_VERSION_SE3NV"
fi

# -----------------------------------------------------------------------------
# Install extras: py3Dmol, JupyterLab, ipywidgets
# -----------------------------------------------------------------------------
echo ""
echo "--- Installing extras (py3Dmol, JupyterLab) into SE3nv ---"
pip install -q py3Dmol jupyterlab ipywidgets

# Register SE3nv as a Jupyter kernel
python -m ipykernel install --user --name "$ENV_SE3NV" --display-name "Python 3 ($ENV_SE3NV)"

# -----------------------------------------------------------------------------
# Download AlphaFold2 parameters (~3.5 GB)
# Used by ColabDesign's AF2 in BOTH envs — share one copy under ~/params.
# -----------------------------------------------------------------------------
AF2_PARAM_FILE=$(find "$HOME/params" -name "params_model_1.npz" 2>/dev/null | head -1)
if [ -z "$AF2_PARAM_FILE" ]; then
    echo ""
    echo "--- Downloading AlphaFold2 params (~3.5 GB) ---"
    mkdir -p "$HOME/params"
    cd "$HOME/params"

    # Validate the tarball download with a size check.
    if [ -f "alphafold_params_2022-12-06.tar" ]; then
        size=$(stat -c%s "alphafold_params_2022-12-06.tar")
        if [ "$size" -lt "$MIN_AF2_TAR_SIZE" ]; then
            echo "  Tarball is truncated at $((size / 1000000)) MB — re-downloading"
            rm -f "alphafold_params_2022-12-06.tar"
        fi
    fi

    if [ ! -f "alphafold_params_2022-12-06.tar" ]; then
        wget --tries=3 --timeout=300 --show-progress \
            https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    fi

    # Verify before extracting
    size=$(stat -c%s "alphafold_params_2022-12-06.tar")
    if [ "$size" -lt "$MIN_AF2_TAR_SIZE" ]; then
        echo "❌ AF2 params tarball is only $((size / 1000000)) MB — download failed"
        exit 1
    fi
    tar -xf alphafold_params_2022-12-06.tar
    rm -f alphafold_params_2022-12-06.tar
else
    echo "✅ AF2 params already present at $AF2_PARAM_FILE"
fi

# Notebook stack done — deactivate before moving to colabfold env work
conda deactivate

fi  # end of "if STREAMLIT_ONLY = 0"

# =============================================================================
# PART 2 — Streamlit UI stack (colabfold env, ColabDesign master, streamlit)
# =============================================================================

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
    echo "--- Installing ColabDesign ($COLABDESIGN_VERSION_CFOLD) into $ENV_CFOLD ---"
    pip install --quiet "git+https://github.com/sokrypton/ColabDesign.git@$COLABDESIGN_VERSION_CFOLD"
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
# Final sanity checks
# =============================================================================

if [ "$STREAMLIT_ONLY" = "0" ]; then
    conda activate "$ENV_SE3NV"
    run_sanity_check_se3nv || true
    conda deactivate
fi

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
echo "  --- For the Streamlit UI (recommended for the workshop) ---"
echo ""
echo "  1. Launch the app:"
echo "       $REPO_ROOT/start_app.sh"
echo ""
echo "  2. From your laptop, open an SSH tunnel:"
echo "       ssh -L 8501:localhost:8501 -i <key.pem> ubuntu@<instance-ip>"
echo ""
echo "  3. Browse to http://localhost:8501"
echo ""
echo "  --- For the Jupyter notebooks (alternative interface) ---"
echo ""
echo "  1. On the AWS instance, start JupyterLab:"
echo "       conda activate $ENV_SE3NV"
echo "       jupyter lab --no-browser --ip=127.0.0.1 --port=8888"
echo ""
echo "  2. From your laptop, open an SSH tunnel:"
echo "       ssh -L 8888:localhost:8888 -i <key.pem> ubuntu@<instance-ip>"
echo ""
echo "  3. Paste the Jupyter URL into your browser; pick 'Python 3 ($ENV_SE3NV)' kernel."
echo ""
echo "  --- Verify a working install later ---"
echo ""
echo "       ./setup.sh --check"
echo ""
