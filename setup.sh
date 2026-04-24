#!/bin/bash
# =============================================================================
# setup.sh — Install RFdiffusion + ColabDesign + AF2 on AWS GPU instance
# =============================================================================
# Target: AWS Deep Learning AMI (Ubuntu 20.04/22.04, PyTorch 2.x)
# Instance: g5.xlarge (A10G) recommended; g4dn.xlarge (T4) works but slower
# Expected runtime: 10-15 min on first install (was 20-30 min with conda)
#
# Uses mamba for fast dependency resolution (5-10x faster than plain conda).
# Automatically repairs the DLAMI's broken AWS-internal conda channel config.
# =============================================================================

set -e  # exit on any error
set -o pipefail

echo ""
echo "=============================================="
echo "  RFdiffusion + ColabDesign install (mamba)"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Locate conda (DLAMI typically has it at /opt/conda or ~/miniconda3)
# -----------------------------------------------------------------------------
if   [ -f /opt/conda/etc/profile.d/conda.sh ]; then
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
# Clone RFdiffusion
# -----------------------------------------------------------------------------
cd "$HOME"
if [ ! -d "RFdiffusion" ]; then
    echo ""
    echo "--- Cloning RFdiffusion ---"
    git clone https://github.com/RosettaCommons/RFdiffusion.git
fi

# -----------------------------------------------------------------------------
# Create conda env using the fast solver
# -----------------------------------------------------------------------------
if ! conda env list | grep -qE "^SE3nv\s"; then
    echo ""
    echo "--- Creating env SE3nv with $SOLVER_CMD (typically 1-3 min) ---"
    cd "$HOME/RFdiffusion"
    if [ ! -f env/SE3nv.yml ]; then
        echo "❌ env/SE3nv.yml not found — RFdiffusion repo layout may have changed"
        exit 1
    fi
    $SOLVER_CMD env create -f env/SE3nv.yml
else
    echo "✅ Env SE3nv already exists"
fi

conda activate SE3nv
echo "✅ Activated: SE3nv"

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
# -----------------------------------------------------------------------------
mkdir -p "$HOME/RFdiffusion/models"
cd "$HOME/RFdiffusion/models"

declare -A WEIGHTS=(
    ["Base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"
    ["Complex_base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    ["Complex_Fold_base_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt"
    ["InpaintSeq_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt"
    ["InpaintSeq_Fold_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt"
    ["ActiveSite_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt"
    ["Base_epoch8_ckpt.pt"]="http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt"
)

echo ""
echo "--- Downloading RFdiffusion weights (~2 GB total) ---"
for name in "${!WEIGHTS[@]}"; do
    if [ ! -f "$name" ]; then
        echo "  $name..."
        wget -q "${WEIGHTS[$name]}"
    else
        echo "  $name (already present)"
    fi
done

# -----------------------------------------------------------------------------
# Install ColabDesign (wraps ProteinMPNN and AF2)
# -----------------------------------------------------------------------------
if ! python -c "import colabdesign" 2>/dev/null; then
    echo ""
    echo "--- Installing ColabDesign ---"
    pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.3
fi

# -----------------------------------------------------------------------------
# Install extras: py3Dmol, JupyterLab, ipywidgets
# -----------------------------------------------------------------------------
echo ""
echo "--- Installing extras (py3Dmol, JupyterLab) ---"
pip install -q py3Dmol jupyterlab ipywidgets

# Register SE3nv as a Jupyter kernel
python -m ipykernel install --user --name SE3nv --display-name "Python 3 (SE3nv)"

# -----------------------------------------------------------------------------
# Download AlphaFold2 parameters (~3.5 GB)
# -----------------------------------------------------------------------------
if [ ! -f "$HOME/params/params/params_model_1.npz" ]; then
    echo ""
    echo "--- Downloading AlphaFold2 params (~3.5 GB) ---"
    mkdir -p "$HOME/params"
    cd "$HOME/params"
    if [ ! -f "alphafold_params_2022-12-06.tar" ]; then
        wget -q https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
    fi
    tar -xf alphafold_params_2022-12-06.tar
    rm -f alphafold_params_2022-12-06.tar
fi

# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
echo ""
echo "--- Running sanity check ---"
python - <<'PY'
import sys
ok = True

try:
    import torch
    print(f"  ✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ❌ torch: {e}"); ok = False

try:
    from colabdesign.mpnn import mk_mpnn_model
    from colabdesign.af   import mk_af_model
    print("  ✅ colabdesign (MPNN + AF2)")
except Exception as e:
    print(f"  ❌ colabdesign: {e}"); ok = False

try:
    import py3Dmol
    print("  ✅ py3Dmol")
except Exception as e:
    print(f"  ❌ py3Dmol: {e}"); ok = False

import os
if os.path.isfile(os.path.expanduser("~/RFdiffusion/models/Base_ckpt.pt")):
    print("  ✅ RFdiffusion weights")
else:
    print("  ❌ RFdiffusion weights missing"); ok = False

if os.path.isfile(os.path.expanduser("~/params/params/params_model_1.npz")):
    print("  ✅ AlphaFold2 params")
else:
    print("  ❌ AF2 params missing"); ok = False

sys.exit(0 if ok else 1)
PY

echo ""
echo "=============================================="
echo "  Install complete."
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. On the AWS instance, start JupyterLab:"
echo "       conda activate SE3nv"
echo "       cd ~"
echo "       jupyter lab --no-browser --ip=127.0.0.1 --port=8888"
echo ""
echo "  2. From your laptop, open an SSH tunnel:"
echo "       ssh -L 8888:localhost:8888 -i <your-key.pem> ubuntu@<instance-ip>"
echo ""
echo "  3. Paste the Jupyter URL (from step 1) into your browser."
echo ""
echo "  4. Open the notebook and select the 'Python 3 (SE3nv)' kernel."
echo ""
