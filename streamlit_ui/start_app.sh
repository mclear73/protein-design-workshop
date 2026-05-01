#!/bin/bash
# =============================================================================
# start_app.sh — launch the Streamlit workshop UI
# =============================================================================
# Run on the AWS instance:
#   ./start_app.sh
#
# Tunnel from your laptop (in another terminal):
#   ./manage.sh streamlit-tunnel
# Then open http://localhost:8501 in your browser.
# =============================================================================

set -e

# Activate the conda env that has ColabDesign + RFdiffusion installed
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate SE3nv

cd "$(dirname "$0")"

# Bind to localhost only — the SSH tunnel exposes it to your laptop securely.
# Disable Streamlit's anonymous usage stats and welcome wizard for a cleaner first run.
exec streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false
