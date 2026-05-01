# Protein Design Workshop

De novo protein design workshop built around **ProteinMPNN** (sequence redesign) and **RFdiffusion** (generative structure design), targeted at industrial biomanufacturing and environmental remediation applications.

Prepared for the Schmidt Sciences Biosciences Initiative retreat.

## What's here

| File | Purpose |
| --- | --- |
| `app.py` | **Streamlit workshop UI** — dropdowns + buttons, the path participants use |
| `start_app.sh` | One-shot launcher for the Streamlit app on the AWS instance |
| `ProteinMPNN_Workshop_Teams.ipynb` | Notebook version of the main activity (for tinkering) |
| `RFdiffusion_Industrial_Demo.ipynb` | Notebook bonus activity — de novo binders to the team targets |
| `setup.sh` | One-shot installer (run on your AWS instance) |
| `manage.sh` | Laptop-side lifecycle helper for AWS instance (start/stop/resize/tunnel) |
| `AWS_DEPLOYMENT.md` | Full AWS deployment guide |

## The three-protein framework

Each team works on one protein. Everyone runs the same steps. The differences in results tell the story.

| Team | Protein | Source | What it teaches |
| --- | --- | --- | --- |
| **1** | **PETase** (*I. sakaiensis*, 6EQE) | Experimental PDB | Well-represented training data — the "easy" case |
| **2** | **ZAR1** (*A. thaliana*, 6J5T) | Experimental PDB, truncated | Plant proteins are under-represented; dynamic oligomer |
| **3** | **CarRP** (*M. circinelloides*, Q9UUQ6) | AlphaFold prediction | Real industrial target — no experimental structure, non-model fungus |

## Quick start on AWS

1. Launch `g5.xlarge` DLAMI Ubuntu instance in `us-east-1` (see `AWS_DEPLOYMENT.md` for full details)
2. SSH in and clone this repo
3. Run setup
4. Start the workshop UI

```bash
# On AWS instance
git clone https://github.com/mclear73/protein-design-workshop.git
cd protein-design-workshop
chmod +x setup.sh start_app.sh
./setup.sh
./start_app.sh
```

```bash
# On laptop (second terminal)
./manage.sh streamlit-tunnel
```

Then open **http://localhost:8501** in your browser.

## Two ways to run the workshop

**`app.py` (Streamlit) is the recommended path for participants.** It exposes the activity as a clean web UI — dropdowns to pick the team's protein, sliders for design parameters, buttons to generate and validate, and an inline 3D viewer for the result. Participants never see the underlying Python.

**The Jupyter notebooks** stay in the repo for two reasons: (1) facilitators preparing the activity may want to inspect or modify the underlying logic, and (2) participants who finish the Streamlit activity early can dig into the code to see how it works. Use them via:

```bash
# On the instance
conda activate SE3nv
jupyter lab --no-browser --ip=127.0.0.1 --port=8888

# On the laptop
./manage.sh tunnel
```

## Before your first deployment: pin the RFdiffusion commit

`setup.sh` pins RFdiffusion to a specific commit hash so upstream changes can't break your install between dry-run and event day. **Before deploying to a fresh instance, replace the placeholder commit hash in `setup.sh` with the actual commit from your tested-and-working setup.**

To find the commit hash from a working instance:

```bash
# On a working AWS instance
cd ~/RFdiffusion && git rev-parse HEAD
```

Then in `setup.sh`, update this line near the top of the file:

```bash
RFDIFFUSION_COMMIT="b44206a2a79f219bb1a649ea50603a284c225050"  # placeholder — replace
```

Replace the hash with what `git rev-parse HEAD` returned. Commit the change. Now every fresh install will use the exact RFdiffusion version you've tested against.

If you ever need to update to a newer RFdiffusion: test the full pipeline (Streamlit app + both notebooks, all three teams) end-to-end against the new commit before changing the pin.

## Verifying an instance before the workshop

Run the setup script in `--check` mode to verify a fresh or rebooted instance is healthy without re-running any installs or downloads:

```bash
./setup.sh --check
```

This validates the conda environment, GPU availability, all required Python imports (including Streamlit), RFdiffusion weight file sizes, and AlphaFold2 parameter files. It exits 0 if everything is workshop-ready, 1 if anything is wrong.

Recommended: run `--check` on the morning of the event before participants arrive.

## Setup logs

Each `setup.sh` run writes a timestamped log to `~/setup-YYYYMMDD-HHMMSS.log` capturing every command and its output. If something fails, that log is the first place to look.

## Acknowledgments

Built on the work of:

* David Baker's lab (RFdiffusion, ProteinMPNN) — University of Washington IPD
* Google DeepMind (AlphaFold2)
* Sergey Ovchinnikov (ColabDesign) — the Colab wrappers that make this all accessible

## License

MIT — see `LICENSE`.
