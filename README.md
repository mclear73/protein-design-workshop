# Protein Design Workshop

De novo protein design workshop built around **ProteinMPNN** (sequence redesign) and **RFdiffusion** (generative structure design), targeted at industrial biomanufacturing and environmental remediation applications.

Prepared for the Schmidt Sciences Biosciences Initiative retreat.

## What's here

| File | Purpose |
|---|---|
| `ProteinMPNN_Workshop_Teams.ipynb` | Main activity — three teams redesign sequences for PETase, ZAR1, or CarRP with ProteinMPNN and validate with AlphaFold2 |
| `RFdiffusion_Industrial_Demo.ipynb` | Bonus activity — same three teams design de novo binders to their targets with RFdiffusion |
| `setup.sh` | One-shot installer (run on your AWS instance) |
| `manage.sh` | Laptop-side lifecycle helper for AWS instance (start/stop/resize/tunnel) |
| `AWS_DEPLOYMENT.md` | Full AWS deployment guide |

## The three-protein framework

Each team works on one protein. Everyone runs the same steps. The differences in results tell the story.

| Team | Protein | Source | What it teaches |
|---|---|---|---|
| **1** | **PETase** (*I. sakaiensis*, 6EQE) | Experimental PDB | Well-represented training data — the "easy" case |
| **2** | **ZAR1** (*A. thaliana*, 6J5T) | Experimental PDB, truncated | Plant proteins are under-represented; dynamic oligomer |
| **3** | **CarRP** (*M. circinelloides*, Q9UUQ6) | AlphaFold prediction | Real industrial target — no experimental structure, non-model fungus |

## Quick start on AWS

1. Launch `g5.xlarge` DLAMI Ubuntu instance in `us-east-1` (see `AWS_DEPLOYMENT.md` for full details)
2. SSH in and clone this repo
3. Run setup
4. Start Jupyter and tunnel

```bash
# On AWS instance
git clone https://github.com/YOUR_USERNAME/protein-design-workshop.git
cd protein-design-workshop
chmod +x setup.sh
./setup.sh
conda activate SE3nv
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

```bash
# On laptop (second terminal)
ssh -L 8888:localhost:8888 -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

Paste the Jupyter URL from the instance terminal into your browser.

## Acknowledgments

Built on the work of:

- David Baker's lab (RFdiffusion, ProteinMPNN) — University of Washington IPD
- Google DeepMind (AlphaFold2)
- Sergey Ovchinnikov (ColabDesign) — the Colab wrappers that make this all accessible

## License

MIT — see `LICENSE`.
