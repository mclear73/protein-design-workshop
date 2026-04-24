# RFdiffusion Industrial/Environmental Demo — AWS Deployment

De novo protein design workshop deployed on your own AWS GPU instance. Built around industrial biomanufacturing and environmental remediation targets (plastic degradation, lignin processing, catalytic cages) rather than biomedical targets.

## What you get

- Self-hosted GPU instance running RFdiffusion + ProteinMPNN + AlphaFold2 validation
- JupyterLab accessed via SSH tunnel (no public ports exposed)
- A pre-configured notebook with three demo modes:
  - **Mode A** — novel scaffold from noise (~5 min)
  - **Mode B** — symmetric catalytic cage (~10 min)
  - **Mode C** — binder to PETase / LCC cutinase / laccase (~20 min)

## Files in this bundle

| File | Purpose |
|---|---|
| `README.md` | This document |
| `setup.sh` | One-shot install script (run on the AWS instance) |
| `RFdiffusion_Industrial_Demo.ipynb` | The workshop notebook |

---

## Prerequisites

- AWS account with permission to launch GPU instances (G instance family)
- SSH key pair in your chosen region
- Local terminal with SSH + a web browser

Service quotas: if your account hasn't launched GPU instances before, you may need to request a service-quota increase for "Running On-Demand G and VT instances". This can take a few hours.

---

## Step 1 · Launch the EC2 instance

From the AWS console:

| Setting | Value |
|---|---|
| **Region** | `us-east-1`, `us-east-2`, or `us-west-2` (best GPU availability) |
| **AMI** | Search: *"Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)"* |
| **Instance type** | `g5.xlarge` recommended — A10G GPU, 24 GB VRAM, ~$1.00/hr |
|   | `g4dn.xlarge` acceptable — T4 GPU, 16 GB VRAM, ~$0.53/hr (slower) |
| **Storage** | 100 GB `gp3` EBS |
| **Security group** | Inbound: **SSH (22) from your IP only**. No other inbound rules. |
| **Key pair** | Use existing, or create new |

Cost reference:
- g5.xlarge: ~$24/day if left running; stop when not in use.
- 100 GB EBS: ~$8/month idle cost (whether running or stopped).

After launch, note the **public IPv4 address** (something like `54.123.45.67`).

---

## Step 2 · Transfer the workshop files

From your laptop, in the directory containing `setup.sh` and the notebook:

```bash
scp -i ~/.ssh/your-key.pem  setup.sh  RFdiffusion_Industrial_Demo.ipynb  ubuntu@<instance-ip>:~/
```

---

## Step 3 · SSH in and run the installer

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

On the instance:

```bash
chmod +x setup.sh
./setup.sh
```

Expect ~20–30 min the first time (weights are ~5.5 GB across RFdiffusion + AF2).

The script is idempotent — if something fails partway, just re-run it; it picks up where it left off.

---

## Step 4 · Start JupyterLab on the instance

Still on the instance:

```bash
conda activate SE3nv
cd ~
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

JupyterLab will print a URL with an embedded token, like:

```
http://127.0.0.1:8888/lab?token=abc123def456...
```

**Copy that full URL.** Leave this terminal running.

---

## Step 5 · Tunnel to your laptop

In a **new terminal on your laptop**:

```bash
ssh -L 8888:localhost:8888 -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
```

Leave this terminal open for the whole session. It forwards port 8888 from the instance to your laptop over the SSH connection — no public Jupyter port, no TLS certificate to manage.

In your browser, paste the URL from Step 4. Replace `127.0.0.1` with `localhost` if needed.

---

## Step 6 · Run the notebook

1. Double-click `RFdiffusion_Industrial_Demo.ipynb` in the JupyterLab file browser.
2. Kernel menu → *Change Kernel* → **Python 3 (SE3nv)**.
3. Run cells top-to-bottom. The first cell verifies the environment; fix any ❌ before proceeding.

---

## Step 7 · Shut down when done

**Stop the instance** from the AWS console when you're not using it — `g5.xlarge` bills by the second while running.

```
EC2 → Instances → select → Instance state → Stop
```

Stopping preserves the EBS volume (and therefore the conda env and weights) at the storage cost (~$8/month). Re-starting the instance later takes ~1 min and you don't need to re-run `setup.sh`.

To permanently clean up: Stop → Terminate the instance, and delete the EBS volume.

---

## Troubleshooting

**`Permission denied (publickey)` on SSH**
Your key file needs `chmod 400 your-key.pem`. Also check the username — DLAMI uses `ubuntu`, some AMIs use `ec2-user`.

**`No GPU detected` during setup**
Run `nvidia-smi`. If nothing returns, you launched a non-GPU instance. Stop and re-launch on a G-family instance.

**`SE3Transformer` build fails**
Sometimes hits a transient compile error. Re-run `setup.sh`; the second attempt usually succeeds. If it persistently fails, check that `nvcc --version` returns a reasonable CUDA version (DLAMI ships with CUDA 11.8 or 12.x).

**`CUDA out of memory`**
You're on a T4 trying Mode C with a large target. Either:
- Switch to `g5.xlarge` (A10G, 24 GB VRAM)
- Use a smaller `BINDER_LENGTH` (e.g., 50–60)
- Use Mode A or B instead

**Jupyter token expired or forgotten**
Kill the Jupyter process (`Ctrl-C` in the server terminal) and restart it. It prints a fresh URL.

**Binder mode fails with contig/chain errors**
The three preset targets (PETase, LCC, laccase) download their PDBs fresh each run. If the download is blocked (rare — RCSB allows public HTTPS), check outbound rules on the security group. PDB IDs, chains, and hotspots are defined in the mode-selection cell — edit them there.

---

## Multi-user considerations

This setup is designed for **one user per instance**. For a multi-person workshop:

**Option A (simplest):** Launch one instance per participant. With the setup script, each takes ~30 min to bring up; use AWS's "Launch more like this" or a launch template. Total cost for a 2-hour workshop with 10 people: ~$25.

**Option B (parallel-friendly):** One larger instance (`g5.4xlarge`, 4 A10Gs) running JupyterHub. More setup overhead; worth it for recurring use.

**Option C (AWS-native):** SageMaker Studio with a GPU instance type. Different cost structure but native multi-user and persistent storage. The notebook runs unchanged; only the install path differs (you'd install into SageMaker's conda env rather than the DLAMI's).

---

## Cost summary

| Scenario | Cost |
|---|---|
| First-time setup on g5.xlarge | ~$0.50 (30 min × $1.00/hr) |
| Running Mode A (monomer) | ~$0.08 per design |
| Running Mode C (binder, full pipeline) | ~$0.45 per design |
| Leaving instance running for a 2-hr workshop | ~$2 |
| EBS storage (instance stopped, between sessions) | ~$8/month |

---

## Security notes

This deployment deliberately exposes only SSH (port 22) to the public internet. JupyterLab listens on `127.0.0.1` only and is reached via SSH tunnel. If you want to expose JupyterLab directly (e.g., for remote participants without SSH setup), you'll need:

- A TLS certificate (free via Let's Encrypt)
- JupyterLab's `--ip=0.0.0.0 --NotebookApp.password=<hashed>` with a strong password
- Inbound port 443 in the security group

For a one-off workshop with trusted participants who have SSH, the tunnel approach is simpler and safer.

---

## What this notebook is *not*

- Not a production design pipeline. For bulk design (thousands of candidates), use RFdiffusion's batch scripts directly rather than this interactive notebook.
- Not a full binder-design workflow. Serious binder design adds Rosetta interface metrics, additional filters (ddG, shape complementarity, ipTM), and multi-temperature ProteinMPNN sampling. This notebook runs the core loop but uses minimal filters — good for teaching, insufficient for publication.
- Not an enzyme-design workflow. For active-site specification and atomic-level enzyme design, use RFdiffusion2 — it has a different input format and different validation criteria.

---

## References

- **RFdiffusion** — Watson et al., *Nature* (2023). [DOI: 10.1038/s41586-023-06415-8](https://doi.org/10.1038/s41586-023-06415-8)
- **RFdiffusion2** — Ahern, Yim, Tischer et al., *Nature* (2025). Atom-level active site scaffolding for enzyme design.
- **ProteinMPNN** — Dauparas et al., *Science* (2022). [DOI: 10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
- **ColabDesign wrapper** — [github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- **AlphaFold2** — Jumper et al., *Nature* (2021).
