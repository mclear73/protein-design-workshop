"""
Protein Design Workshop — Streamlit interface.

Runs on the same AWS GPU instance as the Jupyter notebooks. Wraps the
ProteinMPNN sequence-redesign + AlphaFold2 validation flow in a clean UI
so workshop participants can interact via dropdowns and buttons instead
of Python cells.

Launch on the instance:
    conda activate SE3nv
    streamlit run app.py --server.port 8501 --server.address 127.0.0.1

Tunnel from laptop:
    ./manage.sh streamlit-tunnel
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import streamlit as st

# =============================================================================
# Page config — must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Protein Design Workshop",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Target catalog — same three proteins as the notebooks
# =============================================================================
TARGETS = {
    "PETase": {
        "label":      "PETase — plastic depolymerase (easy / industrial)",
        "source":     "pdb",
        "pdb_id":     "6EQE",
        "chain":      "A",
        "range":      None,
        "organism":   "Ideonella sakaiensis (bacterium)",
        "function":   "Plastic (PET) depolymerase",
        "hotspots":   "A159,A161,A185",
        "blurb": (
            "**The 'easy' case.** Single-domain α/β hydrolase, ~290 aa. "
            "Plenty of training data from related cutinases and esterases. "
            "AI models tend to look good here — this is the well-trained lane."
        ),
    },
    "ZAR1": {
        "label":      "ZAR1 — plant immune receptor (hard case)",
        "source":     "pdb",
        "pdb_id":     "6J6I",  # Wang et al. 2019 Science resistosome; chain C (UniProt Q38834)
        "chain":      "C",     # 6J6I has 5 copies of ZAR1 in chains C, F, G, L, O
        "range":      (1, 200),
        "organism":   "Arabidopsis thaliana (plant)",
        "function":   "NLR immune receptor — forms pentameric resistosome",
        "hotspots":   "A14,A17,A24",
        "blurb": (
            "**The 'hard' case.** Plant NLRs are large, multidomain, and form "
            "dynamic oligomeric complexes. Plant proteins are under-represented "
            "in training data vs. bacterial/human. We truncate to the CC + NB-ARC "
            "domain (~200 aa) to keep things tractable."
        ),
    },
    "CarRP": {
        "label":      "CarRP — carotenoid biosynthesis (real-world industrial)",
        "source":     "afdb",
        "uniprot":    "Q9UUQ6",
        "chain":      "A",
        "range":      None,  # full 614 aa: bifunctional fusion (PSY + LCY domains)
        "organism":   "Mucor circinelloides (fungus)",
        "function":   "Bifunctional lycopene cyclase + phytoene synthase",
        "hotspots":   "A45,A80,A150",
        "blurb": (
            "**The 'real world' case.** Fungal bifunctional enzyme with no "
            "experimental structure — we use an AlphaFold Database prediction. "
            "This is what most industrially-relevant targets actually look like. "
            "Carotenoids are a ~$1.8B global market."
        ),
    },
}

# =============================================================================
# Session state defaults
# =============================================================================
def init_state() -> None:
    defaults = {
        "team":          "PETase",
        "target_loaded": False,
        "input_pdb":     None,
        "n_residues":    None,
        "designs":       None,
        "validated":     None,
        "running":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# =============================================================================
# Cached environment check — runs once per session
# =============================================================================
@st.cache_data(show_spinner=False)
def check_environment() -> dict:
    """Verify the env is workshop-ready. Cached so it doesn't rerun on every interaction."""
    status = {"ok": True, "messages": []}

    try:
        import torch
        if torch.cuda.is_available():
            status["messages"].append(("✅", f"GPU: {torch.cuda.get_device_name(0)}"))
        else:
            status["messages"].append(("❌", "No GPU detected"))
            status["ok"] = False
    except ImportError as e:
        status["messages"].append(("❌", f"torch: {e}"))
        status["ok"] = False

    # ProteinMPNN side (this env, SE3nv)
    try:
        from colabdesign.mpnn import mk_mpnn_model  # noqa: F401
        status["messages"].append(("✅", "ColabDesign (ProteinMPNN)"))
    except ImportError as e:
        status["messages"].append(("❌", f"ColabDesign: {e}"))
        status["ok"] = False

    # ColabFold side (separate env, called via subprocess)
    try:
        import msa_cache  # noqa: F401
        status["messages"].append(("✅", "msa_cache module loaded"))
    except ImportError as e:
        status["messages"].append(("❌", f"msa_cache module: {e}"))
        status["ok"] = False

    # Check the colabfold conda env actually exists and has colabfold_batch
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "colabfold", "which", "colabfold_batch"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            status["messages"].append(("✅", f"colabfold env: {result.stdout.strip()}"))
        else:
            status["messages"].append(("❌", "colabfold_batch not found in `colabfold` env"))
            status["ok"] = False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        status["messages"].append(("❌", f"conda not callable: {e}"))
        status["ok"] = False

    # MSA cache directory
    cache_dir = Path(os.path.expanduser("~/msa_cache"))
    if cache_dir.is_dir():
        n_cached = len(list(cache_dir.glob("*.a3m")))
        status["messages"].append(("✅", f"MSA cache: {n_cached} entries at {cache_dir}"))
    else:
        status["messages"].append(("⚠️", f"MSA cache empty (will be created on first run)"))

    return status


# =============================================================================
# Worker functions — wrap the notebook code, no UI in here
# =============================================================================
def _afdb_pdb_url(uniprot_id: str) -> str:
    """
    Look up the current AFDB prediction URL for a UniProt accession.

    AlphaFold Database tags structures with version numbers (v1, v2, ..., v6
    as of 2026). The version increments when AFDB regenerates the prediction,
    and old version URLs eventually 404. Hardcoding any version is a tripwire,
    so we query the API for the latest one. Falls back to the metadata-stripped
    pdbUrl if it parses, else raises.
    """
    import urllib.request
    import json

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"AFDB API lookup for {uniprot_id} failed: {e}. "
            f"Check that {uniprot_id} has an AlphaFold prediction at "
            f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
        ) from e

    if not data or not isinstance(data, list) or "pdbUrl" not in data[0]:
        raise RuntimeError(
            f"AFDB API returned unexpected payload for {uniprot_id}: {data!r}"
        )
    return data[0]["pdbUrl"]


def download_target(team: str) -> tuple[str, int]:
    """Download and (optionally) truncate the target PDB. Returns (path, n_residues)."""
    target = TARGETS[team]
    Path("inputs").mkdir(exist_ok=True)
    input_pdb = f"inputs/{team}.pdb"

    # Download
    if target["source"] == "pdb":
        url = f"https://files.rcsb.org/download/{target['pdb_id']}.pdb"
    else:
        url = _afdb_pdb_url(target["uniprot"])

    result = subprocess.run(f"wget -q {url} -O {input_pdb}", shell=True)
    if result.returncode != 0 or not os.path.isfile(input_pdb) or os.path.getsize(input_pdb) < 1000:
        size = os.path.getsize(input_pdb) if os.path.isfile(input_pdb) else 0
        raise RuntimeError(f"Download failed: exit={result.returncode}, size={size} bytes")

    # Truncate if requested
    if target["range"] is not None:
        lo, hi = target["range"]
        trunc = f"inputs/{team}_trunc.pdb"
        with open(input_pdb) as fin, open(trunc, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")):
                    if line[21] != target["chain"]:
                        continue
                    try:
                        if not (lo <= int(line[22:26]) <= hi):
                            continue
                    except ValueError:
                        pass
                fout.write(line)
        input_pdb = trunc

    n_res = len({
        line[22:26] for line in open(input_pdb)
        if line.startswith("ATOM") and line[21] == target["chain"]
    })
    return input_pdb, n_res


def design_sequences(input_pdb: str, num: int, temperature: float) -> list[dict]:
    """Run ProteinMPNN to generate sequences. Returns list of design dicts."""
    from colabdesign.mpnn import mk_mpnn_model

    mpnn = mk_mpnn_model()
    mpnn.prep_inputs(pdb_filename=input_pdb, rm_aa="C")
    out = mpnn.sample(num=num, temperature=temperature)

    designs = []
    for i, seq in enumerate(out["seq"]):
        designs.append({
            "idx":      i,
            "sequence": seq.replace("/", ""),
            "score":    float(out["score"][i]),
        })
    return designs


def extract_wt_sequence(pdb_path: str, chain: str) -> str:
    """
    Read the WT amino-acid sequence from a PDB file, in residue-index order.

    Used to compute the MSA cache key for the target. Designs are validated by
    substituting their sequence into the cached a3m of this WT sequence, so
    consistency with what ProteinMPNN sees is what matters here — and
    ProteinMPNN reads the same PDB.
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen_resnums: set[int] = set()
    seq_chars: list[str] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            if resnum in seen_resnums:
                continue
            seen_resnums.add(resnum)
            res = line[17:20]
            seq_chars.append(three_to_one.get(res, "X"))
    return "".join(seq_chars)


def validate_with_colabfold(input_pdb: str, designs: list[dict], chain: str,
                            team: str) -> list[dict]:
    """
    Run ColabFold (MSA-based AF2) on each designed sequence.

    The MSA is generated once per WT target sequence and cached under
    ~/msa_cache/. Per-design validation reuses the cached a3m by substituting
    the designed sequence in as the new query — see msa_cache.py for the full
    rationale.

    Adds plddt/rmsd/ptm/af_pdb to each design dict, in place.
    """
    import msa_cache

    wt_seq = extract_wt_sequence(input_pdb, chain)
    if not wt_seq:
        raise RuntimeError(
            f"Could not extract WT sequence from {input_pdb} chain {chain}"
        )

    # Get or generate the cached MSA for this target's WT sequence
    key = msa_cache.cache_key(team, wt_seq)
    if not msa_cache.cache_exists(key):
        with st.spinner(
            f"Generating MSA for {team} (~3 min, one-time per target)..."
        ):
            key = msa_cache.generate_msa(team, wt_seq, source=f"app:{team}")

    Path("outputs").mkdir(exist_ok=True)
    validated: list[dict] = []
    progress = st.progress(
        0.0, text=f"Validating designs with AlphaFold2 (MSA cached, {msa_cache.NUM_MODELS} models)..."
    )

    for i, d in enumerate(designs):
        # Sanity check: the designed sequence must match WT length, otherwise
        # the cached alignment can't be reused.
        if len(d["sequence"]) != len(wt_seq):
            raise RuntimeError(
                f"Design {i} has length {len(d['sequence'])}, expected {len(wt_seq)}. "
                "ProteinMPNN fixbb should preserve length — check design_sequences()."
            )

        design_out = Path(f"outputs/{team}_design{d['idx']}_af")
        design_out.mkdir(parents=True, exist_ok=True)

        result = msa_cache.validate_design(
            cache_key_str=key,
            design_seq=d["sequence"],
            design_name=f"{team}_design{d['idx']}",
            out_dir=design_out,
        )

        d["plddt"] = float(result["plddt"])
        d["ptm"]   = float(result["ptm"])
        d["af_pdb"] = result["pdb_path"]
        # The design PDB from ColabFold always uses chain A regardless of the
        # input target's chain (which may be C for ZAR1, etc.)
        d["rmsd"]  = msa_cache.calpha_rmsd(
            input_pdb, result["pdb_path"],
            chain_a=chain, chain_b="A",
        )

        validated.append(d)
        progress.progress(
            (i + 1) / len(designs),
            text=f"Validated {i + 1}/{len(designs)} designs",
        )

    progress.empty()
    return validated


def render_structure(orig_pdb: str, design_pdb: str | None = None,
                     chain: str = "A") -> str:
    """
    Build a py3Dmol view as HTML for embedding via st.components.v1.html.

    The design PDB is Kabsch-superimposed onto the WT before display so the
    two structures appear in the same coordinate frame. Without this,
    py3Dmol draws each model in its original frame and they end up
    side-by-side rather than overlaid.

    The WT PDB is filtered to just the workshop's chain — multi-chain inputs
    like 6J6I (ZAR1 resistosome) contain other proteins (kinase, PBL2) that
    would otherwise clutter the viewer.

    The design PDB always has its content on chain A (ColabFold default),
    even when the WT lives on a different chain.
    """
    import py3Dmol
    import msa_cache

    view = py3Dmol.view(width=820, height=480)

    # Show only the workshop's chain from the WT input — drop any other chains
    wt_text = msa_cache.filter_pdb_to_chain(orig_pdb, chain)
    view.addModel(wt_text, "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})

    if design_pdb and os.path.isfile(design_pdb):
        # Superimpose the design onto the WT before adding it to the view.
        # Design PDBs from ColabFold use chain A regardless of the WT's chain.
        aligned_pdb_text = msa_cache.aligned_design_pdb(
            orig_pdb, design_pdb,
            input_chain=chain, design_chain="A",
        )
        view.addModel(aligned_pdb_text, "pdb")
        view.setStyle({"model": 1}, {"cartoon": {
            "colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 90}
        }})

    view.zoomTo()
    return view._make_html()


# =============================================================================
# UI — sidebar (controls) and main pane (results)
# =============================================================================
def render_sidebar() -> dict:
    """Render sidebar controls. Returns the chosen settings."""
    with st.sidebar:
        st.title("🧬 Workshop")
        st.caption("Three teams. Three proteins. One workflow.")

        st.divider()

        team = st.selectbox(
            "**Pick your team's protein:**",
            options=list(TARGETS.keys()),
            format_func=lambda k: TARGETS[k]["label"],
            index=list(TARGETS.keys()).index(st.session_state["team"]),
            key="team_selector",
        )
        # If team changed, clear stale state
        if team != st.session_state["team"]:
            st.session_state.update(
                team=team, target_loaded=False, input_pdb=None,
                n_residues=None, designs=None, validated=None,
            )

        st.markdown(TARGETS[team]["blurb"])

        st.divider()
        st.markdown("**Design parameters**")

        num_designs = st.slider(
            "Number of designs",
            min_value=2, max_value=8, value=4, step=1,
            help="More designs = more diversity to compare, but each adds 2-5 min for AF2 validation.",
        )
        temperature = st.slider(
            "Sampling temperature",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            help="Higher = more diverse sequences. Lower = more conservative.",
        )

        st.divider()

        with st.expander("Environment status"):
            env = check_environment()
            for icon, msg in env["messages"]:
                st.write(f"{icon} {msg}")
            if not env["ok"]:
                st.error("Some checks failed — see setup.sh")

        return {"team": team, "num_designs": num_designs, "temperature": temperature}


def render_target_info(team: str) -> None:
    target = TARGETS[team]
    cols = st.columns([1, 1, 1])
    cols[0].metric("Organism", target["organism"].split(" (")[0])
    cols[1].metric("Function", target["function"][:40] + ("…" if len(target["function"]) > 40 else ""))
    src_label = f"PDB {target.get('pdb_id', '')}" if target["source"] == "pdb" else f"AF-{target.get('uniprot', '')}"
    cols[2].metric("Source", src_label)


def classify_design(d: dict, n_residues: int | None = None) -> tuple[str, str]:
    """
    Classify a validated design's quality.

    Returns (label, emoji) where label ∈ {"self-consistent", "marginal", "divergent"}.

    Decision logic combines three signals — pLDDT (per-residue confidence,
    0–100 scale), pTM (global fold confidence, 0–1), and Cα RMSD to the WT
    (after Kabsch superposition). Using all three is more honest than a binary
    RMSD cutoff: a design that scores pLDDT 92 / pTM 0.90 / RMSD 4 Å on a
    600-residue multi-domain protein has a *correct* fold even though a strict
    2 Å threshold would flag it.

    Length-aware RMSD: per-residue deviation matters more than absolute Å.
    For a 100 aa protein, 5 Å is sloppy; for a 600 aa protein with a flexible
    tail, 5 Å can come entirely from one disordered terminus. We use the
    larger of (5 Å, 0.01 × length) as the self-consistent ceiling — same
    spirit as the published ProteinMPNN/RFdiffusion validation papers, which
    tighten the threshold for short single-domain test cases.

    Marginal vs divergent: marginal means "fold is broadly right, details
    uncertain" — useful as a separate bucket because some workshop runs
    legitimately land here and forcing them into a binary good/bad loses
    signal that the participant should see.
    """
    plddt = d.get("plddt", 0)
    ptm = d.get("ptm", 0)
    rmsd = d.get("rmsd", -1)

    # If RMSD wasn't computed (e.g. chain mismatch), fall back to pLDDT+pTM only
    rmsd_ok_strict = (rmsd < 0) or (rmsd <= max(5.0, 0.01 * (n_residues or 0)))
    rmsd_ok_loose  = (rmsd < 0) or (rmsd <= 8.0)

    if plddt >= 80 and ptm >= 0.70 and rmsd_ok_strict:
        return ("self-consistent", "✅")
    if plddt >= 70 and rmsd_ok_loose:
        return ("marginal", "🟡")
    return ("divergent", "⚠️")


def render_results_table(designs: list[dict], n_residues: int | None = None) -> None:
    """Render the validated-designs results."""
    rows = []
    for d in designs:
        label, emoji = classify_design(d, n_residues)
        rows.append({
            "Design":  f"#{d['idx'] + 1}",
            "MPNN score": f"{d['score']:.3f}",
            "pLDDT":  f"{d['plddt']:.2f}",
            "RMSD (Å)": f"{d['rmsd']:.2f}" if d['rmsd'] >= 0 else "—",
            "pTM":    f"{d['ptm']:.2f}" if d['ptm'] >= 0 else "—",
            "Verdict": f"{emoji} {label}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# =============================================================================
# Main app body
# =============================================================================
def main() -> None:
    settings = render_sidebar()
    team = settings["team"]

    st.title("Protein Design Workshop")
    st.caption(
        "Redesign your team's protein with **ProteinMPNN**, "
        "validate the design with **AlphaFold2**, and see how the result depends on the protein."
    )

    render_target_info(team)
    st.divider()

    # ----- Step 1: Load target -----
    st.subheader("Step 1 — Load target structure")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("📥 Load target", use_container_width=True, disabled=st.session_state.running):
            st.session_state.running = True
            try:
                with st.spinner(f"Downloading {team}..."):
                    pdb, n_res = download_target(team)
                st.session_state.update(
                    target_loaded=True, input_pdb=pdb, n_residues=n_res,
                    designs=None, validated=None,
                )
                st.success(f"Loaded {team}: {n_res} residues")
            except Exception as e:
                st.error(f"Download failed: {e}")
            finally:
                st.session_state.running = False

    with col_b:
        if st.session_state.target_loaded:
            st.info(f"📏 Working structure: **{st.session_state.n_residues} residues**  ·  `{st.session_state.input_pdb}`")

    if not st.session_state.target_loaded:
        st.stop()

    # ----- Step 2: Design + validate -----
    st.divider()
    st.subheader("Step 2 — Design new sequences and validate")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run_design = st.button(
            "✨ Generate & validate designs",
            use_container_width=True,
            disabled=st.session_state.running,
            type="primary",
        )
    with col_b:
        st.caption(
            f"This will generate **{settings['num_designs']}** designs at "
            f"temperature **{settings['temperature']:.2f}** and validate each "
            f"with AlphaFold2 (MSA-cached, 3 models). Expect ~10s per design "
            f"for MPNN + ~2-5 min per design for AF2 depending on target size."
        )

    if run_design:
        st.session_state.running = True
        try:
            t0 = time.time()
            with st.spinner(f"Generating {settings['num_designs']} sequences with ProteinMPNN..."):
                designs = design_sequences(
                    st.session_state.input_pdb,
                    num=settings["num_designs"],
                    temperature=settings["temperature"],
                )
            t_mpnn = time.time() - t0

            t0 = time.time()
            validated = validate_with_colabfold(
                st.session_state.input_pdb,
                designs,
                chain=TARGETS[team]["chain"],
                team=team,
            )
            t_af = time.time() - t0

            st.session_state.update(designs=designs, validated=validated)
            st.success(f"Done. ProteinMPNN: {t_mpnn:.0f}s  ·  AlphaFold2: {t_af:.0f}s")
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.exception(e)
        finally:
            st.session_state.running = False

    if st.session_state.validated is None:
        st.stop()

    # ----- Step 3: Results table + 3D viewer -----
    st.divider()
    st.subheader("Step 3 — Results")

    render_results_table(st.session_state.validated, st.session_state.n_residues)

    # Per-target interpretation of the results. Use the SAME classifier the
    # results table uses, so the callout's condition matches what the table
    # is saying — no more "table says divergent but callout doesn't fire" or
    # vice versa.
    validated = st.session_state.validated
    n_res = st.session_state.n_residues
    classifications = [classify_design(d, n_res)[0] for d in validated]
    any_divergent = "divergent" in classifications
    any_marginal = "marginal" in classifications

    if any_divergent:
        if team == "ZAR1":
            st.warning(
                "**This is the lesson, not a bug.** Look at the structures: the central "
                "coiled-coil bundle aligns reasonably, but the rest of the predicted fold "
                "diverges from the WT — and that's expected for ZAR1.\n\n"
                "Three things stack against the workflow here:\n\n"
                "1. **You truncated a multi-domain protein.** Residues 1–200 cuts the NB-ARC "
                "domain in half. NB-ARC needs the *full* domain (~1–450) to fold; ProteinMPNN "
                "was asked to design a sequence for a half-fold that doesn't exist in nature.\n\n"
                "2. **Plant NLRs are scarce in AF2's training data.** AF2's training was "
                "dominated by bacterial/human/yeast structures. Plant immune receptors are "
                "underrepresented, and pLDDT/pTM reflect that uncertainty (compare ZAR1's "
                f"pTM ≈ {validated[0]['ptm']:.2f} to PETase's typical 0.85+).\n\n"
                "3. **ZAR1 is functionally an oligomer.** In 6J6I it forms a pentameric "
                "resistosome. ProteinMPNN sees chain C as a monomer and designs polar residues "
                "into what should be a buried protein-protein interface — destabilizing the fold.\n\n"
                "**The takeaway for the workshop:** the workflow doesn't fail silently. pLDDT, "
                "pTM, and RMSD all flag the problem. A 'divergent' verdict on a hard target is "
                "a valid scientific result."
            )
            with st.expander("🔓 Try this next: extend to the full NB-ARC domain"):
                st.markdown(
                    "If you ran this with **residues 1–450** of ZAR1 (the full CC + NB-ARC "
                    "tandem) instead of 1–200, you'd give AF2 a complete fold to evaluate. "
                    "Predict: does pLDDT come up? Does the central bundle stay aligned while "
                    "the new C-terminal residues form the missing nucleotide-binding pocket? "
                    "This is the obvious follow-up if you finish early — a valid hypothesis "
                    "test about *why* the truncated case failed."
                )
        else:
            # PETase / CarRP unexpectedly diverging — surface that less commonly
            st.info(
                f"Some designs are flagged as divergent. That's unusual for {team} — "
                "try a lower sampling temperature or rerun. If it persists, inspect the "
                "design sequence for chemistry that doesn't match the fold."
            )
    elif any_marginal:
        # Marginal-only: no warning needed, just a soft note about what the verdict means
        st.info(
            "**'Marginal' verdict.** The fold is broadly right (pLDDT ≥ 70) but tighter "
            "validation thresholds aren't met. Often a flexible terminus or one weak "
            "subdomain is dragging RMSD up while the core is fine — open the 3D viewer "
            "below and look for which region is colored red/orange (low pLDDT)."
        )

    st.markdown("**Visualize one of the designs:**")
    pick = st.selectbox(
        "Design to inspect",
        options=range(len(st.session_state.validated)),
        format_func=lambda i: (
            f"Design #{i + 1}"
            f"  ·  pLDDT {st.session_state.validated[i]['plddt']:.2f}"
            f"  ·  pTM {st.session_state.validated[i]['ptm']:.2f}"
            f"  ·  RMSD {st.session_state.validated[i]['rmsd']:.2f} Å"
        ),
        label_visibility="collapsed",
    )

    chosen = st.session_state.validated[pick]
    html = render_structure(
        st.session_state.input_pdb,
        chosen["af_pdb"],
        chain=TARGETS[team]["chain"],
    )
    st.components.v1.html(html, height=500)

    st.caption(
        "**Gray** = original target backbone.  "
        "**Colored** = AlphaFold2 prediction of the designed sequence "
        "(blue = high confidence, red = low). They should superimpose tightly "
        "if the design is self-consistent."
    )

    with st.expander("Show designed sequence"):
        st.code(chosen["sequence"], language=None)

    # ----- Step 4: Debrief prompts -----
    st.divider()
    with st.expander("🗣️ Debrief — questions for the room"):
        st.markdown("""
        1. **What was your team's best pLDDT and RMSD?**
        2. **Did you notice anything weird?** Low confidence in certain regions, divergent
           predictions, strange sequence preferences (lots of hydrophobics, prolines, etc.)
        3. **How does this match what you'd expect from the preamble?**

        **Likely patterns:**
        - **PETase** usually clean (well-trained data)
        - **ZAR1** often variable (plant proteins under-represented)
        - **CarRP** typically most variable (fungal + AF-predicted starting structure)

        That gradient — from biomedical-adjacent to truly industrial — is the lesson.
        """)


if __name__ == "__main__":
    main()
