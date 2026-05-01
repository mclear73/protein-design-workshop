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

import glob
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
        "pdb_id":     "6J5T",
        "chain":      "A",
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
        "range":      (1, 330),
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

    try:
        from colabdesign.mpnn import mk_mpnn_model  # noqa: F401
        from colabdesign.af import mk_af_model      # noqa: F401
        status["messages"].append(("✅", "ColabDesign (ProteinMPNN + AF2)"))
    except ImportError as e:
        status["messages"].append(("❌", f"ColabDesign: {e}"))
        status["ok"] = False

    af_params = glob.glob(os.path.expanduser("~/params/**/params_model_1.npz"), recursive=True)
    if af_params:
        status["messages"].append(("✅", f"AF2 params: {os.path.basename(af_params[0])}"))
    else:
        status["messages"].append(("❌", "AF2 params not found under ~/params"))
        status["ok"] = False

    return status


# =============================================================================
# Worker functions — wrap the notebook code, no UI in here
# =============================================================================
def download_target(team: str) -> tuple[str, int]:
    """Download and (optionally) truncate the target PDB. Returns (path, n_residues)."""
    target = TARGETS[team]
    Path("inputs").mkdir(exist_ok=True)
    input_pdb = f"inputs/{team}.pdb"

    # Download
    if target["source"] == "pdb":
        url = f"https://files.rcsb.org/download/{target['pdb_id']}.pdb"
    else:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{target['uniprot']}-F1-model_v4.pdb"

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


def validate_with_af2(input_pdb: str, designs: list[dict], chain: str, team: str) -> list[dict]:
    """Run AlphaFold2 on each designed sequence. Adds plddt/rmsd/ptm + saves PDBs."""
    from colabdesign.af import mk_af_model

    af = mk_af_model(protocol="fixbb", use_templates=False, num_recycles=3)
    Path("outputs").mkdir(exist_ok=True)

    validated = []
    progress = st.progress(0.0, text="Validating designs with AlphaFold2...")
    for i, d in enumerate(designs):
        af.prep_inputs(pdb_filename=input_pdb, chain=chain)
        af.set_seq(seq=d["sequence"])
        af.predict(num_recycles=3, verbose=False)

        d["plddt"] = float(af.aux["log"]["plddt"])
        d["rmsd"]  = float(af.aux["log"].get("rmsd", -1))
        d["ptm"]   = float(af.aux["log"].get("ptm", -1))

        af_path = f"outputs/{team}_design{d['idx']}_af.pdb"
        af.save_pdb(af_path)
        d["af_pdb"] = af_path

        validated.append(d)
        progress.progress(
            (i + 1) / len(designs),
            text=f"Validated {i + 1}/{len(designs)} designs",
        )
    progress.empty()
    return validated


def render_structure(orig_pdb: str, design_pdb: str | None = None) -> str:
    """Build a py3Dmol view as HTML for embedding via st.components.v1.html."""
    import py3Dmol

    view = py3Dmol.view(width=820, height=480)
    with open(orig_pdb) as f:
        view.addModel(f.read(), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})

    if design_pdb and os.path.isfile(design_pdb):
        with open(design_pdb) as f:
            view.addModel(f.read(), "pdb")
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
            help="More designs = more diversity to compare, but each adds ~30s for AF2 validation.",
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


def render_results_table(designs: list[dict]) -> None:
    """Render the validated-designs results."""
    rows = []
    for d in designs:
        consistent = d.get("plddt", 0) > 0.7 and (d.get("rmsd", -1) < 2.5 if d.get("rmsd", -1) >= 0 else True)
        verdict = "✅ self-consistent" if consistent else "⚠️ divergent"
        rows.append({
            "Design":  f"#{d['idx'] + 1}",
            "MPNN score": f"{d['score']:.3f}",
            "pLDDT":  f"{d['plddt']:.2f}",
            "RMSD (Å)": f"{d['rmsd']:.2f}" if d['rmsd'] >= 0 else "—",
            "pTM":    f"{d['ptm']:.2f}" if d['ptm'] >= 0 else "—",
            "Verdict": verdict,
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
            f"with AlphaFold2. Expect ~10s per design for MPNN + ~30s per design for AF2."
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
            validated = validate_with_af2(
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

    render_results_table(st.session_state.validated)

    st.markdown("**Visualize one of the designs:**")
    pick = st.selectbox(
        "Design to inspect",
        options=range(len(st.session_state.validated)),
        format_func=lambda i: (
            f"Design #{i + 1}  ·  pLDDT {st.session_state.validated[i]['plddt']:.2f}  ·  "
            f"RMSD {st.session_state.validated[i]['rmsd']:.2f} Å"
        ),
        label_visibility="collapsed",
    )

    chosen = st.session_state.validated[pick]
    html = render_structure(st.session_state.input_pdb, chosen["af_pdb"])
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
