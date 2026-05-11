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
        # The catalytic triad — three residues that perform the bond-cleaving
        # chemistry on PET. Conserved across all serine hydrolases.
        "catalytic_residues": "A160,A206,A237",
        "catalytic_labels":   {"A160": "Ser160", "A206": "Asp206", "A237": "His237"},
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
        # ZAR1's Walker A motif (GxxxxGK[S/T]) is the canonical ATP-binding
        # signature of NB-ARC domains. Empirically (from the MSA conservation
        # plot), it sits at residues 132-138 in our truncated coordinates:
        # G132 ... G137 K138 (T139). K138 is the catalytic lysine that
        # coordinates the ATP β-phosphate.
        "catalytic_residues": "A132,A137,A138",
        "catalytic_labels":   {
            "A132": "Walker A G1",
            "A137": "Walker A G2",
            "A138": "Walker A K (ATP)",
        },
        # ZAR1-specific feature: variants let advanced participants compare
        # the truncated (1-200, divergent prediction) vs extended (1-450,
        # complete NB-ARC) designs side-by-side. The "Try the extended version"
        # button on the results page swaps in the extended range, runs design+
        # validation again, and the page then shows both variants in tabs.
        "variants": {
            "truncated": {"range": (1, 200), "label": "Truncated (1-200)"},
            "extended":  {"range": (1, 450), "label": "Extended (1-450)"},
        },
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
        # CarRP has two catalytic domains. We mark a representative residue
        # from each: the DXXXD motif in the phytoene synthase domain (~D420)
        # and a conserved cyclase residue (~E118). These are approximations —
        # if the MSA shows other positions as more conserved, update from data.
        "catalytic_residues": "A118,A420,A424",
        "catalytic_labels":   {"A118": "cyclase E", "A420": "PSY DXXXD", "A424": "PSY DXXXD"},
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
        # Variant tracking (ZAR1-only feature; other teams ignore these).
        # When the participant clicks "Try the extended version", we snapshot
        # the current state into variants["truncated"] and swap the active
        # state over to the extended range. Both stay in memory so the
        # comparison view can render them side-by-side via tabs.
        #
        # variants[name] is either None or a dict with the same shape as the
        # top-level state: input_pdb, n_residues, designs, validated, range.
        "variants":       {"truncated": None, "extended": None},
        # active_variant tracks which variant the rest of the app is
        # currently designing/validating against. Defaults to "truncated"
        # because that's what ZAR1's "range" field in TARGETS points to,
        # and matches the workshop's default 200-aa flow.
        "active_variant": "truncated",
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
    """
    Download and (optionally) truncate the target PDB. Returns (path, n_residues).

    Variant-aware: for teams with variants defined (ZAR1), the file is saved
    under inputs/<team>_<variant>.pdb so the two variants never collide on
    disk. Range is resolved through get_active_range(), which respects the
    active variant.
    """
    target = TARGETS[team]
    Path("inputs").mkdir(exist_ok=True)

    # For variant-aware teams, include the variant name in the filename so
    # the truncated and extended PDBs coexist on disk.
    if has_variants(team):
        variant_suffix = f"_{st.session_state.active_variant}"
    else:
        variant_suffix = ""
    input_pdb = f"inputs/{team}{variant_suffix}.pdb"
    raw_pdb = f"inputs/{team}{variant_suffix}_raw.pdb"

    # Download (cache: skip if already on disk with non-trivial size)
    if not (Path(raw_pdb).is_file() and os.path.getsize(raw_pdb) > 1000):
        if target["source"] == "pdb":
            url = f"https://files.rcsb.org/download/{target['pdb_id']}.pdb"
        else:
            url = _afdb_pdb_url(target["uniprot"])

        result = subprocess.run(f"wget -q {url} -O {raw_pdb}", shell=True)
        if result.returncode != 0 or not os.path.isfile(raw_pdb) or os.path.getsize(raw_pdb) < 1000:
            size = os.path.getsize(raw_pdb) if os.path.isfile(raw_pdb) else 0
            raise RuntimeError(f"Download failed: exit={result.returncode}, size={size} bytes")

    # Resolve the range for the currently-active variant (None for teams
    # without variants, which means "use full chain"). For ZAR1 truncated
    # this is (1, 200); for ZAR1 extended this is (1, 450).
    active_range = get_active_range(team)

    # Build the variant-specific PDB by chain-filtering and range-truncating
    with open(raw_pdb) as fin, open(input_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] != target["chain"]:
                    continue
                if active_range is not None:
                    lo, hi = active_range
                    try:
                        if not (lo <= int(line[22:26]) <= hi):
                            continue
                    except ValueError:
                        pass
            fout.write(line)

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


# =============================================================================
# MSA inspection — read cached a3m, compute conservation, render the panel
# =============================================================================
# These helpers power the optional "Explore the MSA" expander shown after target
# load. The a3m file is built by msa_cache (during setup_msa_cache.sh prewarm
# or on first design-validation run) and lives at ~/msa_cache/<key>.a3m.
# =============================================================================

# Standard 20 amino acids — used for the conservation computation. Gaps in
# the MSA are ignored (treated as missing data, not a 21st amino acid).
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _find_a3m_path(cache_key: str) -> "Path | None":
    """
    Locate the cached MSA file for this target.

    Thin wrapper around msa_cache.cache_path(). Returns None if the file
    doesn't exist (caller's responsibility to check).
    """
    import msa_cache
    candidate = msa_cache.cache_path(cache_key)
    return candidate if candidate.is_file() else None


def read_msa_a3m(a3m_path) -> list[str]:
    """
    Parse an a3m file into a list of sequences, all aligned to the query.

    a3m format quirks we handle:
      - Lines starting with '#' are ColabFold metadata (e.g. '#265\t1' on the
        first line of monomer a3ms). Skipped.
      - Lines starting with '>' are FASTA-style headers; skipped, but they
        delimit records.
      - Lowercase letters in non-query sequences indicate insertions relative
        to the query — we strip them so all sequences match the query length.
      - Sequences may wrap across multiple lines; we concatenate until the
        next '>' header.

    The query (first record) sets the canonical length. Any record that
    doesn't match the query length after stripping insertions is dropped —
    it usually means the a3m was concatenated from multiple databases with
    inconsistent insertion handling.
    """
    sequences: list[str] = []
    current: list[str] = []
    with open(a3m_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("#"):
                # a3m metadata (e.g. "#265\t1") — skip, don't flush current
                continue
            if line.startswith(">"):
                # New record begins — flush whatever we'd accumulated
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                # Sequence line — strip insertions (lowercase letters)
                current.append("".join(c for c in line if not c.islower()))
        if current:
            sequences.append("".join(current))

    if not sequences:
        return []
    # Query length = length of the FIRST record (the original input sequence).
    # Drop any record that doesn't match — those are usually malformed entries
    # from a3m concatenation across databases.
    query_len = len(sequences[0])
    return [s for s in sequences if len(s) == query_len]


def compute_conservation(sequences: list[str]) -> list[float]:
    """
    Per-position conservation score, expressed as the frequency of the
    most common amino acid at that position. Range: 1/20 (least conserved)
    to 1.0 (perfectly conserved).

    We use most-common-frequency rather than Shannon entropy because it's
    more intuitive to participants ("80% of organisms have a serine here"
    is clearer than "entropy = 0.8 bits"). For workshop purposes the two
    metrics tell the same story.

    Gaps are ignored — they're treated as missing data, not as a 21st
    amino acid. This matches how biologists typically read conservation.
    """
    if not sequences:
        return []
    query_len = len(sequences[0])
    conservation: list[float] = []
    for pos in range(query_len):
        # Count amino acids at this position across all sequences
        col = [s[pos] for s in sequences if s[pos] in _AA_ALPHABET]
        if not col:
            conservation.append(0.0)
            continue
        # Most common AA's frequency
        counts: dict[str, int] = {}
        for aa in col:
            counts[aa] = counts.get(aa, 0) + 1
        max_count = max(counts.values())
        conservation.append(max_count / len(col))
    return conservation


# =============================================================================
# Variant snapshot/restore — ZAR1 truncated vs extended comparison
# =============================================================================
# These helpers let us flip between two variants of the same team without
# losing the previous variant's results. The flow:
#
#   1. Participant runs default flow → state has input_pdb / designs / validated
#   2. They click "Try the extended version" → capture_variant("truncated")
#      copies the current state into st.session_state.variants["truncated"]
#   3. We clear top-level state and download the extended target → flow runs
#      again, populating new top-level state
#   4. On completion, capture_variant("extended") also stores it
#   5. Comparison view reads both variants out of st.session_state.variants
# =============================================================================

def has_variants(team: str) -> bool:
    """True if this team defines alternate variants in TARGETS."""
    return "variants" in TARGETS.get(team, {})


def capture_variant(variant_name: str) -> None:
    """
    Snapshot the current top-level state into st.session_state.variants[name].

    Called right after a design+validation completes, so the variants dict
    always reflects the latest results for each variant the user has run.

    Defensive check: if the current top-level n_residues doesn't plausibly
    match this variant's expected range, refuse to save and warn. This
    prevents the bug where the truncated results accidentally get saved as
    "extended" if the user forgets to re-run validation after switching.
    """
    team = st.session_state.team
    n_res_now = st.session_state.n_residues

    # Sanity check: is the current top-level state actually a fresh run for
    # the named variant? If we're saving as "extended" but n_residues is
    # still the truncated count, something is inconsistent.
    if has_variants(team) and n_res_now is not None:
        expected_range = TARGETS[team]["variants"][variant_name]["range"]
        if expected_range is not None:
            # Allow some slack — structures often have disordered residues
            # missing, so "expected 200" can really mean 140-180 in practice.
            # The point is to detect when truncated (~145) accidentally
            # gets saved as extended (which should be ~350-450).
            expected_min = expected_range[0]
            expected_max = expected_range[1]
            # If our current n_residues is much smaller than the lower bound
            # of the expected range, refuse to save.
            if n_res_now < expected_min + (expected_max - expected_min) * 0.3:
                # Looks like we'd be saving truncated-shaped data as the
                # extended variant (or vice versa). Refuse.
                st.warning(
                    f"⚠️ Refusing to save: top-level state has {n_res_now} residues "
                    f"but variant '{variant_name}' expects roughly "
                    f"{expected_min}-{expected_max}. Re-run **Step 2** (Generate & "
                    f"validate) before snapshotting — the validation step is what "
                    f"refreshes the designs for the active variant."
                )
                return

    st.session_state.variants[variant_name] = {
        "input_pdb":  st.session_state.input_pdb,
        "n_residues": n_res_now,
        "designs":    st.session_state.designs,
        "validated":  st.session_state.validated,
    }


def switch_to_variant(team: str, variant_name: str) -> None:
    """
    Make `variant_name` the active variant: snapshot the current results
    (if any) into the variants dict, restore the requested variant's results
    if they exist, otherwise clear top-level state so the flow runs fresh.

    The PDB filename used for the new variant includes the variant in its
    name (e.g. inputs/ZAR1_extended.pdb), so the two PDBs never collide on
    disk. download_target() picks this up from the active variant when ZAR1.
    """
    # First, snapshot whatever's currently active so we don't lose it
    if st.session_state.validated is not None:
        capture_variant(st.session_state.active_variant)

    st.session_state.active_variant = variant_name

    # If the requested variant has cached results, restore them as top-level
    # state; otherwise, clear top-level so the participant has to re-run.
    cached = st.session_state.variants.get(variant_name)
    if cached is not None:
        st.session_state.update(
            input_pdb=cached["input_pdb"],
            n_residues=cached["n_residues"],
            designs=cached["designs"],
            validated=cached["validated"],
            target_loaded=True,
        )
    else:
        st.session_state.update(
            input_pdb=None,
            n_residues=None,
            designs=None,
            validated=None,
            target_loaded=False,
        )


def get_active_range(team: str) -> "tuple[int, int] | None":
    """
    Resolve the range tuple to use for the currently-active variant of `team`.

    Falls back to TARGETS[team]['range'] for teams that don't define variants
    (PETase, CarRP) — same behavior as before this feature existed.
    """
    target = TARGETS[team]
    if has_variants(team):
        variant_name = st.session_state.active_variant
        return target["variants"][variant_name]["range"]
    return target.get("range")


def render_msa_inspector(team: str, input_pdb: str) -> None:
    """
    Render the MSA inspection panel: conservation plot + top conserved positions.

    Shows nothing destructive if the cache doesn't exist — just an info message
    pointing the user at the design step (which builds the MSA as a side effect).
    """
    import msa_cache
    import matplotlib.pyplot as plt

    target = TARGETS[team]
    chain = target["chain"]

    wt_seq = extract_wt_sequence(input_pdb, chain)
    if not wt_seq:
        st.warning("Could not extract WT sequence — skipping MSA panel.")
        return

    cache_key = msa_cache.cache_key(team, wt_seq)

    # cache_exists() checks BOTH the a3m and the metadata sidecar. If only
    # one is present, the cache entry is partial and we treat it as missing.
    if not msa_cache.cache_exists(cache_key):
        st.info(
            "MSA not built yet for this target. It will be generated automatically "
            "when you run the design step (~3 min, one-time per target)."
        )
        return

    a3m_path = msa_cache.cache_path(cache_key)

    # Parse the MSA and compute conservation
    sequences = read_msa_a3m(a3m_path)
    if len(sequences) < 2:
        st.warning(
            f"MSA contains only {len(sequences)} sequence(s) — that's not enough "
            "for meaningful conservation analysis. The MSA may have failed to build "
            f"properly. Try deleting ~/msa_cache/{cache_key}.a3m and re-running validation."
        )
        return

    conservation = compute_conservation(sequences)

    # ─── Top-line summary ────────────────────────────────────────────────────
    n_seqs = len(sequences)
    n_residues = len(conservation)
    perfectly_conserved = sum(1 for c in conservation if c >= 0.99)
    highly_conserved = sum(1 for c in conservation if c >= 0.90)

    cols = st.columns(4)
    cols[0].metric("MSA depth", f"{n_seqs:,}")
    cols[1].metric("Length", f"{n_residues} aa")
    cols[2].metric("Perfectly conserved", f"{perfectly_conserved}")
    cols[3].metric("Highly conserved (≥90%)", f"{highly_conserved}")

    # ─── Why this matters (collapsible) ──────────────────────────────────────
    with st.expander("ℹ️ Why the MSA matters for AlphaFold", expanded=False):
        st.markdown(
            "AlphaFold doesn't just look at your protein's sequence — it looks at "
            "**hundreds or thousands of related sequences** from across all of life. "
            "When two positions consistently mutate together across evolution "
            "(e.g., a positively charged residue at position X always pairs with "
            "a negatively charged residue at position Y), that's strong evidence "
            "those positions are physically close in 3D — a salt bridge.\n\n"
            "Conservation tells you something different: positions that **never** "
            "change are usually critical. They're either part of the active site, "
            "essential for folding, or both. If you redesign a perfectly conserved "
            "residue, you should not be surprised when the protein stops working.\n\n"
            "When you validate your designs in the next step, AlphaFold reuses this "
            "exact MSA — your designed sequence steps into the query slot, and the "
            "alignment of evolutionary relatives provides the structural prior."
        )

    # ─── Conservation plot ───────────────────────────────────────────────────
    st.markdown("##### Conservation along the sequence")

    fig, ax = plt.subplots(figsize=(11, 2.5))
    positions = list(range(1, n_residues + 1))
    ax.bar(positions, conservation, width=1.0, color="#4477AA",
           edgecolor="none", alpha=0.85)

    # Annotate catalytic / functional residues
    catalytic_resnums: list[int] = []
    catalytic_str = target.get("catalytic_residues", "")
    if catalytic_str:
        for res in catalytic_str.split(","):
            res = res.strip()
            try:
                resnum = int(res[1:])
                catalytic_resnums.append(resnum)
            except ValueError:
                continue

    # Apply chain truncation offset if needed. For variant-aware teams (ZAR1),
    # this resolves to the active variant's range; for others it's just the
    # static "range" field from TARGETS.
    rng = get_active_range(team)
    if rng is not None:
        lo, hi = rng
        # Catalytic residues outside the truncation are dropped (they won't
        # be in this sequence). Inside ones get remapped to 1-indexed coords.
        adjusted = []
        for r in catalytic_resnums:
            if lo <= r <= hi:
                adjusted.append(r - lo + 1)
        catalytic_resnums = adjusted

    # Draw markers and labels
    labels_map = target.get("catalytic_labels", {})
    for r in catalytic_resnums:
        if r < 1 or r > n_residues:
            continue
        ax.axvline(r, color="#CC3311", alpha=0.6, lw=1.2, zorder=3)
        ax.scatter([r], [conservation[r - 1]], color="#CC3311",
                   s=40, zorder=4, edgecolor="white", lw=0.8)

    ax.set_xlabel("Residue position", fontsize=10)
    ax.set_ylabel("Conservation\n(freq. of most common AA)", fontsize=9)
    ax.set_xlim(0.5, n_residues + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    if catalytic_resnums:
        st.caption(
            "🔴 Red markers = known functional residues for this protein. "
            "If the MSA reflects real biology, these should be among the most "
            "conserved positions."
        )

    # ─── Top conserved positions list ────────────────────────────────────────
    st.markdown("##### Most conserved positions")

    # Compute per-position MSA depth (count of non-gap, non-X residues).
    # We use this to filter out positions with artificially high conservation
    # caused by shallow homolog coverage — typically at the C-terminus when
    # most homologs are truncated relative to the query.
    depth = [
        sum(1 for s in sequences if s[pos] in _AA_ALPHABET)
        for pos in range(n_residues)
    ]
    # Median depth across the full sequence sets the "what counts as normal
    # coverage" threshold. Positions with depth below half of that get
    # excluded from the top-conserved ranking (their conservation is unreliable).
    sorted_depth = sorted(depth)
    median_depth = sorted_depth[len(sorted_depth) // 2]
    min_depth = max(median_depth // 2, 2)  # never go below 2 sequences

    # Apply range offset to display the "real" residue numbers, not truncated.
    # E.g., ZAR1 uses range=(1, 200), so position 1 in our local index maps
    # to position 1 in the original PDB; CarRP has range=None, so no offset.
    offset = 0
    if rng is not None:
        offset = rng[0] - 1

    chain_letter = target["chain"]
    catalytic_set = set(catalytic_resnums)

    def is_trivially_conserved(pos_1indexed: int) -> bool:
        """
        Filter positions that are conserved for boring (non-biological) reasons:
          - Position 1 is always Met because translation initiates with M-Met.
            That's not biology, that's the ribosome.
          - Positions with <50% of median MSA depth have shallow homolog
            coverage; their "conservation" reflects a small sample, not a
            strong evolutionary constraint.
        """
        if pos_1indexed == 1:
            return True
        if depth[pos_1indexed - 1] < min_depth:
            return True
        return False

    # Rank by conservation, excluding trivially-conserved positions but
    # preserving them for diagnostic display below if you want to look.
    all_ranked = sorted(
        enumerate(zip(conservation, wt_seq), start=1),
        key=lambda x: -x[1][0],
    )
    real_ranked = [
        (pos, (cons, aa)) for pos, (cons, aa) in all_ranked
        if not is_trivially_conserved(pos)
    ]
    top_n = 10
    top_positions = real_ranked[:top_n]
    top_positions_set = {p for p, _ in top_positions}

    # Always include catalytic residues, even if they're outside the top-N.
    # They get appended to the table after the top conserved entries.
    #
    # Three states for any annotated catalytic residue:
    #   1. Already in top_positions  → no extra row needed
    #   2. In the parsed sequence but not in top_positions → append with rank "—"
    #   3. Outside the parsed sequence (e.g., disordered region missing from
    #      the PDB, or beyond the truncation range) → note in caption,
    #      don't try to fake a table row for data we don't have
    catalytic_extras: list[tuple[int, tuple[float, str]]] = []
    catalytic_missing: list[int] = []   # state 3: annotated but no data
    if catalytic_set:
        for pos in sorted(catalytic_set):
            if pos in top_positions_set:
                continue  # already in the main table (state 1)
            if pos < 1 or pos > n_residues:
                catalytic_missing.append(pos)  # state 3
                continue
            cons = conservation[pos - 1]
            aa = wt_seq[pos - 1]
            catalytic_extras.append((pos, (cons, aa)))  # state 2

    # Build the table. Columns:
    #   Rank      — top-N ranking, or "—" for catalytic positions that didn't
    #                make the top list
    #   Position  — numeric residue position (matches figure x-axis directly)
    #   Residue   — one-letter AA + position (e.g., "S160")
    #   Conservation — percent
    #   Note      — catalytic label or empty
    lines = [
        "| Rank | Position | Residue | Conservation | Note |",
        "|---|---|---|---|---|",
    ]

    def format_row(rank_label: str, pos_1indexed: int, cons: float, aa: str) -> str:
        real_resnum = pos_1indexed + offset
        residue_key = f"{chain_letter}{real_resnum}"
        label = labels_map.get(residue_key, "")
        if label:
            note = f"🔴 {label}"
        elif pos_1indexed in catalytic_set:
            note = "🔴 catalytic"
        else:
            note = ""
        return f"| {rank_label} | {real_resnum} | {aa}{real_resnum} | {cons:.1%} | {note} |"

    for rank, (pos_1indexed, (cons, aa)) in enumerate(top_positions, start=1):
        lines.append(format_row(str(rank), pos_1indexed, cons, aa))

    # Catalytic residues that didn't rank in the top — append with "—" rank
    if catalytic_extras:
        for pos_1indexed, (cons, aa) in catalytic_extras:
            lines.append(format_row("—", pos_1indexed, cons, aa))

    st.markdown("\n".join(lines))

    # Diagnostic info about what was filtered, for transparency
    n_filtered_depth = sum(1 for d in depth if d < min_depth)
    filter_notes = []
    if n_filtered_depth > 0:
        filter_notes.append(
            f"{n_filtered_depth} positions excluded for shallow MSA coverage "
            f"(<{min_depth} sequences; median is {median_depth})"
        )
    filter_notes.append(
        "Position 1 (always Met) excluded as a translation artifact"
    )
    # If any annotated catalytic residues are outside the parsed sequence
    # (e.g., disordered loops missing from the PDB), call that out explicitly
    # so participants don't wonder why a labeled residue isn't visible.
    if catalytic_missing:
        # Translate back to original residue numbers for display
        missing_real = [p + offset for p in catalytic_missing]
        filter_notes.append(
            f"Catalytic residues outside resolved structure: "
            f"{', '.join(str(p) for p in missing_real)} "
            f"(sequence only covers residues {1 + offset}–{n_residues + offset})"
        )
    st.caption(
        "Positions that never change across hundreds of related proteins are doing "
        "essential work — either catalysis, folding, or both. "
        + ("If catalytic residues didn't make the top 10, they're listed below "
           "with rank '—'. " if catalytic_extras else "")
        + " · ".join(filter_notes) + "."
    )


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
                     chain: str = "A",
                     view_style: str = "cartoon",
                     surface_color: str = "plddt") -> str:
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

    Visualization parameters:
    - view_style: one of "cartoon", "cartoon+surface", "surface". Controls
      what's rendered for the design (the WT target is always cartoon-only
      in light gray as a reference scaffold).
    - surface_color: one of "plddt", "hydrophobicity". Only meaningful when
      view_style includes a surface. pLDDT uses AF2's B-factor column with
      a red→blue gradient matching the cartoon color scheme. Hydrophobicity
      colors by residue chemistry: yellow for nonpolar residues, lightblue
      for polar/charged.
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

        # Cartoon layer for the design — colored by pLDDT (B-factor) using
        # the existing red→orange→yellow→green→blue gradient. Drawn only
        # when view_style is "cartoon" or "cartoon+surface". When view_style
        # is "surface", we hide the cartoon entirely so the surface is the
        # only thing on screen.
        if view_style in ("cartoon", "cartoon+surface"):
            view.setStyle({"model": 1}, {"cartoon": {
                "colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 90}
            }})
        else:  # "surface" — hide cartoon for design
            view.setStyle({"model": 1}, {})

        # Surface layer for the design — only when view_style requests it.
        # The surface uses py3Dmol's SAS (solvent-accessible surface) which
        # is a smooth molecular envelope. Opacity matters: opaque surfaces
        # hide everything inside, transparent surfaces let the cartoon show
        # through ("cartoon+surface" mode).
        if view_style in ("cartoon+surface", "surface"):
            # Opacity differs by mode: see-through when overlaid on cartoon,
            # fully visible when the surface is the main subject.
            opacity = 0.55 if view_style == "cartoon+surface" else 0.92

            if surface_color == "plddt":
                # Color the surface by AF2 confidence — matches the cartoon's
                # color scheme so the visual story stays coherent. The
                # gradient floor/ceiling (50/90) is the same as the cartoon
                # gradient above to keep the two color schemes consistent.
                surface_style = {
                    "opacity": opacity,
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    },
                }
            else:  # "hydrophobicity"
                # Kyte-Doolittle informed coloring: yellow for hydrophobic
                # residues (A, V, L, I, M, F, W, P, G — i.e. the ones that
                # tend to bury away from solvent), light blue for polar and
                # charged. This is the surface coloring that reveals binding
                # pockets and membrane interfaces: yellow patches are where
                # ligands dock; blue patches face water.
                HYDROPHOBIC = {
                    "ALA", "VAL", "LEU", "ILE", "MET",
                    "PHE", "TRP", "PRO", "GLY", "CYS",
                }
                ALL_AA = HYDROPHOBIC | {
                    "ARG", "LYS", "ASP", "GLU",
                    "SER", "THR", "ASN", "GLN",
                    "TYR", "HIS",
                }
                color_map = {
                    res: ("#e8c547" if res in HYDROPHOBIC else "#7ec4ff")
                    for res in ALL_AA
                }
                surface_style = {
                    "opacity": opacity,
                    "colorscheme": {
                        "prop": "resn",
                        "map": color_map,
                    },
                }

            # Add the surface to the design model only (not the WT scaffold).
            # The {"model": 1} selection limits the surface to the AF2
            # prediction; the gray WT cartoon stays unobscured as a reference.
            view.addSurface(py3Dmol.SAS, surface_style, {"model": 1})

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
        # If team changed, clear stale state — including any ZAR1 variants
        # that might have been captured during a previous session with that team.
        if team != st.session_state["team"]:
            st.session_state.update(
                team=team, target_loaded=False, input_pdb=None,
                n_residues=None, designs=None, validated=None,
                variants={"truncated": None, "extended": None},
                active_variant="truncated",
            )

        st.markdown(TARGETS[team]["blurb"])

        st.divider()
        st.markdown("**Design parameters**")

        num_designs = st.slider(
            "Number of designs",
            min_value=2, max_value=8, value=3, step=1,
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
    """
    Render the Organism/Function/Source card row above Step 1.

    Uses a custom Markdown layout instead of st.metric because metric widgets
    enforce single-line values — long function descriptions like
    "NLR immune receptor — forms pentameric resistosome" get visually cut off
    or truncated with ellipsis, losing information. The label/value pairs
    below use normal CSS line wrapping, so the full text is always readable
    regardless of length. Field labels are styled to look like st.metric's
    labels (small, muted) so the visual rhythm of the page is preserved.
    """
    target = TARGETS[team]

    # Strip the parenthetical from organism ("Arabidopsis thaliana (plant)" →
    # "Arabidopsis thaliana") for visual density. The plant/animal/fungal
    # context is already conveyed by the team's blurb.
    organism = target["organism"].split(" (")[0]
    function = target["function"]
    src_label = (
        f"PDB {target.get('pdb_id', '')}"
        if target["source"] == "pdb"
        else f"AF-{target.get('uniprot', '')}"
    )

    # Inline style: label small/muted/uppercase like st.metric, value at
    # readable size with wrapping. line-height tightened so two-line values
    # don't push the cards visually unbalanced.
    label_style = (
        "font-size: 0.78rem; color: rgba(250,250,250,0.55); "
        "text-transform: none; margin-bottom: 0.25rem; font-weight: 400;"
    )
    value_style = (
        "font-size: 1.05rem; line-height: 1.35; font-weight: 500; "
        "color: rgba(250,250,250,0.95);"
    )

    cols = st.columns([1, 1.4, 1])  # give Function a bit more room
    with cols[0]:
        st.markdown(
            f'<div style="{label_style}">Organism</div>'
            f'<div style="{value_style}">{organism}</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div style="{label_style}">Function</div>'
            f'<div style="{value_style}">{function}</div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div style="{label_style}">Source</div>'
            f'<div style="{value_style}">{src_label}</div>',
            unsafe_allow_html=True,
        )


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


def render_variant_comparison(team: str) -> None:
    """
    Render the truncated-vs-extended comparison summary for ZAR1.

    Shows aggregate metrics for each variant (mean pLDDT, mean pTM, mean RMSD,
    verdict distribution) in a compact table so participants can see at a
    glance how the two cases differ. Detailed per-design results live below
    in the tabs.
    """
    target = TARGETS[team]
    variants = st.session_state.variants

    summary_rows = []
    for variant_name, variant_data in variants.items():
        if variant_data is None:
            continue
        validated = variant_data["validated"]
        n_res = variant_data["n_residues"]
        variant_label = target["variants"][variant_name]["label"]

        classifications = [classify_design(d, n_res)[0] for d in validated]
        # Verdict distribution: most common verdict + count
        from collections import Counter
        cls_counts = Counter(classifications)
        verdict_summary = ", ".join(
            f"{count} {label}" for label, count in cls_counts.most_common()
        )

        mean_plddt = sum(d["plddt"] for d in validated) / len(validated)
        mean_ptm = sum(d["ptm"] for d in validated if d["ptm"] >= 0)
        n_ptm = sum(1 for d in validated if d["ptm"] >= 0)
        mean_ptm = mean_ptm / n_ptm if n_ptm else float("nan")
        mean_rmsd = sum(d["rmsd"] for d in validated if d["rmsd"] >= 0)
        n_rmsd = sum(1 for d in validated if d["rmsd"] >= 0)
        mean_rmsd = mean_rmsd / n_rmsd if n_rmsd else float("nan")

        summary_rows.append({
            "Variant":     variant_label,
            "Length":      f"{n_res} aa",
            "Mean pLDDT":  f"{mean_plddt:.2f}",
            "Mean pTM":    f"{mean_ptm:.3f}" if n_ptm else "—",
            "Mean RMSD (Å)": f"{mean_rmsd:.2f}" if n_rmsd else "—",
            "Verdicts":    verdict_summary,
        })

    st.markdown("##### Variants compared at a glance")
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    st.caption(
        "**Interpreting the comparison:** ZAR1 has three stacking failure modes "
        "(truncation + plant-NLR scarcity in training data + oligomer-as-monomer "
        "design). The truncated→extended swap fixes failure mode 1 but leaves "
        "modes 2 and 3 in place.\n\n"
        "- **If extended's metrics are clearly better** (higher pLDDT, lower RMSD), "
        "truncation was the dominant problem.\n"
        "- **If extended's metrics are similar** (as is often the case with only "
        "2-4 designs), the other failure modes are also significant — extending "
        "the domain alone isn't enough.\n\n"
        "Either result is a real scientific finding. With small N (2-4 designs) "
        "the comparison is noisy; for a publishable claim you'd want N≥10 per "
        "variant. For workshop purposes, the conclusion is qualitative: did the "
        "verdict distribution change?"
    )


def render_variant_detail(team: str, variant_name: str | None = None) -> None:
    """
    Render the detailed view for ONE variant: results table, divergent/marginal
    callouts, structure viewer, designed sequence.

    Reads from the variant-specific snapshot in st.session_state.variants if
    variant_name is given, otherwise reads from top-level state (single-variant
    flow for non-ZAR1 teams).
    """
    target = TARGETS[team]

    if variant_name is not None:
        data = st.session_state.variants[variant_name]
        if data is None:
            st.info(f"No results for the {variant_name} variant yet.")
            return
        input_pdb  = data["input_pdb"]
        n_res      = data["n_residues"]
        validated  = data["validated"]
        variant_label = target["variants"][variant_name]["label"]
    else:
        input_pdb  = st.session_state.input_pdb
        n_res      = st.session_state.n_residues
        validated  = st.session_state.validated
        variant_label = None  # single-variant flow doesn't show a header

    # Optional variant header to remind which one we're looking at
    if variant_label:
        st.markdown(f"**Variant:** {variant_label}  ·  **Length:** {n_res} aa")

    render_results_table(validated, n_res)

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
        else:
            st.info(
                f"Some designs are flagged as divergent. That's unusual for {team} — "
                "try a lower sampling temperature or rerun. If it persists, inspect the "
                "design sequence for chemistry that doesn't match the fold."
            )
    elif any_marginal:
        st.info(
            "**'Marginal' verdict.** The fold is broadly right (pLDDT ≥ 70) but tighter "
            "validation thresholds aren't met. Often a flexible terminus or one weak "
            "subdomain is dragging RMSD up while the core is fine — open the 3D viewer "
            "below and look for which region is colored red/orange (low pLDDT)."
        )

    # Structure viewer
    st.markdown("**Visualize one of the designs:**")

    # Visualization style controls. The radio widgets are placed side-by-side
    # in a 3-column row to keep vertical space tight. Each variant tab gets
    # its own keys so the user's choices persist independently per tab.
    style_key = f"view_style_{variant_name or 'single'}"
    color_key = f"surface_color_{variant_name or 'single'}"

    col_pick, col_style, col_color = st.columns([2, 1.4, 1.4])
    with col_pick:
        selectbox_key = f"design_picker_{variant_name or 'single'}"
        pick = st.selectbox(
            "Design to inspect",
            options=range(len(validated)),
            format_func=lambda i: (
                f"Design #{i + 1}"
                f"  ·  pLDDT {validated[i]['plddt']:.2f}"
                f"  ·  pTM {validated[i]['ptm']:.2f}"
                f"  ·  RMSD {validated[i]['rmsd']:.2f} Å"
            ),
            key=selectbox_key,
        )
    with col_style:
        view_style = st.radio(
            "View",
            options=["cartoon", "cartoon+surface", "surface"],
            format_func=lambda s: {
                "cartoon": "Cartoon",
                "cartoon+surface": "Cartoon + surface",
                "surface": "Surface only",
            }[s],
            key=style_key,
        )
    with col_color:
        # The color radio is only meaningful when a surface is showing.
        # Disable when in pure-cartoon mode so the UI doesn't pretend the
        # choice matters there.
        surface_disabled = (view_style == "cartoon")
        surface_color = st.radio(
            "Surface color",
            options=["plddt", "hydrophobicity"],
            format_func=lambda s: {
                "plddt": "By pLDDT",
                "hydrophobicity": "By hydrophobicity",
            }[s],
            key=color_key,
            disabled=surface_disabled,
        )

    chosen = validated[pick]
    html = render_structure(
        input_pdb,
        chosen["af_pdb"],
        chain=target["chain"],
        view_style=view_style,
        surface_color=surface_color,
    )
    st.components.v1.html(html, height=500)

    # Caption adapts to the active view mode. Each mode reveals a different
    # aspect of the design — cartoon for fold topology, surface for shape
    # and binding pockets, hydrophobicity for ligand-binding interpretation.
    if view_style == "cartoon":
        st.caption(
            "**Gray** = original target backbone.  "
            "**Colored cartoon** = AlphaFold2 prediction of the designed "
            "sequence (blue = high confidence, red = low). They should "
            "superimpose tightly if the design is self-consistent."
        )
    elif view_style == "cartoon+surface":
        if surface_color == "plddt":
            st.caption(
                "**Gray cartoon** = target backbone.  **Colored cartoon "
                "+ translucent surface** = designed sequence's AF2 "
                "prediction; both colored by pLDDT (red = low confidence, "
                "blue = high). The surface shows the shape and bulk of the "
                "design; the cartoon shows the underlying fold."
            )
        else:
            st.caption(
                "**Gray cartoon** = target backbone.  **Translucent "
                "surface** colored by residue chemistry — **yellow** = "
                "hydrophobic (A, V, L, I, M, F, W, P, G, C), **light blue** "
                "= polar and charged. Yellow patches mark where ligands "
                "would bind or where the protein contacts hydrophobic "
                "partners; blue surfaces face the solvent."
            )
    else:  # "surface"
        if surface_color == "plddt":
            st.caption(
                "**Gray cartoon** = target backbone.  **Solid colored "
                "surface** = AF2 prediction of the design, colored by "
                "pLDDT confidence. Red regions are where AF2 was uncertain "
                "about the predicted position."
            )
        else:
            st.caption(
                "**Gray cartoon** = target backbone.  **Solid surface** "
                "colored by residue chemistry. **Yellow** patches reveal "
                "hydrophobic binding pockets and protein-protein interface "
                "surfaces; **light blue** is solvent-exposed polar/charged "
                "residues. For ZAR1: look for a yellow patch around the "
                "Walker A motif (≈positions 132-138 of the truncated "
                "coordinates) — that's the ATP-binding pocket signature."
            )

    with st.expander("Show designed sequence"):
        st.code(chosen["sequence"], language=None)


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

    # ─── MSA Inspector (optional pedagogy expander) ─────────────────────────
    # Shown right after target load, before design. If the MSA cache exists
    # (workshop-day state — setup_msa_cache.sh prewarms it), displays a
    # conservation plot. Otherwise, a non-blocking note about lazy build.
    # Wrapped in an expander so it doesn't dominate the page for users who
    # want to push straight to design.
    with st.expander("🧬 Explore the multiple sequence alignment (optional)",
                     expanded=False):
        st.markdown(
            "Before AlphaFold predicts a structure, it searches the entire database "
            "of known protein sequences for relatives of your target. The collection "
            "of those sequences — the **multiple sequence alignment**, or MSA — is "
            "the single biggest reason modern structure prediction works as well as "
            "it does."
        )
        try:
            render_msa_inspector(team, st.session_state.input_pdb)
        except Exception as e:
            st.warning(f"MSA inspector failed to render: {e}")
            st.caption("This won't affect the design step — proceed normally.")

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
            # Snapshot this run into the variants dict so the comparison
            # view can pick it up later. No-op for teams without variants.
            if has_variants(team):
                capture_variant(st.session_state.active_variant)
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

    # Decide between single-variant view and comparison view.
    # Comparison view fires only for ZAR1 (the only team with variants defined)
    # AND only when BOTH variants have completed runs.
    both_variants_done = (
        has_variants(team)
        and st.session_state.variants.get("truncated") is not None
        and st.session_state.variants.get("extended") is not None
    )

    if both_variants_done:
        # ──────────────────────────────────────────────────────────────────
        # Comparison mode: summary table + tabs containing each variant's
        # full detail panel
        # ──────────────────────────────────────────────────────────────────
        render_variant_comparison(team)

        st.markdown("---")
        st.markdown("##### Per-variant detail")
        st.caption(
            "Each tab below contains the full results, structure viewer, and "
            "designed sequence for one variant. Switch between tabs to compare "
            "how the prediction quality changes with target length."
        )

        # Build tabs in a predictable order — truncated first, extended second
        ordered_variants = ["truncated", "extended"]
        tab_labels = [
            TARGETS[team]["variants"][v]["label"] for v in ordered_variants
        ]
        tabs = st.tabs(tab_labels)
        for tab, variant_name in zip(tabs, ordered_variants):
            with tab:
                render_variant_detail(team, variant_name)
    else:
        # ──────────────────────────────────────────────────────────────────
        # Single-variant mode (always for PETase/CarRP; for ZAR1 until both
        # variants are done)
        # ──────────────────────────────────────────────────────────────────
        render_variant_detail(team, variant_name=None)

        # For ZAR1 only: offer the "Try the extended version" button right
        # below the detail. The button triggers a variant switch — top-level
        # state is cleared, the extended target gets downloaded, and the
        # participant runs Step 2 again with the extended sequence. After
        # that, both variants are done and the next rerun lands in
        # comparison mode.
        if (
            has_variants(team)
            and st.session_state.active_variant == "truncated"
            and st.session_state.variants.get("extended") is None
        ):
            st.markdown("---")
            with st.container():
                st.markdown(
                    "##### 🔬 Try the extended version (1–450)"
                )
                st.markdown(
                    "You ran ZAR1 on the **truncated CC domain (1–200)** "
                    "and saw the prediction diverge. "
                    "**Now try the same workflow on the extended CC + NB-ARC "
                    "tandem (1–450).** Does pLDDT come up? Does the central "
                    "bundle stay aligned while the new C-terminal residues "
                    "form the missing nucleotide-binding pocket? This is the "
                    "obvious follow-up — a real hypothesis test about *why* "
                    "the truncated case failed."
                )
                st.warning(
                    "**Heads-up: this will take ~3-5× longer than the truncated run.** "
                    "Extended is 2.25× the residues; AF2's runtime scales worse than "
                    f"linearly with length. Expect ~{settings['num_designs'] * 5}–"
                    f"{settings['num_designs'] * 10} min for {settings['num_designs']} "
                    "designs. ProteinMPNN itself is still fast. You'll also wait for "
                    "a fresh MSA build (~3 min) because the longer sequence has its "
                    "own cache key."
                )
                if st.button(
                    "🚀 Run extended version",
                    type="primary",
                    disabled=st.session_state.running,
                    key="run_extended",
                ):
                    # Snapshot the current truncated run (it should already be
                    # in variants["truncated"] from the post-validate capture,
                    # but call this defensively in case state got reset).
                    if st.session_state.validated is not None:
                        capture_variant("truncated")
                    # Switch to the extended variant — clears top-level state
                    # so the target-load + design flow re-runs
                    switch_to_variant(team, "extended")
                    st.rerun()

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
