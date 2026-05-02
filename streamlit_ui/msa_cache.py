"""
MSA cache + ColabFold validation for the protein design workshop.

ARCHITECTURE
------------
Streamlit runs in the SE3nv conda env (ProteinMPNN side). ColabFold runs in a
separate `colabfold` env (MSA-based AF2). They communicate via files. This
module is the bridge: it lives in SE3nv, shells out to the colabfold env via
`conda run`, and returns parsed results.

CACHE STRATEGY
--------------
For each WORKSHOP TARGET we cache the MSA generated from its WT sequence under
~/msa_cache/<key>.a3m, with a sidecar <key>.json recording sequence + length +
date. To validate a DESIGNED sequence (which has the same length as the target
because ProteinMPNN fixbb keeps the backbone), we:

  1. Read the cached WT a3m
  2. Replace the first record (the query) with the designed sequence
  3. Hand the resulting a3m to colabfold_batch as input

ColabFold treats whatever is the first record of an a3m as the query and skips
the MSA server entirely. The aligned homologs in the rest of the file provide
the coevolutionary signal AF2 needs. This is an approximation — strictly the
homologs of the design might be different from those of the WT — but it's a
widely-used trick for design self-consistency validation, and it's the only
way to keep per-design validation under workshop time budget.

For ARBITRARY user-supplied targets, we run a one-shot colabfold MSA query for
the WT sequence and cache the result; subsequent designs against that target
reuse the cache.

NOT THREAD-SAFE. Streamlit runs single-threaded per session, which is fine
here, but don't call this from a multiprocessing pool.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

# =============================================================================
# Paths
# =============================================================================
CACHE_DIR = Path(os.path.expanduser("~/msa_cache"))
COLABFOLD_ENV = "colabfold"  # name of the conda env that has colabfold_batch

# How many AF2 models to run per design. Stage 2 used 5; for in-workshop
# per-design validation we use 3 (decision recorded in handoff).
NUM_MODELS = 3


# =============================================================================
# Errors
# =============================================================================
class MSACacheError(RuntimeError):
    """Raised when MSA generation, caching, or substitution fails."""


class ColabFoldError(RuntimeError):
    """Raised when the colabfold_batch subprocess fails."""


# =============================================================================
# Cache key + a3m helpers
# =============================================================================
def cache_key(name: str, sequence: str) -> str:
    """
    Build a stable cache key from a target name plus a sequence hash.

    Including the sequence hash means that if a target is renamed or its WT
    sequence is changed (e.g. CarRP range was extended), the cache invalidates
    automatically and a fresh MSA is generated.
    """
    seq_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:12]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return f"{safe_name}_{seq_hash}"


def _read_a3m(path: Path) -> list[tuple[str, str]]:
    """
    Parse an a3m file into (header, sequence) pairs.

    a3m format is FASTA-ish but allows lowercase letters in non-query records
    to mark insertions relative to the query. We preserve the lines verbatim.
    """
    records: list[tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            elif line:
                seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _write_a3m(records: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f"{header}\n{seq}\n")


def _substitute_query(cached_a3m: Path, design_seq: str, design_name: str,
                      out_path: Path) -> None:
    """
    Read cached a3m, replace the first record (query) with the designed
    sequence, write to out_path. The designed sequence MUST have the same
    length as the cached query (validated by caller).
    """
    records = _read_a3m(cached_a3m)
    if not records:
        raise MSACacheError(f"Cached a3m at {cached_a3m} is empty")

    cached_query_seq = records[0][1]
    if len(cached_query_seq) != len(design_seq):
        raise MSACacheError(
            f"Length mismatch: cached query is {len(cached_query_seq)} aa, "
            f"designed sequence is {len(design_seq)} aa. Cannot substitute "
            "without breaking the alignment."
        )

    # Replace just the first record; aligned homologs are kept intact.
    records[0] = (f">{design_name}", design_seq)
    _write_a3m(records, out_path)


# =============================================================================
# Cache management
# =============================================================================
def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.a3m"


def cache_meta_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_exists(key: str) -> bool:
    """An entry is present iff both the a3m and its metadata sidecar exist."""
    return cache_path(key).is_file() and cache_meta_path(key).is_file()


def load_cache_meta(key: str) -> dict:
    with open(cache_meta_path(key)) as f:
        return json.load(f)


def _save_cache_meta(key: str, sequence: str, source: str) -> None:
    meta = {
        "key":      key,
        "length":   len(sequence),
        "sequence": sequence,
        "source":   source,
        "created":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(cache_meta_path(key), "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# MSA generation (the slow path: hits the ColabFold server)
# =============================================================================
def _run_colabfold(cmd: list[str], cwd: Optional[Path] = None,
                   timeout: int = 1800) -> subprocess.CompletedProcess:
    """
    Run colabfold_batch in the colabfold conda env. Captures stdout/stderr.
    """
    full_cmd = ["conda", "run", "-n", COLABFOLD_ENV, "--no-capture-output"] + cmd
    try:
        return subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ColabFoldError(
            f"colabfold_batch timed out after {timeout}s. "
            f"Command: {' '.join(full_cmd)}"
        ) from e
    except FileNotFoundError as e:
        raise ColabFoldError(
            "`conda` not on PATH. The Streamlit process needs to be launched "
            "from a shell where `conda` is available (e.g. after `conda init`)."
        ) from e


def generate_msa(name: str, sequence: str, source: str = "user") -> str:
    """
    Generate an MSA for the given sequence by querying the ColabFold server,
    cache the result, and return the cache key.

    Idempotent: if the cache key already exists, returns immediately without
    re-querying.

    `source` is recorded in the metadata sidecar — useful for telling
    workshop-pre-warmed entries apart from user-uploaded ones.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = cache_key(name, sequence)
    if cache_exists(key):
        return key

    # Run colabfold_batch in --msa-only mode to get the a3m without doing a
    # structure prediction (much faster — we just want the alignment).
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "query.fasta"
        fasta.write_text(f">{name}\n{sequence}\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _run_colabfold(
            ["colabfold_batch", str(fasta), str(out_dir), "--msa-only"],
        )
        if result.returncode != 0:
            raise MSACacheError(
                f"colabfold_batch --msa-only failed (exit {result.returncode})\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Find the a3m. ColabFold names it after the FASTA header.
        a3ms = list(out_dir.glob("*.a3m"))
        # Skip per-database intermediate a3ms (those live under */env/*.a3m or
        # have e.g. "_uniref.a3m" suffixes); we want the merged top-level one.
        merged = [p for p in a3ms if "_env" not in p.parts and not p.name.endswith(("_uniref.a3m", "_smag.a3m"))]
        if not merged:
            raise MSACacheError(
                f"No merged .a3m found in {out_dir}. Files present: {[p.name for p in a3ms]}"
            )

        shutil.copy(merged[0], cache_path(key))
        _save_cache_meta(key, sequence, source)

    return key


# =============================================================================
# Per-design validation
# =============================================================================
def validate_design(
    cache_key_str: str,
    design_seq: str,
    design_name: str,
    out_dir: Path,
) -> dict:
    """
    Run AF2 (via ColabFold) on a single designed sequence using the cached MSA.

    Returns a dict with:
        plddt:    mean pLDDT of the rank-1 model (0-100)
        ptm:      pTM of the rank-1 model (0-1)
        pdb_path: path to the rank-1 PDB
        all_models: list of dicts per model with plddt/ptm
    """
    if not cache_exists(cache_key_str):
        raise MSACacheError(
            f"No cached MSA for key '{cache_key_str}'. "
            f"Call generate_msa() first, or run setup_msa_cache.sh."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a per-design a3m with the design as the new query sequence
    design_a3m = out_dir / f"{design_name}.a3m"
    _substitute_query(
        cached_a3m=cache_path(cache_key_str),
        design_seq=design_seq,
        design_name=design_name,
        out_path=design_a3m,
    )

    # Run colabfold_batch with the per-design a3m. Because the a3m already
    # contains the alignment, the MSA server is not contacted.
    result = _run_colabfold([
        "colabfold_batch",
        str(design_a3m),
        str(out_dir),
        "--num-models", str(NUM_MODELS),
    ])
    if result.returncode != 0:
        raise ColabFoldError(
            f"colabfold_batch failed (exit {result.returncode})\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    # Parse outputs: rank_001 PDB and the per-model JSON scores.
    rank1_pdbs = sorted(out_dir.glob(f"{design_name}_*rank_001*.pdb"))
    if not rank1_pdbs:
        raise ColabFoldError(
            f"No rank_001 PDB found in {out_dir}. "
            f"Files: {[p.name for p in out_dir.glob('*.pdb')]}"
        )
    rank1_pdb = rank1_pdbs[0]

    # Aggregate scores across all models for diagnostic purposes
    all_models: list[dict] = []
    for sj in sorted(out_dir.glob(f"{design_name}_scores_*.json")):
        with open(sj) as f:
            scores = json.load(f)
        plddt_arr = scores.get("plddt", [])
        mean_plddt = sum(plddt_arr) / len(plddt_arr) if plddt_arr else float("nan")
        all_models.append({
            "file":  sj.name,
            "plddt": mean_plddt,
            "ptm":   float(scores.get("ptm", -1)),
        })

    # Rank-1 model — match by filename
    rank1_score_files = sorted(out_dir.glob(f"{design_name}_scores_rank_001*.json"))
    if not rank1_score_files:
        raise ColabFoldError(f"No rank_001 scores JSON in {out_dir}")
    with open(rank1_score_files[0]) as f:
        rank1_scores = json.load(f)
    rank1_plddt_arr = rank1_scores.get("plddt", [])
    rank1_plddt = sum(rank1_plddt_arr) / len(rank1_plddt_arr) if rank1_plddt_arr else float("nan")

    return {
        "plddt":      rank1_plddt,
        "ptm":        float(rank1_scores.get("ptm", -1)),
        "pdb_path":   str(rank1_pdb),
        "all_models": all_models,
    }


# =============================================================================
# Cα superposition helpers (numpy Kabsch, no Biopython dependency)
# =============================================================================
def _ca_records(pdb_path: str, chain: str = "A") -> list[tuple[int, tuple[float, float, float], str]]:
    """
    Read Cα atoms from a PDB.

    Returns a list of (resnum, (x, y, z), atom_line) tuples — one per Cα.
    The full ATOM line is kept so we can rewrite the file with transformed
    coordinates while preserving everything else.

    Altloc handling: high-resolution crystal structures often include multiple
    alternative conformations per residue (altloc A/B/C/...). We keep only
    altloc=' ' (no altloc) or altloc='A' (the primary alternative). Without
    this dedup, structures like 6EQE (0.92 Å PETase) yield more Cα records
    than the chain has residues, breaking position-based alignment downstream.
    """
    out = []
    seen_resnums: set[int] = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            try:
                resnum = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            # Belt-and-suspenders: even after altloc filter, defend against
            # duplicate Cα records for the same residue number.
            if resnum in seen_resnums:
                continue
            seen_resnums.add(resnum)
            out.append((resnum, (x, y, z), line))
    return out


def _kabsch(a: "np.ndarray", b: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Kabsch superposition. Returns (rotation_matrix, translation_vector, rmsd).

    Solves: minimize ||R @ (a - centroid_a) + t - b||² over R, t.
    Convention chosen so rotated_a = (a - centroid_a) @ rot.T + centroid_b.
    """
    import numpy as np

    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    a_c = a - centroid_a
    b_c = b - centroid_b
    h = a_c.T @ b_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = a_c @ rot.T
    diff = a_rot - b_c
    rmsd = float(np.sqrt((diff * diff).sum() / len(a)))
    translation = centroid_b - centroid_a @ rot.T
    return rot, translation, rmsd


def calpha_rmsd(pdb_a: str, pdb_b: str, chain_a: str = "A",
                chain_b: str = "A") -> float:
    """
    Compute Cα RMSD between two PDB files after Kabsch superposition.

    Aligns on the COMMON LENGTH (min of the two Cα counts), starting from the
    N-terminus. This is appropriate for the workshop case where the predicted
    PDB may have a few extra/missing residues at the termini relative to the
    input — we get a meaningful number rather than a sentinel, and it's still
    a valid measure of structural similarity for the bulk of the chain.

    chain_a and chain_b can differ. The input PDB might use chain C (e.g. ZAR1
    in 6J6I) while the ColabFold-generated design always uses chain A.

    Returns RMSD in Å, or -1.0 if neither chain has any Cα atoms.
    """
    import numpy as np

    a = _ca_records(pdb_a, chain_a)
    b = _ca_records(pdb_b, chain_b)
    if not a or not b:
        return -1.0

    n = min(len(a), len(b))
    a_xyz = np.array([rec[1] for rec in a[:n]])
    b_xyz = np.array([rec[1] for rec in b[:n]])
    _, _, rmsd = _kabsch(a_xyz, b_xyz)
    return rmsd


def aligned_design_pdb(input_pdb: str, design_pdb: str,
                       input_chain: str = "A", design_chain: str = "A") -> str:
    """
    Return the design PDB's contents as a string, with all atom coordinates
    rotated/translated to superimpose onto the input PDB.

    Used by the 3D viewer so the WT and the design are drawn in the same
    coordinate frame — without this, py3Dmol shows them side-by-side in their
    original (unrelated) frames.

    The Kabsch fit is computed on Cα atoms of the common-length prefix; the
    same rotation+translation is then applied to every atom (CA, sidechain,
    HETATM, etc.) so the whole design moves coherently.

    input_chain and design_chain can differ — the WT may live in chain C of
    a multi-chain PDB while ColabFold always emits the design as chain A.
    """
    import numpy as np

    wt_ca = _ca_records(input_pdb, input_chain)
    design_ca = _ca_records(design_pdb, design_chain)
    if not wt_ca or not design_ca:
        # Can't align — return the design unmodified
        with open(design_pdb) as f:
            return f.read()

    n = min(len(wt_ca), len(design_ca))
    a_xyz = np.array([rec[1] for rec in design_ca[:n]])  # design = source
    b_xyz = np.array([rec[1] for rec in wt_ca[:n]])      # wt     = target
    rot, translation, _ = _kabsch(a_xyz, b_xyz)

    # Apply rot+translation to every ATOM/HETATM line in the design PDB
    out_lines: list[str] = []
    with open(design_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                out_lines.append(line)
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                out_lines.append(line)
                continue
            v = np.array([x, y, z])
            new = v @ rot.T + translation
            new_line = (
                line[:30]
                + f"{new[0]:8.3f}{new[1]:8.3f}{new[2]:8.3f}"
                + line[54:]
            )
            out_lines.append(new_line)

    return "".join(out_lines)


def filter_pdb_to_chain(pdb_path: str, chain: str) -> str:
    """
    Return PDB contents as a string, keeping only ATOM/HETATM records on the
    given chain (and other non-coordinate lines like HEADER/TER/END).

    Used by the 3D viewer to show only the WT chain we care about, instead of
    all chains in a multi-chain PDB (e.g. 6J6I has the ZAR1 chains plus the
    kinase chains; we only want to display chain C alongside the design).
    """
    out = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if line[21] == chain:
                    out.append(line)
            else:
                out.append(line)
    return "".join(out)
