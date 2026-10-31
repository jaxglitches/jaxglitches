"""Shared setup for every figure in the paper.

Resolves the repository root (so that `notebooks/noise.py` is importable), fixes a
common matplotlib style, and defines the figure output directory.

Every figure of `paper/main.tex` is written through `save()` below -- the validation
notebooks, the two analysis notebooks and the standalone `make_fig_*.py` scripts all
import it -- which makes this module the one place where reproducibility can be
enforced rather than asked for. It does three things towards that:

* **Deterministic bytes.** Matplotlib stamps the wall clock into every PDF it writes,
  so re-exporting an *identical* figure still changed the file. `save()` pins
  `CreationDate` to `SOURCE_DATE_EPOCH` (default: the Unix epoch), so an unchanged
  figure re-exports byte for byte and `git status` means something again.

* **A manifest.** `save()` records into `paper/figures/MANIFEST.json` the SHA-256 of
  the figure, the producer that wrote it, the seeds it used, the SHA-256 of every
  data file it read, the git commit, and the versions of the packages involved. A
  figure drawn from a two-hour GPU chain is only reproducible if the chain is the
  same file, and hashing the inputs is what pins that down.

* **A declared environment.** `provenance()` collects what the numbers actually
  depend on, the JAX backend and the x64 flag included: Sec. "Validation" quotes
  residuals at $10^{-16}$, and CPU and GPU do differ in the last bits.

`../make_figures.py` drives all of it, and can verify the manifest without
recomputing anything.

Figure widths match the two-column layout of main.tex:
    COL   = 8.4 cm  (3.31 in)  -- single column,  \\includegraphics in figure
    FULL  = 17.4 cm (6.85 in)  -- both columns,   \\includegraphics in figure*
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# repo root = parent of paper/
REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(exist_ok=True)

NOTEBOOK_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))   # makes `import noise` work

COL_IN = 3.31
FULL_IN = 6.85

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})

# colour-blind safe qualitative palette (Okabe-Ito)
C = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "grey":   "#666666",
}

EPS = 2.220446049250313e-16   # float64 machine epsilon


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

MANIFEST = FIG_DIR / "MANIFEST.json"

# Reproducible-builds convention: a fixed timestamp, overridable from the
# environment, in place of the wall clock matplotlib would otherwise embed.
SOURCE_DATE_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
PDF_METADATA = {
    "CreationDate": datetime.datetime.fromtimestamp(SOURCE_DATE_EPOCH,
                                                    datetime.timezone.utc),
}

# Packages whose version can change a number in the paper. `jaxglitches` and `jax`
# obviously; `lisaglitch`/`lisainstrument`/`pytdi` because Checks 4 and 5 compare
# against them; `jaxgb`/`jexplore` because they supply the binary waveform and the
# sampler.
_PACKAGES = ("jaxglitches", "jax", "jaxlib", "numpy", "scipy", "matplotlib",
             "jaxgb", "jexplore", "lisaglitch", "lisainstrument", "pytdi",
             "lisaorbits")

# Set by `set_source()`, or by the JG_FIG_SOURCE environment variable that
# `paper/make_figures.py` exports for each step it runs.
_SOURCE = os.environ.get("JG_FIG_SOURCE")

_PROVENANCE = None      # cached: shelling out to git once per process is enough


def set_source(name):
    """Declare which file is producing the figures, for the manifest.

    Scripts are detected from the call stack and need not call this. Notebooks do:
    a notebook cell has no `__file__`, so without it the producer would be recorded
    as unknown. Call it once, in the first cell::

        set_source("paper/validation/01_waveform_accuracy.ipynb")
    """
    global _SOURCE
    _SOURCE = name
    return name


def _git(*args):
    try:
        out = subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance():
    """What the numbers in a figure depend on: commit, packages, JAX backend."""
    global _PROVENANCE
    if _PROVENANCE is not None:
        return _PROVENANCE

    pkgs = {}
    for name in _PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pass

    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "packages": pkgs,
    }

    # Only report JAX if the producer actually imported it: pure redraw scripts do
    # not, and importing it here to ask would be both slow and misleading.
    jax = sys.modules.get("jax")
    if jax is not None:
        try:
            prov["jax_backend"] = jax.default_backend()
            prov["jax_devices"] = [str(d) for d in jax.devices()]
            prov["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
        except Exception:                                    # noqa: BLE001
            pass

    _PROVENANCE = prov
    return prov


def sha256(path):
    """SHA-256 of a file's *content*, or None if it is not there.

    `.npz` files are hashed member by member rather than as a container. A zip
    stamps the wall clock into every entry, so two `np.savez_compressed` calls
    writing identical arrays produce different bytes -- which would make every
    figure drawn from a chain look stale the moment the chain was rewritten, for no
    reason. Hashing the `.npy` payloads instead compares what the file says.
    """
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    if path.suffix == ".npz":
        try:
            with zipfile.ZipFile(path) as zf:
                for name in sorted(zf.namelist()):
                    h.update(name.encode())
                    h.update(zf.read(name))
            return h.hexdigest()
        except (zipfile.BadZipFile, OSError):
            h = hashlib.sha256()        # fall through to raw bytes
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    """Repository-relative path where possible, so the manifest is portable."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _caller_source():
    """The repository file that called `save()`, from the stack.

    Frames outside the repository (nbclient, ipykernel) are skipped rather than
    reported, so a notebook that forgot `set_source()` records `null` instead of
    something from site-packages.
    """
    if _SOURCE is not None:
        return _SOURCE
    here = Path(__file__).resolve()
    venv = REPO_ROOT / ".venv"
    frame = sys._getframe(1)
    while frame is not None:
        f = frame.f_globals.get("__file__")
        if f:
            p = Path(f).resolve()
            if p != here and REPO_ROOT in p.parents and venv not in p.parents:
                return _rel(p)
        frame = frame.f_back
    return None


def _record(name, path, source, seed, inputs, note, reproducible):
    """Add one entry to paper/figures/MANIFEST.json."""
    if os.environ.get("JG_NO_MANIFEST"):
        return
    entry = {
        "output": _rel(path),
        "sha256": sha256(path),
        "bytes": Path(path).stat().st_size,
        "produced_by": source,
        "reproducible": reproducible,
        "provenance": provenance(),
    }
    if seed is not None:
        entry["seed"] = seed
    if inputs:
        entry["inputs"] = {_rel(p): sha256(p) for p in inputs}
    if note:
        entry["note"] = note

    data = {}
    if MANIFEST.is_file():
        try:
            data = json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            data = {}
    data[name] = entry

    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(MANIFEST)


def save(fig, name, *, seed=None, inputs=None, source=None, note=None,
         reproducible=True):
    """Save `fig` as a PDF in paper/figures, deterministically, and record it.

    Parameters
    ----------
    fig, name
        The figure and its stem; `name.pdf` is written into `paper/figures`.
    seed
        The RNG seed(s) the producer used, for the manifest. Pass whatever the
        producer keys its randomness on -- an int, or a dict of them.
    inputs
        Data files the figure was drawn from (`dataset.npz`, a chain, ...). Their
        SHA-256 goes in the manifest: a figure made from a two-hour sampler run is
        only reproducible together with the run.
    source
        Producer path, if the stack cannot be trusted. Prefer `set_source()`.
    note
        Free text kept with the entry.
    reproducible
        Set False for output that cannot be byte-reproducible by construction --
        the benchmark table, whose content is wall-clock timings.
    """
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    _record(name, out, source or _caller_source(), seed, inputs, note, reproducible)
    print(f"saved {out.relative_to(REPO_ROOT)}")
    return out


def save_text(content, name, *, seed=None, inputs=None, source=None, note=None,
              reproducible=True):
    """The same, for the generated LaTeX tables (`tab_*.tex`)."""
    out = FIG_DIR / name
    out.write_text(content if content.endswith("\n") else content + "\n")
    _record(Path(name).stem, out, source or _caller_source(), seed, inputs, note,
            reproducible)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return out


def machine_precision_band(ax, label="float64 eps"):
    """Shade the region below machine epsilon on a log-scale axis."""
    ax.axhline(EPS, color=C["grey"], ls=":", lw=0.8)
    ax.text(0.02, EPS * 1.6, label, transform=ax.get_yaxis_transform(),
            fontsize=6, color=C["grey"], va="bottom")
