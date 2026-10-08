# Reproducing the paper's figures

!!! note "Separate paper workspace"
    The `paper/` tree and stored `.npz` chains described here are excluded from
    this repository. These instructions document the original research workflow;
    they require that separate workspace and its inputs.

Every figure and generated table in `paper/main.tex` is produced by code in this
repository, and `paper/make_figures.py` is the index of which producer makes which:

```sh
uv run python paper/make_figures.py --list     # what makes what, what it reads, how long
uv run python paper/make_figures.py --check    # verify the figures against the manifest
uv run python paper/make_figures.py --run draw # redraw all 10 stored-chain figures, ~1 min
```

Producers are grouped into tiers by cost. `draw` redraws from the stored `.npz`
chains and computes nothing; `validation` runs the validation notebooks (~14 min);
`derived` builds the convergence table; `sample` is the MCMC that produced the stored
chains in the first place — hours, mostly on a GPU.

Those `.npz` chains are **not** in version control (`notebooks/**/*.npz` is ignored,
as is `paper/`), so the manifest below is the only record tying a published figure to
the sampler run behind it. Keep them together.

Three properties make this reproducible rather than merely scripted:

1. **Figures are byte-deterministic.** Every producer writes through one `save()`
   (`paper/validation/_style.py`), which pins the PDF timestamp to
   `SOURCE_DATE_EPOCH`. Re-running a producer on unchanged input gives a
   byte-identical file, so a changed figure in `git status` means a changed *figure*.

2. **Provenance is recorded, not remembered.** `paper/figures/MANIFEST.json` holds,
   per figure: its SHA-256, the producer that wrote it, the seeds it used, the
   SHA-256 of every data file it read, the git commit, and the versions of `jax`,
   `jaxglitches`, `lisaglitch`, `lisainstrument`, `pytdi` and the rest, plus the JAX
   backend — which matters, because Appendix B.4 of the paper quotes residuals at
   `1e-16`, where CPU and GPU differ. The commit it records is this repository's;
   the paper tree and the chains are outside it, which is precisely why their
   hashes are worth writing down.

3. **Staleness is detectable.** `--check` re-hashes outputs *and inputs*, so a figure
   drawn before its chain was re-sampled is reported rather than silently shipped.
   `.npz` inputs are hashed member by member, so a rewritten-but-identical chain does
   not raise a false alarm.

Verified by running every producer outside the `sample` tier twice from scratch:
every output comes back byte-identical except the two benchmark outputs,
`fig_benchmark.pdf` and `tab_benchmark.tex`, whose content is wall-clock timings and
which are flagged `reproducible: false` in the manifest. (`fig_wdm_tiling.pdf` comes
from a 40-minute sampler notebook and is registered in the manifest rather than
re-run.) Five producers had to give up the GPU to get there
(`05_heterodyne.ipynb`, `unequal_vs_equal_arm.ipynb`, `compare_other_codes.ipynb`,
`exact_vs_hybrid.ipynb`, `run_convergence.py`): GPU
reductions are not bit-reproducible, and each was rewriting its figure on every run
at a relative `1e-6` that changed nothing it reports. They set `JAX_PLATFORMS=cpu`,
overridable, and each says why at the top of the file. The MCMC in the `sample` tier
still runs on the GPU and is reproducible in the ordinary seeded sense.

One figure is deliberately environment-dependent and says so: panel (d) of
`fig_gradients` visualises how the installed JAX version differentiates a function
that is not differentiable at the point in question. Under jax 0.10 the Hessian comes
out identically zero where older versions returned a finite but wrong matrix; the
notebook detects which it got and draws that one, instead of crashing on
`np.linalg.inv` as it had started to do.
