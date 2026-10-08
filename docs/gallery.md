# Research notebooks

These studies are rendered from the repository's saved notebooks, including
code, tables, and plots. They are **not executed by CI or the docs build**.
Their saved findings are retained below; displaying them does not constitute
fresh scientific validation. The small [quick-start example](example.md) remains
an executed CI check.

| Study | View |
| --- | --- |
| Equal-arm approximation and frozen unequal delays | [Notebook](notebook-gallery/unequal_vs_equal_arm.md) |
| Comparisons with published results | [Notebook](notebook-gallery/compare_other_codes.md) |
| Exact versus hybrid likelihood | [Notebook](notebook-gallery/exact_vs_hybrid.md) |
| Unresolved short glitches | [Script and usage](notebook-gallery/short_glitch.md) |
| Raw response, TDI, and external simulation checks | [Notebook](notebook-gallery/raw_and_tdi_tests.md) |

Each page provides a download and a link to its original source. The raw/TDI
notebook contains six embedded plots. The other three notebooks contain saved
text and numeric tables, but their paper figures were written to external PDFs
and are not embedded in the saved notebooks.
`paper/validation/04_end_to_end.ipynb` and the paper appendices are not included
in this checkout. The available end-to-end validation is in section 5 of the raw/TDI notebook.
Stored chains and other ignored `.npz` files may be needed to rerun these studies;
installing the package alone does not supply those inputs.

## What each study addresses

[unequal_vs_equal_arm.ipynb](notebook-gallery/unequal_vs_equal_arm.md) goes one step further and asks
what the equal-arm approximation *costs*: it injects with the six frozen delays of a
Keplerian constellation, fits with the equal-arm template, and measures the resulting
bias over a year of epochs. Appendix D of the paper reports the answer. Two findings
are worth knowing before reusing that code — the T channel stops being null once the
arms are unequal, so the equal-arm `S_T` of `notebooks/noise.py` must not be used with
an unequal-arm signal; and the equal-arm `T` an analyst picks matters, the epoch's mean
delay being a factor four better than the design value.

[compare_other_codes.ipynb](notebook-gallery/compare_other_codes.md) checks the package against
published results rather than against itself: the optimal SNRs of the seven glitches
tabulated by Muratore et al. (2025), the LISA Pathfinder population projected to LISA
by Baghi et al. (2022), the two-year population percentiles of Boumerdassi et al.
(2026), and the parameter degeneracy Sauter et al. (2025) report for sub-sample
glitches. Appendix C of the paper reports what came out. Two of the four agree
sharply; the other two turn on what a published amplitude column means, and the
notebook says so rather than picking the flattering reading.

[exact_vs_hybrid.ipynb](notebook-gallery/exact_vs_hybrid.md) asks whether the binned likelihood the
paper samples from gives the same posterior as the exact one over all 93,697 Fourier
bins. Same stored data, same priors, same sampler, same Newton start — the only thing
that changes is the grid. The awkward part is knowing what agreement to demand, since
two chains of any finite length disagree, so the notebook measures that too: in each TDI
generation a second hybrid chain differs from the first only in its random seed, and that
control is the yardstick. In root-mean-square over the seven parameters the binning comes
out smaller than it, on both the posterior widths and the medians, in both generations.
Appendix E of the paper reports the numbers, and Sec. 3.2 summarises them.

[short_glitch.py](notebook-gallery/short_glitch.md) asks what is left of a glitch too short for the
analysis band to resolve, i.e. with its knee 1/(2 pi tau) above the 3 mHz top of the
band (tau below 53 s). Below the knee the glitch is a velocity step at the pulse
centroid t0 + 2 tau, so the kick and the centroid stay measured -- a chain at tau = 1 s
gives Deltav to 2.5% (about 1/rho) and the centroid to 2 s -- while tau becomes an upper
limit and t0 is lost in its degeneracy with tau. The Fisher matrix's constant 3.8% on
Deltav at small tau is a linearisation artefact, which the chain shows.

The waveforms are also checked end to end against the external LISA simulation
chain — a glitch injected with `lisaglitch`, propagated by `lisainstrument` and
combined into Michelson TDI by `pytdi` — in
`paper/validation/04_end_to_end.ipynb` and §5 of
[raw_and_tdi_tests.ipynb](notebook-gallery/raw_and_tdi_tests.md). With every light travel time set
to an integer number of samples nothing in the chain interpolates and the
agreement is exact to 2e-15 across all six channels and both TDI generations.
`lisainstrument` and `pytdi` are not package dependencies:

```sh
uv sync --extra notebook --extra simulation
```

