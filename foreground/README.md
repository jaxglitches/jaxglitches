# Galactic-foreground rerun (sandbox)

A copy of the analysis scripts of `../notebooks` and `../paper/validation/_style.py`,
run with a noise model that adds the stationary (yearly-averaged) Galactic confusion
foreground to the instrumental A and E spectra. Nothing here writes outside this folder:
`pyproject.toml` is a marker that makes the notebooks resolve their repository root to
`foreground/`, and `_style.py` writes figures and its manifest to `foreground/paper/figures`.

The only change to the copied code is `notebooks/noise.py` (`psd_tdi1` now adds `_foreground_ae`); `notebooks/glitch_GB/run_wdm_segments.py` also guards one singular Hessian, and `notebooks/glitch_GB/run_unmodelled.py` (`run_chain`) caps and reflects the start ball into the prior box: the glitch-free maxima sit on the log fdot floor, and walkers started below it never moved. The same latent issue exists in the original `notebooks/glitch_GB/run_unmodelled.py` but never triggers without the foreground.

Later additions (2026-10-31):
- `notebooks/noise.py` also draws the check figure of the noise model: `python noise.py [out.pdf]`.
- `glitch_and_gb_wdm.ipynb` and `run_wdm_tdi2.py` read their chain length from `WDM_NBURN` / `WDM_NSAMP` (defaults 2 000 / 10 000).
- `run_knee_scan.py --figure` labels the fiducial line with the foreground SNR, 27.7.
- `paper/validation/05_heterodyne.ipynb` uses the foreground glitch widths, sigma(t0) = 19.5 s.
- `notebooks/glitch_GB/run_start_check.py` is new: it compares the production chain start (one Newton step) with the backtracked maximum, against a second seed.
- `notebooks/glitch_only/exact_vs_hybrid.ipynb` and `unequal_vs_equal_arm.ipynb` are copies of the originals, run here with the foreground.
