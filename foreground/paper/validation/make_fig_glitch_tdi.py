"""fig_glitch_tdi.pdf -- the fiducial glitch, in time and frequency, through both TDI generations.

Section "Glitches as instrumental artefacts" describes the glitch in words and equations:
a smoothed velocity step on one link, turned by TDI into a pulse, with a spectral knee
at 1/(2 pi tau) that separates a flat low-frequency response from a steep fall-off.
This draws it, for the injection of Sec. "The simulated data set" (t0 = 400 s,
Deltav = 1e-11 m/s, tau = 300 s, test mass 12, equal arms).

Columns go single link -> TDI-1 -> TDI-2, rows time domain -> frequency domain, so each
time-domain panel sits above its own spectrum:

  (a) the single-link fractional-frequency perturbation, which asymptotes to Deltav/c;
  (b), (c) the A, E, T channels in TDI-1 and TDI-2;
  (d) the single-link spectrum, with its two asymptotes and the knee between them;
  (e), (f) the A, E, T spectra against the noise of each channel, in the normalisation
      whose squared ratio integrates over ln f to rho^2, so the gap between a signal
      and its noise is the SNR density.

(e) and (f) share a vertical axis. The second generation lowers signal and noise by the
same factor |1 - D^4| = 2|sin(4 pi f L)| at every frequency, so both sets of curves move
down together and the gaps do not change: that is the frequency-domain identity the null
test of Sec. "TDI-1 against TDI-2: a null test" relies on, seen on one plot.

The signal comes from `jaxglitches` and the noise from `notebooks/noise.py`; the only
data file read is `dataset.npz`, for the injected parameters, the arm length and the
binary's frequency, so that the figure cannot drift from the injection it illustrates.
The printed checks tie the plot to the numbers quoted in the text.
"""
import os
import sys
from pathlib import Path

# Nothing here is heavy, and the CPU keeps the printed numbers and the PDF's bytes the
# same run to run.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _style import FULL_IN, C, save  # noqa: E402

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import jaxglitches as jg  # noqa: E402
import noise as ns  # noqa: E402

DATASET = HERE.parents[1] / "notebooks" / "glitch_GB" / "dataset.npz"
C_SI = 299792458.0
F_MAX = 3e-3                          # top of the analysis band, Eq. (dt)
CHANNELS = (("A", C["blue"]), ("E", C["red"]), ("T", C["green"]))


def main():
    ds = np.load(DATASET)
    t0, dv, tau = (float(v) for v in ds["glitch_true"])
    arm = float(ds["T_ARM"])
    f0_gb = float(ds["gb_true"][0])
    f_knee = 1.0 / (2.0 * np.pi * tau)
    p = jnp.array([t0, dv, tau])
    assert abs(arm - jg.waveform.T_ARM_s) < 1e-12, "templates assume the design arm"

    # ---- time domain -------------------------------------------------------
    t = np.arange(-300.0, 2700.0, 0.25)                 # relative to the onset
    tt = jnp.asarray(t0 + t)
    # Eq. (dnu_time) of the paper. `raw_glitch_t` carries the opposite overall sign,
    # compensated in the package by the sign of its transfer functions, so that the
    # TDI outputs agree with Eqs. (TFX1)-(TFY1) as written.
    dnu = -np.asarray(jg.raw_glitch_t(p, tt))
    td = {g: np.asarray(jg.clean_signal_t(p, tt, tdi=g)) for g in (1, 2)}

    # ---- frequency domain ----------------------------------------------------
    f = np.logspace(-5.0, np.log10(2e-2), 4000)
    fj = jnp.asarray(f)
    dnu_f = np.abs(np.asarray(jg.raw_glitch_f(p, fj)))
    fd = {g: np.abs(np.asarray(jg.clean_signal_f(p, fj, tdi=g))) for g in (1, 2)}
    psd = {1: np.stack(ns.psd_tdi1(fj), axis=-1), 2: np.stack(ns.psd_tdi2(fj), axis=-1)}
    char = {g: 2.0 * np.sqrt(f)[:, None] * fd[g] for g in (1, 2)}
    asd = {g: np.sqrt(psd[g]) for g in (1, 2)}

    # ---- checks against the numbers the text quotes ---------------------------
    band = f <= F_MAX
    rho = {g: np.sqrt(np.trapezoid((char[g] / asd[g])[band] ** 2, np.log(f[band]), axis=0))
           for g in (1, 2)}
    print(f"glitch: t0 = {t0:.0f} s, Deltav = {dv:.1e} m/s, tau = {tau:.0f} s, "
          f"knee {f_knee * 1e3:.3f} mHz")
    for g in (1, 2):
        print(f"  TDI-{g}: rho (A, E, T) = " + ", ".join(f"{v:.2f}" for v in rho[g])
              + f"; total {np.sqrt(np.sum(rho[g] ** 2)):.2f} on [{f[0]:.0e} Hz, 3 mHz]")
    ratio = (char[2] / asd[2]) / (char[1] / asd[1])
    print(f"  SNR density TDI-2 / TDI-1: max |ratio - 1| = {np.abs(ratio[band] - 1).max():.1e}")
    for g in (1, 2):
        peak = np.abs(td[g]).max(axis=0)
        i = np.argmax(np.abs(td[g][:, 1]))
        print(f"  TDI-{g} time domain: peak |A|, |E|, |T| = "
              + ", ".join(f"{v:.2e}" for v in peak) + f"; E peaks at t - t0 = {t[i]:.0f} s")

    # ---- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(FULL_IN, 3.9))

    ax = axes[0, 0]
    ax.plot(t, dnu / 1e-20, color="k", lw=1.2)
    ax.axhline(dv / C_SI / 1e-20, color=C["grey"], lw=0.8, ls=":")
    ax.text(t[-1] - 60, dv / C_SI / 1e-20 * 0.94, r"$\Delta v/c$", ha="right", va="top",
            fontsize=7, color=C["grey"])
    ax.set_ylabel(r"fractional frequency  [$10^{-20}$]")
    ax.set_title("(a) single link", loc="left")

    for col, g, exp in ((1, 1, 21), (2, 2, 22)):
        ax = axes[0, col]
        for i, (name, colour) in enumerate(CHANNELS):
            ax.plot(t, td[g][:, i] / 10.0 ** -exp, color=colour, lw=1.1, label=name)
        ax.axhline(0.0, color=C["grey"], lw=0.5)
        ax.set_ylabel(rf"fractional frequency  [$10^{{-{exp}}}$]")
        ax.set_title(f"({'bc'[col - 1]}) TDI-{g}", loc="left")
    axes[0, 1].legend(loc="lower right", ncol=3, handlelength=1.2, columnspacing=0.9)
    for ax in axes[0]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel(r"$t-t_0$  [s]")

    ax = axes[1, 0]
    ax.loglog(f, dnu_f, color="k", lw=1.2)
    # The two asymptotes of Sec. "The spectral knee", each drawn only on its own side of
    # the knee (to a factor 4 past it), so that neither can be mistaken for the other.
    low = dv / (C_SI * 2.0 * np.pi * f)
    high = dv / (C_SI * (2.0 * np.pi * f) ** 3 * tau ** 2)
    ax.loglog(f[f < 4 * f_knee], low[f < 4 * f_knee], color=C["grey"], lw=0.9, ls=":")
    ax.loglog(f[f > f_knee / 4], high[f > f_knee / 4], color=C["grey"], lw=0.9, ls=":")
    ax.set_ylim(dnu_f.min() / 4, dnu_f.max() * 4)
    ax.text(2e-5, low[np.searchsorted(f, 2e-5)] / 6, r"$\propto f^{-1}$", fontsize=7,
            color=C["grey"], ha="left", va="top")
    ax.text(3e-3, high[np.searchsorted(f, 3e-3)] / 6, r"$\propto f^{-3}$", fontsize=7,
            color=C["grey"], ha="right", va="top")
    ax.text(f_knee * 0.85, dnu_f.min() * 1.5, r"$f_{\rm knee}$", fontsize=7,
            color=C["grey"], ha="right", va="bottom")
    ax.set_ylabel(r"$|\Delta\tilde\nu/\nu_{0}|$  [Hz$^{-1}$]")
    ax.set_title("(d) single link", loc="left")

    lo = min(asd[2][:, 2].min(), char[2][:, 2].min()) / 3
    hi = max(asd[1].max(), char[1].max()) * 3
    for col, g in ((1, 1), (2, 2)):
        ax = axes[1, col]
        for i, (name, colour) in enumerate(CHANNELS):
            ax.loglog(f, char[g][:, i], color=colour, lw=1.1)
        ax.loglog(f, asd[g][:, 0], color=C["grey"], lw=1.0, label="noise, A and E")
        ax.loglog(f, asd[g][:, 2], color=C["grey"], lw=1.0, ls="--", label="noise, T")
        ax.set_ylim(lo, hi)
        ax.set_ylabel(r"$2\sqrt{f}\,|\tilde h|$  and  $\sqrt{S_{c}}$  [Hz$^{-1/2}$]")
        ax.set_title(f"({'ef'[col - 1]}) TDI-{g}", loc="left")
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([f0_gb], [0.965], marker="v", ms=4, color=C["grey"], transform=trans,
                clip_on=False)
        ax.text(f0_gb * 0.8, 0.965, r"$f_{0}$", transform=trans, fontsize=6.5,
                color=C["grey"], ha="right", va="center")
    axes[1, 1].legend(loc="lower left", fontsize=6.5, handlelength=1.8)
    for ax in axes[1]:
        ax.axvline(f_knee, color=C["grey"], lw=0.8, ls=":")
        ax.axvspan(F_MAX, f[-1], color=C["grey"], alpha=0.12, lw=0)
        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequency [Hz]")

    save(fig, "fig_glitch_tdi", inputs=[DATASET])


if __name__ == "__main__":
    main()
