"""
LISA noise PSDs and frequency-domain noise sampling — outside jaxglitches package.

Units and conventions
---------------------
The `jaxglitches` templates (`clean_signal_f`, `raw_glitch_f`, ...) return the
*continuous Fourier transform* of a TDI combination of the fractional frequency
perturbation,

    h_fd[k, c] = H_c(f_k) = \\int h_c(t) e^{-2 pi i f_k t} dt ,

so `h_fd` carries units of seconds (dimensionless signal x time).  For the noise
to be addable to the signal it has to live in exactly the same units, which
requires two separate things to be right:

1. **TDI fractional frequency, not strain.**  The PSDs below are the standard
   equal-arm AET spectra *in fractional frequency*, i.e. the units the TDI
   variables themselves are in.  They are NOT strain-referenced sensitivity
   curves: a strain curve is this divided by the instrument response
   ~16 x^2 sin^2 x (x = 2 pi f L / c), which grows like f^4 across the band and
   would silently reweight the analysis.  See `psd_tdi1` for the expressions.

2. **Bin variance, not one-sided PSD.**  A one-sided PSD S_1(f) has units of
   1/Hz; the variance of a continuous-FT bin over an observation of length
   T_obs is

       sigma_k^2 = (T_obs / 2) * S_1(f_k)                         [s^2]

   which is what `sample_noise_fd` needs and what the likelihood in
   `jaxglitches.likelihood` divides by.  With this convention the matched-filter
   SNR

       rho^2 = 4 \\int |H|^2 / S_1 df  ->  2 sum_k |H_k|^2 / sigma_k^2

   reduces exactly to `jaxglitches.snr`, which computes 2 sum conj(a) b / psd.

Accordingly `psd_tdi1`/`psd_tdi2` return the physical one-sided PSD [1/Hz],
while `psd_tdi1_array`/`psd_tdi2_array` return the bin variance [s^2] and
therefore *require* `t_obs`.  Passing the wrong one is the failure mode this
module exists to prevent, so `t_obs` is keyword-only and has no default.

TDI-1 vs TDI-2
--------------
The TDI-1 and TDI-2 combinations carry the same physical signal with different
transfer functions (see waveform.py):

    h_X^{TDI1}(f) ~ TF1(f) * dnu(f),   TF1 = exp(-4iTf.2pi) - 1
    h_X^{TDI2}(f) ~ TF1(f)^2 * dnu(f)

so the noise is amplified by |TF1|^2:

    S_TDI2(f) = |TF1(f)|^2 . S_TDI1(f) = 4 sin^2(4 pi f T) . S_TDI1(f)

With this correction the optimal SNR is identical in both generations, as it
must be physically.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Default LISA noise parameters (LISA red-book / SciRD values)
_ARM_m  = 2.5e9        # arm length (m)
_C_SI   = 299792458.0
_A_DEF  = 3.0e-15      # test-mass acceleration noise (m/s^2/sqrt(Hz))
_P_DEF  = 15.0e-12     # OMS optical path noise (m/sqrt(Hz))


# ---------------------------------------------------------------------------
# Single-link noise contributions, in fractional frequency
# ---------------------------------------------------------------------------

def _single_link_psd(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    Test-mass and optical-metrology noise per link, in fractional frequency.

        S_pm(f) = A^2 [1 + (0.4 mHz/f)^2][1 + (f/8 mHz)^4] / (2 pi f c)^2
        S_op(f) = P^2 [1 + (2 mHz/f)^4]  (2 pi f / c)^2

    Both have units of 1/Hz.  The acceleration noise is divided by (2 pi f)^2 c^2
    because an acceleration must be integrated once to a velocity and then
    referred to c to become a fractional frequency; the optical noise is
    multiplied by (2 pi f / c)^2 because a displacement must be differentiated
    once.  This is the origin of their opposite frequency slopes.
    """
    M_acc = (1.0 + (4e-4 / f) ** 2) * (1.0 + (f / 8e-3) ** 4)
    M_oms = 1.0 + (2e-3 / f) ** 4
    S_pm = A ** 2 * M_acc / (2.0 * jnp.pi * f * _C_SI) ** 2
    S_op = P ** 2 * M_oms * (2.0 * jnp.pi * f / _C_SI) ** 2
    return S_pm, S_op


# ---------------------------------------------------------------------------
# TDI-1 one-sided PSD (A, E, T channels), fractional frequency
# ---------------------------------------------------------------------------

def _psd_tdi1_instr(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    One-sided TDI-1 *instrumental* noise PSD for the A, E, T channels, in fractional frequency.

    With x = 2 pi f L / c the standard equal-arm expressions are

        S_A = S_E = 8 sin^2 x [4 (1 + cos x + cos^2 x) S_pm + (2 + cos x) S_op]
        S_T       = 16 sin^2 x [2 (1 - cos x)^2 S_pm + (1 - cos x) S_op]

    Units: 1/Hz.  This is the *physical* PSD; to obtain the per-bin variance the
    likelihood consumes, multiply by T_obs/2 (see `bin_variance`, or use
    `psd_tdi1_array`, which does it for you).

    Parameters
    ----------
    f : array of positive frequencies (Hz).
    A : test-mass acceleration noise (m/s^2/sqrt(Hz)).
    P : OMS optical path noise (m/sqrt(Hz)).
    L : arm length (m).

    Returns
    -------
    S_A, S_E, S_T : arrays of shape (len(f),), units 1/Hz.
    """
    x = 2.0 * jnp.pi * f * (L / _C_SI)
    S_pm, S_op = _single_link_psd(f, A=A, P=P, L=L)

    cx = jnp.cos(x)
    s2 = jnp.sin(x) ** 2
    S_A = 8.0 * s2 * (4.0 * (1.0 + cx + cx ** 2) * S_pm + (2.0 + cx) * S_op)
    S_E = S_A
    S_T = 16.0 * s2 * (2.0 * (1.0 - cx) ** 2 * S_pm + (1.0 - cx) * S_op)
    return S_A, S_E, S_T


# ---------------------------------------------------------------------------
# Galactic confusion foreground (stationary, yearly averaged)
# ---------------------------------------------------------------------------
# Robson, Cornish & Liu (2019), Eq. (14) and Table 1, one-year fit:
#   S_c(f) = A f^{-7/3} exp(-f^alpha + beta f sin(kappa f)) [1 + tanh(gamma (f_k - f))]
# S_c is a sky-averaged *strain* PSD in the convention of their Eq. (13), the
# instrumental sensitivity S_n(f) of the same paper. Both are referred to strain by the
# same sky-averaged response, so the foreground in a TDI channel is the instrumental
# channel PSD times S_c/S_n: S_conf^A = S_A^instr * S_c / S_n. The ratio is independent
# of how many channels S_n is normalised to, because S_c and S_n share that normalisation.
# T is insensitive to gravitational waves at these frequencies and gets no foreground.
_FG = dict(A=9e-45, alpha=0.171, beta=292.0, kappa=1020.0, gamma=1680.0, fk=0.00215)


def galactic_confusion_strain(f, p=_FG):
    """Robson et al. (2019) one-year confusion noise S_c(f), strain, 1/Hz."""
    return (p["A"] * f ** (-7.0 / 3.0)
            * jnp.exp(-f ** p["alpha"] + p["beta"] * f * jnp.sin(p["kappa"] * f))
            * (1.0 + jnp.tanh(p["gamma"] * (p["fk"] - f))))


def robson_instrumental_strain(f, L: float = _ARM_m):
    """Robson et al. (2019) Eq. (13): instrumental sky-averaged sensitivity, 1/Hz."""
    fstar = _C_SI / (2.0 * jnp.pi * L)
    P_oms = (1.5e-11) ** 2 * (1.0 + (2e-3 / f) ** 4)
    P_acc = (3e-15) ** 2 * (1.0 + (4e-4 / f) ** 2) * (1.0 + (f / 8e-3) ** 4)
    return (10.0 / (3.0 * L ** 2)
            * (P_oms + 2.0 * (1.0 + jnp.cos(f / fstar) ** 2) * P_acc / (2.0 * jnp.pi * f) ** 4)
            * (1.0 + 0.6 * (f / fstar) ** 2))


def _foreground_ae(f, S_A_instr, L: float = _ARM_m):
    """Foreground PSD in the A (= E) TDI-1 channel, fractional frequency, 1/Hz."""
    return S_A_instr * galactic_confusion_strain(f) / robson_instrumental_strain(f, L=L)


def psd_tdi1_instrumental(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """The instrumental TDI-1 A, E, T PSD alone (no foreground), 1/Hz."""
    return _psd_tdi1_instr(f, A=A, P=P, L=L)


def psd_tdi1(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    One-sided TDI-1 noise PSD for A, E, T: instrumental + Galactic confusion foreground.

    The foreground (`_foreground_ae`) is added to A and E only. Everything downstream --
    the TDI-2 PSD, the bin variances, the noise draws -- inherits it through this function.
    """
    S_A, S_E, S_T = _psd_tdi1_instr(f, A=A, P=P, L=L)
    S_fg = _foreground_ae(f, S_A, L=L)
    return S_A + S_fg, S_E + S_fg, S_T


# ---------------------------------------------------------------------------
# TDI-2 one-sided PSD  (= TDI-1 x |TF1|^2)
# ---------------------------------------------------------------------------

def psd_tdi2(f, A: float = _A_DEF, P: float = _P_DEF, L: float = _ARM_m):
    """
    One-sided TDI-2 noise PSD for the A, E, T channels, in fractional frequency.

        S_TDI2(f) = 4 sin^2(4 pi f T) . S_TDI1(f),   T = L/c

    The factor 4 sin^2(4 pi f T) = |TF1(f)|^2 is the squared magnitude of the
    TDI-1 transfer function.  With this PSD the optimal SNR matches TDI-1.

    Note that |TF1|^2 vanishes at f = k/(4T): the TDI-2 PSD has nulls where the
    TDI-1 PSD does not.  Any PSD-weighted quantity has to be handled with care
    there, which is why quantitative comparisons in this project are done in
    TDI-1 wherever the choice is free.

    Returns
    -------
    S_A, S_E, S_T : arrays of shape (len(f),), units 1/Hz.
    """
    T = L / _C_SI
    S_A1, S_E1, S_T1 = psd_tdi1(f, A=A, P=P, L=L)
    factor = 4.0 * jnp.sin(4.0 * jnp.pi * f * T) ** 2
    return factor * S_A1, factor * S_E1, factor * S_T1


# ---------------------------------------------------------------------------
# One-sided PSD -> per-bin variance
# ---------------------------------------------------------------------------

def bin_variance(psd_one_sided, t_obs: float):
    """
    Convert a one-sided PSD [1/Hz] to the variance of a continuous-FT bin [s^2].

        sigma_k^2 = (t_obs / 2) * S_1(f_k)

    This is the quantity `sample_noise_fd` draws from and the quantity
    `jaxglitches.likelihood` divides by, so that

        E[|n_fd[k, c]|^2] = sigma_k^2

    and the matched-filter SNR comes out dimensionless and physical.
    """
    return 0.5 * t_obs * psd_one_sided


# ---------------------------------------------------------------------------
# Convenience: per-bin variance arrays, ready for the likelihood
# ---------------------------------------------------------------------------

def psd_tdi1_array(f, *, t_obs: float, A: float = _A_DEF, P: float = _P_DEF,
                   L: float = _ARM_m):
    """
    TDI-1 per-bin variance as an (F, 3) array — columns [A, E, T], units s^2.

    `t_obs` is keyword-only and mandatory on purpose: the difference between a
    one-sided PSD and a bin variance is a silent factor T_obs/2 in every SNR, so
    it must not be defaultable.
    """
    S_A, S_E, S_T = psd_tdi1(f, A=A, P=P, L=L)
    return bin_variance(jnp.stack([S_A, S_E, S_T], axis=-1), t_obs)


def psd_tdi2_array(f, *, t_obs: float, A: float = _A_DEF, P: float = _P_DEF,
                   L: float = _ARM_m):
    """
    TDI-2 per-bin variance as an (F, 3) array — columns [A, E, T], units s^2.

    See `psd_tdi1_array` for why `t_obs` is mandatory.
    """
    S_A, S_E, S_T = psd_tdi2(f, A=A, P=P, L=L)
    return bin_variance(jnp.stack([S_A, S_E, S_T], axis=-1), t_obs)


# ---------------------------------------------------------------------------
# Frequency-domain noise sampling
# ---------------------------------------------------------------------------

@jax.jit
def sample_noise_fd(key, psd_fd):
    """
    Draw frequency-domain Gaussian noise matching the supplied per-bin variance.

    Each positive-frequency bin k satisfies  E[|n[k,c]|^2] = psd_fd[k,c].
    The DC bin (index 0) is set to zero.

    Parameters
    ----------
    key    : JAX PRNG key.
    psd_fd : real (F, C) array — per-bin variance [s^2], i.e. the output of
             `psd_tdi1_array(f_safe, t_obs=...)` or `psd_tdi2_array(...)`.
             Passing a bare one-sided PSD here gives noise too small by
             sqrt(T_obs/2).

    Returns
    -------
    n_fd : complex (F, C) array, same units as `clean_signal_f` (seconds).
    """
    shape = psd_fd.shape
    k_r, k_i = jax.random.split(key)
    z_r = jax.random.normal(k_r, shape)
    z_i = jax.random.normal(k_i, shape)
    n_fd = jnp.sqrt(psd_fd / 2.0) * (z_r + 1j * z_i)
    # zero DC bin to avoid issues at f=0
    return n_fd.at[0].set(0.0 + 0.0j)


# ---------------------------------------------------------------------------
# Check figure: instrumental, foreground and total PSDs
# ---------------------------------------------------------------------------

def plot_psd_foreground(out="fig_psd_foreground.pdf"):
    """
    Instrumental, Galactic-foreground and total A-channel noise, TDI-1 and TDI-2, plus the
    same spectra referred to strain against Robson, Cornish & Liu (2019) S_n and S_n + S_c,
    which they must reproduce by construction. Prints the total/instrumental amplitude
    ratio at a few frequencies.

        python noise.py [out.pdf]
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = jnp.geomspace(1e-5, 1e-1, 4000)
    Ai, _, _ = psd_tdi1_instrumental(f)
    fg = _foreground_ae(f, Ai)
    A1, _, T1 = psd_tdi1(f)
    A2, _, T2 = psd_tdi2(f)
    gen2 = A2 / A1
    Sn, Sc = robson_instrumental_strain(f), galactic_confusion_strain(f)
    R = Ai / Sn                                  # strain -> fractional frequency, A channel
    f_ = np.asarray(f)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (tot, ins, fgc, tt, title) in zip(axs[:2], (
            (A1, Ai, fg, T1, "TDI-1"),
            (A2, gen2 * Ai, gen2 * fg, T2, "TDI-2"))):
        ax.loglog(f_, np.sqrt(np.asarray(ins)), color="C0", lw=1.2, label="A, instrumental")
        ax.loglog(f_, np.sqrt(np.asarray(fgc)), color="C1", lw=1.2, label="A, Galactic foreground")
        ax.loglog(f_, np.sqrt(np.asarray(tot)), color="k", lw=1.6, ls="--", label="A, total")
        ax.loglog(f_, np.sqrt(np.asarray(tt)), color="C2", lw=0.9, label="T (no foreground)")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(1e-24, 1e-17)
    for ax in axs:
        ax.axvspan(3e-3, 1e-1, color="0.85", lw=0)
        ax.axvline(2e-3, color="C3", lw=0.8, ls=":")
        ax.axvline(1 / (2 * np.pi * 300), color="C4", lw=0.8, ls=":")
        ax.set_xlabel("frequency [Hz]")
        ax.grid(alpha=0.3, which="both", lw=0.3)
    axs[0].set_ylabel(r"$\sqrt{S}$ [Hz$^{-1/2}$], fractional frequency")
    axs[2].loglog(f_, np.sqrt(np.asarray(Sn)), color="C0", lw=1.2, label=r"Robson+19 $S_n$")
    axs[2].loglog(f_, np.sqrt(np.asarray(Sn + Sc)), color="k", lw=1.2, label=r"Robson+19 $S_n+S_c$")
    axs[2].loglog(f_, np.sqrt(np.asarray(A1 / R)), color="C1", lw=1.0, ls="--", label=r"ours, $S_A/R$")
    axs[2].set_ylim(1e-20, 1e-14)
    axs[2].set_ylabel(r"strain ASD [Hz$^{-1/2}$]")
    axs[2].set_title("referred to strain (check)", fontsize=9)
    axs[0].legend(fontsize=6.5, loc="upper right")
    axs[2].legend(fontsize=6.5, loc="upper right")
    axs[1].text(2.1e-3, 2e-24, r"$f_0$", color="C3", fontsize=7)
    axs[1].text(0.55e-3, 2e-24, r"$f_{\rm knee}$", color="C4", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    for v in (3e-4, 5.3e-4, 1e-3, 2e-3, 3e-3):
        r = float(np.sqrt(np.interp(v, f_, np.asarray(A1 / Ai))))
        print(f"{v * 1e3:5.2f} mHz: total/instrumental ASD = {r:.2f}")
    print(f"saved {out}")


if __name__ == "__main__":
    import sys
    plot_psd_foreground(*sys.argv[1:2])
