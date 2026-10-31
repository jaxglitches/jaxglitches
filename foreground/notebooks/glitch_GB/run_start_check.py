"""Does the start of the fiducial chain leave a trace? (Sec. "Locating and characterising the peak")

The production chains of `glitch_and_gb.ipynb` start from a ball around the single,
undamped Newton step from the injection. This runs three binned TDI-1 chains of the same
length and compares them:

    single   the production start, seed 1
    backtr   the backtracked maximum of `fd_pipeline.laplace`, seed 1
    seed     the production start, seed 3 -- the control: two seeds, same start

and prints, for backtr and for seed against single, the largest and rms relative width
difference and the rms median difference in units of the single-start width. Writes
`start_check.npz` next to this script. A minute on one GPU.

    python run_start_check.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))           # noise.py
import fd_pipeline as fp
import noise as ns
from jaxglitches.priors import _DELTAV_MIN, _DELTAV_MAX, _TAU_MIN, _TAU_MAX

N_BURN, N_SAMP = 2_000, 20_000                 # as glitch_and_gb.ipynb
FALLBACK = jnp.array([1e-8, 0.2, 0.02, 0.05, 0.5, 0.02, 0.05])   # as glitch_and_gb.ipynb


def main():
    DS = np.load(HERE / "dataset.npz")
    grid = fp.make_grid(int(DS["N"]), float(DS["DT"]))
    model, n_gb = fp.make_gb_model(grid["T_OBS"], int(DS["N_GB"]))
    coords = fp.Coords(free_sky=False)
    th = coords.to_sampling(jnp.asarray(DS["gb_true"]), jnp.asarray(DS["glitch_true"]))
    bounds = {"log_f0": (np.log(1e-4), np.log(3e-3)),
              "log_fdot": (np.log(1e-22), np.log(1e-15)),
              "log_A_gb": (np.log(1e-25), np.log(1e-20)),
              "psi": (0.0, float(np.pi)), "t0": (0.0, float(grid["T_OBS"])),
              "log_Ag": (float(np.log(_DELTAV_MIN * _TAU_MIN)),
                         float(np.log(_DELTAV_MAX * _TAU_MAX))),
              "log_tau": (float(np.log(_TAU_MIN)), float(np.log(_TAU_MAX)))}
    log_prior = fp.make_log_prior(coords, bounds)
    psd = ns.psd_tdi1_array(grid["f_safe"], t_obs=grid["T_OBS"])
    ll, _ = fp.build_hybrid(grid, jnp.asarray(DS["data_tdi1"]), psd, int(DS["k_min"]),
                            model, n_gb, coords, tdi=1)
    lp = jax.jit(lambda t: ll(t) + log_prior(t))

    # the production start: one undamped Newton step, its Laplace diagonal as the ball
    g, H = jax.grad(lp)(th), jax.hessian(lp)(th)
    x_one = th - jnp.linalg.solve(H, g)
    s_raw = jnp.sqrt(jnp.abs(jnp.diag(-jnp.linalg.inv(H))))
    s_one = jnp.where(jnp.isfinite(s_raw) & (s_raw > 1e-10) & (s_raw < 10.0), s_raw, FALLBACK)
    x_bt, s_bt = fp.laplace(lp, th, FALLBACK)
    print(f"log-posterior: truth {float(lp(th)):.3f}  one step {float(lp(x_one)):.3f}  "
          f"backtracked {float(lp(x_bt)):.3f}")

    runs = {}
    for name, x0, s0, seed in (("single", x_one, s_one, 1), ("backtr", x_bt, s_bt, 1),
                               ("seed", x_one, s_one, 3)):
        ch = np.asarray(fp.run_chain(ll, log_prior, x0, s0, 7, seed,
                                     nburn=N_BURN, nsamp=N_SAMP))
        runs[name] = ch
        print(f"  {name:7s} chain {ch.shape}", flush=True)

    ref = runs["single"]
    sig, med = ref.std(axis=0), np.median(ref, axis=0)
    out = {k: v for k, v in runs.items()}
    for name in ("backtr", "seed"):
        dw = runs[name].std(axis=0) / sig - 1.0
        dm = (np.median(runs[name], axis=0) - med) / sig
        out[f"{name}_dwidth"], out[f"{name}_dmedian"] = dw, dm
        print(f"{name:7s} vs single: width max {100 * np.abs(dw).max():.1f}%  "
              f"rms {100 * np.sqrt(np.mean(dw ** 2)):.1f}%;  median rms "
              f"{np.sqrt(np.mean(dm ** 2)):.3f} sigma, max {np.abs(dm).max():.3f} sigma")
    np.savez_compressed(HERE / "start_check.npz", **out)
    print(f"saved {HERE / 'start_check.npz'}")


if __name__ == "__main__":
    main()
