# jaxglitches

![jaxglitches logo](docs/assets/branding/jaxglitches-logo.svg)

[![PyPI](https://img.shields.io/pypi/v/jaxglitches)](https://pypi.org/project/jaxglitches/)
[![JAX](https://img.shields.io/badge/JAX-differentiable-5B4B9A)](https://github.com/jax-ml/jax)
[![Tests](https://github.com/jaxglitches/jaxglitches/actions/workflows/ci.yml/badge.svg)](https://github.com/jaxglitches/jaxglitches/actions/workflows/ci.yml)

Differentiable LISA glitch waveforms, TDI responses, and inference utilities
built with JAX. Float64 is enabled on import.

[Documentation](https://jaxglitches.github.io/jaxglitches/) ·
[Quick-start notebook](docs/examples/quickstart.ipynb) ·
[Research notebooks](https://jaxglitches.github.io/jaxglitches/gallery/) ·
[MIT license](LICENSE)

## Install

Python 3.12 or later:

```sh
pip install jaxglitches
```

Optional extras add notebook tools, Galactic-binary waveforms (`joint`), wavelet
transforms (`wdm`), external simulation tools (`simulation`), and CUDA support
(`gpu` or `gpu-cuda13`). See the
[installation guide](https://jaxglitches.github.io/jaxglitches/getting-started/)
and [GPU notes](https://jaxglitches.github.io/jaxglitches/gpu/).

## Quick start

```python
import jax.numpy as jnp
import jaxglitches as jg

freq = jg.freq_grid(t_obs=1800.0, dt=0.5)
params = jnp.array([400.0, 1.2e-13, 0.79])  # t0 (s), Deltav (m/s), tau (s)
response = jg.clean_signal_f(params, freq, tdi=2, basis="AET")  # (1801, 3)
```

## Package scope

- Analytic time- and frequency-domain responses for a single-exponential
  test-mass glitch on link 12, with TDI-1/TDI-2 and AET/XYZ channels.
- Equal-arm and frozen unequal-arm signal builders, with per-link delays
  obtained from `lisaorbits`.
- Gaussian likelihoods, matched-filter SNR, Fisher matrices, and
  LPF-motivated priors with physical and sampling coordinates.
- Synthetic catalogues resampled from the LISA Pathfinder glitch population.

Core dependencies are JAX, NumPy, `lisaglitch`, and `lisaorbits`.
The notebooks' noise models live in `notebooks/noise.py`, outside the package.
Check the [response and noise conventions](https://jaxglitches.github.io/jaxglitches/conventions/)
before combining templates with data.

## Development

```sh
git clone https://github.com/jaxglitches/jaxglitches.git
cd jaxglitches
uv sync --locked
uv run --no-sync pytest -q
```

See [development and releases](docs/development.md) for documentation builds,
CI, and PyPI publishing. Longer validation studies are shown in the
[notebook gallery](https://jaxglitches.github.io/jaxglitches/gallery/) with their
saved outputs. The original [paper reproduction workflow](docs/reproducing-paper.md)
requires the separate paper workspace and stored analysis inputs.
