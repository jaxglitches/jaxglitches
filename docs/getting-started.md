# Getting started

## Install

Python 3.12 or later is required. Until the first PyPI release is published,
install from the canonical repository:

```sh
pip install git+https://github.com/jaxglitches/jaxglitches.git
```

Once a release is available:

```sh
pip install jaxglitches
```

For an editable development installation:

```sh
git clone https://github.com/jaxglitches/jaxglitches.git
cd jaxglitches
uv sync --locked
```

## Build a waveform

```python
import jax.numpy as jnp
import jaxglitches as jg

freq = jg.freq_grid(t_obs=1800.0, dt=0.5)
params = jnp.array([400.0, 1.2e-13, 0.79])
response = jg.clean_signal_f(params, freq, tdi=2, basis="XYZ")
print(response.shape)  # (1801, 3)
```

The parameter vector contains onset time in seconds, velocity kick in m/s,
and decay time in seconds. Columns above are X, Y, Z; the default basis is AET.
See [conventions](conventions.md) before comparing against data or noise models.

Run the [example notebook](example.md) for time-domain plots and JAX derivatives.

## Optional dependencies

| Extra | Purpose |
| --- | --- |
| `notebook` | Plotting, notebook execution, and Jexplore sampling |
| `joint` | Galactic-binary responses through `jaxgb` |
| `wdm` | Wavelet transforms through `wdm-transform` |
| `simulation` | `lisainstrument` and `pytdi` validation tools |
| `gpu` | JAX CUDA 12 support on Linux x86_64 |
| `gpu-cuda13` | JAX CUDA 13 support on Linux x86_64 |

For the joint glitch/binary notebooks:

```sh
uv sync --locked --extra notebook --extra joint --extra wdm
```

GPU extras require a compatible NVIDIA driver. Verify the available devices
with `python -c "import jax; print(jax.devices())"` after installation.
