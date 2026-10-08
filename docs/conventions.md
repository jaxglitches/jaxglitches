# Conventions

## Parameters and channels

Physical parameter vectors have order `[t0, Deltav, tau]`: onset in seconds,
velocity kick in m/s, and exponential decay time in seconds. Sampling-coordinate
helpers transform kick and decay to log coordinates; use `to_sampling` and
`to_physical` when crossing that boundary.

`clean_signal_f` returns an `(F, 3)` complex array and `clean_signal_t` an
`(N, 3)` real array. The default channels are A, E, T. Set `basis="XYZ"` for
X, Y, Z. Set `tdi=1` or `tdi=2` explicitly when matching an external response.
Zero-frequency values in the Fourier signal builder are set to zero.

## Fourier and noise normalization

Keep the Fourier grid, observation duration, windowing, response, and noise
normalization consistent. The notebook noise module describes its continuous
Fourier scaling and the conversion to the discrete covariance used by the
package likelihood. Inspect `notebooks/noise.py` and the
[likelihood API](api.md#jaxglitches.likelihood) before supplying an external PSD.
A taper or cropped observation window must be reflected in the template too.

## Arm delays

Equal-arm helpers use a single light travel time. Unequal-arm helpers accept six
link delays in order `(12, 23, 31, 13, 32, 21)`, frozen at the glitch epoch.
`link_ltt` obtains these from an orbit object. Frozen delays are an approximation
when orbital evolution matters across the response window.

## Numerical precision

The package enables JAX float64 on import. Comparisons of independently compiled
responses can differ at machine precision, especially where channel combinations
cancel. Numerical agreement and passing tests do not establish sampler mixing
or scientific recovery for a new dataset.
