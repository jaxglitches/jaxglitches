# jaxglitches

![jaxglitches logo](assets/branding/jaxglitches-logo.svg)

Differentiable LISA glitch waveforms, time-delay interferometry (TDI) responses,
and parameter-estimation utilities built with JAX. Importing the package enables
float64 arithmetic.

The package models a single-exponential test-mass glitch on link 12, with
TDI generations 1 and 2, AET or XYZ channels, and equal or frozen unequal arms.
It also provides Gaussian likelihoods, Fisher matrices, priors, and synthetic
catalogues resampled from LISA Pathfinder glitch populations.

Start with the [installation and waveform example](getting-started.md), check
[units and response conventions](conventions.md), try the [example notebook](example.md), or browse the
[API reference](api.md).

The core dependencies are JAX, NumPy, `lisaglitch`, and `lisaorbits`.
Noise models used in the example notebooks live in `notebooks/noise.py` and
are not part of the installed package. Galactic-binary waveforms, wavelet
transforms, and simulation tools are optional extras.

Explore the [research notebook gallery](gallery.md) for longer validation and
approximation studies with saved outputs.

## References

- [Source code and notebooks](https://github.com/jaxglitches/jaxglitches)
- [LISA glitch simulation package](https://gitlab.in2p3.fr/lisa-simulation/glitch)
- [Baghi et al. (2022), LISA Pathfinder glitch populations](https://arxiv.org/abs/2112.07490)

## Logo assets

Download the [wordmark (SVG)](assets/branding/jaxglitches-logo.svg),
[transparent waveform mark (PNG)](assets/branding/jaxglitches-mark.png), or
[square icon (PNG)](assets/branding/jaxglitches-icon.png) for use elsewhere.
