# GPU

The benchmark numbers below are retained from the original research README.
Its referenced `paper/validation/` assets are not distributed in this checkout.


The package is pure JAX with no custom kernels, so the same code runs on GPU
unchanged — you only need a CUDA-enabled `jaxlib`. Check your driver with
`nvidia-smi` and install the matching extra:

```sh
uv sync --extra gpu           # CUDA 12 build, driver >= 525
uv sync --extra gpu-cuda13    # CUDA 13 build, driver >= 580
```

Verify with `python -c "import jax; print(jax.devices())"`; if it prints
`[CpuDevice(...)]` the plugin does not match the driver and JAX has silently
fallen back to CPU.

Everything stays in float64 on GPU. That matters here — the glitch phase
`exp(-2i pi f t0)` with `t0` up to a year needs the full double mantissa — and it
is not free on consumer hardware, where FP64 runs at a fraction of the FP32 rate.
It turns out not to hurt: these kernels are memory-bound rather than FLOP-bound,
and on an RTX 2000 Ada the template is ~5x faster than on 22 CPU cores for grids
above 1e4 bins. The Hessian benefits far more (14.7x one likelihood on CPU
vs 1.9x on GPU). See `paper/validation/03_benchmarks.ipynb`.

#### Benchmarking todo

Two gaps in the numbers above, both of which need hardware or a run this machine
cannot provide:

- [ ] **A second GPU.** Every benchmark here is on one Ada-generation laptop part,
  whose FP64 rate is 1/64 of FP32. A data-centre card with a 1/2 FP64 ratio (A100,
  H100) would separate the memory-bandwidth-bound reading above from a residual FLOP
  limitation: if that reading is right, the speed-up should still saturate near the
  bandwidth ratio rather than scale with FP64 throughput. One run on a cluster node
  settles it.
- [ ] **End-to-end sampling benchmark.** The year-long joint run reaches the MAP in
  3.6 s and does 16 walkers x 22,000 iterations in 52 s, but those numbers are
  reported in passing, per run. What is missing is the scaling in walkers and in
  device, measured and reported as a benchmark in its own right.

