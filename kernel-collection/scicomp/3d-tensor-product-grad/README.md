Quick peek at various optimizations for memory-bound tensor contractions that
would typically be found in matrix-free FEM solvers.

Typical tensor shape is $(n_e, n, n, n)$, where $n$ is $O(10)$, and $n_e$ varies
in magnitude based on the parallel decomposition, but $O(10^5)$ elements per GPU
is probably where we'd expect to sit.

The contraction we carry out is applying an operator of shape $(n, n)$ to each
spatial axis to do things like differentiate, interpolate, etc. For example,
differentiation looks something like

$\partial_x u^e_{ijk} = \sum_{\ell} A_{i\ell} u^e_{\ell  jk}$

$\partial_y u^e_{ijk} = \sum_{\ell} A_{j\ell} u^e_{i\ell  k}$

$\partial_z u^e_{ijk} = \sum_{\ell} A_{k\ell} u^e_{ij\ell k}$

Hence, we need $O(n^3)$ entries and perform $O(n^4)$ FLOPs per contraction. The
higher order our discretization, the larger $n$ will be. Since the number of
elements scales both the total number of entries and total number of FLOPs, it
suffices to look at the arithmetic intensity of the contraction within a single
element. Assuming we only need to load data necessary for a computation once,

$\frac{n^4 \text{FLOPs}}{b \cdot n^3 \text{bytes}} = \frac{n\text{FLOP}}{b
\text{byte}}$

So, in general, our arithmetic intensity is $\frac{n}{b}$. This is the same as
matrix multiplication, but unfortunately we are not so lucky that our $n$ is as
large as in matrix multiplication. For $n = 8$ with FP64, our arithmetic
intensity is a whopping 1 FLOP/byte. For $n = 20$, we have 2.5 FLOPs/byte. 

**This means kernels which are *only* performing contractions to compute the
derivative or apply a mass matrix will likely always be memory-bound. Therefore,
it would be wise to try to increase the arithmetic intensity by forming full
operators (like advection, diffusion, etc.) via fusion.**

Here, we only take a look at computing a gradient and do not consider fused
operators.

On a 3090 with FP32, 100k elements 8 DOFs per element:
```
===== Naive kernel =====
Runtime: 1.3589 ms
GFLOP/s: 602.8352
Gbyte/s: 150.7090
========================
===== Shared memory kernel =====
Runtime: 1.2421 ms
GFLOP/s: 659.5218
Gbyte/s: 164.8807
================================
Passed
===== Shared memory kernel, no conflicts =====
Runtime: 1.1346 ms
GFLOP/s: 722.0216
Gbyte/s: 180.5056
==============================================
Passed
===== Register tiled kernel =====
Runtime: 1.0291 ms
GFLOP/s: 796.0199
Gbyte/s: 199.0052
=================================
Passed
===== Register tiled kernel, const A =====
Runtime: 2.1737 ms
GFLOP/s: 376.8641
Gbyte/s: 94.2161
==========================================
Passed
===== Register tiled kernel, 4 elements per block =====
Runtime: 1.0488 ms
GFLOP/s: 781.1069
Gbyte/s: 195.2770
========================================================
Passed
===== Register tiled explicit unroll =====
Runtime: 1.0045 ms
GFLOP/s: 815.4944
Total GB: 0.2048
Gbyte/s: 203.8738
==========================================
Passed
```

## Final kernel summary statistics from Nsight
![Summary statistics](./nsight-register-tiled-explicit-unroll.png "Final kernel
summary statistics")

## TODO / stuff for later
- Final push would be to entirely eliminate shared memory bank conflicts
- Pipelining likely not worth it since we don't have enough compute to cover the
  loads 
- Vectorized loads from shared could also add a nice boost 
