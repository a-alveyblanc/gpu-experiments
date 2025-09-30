# Why GEMM?
- [Matricization](https://en.wikipedia.org/wiki/Tensor_reshaping#Mode-m_Flattening_/_Mode-m_Matrixization)
guarantees that any arbitrary tensor contraction can be rewritten as a GEMM

# Strides and hardware axes
- Want vectorized loads -> all threads in a warp need to access contiguous data
- No way to get contiguous data? -> data reuse as much as possible
- Axis stride (stride): given a particular axis, its stride is the number of
array elements to "jump" over to get to the next entry in that axis 
    - C-ordering, F-ordering: column- vs. row-major ordering
- Arrays have strides and so do GPU hardware axes
    - Match fastest array axis with fastest hardware axis
    - Get: contiguous loads of array elements along fastest array axis 
    - For matmul: warp-level data reuse in rows of A, coalesced load in B,
    coalesced store in C
- Up to 3D: `(threadIdx.z, threadIdx.y, threadIdx.x)` are the axes in
speed-descending order
    - For matmul: this means we want `threadIdx.x` to iterate over *columns* of 
    B and `threadIdx.y` to iterate over rows of A
    - Dimensions > 3 will require some indexing math involving integer and
    modular division
    - **Note**: this indexing math is taken care of under-the-hood for
    dimensions <= 3

# Tiling
- Many levels of block linear algebra
    - On CPUs: typically one level for each level of cache
    - On GPUs: one level per abstraction, i.e. block, warp, thread
        - Corresponding to SMEM, tensor cores, registers

# General tricks / nice-to-know info
- Avoid using `threadIdx.?`, `blockDim.?`, etc. in loops if possible because the
  compiler can only reason so much about what those things mean
- When shared memory is used `ncu` does not report L1 hit rate
    - Global loads to shared bypass L1, so if everything goes through L1 then
      hit rate is 0%

# View of the loops
- Easier to see what's happening by splitting loops instead of drawing pictures
- The basic idea is that we split loops to achieve the desired granularity, then
  add "implementation details" like fetching to shared, registers, etc. later
- This is how [scheduling languages](https://arxiv.org/abs/2410.19927) work,
i.e. by incrementally transforming an expression into something that efficiently
runs on a computer (GPU)
## Algorithm
$$
C_{ij} = \sum_k A_{ik} B_{kj}
$$

## Naive view
```python
for i in range(0, M):     # parallel over blocks
  for j in range(0, N):   # parallel over blocks
    for k in range(0, K): # sequential within each block
      # do work
```

## Shared memory tiling (1-level of memory)
```python
# outer loop level - over threadblocks
for iouter in range(0, M / BM):     # parallel over tiles 
  for jouter in range(0, N / BN):   # parallel over tiles 
    for kouter in range(0, K / BK): # sequential within each tile 
      # fetch tile from gmem -> smem

      for iinner in range(0, BM):     # parallel over threads
        for jinner in range(0, BN):   # parallel over threads
          for kinner in range(0, BK): # sequential within each thread 
            # do work
```

## 1D thread block tiling (2-levels of memory)
```python
# outer loop level - over threadblocks
for iouter in range(0, M / BM):     # parallel over tiles 
  for jouter in range(0, N / BN):   # parallel over tiles 
    for kouter in range(0, K / BK): # sequential within each tile 
      # fetch tile from gmem -> smem 
      
      for iinner_outer in range(0, BM / TM): # parallel over threads 
        for jinner in range(0, BN):          # parallel over threads
          for kinner in range(0, BK):        # sequential within each thread

            for iinner_inner in range(0, TM): # sequential within each thread 
              # do work
```

## 2D thread block tiling (2-levels of memory)
```python
# outer loop level - over threadblocks
for iouter in range(0, M / BM):     # parallel over tiles 
  for jouter in range(0, N / BN):   # parallel over tiles 
    for kouter in range(0, K / BK): # sequential within each tile 
      # fetch tile from gmem -> smem 
      
      # inner loop level - over threads
      for iinner_outer in range(0, BM / TM):   # parallel over threads
        for jinner_outer in range(0, BN / TN): # parallel over threads
          for kinner in range(0, BK):          # sequential within each thread
            # load b temporary

            for iinner_inner in range(0, TM):   # sequential within each thread 
              for jinner_inner in range(0, TN): # sequential within each thread 
                # do work
```
