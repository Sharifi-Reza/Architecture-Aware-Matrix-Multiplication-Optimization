# Architecture-Aware Matrix Multiplication Optimization

This project takes a naive triple-loop matrix multiplication in C and makes it about **114× faster**, one hardware-motivated step at a time. Each phase targets a specific part of the CPU: registers, the cache hierarchy, the floating-point pipeline, SIMD units, and finally multiple cores. Every phase was benchmarked on real hardware and explained in terms of the processor's architecture.

All code is compiled with **`-O0`**, so the compiler does not optimize anything on its own. Every speedup reported here comes from the hand-written code.

This was a term project for the **Microprocessor Systems Design** course at Sharif University of Technology. A report for each phase is in [`reports/`](reports/) (in Persian).

---

## Test Hardware

| | |
|---|---|
| CPU | Intel Core i7, 11th generation, 4 cores |
| L1 cache | 80 KB per core |
| L2 cache | 1.25 MB per core |
| L3 cache | 12 MB (shared) |
| SIMD | AVX / AVX2 (256-bit) |
| OS / toolchain | Windows, MinGW-w64 GCC (Code::Blocks) |

---

## The Optimization Path at a Glance (n = 2048)

Each step keeps the techniques from the steps before it.

| Step | Implementation | Time (s) | Speedup vs. previous | Cumulative speedup |
|---|---|---|---|---|
| Phase 0 | `matrix_mult_index` (reference) | 77.8 | — | 1× |
| Phase 1 | `matrix_mult_ptr_reg` | 44.2 | 1.8× | 1.8× |
| Phase 2 | `matrix_mult_transpose` | 8.75 | 5.1× | 8.9× |
| Phase 3 | `matrix_mult_unrolled_4` | 4.96 | 1.8× | 15.7× |
| Phase 4 | `matrix_mult_cache_friendly_vec` | 2.25 | 2.2× | 34.6× |
| Phase 5 | `matrix_mult_omp_vec` | 0.68 | 3.3× | **~114×** |

Times are either averages over several runs or the value the runs settled on.

---

## Phase-by-Phase Results

### Phase 1: Pointers and registers

Phase 1 replaces index arithmetic (`a[i*n + k]`) with pointers that are incremented as the loops advance, and adds `register` hints for the hottest variables. At `-O0` the compiler keeps every variable in memory unless told otherwise, so the `register` hints have a real effect.

| n | `index` | `ptr_reg` | `ptr_no_reg` |
|---|---|---|---|
| 100 | 0.01 | 0.00 | 0.00 |
| 200 | 0.03 | 0.01 | 0.02 |
| 300 | 0.08 | 0.03 | 0.06 |
| 400 | 0.21 | 0.06 | 0.13 |
| 800 | 1.80 | 0.51 | 1.13 |

Pointers with register hints give about a **3× speedup** at small sizes. Without the hints, the speedup is only about 1.6–2×.

**Power-of-two anomaly.** Running `ptr_reg` over a finer range of sizes exposed a sharp slowdown at exactly n = 1024:

| n | 900 | 1000 | **1024** | 1100 |
|---|---|---|---|---|
| Time (s) | 0.70 | 1.25 | **2.30** | 1.60 |

When n is a power of two, consecutive elements of a column are spaced exactly 8 KB apart in memory. Those addresses all map to the same cache set, so they keep evicting each other (**conflict misses**) even though most of the cache is empty. A slightly larger matrix, n = 1100, runs faster than n = 1024.

### Phase 2: Memory access order

**Transpose.** Phase 2 first copies **B** into its transpose. The inner loop then reads both matrices sequentially along rows, instead of jumping down a column of B with a stride of `n`.

| n | `ptr_reg` | `transpose` | Speedup |
|---|---|---|---|
| 512 | 0.25 | 0.12 | 2.08× |
| 1024 | 2.30 | 1.00 | 2.30× |
| 2048 | 44.2 | 8.75 | 5.05× |
| 4096 | 537 | 69 | **7.78×** |

The speedup grows with n. Once the matrices no longer fit in cache, the column-wise access pattern becomes more and more costly.

**Blocking (tiling).** `matrix_mult_block` splits the computation into tiles of size `block_size`, using `i-k-j` order inside each tile. The table below shows a block-size sweep, with times in seconds.

| n \ block size | 8 | 16 | 32 | 64 | 128 | 256 | 512 | Best speedup vs. `ptr_reg` |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.10 | 0.09 | 0.09 | **0.08** | **0.08** | — | — | 3.1× |
| 1024 | 0.80 | 0.72 | 0.77 | 0.63 | **0.56** | 0.57 | **0.56** | 4.1× |
| 2048 | 6.52 | 6.14 | 6.30 | 5.26 | 4.84 | **4.50** | — | 9.8× |
| 4096 | 82.0 | 68.5 | 70.0 | 57.7 | 51.0 | **44.1** | 46.0 | **12.2×** |

The block sizes line up with the cache hierarchy:

- **8 and 16:** too small. The six nested loops add overhead, and the tiles use only a fraction of the L1 cache.
- **64:** three 64×64 tiles of doubles take about 98 KB, close to the 80 KB L1. This is where the large improvement starts.
- **128 and 256:** the three tiles (about 393 KB and about 1.5 MB) fit in L2 and part of L3. These give the best balance.
- **512:** each tile is about 2 MB, more than the L2 can hold. Performance drops again.

Larger matrices favor larger blocks. When the data has to come from RAM, it pays to do as much computation as possible on each block once it is loaded.

### Phase 3: Loop unrolling and the floating-point pipeline

Phase 3 builds on the transpose version. It unrolls the inner loop by factors of 2, 4, and 8, with a **separate accumulator for each unrolled step**. Separate accumulators remove the dependency on a single running sum, so several independent multiply-add operations can be in the FPU pipeline at once.

| n | `transpose` | Unroll ×2 | Unroll ×4 | Unroll ×8 | Best speedup |
|---|---|---|---|---|---|
| 512 | 0.12 | 0.06 | **0.05** | **0.05** | 2.4× |
| 1024 | 1.00 | 0.52 | **0.40** | 0.41 | 2.5× |
| 2048 | 8.75 | 5.23 | **4.96** | 5.07 | 1.76× |
| 4096 | 69.0 | 48.2 | **44.1** | 45.0 | 1.56× |

An unroll factor of 4 is the best choice. Going to 8 brings no further improvement.

**Microbenchmark.** A separate benchmark timed loops containing more and more independent add and multiply instructions. Execution time stayed flat up to about **6 independent operations in flight**, then started to rise. This suggests pipeline depth × number of execution units ≈ 6. That would mean either a depth-3 pipeline with 2 units, or a depth-6 pipeline with 1 unit. This explains why an unroll factor of 4 already keeps the FPU full, while 8 asks for more parallelism than the hardware can provide.

### Phase 4: Cache-friendly strip mining and AVX vectorization

**Cache-friendly.** This step processes the columns of the transposed B in strips of 64. The strip stays in L1/L2 while every row of A passes over it.

| n | `transpose` | `cache_friendly` | Speedup |
|---|---|---|---|
| 512 | 0.12 | 0.12 | 1.00× |
| 1024 | 1.00 | 0.95 | 1.05× |
| 2048 | 8.75 | 7.51 | 1.16× |
| 4096 | 69.0 | 62.9 | 1.10× |

At small sizes the transposed version is already fast, helped by the hardware prefetchers, so strip mining adds nothing. It only helps once the matrices are larger than L2 and L3.

**Vectorization.** `matrix_mult_cache_friendly_vec` uses GCC vector extensions (`typedef double v4df __attribute__((vector_size(32)))`). Each 256-bit AVX register holds 4 doubles. With four vector accumulators, each loop iteration processes **16 elements**, which combines SIMD with the unrolling from Phase 3.

| n | `transpose` | `cache_friendly_vec` | Speedup |
|---|---|---|---|
| 512 | 0.12 | 0.04 | 3.00× |
| 1024 | 1.00 | 0.28 | 3.57× |
| 2048 | 8.75 | 2.25 | 3.88× |
| 4096 | 69.0 | 21.10 | 3.27× |

The result is close to the theoretical 4× for 4-wide SIMD.

### Phase 5: OpenMP multithreading

`matrix_mult_omp_vec` keeps everything from earlier phases and spreads the work across cores:

- The transpose itself runs in parallel.
- A **single parallel region** is opened outside the block loop. Threads are created once rather than for every block, which avoids repeated fork/join overhead.
- **`schedule(static)`** gives each thread a contiguous range of rows. This keeps memory access local and adds no scheduling overhead at runtime.
- Timing uses `omp_get_wtime()` for wall-clock time. `clock()` can report CPU time summed across all threads.

| n | Phase 4: `cache_friendly_vec` (1 thread) | Phase 5: `omp_vec` | Speedup |
|---|---|---|---|
| 1024 | 0.28 | 0.08 | 3.50× |
| 2048 | 2.25 | 0.68 | 3.30× |
| 4096 | 21.10 | 6.33 | 3.34× |

On 4 cores the result is about 3.4×, against an ideal of 4×. The gap comes from thread management, synchronization, and memory bandwidth saturation at large sizes.

---

## Repository Structure

```
.
├── main.c        # All implementations (Phases 0–5) and the interactive benchmark driver
└── reports/
    ├── phase1-401101932.pdf   # Pointers & registers, power-of-two anomaly
    ├── phase2_401101932.pdf   # Transpose, blocking and block-size sweep
    ├── phase3_401101932.pdf   # Loop unrolling, FPU pipeline microbenchmark
    ├── phase4_401101932.pdf   # Cache-friendly strip mining, AVX vectorization
    └── phase5-401101932.pdf   # OpenMP multithreading, final comparison
```

| Function | Phase |
|---|---|
| `matrix_mult_index` | 0 – Reference |
| `matrix_mult_ptr_reg`, `matrix_mult_ptr_no_reg` | 1 |
| `matrix_mult_transpose`, `matrix_mult_block` | 2 |
| `matrix_mult_unrolled_2`, `_4`, `_8` | 3 |
| `matrix_mult_cache_friendly`, `matrix_mult_cache_friendly_vec` | 4 |
| `matrix_mult_omp_vec` | 5 |

---

## Building and Running

The code uses `_aligned_malloc` / `_aligned_free` from `<malloc.h>`, so it builds as-is on **Windows with MinGW-w64 GCC**:

```bash
gcc -O0 -mavx2 -Wall -Wextra -fopenmp -o matmul main.c
./matmul
```

To build on Linux, replace `_aligned_malloc(size, 64)` with `aligned_alloc(64, size)` and `_aligned_free` with `free`.

To choose the number of threads for Phase 5:

```bash
set OMP_NUM_THREADS=4       # Windows cmd
export OMP_NUM_THREADS=4    # Linux / macOS
```

### Usage

The program is interactive. It asks for the matrix size `n`, fills two random matrices, and then asks (y/n) whether to run each implementation in turn, printing the execution time for each one.

**Correctness check:** the first implementation you run becomes the reference. The most recent other implementation is compared against it element by element, with a relative tolerance of 1e-10. The program then prints either `Matrixes are equivalent` or the index of the first mismatch.

---



## Author

**Mohammadreza Sharifi**
B.Sc. Electrical Engineering (Electronics), Sharif University of Technology
