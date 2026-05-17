<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-24 | Updated: 2026-03-24 -->

# triton_code

## Purpose
Triton GPU kernel implementations for relocated attention. Contains Flash Attention v2-based kernels adapted for sparse/local attention patterns with shared text regions, plus benchmarking and correctness test suites.

## Key Files

| File | Description |
|------|-------------|
| `reloc_triton_kernel_parallel.py` | Main Triton kernel — Flash Attention v2 adapted for relocated sparse attention with parallel computation of shared (text) and local (image tile) regions |
| `reloc_triton_kernel_parallel_sliding.py` | Sliding window variant — adds offset-based sliding to the parallel kernel for varying attention patterns across layers |
| `benchmark.py` | Performance benchmark comparing Triton kernel vs text SDPA vs full SDPA vs parallel variant across configurations |
| `benchmark_no_shared.py` | Benchmark isolating the impact of the shared region (`n_shared=0` vs `n_shared=768` vs full SDPA) |
| `benchmark_vs_sdpa.py` | Direct comparison benchmark: Triton sparse attention kernel vs PyTorch `F.scaled_dot_product_attention` |
| `calc_flops.py` | FLOPs calculator for sparse vs full attention — computes theoretical speedup from sparsity |
| `flop.py` | Additional FLOPs analysis utilities |
| `test_clean.py` | Clean test for reorder + sliding window behavior — verifies `reorder(offset=0) + kernel(offset=X)` correctness |
| `test_simple_equivalence.py` | Simplified round-trip test: reorder → kernel → recover cycle verification |

## For AI Agents

### Working In This Directory
- **Kernel architecture**: Based on Flash Attention v2 (Tri Dao) with modifications for sparse attention patterns. Uses Triton's `@triton.jit` decorator for GPU compilation
- **Two kernel variants**:
  - `reloc_triton_kernel_parallel.py`: Fixed offset, parallel shared/local computation
  - `reloc_triton_kernel_parallel_sliding.py`: Per-layer sliding window offsets
- **Entry point**: The `attention()` function in each kernel file is the main callable. Parameters include `q, k, v, N_CTX_shared, N_CTX, causal, sm_scale, GROUPS, USE_FP32`
- **TMA support**: Kernels detect and optionally use Tensor Memory Access (SM >= 9.0 / Hopper)
- **Device setup**: Uses `triton.runtime.driver.active.get_active_torch_device()` for device detection
- Tests import from parent directory (`sys.path.insert`) to access `reorder_utils_sliding_shared`

### Testing Requirements
- Run tests with CUDA available — CPU fallback will skip kernel tests
- `test_clean.py`: Verifies sliding window correctness end-to-end
- `test_simple_equivalence.py`: Verifies reorder/recover round-trip fidelity
- Always validate with `torch.allclose` against PyTorch SDPA reference
- Test with both `group_sizes = [4, 16]` and `height_widths = [1024, 2048]`

### Common Patterns
- Block sizes: `BLOCK_M` and `BLOCK_N` are Triton constexpr parameters (typically 64 or 128)
- Head dimension: `HEAD_DIM = 128` (FLUX standard)
- Number of attention heads: 24
- BF16 precision for production, FP32 accumulation optional via `USE_FP32` flag
- Benchmarks use `torch.utils.benchmark.Timer` for reliable GPU timing with warmup

### Performance Notes
- Sparse attention FLOPs scale as `O(N_shared * N_total + N_local_group * N_local_group * GROUPS)` vs `O(N_total^2)` for full attention
- Shared region (text tokens) attends to all tokens; image tile tokens attend only within their tile + shared
- Set `TRITON_DISABLE_CACHE=1` during development to force recompilation

## Dependencies

### Internal
- `../reorder_utils_sliding_shared.py` — index computation for sliding window tests
- `../reorder_utils.py` — basic reorder functions

### External
- `triton` 3.3+ — kernel compiler and JIT
- `torch` — tensor operations, CUDA events, benchmarking
- `pytest` — test framework (imported but tests also run standalone)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
