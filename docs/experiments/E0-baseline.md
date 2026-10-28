# E0: ABACUS 基线（1 核 + 当前 GPU 环境）

## 环境

- ABACUS：`/root/abacus-develop/build_ci/abacus_basic_para`，v3.11.0-beta7，
  1 MPI / `OMP_NUM_THREADS=1`
- ad-dft：RTX 3070 Ti 8GB / WSL2，JAX 0.12.0.1 + CUDA 12.9，FP64
- 赝势：`Si.pz-vbc.UPF`，sha256 `6ab22ac0…d699b`
- 案例：`tests/regress/si-gamma`，Γ 点，`nbands=6`，`fixed`，`symmetry 0`

## 基准

| `ecutwfc` | `nG` | `nfft` | ABACUS | ad-dft SCF | 比值 |
|---|---:|---:|---:|---:|---:|
| 20 Ry | 411 | 13824 | 0.184 s | 0.45 s | 2.4× |
| 40 Ry | 1139 | 32768 | 0.35 s | 1.53 s | 4.4× |
| 100 Ry | 4477 | 110592 | 0.68 s | 35.2 s | 52× |

100 Ry ad-dft 分项：

| 项 | 单次 | 调用 | 总计 |
|---|---:|---:|---:|
| dense eigh | 2.36 s | 15 | 35.4 s |
| H 组装 | 9 ms | 15 | 0.135 s |
| Hartree | 2 ms | 15 | 0.03 s |
| density | 2 ms | 15 | 0.03 s |
| XC | 2 ms | 15 | 0.03 s |

## 结论

- 100 Ry 双方都是 10 次迭代；
- ad-dft 的差距 ~99% 来自 dense eigh；
- 10× 目标要求 `SCF+force+stress < 0.068 s`（vs 1 核）；
- 下一步 E1：matrix-free Hψ，目标单次 <1 ms（100 Ry）。

## 复现

```bash
export PYTHONPATH=src:tests:.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export ADDFT_NATIVE_XC=1 ADDFT_NATIVE_EIGH=1
python -m pytest tests/test_native_pack.py -q
```

详细对照见 `docs/reports/ABACUS_COMPARISON.md`。
