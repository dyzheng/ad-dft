"""伴随成本/显存基准：SCF vs 一次能量+梯度。

用法：
    python bench/adjoint.py 729
    python bench/adjoint.py 3375
    python bench/adjoint.py all     # 分别在子进程测，比较峰值 RSS
"""

import json
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

import addft as ad

CASES = {
    729: dict(ecut=2.0, density_factor=4.0),
    3375: dict(ecut=3.5, density_factor=4.0),
}


def run(nfft_target, repeats=3):
    kw = CASES[nfft_target]
    s = ad.setup(lattice=jnp.eye(3) * 6.0,
                 frac=jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]]),
                 nband=2, **kw)
    e_fn = ad.make_energy(s.static, mode="adjoint")

    t_scfs, t_grads = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        st = ad.scf(s)
        jax.block_until_ready(st.rho_r)
        t_scfs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        e = e_fn(s.geo)
        g = jax.grad(e_fn)(s.geo)
        jax.block_until_ready((e, g))
        t_grads.append(time.perf_counter() - t0)

    med = lambda xs: sorted(xs)[len(xs) // 2]
    t_scf, t_grad = med(t_scfs), med(t_grads)
    return dict(nfft=s.static.basis.nfft, t_scf=t_scf, t_grad=t_grad,
                ratio=t_grad / t_scf,
                rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "all":
        out = {}
        for n in CASES:
            r = subprocess.run([sys.executable, __file__, str(n)],
                               capture_output=True, text=True, check=True)
            out[n] = json.loads(r.stdout.strip().splitlines()[-1])
        for n, r in out.items():
            print(f"nfft={r['nfft']}: scf={r['t_scf']:.2f}s "
                  f"grad={r['t_grad']:.2f}s ratio={r['ratio']:.2f} "
                  f"rss={r['rss_kb']/1024:.0f}MB")
        lo, hi = out[min(CASES)], out[max(CASES)]
        print(f"RSS ratio (3375/729) = {hi['rss_kb']/lo['rss_kb']:.2f}")
    else:
        print(json.dumps(run(int(target))))
