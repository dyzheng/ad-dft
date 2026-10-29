# Performance dashboard

Reproducible comparison of SCF/back-end variants.

```bash
# quick: 20/40 Ry, all schemes
python bench/dashboard.py --cases 20,40

# include stress and 100 Ry (slow)
python bench/dashboard.py --cases 20,40,100 --stress

# regenerate HTML/Markdown from existing latest.json
python bench/dashboard.py --no-run

# live view
python -m http.server 8000 -d bench/results
# open http://localhost:8000/dashboard.html
```

Schemes:

- `device_broyden`: device SCF + native XC/eigh/pack
- `python_diis`: Python DIIS + native XC/eigh/pack
- `no_native_pack`: device SCF with JAX scatter/gather
- `no_native_eigh`: device SCF with XLA `jnp.linalg.eigh`
- `all_jax`: all-native disabled, Python DIIS

Scaling:

```bash
python bench/scaling.py --sizes 2,8,16 --ecutwfc 20
python bench/dashboard.py --no-run   # include scaling in the dashboard
```

Outputs: `bench/results/latest.json`, `dashboard.md`, `dashboard.html`,
`history.jsonl`, `scaling-*.json/md` (ignored by git).


## Self-evolution

```bash
python bench/evolve.py --generations 3 --population 6 --elite 2 --seed 3
```

Searches Davidson `max_sub/cold/warm/precond/alpha` and Broyden `hi/lo` in
fresh processes, warm-times SCF, and writes `bench/results/evolution.json/md`.
Davidson-related env knobs also include `ADDFT_DAVID_EIG_TOL` (default 1e-3).
