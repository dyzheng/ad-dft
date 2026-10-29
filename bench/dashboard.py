"""Performance dashboard for ad-dft.

Runs the same case under several backends/drivers in fresh processes, then
writes bench/results/{latest.json,dashboard.md,dashboard.html}.

Usage:
    python bench/dashboard.py --cases 20,40
    python bench/dashboard.py --cases 20,40,100 --stress
    python bench/dashboard.py --no-run          # regenerate from latest.json
    python bench/dashboard.py --cases 20,40 --watch 60

For a live browser view:
    python -m http.server 8000 -d bench/results
    # open http://localhost:8000/dashboard.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "bench" / "results"
WORKER = ROOT / "bench" / "dashboard_worker.py"
CASE_SRC = ROOT / "tests" / "regress" / "si-gamma"

CASES = {"20": 20.0, "40": 40.0, "100": 100.0}

SCHEMES = {
    "device_broyden": dict(driver="auto", xc="1", eigh="1", pack="1"),
    "python_diis": dict(driver="python", xc="1", eigh="1", pack="1"),
    "no_native_pack": dict(driver="auto", xc="1", eigh="1", pack="0"),
    "no_native_eigh": dict(driver="auto", xc="1", eigh="0", pack="1"),
    "all_jax": dict(driver="python", xc="0", eigh="0", pack="0"),
    "davidson": dict(driver="auto", xc="1", eigh="1", pack="1",
                     eigh_solver="davidson"),
}


def make_case(ecutwfc: float) -> tuple[Path, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix=f"addft-dash-{int(ecutwfc)}-")
    dst = Path(tmp.name)
    for name in ("INPUT", "KPT", "STRU"):
        shutil.copy2(CASE_SRC / name, dst / name)
    lines = []
    for ln in (dst / "INPUT").read_text().splitlines():
        if ln.strip().startswith("suffix"):
            ln = "suffix            dash"
        elif ln.strip().startswith("ecutwfc"):
            ln = f"ecutwfc           {ecutwfc:g}"
        lines.append(ln)
    (dst / "INPUT").write_text("\n".join(lines) + "\n")
    return dst, tmp


def load_evolution() -> list:
    f = RESULT_DIR / "evolution.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except Exception:
        return []


def load_scaling() -> list:
    rows = []
    for f in sorted(RESULT_DIR.glob("scaling-*.json")):
        try:
            rows.extend(json.loads(f.read_text()))
        except Exception:
            pass
    return rows


def run_worker(case_dir: Path, scheme: str, mode: str, repeats: int,
               stress: bool) -> dict:
    cfg = SCHEMES[scheme]
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}:{ROOT}",
        "ADDFT_NATIVE_XC": cfg["xc"],
        "ADDFT_NATIVE_EIGH": cfg["eigh"],
        "ADDFT_NATIVE_PACK": cfg["pack"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
    })
    cmd = [sys.executable, str(WORKER), "--case", str(case_dir),
           "--scheme", scheme, "--mode", mode,
           "--driver", cfg["driver"], "--repeats", str(repeats),
           "--eigh-solver", cfg.get("eigh_solver", "auto")]
    if stress:
        cmd.append("--stress")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return {"mode": mode, "scheme": scheme, "error": p.stderr[-2000:]}
    line = [x for x in p.stdout.strip().splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def collect(cases: list[str], schemes: list[str], repeats: int,
            stress: bool) -> dict:
    out = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "cases": [],
        "scf": [],
        "micro": [],
    }
    try:
        import jax
        out["jax"] = jax.__version__
        out["devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    for key in cases:
        ecut = CASES[key]
        case_dir, tmp = make_case(ecut)
        out["cases"].append({"name": f"{int(ecut)} Ry", "ecutwfc": ecut,
                             "path": str(case_dir)})
        try:
            for scheme in schemes:
                print(f"[dash] {ecut:g} Ry / {scheme} / scf", flush=True)
                r = run_worker(case_dir, scheme, "scf", repeats, stress)
                r["case"] = f"{int(ecut)} Ry"
                out["scf"].append(r)
            print(f"[dash] {ecut:g} Ry / micro", flush=True)
            r = run_worker(case_dir, "device_broyden", "micro",
                           max(1, repeats), False)
            r["case"] = f"{int(ecut)} Ry"
            out["micro"].append(r)
        finally:
            tmp.cleanup()
    return out


def _fmt(x, unit=""):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:.3g}{unit}"


def write_markdown(data: dict) -> str:
    lines = [f"# ad-dft performance dashboard", "",
             f"Updated: `{data.get('timestamp')}`  ",
             f"JAX: `{data.get('jax','?')}`  ",
             f"Devices: `{', '.join(data.get('devices',[]))}`", "",
             "## SCF", "",
             "| case | scheme | nG | nfft | iters | SCF | E+F | stress | residual |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data.get("scf", []):
        if "error" in r:
            lines.append(f"| {r.get('case')} | {r.get('scheme')} | — | — | — | ERROR | — | — | — |")
            continue
        lines.append(
            f"| {r['case']} | {r['scheme']} | {r['nG']} | {r['nfft']} | "
            f"{r['n_iter']} | {_fmt(r.get('t_scf'),' s')} | "
            f"{_fmt(r.get('t_ef'),' s')} | {_fmt(r.get('t_stress'),' s')} | "
            f"{_fmt(r.get('residual'))} |")
    lines += ["", "## Scaling", "",
              "| atoms | nG | nfft | iters | ABACUS 1-core | ad-dft SCF | ratio |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += ["", "## Evolution", "",
              "| gen | max_sub | cold | warm | precond | hi | lo | SCF | iters | residual | score |",
              "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for h in data.get("evolution", []):
        c, r = h.get("candidate", {}), h.get("result", {})
        if "error" in r:
            continue
        lines.append(f"| {h.get('generation')} | {c.get('max_sub')} | {c.get('cold')} | "
                     f"{c.get('warm')} | {c.get('precond')} | {c.get('hi')} | {c.get('lo')} | "
                     f"{_fmt(r.get('t_scf'),' s')} | {r.get('n_iter')} | "
                     f"{_fmt(r.get('residual'))} | {_fmt(h.get('score'))} |")
    lines += ["", "## Scaling", "",
              "| atoms | nG | nfft | iters | ABACUS 1-core | ad-dft SCF | ratio |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in data.get("scaling", []):
        ab, adt = r.get("t_abacus"), r.get("t_addft")
        ratio = (adt / ab) if (ab and adt) else None
        lines.append(f"| {r.get('n_atoms')} | {r.get('nG') or '—'} | "
                     f"{r.get('nfft') or '—'} | {r.get('n_iter') or '—'} | "
                     f"{_fmt(ab,' s')} | {_fmt(adt,' s')} | {_fmt(ratio,'×')} |")
    lines += ["", "## Microbench", "",
              "| case | nG | nfft | Hψ FP64 | Hψ FP32 | eigh native | eigh dense |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in data.get("micro", []):
        if "error" in r:
            continue
        lines.append(
            f"| {r['case']} | {r['nG']} | {r['nfft']} | "
            f"{_fmt(r.get('hpsi_fp64'),' ms')} | {_fmt(r.get('hpsi_fp32'),' ms')} | "
            f"{_fmt(r.get('eigh_native'),' s')} | {_fmt(r.get('eigh_dense'),' s')} |")
    return "\n".join(lines) + "\n"


def write_html(data: dict) -> str:
    blob = json.dumps(data, ensure_ascii=False)
    return """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ad-dft performance dashboard</title>
<style>
:root{--bg:#0b1020;--panel:#141b31;--text:#e7ecff;--muted:#93a0c4;--accent:#5dd3ff;--good:#56e39f;--bad:#ff7b8a}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:22px 28px;border-bottom:1px solid #26304d;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:20px;margin:0} .muted{color:var(--muted)} main{padding:22px 28px;max-width:1400px;margin:auto}
section{margin-bottom:26px} h2{font-size:15px;color:var(--accent);margin:0 0 10px}
table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:10px;overflow:hidden}
th,td{padding:9px 10px;border-bottom:1px solid #232d4a;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--muted);font-weight:600;font-size:12px}
tr.best td:first-child{border-left:3px solid var(--good)} td.best{color:var(--good);font-weight:700}
.bar{height:7px;background:#25304f;border-radius:4px;min-width:80px}.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),#8a7dff);border-radius:4px}
.err{color:var(--bad)} .pill{display:inline-block;padding:2px 7px;border-radius:99px;background:#223055;color:var(--muted);font-size:12px}
#updated{font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>ad-dft performance dashboard</h1>
<span class="muted">updated <span id="updated"></span></span>
<span class="pill" id="devices"></span>
<span class="muted">auto-refresh 5s · serve with <code>python -m http.server 8000 -d bench/results</code></span>
</header><main>
<section><h2>SCF variants</h2><div id="scf"></div></section>
<section><h2>Evolution</h2><div id="evolution"></div></section>
<section><h2>Scaling</h2><div id="scaling"></div></section>
<section><h2>Microbench</h2><div id="micro"></div></section>
<section><h2>Notes</h2><p class="muted">Times are warm medians. SCF is the primary comparison; E+F/stress include one SCF each. 100 Ry is dominated by dense eigh.</p></section>
</main>
<script>
const EMBEDDED = __DATA__;
let DATA = EMBEDDED;
const fmt=(x,u="")=>x==null?"—":(typeof x==="string"?x:x.toPrecision(3)+u);
function table(rows, cols, bestKey){
  let h='<table><thead><tr>'+cols.map(c=>`<th>${c[0]}</th>`).join('')+'</tr></thead><tbody>';
  const valid=rows.filter(r=>!r.error);
  const min=bestKey?Math.min(...valid.map(r=>r[bestKey]).filter(x=>x!=null)):null;
  rows.forEach(r=>{
    if(r.error){h+=`<tr><td>${r.case||""}</td><td>${r.scheme}</td><td colspan="${cols.length-2}" class="err">ERROR</td></tr>`;return;}
    const best=bestKey&&r[bestKey]===min;
    h+=`<tr>${cols.map(c=>`<td class="${c[1]&&best?'best':''}">${c[2](r)}</td>`).join('')}</tr>`;
  });
  return h+'</tbody></table>';
}
function renderEvolution(){
  const rows=DATA.evolution||[];
  if(!rows.length){document.getElementById('evolution').innerHTML='<p class="muted">No evolution data. Run <code>python bench/evolve.py</code>.</p>';return;}
  const cols=[['gen',0,h=>h.generation],['max_sub',0,h=>h.candidate.max_sub],['cold',0,h=>h.candidate.cold],['warm',0,h=>h.candidate.warm],['precond',0,h=>h.candidate.precond],['hi',0,h=>h.candidate.hi],['lo',0,h=>h.candidate.lo],['SCF',0,h=>h.result.error?'ERROR':fmt(h.result.t_scf,' s')],['iters',0,h=>h.result.n_iter||'—'],['residual',0,h=>fmt(h.result.residual)],['score',0,h=>fmt(h.score)]];
  document.getElementById('evolution').innerHTML=table(rows.map(h=>({...h,error:h.result.error})),cols,'score');
}
function renderScaling(){
  const rows=DATA.scaling||[];
  if(!rows.length){document.getElementById('scaling').innerHTML='<p class="muted">No scaling data. Run <code>python bench/scaling.py --sizes 2,8,16</code>.</p>';return;}
  const cols=[['atoms',0,r=>r.n_atoms],['nG',0,r=>r.nG||'—'],['nfft',0,r=>r.nfft||'—'],
    ['iters',0,r=>r.n_iter||'—'],['ABACUS',0,r=>fmt(r.t_abacus,' s')],
    ['ad-dft',1,r=>fmt(r.t_addft,' s')],['ratio',0,r=>r.t_addft&&r.t_abacus?fmt(r.t_addft/r.t_abacus,'×'):'—']];
  document.getElementById('scaling').innerHTML=table(rows,cols,'t_addft');
}
function render(){
  document.getElementById('updated').textContent=DATA.timestamp||'—';
  document.getElementById('devices').textContent=(DATA.devices||[]).join(' · ');
  const scf=DATA.scf||[];
  const cols=[
    ['case',0,r=>r.case||''],['scheme',0,r=>r.scheme],
    ['nG',0,r=>r.nG],['nfft',0,r=>r.nfft],['iters',0,r=>r.n_iter],
    ['SCF',1,r=>fmt(r.t_scf,' s')],['E+F',1,r=>fmt(r.t_ef,' s')],
    ['stress',1,r=>fmt(r.t_stress,' s')],['residual',0,r=>fmt(r.residual)],
    ['SCF bar',0,r=>`<div class="bar"><i style="width:${Math.max(2,100*(r.t_scf||0)/Math.max(...scf.map(x=>x.t_scf||0),1e-9))}%"></i></div>`]
  ];
  document.getElementById('scf').innerHTML=table(scf,cols,'t_scf');
  renderEvolution();
  renderScaling();
  const micro=DATA.micro||[];
  const mcols=[
    ['case',0,r=>r.case||''],['nG',0,r=>r.nG],['nfft',0,r=>r.nfft],
    ['Hψ FP64',1,r=>fmt(r.hpsi_fp64,' ms')],['Hψ FP32',1,r=>fmt(r.hpsi_fp32,' ms')],
    ['eigh native',1,r=>fmt(r.eigh_native,' s')],['eigh dense',1,r=>fmt(r.eigh_dense,' s')]
  ];
  document.getElementById('micro').innerHTML=table(micro,mcols,null);
}
async function refresh(){
  try{const r=await fetch('latest.json?t='+Date.now()); if(r.ok){DATA=await r.json();}}catch(e){}
  render();
}
render(); setInterval(refresh,5000);
</script></body></html>
""".replace("__DATA__", blob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="20,40")
    p.add_argument("--schemes", default=",".join(SCHEMES))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--watch", type=int, default=0,
                   help="rerun every N seconds (0 = once)")
    args = p.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    latest = RESULT_DIR / "latest.json"
    if args.no_run and latest.exists():
        data = json.loads(latest.read_text())
        data["scaling"] = load_scaling()
        data["evolution"] = load_evolution()
    else:
        cases = [x.strip() for x in args.cases.split(",") if x.strip()]
        schemes = [x.strip() for x in args.schemes.split(",") if x.strip()]
        bad = [x for x in cases if x not in CASES] + [x for x in schemes if x not in SCHEMES]
        if bad:
            raise SystemExit(f"unknown cases/schemes: {bad}")
        data = collect(cases, schemes, args.repeats, args.stress)
        data["scaling"] = load_scaling()
        data["evolution"] = load_evolution()
        latest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        with (RESULT_DIR / "history.jsonl").open("a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    (RESULT_DIR / "dashboard.md").write_text(write_markdown(data))
    (RESULT_DIR / "dashboard.html").write_text(write_html(data), encoding="utf-8")
    print(f"[dash] wrote {RESULT_DIR/'dashboard.html'}")
    if args.watch:
        while True:
            time.sleep(args.watch)
            data = collect([x.strip() for x in args.cases.split(",") if x.strip()],
                           [x.strip() for x in args.schemes.split(",") if x.strip()],
                           args.repeats, args.stress)
            latest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            with (RESULT_DIR / "history.jsonl").open("a") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            (RESULT_DIR / "dashboard.md").write_text(write_markdown(data))
            (RESULT_DIR / "dashboard.html").write_text(write_html(data), encoding="utf-8")


if __name__ == "__main__":
    main()
