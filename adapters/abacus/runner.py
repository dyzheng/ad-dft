"""运行 ABACUS 并解析 running_scf.log（固定二进制提交 c17cbac9b）。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

ABACUS = Path("/root/abacus-develop/build/abacus_basic_para")
HA_TO_EV = 27.211386245988


def run_case(case_dir, suffix="r3", timeout=600):
    case_dir = Path(case_dir)
    cmd = ([ "mpirun", "-np", "1", str(ABACUS)]
           if shutil.which("mpirun") else [str(ABACUS)])
    subprocess.run(cmd, cwd=case_dir, check=True, timeout=timeout)
    return parse_running_scf(case_dir / f"OUT.{suffix}" / "running_scf.log")


def _scan_block(text, header):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header in line:
            rows = []
            for ln in lines[i + 1:i + 40]:
                s = ln.strip()
                if not s:
                    if rows:
                        break
                    continue
                parts = s.split()
                try:
                    vals = [float(x) for x in parts[-3:]]
                except ValueError:
                    continue
                if len(vals) == 3:
                    rows.append(vals)
                if "TOTAL-" in s and header not in s:
                    break
            return np.asarray(rows)
    raise ValueError(f"找不到 {header}")


def parse_running_scf(path):
    text = Path(path).read_text(errors="ignore")
    # FINAL_ETOT_IS 的单位是 eV
    etot_ev = float(re.search(r"FINAL_ETOT_IS\s+([-\d.eE+]+)", text).group(1))
    forces = _scan_block(text, "TOTAL-FORCE (eV/Angstrom)")
    stress = _scan_block(text, "TOTAL-STRESS (kbar)")[:3]
    return dict(etot_ev=etot_ev, etot_ha=etot_ev / HA_TO_EV,
                forces=forces, stress=stress)
