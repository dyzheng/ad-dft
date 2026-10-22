"""ABACUS 输入适配层（K16 骨架）：INPUT / KPT → 本项目的数据模型。

三条映射规则（spec/units.md）：
  1. 单位在边界转换一次，内部统一原子单位，不允许混用；
  2. 泛函用 libxc ID 而不是字符串（"PBE" 在不同库里不等价）；
  3. 任何未映射的参数必须**显式报错**，不允许静默忽略——
     静默忽略是"看起来跑了但结果不对"的主要来源。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# ABACUS 内部是 Ry，本项目的 Ha 桥只允许出现在这一层
RY_TO_HA = 0.5

#: ABACUS INPUT 参数 → 本项目字段（只映射 V0–V2 需要的子集）
PARAM_MAP = {
    "ecutwfc": ("basis.ecut_ha", lambda v: float(v) * RY_TO_HA),
    "scf_thr": ("scf.threshold", float),
    "scf_nmax": ("scf.max_iter", int),
    "smearing_method": ("occupation.method", str),
    "smearing_sigma": ("occupation.sigma_ha", lambda v: float(v) * RY_TO_HA),
    "mixing_type": ("scf.mixer", str),
    "mixing_beta": ("scf.beta", float),
    "nspin": ("spin.nspin", int),
    "symmetry": ("symmetry.enabled", lambda v: int(v) != 0),
    "dft_functional": ("xc.name", str),
    "basis_type": ("basis.type", str),
    "device": ("runtime.device", str),
    "calculation": ("task.type", str),
}

#: 明确不支持的参数：遇到就报错（不是忽略）
UNSUPPORTED = {
    "dft_plus_u": "DFT+U 排在 G4（K15 之后）",
    "orbital_dir": "NAO 基组排在 G5",
    "vdw_method": "vdW 修正不在范围内",
    "nspin": None,  # 支持 1/2，其它报错
}

XC_TO_LIBXC = {
    "lda": ["XC_LDA_X", "XC_LDA_C_PZ"],
    "pbe": ["XC_GGA_X_PBE", "XC_GGA_C_PBE"],
    "pz": ["XC_LDA_X", "XC_LDA_C_PZ"],
}


class UnmappedParameterError(KeyError):
    pass


def parse_input(path: str | Path) -> dict:
    """读 ABACUS INPUT（`key value` 逐行，忽略 # 注释）。"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        key, val = parts[0].lower(), parts[1]
        if key in UNSUPPORTED and UNSUPPORTED[key] is not None:
            raise UnmappedParameterError(f"{key}: {UNSUPPORTED[key]}")
        if key not in PARAM_MAP:
            raise UnmappedParameterError(
                f"未映射的 INPUT 参数 `{key}`。按 spec/units.md §3 规则 3，"
                f"静默忽略会导致'看起来跑了但结果不对'；请在 PARAM_MAP 里补映射。")
        field, conv = PARAM_MAP[key]
        out[field] = conv(val)
    return out


def parse_kpt(path: str | Path) -> dict:
    """读 ABACUS KPT。

    格式：
        0            # Gamma 或
        Gamma
        Monkhorst-Pack
        2 2 2 0 0 0
    """
    tokens = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            tokens.extend(line.split())
    if not tokens:
        raise ValueError("空 KPT 文件")
    if tokens[0] == "Gamma" or "Gamma" in tokens:
        return {"mode": "gamma"}
    idx = tokens.index("Monkhorst-Pack") if "Monkhorst-Pack" in tokens else 0
    nums = [int(float(x)) for x in tokens[idx + 1: idx + 7]]
    return {"mode": "mp", "mesh": tuple(nums[:3]), "shift": tuple(x / 2.0 for x in nums[3:6])}


def parse_stru(path: str | Path):
    """读 ABACUS STRU（LATTICE_CONSTANT / LATTICE_VECTORS / ATOMIC_POSITIONS）。

    返回 (lattice_bohr, frac, species, magmom)。
    """
    text = Path(path).read_text(encoding="utf-8").splitlines()
    lat0 = 1.0
    vecs, species, coords, magmoms = [], [], [], []
    i = 0
    n = len(text)
    while i < n:
        line = text[i].split("#")[0].strip()
        if line.startswith("LATTICE_CONSTANT"):
            lat0 = float(line.split()[1])
        elif line.startswith("LATTICE_VECTORS"):
            for j in range(3):
                vecs.append([float(x) for x in text[i + 1 + j].split()[:3]])
            i += 3
        elif line.startswith("ATOMIC_POSITIONS"):
            cart = "Cartesian" in text[i]
            i += 1
            while i < n:
                s = text[i].split("#")[0].strip()
                if not s:
                    i += 1
                    continue
                if re.match(r"^[A-Za-z]", s) and s.lower() not in ("direct", "cartesian"):
                    species.append(s)
                    i += 1
                    continue
                nums = s.split()
                if len(nums) >= 3 and re.match(r"^[-+.\d]", nums[0]):
                    xyz = [float(x) for x in nums[:3]]
                    magmoms.append(float(nums[3]) if len(nums) > 3 else 0.0)
                    coords.append(xyz)
                i += 1
            break
        i += 1
    lat = np.array(vecs) * lat0  # Bohr（ABACUS 的 lat0 以 Bohr 计）
    frac = np.array(coords)
    if len(coords) and max(abs(x) for x in coords[0]) > 2.0:  # 粗判 Cartesian
        frac = frac @ np.linalg.inv(lat)
    return lat, frac, species, np.array(magmoms)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1:
        print(parse_input(sys.argv[1]))
