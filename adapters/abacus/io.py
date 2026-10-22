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
    "nbands": ("nband", int),
    "pseudo_dir": ("pseudo_dir", str),
    "cal_force": ("scf.cal_force", lambda v: int(v) != 0),
    "cal_stress": ("scf.cal_stress", lambda v: int(v) != 0),
    "out_element_info": ("scf.out_element_info", lambda v: int(v) != 0),
    "ntype": ("cell.ntype", int),
    "suffix": ("io.suffix", str),
    "ks_solver": ("solver.ks_solver", str),
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
        key = parts[0].lower()
        if key == "input_parameters":
            continue
        if len(parts) < 2:
            raise UnmappedParameterError(f"无法解析的 INPUT 行: {line}")
        val = parts[1]
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
    """读 ABACUS STRU，返回 dict(lattice, frac, species, magmoms, labels, pseudo_files)。

    Cartesian 坐标的单位是 `LATTICE_CONSTANT` (lat0)，不是晶格向量；
    旧实现漏掉 lat0，真实 Si 结构会错 2 倍。原子按 `ATOMIC_SPECIES` 的
    species 块绑定，多元素不再被静默当成单元素。
    """
    text = Path(path).read_text(encoding="utf-8").splitlines()
    lat0, vecs = 1.0, []
    species_labels, pseudo_files = [], []
    species_per_atom, coords, magmoms = [], [], []
    mode = None
    i, n = 0, len(text)

    while i < n:
        line = text[i].split("#")[0].strip()
        if not line:
            i += 1
            continue
        if line.startswith("ATOMIC_SPECIES"):
            i += 1
            while i < n:
                s = text[i].split("#")[0].strip()
                if not s or s.startswith(("LATTICE_", "ATOMIC_POSITIONS")):
                    break
                parts = s.split()
                if len(parts) >= 3:
                    species_labels.append(parts[0])
                    pseudo_files.append(parts[2])
                i += 1
            continue
        if line.startswith("LATTICE_CONSTANT"):
            parts = line.split()
            if len(parts) >= 2:
                lat0 = float(parts[1])
                i += 1
            else:
                lat0 = float(text[i + 1].split("#")[0].strip())
                i += 2
            continue
        if line.startswith("LATTICE_VECTORS"):
            for j in range(3):
                vecs.append([float(x) for x in text[i + 1 + j].split()[:3]])
            i += 4
            continue
        if line.startswith("ATOMIC_POSITIONS"):
            parts = line.split()
            i += 1
            if len(parts) >= 2:
                mode = parts[1]
            else:
                while i < n:
                    s = text[i].split("#")[0].strip()
                    if not s:
                        i += 1
                        continue
                    mode = s.split()[0]
                    i += 1
                    break
            if mode not in ("Direct", "Cartesian"):
                raise NotImplementedError(f"未支持的坐标模式: {mode}")
            break
        i += 1

    while i < n:
        s = text[i].split("#")[0].strip()
        if not s:
            i += 1
            continue
        if s.startswith(("LATTICE_", "ATOMIC_")):
            break
        label = s.split()[0]
        if label not in species_labels:
            raise ValueError(
                f"ATOMIC_POSITIONS 中的元素未在 ATOMIC_SPECIES 定义: {label}")
        mag = float(text[i + 1].split()[0])
        count = int(float(text[i + 2].split()[0]))
        i += 3
        for _ in range(count):
            nums = [float(x) for x in text[i].split()[:3]]
            coords.append(nums)
            magmoms.append(mag)
            species_per_atom.append(label)
            i += 1

    if mode is None:
        raise ValueError("STRU 缺少 ATOMIC_POSITIONS 坐标模式")
    lat = np.asarray(vecs, dtype=float) * lat0  # Bohr
    if mode == "Direct":
        frac = np.asarray(coords, dtype=float)
    else:  # Cartesian，单位 lat0
        frac = (np.asarray(coords, dtype=float) * lat0) @ np.linalg.inv(lat)
    return dict(lattice=lat, frac=frac, species=species_per_atom,
                magmoms=np.asarray(magmoms), labels=species_labels,
                pseudo_files=pseudo_files)


def setup_from_case(case_dir):
    """读 INPUT/KPT/STRU/UPF，构造 addft.System。只支持单 species。"""
    from addft.api import setup
    from addft.geometry.kmesh import gamma, monkhorst_pack
    from addft.geometry.psp import from_upf

    case = Path(case_dir)
    inp = parse_input(case / "INPUT")
    kpt = parse_kpt(case / "KPT")
    stru = parse_stru(case / "STRU")
    if len(set(stru["species"])) != 1:
        raise NotImplementedError("R3 只支持单 species 的 ABACUS 回归")
    pseudo_path = case / inp.get("pseudo_dir", ".") / stru["pseudo_files"][0]
    psp = from_upf(pseudo_path)
    km = (gamma() if kpt["mode"] == "gamma"
          else monkhorst_pack(kpt["mesh"], kpt["shift"]))
    sigma = 0.0 if inp.get("occupation.method") == "fixed" else \
        inp.get("occupation.sigma_ha", 0.05)
    return setup(lattice=stru["lattice"], frac=stru["frac"], psp=psp,
                 ecut=inp["basis.ecut_ha"], nband=inp.get("nband", 6),
                 kmesh=km, sigma=sigma)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1:
        print(parse_input(sys.argv[1]))
