"""UPF v1/v2 文本/XML 解析（只读，不做单位转换）。

`read_upf` 返回的 `dion`、`vloc_at`、`betar` 仍是 UPF 原始的 Ry 制；
Ry→Ha 的 `/2` 只在 `psp.from_upf` / adapters 里出现一次。
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")


def _floats(text: str) -> np.ndarray:
    if text is None:
        return np.zeros((0,))
    vals = [float(x.replace("D", "E").replace("d", "e"))
            for x in _NUM_RE.findall(text)]
    return np.asarray(vals, dtype=float)


def _nums(el) -> np.ndarray:
    if el is None:
        return _floats(None)
    return _floats(el if isinstance(el, str) else el.text)


def _block(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    return m.group(1) if m else None


def _read_v2(root):
    h = root.find("PP_HEADER")
    mesh_decl = int(root.find("PP_MESH/PP_R").attrib["size"])
    mesh = mesh_decl - (1 if mesh_decl % 2 == 0 else 0)  # ABACUS 偶数网格减 1
    r = _nums(root.find("PP_MESH/PP_R"))[:mesh]
    rab = _nums(root.find("PP_MESH/PP_RAB"))[:mesh]
    vloc = _nums(root.find("PP_LOCAL"))[:mesh]  # Ry
    nbeta = int(h.attrib["number_of_proj"])
    lll, betar = [], []
    for ib in range(1, nbeta + 1):
        el = root.find(f"PP_NONLOCAL/PP_BETA.{ib}")
        lll.append(int(el.attrib["angular_momentum"]))
        betar.append(_nums(el)[:mesh])
    dij = _nums(root.find("PP_NONLOCAL/PP_DIJ"))[:nbeta * nbeta].reshape(nbeta, nbeta)
    nlcc = h.attrib.get("core_correction", "F") in ("T", "true", "1")
    rho_atc = (_nums(root.find("PP_NLCC"))[:mesh]
               if nlcc else np.zeros(mesh))
    return dict(zval=float(h.attrib["z_valence"]), mesh=mesh, r=r, rab=rab,
                vloc_at=vloc, nbeta=nbeta, lll=np.asarray(lll, dtype=int),
                betar=np.asarray(betar), dion=dij, nlcc=nlcc,
                rho_atc=rho_atc)


def _read_v1_text(text: str):
    header = _block(text, "PP_HEADER")
    if header is None:
        raise ValueError("UPF v1 缺少 PP_HEADER")
    zval = float(re.search(r"([-\d.]+)\s+Z valence", header).group(1))
    mesh = int(re.search(r"(\d+)\s+Number of points in mesh", header).group(1))
    mesh = mesh - (1 if mesh % 2 == 0 else 0)
    nchi, nbeta = map(int, re.search(
        r"(\d+)\s+(\d+)\s+Number of Wavefunctions, Number of Projectors",
        header).groups())
    r = _nums(_block(text, "PP_R"))[:mesh]
    rab = _nums(_block(text, "PP_RAB"))[:mesh]
    vloc = _nums(_block(text, "PP_LOCAL"))[:mesh]
    lll, betar = [], []
    for blk in re.findall(r"<PP_BETA>(.*?)</PP_BETA>", text, flags=re.S):
        vals = _floats(blk)
        idx, l, kb = int(vals[0]), int(vals[1]), int(vals[2])
        lll.append(l)
        b = vals[3:3 + min(kb, mesh)]
        betar.append(np.pad(b, (0, mesh - len(b))))
    dvals = _floats(_block(text, "PP_DIJ"))
    nd = int(dvals[0])
    d = np.zeros((nbeta, nbeta))
    for k in range(nd):
        i, j, v = int(dvals[1 + 3 * k]), int(dvals[2 + 3 * k]), dvals[3 + 3 * k]
        d[i - 1, j - 1] = d[j - 1, i - 1] = v  # Ry
    nlcc_text = _block(text, "PP_NLCC")
    return dict(zval=zval, mesh=mesh, r=r, rab=rab, vloc_at=vloc,
                nbeta=nbeta, lll=np.asarray(lll, dtype=int),
                betar=np.asarray(betar), dion=d,
                nlcc=nlcc_text is not None,
                rho_atc=(_nums(nlcc_text)[:mesh] if nlcc_text is not None
                         else np.zeros(mesh)))


def read_upf(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if "<UPF" in text:
        root = ET.fromstring(text)
        return _read_v2(root)
    return _read_v1_text(text)
