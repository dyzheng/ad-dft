import numpy as np

from addft.geometry.upf import read_upf


def test_read_si_v1():
    raw = read_upf("/root/abacus-develop/tests/PP_ORB/Si.pz-vbc.UPF")
    assert raw["zval"] == 4.0
    assert raw["nbeta"] == 2
    assert list(raw["lll"]) == [0, 1]
    assert raw["r"].shape == (raw["mesh"],)
    assert raw["betar"].shape == (2, raw["mesh"])
    # v1 的 PP_DIJ 是 Ry；raw 层不转单位
    assert abs(raw["dion"][0, 0] - 1.52388501179) < 1e-12


def test_read_si_v2():
    raw = read_upf("/root/abacus-develop/tests/PP_ORB/Si_ONCV_PBE-1.0.upf")
    assert raw["zval"] == 4.0
    assert raw["nbeta"] == 4
    assert list(raw["lll"]) == [0, 0, 1, 1]
    assert raw["betar"].shape[1] == raw["mesh"]
    assert raw["r"].shape == (raw["mesh"],)
    assert raw["mesh"] == 601  # 602 → 奇数网格 601


def _parse_nonlocal(path):
    import re
    from pathlib import Path as _P
    import numpy as np

    text = _P(path).read_text()
    nbeta = int(re.search(r"<DIJ>\s*(\d+)", text).group(1))
    dij = np.array([float(x) for x in re.findall(
        r"^\s*\d+\s+\d+\s+([-\d.eE+]+)\s*$", text, flags=re.M)])
    dij = dij.reshape(nbeta, nbeta)
    betas = []
    for blk in re.findall(r"<PP_BETA>(.*?)</PP_BETA>", text, flags=re.S):
        lines = blk.strip().splitlines()
        l = int(lines[1].split()[0])
        cut = int(lines[2].split()[0])
        rows = np.array([[float(x) for x in ln.split()[:3]]
                         for ln in lines[3:3 + cut]])
        betas.append((l, rows))
    return nbeta, dij, betas


def test_against_abacus_element_info(tmp_path):
    """与 ABACUS out_element_info 逐元素对照（需要本地 ABACUS + MPI）。"""
    import os
    import shutil
    import subprocess

    import numpy as np
    import pytest

    if os.environ.get("ADDFT_RUN_ABACUS") != "1":
        pytest.skip("设置 ADDFT_RUN_ABACUS=1 才运行 ABACUS 对照")
    abacus = "/root/abacus-develop/build/abacus_basic_para"
    pseudo_dir = "/root/abacus-develop/tests/PP_ORB"
    (tmp_path / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix r3e\npseudo_dir %s\nntype 1\nnbands 6\n"
        "symmetry 0\necutwfc 20\nscf_thr 1e-7\nscf_nmax 30\nbasis_type pw\n"
        "ks_solver dav_subspace\nsmearing_method fixed\nout_element_info 1\n"
        % pseudo_dir)
    (tmp_path / "STRU").write_text(
        "ATOMIC_SPECIES\nSi 1.000 Si.pz-vbc.UPF\n\nLATTICE_CONSTANT\n10.2\n\n"
        "LATTICE_VECTORS\n0.0 0.5 0.5\n0.5 0.0 0.5\n0.5 0.5 0.0\n\n"
        "ATOMIC_POSITIONS\nCartesian\nSi\n0.0\n2\n"
        "0.00 0.00 0.00 0 0 0\n0.25 0.25 0.25 1 1 1\n")
    (tmp_path / "KPT").write_text("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")
    if shutil.which("mpirun"):
        cmd = ["mpirun", "-np", "1", abacus]
    else:
        cmd = [abacus]
    subprocess.run(cmd, cwd=tmp_path, check=True, timeout=900)

    from addft.geometry.psp import from_upf, vloc_q
    from addft.geometry.upf import read_upf

    raw = read_upf(f"{pseudo_dir}/Si.pz-vbc.UPF")
    psp = from_upf(f"{pseudo_dir}/Si.pz-vbc.UPF")
    nbeta, dij, betas = _parse_nonlocal(tmp_path / "OUT.r3e" / "Si" / "Si.NONLOCAL")
    assert nbeta == raw["nbeta"]
    assert np.allclose(dij, raw["dion"], atol=1e-12)
    for (l, rows), raw_l, raw_b in zip(betas, raw["lll"], raw["betar"]):
        cut = rows.shape[0]
        assert l == int(raw_l)
        assert np.allclose(rows[:, 0], raw["r"][:cut], atol=1e-12)
        assert np.allclose(rows[:, 1], raw_b[:cut], atol=1e-12)
        assert np.allclose(rows[:, 2], raw["rab"][:cut], atol=1e-12)

    v = np.loadtxt(tmp_path / "OUT.r3e" / "Si" / "v_loc_g.dat")
    q = np.sqrt(v[:, 0])
    omega = (10.2 ** 3) * 0.25  # fcc 原胞体积 = a^3/4
    import jax.numpy as jnp
    ours = np.asarray(vloc_q(psp, jnp.array([0]), jnp.asarray(q))) / omega / 2.0
    assert np.allclose(ours[:, 0], v[:, 1], rtol=1e-8, atol=1e-10)
