from pathlib import Path

import numpy as np
import pytest

from adapters.abacus.io import parse_input, parse_kpt, parse_stru

SI_STRU = Path("/root/abacus-develop/examples/02_scf/01_pw_Si2/STRU")


def test_si_cartesian_positions_use_lat0():
    out = parse_stru(SI_STRU)
    # 2 个 Si；Cartesian 0.25 的单位是 lat0=10.2 Bohr
    assert out["frac"].shape == (2, 3)
    assert out["species"] == ["Si", "Si"]
    # 位置差 = (0.25,0.25,0.25)*10.2 Bohr 投到分数坐标
    d = (out["frac"][1] - out["frac"][0]) @ out["lattice"]
    assert abs(float(d[0] - 2.55)) < 1e-12


def test_parse_regression_inputs():
    inp = parse_input("tests/regress/si-gamma/INPUT")
    assert inp["basis.ecut_ha"] == pytest.approx(10.0)
    assert inp["nband"] == 6
    assert inp["occupation.method"] == "fixed"
    kpt = parse_kpt("tests/regress/si-gamma/KPT")
    assert kpt["mode"] == "gamma"


def test_setup_from_case_uses_real_upf():
    import addft as ad

    s = ad.setup_from_case("tests/regress/si-gamma")
    assert s.static.psp.kind == 1
    assert s.static.nelec == pytest.approx(8.0)
    assert s.static.spec_idx.shape == (2,)
    assert s.static.basis.nG > 0


def test_runner_parses_log(tmp_path):
    from adapters.abacus.runner import parse_running_scf

    log = tmp_path / "running_scf.log"
    log.write_text(
        "FINAL_ETOT_IS -123.456 eV\n"
        "TOTAL-FORCE (eV/Angstrom)\n"
        "-------------------------------------\n"
        " Si1  0.0 0.0 -0.003\n"
        " Si2  0.0 0.0  0.003\n"
        "\nTOTAL-STRESS (kbar)\n"
        "-------------------------------------\n"
        " 1.0 2.0 3.0\n 4.0 5.0 6.0\n 7.0 8.0 9.0\n")
    out = parse_running_scf(log)
    assert out["etot_ha"] == pytest.approx(-123.456 / 27.211386245988)
    assert np.allclose(out["forces"], [[0, 0, -0.003], [0, 0, 0.003]])
    assert np.allclose(out["stress"][2], [7, 8, 9])


def test_multi_species_is_explicitly_rejected(tmp_path):
    import addft as ad

    for f in ("INPUT", "KPT"):
        (tmp_path / f).write_text((Path("tests/regress/si-gamma") / f).read_text())
    (tmp_path / "STRU").write_text(
        "ATOMIC_SPECIES\nSi 1.0 Si.pz-vbc.UPF\nO 16.0 Si.pz-vbc.UPF\n\n"
        "LATTICE_CONSTANT\n10.2\n\nLATTICE_VECTORS\n0.0 0.5 0.5\n"
        "0.5 0.0 0.5\n0.5 0.5 0.0\n\nATOMIC_POSITIONS\nDirect\n"
        "Si\n0.0\n1\n0 0 0\nO\n0.0\n1\n0.25 0.25 0.25\n")
    with pytest.raises(NotImplementedError):
        ad.setup_from_case(tmp_path)
