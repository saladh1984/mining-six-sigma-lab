import numpy as np
import pytest
from statistics_engine.capability import (capability, sigma_within_individuals,
                                          sigma_within_from_subgroups, intervalo_confianza_cpk)


def test_cp_con_sigma_conocida():
    """Proceso centrado con sigma=1 y ventana de 12 -> Cp = 2."""
    rng = np.random.default_rng(0)
    x = rng.normal(100, 1, 5000)
    r = capability(x, lie=94, lse=106)
    assert r.cp == pytest.approx(2.0, rel=0.05)
    assert r.pp == pytest.approx(2.0, rel=0.05)
    assert r.cpk == pytest.approx(r.cp, rel=0.05)   # centrado -> Cpk ~ Cp


def test_descentrado_reduce_cpk_pero_no_cp():
    rng = np.random.default_rng(1)
    x = rng.normal(103, 1, 5000)
    r = capability(x, lie=94, lse=106)
    assert r.cp == pytest.approx(2.0, rel=0.06)
    assert r.cpk < r.cp
    assert r.cpk == pytest.approx(1.0, rel=0.1)     # (106-103)/(3*1)


def test_identidad_cpk_min_cpu_cpl():
    rng = np.random.default_rng(2)
    r = capability(rng.normal(50, 2, 400), lie=40, lse=58)
    assert r.cpk == pytest.approx(min(r.cpu, r.cpl))
    assert r.ppk == pytest.approx(min(r.ppu, r.ppl))


def test_unilateral():
    rng = np.random.default_rng(3)
    r = capability(rng.normal(30, 2, 400), lse=38)
    assert r.cp is None and r.cpl is None
    assert r.cpk == pytest.approx(r.cpu)


def test_cp_mayor_que_pp_con_deriva_entre_subgrupos():
    """Si hay deriva entre subgrupos, sigma_total > sigma_dentro -> Cp > Pp."""
    rng = np.random.default_rng(4)
    sub = np.array([rng.normal(100 + 3 * (i % 5), 1, 5) for i in range(40)])
    r = capability(sub.ravel(), lie=88, lse=112, subgrupos=sub)
    assert r.sigma_overall > r.sigma_within
    assert r.cp > r.pp
    assert any("entre subgrupos" in a for a in r.advertencias)


def test_sigma_within_metodos():
    rng = np.random.default_rng(5)
    sub = rng.normal(0, 1, (200, 5))
    sr, _ = sigma_within_from_subgroups(sub, "R")
    ss, _ = sigma_within_from_subgroups(sub, "S")
    assert sr == pytest.approx(1.0, rel=0.08)
    assert ss == pytest.approx(1.0, rel=0.08)
    si, _ = sigma_within_individuals(rng.normal(0, 1, 3000))
    assert si == pytest.approx(1.0, rel=0.08)


def test_ppm_observado_es_conteo_real():
    x = [1, 2, 3, 4, 50]      # 1 de 5 fuera del LSE=10 -> 200000 ppm
    r = capability(x, lse=10)
    assert r.ppm_observado == pytest.approx(200000.0)


def test_errores():
    with pytest.raises(ValueError):
        capability([1, 2, 3])                       # sin limites
    with pytest.raises(ValueError):
        capability([1, 2, 3], lie=10, lse=5)        # limites invertidos
    with pytest.raises(ValueError):
        capability([5, 5, 5, 5], lie=1, lse=9)      # sin variacion


def test_ic_cpk():
    lo, hi = intervalo_confianza_cpk(1.33, 100)
    assert lo < 1.33 < hi
    lo2, hi2 = intervalo_confianza_cpk(1.33, 1000)
    assert (hi2 - lo2) < (hi - lo)                  # mas datos -> IC mas estrecho
