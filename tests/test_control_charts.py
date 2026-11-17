import numpy as np
import pytest
from statistics_engine.control_charts import (xbar_r, xbar_s, i_mr, p_chart, np_chart,
                                              c_chart, u_chart, nelson_rules)

SUB = [[2, 4, 6, 4, 4], [3, 3, 5, 3, 6], [5, 5, 5, 5, 5]]
# medias 4.0, 4.0, 5.0 -> Xbarbar = 13/3; rangos 4, 3, 0 -> Rbar = 7/3


def test_xbar_r_calculado_a_mano():
    x, r = xbar_r(SUB)
    assert x.centro[0] == pytest.approx(13 / 3)
    assert r.centro[0] == pytest.approx(7 / 3)
    assert x.lcs[0] == pytest.approx(13 / 3 + 0.577 * 7 / 3, abs=0.01)
    assert x.lci[0] == pytest.approx(13 / 3 - 0.577 * 7 / 3, abs=0.01)
    assert r.lcs[0] == pytest.approx(2.114 * 7 / 3, abs=0.01)
    assert r.lci[0] == pytest.approx(0.0)


def test_xbar_s_usa_c4():
    x, s = xbar_s(SUB)
    sds = np.array(SUB).std(axis=1, ddof=1)
    assert s.centro[0] == pytest.approx(sds.mean())
    assert x.lcs[0] == pytest.approx(13 / 3 + 1.427 * sds.mean(), abs=0.01)


def test_i_mr_calculado_a_mano():
    i, mr = i_mr([1, 3, 2, 5])
    assert mr.centro[0] == pytest.approx(2.0)            # MR = 2,1,3 -> MRbar = 2
    assert i.sigma == pytest.approx(2 / 1.128, abs=1e-6)
    assert i.centro[0] == pytest.approx(2.75)
    assert i.lcs[0] == pytest.approx(2.75 + 3 * 2 / 1.128, abs=1e-6)
    assert len(mr.puntos) == 3


def test_p_chart():
    r = p_chart([2, 4], 100)
    se = np.sqrt(0.03 * 0.97 / 100)
    assert r.centro[0] == pytest.approx(0.03)
    assert r.lcs[0] == pytest.approx(0.03 + 3 * se)
    assert r.lci[0] == pytest.approx(0.0)                 # truncado en cero


def test_p_chart_limites_variables():
    r = p_chart([5, 5], [100, 400])
    assert r.lcs[0] > r.lcs[1]                            # n mayor -> limites mas estrechos


def test_np_c_u():
    n = np_chart([2, 4], 100)
    assert n.centro[0] == pytest.approx(3.0)
    assert n.lcs[0] == pytest.approx(3 + 3 * np.sqrt(3 * 0.97))
    c = c_chart([2, 4, 6])
    assert c.centro[0] == pytest.approx(4.0)
    assert c.lcs[0] == pytest.approx(10.0) and c.lci[0] == pytest.approx(0.0)
    u = u_chart([4, 6], [2, 3])
    assert u.centro[0] == pytest.approx(2.0)
    assert u.lcs[0] == pytest.approx(2 + 3 * np.sqrt(2 / 2))
    assert u.lcs[1] == pytest.approx(2 + 3 * np.sqrt(2 / 3))


def test_pbar_es_ponderado_no_promedio_de_proporciones():
    """pbar = total/total, NO la media de las p_i. Con n desigual difieren."""
    r = p_chart([1, 10], [10, 100])
    assert r.centro[0] == pytest.approx(11 / 110)
    assert r.centro[0] != pytest.approx(np.mean([1 / 10, 10 / 100]) + 1e-9, abs=0)


def test_proceso_estable_no_genera_senales():
    rng = np.random.default_rng(123)
    x, r = xbar_r(rng.normal(0, 1, (25, 5)))
    assert x.en_control and r.en_control


def test_detecta_corrimiento():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 1, (30, 5))
    m[20:] += 3.0
    x, _ = xbar_r(m)
    assert 1 in x.violaciones
    assert any(i >= 20 for i in x.fuera_de_control)


# ----------------------------- reglas de Nelson -----------------------------
def test_regla_1():
    v = nelson_rules([0, 0, 4.0, 0], np.zeros(4), 1.0)
    assert v[1] == [2]


def test_regla_2_nueve_del_mismo_lado():
    v = nelson_rules([0.5] * 9, np.zeros(9), 1.0)
    assert 2 in v and len(v[2]) == 9


def test_regla_3_seis_en_tendencia():
    v = nelson_rules([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], np.zeros(7), 10.0)
    assert 3 in v


def test_regla_4_catorce_alternando():
    x = [0.1 * (-1) ** i for i in range(15)]
    v = nelson_rules(x, np.zeros(15), 10.0)
    assert 4 in v


def test_regla_5_dos_de_tres_mas_alla_de_2sigma():
    v = nelson_rules([2.5, 0.0, 2.6], np.zeros(3), 1.0)
    assert 5 in v


def test_regla_6_cuatro_de_cinco_mas_alla_de_1sigma():
    v = nelson_rules([1.2, 1.3, 0.0, 1.4, 1.5], np.zeros(5), 1.0)
    assert 6 in v


def test_regla_7_estratificacion():
    v = nelson_rules([0.1] * 15, np.zeros(15), 1.0)
    assert 7 in v


def test_regla_8_mezcla():
    x = [1.5, -1.5] * 4
    v = nelson_rules(x, np.zeros(8), 1.0)
    assert 8 in v


def test_cartas_de_atributos_no_usan_reglas_de_zona():
    """R, S, p, np, c y u tienen estadísticos asimétricos: solo reglas 1-4."""
    c = c_chart(list(np.random.default_rng(0).poisson(5, 40)))
    assert all(r <= 4 for r in c.violaciones)
    assert c.sigma is None


def test_to_frame():
    x, _ = xbar_r(SUB)
    df = x.to_frame()
    assert {"subgrupo", "valor", "LC", "LCI", "LCS", "senal"} <= set(df.columns)
    assert len(df) == 3
