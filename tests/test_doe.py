import numpy as np
import pandas as pd
import pytest
from statistics_engine.doe import (design_matrix, factorial_analysis,
                                   main_effect_plot_data, interaction_plot_data)


def test_matriz_de_diseno_2k():
    d = design_matrix({"A": (10, 20), "B": (1, 2)})
    assert len(d) == 4
    assert sorted(map(tuple, d[["A", "B"]].to_numpy())) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    assert set(d["A_real"]) == {10, 20}
    assert len(design_matrix({"A": (0, 1), "B": (0, 1), "C": (0, 1)}, replicas=2)) == 16


def test_columnas_ortogonales():
    d = design_matrix({"A": (0, 1), "B": (0, 1), "C": (0, 1)})
    M = d[["A", "B", "C"]].to_numpy()
    assert np.allclose(M.T @ M, 8 * np.eye(3))      # ortogonales y balanceadas


def test_efectos_calculados_a_mano():
    """2^2 sin replicas con respuestas 10, 20, 30, 50 en orden estándar.

    Orden: (-1,-1)=10, (+1,-1)=20, (-1,+1)=30, (+1,+1)=50
    Efecto A = [(20+50) - (10+30)] / 2 = 15
    Efecto B = [(30+50) - (10+20)] / 2 = 25
    Efecto AB = [(10+50) - (20+30)] / 2 = 5
    """
    d = design_matrix({"A": (0, 1), "B": (0, 1)})
    d["y"] = [10, 20, 30, 50]
    e = factorial_analysis(d, ["A", "B"], "y")["efectos"].set_index("termino")["efecto"]
    assert e["A"] == pytest.approx(15.0)
    assert e["B"] == pytest.approx(25.0)
    assert e["A:B"] == pytest.approx(5.0)


def test_coeficiente_es_la_mitad_del_efecto():
    d = design_matrix({"A": (0, 1), "B": (0, 1)})
    d["y"] = [10, 20, 30, 50]
    t = factorial_analysis(d, ["A", "B"], "y")["efectos"]
    assert np.allclose(t["coeficiente"], t["efecto"] / 2)


def test_media_global_es_el_intercepto():
    d = design_matrix({"A": (0, 1), "B": (0, 1)})
    d["y"] = [10, 20, 30, 50]
    r = factorial_analysis(d, ["A", "B"], "y")
    assert r["media_global"] == pytest.approx(27.5)
    assert r["coeficientes"]["Intercepto"] == pytest.approx(27.5)


def test_anova_replicado_descompone_la_suma_de_cuadrados():
    rng = np.random.default_rng(20)
    d = design_matrix({"A": (0, 1), "B": (0, 1)}, replicas=4)
    d["y"] = 10 - 2 * d.A + 1 * d.B + 0.5 * d.A * d.B + rng.normal(0, 0.3, len(d))
    r = factorial_analysis(d, ["A", "B"], "y")
    t = r["tabla_anova"].set_index("termino")
    sc_efectos = t.loc[["A", "B", "A:B"], "SC"].sum()
    assert sc_efectos + t.loc["Error puro", "SC"] == pytest.approx(t.loc["Total", "SC"])
    assert t.loc["Error puro", "gl"] == 16 - 4
    assert set(r["significativos"]) == {"A", "B", "A:B"}
    assert r["s_error"] == pytest.approx(0.3, abs=0.15)


def test_efecto_nulo_no_es_significativo():
    rng = np.random.default_rng(21)
    d = design_matrix({"A": (0, 1), "B": (0, 1)}, replicas=6)
    d["y"] = 10 - 2 * d.A + rng.normal(0, 0.4, len(d))   # B no influye
    r = factorial_analysis(d, ["A", "B"], "y")
    assert "A" in r["significativos"] and "B" not in r["significativos"]


def test_lenth_en_diseno_sin_replicas():
    rng = np.random.default_rng(22)
    d = design_matrix({"A": (0, 1), "B": (0, 1), "C": (0, 1)})
    d["y"] = 50 + 6 * d.A - 0.2 * d.B + 0.1 * d.C + rng.normal(0, 0.25, len(d))
    r = factorial_analysis(d, ["A", "B", "C"], "y")
    assert "Lenth" in r["metodo"]
    assert "A" in r["significativos"]
    assert r["pse_lenth"] > 0


def test_advierte_sobre_interaccion():
    rng = np.random.default_rng(23)
    d = design_matrix({"A": (0, 1), "B": (0, 1)}, replicas=5)
    d["y"] = 10 + 1 * d.A + 1 * d.B + 2.5 * d.A * d.B + rng.normal(0, 0.2, len(d))
    adv = factorial_analysis(d, ["A", "B"], "y")["advertencias"]
    assert any("nteracci" in a and "aislada" in a for a in adv)


def test_rechaza_codificacion_invalida():
    d = pd.DataFrame({"A": [0, 1, 0, 1], "B": [0, 0, 1, 1], "y": [1, 2, 3, 4]})
    with pytest.raises(ValueError):
        factorial_analysis(d, ["A", "B"], "y")


def test_datos_para_graficos():
    d = design_matrix({"A": (0, 1), "B": (0, 1)}, replicas=2)
    d["y"] = [1, 2, 3, 4, 1, 2, 3, 4]
    me = main_effect_plot_data(d, ["A", "B"], "y")
    assert len(me) == 4 and set(me["factor"]) == {"A", "B"}
    assert len(interaction_plot_data(d, "A", "B", "y")) == 4
