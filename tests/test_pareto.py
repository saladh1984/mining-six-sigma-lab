import pytest
from statistics_engine.pareto import pareto, matriz_causa_efecto


def test_ordena_y_acumula():
    r = pareto(["A", "B", "C"], [10, 50, 40])
    t = r["tabla"]
    assert list(t["categoria"]) == ["B", "C", "A"]
    assert list(t["pct"]) == pytest.approx([50.0, 40.0, 10.0])
    assert list(t["pct_acumulado"]) == pytest.approx([50.0, 90.0, 100.0])
    assert t["pct_acumulado"].iloc[-1] == pytest.approx(100.0)


def test_pocos_vitales_incluye_la_categoria_que_cruza_el_corte():
    r = pareto(["A", "B", "C", "D"], [50, 30, 15, 5], corte=80)
    assert set(r["pocos_vitales"]) == {"A", "B", "C"}
    assert r["pct_cubierto_por_vitales"] == pytest.approx(95.0)


def test_detecta_distribucion_plana():
    r = pareto([f"C{i}" for i in range(10)], [10] * 10)
    assert r["concentrado"] is False
    assert "NO está concentrada" in r["lectura"]


def test_agrupa_categorias_repetidas():
    r = pareto(["A", "A", "B"], [10, 20, 5])
    assert r["tabla"].set_index("categoria").loc["A", "valor"] == 30


def test_rechaza_negativos_y_total_cero():
    with pytest.raises(ValueError):
        pareto(["A"], [-1])
    with pytest.raises(ValueError):
        pareto(["A", "B"], [0, 0])


def test_matriz_causa_efecto_pondera():
    r = matriz_causa_efecto(["c1", "c2"], ["s1", "s2"], [10, 5],
                            {"c1": [9, 1], "c2": [3, 9]})
    t = r["tabla"].set_index("causa")
    assert t.loc["c1", "puntaje"] == pytest.approx(10 * 9 + 5 * 1)
    assert t.loc["c2", "puntaje"] == pytest.approx(10 * 3 + 5 * 9)
    assert r["tabla"].iloc[0]["causa"] == "c1"


def test_matriz_rechaza_dimension_incorrecta():
    with pytest.raises(ValueError):
        matriz_causa_efecto(["c1"], ["s1", "s2"], [1, 1], {"c1": [9]})
