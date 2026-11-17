import numpy as np
import pandas as pd
import pytest
from statistics_engine.gagerr import gage_rr_anova
from data.generators import datos_gage_rr


def test_descomposicion_de_la_suma_de_cuadrados():
    d = datos_gage_rr(semilla=1)
    t = gage_rr_anova(d, "pieza", "operador", "medicion")["tabla_anova"].set_index("Fuente")
    suma = t.loc[["pieza", "operador", "pieza x operador", "Repetibilidad (error)"], "SC"].sum()
    assert suma == pytest.approx(t.loc["Total", "SC"])
    assert t.loc["Total", "gl"] == 10 * 3 * 3 - 1


def test_componentes_de_varianza_suman_el_total():
    r = gage_rr_anova(datos_gage_rr(semilla=2), "pieza", "operador", "medicion")
    c = r["componentes"].set_index("Fuente")
    assert (c.loc["Gage R&R total", "Varianza"] + c.loc["Variación de pieza (proceso)", "Varianza"]
            == pytest.approx(c.loc["Variación total", "Varianza"]))
    assert c.loc["Variación total", "%Contribucion"] == pytest.approx(100.0)


def test_recupera_repetibilidad_conocida():
    d = datos_gage_rr(piezas=15, operadores=3, replicas=4, semilla=3,
                      sigma_pieza=5.0, sigma_repetibilidad=1.0, sesgos_operador=(0, 0, 0))
    r = gage_rr_anova(d, "pieza", "operador", "medicion")
    c = r["componentes"].set_index("Fuente")
    assert np.sqrt(c.loc["  Repetibilidad (equipo)", "Varianza"]) == pytest.approx(1.0, rel=0.15)
    assert np.sqrt(c.loc["Variación de pieza (proceso)", "Varianza"]) == pytest.approx(5.0, rel=0.25)


def test_sin_sesgo_de_operador_la_reproducibilidad_es_casi_nula():
    d = datos_gage_rr(piezas=12, operadores=3, replicas=4, semilla=4,
                      sigma_pieza=6.0, sigma_repetibilidad=1.0, sesgos_operador=(0.0, 0.0, 0.0))
    c = gage_rr_anova(d, "pieza", "operador", "medicion")["componentes"].set_index("Fuente")
    assert c.loc["  Reproducibilidad", "%Contribucion"] < 2.0


def test_sistema_malo_se_declara_inaceptable():
    d = datos_gage_rr(piezas=10, operadores=3, replicas=3, semilla=5,
                      sigma_pieza=1.0, sigma_repetibilidad=3.0, sesgos_operador=(0, 2.0, -2.0))
    r = gage_rr_anova(d, "pieza", "operador", "medicion")
    assert r["pct_rr_estudio"] > 30
    assert "INACEPTABLE" in r["veredicto"]
    assert r["ndc"] < 5


def test_sistema_bueno_se_declara_aceptable():
    d = datos_gage_rr(piezas=10, operadores=3, replicas=3, semilla=6,
                      sigma_pieza=10.0, sigma_repetibilidad=0.25, sesgos_operador=(0, 0.1, -0.1))
    r = gage_rr_anova(d, "pieza", "operador", "medicion")
    assert r["pct_rr_estudio"] < 10 and "ACEPTABLE" in r["veredicto"] and r["ndc"] >= 5


def test_pct_estudio_usa_desviaciones_y_contribucion_varianzas():
    r = gage_rr_anova(datos_gage_rr(semilla=7), "pieza", "operador", "medicion")
    c = r["componentes"].set_index("Fuente")
    v = c.loc["Gage R&R total", "%Contribucion"] / 100
    s = c.loc["Gage R&R total", "%VariacionEstudio"] / 100
    assert s == pytest.approx(np.sqrt(v), rel=1e-6)


def test_rechaza_diseno_desbalanceado():
    d = datos_gage_rr(semilla=8).iloc[:-1]
    with pytest.raises(ValueError, match="balanceado"):
        gage_rr_anova(d, "pieza", "operador", "medicion")


def test_rechaza_sin_replicas():
    d = datos_gage_rr(replicas=1, semilla=9)
    with pytest.raises(ValueError, match="replicas"):
        gage_rr_anova(d, "pieza", "operador", "medicion")


def test_porcentaje_tolerancia():
    r = gage_rr_anova(datos_gage_rr(semilla=10), "pieza", "operador", "medicion", tolerancia=40)
    c = r["componentes"].set_index("Fuente")
    assert c.loc["Gage R&R total", "%Tolerancia(P/T)"] == pytest.approx(
        100 * c.loc["Gage R&R total", "VarEstudio(6s)"] / 40)
