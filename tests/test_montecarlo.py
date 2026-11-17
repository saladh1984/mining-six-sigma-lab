"""Propagación de incertidumbre y análisis de sensibilidad."""
import numpy as np
import pytest

from statistics_engine.montecarlo import (Incierto, crear_generador, simular,
                                          van_montecarlo)


# --------------------------------------------------------- distribuciones --
def test_fija_no_tiene_dispersion():
    rng = crear_generador(1)
    x = Incierto("a", "fija", {"valor": 7.5}).muestrear(1000, rng)
    assert np.allclose(x, 7.5)


def test_normal_recupera_sus_momentos():
    rng = crear_generador(2)
    x = Incierto("a", "normal", {"media": 50, "sd": 4}).muestrear(200_000, rng)
    assert x.mean() == pytest.approx(50, abs=0.1)
    assert x.std(ddof=1) == pytest.approx(4, rel=0.02)


def test_uniforme_respeta_el_rango():
    rng = crear_generador(3)
    x = Incierto("a", "uniforme", {"minimo": 2, "maximo": 9}).muestrear(50_000, rng)
    assert x.min() >= 2 and x.max() <= 9
    assert x.mean() == pytest.approx(5.5, abs=0.05)


def test_triangular_media_teorica():
    """Media de una triangular = (a + m + b) / 3."""
    rng = crear_generador(4)
    x = Incierto("a", "triangular", {"minimo": 1, "moda": 2, "maximo": 9}).muestrear(200_000, rng)
    assert x.mean() == pytest.approx((1 + 2 + 9) / 3, rel=0.01)
    assert x.min() >= 1 and x.max() <= 9


def test_pert_media_teorica_y_rango():
    """Media PERT = (a + 4m + b) / 6, y pondera más la moda que la triangular."""
    rng = crear_generador(5)
    p = Incierto("a", "pert", {"minimo": 1, "moda": 2, "maximo": 9})
    t = Incierto("a", "triangular", {"minimo": 1, "moda": 2, "maximo": 9})
    xp = p.muestrear(200_000, rng)
    xt = t.muestrear(200_000, rng)
    assert xp.mean() == pytest.approx((1 + 4 * 2 + 9) / 6, rel=0.01)
    assert xp.min() >= 1 and xp.max() <= 9
    assert xp.std(ddof=1) < xt.std(ddof=1)       # más concentrada en la moda


def test_lognormal_es_positiva_y_asimetrica():
    rng = crear_generador(6)
    x = Incierto("a", "lognormal", {"media": 100, "sd": 30}).muestrear(200_000, rng)
    assert x.min() > 0
    assert x.mean() == pytest.approx(100, rel=0.02)
    assert x.std(ddof=1) == pytest.approx(30, rel=0.03)
    assert np.median(x) < x.mean()               # cola a la derecha


def test_recorte_fisico():
    rng = crear_generador(7)
    x = Incierto("a", "normal", {"media": 1, "sd": 5}, minimo_fisico=0).muestrear(10_000, rng)
    assert x.min() >= 0


def test_distribucion_desconocida_y_parametros_invalidos():
    with pytest.raises(ValueError, match="desconocida"):
        Incierto("a", "cauchy", {})
    with pytest.raises(ValueError, match="requiere"):
        Incierto("a", "normal", {"media": 1})
    with pytest.raises(ValueError, match="negativa"):
        Incierto("a", "normal", {"media": 1, "sd": -1})
    with pytest.raises(ValueError, match="mínimo <= moda"):
        Incierto("a", "triangular", {"minimo": 5, "moda": 1, "maximo": 9})
    with pytest.raises(ValueError, match="menor que el máximo"):
        Incierto("a", "uniforme", {"minimo": 9, "maximo": 2})


# ---------------------------------------------------------- propagación ---
def test_reproducibilidad_por_semilla():
    e = [Incierto("x", "normal", {"media": 0, "sd": 1})]
    f = lambda x: x * 2
    a = simular(e, f, n=500, semilla=42)
    b = simular(e, f, n=500, semilla=42)
    c = simular(e, f, n=500, semilla=43)
    assert np.allclose(a.salida, b.salida)
    assert not np.allclose(a.salida, c.salida)


def test_propagacion_analitica_conocida():
    """Para z = x + y con x e y normales independientes: var(z) = var(x)+var(y)."""
    e = [Incierto("x", "normal", {"media": 10, "sd": 3}),
         Incierto("y", "normal", {"media": 5, "sd": 4})]
    r = simular(e, lambda x, y: x + y, n=200_000, semilla=8)
    assert r.salida.mean() == pytest.approx(15, abs=0.1)
    assert r.salida.std(ddof=1) == pytest.approx(np.sqrt(9 + 16), rel=0.02)


def test_entradas_fijas_dan_salida_constante():
    e = [Incierto("x", "fija", {"valor": 3}), Incierto("y", "fija", {"valor": 4})]
    r = simular(e, lambda x, y: x * y, n=100, semilla=9)
    assert np.allclose(r.salida, 12)
    assert r.resumen()["desv_std"] == pytest.approx(0.0)


def test_modelo_no_vectorizado_falla_con_mensaje_claro():
    e = [Incierto("x", "normal", {"media": 0, "sd": 1})]
    with pytest.raises(ValueError, match="vectorizado"):
        simular(e, lambda x: 1.0, n=100, semilla=10)


def test_nombres_repetidos():
    e = [Incierto("x", "fija", {"valor": 1}), Incierto("x", "fija", {"valor": 2})]
    with pytest.raises(ValueError, match="repetidos"):
        simular(e, lambda x: x, n=50)


def test_error_montecarlo_baja_como_raiz_de_n():
    e = [Incierto("x", "normal", {"media": 0, "sd": 1})]
    chico = simular(e, lambda x: x, n=2_000, semilla=11)
    grande = simular(e, lambda x: x, n=32_000, semilla=11)
    # 16 veces mas simulaciones -> error aproximadamente 4 veces menor
    assert grande.error_montecarlo() == pytest.approx(chico.error_montecarlo() / 4, rel=0.25)


def test_probabilidades_e_intervalo():
    e = [Incierto("x", "normal", {"media": 0, "sd": 1})]
    r = simular(e, lambda x: x, n=200_000, semilla=12)
    assert r.probabilidad_mayor_que(0) == pytest.approx(0.5, abs=0.01)
    assert r.probabilidad_menor_que(-1.96) == pytest.approx(0.025, abs=0.003)
    lo, hi = r.intervalo(0.95)
    assert lo == pytest.approx(-1.96, abs=0.05) and hi == pytest.approx(1.96, abs=0.05)
    with pytest.raises(ValueError):
        r.intervalo(1.5)


# -------------------------------------------------------- sensibilidad ----
def test_sensibilidad_identifica_la_entrada_dominante():
    e = [Incierto("domina", "normal", {"media": 0, "sd": 10}),
         Incierto("menor", "normal", {"media": 0, "sd": 0.1})]
    r = simular(e, lambda domina, menor: domina + menor, n=20_000, semilla=13)
    s = r.sensibilidad()
    assert s.iloc[0]["entrada"] == "domina"
    assert s.iloc[0]["contribucion_pct"] > 95
    assert s["contribucion_pct"].sum() == pytest.approx(100.0)


def test_sensibilidad_detecta_el_signo():
    e = [Incierto("sube", "normal", {"media": 0, "sd": 1}),
         Incierto("baja", "normal", {"media": 0, "sd": 1})]
    r = simular(e, lambda sube, baja: sube - baja, n=20_000, semilla=14)
    s = r.sensibilidad().set_index("entrada")
    assert s.loc["sube", "correlacion_spearman"] > 0
    assert s.loc["baja", "correlacion_spearman"] < 0


def test_entrada_fija_no_contribuye_a_la_incertidumbre():
    e = [Incierto("incierta", "normal", {"media": 0, "sd": 1}),
         Incierto("conocida", "fija", {"valor": 100})]
    r = simular(e, lambda incierta, conocida: incierta * conocida, n=5_000, semilla=15)
    s = r.sensibilidad().set_index("entrada")
    assert bool(s.loc["conocida", "fija"]) is True
    assert s.loc["conocida", "contribucion_pct"] == pytest.approx(0.0)
    assert s.loc["incierta", "contribucion_pct"] == pytest.approx(100.0)


def test_sensibilidad_capta_relaciones_no_lineales():
    """Spearman detecta monotonía aunque la relación no sea lineal."""
    e = [Incierto("x", "uniforme", {"minimo": 1, "maximo": 10})]
    r = simular(e, lambda x: x ** 3, n=10_000, semilla=16)
    assert r.sensibilidad().iloc[0]["correlacion_spearman"] == pytest.approx(1.0, abs=1e-6)


def test_convergencia_estabiliza():
    e = [Incierto("x", "normal", {"media": 7, "sd": 2})]
    r = simular(e, lambda x: x, n=40_000, semilla=17)
    c = r.convergencia()
    assert c["error_std"].iloc[-1] < c["error_std"].iloc[0]
    assert c["media_acumulada"].iloc[-1] == pytest.approx(7, abs=0.1)


# ------------------------------------------------------------------ VAN ---
def test_van_coincide_con_el_calculo_deterministico_si_todo_es_fijo():
    r = van_montecarlo(
        Incierto("inversion", "fija", {"valor": 100_000}),
        Incierto("beneficio_anual", "fija", {"valor": 30_000}),
        Incierto("tasa", "fija", {"valor": 0.10}),
        horizonte=5, n=100, semilla=18)
    esperado = sum(30_000 / 1.1 ** t for t in range(1, 6)) - 100_000
    assert r.salida[0] == pytest.approx(esperado)
    assert np.allclose(r.salida, esperado)


def test_van_entrega_probabilidad_de_perdida():
    r = van_montecarlo(
        Incierto("inversion", "triangular", {"minimo": 90_000, "moda": 100_000, "maximo": 150_000}),
        Incierto("beneficio_anual", "triangular", {"minimo": 10_000, "moda": 25_000, "maximo": 35_000}),
        Incierto("tasa", "uniforme", {"minimo": 0.08, "maximo": 0.18}),
        horizonte=5, n=20_000, semilla=19)
    p = r.probabilidad_menor_que(0)
    assert 0 < p < 1                       # hay riesgo real, no certeza
    assert r.resumen()["desv_std"] > 0
    s = r.sensibilidad()
    assert set(s["entrada"]) == {"inversion", "beneficio_anual", "tasa"}


def test_van_horizonte_invalido():
    i = Incierto("inversion", "fija", {"valor": 1})
    b = Incierto("beneficio_anual", "fija", {"valor": 1})
    t = Incierto("tasa", "fija", {"valor": 0.1})
    with pytest.raises(ValueError, match="horizonte"):
        van_montecarlo(i, b, t, horizonte=0)
