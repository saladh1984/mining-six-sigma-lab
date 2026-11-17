import numpy as np
import pandas as pd
import pytest
from data import generators as gen
from data.cases import CASOS, obtener
from statistics_engine.regression import ols
from statistics_engine.doe import factorial_analysis


def test_reproducibilidad_por_semilla():
    a = gen.caso1_ciclo_camiones(n=50, semilla=99)
    b = gen.caso1_ciclo_camiones(n=50, semilla=99)
    c = gen.caso1_ciclo_camiones(n=50, semilla=100)
    pd.testing.assert_frame_equal(a, b)
    assert not a["ciclo_total_min"].equals(c["ciclo_total_min"])


@pytest.mark.parametrize("cid", list(CASOS))
def test_cada_caso_declara_variables_existentes(cid):
    caso = obtener(cid)
    d = caso.datos()
    for v in caso.variables:
        assert v.nombre in d.columns, f"{cid}: falta {v.nombre}"
    assert caso.respuesta in d.columns
    assert d.isna().sum().sum() == 0
    assert len(d) >= 100


@pytest.mark.parametrize("cid", list(CASOS))
def test_limites_de_especificacion_son_coherentes(cid):
    for v in obtener(cid).variables:
        if v.lie is not None and v.lse is not None:
            assert v.lie < v.lse
        if v.objetivo is not None:
            if v.lie is not None:
                assert v.objetivo >= v.lie
            if v.lse is not None:
                assert v.objetivo <= v.lse


def test_caso1_ciclo_es_la_suma_de_sus_componentes():
    d = gen.caso1_ciclo_camiones(n=200, semilla=5)
    comp = ["t_carguio_min", "t_transporte_min", "t_descarga_min", "t_retorno_min", "t_espera_pala_min"]
    assert np.allclose(d[comp].sum(axis=1), d["ciclo_total_min"], atol=0.01)


def test_caso1_productividad_es_consistente():
    d = gen.caso1_ciclo_camiones(n=100, semilla=6)
    assert np.allclose(d["productividad_tph"], d["carga_ton"] / (d["ciclo_total_min"] / 60), atol=0.05)


def test_caso1_la_regresion_recupera_la_estructura_del_generador():
    """El modelo generador es conocido: el análisis debe reencontrarlo.

    t_carguio = 2.60 + 0.0180*carga + 0.45*noche
    """
    d = gen.caso1_ciclo_camiones(n=3000, semilla=7)
    d["noche"] = (d["turno"] == "Noche").astype(float)
    r = ols(d, "t_carguio_min", ["carga_ton", "noche"])
    c = r["coeficientes"].set_index("termino")
    # Se verifica COBERTURA del intervalo de confianza, no igualdad puntual: el
    # intercepto y la pendiente estan correlacionados (la carga se centra en 218 t,
    # no en cero), de modo que cada estimacion puntual se desvia compensando a la
    # otra aunque el modelo sea correcto.
    verdad = {"const": 2.60, "carga_ton": 0.0180, "noche": 0.45}
    for termino, valor in verdad.items():
        assert c.loc[termino, "IC_inf"] <= valor <= c.loc[termino, "IC_sup"], termino
    assert r["s_residual"] == pytest.approx(0.55, abs=0.05)
    # Y que la prediccion reproduce el nivel correcto a la carga media
    assert r["ajustados"].mean() == pytest.approx(d["t_carguio_min"].mean(), abs=1e-6)


def test_caso1_la_mejora_reduce_media_y_dispersion():
    base = gen.caso1_ciclo_camiones(n=2000, semilla=8)
    mej = gen.caso1_ciclo_camiones(n=2000, semilla=8, mejora=True)
    assert mej["ciclo_total_min"].mean() < base["ciclo_total_min"].mean()
    assert mej["ciclo_total_min"].std() < base["ciclo_total_min"].std()
    assert mej["t_espera_pala_min"].mean() < base["t_espera_pala_min"].mean()


def test_atributos_son_conteos_validos():
    d = gen.datos_atributos(k=30, semilla=3)
    assert (d["defectuosas"] <= d["n_inspeccionadas"]).all()
    assert (d["defectuosas"] >= 0).all() and (d["defectos"] >= 0).all()
    assert len(d) == 30


def test_gage_rr_es_cruzado_y_balanceado():
    d = gen.datos_gage_rr(piezas=10, operadores=3, replicas=3)
    assert len(d) == 90
    assert d.groupby(["pieza", "operador"]).size().nunique() == 1


def test_doe_carguio_recupera_sus_efectos():
    """Verdad: efecto A = -1.60, efecto B = -0.70, interacción = +0.35."""
    d = gen.datos_doe_carguio(replicas=30, semilla=11)
    e = factorial_analysis(d, ["A", "B"], "t_carguio_min")["efectos"].set_index("termino")["efecto"]
    assert e["A"] == pytest.approx(-1.60, abs=0.08)
    assert e["B"] == pytest.approx(-0.70, abs=0.08)
    assert e["A:B"] == pytest.approx(0.35, abs=0.08)


def test_doe_molienda_es_2a3_sin_replicas():
    d = gen.datos_doe_molienda()
    assert len(d) == 8
    r = factorial_analysis(d, ["A", "B", "C"], "recuperacion_pct")
    assert "Lenth" in r["metodo"]
    assert "C" in r["significativos"]      # efecto dominante (-4.40)
    assert "B" in r["significativos"]      # efecto secundario (+2.00)
    assert "A:C" not in r["significativos"]  # efecto nulo


def test_subgrupos_desde_serie():
    m = gen.subgrupos_desde_serie(np.arange(23), n=5)
    assert m.shape == (4, 5)
    with pytest.raises(ValueError):
        gen.subgrupos_desde_serie([1, 2, 3], n=5)


def test_hoja_de_verificacion():
    d = gen.hoja_verificacion()
    assert {"categoria", "turno", "eventos", "minutos_perdidos"} <= set(d.columns)
    assert (d["eventos"] >= 0).all() and (d["minutos_perdidos"] >= 0).all()
