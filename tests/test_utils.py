"""Pruebas de las utilidades de datos, entrada/salida y presentación."""
import io

import numpy as np
import pandas as pd
import pytest

from data.cases import obtener
from data.generators import subgrupos_racionales
from statistics_engine.capability import capability
from statistics_engine.regression import correlation_matrix, ols
from utils.io import a_csv, resultado_a_frame, validar_csv


@pytest.fixture
def datos():
    return obtener("caso1").datos(n=300, semilla=5)


# ------------------------------------------------- subgrupos racionales ----
def test_subgrupos_consecutivos(datos):
    m, desc = subgrupos_racionales(datos, "ciclo_total_min", None, 5)
    assert m.shape[1] == 5 and m.shape[0] == 60
    assert "sin estratificar" in desc


def test_subgrupos_estratificados_son_homogeneos(datos):
    """Cada subgrupo debe provenir de un solo nivel del factor."""
    n = 5
    bloques = []
    for _, g in datos.groupby("fase", observed=True, sort=True):
        x = g["ciclo_total_min"].to_numpy()
        k = x.size // n
        bloques.append(x[: k * n].reshape(k, n))
    esperado = np.vstack(bloques)
    m, desc = subgrupos_racionales(datos, "ciclo_total_min", "fase", n)
    assert np.allclose(m, esperado)
    assert "fase" in desc


def test_estratificar_reduce_sigma_dentro_cuando_el_factor_importa(datos):
    """La distancia depende de la fase: estratificar por fase debe reducir
    sigma dentro de subgrupo y aumentar la razón Cp/Pp."""
    m_cons, _ = subgrupos_racionales(datos, "ciclo_total_min", None, 5)
    m_fase, _ = subgrupos_racionales(datos, "ciclo_total_min", "fase", 5)
    r_cons = capability(datos["ciclo_total_min"], lse=38, subgrupos=m_cons)
    r_fase = capability(datos["ciclo_total_min"], lse=38, subgrupos=m_fase)
    assert r_fase.sigma_within < r_cons.sigma_within * 0.8
    assert r_fase.sigma_overall == pytest.approx(r_cons.sigma_overall)  # sigma total no cambia
    assert r_fase.cpk > r_cons.cpk


def test_subgrupos_insuficientes(datos):
    with pytest.raises(ValueError):
        subgrupos_racionales(datos.head(4), "ciclo_total_min", None, 5)


# ------------------------------------------------------- columnas duplicadas
def test_correlacion_tolera_columnas_repetidas(datos):
    """Una variable declarada a la vez como predictor y como componente no debe
    romper la matriz: se deduplican conservando el orden."""
    cols = ["ciclo_total_min", "distancia_km", "t_espera_pala_min", "t_espera_pala_min"]
    r = correlation_matrix(datos, "pearson", cols)
    assert r["r"].shape == (3, 3)
    assert list(r["r"].columns) == ["ciclo_total_min", "distancia_km", "t_espera_pala_min"]


def test_correlacion_requiere_dos_variables_distintas(datos):
    with pytest.raises(ValueError):
        correlation_matrix(datos, "pearson", ["distancia_km", "distancia_km"])


def test_ols_deduplica_predictores(datos):
    r1 = ols(datos, "ciclo_total_min", ["distancia_km", "distancia_km"])
    r2 = ols(datos, "ciclo_total_min", ["distancia_km"])
    assert r1["r2"] == pytest.approx(r2["r2"])


def test_ols_rechaza_respuesta_como_predictor(datos):
    with pytest.raises(ValueError, match="no puede ser"):
        ols(datos, "ciclo_total_min", ["ciclo_total_min", "distancia_km"])


# ------------------------------------------------------------------- CSV ---
def test_csv_ida_y_vuelta(datos):
    b = a_csv(datos.head(20))
    df, msgs = validar_csv(io.BytesIO(b))
    assert df is not None
    assert df.shape == (20, datos.shape[1])
    assert df["ciclo_total_min"].iloc[0] == pytest.approx(datos["ciclo_total_min"].iloc[0], abs=1e-5)


def test_csv_acepta_separador_coma():
    b = "a,b\n1.5,2.5\n3.5,4.5\n".encode()
    df, msgs = validar_csv(io.BytesIO(b))
    assert df is not None and df.shape == (2, 2)
    assert df["a"].iloc[0] == pytest.approx(1.5)


def test_csv_invalido_devuelve_mensajes():
    df, msgs = validar_csv(io.BytesIO(b"\x00\x01\x02binario"))
    assert df is None and msgs


def test_csv_reporta_faltantes():
    b = "a;b\n1,0;\n2,0;3,0\n".encode()
    df, msgs = validar_csv(io.BytesIO(b))
    assert df is not None
    assert any("faltantes" in m for m in msgs)


def test_resultado_a_frame(datos):
    r = capability(datos["ciclo_total_min"], lse=38)
    t = resultado_a_frame(r)
    assert set(t.columns) == {"indicador", "valor"}
    assert "cpk" in set(t["indicador"])


# ------------------------------------------------------------ presentación -
def test_etiquetas_de_presentacion_no_alteran_los_datos():
    from components.ui import ETIQUETAS, _etiquetar
    df = pd.DataFrame({"senal": [True], "categoria": ["A"], "valor": [1.0], "intacta": [2]})
    vista = _etiquetar(df)
    assert list(vista.columns) == ["señal", "categoría", "valor", "intacta"]
    assert list(df.columns) == ["senal", "categoria", "valor", "intacta"]  # original sin tocar
    assert vista["señal"].iloc[0] == df["senal"].iloc[0]


def test_fichas_didacticas_completas():
    from components.didactica import FICHAS
    assert len(FICHAS) >= 14
    for clave, f in FICHAS.items():
        assert f.titulo and f.que_es and f.como_funciona and f.como_se_interpreta, clave
        assert f.procedimiento, clave
        assert f.limitaciones, clave
