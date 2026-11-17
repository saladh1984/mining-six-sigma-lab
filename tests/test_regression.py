import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statistics_engine.regression import correlation_matrix, scatter_fit, ols


@pytest.fixture
def datos():
    rng = np.random.default_rng(10)
    n = 400
    x1, x2 = rng.uniform(0, 10, n), rng.uniform(0, 5, n)
    return pd.DataFrame({"y": 3 + 2 * x1 - 1.5 * x2 + rng.normal(0, 1, n), "x1": x1, "x2": x2})


def test_ols_recupera_coeficientes(datos):
    r = ols(datos, "y", ["x1", "x2"])
    c = dict(zip(r["coeficientes"]["termino"], r["coeficientes"]["coeficiente"]))
    assert c["const"] == pytest.approx(3, abs=0.3)
    assert c["x1"] == pytest.approx(2, abs=0.05)
    assert c["x2"] == pytest.approx(-1.5, abs=0.1)
    assert r["s_residual"] == pytest.approx(1.0, abs=0.1)


def test_r2_ajustado_menor_que_r2(datos):
    r = ols(datos, "y", ["x1", "x2"])
    assert r["r2_ajustado"] < r["r2"] <= 1.0


def test_r2_coincide_con_statsmodels(datos):
    r = ols(datos, "y", ["x1", "x2"])
    m = sm.OLS(datos["y"], sm.add_constant(datos[["x1", "x2"]])).fit()
    assert r["r2"] == pytest.approx(m.rsquared)
    assert r["f"] == pytest.approx(m.fvalue)


def test_regresion_simple_sufre_sesgo_por_variable_omitida():
    """Con predictores correlacionados, omitir uno sesga el coeficiente del otro."""
    rng = np.random.default_rng(11)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 0.6, n)
    y = 1 + 2 * x1 + 3 * x2 + rng.normal(0, 0.5, n)
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    simple = ols(d, "y", ["x1"])["coeficientes"].set_index("termino").loc["x1", "coeficiente"]
    multi = ols(d, "y", ["x1", "x2"])["coeficientes"].set_index("termino").loc["x1", "coeficiente"]
    assert multi == pytest.approx(2.0, abs=0.1)
    assert simple > 3.5          # sesgado hacia arriba


def test_vif_detecta_multicolinealidad():
    rng = np.random.default_rng(12)
    x1 = rng.normal(0, 1, 300)
    d = pd.DataFrame({"x1": x1, "x2": x1 + rng.normal(0, 0.02, 300)})
    d["y"] = d.x1 + rng.normal(0, 1, 300)
    r = ols(d, "y", ["x1", "x2"])
    assert r["diagnostico"]["vif"]["VIF"].max() > 10
    assert any("ulticolinealidad" in a for a in r["advertencias"])


def test_breusch_pagan_detecta_heterocedasticidad():
    rng = np.random.default_rng(13)
    x = rng.uniform(1, 10, 400)
    d = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, x, 400)})
    r = ols(d, "y", ["x"])
    assert r["diagnostico"]["breusch_pagan"]["p_valor"] < 0.05
    assert any("eterocedasticidad" in a for a in r["advertencias"])


def test_correlacion_pearson_vs_spearman():
    x = np.arange(1, 51, dtype=float)
    d = pd.DataFrame({"x": x, "y": x ** 3})   # monotona pero no lineal
    rp = correlation_matrix(d, "pearson")["r"].loc["x", "y"]
    rs = correlation_matrix(d, "spearman")["r"].loc["x", "y"]
    assert rs == pytest.approx(1.0)
    assert rp < 0.98


def test_correlacion_perfecta_y_simetria():
    d = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "c": [4.0, 3, 2, 1]})
    m = correlation_matrix(d)["r"]
    assert m.loc["a", "b"] == pytest.approx(1.0)
    assert m.loc["a", "c"] == pytest.approx(-1.0)
    assert m.loc["a", "b"] == m.loc["b", "a"]
    assert np.allclose(np.diag(m), 1.0)


def test_scatter_fit_r2_es_cuadrado_de_r():
    rng = np.random.default_rng(14)
    x = rng.normal(0, 1, 200)
    s = scatter_fit(x, 2 * x + rng.normal(0, 1, 200))
    assert s["r2"] == pytest.approx(s["r"] ** 2)
    assert s["pendiente"] == pytest.approx(2.0, abs=0.2)
    assert len(s["ic_pred_sup"]) == len(s["x_linea"])
    assert np.all(s["ic_pred_sup"] >= s["ic_media_sup"])   # prediccion mas ancha que media
