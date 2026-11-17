import numpy as np
import pytest
from statistics_engine.descriptive import describe, boxplot_stats, percentiles, describe_frame
import pandas as pd

X = [2, 4, 4, 4, 5, 5, 7, 9]  # media 5, sd poblacional 2, sd muestral sqrt(32/7)


def test_valores_conocidos():
    d = describe(X)
    assert d.n == 8
    assert d.media == pytest.approx(5.0)
    assert d.mediana == pytest.approx(4.5)
    assert d.moda == pytest.approx(4.0)
    assert d.varianza == pytest.approx(32 / 7)
    assert d.desv_std == pytest.approx(np.sqrt(32 / 7))
    assert d.coef_variacion == pytest.approx(np.sqrt(32 / 7) / 5)
    assert d.rango == pytest.approx(7.0)
    assert d.error_std_media == pytest.approx(np.sqrt(32 / 7) / np.sqrt(8))


def test_sin_moda():
    d = describe([1.0, 2.0, 3.0, 4.0], moda_decimales=3)
    assert d.moda is None
    assert "sin moda" in d.moda_metodo


def test_cuartiles_y_iqr():
    d = describe(list(range(1, 101)))
    assert d.q1 == pytest.approx(25.75)
    assert d.q3 == pytest.approx(75.25)
    assert d.iqr == pytest.approx(d.q3 - d.q1)


def test_nan_se_excluyen():
    d = describe([1.0, 2.0, np.nan, 3.0])
    assert d.n == 3 and d.n_missing == 1


def test_muestra_insuficiente():
    with pytest.raises(ValueError):
        describe([1.0])


def test_outliers_tukey():
    b = boxplot_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    assert b["n_outliers"] == 1 and b["outliers"] == [100.0]
    assert b["bigote_superior"] == 9.0


def test_percentiles_y_frame():
    p = percentiles(list(range(101)), ps=(50,))
    assert p.loc[0, "valor"] == pytest.approx(50.0)
    f = describe_frame(pd.DataFrame({"a": X, "b": X}))
    assert set(f["variable"]) == {"a", "b"} and f.shape[0] == 2
