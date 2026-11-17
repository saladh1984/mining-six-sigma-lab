"""Verifica los factores de carta contra la tabla publicada (Montgomery, Apendice VI)."""
import pytest
from statistics_engine import constants as k

TABLA = {  # n: (A2, D3, D4, A3, B3, B4, c4)
    2: (1.880, 0.000, 3.267, 2.659, 0.000, 3.267, 0.7979),
    3: (1.023, 0.000, 2.574, 1.954, 0.000, 2.568, 0.8862),
    4: (0.729, 0.000, 2.282, 1.628, 0.000, 2.266, 0.9213),
    5: (0.577, 0.000, 2.114, 1.427, 0.000, 2.089, 0.9400),
    6: (0.483, 0.000, 2.004, 1.287, 0.030, 1.970, 0.9515),
    7: (0.419, 0.076, 1.924, 1.182, 0.118, 1.882, 0.9594),
    8: (0.373, 0.136, 1.864, 1.099, 0.185, 1.815, 0.9650),
    10: (0.308, 0.223, 1.777, 0.975, 0.284, 1.716, 0.9727),
}


@pytest.mark.parametrize("n,esperado", TABLA.items())
def test_factores_contra_tabla(n, esperado):
    a2, d3l, d4u, a3, b3v, b4v, c4v = esperado
    assert k.a2(n) == pytest.approx(a2, abs=0.002)
    assert k.d3_lower(n) == pytest.approx(d3l, abs=0.002)
    assert k.d4_upper(n) == pytest.approx(d4u, abs=0.002)
    assert k.a3(n) == pytest.approx(a3, abs=0.002)
    assert k.b3(n) == pytest.approx(b3v, abs=0.002)
    assert k.b4(n) == pytest.approx(b4v, abs=0.002)
    assert k.c4(n) == pytest.approx(c4v, abs=0.0002)


def test_d2_fuera_de_rango():
    with pytest.raises(ValueError):
        k.d2(1)
    with pytest.raises(ValueError):
        k.d2(30)


def test_c4_tiende_a_uno():
    assert k.c4(200) == pytest.approx(1.0, abs=0.002)
