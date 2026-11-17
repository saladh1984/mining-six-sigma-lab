"""Fase I / Fase II, exclusión de causas asignables y cartas de memoria."""
import numpy as np
import pytest

from statistics_engine.control_charts import (CusumResult, LimitesControl, c_chart,
                                              cusum, ewma, i_mr, np_chart, p_chart,
                                              u_chart, xbar_r, xbar_s)


# ------------------------------------------------------- Fase I / Fase II --
def test_fase_ii_congela_los_limites():
    rng = np.random.default_rng(0)
    base = rng.normal(50, 2, (25, 5))
    x1, _ = xbar_r(base)
    nuevos = rng.normal(50, 2, (15, 5))
    x2, _ = xbar_r(nuevos, limites=x1.limites)
    assert x2.centro[0] == pytest.approx(x1.centro[0])
    assert x2.lcs[0] == pytest.approx(x1.lcs[0])
    assert x2.lci[0] == pytest.approx(x1.lci[0])
    assert "FASE II" in x2.origen_limites


def test_fase_ii_detecta_el_corrimiento_que_recalcular_oculta():
    """El punto central de todo el módulo: recalcular límites esconde el corrimiento."""
    rng = np.random.default_rng(1)
    base = rng.normal(50, 2, (30, 5))
    corridos = rng.normal(56, 2, (20, 5))      # corrimiento de 3 sigma
    x_base, _ = xbar_r(base)
    con_congelados, _ = xbar_r(corridos, limites=x_base.limites)
    recalculando, _ = xbar_r(corridos)
    assert len(con_congelados.fuera_de_control) == 20      # los detecta todos
    assert len(recalculando.fuera_de_control) == 0          # no detecta ninguno
    assert recalculando.centro[0] > x_base.centro[0] + 4    # la línea central se mudó


def test_fase_i_declara_su_origen():
    rng = np.random.default_rng(2)
    x, _ = xbar_r(rng.normal(0, 1, (20, 5)))
    assert "FASE I" in x.origen_limites
    assert x.limites.k_base == 20
    assert x.limites.tipo == "xbar_r"


@pytest.mark.parametrize("carta,args", [
    ("p", ([3, 5, 4, 6, 2, 5], 150)),
    ("np", ([3, 5, 4, 6, 2, 5], 150)),
    ("c", ([6, 4, 7, 5, 8, 6],)),
    ("u", ([6, 4, 7, 5, 8, 6], [10, 9, 11, 10, 12, 10])),
])
def test_atributos_tambien_congelan(carta, args):
    f = {"p": p_chart, "np": np_chart, "c": c_chart, "u": u_chart}[carta]
    r1 = f(*args)
    r2 = f(*args, limites=r1.limites)
    assert r2.centro[0] == pytest.approx(r1.centro[0])
    assert "FASE II" in r2.origen_limites


def test_limites_de_otra_carta_se_rechazan():
    rng = np.random.default_rng(3)
    m = rng.normal(0, 1, (20, 5))
    x, _ = xbar_r(m)
    with pytest.raises(ValueError, match="no son intercambiables|aplicando a una carta"):
        xbar_s(m, limites=x.limites)


def test_limites_con_otro_tamano_de_subgrupo_se_rechazan():
    rng = np.random.default_rng(4)
    x, _ = xbar_r(rng.normal(0, 1, (20, 5)))
    with pytest.raises(ValueError, match="tamaño"):
        xbar_r(rng.normal(0, 1, (20, 4)), limites=x.limites)


# ------------------------------------------------------------- exclusión --
def test_excluir_cambia_los_limites_pero_no_los_puntos_graficados():
    rng = np.random.default_rng(5)
    m = rng.normal(50, 2, (30, 5))
    m[11] += 9
    con, _ = xbar_r(m)
    sin, _ = xbar_r(m, excluir=[11])
    assert len(sin.puntos) == len(con.puntos) == 30     # se grafica igual
    assert sin.centro[0] < con.centro[0]                # pero no contamina el centro
    assert sin.lcs[0] < con.lcs[0]                      # ni ensancha los límites
    assert sin.to_frame().loc[11, "excluido"]
    assert 11 in sin.excluidos


def test_senales_nuevas_ignora_los_ya_excluidos():
    rng = np.random.default_rng(6)
    m = rng.normal(50, 2, (30, 5))
    m[11] += 9
    r, _ = xbar_r(m, excluir=[11])
    assert 11 in r.fuera_de_control          # sigue señalado en el gráfico
    assert 11 not in r.senales_nuevas        # pero ya está explicado


def test_excluir_indice_invalido():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="fuera de rango"):
        xbar_r(rng.normal(0, 1, (10, 5)), excluir=[99])


def test_excluir_casi_todo_falla():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="menos de 2"):
        xbar_r(rng.normal(0, 1, (5, 5)), excluir=[0, 1, 2, 3])


def test_i_mr_descarta_los_rangos_moviles_que_tocan_un_punto_excluido():
    x = [10, 10, 10, 40, 10, 10, 10]        # el 40 contamina DOS rangos móviles
    con, mr_con = i_mr(x)
    sin, mr_sin = i_mr(x, excluir=[3])
    assert mr_sin.centro[0] < mr_con.centro[0]
    assert mr_sin.centro[0] == pytest.approx(0.0)   # el resto es constante


# ------------------------------------------------------------------ EWMA --
def test_ewma_recursion():
    """z_i = lam*x_i + (1-lam)*z_{i-1}, con z_0 = mu0."""
    x = [10.0, 12.0, 11.0]
    r = ewma(x, lam=0.5, mu0=10.0, sigma=1.0)
    esperado, z = [], 10.0
    for v in x:
        z = 0.5 * v + 0.5 * z
        esperado.append(z)
    assert np.allclose(r.puntos, esperado)


def test_ewma_limites_se_ensanchan_y_convergen():
    rng = np.random.default_rng(9)
    r = ewma(rng.normal(0, 1, 120), lam=0.2, L=3.0, mu0=0.0, sigma=1.0)
    ancho = r.lcs - r.centro
    assert ancho[0] < ancho[5] < ancho[30]                       # se ensanchan
    asintota = 3.0 * np.sqrt(0.2 / 1.8)
    assert ancho[-1] == pytest.approx(asintota, rel=0.01)        # y convergen


def test_ewma_detecta_corrimiento_pequeno_antes_que_shewhart():
    """Resultado conocido: ante 1 sigma, EWMA/CUSUM superan a Shewhart."""
    detecta_sh = detecta_ew = 0
    for rep in range(60):
        r = np.random.default_rng(500 + rep)
        base = r.normal(100, 1, 50)
        serie = np.r_[base, r.normal(101, 1, 50)]
        i_b, _ = i_mr(base)
        e_b = ewma(base, lam=0.2, L=3.0)
        i_2, _ = i_mr(serie, limites=i_b.limites)
        e_2 = ewma(serie, lam=0.2, L=3.0, limites=e_b.limites)
        detecta_sh += any(i >= 50 for i in i_2.violaciones.get(1, []))
        detecta_ew += any(i >= 50 for i in e_2.fuera_de_control)
    assert detecta_ew > detecta_sh


def test_ewma_lambda_invalido():
    with pytest.raises(ValueError):
        ewma([1, 2, 3], lam=0.0)
    with pytest.raises(ValueError):
        ewma([1, 2, 3], lam=1.5)


def test_ewma_con_lambda_uno_equivale_a_individuales():
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 60)
    e = ewma(x, lam=1.0, L=3.0, mu0=0.0, sigma=1.0)
    assert np.allclose(e.puntos, x)
    assert e.lcs[-1] == pytest.approx(3.0, rel=1e-6)


# ----------------------------------------------------------------- CUSUM --
def test_cusum_recursion_y_umbral():
    x = [10, 10, 13, 13, 13]
    c = cusum(x, k_sigma=0.5, h_sigma=4.0, mu0=10.0, sigma=1.0)
    assert c.K == pytest.approx(0.5) and c.H == pytest.approx(4.0)
    esperado_p, acc = [], 0.0
    for v in x:
        acc = max(0.0, v - 10.5 + acc)
        esperado_p.append(acc)
    assert np.allclose(c.c_mas, esperado_p)
    assert np.allclose(c.c_menos, 0.0)
    assert c.senales_mas and not c.senales_menos


def test_cusum_proceso_estable_tiene_pocas_falsas_alarmas():
    """Con k=0.5 y h=4 la longitud media de racha en control es ~168, de modo
    que en 200 observaciones se espera alrededor de una excursión falsa."""
    rng = np.random.default_rng(12)
    c = cusum(rng.normal(0, 1, 200), mu0=0.0, sigma=1.0)
    assert len(c.excursiones) <= 2


def test_cusum_cuenta_excursiones_no_puntos():
    """Sin reinicio, una sola excursión deja varios puntos sobre H."""
    serie = [0.0] * 10 + [4.0] * 10 + [0.0] * 10
    c = cusum(serie, k_sigma=0.5, h_sigma=4.0, mu0=0.0, sigma=1.0)
    assert len(c.senales_mas) > len(c.excursiones_mas)
    assert c.excursiones_mas == [min(c.senales_mas)]


def test_cusum_con_reinicio_corta_la_acumulacion():
    serie = [0.0] * 5 + [4.0] * 10
    sin = cusum(serie, h_sigma=4.0, mu0=0.0, sigma=1.0, reiniciar_tras_senal=False)
    con = cusum(serie, h_sigma=4.0, mu0=0.0, sigma=1.0, reiniciar_tras_senal=True)
    assert sin.c_mas.max() > con.c_mas.max()
    assert con.reiniciado is True and sin.reiniciado is False


def test_cusum_detecta_corrimiento_sostenido():
    rng = np.random.default_rng(13)
    serie = np.r_[rng.normal(100, 1, 40), rng.normal(101.5, 1, 40)]
    c = cusum(serie, mu0=100.0, sigma=1.0)
    assert any(i >= 40 for i in c.senales_mas)
    assert not c.senales_menos


def test_cusum_detecta_corrimiento_a_la_baja():
    rng = np.random.default_rng(14)
    serie = np.r_[rng.normal(100, 1, 40), rng.normal(98.5, 1, 40)]
    c = cusum(serie, mu0=100.0, sigma=1.0)
    assert any(i >= 40 for i in c.senales_menos)
    assert not c.senales_mas


def test_cusum_to_frame():
    c = cusum([1, 2, 3, 4, 5], mu0=3.0, sigma=1.0)
    df = c.to_frame()
    assert {"observacion", "C_mas", "C_menos", "H", "senal_superior",
            "senal_inferior", "excluido"} == set(df.columns)
    assert len(df) == 5


def test_resumen_de_limites_es_exportable():
    rng = np.random.default_rng(15)
    x, _ = xbar_r(rng.normal(0, 1, (20, 5)))
    r = x.limites.resumen()
    assert {"campo", "valor"} == set(r.columns)
    assert "tipo de carta" in set(r["campo"])
