import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statistics_engine.inference import normality, levene, two_sample_t, paired_t, one_way_anova


def test_t_dos_muestras_contra_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 50), rng.normal(11, 2, 50)
    r = two_sample_t(a, b, equal_var=True)
    t, p = stats.ttest_ind(a, b, equal_var=True)
    assert r.estadistico == pytest.approx(t) and r.p_valor == pytest.approx(p)
    assert r.gl == pytest.approx(98)


def test_welch_se_elige_con_varianzas_distintas():
    rng = np.random.default_rng(1)
    r = two_sample_t(rng.normal(0, 1, 60), rng.normal(0, 6, 60))
    assert "Welch" in r.nombre and r.detalles["varianzas_iguales_asumidas"] is False
    assert r.gl < 118


def test_ic_contiene_la_diferencia_verdadera():
    rng = np.random.default_rng(2)
    dentro = sum(
        (lambda r: r.ic_inferior <= -3.0 <= r.ic_superior)(
            two_sample_t(rng.normal(10, 2, 60), rng.normal(13, 2, 60)))
        for _ in range(200))
    assert 0.90 <= dentro / 200 <= 1.0       # cobertura nominal 95%


def test_t_pareada_equivale_a_t_de_una_muestra_de_diferencias():
    rng = np.random.default_rng(3)
    a = rng.normal(50, 5, 30)
    b = a - 2 + rng.normal(0, 1, 30)
    r = paired_t(a, b)
    t1, p1 = stats.ttest_1samp(b - a, 0)
    assert r.estadistico == pytest.approx(-t1)   # signo: scipy compara (a, b)
    assert r.p_valor == pytest.approx(p1)
    assert r.efecto_valor == pytest.approx(np.mean(b - a))


def test_pareada_rechaza_tamanos_distintos():
    with pytest.raises(ValueError):
        paired_t([1, 2, 3], [1, 2])


def test_anova_contra_scipy_y_descomposicion_de_sc():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"y": np.r_[rng.normal(0, 1, 25), rng.normal(2, 1, 25), rng.normal(0, 1, 25)],
                       "g": ["A"] * 25 + ["B"] * 25 + ["C"] * 25})
    r = one_way_anova(df, "y", "g")
    f, p = stats.f_oneway(*[g["y"].to_numpy() for _, g in df.groupby("g")])
    assert r["F"] == pytest.approx(f) and r["p_valor"] == pytest.approx(p)
    t = r["tabla_anova"]
    assert t.loc[0, "SC"] + t.loc[1, "SC"] == pytest.approx(t.loc[2, "SC"])
    assert t.loc[0, "gl"] == 2 and t.loc[1, "gl"] == 72
    assert 0 <= r["eta_cuadrado"] <= 1
    assert r["rechaza_h0"] is True


def test_anova_sin_diferencias_no_rechaza():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"y": rng.normal(0, 1, 90), "g": ["A", "B", "C"] * 30})
    assert one_way_anova(df, "y", "g")["rechaza_h0"] is False


def test_tukey_identifica_el_grupo_distinto():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"y": np.r_[rng.normal(0, .5, 40), rng.normal(5, .5, 40), rng.normal(0, .5, 40)],
                       "g": ["A"] * 40 + ["B"] * 40 + ["C"] * 40})
    tk = one_way_anova(df, "y", "g")["tukey"]
    rech = {(r.group1, r.group2): r.reject for r in tk.itertuples()}
    assert rech[("A", "B")] and rech[("B", "C")] and not rech[("A", "C")]


def test_normalidad_detecta_normal_y_no_normal():
    rng = np.random.default_rng(7)
    assert normality(rng.normal(0, 1, 300))["shapiro"]["rechaza_normalidad"] is False
    assert normality(rng.exponential(1, 300))["shapiro"]["rechaza_normalidad"] is True
    assert normality(rng.exponential(1, 300))["anderson_darling"]["rechaza_normalidad"] is True


def test_levene_detecta_varianzas_distintas():
    rng = np.random.default_rng(8)
    assert levene(rng.normal(0, 1, 80), rng.normal(0, 5, 80)).rechaza_h0 is True
    assert levene(rng.normal(0, 1, 80), rng.normal(3, 1, 80)).rechaza_h0 is False
