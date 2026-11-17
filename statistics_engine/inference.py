"""Pruebas de hipótesis y verificación de supuestos.

SOBRE EL p-VALOR (requisito 11 de la especificación)
----------------------------------------------------
El p-valor es P(datos al menos tan extremos como los observados | H0 cierta).
NO es la probabilidad de que H0 sea cierta, ni la probabilidad de haberse
equivocado, ni una medida del tamaño del efecto. Un p pequeño con n grande
puede corresponder a una diferencia operacionalmente irrelevante; por eso cada
prueba devuelve además el tamaño del efecto y el intervalo de confianza, que
son lo que se debe usar para decidir.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


@dataclass
class TestResult:
    nombre: str
    hipotesis_nula: str
    estadistico: float
    gl: float | None
    p_valor: float
    alfa: float
    rechaza_h0: bool
    efecto_nombre: str | None = None
    efecto_valor: float | None = None
    ic_inferior: float | None = None
    ic_superior: float | None = None
    detalles: dict = field(default_factory=dict)
    supuestos: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------- normalidad -------
def normality(x: Sequence[float], alfa: float = 0.05) -> dict:
    """Shapiro-Wilk, Anderson-Darling y D'Agostino-Pearson.

    Advertencia: con n grande cualquier desviación trivial de la normalidad se
    vuelve significativa; con n pequeño la prueba casi no tiene potencia. El
    gráfico Q-Q debe pesar tanto como el p-valor.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    n = s.size
    if n < 3:
        raise ValueError("Se requieren al menos 3 observaciones.")
    out = {"n": n, "alfa": alfa}

    w, p = stats.shapiro(s)
    out["shapiro"] = {"estadistico": float(w), "p_valor": float(p), "rechaza_normalidad": bool(p < alfa)}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ad = stats.anderson(s, dist="norm")
    i = list(ad.significance_level).index(5.0) if 5.0 in list(ad.significance_level) else 2
    out["anderson_darling"] = {
        "estadistico": float(ad.statistic),
        "valor_critico_5pct": float(ad.critical_values[i]),
        "rechaza_normalidad": bool(ad.statistic > ad.critical_values[i]),
    }

    if n >= 8:
        k2, pk = stats.normaltest(s)
        out["dagostino"] = {"estadistico": float(k2), "p_valor": float(pk), "rechaza_normalidad": bool(pk < alfa)}

    out["asimetria"] = float(stats.skew(s, bias=False))
    out["curtosis_exceso"] = float(stats.kurtosis(s, fisher=True, bias=False))
    out["qq"] = _qq_points(s)
    out["conclusion"] = (
        "Se rechaza la normalidad" if out["shapiro"]["rechaza_normalidad"]
        else "No hay evidencia suficiente contra la normalidad"
    ) + f" (Shapiro-Wilk, alfa={alfa})."
    if n > 300:
        out["advertencia"] = ("n>300: las pruebas de normalidad detectan desviaciones "
                              "sin relevancia práctica. Priorice el gráfico Q-Q.")
    if n < 20:
        out["advertencia"] = ("n<20: potencia baja. 'No rechazar' no equivale a "
                              "'los datos son normales'.")
    return out


def _qq_points(s: np.ndarray) -> dict:
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    return {"teoricos": osm.tolist(), "observados": osr.tolist(),
            "pendiente": float(slope), "intercepto": float(intercept), "r": float(r)}


def levene(*grupos, alfa: float = 0.05) -> TestResult:
    """Homogeneidad de varianzas (Levene con mediana = Brown-Forsythe)."""
    gs = [pd.Series(g, dtype="float64").dropna().to_numpy() for g in grupos]
    w, p = stats.levene(*gs, center="median")
    return TestResult(
        "Levene (Brown-Forsythe)", "Las varianzas de todos los grupos son iguales",
        float(w), None, float(p), alfa, bool(p < alfa),
        detalles={"varianzas": [float(np.var(g, ddof=1)) for g in gs],
                  "n_por_grupo": [int(g.size) for g in gs]},
        supuestos=["Robusta ante no normalidad gracias al centrado por la mediana."],
    )


# ---------------------------------------------------------- pruebas t -----
def two_sample_t(a, b, alfa: float = 0.05, equal_var: bool | None = None) -> TestResult:
    """Prueba t para dos muestras independientes.

    equal_var=None: se decide con Levene. Si las varianzas difieren se usa
    Welch, que es el valor por defecto recomendado en la práctica moderna.
    Tamaño del efecto: d de Cohen con sd agrupada.
    """
    x = pd.Series(a, dtype="float64").dropna().to_numpy()
    y = pd.Series(b, dtype="float64").dropna().to_numpy()
    n1, n2 = x.size, y.size
    if min(n1, n2) < 2:
        raise ValueError("Cada grupo requiere al menos 2 observaciones.")

    lev = levene(x, y, alfa=alfa)
    if equal_var is None:
        equal_var = not lev.rechaza_h0

    t, p = stats.ttest_ind(x, y, equal_var=equal_var)
    s1, s2 = x.var(ddof=1), y.var(ddof=1)
    if equal_var:
        gl = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / gl
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = np.sqrt(s1 / n1 + s2 / n2)
        gl = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
    diff = float(x.mean() - y.mean())
    tcrit = stats.t.isf(alfa / 2, gl)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    return TestResult(
        "Prueba t de dos muestras " + ("(varianzas iguales)" if equal_var else "(Welch)"),
        "Las medias poblacionales son iguales (mu1 - mu2 = 0)",
        float(t), float(gl), float(p), alfa, bool(p < alfa),
        "Diferencia de medias (d de Cohen)", diff,
        float(diff - tcrit * se), float(diff + tcrit * se),
        detalles={"media_1": float(x.mean()), "media_2": float(y.mean()),
                  "sd_1": float(np.sqrt(s1)), "sd_2": float(np.sqrt(s2)),
                  "n_1": n1, "n_2": n2, "error_std_diferencia": float(se),
                  "d_cohen": float(diff / sp) if sp > 0 else None,
                  "levene_p": lev.p_valor, "varianzas_iguales_asumidas": equal_var},
        supuestos=["Observaciones independientes dentro y entre grupos.",
                   "Normalidad de cada grupo (o n suficiente para el TCL).",
                   "Welch no requiere varianzas iguales; la versión agrupada si."],
    )


def paired_t(antes, despues, alfa: float = 0.05) -> TestResult:
    """Prueba t pareada: se aplica a la diferencia dentro de cada par.

    Usarla solo cuando cada observación de 'antes' y 'después' corresponde a la
    MISMA unidad (el mismo camion, la misma ruta, el mismo turno). Si las
    muestras son independientes, esta prueba es invalida.
    """
    a = pd.Series(antes, dtype="float64")
    b = pd.Series(despues, dtype="float64")
    if a.size != b.size:
        raise ValueError("Las muestras pareadas deben tener el mismo tamaño.")
    par = pd.DataFrame({"a": a, "b": b}).dropna()
    d = (par["b"] - par["a"]).to_numpy()
    n = d.size
    if n < 2:
        raise ValueError("Se requieren al menos 2 pares completos.")
    t, p = stats.ttest_rel(par["a"], par["b"])
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.isf(alfa / 2, n - 1)
    return TestResult(
        "Prueba t pareada", "La diferencia media entre pares es cero",
        float(t), float(n - 1), float(p), alfa, bool(p < alfa),
        "Diferencia media (después - antes)", float(d.mean()),
        float(d.mean() - tcrit * se), float(d.mean() + tcrit * se),
        detalles={"n_pares": n, "sd_diferencias": float(sd),
                  "d_cohen": float(d.mean() / sd) if sd > 0 else None,
                  "media_antes": float(par["a"].mean()), "media_despues": float(par["b"].mean())},
        supuestos=["Las diferencias son aproximadamente normales.",
                   "Los pares son independientes entre si."],
    )


def one_way_anova(datos: pd.DataFrame, variable: str, factor: str, alfa: float = 0.05) -> dict:
    """ANOVA de un factor + Levene + Tukey HSD + eta cuadrado.

    ANOVA responde 'al menos un grupo difiere'. NO dice cual: para eso se usa
    Tukey HSD, que controla el error tipo I de todas las comparaciones.
    """
    df = datos[[variable, factor]].dropna()
    grupos = [g[variable].to_numpy() for _, g in df.groupby(factor, observed=True)]
    niveles = [k for k, _ in df.groupby(factor, observed=True)]
    if len(grupos) < 2:
        raise ValueError("Se requieren al menos 2 niveles del factor.")

    f, p = stats.f_oneway(*grupos)
    k, n = len(grupos), df.shape[0]
    gran = df[variable].mean()
    ss_b = sum(len(g) * (g.mean() - gran) ** 2 for g in grupos)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in grupos)
    ss_t = ss_b + ss_w
    gl_b, gl_w = k - 1, n - k
    eta2 = ss_b / ss_t if ss_t > 0 else np.nan
    omega2 = (ss_b - gl_b * (ss_w / gl_w)) / (ss_t + ss_w / gl_w) if gl_w > 0 else np.nan

    tabla = pd.DataFrame({
        "Fuente": [f"Entre grupos ({factor})", "Dentro de grupos (error)", "Total"],
        "SC": [ss_b, ss_w, ss_t],
        "gl": [gl_b, gl_w, n - 1],
        "CM": [ss_b / gl_b, ss_w / gl_w, np.nan],
        "F": [f, np.nan, np.nan],
        "p": [p, np.nan, np.nan],
    })

    lev = levene(*grupos, alfa=alfa)
    tuk = pairwise_tukeyhsd(df[variable], df[factor], alpha=alfa)
    tukey = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])

    resumen = df.groupby(factor, observed=True)[variable].agg(
        n="count", media="mean", sd=lambda z: z.std(ddof=1)).reset_index()

    return {
        "tabla_anova": tabla, "F": float(f), "p_valor": float(p), "alfa": alfa,
        "rechaza_h0": bool(p < alfa), "gl_entre": gl_b, "gl_dentro": gl_w,
        "eta_cuadrado": float(eta2), "omega_cuadrado": float(omega2),
        "niveles": [str(x) for x in niveles], "resumen_grupos": resumen,
        "levene": lev.to_dict(), "tukey": tukey,
        "sigma_error": float(np.sqrt(ss_w / gl_w)),
        "supuestos": [
            "Independencia de las observaciones (la más importante; no es verificable "
            "con los datos, depende del diseño del muestreo).",
            f"Homogeneidad de varianzas: Levene p={lev.p_valor:.4f} -> "
            + ("se rechaza, ANOVA clasico no es apropiado; use Welch o transforme."
               if lev.rechaza_h0 else "no se rechaza."),
            "Normalidad de los residuos; con grupos balanceados y n moderado el ANOVA "
            "es razonablemente robusto.",
        ],
        "interpretacion_p": (
            f"p={p:.5f} es la probabilidad de observar una F de {f:.3f} o mayor SI todas "
            "las medias fueran iguales. No es la probabilidad de que sean iguales."
        ),
    }
