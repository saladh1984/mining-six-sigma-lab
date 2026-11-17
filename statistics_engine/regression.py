"""Correlación, regresión lineal simple y múltiple, y diagnóstico de residuos.

CORRELACIÓN NO ES CAUSALIDAD (requisito 11 de la especificación)
-----------------------------------------------------------------
Una correlación alta entre distancia y tiempo de ciclo es compatible con:
(a) la distancia causa el tiempo, (b) ambas dependen de un tercer factor (p. ej.
la fase del rajo), (c) coincidencia muestral. Solo un experimento controlado o
un argumento causal externo permite afirmar (a). La regresión ajusta una
asociación condicional, no un mecanismo.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


# ------------------------------------------------------- correlacion -----
def correlation_matrix(df: pd.DataFrame, metodo: str = "pearson",
                       columnas: Sequence[str] | None = None) -> dict:
    """Matriz de correlación con matriz de p-valores asociada.

    pearson  : mide asociación LINEAL; sensible a outliers; supone pares
               aproximadamente normales para el p-valor.
    spearman : correlación de rangos; mide asociación MONOTONA (no solo lineal);
               robusta a outliers y no requiere normalidad.
    """
    cols = list(columnas) if columnas else df.select_dtypes("number").columns.tolist()
    # Una columna repetida haria que df[cols] devuelva un DataFrame por nombre y
    # romperia pearsonr/spearmanr. Se eliminan duplicados conservando el orden.
    cols = list(dict.fromkeys(cols))
    if len(cols) < 2:
        raise ValueError("Se requieren al menos dos variables distintas.")
    d = df[cols].dropna()
    n = d.shape[0]
    if n < 3:
        raise ValueError("Se requieren al menos 3 observaciones completas.")
    f = stats.pearsonr if metodo == "pearson" else stats.spearmanr
    k = len(cols)
    r = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    for i in range(k):
        for j in range(i + 1, k):
            res = f(d[cols[i]], d[cols[j]])
            rv, pv = float(res[0]), float(res[1])
            r.iloc[i, j] = r.iloc[j, i] = rv
            p.iloc[i, j] = p.iloc[j, i] = pv
    return {"metodo": metodo, "n": n, "r": r, "p": p, "r2": r**2,
            "nota": "r^2 aquí es el cuadrado de la correlación bivariada, no el R2 "
                    "de un modelo múltiple."}


def scatter_fit(x, y, alfa: float = 0.05) -> dict:
    """Dispersión con recta de mínimos cuadrados, R2 y bandas de confianza."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = d.shape[0]
    if n < 3:
        raise ValueError("Se requieren al menos 3 pares completos.")
    lr = stats.linregress(d["x"], d["y"])
    xs = np.linspace(d["x"].min(), d["x"].max(), 120)
    ys = lr.intercept + lr.slope * xs
    sxx = ((d["x"] - d["x"].mean()) ** 2).sum()
    resid = d["y"] - (lr.intercept + lr.slope * d["x"])
    s_err = np.sqrt((resid**2).sum() / (n - 2))
    tcrit = stats.t.isf(alfa / 2, n - 2)
    se_mean = s_err * np.sqrt(1 / n + (xs - d["x"].mean()) ** 2 / sxx)
    se_pred = s_err * np.sqrt(1 + 1 / n + (xs - d["x"].mean()) ** 2 / sxx)
    return {"pendiente": float(lr.slope), "intercepto": float(lr.intercept),
            "r": float(lr.rvalue), "r2": float(lr.rvalue**2), "p_valor": float(lr.pvalue),
            "error_std_pendiente": float(lr.stderr), "n": n, "s_residual": float(s_err),
            "x_linea": xs, "y_linea": ys,
            "ic_media_inf": ys - tcrit * se_mean, "ic_media_sup": ys + tcrit * se_mean,
            "ic_pred_inf": ys - tcrit * se_pred, "ic_pred_sup": ys + tcrit * se_pred,
            "ecuacion": f"y = {lr.intercept:.4f} + {lr.slope:.4f} * x"}


# --------------------------------------------------------- regresion -----
def ols(df: pd.DataFrame, y: str, xs: Sequence[str], alfa: float = 0.05) -> dict:
    """Mínimos cuadrados ordinarios (simple o múltiple) con diagnóstico completo."""
    xs = list(dict.fromkeys(xs))
    if y in xs:
        raise ValueError(f"La variable respuesta '{y}' no puede ser también predictor.")
    cols = [y] + list(xs)
    d = df[cols].dropna()
    n, k = d.shape[0], len(xs)
    if n <= k + 1:
        raise ValueError(f"n={n} insuficiente para {k} predictores.")
    X = sm.add_constant(d[list(xs)], has_constant="add")
    m = sm.OLS(d[y], X).fit()

    coef = pd.DataFrame({
        "termino": m.params.index, "coeficiente": m.params.values,
        "error_std": m.bse.values, "t": m.tvalues.values, "p_valor": m.pvalues.values,
        "IC_inf": m.conf_int(alpha=alfa)[0].values, "IC_sup": m.conf_int(alpha=alfa)[1].values,
    })
    terminos = " + ".join(f"{c:+.4f}*{v}" for v, c in m.params.items() if v != "const")
    ecuacion = f"{y} = {m.params.get('const', 0.0):.4f} {terminos}"

    diag = _diagnostico(m, d, list(xs))
    return {
        "modelo": m, "n": n, "k": k, "y": y, "x": list(xs),
        "coeficientes": coef, "ecuacion": ecuacion,
        "r2": float(m.rsquared), "r2_ajustado": float(m.rsquared_adj),
        "f": float(m.fvalue), "f_p_valor": float(m.f_pvalue),
        "s_residual": float(np.sqrt(m.mse_resid)), "aic": float(m.aic), "bic": float(m.bic),
        "ajustados": m.fittedvalues.to_numpy(), "residuos": m.resid.to_numpy(),
        "residuos_estudentizados": m.get_influence().resid_studentized_internal,
        "diagnostico": diag, "resumen_texto": str(m.summary()),
        "advertencias": _advertencias_ols(m, diag, n, k),
    }


def _diagnostico(m, d, xs) -> dict:
    resid = m.resid
    out = {"durbin_watson": float(durbin_watson(resid))}
    sh_w, sh_p = stats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
    out["shapiro_residuos"] = {"estadistico": float(sh_w), "p_valor": float(sh_p)}
    try:
        lm, lm_p, f, f_p = het_breuschpagan(resid, m.model.exog)
        out["breusch_pagan"] = {"LM": float(lm), "p_valor": float(lm_p)}
    except Exception as e:  # pragma: no cover
        out["breusch_pagan"] = {"error": str(e)}
    if len(xs) > 1:
        X = sm.add_constant(d[xs], has_constant="add")
        out["vif"] = pd.DataFrame({
            "variable": xs,
            "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(xs))],
        })
    inf = m.get_influence()
    cooks = inf.cooks_distance[0]
    out["cook_max"] = float(np.max(cooks))
    out["observaciones_influyentes"] = (np.where(cooks > 4 / len(resid))[0] + 1).tolist()
    return out


def _advertencias_ols(m, diag, n, k) -> list[str]:
    a = []
    dw = diag["durbin_watson"]
    if dw < 1.5 or dw > 2.5:
        a.append(f"Durbin-Watson={dw:.3f} (lejos de 2): hay autocorrelación en los "
                 "residuos. Si los datos son una serie temporal, los errores estándar "
                 "y los p-valores están subestimados.")
    bp = diag.get("breusch_pagan", {}).get("p_valor")
    if bp is not None and bp < 0.05:
        a.append(f"Breusch-Pagan p={bp:.4f}: heterocedasticidad. La varianza del error "
                 "no es constante; use errores robustos (HC3) o transforme la respuesta.")
    sp = diag["shapiro_residuos"]["p_valor"]
    if sp == sp and sp < 0.05:
        a.append(f"Residuos no normales (Shapiro p={sp:.4f}). Los IC y p-valores son "
                 "aproximados; con n grande el impacto suele ser menor.")
    if "vif" in diag:
        vmax = diag["vif"]["VIF"].max()
        if vmax > 10:
            a.append(f"VIF máximo={vmax:.1f} > 10: multicolinealidad severa. Los "
                     "coeficientes individuales son inestables y su signo puede invertirse; "
                     "el R2 global sigue siendo válido.")
        elif vmax > 5:
            a.append(f"VIF máximo={vmax:.1f} > 5: multicolinealidad moderada.")
    if diag["observaciones_influyentes"]:
        a.append(f"Observaciones con distancia de Cook > 4/n: "
                 f"{diag['observaciones_influyentes'][:12]}. Revise si son errores de "
                 "registro o condiciones operacionales distintas.")
    if k > 1 and n < 10 * k:
        a.append(f"n={n} con {k} predictores (<10 por predictor): riesgo de sobreajuste. "
                 "Compare R2 y R2 ajustado.")
    a.append("La regresión estima asociación condicional, no causalidad. Un coeficiente "
             "es el cambio esperado en la respuesta por unidad del predictor MANTENIENDO "
             "CONSTANTES los demas, lo que puede no ser operacionalmente posible.")
    return a
