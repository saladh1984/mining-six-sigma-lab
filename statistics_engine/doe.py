"""Diseño de experimentos factorial 2^k (k = 2 o 3), con o sin replicas.

Codificación: cada factor se lleva a -1 (nivel bajo) y +1 (nivel alto). Esto
hace los efectos comparables entre si aunque las unidades fisicas difieran
(toneladas de balde vs. angulo de posicionamiento).

Efecto principal de A = media(respuesta | A=+1) - media(respuesta | A=-1)
Coeficiente del modelo codificado = efecto / 2

Un efecto de interacción AB significa que el efecto de A DEPENDE del nivel de
B. Cuando una interacción es relevante, los efectos principales no se pueden
interpretar por separado.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def design_matrix(factores: Mapping[str, tuple[float, float]], replicas: int = 1,
                  aleatorizar: bool = False, semilla: int | None = None) -> pd.DataFrame:
    """Matriz experimental 2^k en orden estándar (Yates).

    factores: {"A": (bajo, alto), "B": (bajo, alto), ...}
    Devuelve columnas codificadas (A, B, ...) y reales (A_real, ...).
    """
    nombres = list(factores)
    k = len(nombres)
    if k < 2 or k > 4:
        raise ValueError("Implementado para k = 2, 3 o 4 factores.")
    filas = list(product(*[(-1, 1)] * k))
    # orden estandar de Yates: el primer factor alterna mas rapido
    filas = sorted(filas, key=lambda r: tuple(reversed(r)))
    base = pd.DataFrame(filas, columns=nombres)
    base.insert(0, "corrida", np.arange(1, len(base) + 1))
    df = pd.concat([base.assign(replica=i + 1) for i in range(replicas)], ignore_index=True)
    for f in nombres:
        lo, hi = factores[f]
        df[f"{f}_real"] = np.where(df[f] == -1, lo, hi)
    if aleatorizar:
        df = df.sample(frac=1, random_state=semilla).reset_index(drop=True)
        df.insert(0, "orden_ejecucion", np.arange(1, len(df) + 1))
    return df


def _terminos(nombres: Sequence[str]) -> list[tuple[str, ...]]:
    t: list[tuple[str, ...]] = []
    for orden in range(1, len(nombres) + 1):
        t += list(combinations(nombres, orden))
    return t


def factorial_analysis(df: pd.DataFrame, factores: Sequence[str], respuesta: str,
                       alfa: float = 0.05) -> dict:
    """Efectos, coeficientes, ANOVA (si hay replicas) y método de Lenth (si no)."""
    nombres = list(factores)
    d = df[nombres + [respuesta]].dropna().copy()
    k = len(nombres)
    if not set(np.unique(d[nombres].to_numpy())).issubset({-1, 1, -1.0, 1.0}):
        raise ValueError("Los factores deben estar codificados en -1 y +1.")

    n_corridas = 2**k
    N = d.shape[0]
    replicas = N / n_corridas
    if abs(replicas - round(replicas)) > 1e-9:
        raise ValueError("Diseño desbalanceado: cada corrida debe tener igual número de replicas.")
    replicas = int(round(replicas))

    y = d[respuesta].to_numpy(dtype=float)
    terms = _terminos(nombres)
    filas = []
    for t in terms:
        col = np.ones(N)
        for f in t:
            col = col * d[f].to_numpy(dtype=float)
        contraste = float((col * y).sum())
        efecto = contraste / (replicas * 2 ** (k - 1))
        ss = (N * efecto**2) / 4.0
        filas.append({"termino": ":".join(t), "orden": len(t), "efecto": efecto,
                      "coeficiente": efecto / 2, "contraste": contraste, "SC": ss})
    efectos = pd.DataFrame(filas)
    efectos["pct_contribucion"] = 100 * efectos["SC"] / efectos["SC"].sum()

    media_global = float(y.mean())
    out = {
        "k": k, "n_corridas": n_corridas, "replicas": replicas, "N": N,
        "media_global": media_global, "efectos": efectos, "factores": nombres,
        "respuesta": respuesta,
        "medias_por_corrida": d.groupby(nombres, as_index=False)[respuesta].agg(
            media="mean", n="count", sd=lambda z: z.std(ddof=1)),
    }

    if replicas >= 2:
        out.update(_anova_replicado(d, nombres, respuesta, efectos, replicas, k, N, alfa))
        out["metodo"] = "ANOVA con error puro (diseño replicado)"
    else:
        out.update(_lenth(efectos, alfa))
        out["metodo"] = ("Método de Lenth (diseño sin replicas: no hay estimación "
                         "independiente del error, se usa la pseudo-desviación estándar)")

    coef = {"Intercepto": media_global}
    coef.update(dict(zip(efectos["termino"], efectos["coeficiente"])))
    out["ecuacion_codificada"] = (
        f"{respuesta} = {media_global:.4f} " +
        " ".join(f"{c:+.4f}*{t}" for t, c in list(coef.items())[1:])
    )
    out["coeficientes"] = coef
    out["advertencias"] = _advertencias_doe(out, replicas, k)
    return out


def _anova_replicado(d, nombres, respuesta, efectos, replicas, k, N, alfa) -> dict:
    medias = d.groupby(nombres)[respuesta].transform("mean")
    ss_error = float(((d[respuesta] - medias) ** 2).sum())
    gl_error = N - 2**k
    cm_error = ss_error / gl_error
    tab = efectos[["termino", "SC"]].copy()
    tab["gl"] = 1
    tab["CM"] = tab["SC"]
    tab["F"] = tab["CM"] / cm_error
    tab["p"] = stats.f.sf(tab["F"], 1, gl_error)
    tab["significativo"] = tab["p"] < alfa
    total = pd.DataFrame([
        {"termino": "Error puro", "SC": ss_error, "gl": gl_error, "CM": cm_error,
         "F": np.nan, "p": np.nan, "significativo": False},
        {"termino": "Total", "SC": float(((d[respuesta] - d[respuesta].mean()) ** 2).sum()),
         "gl": N - 1, "CM": np.nan, "F": np.nan, "p": np.nan, "significativo": False},
    ])
    tabla = pd.concat([tab, total], ignore_index=True)
    ss_mod = float(efectos["SC"].sum())
    ss_tot = ss_mod + ss_error
    return {"tabla_anova": tabla, "gl_error": gl_error, "cm_error": cm_error,
            "s_error": float(np.sqrt(cm_error)),
            "r2": ss_mod / ss_tot if ss_tot > 0 else np.nan,
            "significativos": tab.loc[tab["significativo"], "termino"].tolist()}


def _lenth(efectos: pd.DataFrame, alfa: float) -> dict:
    """Pseudo-error de Lenth (1989) para diseños sin replicas."""
    e = efectos["efecto"].to_numpy()
    m = e.size
    s0 = 1.5 * np.median(np.abs(e))
    sel = np.abs(e) < 2.5 * s0
    pse = 1.5 * np.median(np.abs(e[sel])) if sel.any() else s0
    escala = float(np.max(np.abs(e)))
    degenerado = pse <= 1e-12 * max(escala, 1.0)
    if degenerado:
        # Mas de la mitad de los efectos son exactamente cero (datos sin ruido o
        # redondeados). La PSE colapsa y todo efecto no nulo pareceria activo.
        pse = float(np.std(e, ddof=1)) if e.size > 1 else escala
    gl = m / 3
    tcrit = float(stats.t.isf(alfa / 2, max(gl, 1)))
    me = tcrit * pse
    gamma = (1 + 0.95 ** (1 / m)) / 2
    sme = float(stats.t.isf(1 - gamma, max(gl, 1))) * pse
    tab = efectos.copy()
    tab["|efecto|"] = np.abs(tab["efecto"])
    tab["supera_ME"] = tab["|efecto|"] > me
    tab["supera_SME"] = tab["|efecto|"] > sme
    return {"pse_lenth": float(pse), "pse_degenerada": bool(degenerado), "margen_error_ME": float(me),
            "margen_simultaneo_SME": float(sme), "tabla_lenth": tab,
            "significativos": tab.loc[tab["supera_ME"], "termino"].tolist()}


def main_effect_plot_data(df: pd.DataFrame, factores: Sequence[str], respuesta: str) -> pd.DataFrame:
    filas = []
    for f in factores:
        for nivel in (-1, 1):
            sub = df.loc[df[f] == nivel, respuesta]
            filas.append({"factor": f, "nivel": "Bajo (-1)" if nivel == -1 else "Alto (+1)",
                          "codigo": nivel, "media": float(sub.mean()), "n": int(sub.size)})
    return pd.DataFrame(filas)


def interaction_plot_data(df: pd.DataFrame, f1: str, f2: str, respuesta: str) -> pd.DataFrame:
    g = df.groupby([f1, f2], as_index=False)[respuesta].mean()
    g[f1] = g[f1].map({-1: "Bajo (-1)", 1: "Alto (+1)"})
    g[f2] = g[f2].map({-1: "Bajo (-1)", 1: "Alto (+1)"})
    return g.rename(columns={respuesta: "media"})


def _advertencias_doe(out, replicas, k) -> list[str]:
    a = []
    if out.get("pse_degenerada"):
        a.append("La pseudo-desviación de Lenth colapso a cero porque más de la mitad "
                 "de los efectos son exactamente nulos (datos sin ruido o muy "
                 "redondeados). Se sustituyo por la desviación estándar de los efectos; "
                 "la significancia reportada es solo orientativa.")
    if replicas == 1:
        a.append("Diseño sin replicas: no existe estimación independiente del error "
                 "experimental. El método de Lenth supone que la mayoría de los efectos "
                 "son nulos (principio de escasez de efectos). No hay p-valores exactos.")
        if out["k"] == 3:
            a.append("Con k=3 solo hay 7 efectos estimables y los grados de libertad "
                     "efectivos de Lenth son ~2.3: la potencia es muy baja y solo se "
                     "detectan efectos dominantes. Si varios factores importan a la vez, "
                     "replique el diseño para obtener un error puro.")
    sig = out.get("significativos", [])
    inter = [t for t in sig if ":" in t]
    if inter:
        a.append(f"Interacción(es) relevante(s): {', '.join(inter)}. Los efectos "
                 "principales involucrados NO deben interpretarse de forma aislada: "
                 "el efecto de un factor cambia según el nivel del otro.")
    a.append("Un diseño 2^k solo detecta efectos LINEALES entre los dos niveles "
             "elegidos. No puede detectar curvatura; para eso se requieren puntos "
             "centrales o un diseño de superficie de respuesta.")
    a.append("Las conclusiones son válidas únicamente dentro del rango experimentado. "
             "Extrapolar fuera de los niveles bajo y alto no está respaldado.")
    a.append("La aleatorización del orden de ejecución es lo que protege contra "
             "factores no controlados que derivan en el tiempo (temperatura, turno, "
             "desgaste). Sin aleatorizar, los efectos pueden estar confundidos.")
    return a
