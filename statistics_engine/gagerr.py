"""Gage R&R cruzado por el método ANOVA (AIAG MSA 4a edición).

QUE MIDE
--------
Antes de analizar la variabilidad de un proceso hay que saber cuanta de esa
variabilidad proviene del SISTEMA DE MEDICIÓN y no del proceso. Si el
instrumento aporta el 40% de la variación observada, buena parte de lo que
parece variabilidad del proceso es ruido de medición.

    Variación total observada = Variación del proceso (piezas)
                              + Repetibilidad (mismo operador, mismo item)
                              + Reproducibilidad (entre operadores)

Repetibilidad  : variación del equipo. Un mismo operador mide el mismo item
                 varias veces y no obtiene el mismo valor.
Reproducibilidad: variación del operador (y de su interacción con la pieza).
R&R            : repetibilidad + reproducibilidad.

DISEÑO REQUERIDO
----------------
Cruzado y balanceado: cada operador mide CADA pieza, r veces. Lo típico es
10 piezas x 3 operadores x 2-3 replicas. No se puede ejecutar un Gage R&R
sobre una serie temporal cualquiera: requiere este diseño específico.

CRITERIOS AIAG (sobre %Variación de estudio, no sobre %Contribución)
--------------------------------------------------------------------
    %R&R < 10%        sistema de medición aceptable
    10% <= %R&R <= 30% condicionalmente aceptable según criticidad y costo
    %R&R > 30%        inaceptable
    ndc >= 5          número de categorías distinguibles suficiente
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def gage_rr_anova(df: pd.DataFrame, pieza: str, operador: str, medicion: str,
                  tolerancia: float | None = None, alfa_interaccion: float = 0.25) -> dict:
    """Gage R&R cruzado por ANOVA de dos factores con interacción.

    Modelo:  y_ijk = mu + P_i + O_j + (PO)_ij + e_ijk
    Si la interacción no es significativa (p > alfa_interacción, AIAG usa 0.25),
    se agrupa con el error y se reestima el modelo sin interacción.
    """
    d = df[[pieza, operador, medicion]].dropna()
    p = d[pieza].nunique()
    o = d[operador].nunique()
    conteo = d.groupby([pieza, operador]).size()
    if conteo.nunique() != 1:
        raise ValueError("El diseño debe ser balanceado: igual número de replicas "
                         "en cada combinación pieza x operador.")
    r = int(conteo.iloc[0])
    if r < 2:
        raise ValueError("Se requieren al menos 2 replicas para estimar la repetibilidad.")
    if p < 2 or o < 2:
        raise ValueError("Se requieren al menos 2 piezas y 2 operadores.")

    y = d[medicion].to_numpy(dtype=float)
    N = y.size
    gran = y.mean()
    mp = d.groupby(pieza)[medicion].mean()
    mo = d.groupby(operador)[medicion].mean()
    mpo = d.groupby([pieza, operador])[medicion].mean()

    ss_p = o * r * float(((mp - gran) ** 2).sum())
    ss_o = p * r * float(((mo - gran) ** 2).sum())
    ss_po = r * float(sum((mpo.loc[(i, j)] - mp[i] - mo[j] + gran) ** 2
                          for i in mp.index for j in mo.index))
    ss_e = float(sum((row[medicion] - mpo.loc[(row[pieza], row[operador])]) ** 2
                     for _, row in d.iterrows()))
    ss_t = float(((y - gran) ** 2).sum())

    gl_p, gl_o, gl_po, gl_e = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    ms_p, ms_o, ms_po, ms_e = ss_p / gl_p, ss_o / gl_o, ss_po / gl_po, ss_e / gl_e

    f_po = ms_po / ms_e
    p_po = float(stats.f.sf(f_po, gl_po, gl_e))
    interaccion_significativa = p_po <= alfa_interaccion

    tabla = pd.DataFrame({
        "Fuente": [pieza, operador, f"{pieza} x {operador}", "Repetibilidad (error)", "Total"],
        "SC": [ss_p, ss_o, ss_po, ss_e, ss_t],
        "gl": [gl_p, gl_o, gl_po, gl_e, N - 1],
        "CM": [ms_p, ms_o, ms_po, ms_e, np.nan],
        "F": [ms_p / ms_po, ms_o / ms_po, f_po, np.nan, np.nan],
        "p": [float(stats.f.sf(ms_p / ms_po, gl_p, gl_po)),
              float(stats.f.sf(ms_o / ms_po, gl_o, gl_po)), p_po, np.nan, np.nan],
    })

    if interaccion_significativa:
        v_rep = ms_e
        v_po = max(0.0, (ms_po - ms_e) / r)
        v_op = max(0.0, (ms_o - ms_po) / (p * r))
        v_pieza = max(0.0, (ms_p - ms_po) / (o * r))
        modelo = "con interacción pieza x operador"
    else:
        ms_e2 = (ss_po + ss_e) / (gl_po + gl_e)
        v_rep = ms_e2
        v_po = 0.0
        v_op = max(0.0, (ms_o - ms_e2) / (p * r))
        v_pieza = max(0.0, (ms_p - ms_e2) / (o * r))
        modelo = (f"sin interacción (p={p_po:.3f} > {alfa_interaccion}; se agrupa "
                  "con el error según AIAG)")

    v_reprod = v_op + v_po
    v_rr = v_rep + v_reprod
    v_total = v_rr + v_pieza

    comp = pd.DataFrame({
        "Fuente": ["Gage R&R total", "  Repetibilidad (equipo)", "  Reproducibilidad",
                   f"    {operador}", f"    {pieza} x {operador}",
                   f"Variación de {pieza} (proceso)", "Variación total"],
        "Varianza": [v_rr, v_rep, v_reprod, v_op, v_po, v_pieza, v_total],
    })
    comp["DesvEst"] = np.sqrt(comp["Varianza"])
    comp["%Contribucion"] = 100 * comp["Varianza"] / v_total if v_total > 0 else np.nan
    comp["VarEstudio(6s)"] = 6 * comp["DesvEst"]
    comp["%VariacionEstudio"] = 100 * comp["DesvEst"] / np.sqrt(v_total) if v_total > 0 else np.nan
    if tolerancia:
        comp["%Tolerancia(P/T)"] = 100 * comp["VarEstudio(6s)"] / tolerancia

    pct_rr = float(comp.loc[0, "%VariacionEstudio"])
    ndc = float(1.41 * np.sqrt(v_pieza) / np.sqrt(v_rr)) if v_rr > 0 else np.inf

    if pct_rr < 10:
        veredicto = "ACEPTABLE (%R&R < 10%): el sistema de medición es adecuado."
    elif pct_rr <= 30:
        veredicto = ("CONDICIONAL (10% <= %R&R <= 30%): aceptable según la criticidad "
                     "de la aplicación y el costo de mejorar el sistema.")
    else:
        veredicto = ("INACEPTABLE (%R&R > 30%): el sistema de medición no sirve para "
                     "controlar este proceso. Corregirlo ANTES de analizar la "
                     "variabilidad del proceso.")

    return {
        "n_piezas": p, "n_operadores": o, "n_replicas": r, "N": N,
        "tabla_anova": tabla, "componentes": comp, "modelo": modelo,
        "interaccion_significativa": interaccion_significativa, "p_interaccion": p_po,
        "pct_rr_estudio": pct_rr,
        "pct_rr_contribucion": float(comp.loc[0, "%Contribucion"]),
        "ndc": int(np.floor(ndc)) if np.isfinite(ndc) else None,
        "ndc_exacto": ndc, "tolerancia": tolerancia, "veredicto": veredicto,
        "notas": [
            "%Contribución se calcula sobre VARIANZAS y %VariacionEstudio sobre "
            "DESVIACIONES ESTÁNDAR: no son comparables entre si. El criterio AIAG "
            "de 10/30% aplica a %VariacionEstudio.",
            f"ndc = 1.41 x (sigma_pieza / sigma_R&R) = {ndc:.2f}. Se requiere ndc >= 5 "
            "para que el sistema distinga categorías útiles dentro del rango de piezas.",
            "El %R&R depende de la variación de las PIEZAS elegidas. Si se seleccionan "
            "piezas muy similares entre si, el %R&R sale artificialmente alto aunque el "
            "instrumento sea bueno. Las piezas deben cubrir el rango real del proceso.",
            "Este análisis evalua PRECISIÓN (repetibilidad y reproducibilidad). No "
            "evalua EXACTITUD (sesgo, linealidad, estabilidad), que requieren un patron "
            "de referencia trazable.",
        ],
    }
