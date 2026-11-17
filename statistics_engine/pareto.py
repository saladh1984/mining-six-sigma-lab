"""Diagrama de Pareto y matriz causa-efecto.

El principio de Pareto es una REGULARIDAD EMPIRICA, no una ley. En algunos
procesos la distribución es plana y no existe un "pocos vitales": en ese caso
el gráfico lo muestra y la conclusión correcta es que no hay una causa
dominante, no forzar un corte en 80%.

Cuidado con la unidad de medida: ordenar por FRECUENCIA de ocurrencia y
ordenar por IMPACTO (minutos perdidos, toneladas no movidas, costo) suele dar
rankings distintos. El Pareto relevante para decidir es el de impacto.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def pareto(categorias: Sequence[str], valores: Sequence[float],
           corte: float = 80.0, etiqueta_valor: str = "Frecuencia") -> dict:
    """Tabla de Pareto ordenada con porcentaje acumulado."""
    df = pd.DataFrame({"categoria": list(categorias), "valor": np.asarray(valores, dtype=float)})
    if (df["valor"] < 0).any():
        raise ValueError("Los valores de un Pareto deben ser no negativos.")
    df = df.groupby("categoria", as_index=False)["valor"].sum()
    df = df.sort_values("valor", ascending=False).reset_index(drop=True)
    total = float(df["valor"].sum())
    if total <= 0:
        raise ValueError("El total es cero: no hay nada que priorizar.")
    df["pct"] = 100 * df["valor"] / total
    df["pct_acumulado"] = df["pct"].cumsum()
    df["rank"] = np.arange(1, len(df) + 1)
    df["vital"] = df["pct_acumulado"] <= corte
    if not df["vital"].any():
        df.loc[0, "vital"] = True
    else:  # incluir la categoria que cruza el corte
        i = int(df["vital"].sum())
        if i < len(df):
            df.loc[i, "vital"] = True

    vitales = df.loc[df["vital"], "categoria"].tolist()
    pct_vitales = float(df.loc[df["vital"], "pct"].sum())
    n_vit, n_tot = len(vitales), len(df)
    concentrado = (n_vit / n_tot) <= 0.5

    return {
        "tabla": df, "total": total, "etiqueta_valor": etiqueta_valor, "corte": corte,
        "pocos_vitales": vitales, "n_vitales": n_vit, "n_categorias": n_tot,
        "pct_cubierto_por_vitales": pct_vitales,
        "concentrado": concentrado,
        "lectura": (
            f"{n_vit} de {n_tot} categorías concentran el {pct_vitales:.1f}% del total "
            f"de {etiqueta_valor.lower()}."
            + ("" if concentrado else
               " La distribución NO está concentrada: no existe un grupo pequeño de "
               "causas dominantes, de modo que atacar las primeras categorías rendirá "
               "poco. Revise si la clasificación de categorías es la adecuada.")
        ),
    }


def matriz_causa_efecto(causas: Sequence[str], salidas: Sequence[str],
                        pesos_salida: Sequence[float],
                        calificaciones: Mapping[str, Sequence[float]]) -> dict:
    """Matriz causa-efecto (C&E / XY matrix) ponderada.

    pesos_salida   : importancia de cada salida para el cliente (1-10).
    calificaciones : {causa: [relación con cada salida]} en escala 0,1,3,9.
    Puntaje de la causa = sum_j peso_j * relación_ij
    """
    w = np.asarray(pesos_salida, dtype=float)
    if w.size != len(salidas):
        raise ValueError("pesos_salida debe tener un peso por salida.")
    filas = []
    for c in causas:
        rel = np.asarray(calificaciones[c], dtype=float)
        if rel.size != len(salidas):
            raise ValueError(f"La causa '{c}' debe calificarse contra todas las salidas.")
        filas.append({"causa": c, **{s: r for s, r in zip(salidas, rel)},
                      "puntaje": float((w * rel).sum())})
    df = pd.DataFrame(filas).sort_values("puntaje", ascending=False).reset_index(drop=True)
    tot = df["puntaje"].sum()
    df["pct"] = 100 * df["puntaje"] / tot if tot > 0 else np.nan
    df["pct_acumulado"] = df["pct"].cumsum()
    return {"tabla": df, "pesos": dict(zip(salidas, w.tolist())),
            "nota": "El puntaje ordena las causas por su relación declarada con las "
                    "salidas críticas. Es una priorización basada en juicio experto, "
                    "no evidencia estadística: sirve para decidir que analizar, no "
                    "para concluir."}
