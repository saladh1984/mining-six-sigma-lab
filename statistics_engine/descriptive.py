"""Estadística descriptiva.

Todas las medidas se calculan sobre los datos entregados. Ninguna se fija por
defecto ni se aproxima.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DescriptiveSummary:
    n: int
    n_missing: int
    media: float
    mediana: float
    moda: float | None
    moda_metodo: str
    desv_std: float          # muestral, ddof=1
    varianza: float          # muestral, ddof=1
    coef_variacion: float | None   # sd/media, adimensional (None si media ~ 0)
    minimo: float
    maximo: float
    rango: float
    q1: float
    q2: float
    q3: float
    iqr: float
    asimetria: float         # sesgo muestral (Fisher-Pearson ajustado)
    curtosis_exceso: float   # exceso sobre la normal (normal = 0)
    error_std_media: float

    def to_dict(self) -> dict:
        return asdict(self)


def describe(x: Sequence[float], moda_decimales: int = 1) -> DescriptiveSummary:
    """Resumen descriptivo completo de una variable continua.

    moda_decimales: la moda de una variable continua no está definida sobre los
    datos crudos (casi toda observación es única). Se reporta la moda de los
    datos redondeados a este número de decimales, y el método queda registrado
    en `moda_método` para que la limitación sea explicita en la interfaz.
    """
    s = pd.Series(x, dtype="float64")
    n_missing = int(s.isna().sum())
    s = s.dropna()
    n = int(s.size)
    if n < 2:
        raise ValueError("Se requieren al menos 2 observaciones válidas.")

    media = float(s.mean())
    sd = float(s.std(ddof=1))

    redondeada = s.round(moda_decimales)
    conteo = redondeada.value_counts()
    if conteo.iloc[0] <= 1:
        moda, metodo = None, f"sin moda (ningun valor se repite al redondear a {moda_decimales} dec.)"
    else:
        moda = float(conteo.index[0])
        metodo = f"valor más frecuente tras redondear a {moda_decimales} decimales"

    q1, q2, q3 = (float(v) for v in np.percentile(s, [25, 50, 75]))

    return DescriptiveSummary(
        n=n,
        n_missing=n_missing,
        media=media,
        mediana=float(s.median()),
        moda=moda,
        moda_metodo=metodo,
        desv_std=sd,
        varianza=float(s.var(ddof=1)),
        coef_variacion=(sd / media) if abs(media) > 1e-12 else None,
        minimo=float(s.min()),
        maximo=float(s.max()),
        rango=float(s.max() - s.min()),
        q1=q1, q2=q2, q3=q3, iqr=q3 - q1,
        asimetria=float(stats.skew(s, bias=False)),
        curtosis_exceso=float(stats.kurtosis(s, fisher=True, bias=False)),
        error_std_media=sd / np.sqrt(n),
    )


def percentiles(x: Sequence[float], ps: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> pd.DataFrame:
    """Percentiles por interpolación lineal (método 7, el de numpy/R por defecto)."""
    s = pd.Series(x, dtype="float64").dropna()
    vals = np.percentile(s, ps)
    return pd.DataFrame({"percentil": list(ps), "valor": vals})


def boxplot_stats(x: Sequence[float], k: float = 1.5) -> dict:
    """Estadísticos del diagrama de cajas y outliers por la regla de Tukey.

    Bigotes: último dato dentro de [Q1 - k*IQR, Q3 + k*IQR]. Un punto fuera NO
    es necesariamente un error: es un candidato a investigación.
    """
    s = pd.Series(x, dtype="float64").dropna()
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    dentro = s[(s >= lo) & (s <= hi)]
    outliers = s[(s < lo) | (s > hi)]
    return {
        "q1": float(q1), "mediana": float(s.median()), "q3": float(q3), "iqr": float(iqr),
        "limite_inferior_tukey": float(lo), "limite_superior_tukey": float(hi),
        "bigote_inferior": float(dentro.min()), "bigote_superior": float(dentro.max()),
        "outliers": outliers.tolist(), "n_outliers": int(outliers.size), "k": k,
    }


def describe_frame(df: pd.DataFrame, columnas: Sequence[str] | None = None) -> pd.DataFrame:
    """Tabla de resumen descriptivo para varias variables numéricas."""
    cols = list(columnas) if columnas else df.select_dtypes("number").columns.tolist()
    filas = []
    for c in cols:
        d = describe(df[c]).to_dict()
        d["variable"] = c
        filas.append(d)
    out = pd.DataFrame(filas)
    return out[["variable"] + [c for c in out.columns if c != "variable"]]
