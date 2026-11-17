"""Cartas de control (SPC) y reglas de patrones no aleatorios.

LÍMITES DE CONTROL != LÍMITES DE ESPECIFICACIÓN
-----------------------------------------------
Los límites de control se CALCULAN A PARTIR DE LOS DATOS y describen lo que el
proceso hace (su voz). Los límites de especificación los fija el cliente o el
requisito técnico y describen lo que el proceso debería hacer (la voz del
cliente). Nunca se dibujan especificaciones sobre una carta de medias: la media
de subgrupos tiene menos dispersión que las observaciones individuales, así que
compararlas con la especificación produce un optimismo falso.

Cartas implementadas
--------------------
Variables (datos continuos)
    xbar_r   : medias y rangos            (subgrupos 2 <= n <= 10 aprox.)
    xbar_s   : medias y desv. estándar    (subgrupos n >= 9, o n variable)
    i_mr     : individuales y rango movil (n = 1)
Atributos (datos de conteo)
    p        : proporción de defectuosos,  n variable o constante  (binomial)
    np_chart : número de defectuosos,      n constante             (binomial)
    c        : número de defectos por unidad de inspección constante (Poisson)
    u        : defectos por unidad,        area de oportunidad variable (Poisson)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import a2, a3, b3, b4, c4, d2, d3_lower, d4_upper

NELSON_DESC = {
    1: "1 punto fuera de los límites de control (más de 3 sigma de la línea central)",
    2: "9 puntos consecutivos del mismo lado de la línea central",
    3: "6 puntos consecutivos en ascenso o descenso continuo",
    4: "14 puntos consecutivos alternando arriba y abajo",
    5: "2 de 3 puntos consecutivos más alla de 2 sigma, del mismo lado",
    6: "4 de 5 puntos consecutivos más alla de 1 sigma, del mismo lado",
    7: "15 puntos consecutivos dentro de 1 sigma (ambos lados): estratificación",
    8: "8 puntos consecutivos más alla de 1 sigma (ambos lados): mezcla",
}


@dataclass
class ChartResult:
    nombre: str
    etiqueta_y: str
    puntos: np.ndarray
    centro: np.ndarray            # linea central (vector, por si es variable)
    lci: np.ndarray               # limite de control inferior
    lcs: np.ndarray               # limite de control superior
    sigma: float | None           # sigma del estadistico graficado (si constante)
    violaciones: dict[int, list[int]] = field(default_factory=dict)
    n_subgrupo: np.ndarray | None = None
    notas: list[str] = field(default_factory=list)

    @property
    def fuera_de_control(self) -> list[int]:
        """Índices (base 0) con al menos una señal de cualquier regla."""
        return sorted({i for idxs in self.violaciones.values() for i in idxs})

    @property
    def en_control(self) -> bool:
        return not self.fuera_de_control

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "subgrupo": np.arange(1, len(self.puntos) + 1),
            "valor": self.puntos, "LC": self.centro, "LCI": self.lci, "LCS": self.lcs,
        })
        if self.n_subgrupo is not None:
            df.insert(1, "n", self.n_subgrupo)
        for r in sorted(NELSON_DESC):
            df[f"regla_{r}"] = df.index.isin(self.violaciones.get(r, []))
        df["senal"] = df[[f"regla_{r}" for r in sorted(NELSON_DESC)]].any(axis=1)
        return df


# ------------------------------------------------------------- reglas ------
def nelson_rules(puntos, centro, sigma, reglas: Sequence[int] | None = None) -> dict[int, list[int]]:
    """Reglas de Nelson (1984). Las reglas de zona (5-8) requieren sigma.

    Las reglas de zona suponen límites simetricos y aproximadamente normales:
    son válidas en cartas Xbar e I, pero NO en R, S, p, np, c ni u, cuyos
    estadísticos son asimétricos. En esas cartas se aplican solo 1, 2, 3 y 4.
    """
    x = np.asarray(puntos, dtype=float)
    cl = np.asarray(centro, dtype=float)
    n = x.size
    activas = list(reglas) if reglas is not None else ([1, 2, 3, 4] if sigma is None else list(range(1, 9)))
    v: dict[int, list[int]] = {}

    def add(r, idxs):
        if idxs:
            v[r] = sorted(set(idxs))

    if 1 in activas and sigma is not None:
        add(1, np.where(np.abs(x - cl) > 3 * sigma + 1e-12)[0].tolist())

    lado = np.sign(x - cl)

    if 2 in activas:
        hit = []
        for i in range(n - 8):
            w = lado[i:i + 9]
            if np.all(w > 0) or np.all(w < 0):
                hit += list(range(i, i + 9))
        add(2, hit)

    if 3 in activas:
        hit = []
        d = np.diff(x)
        for i in range(n - 6):
            w = d[i:i + 6]
            if np.all(w > 0) or np.all(w < 0):
                hit += list(range(i, i + 7))
        add(3, hit)

    if 4 in activas:
        hit = []
        d = np.sign(np.diff(x))
        for i in range(n - 14):
            w = d[i:i + 14]
            if np.all(w != 0) and np.all(w[:-1] * w[1:] < 0):
                hit += list(range(i, i + 15))
        add(4, hit)

    if sigma is not None and sigma > 0:
        z = (x - cl) / sigma
        if 5 in activas:
            hit = []
            for i in range(n - 2):
                w = z[i:i + 3]
                if (w > 2).sum() >= 2 or (w < -2).sum() >= 2:
                    hit += list(range(i, i + 3))
            add(5, hit)
        if 6 in activas:
            hit = []
            for i in range(n - 4):
                w = z[i:i + 5]
                if (w > 1).sum() >= 4 or (w < -1).sum() >= 4:
                    hit += list(range(i, i + 5))
            add(6, hit)
        if 7 in activas:
            hit = []
            for i in range(n - 14):
                if np.all(np.abs(z[i:i + 15]) < 1):
                    hit += list(range(i, i + 15))
            add(7, hit)
        if 8 in activas:
            hit = []
            for i in range(n - 7):
                if np.all(np.abs(z[i:i + 8]) > 1):
                    hit += list(range(i, i + 8))
            add(8, hit)
    return v


# ------------------------------------------------- cartas de variables -----
def _as_subgroup_matrix(datos) -> np.ndarray:
    m = np.asarray(datos, dtype=float)
    if m.ndim != 2:
        raise ValueError("Se espera una matriz k subgrupos x n observaciones.")
    return m


def xbar_r(datos) -> tuple[ChartResult, ChartResult]:
    """Carta Xbar-R. sigma_hat = Rbar/d2(n). Devuelve (carta Xbar, carta R).

    Se evalua PRIMERO la carta R: si la dispersión no es estable, los límites
    de la carta de medias se calculan con un Rbar contaminado y no son válidos.
    """
    m = _as_subgroup_matrix(datos)
    k, n = m.shape
    medias, rangos = m.mean(axis=1), m.max(axis=1) - m.min(axis=1)
    xbarbar, rbar = float(medias.mean()), float(rangos.mean())
    sigma_hat = rbar / d2(n)

    r_chart = ChartResult(
        "Carta R", "Rango del subgrupo", rangos,
        np.full(k, rbar), np.full(k, d3_lower(n) * rbar), np.full(k, d4_upper(n) * rbar),
        sigma=None, n_subgrupo=np.full(k, n),
        notas=[f"Rbar={rbar:.4f}; D3={d3_lower(n):.3f}, D4={d4_upper(n):.3f} para n={n}.",
               "Los límites de R son asimétricos: no se aplican las reglas de zona 5-8."],
    )
    r_chart.violaciones = _limit_rules(rangos, r_chart.lci, r_chart.lcs) | nelson_rules(rangos, r_chart.centro, None)

    x_chart = ChartResult(
        "Carta Xbar", "Media del subgrupo", medias,
        np.full(k, xbarbar), np.full(k, xbarbar - a2(n) * rbar), np.full(k, xbarbar + a2(n) * rbar),
        sigma=sigma_hat / np.sqrt(n), n_subgrupo=np.full(k, n),
        notas=[f"Xbarbar={xbarbar:.4f}; A2={a2(n):.3f} para n={n}.",
               f"sigma del proceso estimada = Rbar/d2 = {sigma_hat:.4f}.",
               "No superponga límites de especificación sobre esta carta."],
    )
    x_chart.violaciones = nelson_rules(medias, x_chart.centro, x_chart.sigma)
    return x_chart, r_chart


def xbar_s(datos) -> tuple[ChartResult, ChartResult]:
    """Carta Xbar-S. sigma_hat = Sbar/c4(n). Preferible a Xbar-R con n >= 9."""
    m = _as_subgroup_matrix(datos)
    k, n = m.shape
    medias, sds = m.mean(axis=1), m.std(axis=1, ddof=1)
    xbarbar, sbar = float(medias.mean()), float(sds.mean())
    sigma_hat = sbar / c4(n)

    s_chart = ChartResult(
        "Carta S", "Desviación estándar del subgrupo", sds,
        np.full(k, sbar), np.full(k, b3(n) * sbar), np.full(k, b4(n) * sbar),
        sigma=None, n_subgrupo=np.full(k, n),
        notas=[f"Sbar={sbar:.4f}; B3={b3(n):.3f}, B4={b4(n):.3f} para n={n}."],
    )
    s_chart.violaciones = _limit_rules(sds, s_chart.lci, s_chart.lcs) | nelson_rules(sds, s_chart.centro, None)

    x_chart = ChartResult(
        "Carta Xbar (S)", "Media del subgrupo", medias,
        np.full(k, xbarbar), np.full(k, xbarbar - a3(n) * sbar), np.full(k, xbarbar + a3(n) * sbar),
        sigma=sigma_hat / np.sqrt(n), n_subgrupo=np.full(k, n),
        notas=[f"Xbarbar={xbarbar:.4f}; A3={a3(n):.3f} para n={n}.",
               f"sigma del proceso estimada = Sbar/c4 = {sigma_hat:.4f}."],
    )
    x_chart.violaciones = nelson_rules(medias, x_chart.centro, x_chart.sigma)
    return x_chart, s_chart


def i_mr(x) -> tuple[ChartResult, ChartResult]:
    """Carta de individuales y rango movil. sigma_hat = MRbar/1.128.

    Se usa cuando no hay subgrupos racionales (una medición por periodo). Es
    sensible a la no normalidad, más que la carta Xbar, porque no se beneficia
    del teorema central del límite.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    k = s.size
    if k < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    mr = np.abs(np.diff(s))
    mrbar = float(mr.mean())
    sigma_hat = mrbar / d2(2)
    xbar = float(s.mean())

    i_chart = ChartResult(
        "Carta I (individuales)", "Observación individual", s,
        np.full(k, xbar), np.full(k, xbar - 3 * sigma_hat), np.full(k, xbar + 3 * sigma_hat),
        sigma=sigma_hat,
        notas=[f"Xbar={xbar:.4f}; MRbar={mrbar:.4f}; sigma=MRbar/1.128={sigma_hat:.4f}.",
               "Más sensible a la no normalidad que la carta Xbar: verifique la distribución."],
    )
    i_chart.violaciones = nelson_rules(s, i_chart.centro, sigma_hat)

    mr_chart = ChartResult(
        "Carta MR (rango movil)", "Rango movil (orden 2)", mr,
        np.full(k - 1, mrbar), np.full(k - 1, d3_lower(2) * mrbar), np.full(k - 1, d4_upper(2) * mrbar),
        sigma=None,
        notas=["Los rangos moviles están autocorrelacionados por construcción: "
               "las reglas de rachas sobre la carta MR producen falsas alarmas. "
               "Use principalmente la regla 1."],
    )
    mr_chart.violaciones = _limit_rules(mr, mr_chart.lci, mr_chart.lcs)
    return i_chart, mr_chart


# ------------------------------------------------- cartas de atributos -----
def p_chart(defectuosos, n) -> ChartResult:
    """Carta p: proporción de unidades defectuosas. Modelo binomial.

    Límites variables si n cambia:  pbar +/- 3*sqrt(pbar(1-pbar)/n_i)
    pbar = total defectuosos / total inspeccionados  (NO el promedio de las p_i).
    """
    d = np.asarray(defectuosos, dtype=float)
    nn = np.asarray(n, dtype=float) * np.ones_like(d)
    if np.any(d > nn) or np.any(d < 0):
        raise ValueError("Los defectuosos deben estar entre 0 y n.")
    p = d / nn
    pbar = float(d.sum() / nn.sum())
    se = np.sqrt(pbar * (1 - pbar) / nn)
    res = ChartResult(
        "Carta p", "Proporción defectuosa", p,
        np.full(p.size, pbar), np.maximum(0.0, pbar - 3 * se), np.minimum(1.0, pbar + 3 * se),
        sigma=None, n_subgrupo=nn,
        notas=[f"pbar = {pbar:.5f} = {int(d.sum())}/{int(nn.sum())}.",
               "Supuestos: unidades independientes y probabilidad de defecto constante. "
               "Si la variabilidad observada excede la binomial (sobredispersión), "
               "los límites son demasiado estrechos y aparecen falsas alarmas."],
    )
    if np.any(nn * pbar < 5):
        res.notas.append("Aviso: n*pbar < 5 en algun subgrupo; la aproximación normal "
                         "de los límites 3-sigma es pobre.")
    res.violaciones = _limit_rules(p, res.lci, res.lcs) | nelson_rules(p, res.centro, None)
    return res


def np_chart(defectuosos, n: int) -> ChartResult:
    """Carta np: número de defectuosos, con n CONSTANTE."""
    d = np.asarray(defectuosos, dtype=float)
    k = d.size
    pbar = float(d.sum() / (n * k))
    npbar = n * pbar
    se = np.sqrt(npbar * (1 - pbar))
    res = ChartResult(
        "Carta np", "Número de defectuosos", d,
        np.full(k, npbar), np.full(k, max(0.0, npbar - 3 * se)), np.full(k, npbar + 3 * se),
        sigma=None, n_subgrupo=np.full(k, n),
        notas=[f"pbar={pbar:.5f}; n*pbar={npbar:.3f}.",
               "Requiere tamaño de muestra constante. Si n varía, use la carta p."],
    )
    res.violaciones = _limit_rules(d, res.lci, res.lcs) | nelson_rules(d, res.centro, None)
    return res


def c_chart(defectos) -> ChartResult:
    """Carta c: número de defectos por unidad de inspección CONSTANTE. Poisson."""
    c = np.asarray(defectos, dtype=float)
    k = c.size
    cbar = float(c.mean())
    se = np.sqrt(cbar)
    res = ChartResult(
        "Carta c", "Número de defectos", c,
        np.full(k, cbar), np.full(k, max(0.0, cbar - 3 * se)), np.full(k, cbar + 3 * se),
        sigma=None,
        notas=[f"cbar={cbar:.4f}; sigma=sqrt(cbar)={se:.4f}.",
               "Supuesto Poisson: media = varianza, defectos independientes y area de "
               "oportunidad constante. Si la varianza excede la media, hay "
               "sobredispersión y el modelo no aplica."],
    )
    if cbar < 5:
        res.notas.append("Aviso: cbar < 5; la Poisson es marcadamente asimétrica y los "
                         "límites 3-sigma simetricos son aproximados.")
    res.violaciones = _limit_rules(c, res.lci, res.lcs) | nelson_rules(c, res.centro, None)
    return res


def u_chart(defectos, unidades) -> ChartResult:
    """Carta u: defectos por unidad, con area de oportunidad VARIABLE.

    ubar = total defectos / total unidades;  límites: ubar +/- 3*sqrt(ubar/n_i)
    """
    c = np.asarray(defectos, dtype=float)
    nn = np.asarray(unidades, dtype=float) * np.ones_like(c)
    if np.any(nn <= 0):
        raise ValueError("El número de unidades debe ser positivo.")
    u = c / nn
    ubar = float(c.sum() / nn.sum())
    se = np.sqrt(ubar / nn)
    res = ChartResult(
        "Carta u", "Defectos por unidad", u,
        np.full(u.size, ubar), np.maximum(0.0, ubar - 3 * se), ubar + 3 * se,
        sigma=None, n_subgrupo=nn,
        notas=[f"ubar = {ubar:.5f} = {c.sum():.0f}/{nn.sum():.0f}.",
               "Límites variables cuando el area de oportunidad cambia entre subgrupos."],
    )
    res.violaciones = _limit_rules(u, res.lci, res.lcs) | nelson_rules(u, res.centro, None)
    return res


def _limit_rules(x, lci, lcs) -> dict[int, list[int]]:
    """Regla 1 para cartas con límites asimétricos o variables."""
    x = np.asarray(x, dtype=float)
    idx = np.where((x > np.asarray(lcs) + 1e-12) | (x < np.asarray(lci) - 1e-12))[0].tolist()
    return {1: idx} if idx else {}
