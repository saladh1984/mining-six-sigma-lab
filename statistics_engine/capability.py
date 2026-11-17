"""Análisis de capacidad del proceso: Cp, Cpk, Pp, Ppk, Cpm, Z y PPM.

DISTINCIÓN CENTRAL (requisito 11 de la especificación)
------------------------------------------------------
* Cp / Cpk  -> CAPACIDAD POTENCIAL. Usan sigma DENTRO de subgrupo (variabilidad
  de corto plazo, "lo mejor que el proceso puede hacer si está estable").
      sigma_within = Rbar/d2(n)   o   Sbar/c4(n)   o  MRbar/d2(2) si n=1
* Pp / Ppk  -> DESEMPEÑO REAL. Usan sigma TOTAL (desviación estándar muestral
  de todos los datos, ddof=1), que incluye la variación entre subgrupos.
      sigma_overall = s

Si Cp >> Pp, hay variación entre subgrupos (deriva, turnos, operadores): el
proceso no es estable y los índices de capacidad NO son interpretables todavía.

PRERREQUISITOS (la interfaz debe bloquear o advertir si no se cumplen)
---------------------------------------------------------------------
1. Proceso estable: sin señales fuera de control. Un Cpk sobre un proceso
   inestable no predice nada futuro.
2. Distribución aproximadamente normal: Cp/Cpk convierten sigmas en fracción
   defectuosa suponiendo normalidad. Con datos sesgados (comun en tiempos de
   ciclo) el PPM estimado puede errar por ordenes de magnitud.
3. Límites de especificación (LIE/LSE) provienen del cliente o del requisito
   técnico. NO se calculan de los datos y NO son límites de control.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .constants import c4, d2


@dataclass
class CapabilityResult:
    n: int
    media: float
    sigma_within: float
    sigma_overall: float
    metodo_sigma_within: str
    lie: float | None
    lse: float | None
    objetivo: float | None
    cp: float | None
    cpk: float | None
    cpu: float | None
    cpl: float | None
    pp: float | None
    ppk: float | None
    ppu: float | None
    ppl: float | None
    cpm: float | None
    z_bench_within: float | None
    ppm_observado: float
    ppm_esperado_within: float | None
    ppm_esperado_overall: float | None
    advertencias: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def sigma_within_from_subgroups(df_subgrupos: np.ndarray, metodo: str = "R") -> tuple[float, str]:
    """sigma de corto plazo a partir de una matriz (k subgrupos x n observaciones).

    método "R": Rbar/d2(n)   -- tradicional, recomendado para n <= 8
    método "S": Sbar/c4(n)   -- más eficiente para n >= 9
    """
    m = np.asarray(df_subgrupos, dtype=float)
    if m.ndim != 2:
        raise ValueError("Se espera una matriz de k subgrupos x n observaciones.")
    k, n = m.shape
    if n < 2:
        raise ValueError("Para n=1 use sigma_within_individuals().")
    if metodo.upper() == "R":
        rbar = float(np.mean(m.max(axis=1) - m.min(axis=1)))
        return rbar / d2(n), f"Rbar/d2(n={n}), k={k} subgrupos"
    if metodo.upper() == "S":
        sbar = float(np.mean(m.std(axis=1, ddof=1)))
        return sbar / c4(n), f"Sbar/c4(n={n}), k={k} subgrupos"
    raise ValueError("método debe ser 'R' o 'S'.")


def sigma_within_individuals(x: Sequence[float]) -> tuple[float, str]:
    """sigma de corto plazo para datos individuales: MRbar/d2(2), d2(2)=1.128.

    El rango movil de orden 2 estima la variación entre observaciones
    consecutivas, es decir la variabilidad de corto plazo.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    if s.size < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    mr = np.abs(np.diff(s))
    return float(mr.mean()) / d2(2), "MRbar/d2(2)=MRbar/1.128 (datos individuales)"


def capability(
    x: Sequence[float],
    lie: float | None = None,
    lse: float | None = None,
    objetivo: float | None = None,
    subgrupos: np.ndarray | None = None,
    metodo_within: str = "R",
) -> CapabilityResult:
    """Calcula los índices de capacidad y desempeño.

    x         : todas las observaciones (vector plano).
    lie, lse  : límites de especificación. Al menos uno es obligatorio.
    objetivo  : valor nominal. Solo se usa para Cpm.
    subgrupos : matriz k x n opcional. Si se entrega, sigma_within sale de los
                subgrupos; si no, se usa el rango movil (datos individuales).
    """
    s = pd.Series(x, dtype="float64").dropna()
    n = int(s.size)
    if n < 2:
        raise ValueError("Se requieren al menos 2 observaciones válidas.")
    if lie is None and lse is None:
        raise ValueError("Debe especificar al menos un límite (LIE o LSE).")
    if lie is not None and lse is not None and lie >= lse:
        raise ValueError("LIE debe ser menor que LSE.")

    mu = float(s.mean())
    sigma_overall = float(s.std(ddof=1))
    if subgrupos is not None:
        sigma_w, metodo = sigma_within_from_subgroups(subgrupos, metodo_within)
    else:
        sigma_w, metodo = sigma_within_individuals(s)

    if sigma_w <= 0 or sigma_overall <= 0:
        raise ValueError("La desviación estándar estimada es cero: datos constantes.")

    def _idx(sigma: float):
        cpu = (lse - mu) / (3 * sigma) if lse is not None else None
        cpl = (mu - lie) / (3 * sigma) if lie is not None else None
        cp = (lse - lie) / (6 * sigma) if (lie is not None and lse is not None) else None
        candidatos = [v for v in (cpu, cpl) if v is not None]
        cpk = min(candidatos) if candidatos else None
        return cp, cpk, cpu, cpl

    cp, cpk, cpu, cpl = _idx(sigma_w)
    pp, ppk, ppu, ppl = _idx(sigma_overall)

    # Cpm penaliza el alejamiento del objetivo: tau^2 = sigma^2 + (mu - T)^2
    cpm = None
    if objetivo is not None and lie is not None and lse is not None:
        tau = np.sqrt(sigma_overall**2 + (mu - objetivo) ** 2)
        cpm = (lse - lie) / (6 * tau)

    # PPM observado: cuenta directa, sin supuesto distribucional
    fuera = 0
    if lie is not None:
        fuera += int((s < lie).sum())
    if lse is not None:
        fuera += int((s > lse).sum())
    ppm_obs = 1e6 * fuera / n

    def _ppm_esperado(sigma: float) -> float:
        p = 0.0
        if lie is not None:
            p += stats.norm.cdf(lie, loc=mu, scale=sigma)
        if lse is not None:
            p += stats.norm.sf(lse, loc=mu, scale=sigma)
        return 1e6 * p

    ppm_w, ppm_o = _ppm_esperado(sigma_w), _ppm_esperado(sigma_overall)
    z_bench = float(stats.norm.isf(ppm_w / 1e6)) if 0 < ppm_w < 1e6 else None

    adv = _advertencias(s, n, cp, pp, sigma_w, sigma_overall)

    return CapabilityResult(
        n=n, media=mu, sigma_within=sigma_w, sigma_overall=sigma_overall,
        metodo_sigma_within=metodo, lie=lie, lse=lse, objetivo=objetivo,
        cp=cp, cpk=cpk, cpu=cpu, cpl=cpl, pp=pp, ppk=ppk, ppu=ppu, ppl=ppl,
        cpm=cpm, z_bench_within=z_bench, ppm_observado=ppm_obs,
        ppm_esperado_within=ppm_w, ppm_esperado_overall=ppm_o, advertencias=adv,
    )


def _advertencias(s, n, cp, pp, sigma_w, sigma_overall) -> list[str]:
    adv: list[str] = []
    if n < 30:
        adv.append(
            f"n={n} < 30: los índices tienen un intervalo de confianza muy amplio. "
            "Minitab y AIAG recomiendan al menos 100 observaciones para Pp/Ppk."
        )
    if n >= 8:
        p_sw = float(stats.shapiro(s)[1]) if n <= 5000 else float(stats.normaltest(s)[1])
        if p_sw < 0.05:
            adv.append(
                f"Normalidad rechazada (p={p_sw:.4f}). Cp/Cpk/Pp/Ppk suponen normalidad "
                "para traducir sigmas en PPM; con datos sesgados el PPM esperado puede "
                "errar por ordenes de magnitud. Considere una transformación o un "
                "método no normal (percentiles de Clements / ajuste Johnson)."
            )
    if cp is not None and pp is not None and pp > 0 and cp / pp > 1.33:
        adv.append(
            f"Cp/Pp = {cp/pp:.2f} > 1.33: sigma entre subgrupos domina a sigma dentro de "
            "subgrupo. Hay variación especial (deriva, turno, operador, equipo). El "
            "proceso probablemente NO está en control estadístico y la capacidad "
            "potencial no es alcanzable sin eliminar esa causa primero."
        )
    adv.append(
        "Recordatorio: estos índices solo son interpretables si el proceso está en "
        "control estadístico. Verifique la carta de control ANTES de leer el Cpk."
    )
    return adv


def intervalo_confianza_cpk(cpk: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """IC aproximado para Cpk (Bissell, 1990), válido bajo normalidad.

        Cpk +/- z * sqrt( 1/(9n) + Cpk^2/(2(n-1)) )
    """
    if n < 2:
        raise ValueError("n >= 2.")
    z = float(stats.norm.isf((1 - conf) / 2))
    se = np.sqrt(1.0 / (9 * n) + cpk**2 / (2 * (n - 1)))
    return float(cpk - z * se), float(cpk + z * se)
