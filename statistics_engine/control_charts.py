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
class LimitesControl:
    """Límites estimados en una FASE I, listos para congelarse y aplicarse a datos nuevos.

    En control estadístico de procesos el trabajo tiene dos fases:

    FASE I  (estudio retrospectivo): se reúne un periodo base, se calculan los
            límites, se investigan y eliminan las causas especiales, y se
            recalculan hasta obtener un periodo estable.
    FASE II (vigilancia prospectiva): esos límites se CONGELAN y los datos
            nuevos se comparan contra ellos.

    Recalcular los límites con cada lote nuevo, que es el error más común, hace
    que un proceso corrido arrastre sus propios límites y el corrimiento quede
    oculto: exactamente lo que la carta debía detectar.
    """
    tipo: str                       # xbar_r | r | xbar_s | s | i | mr | ewma | p | np | c | u
    centro: float
    parametros: dict                # rbar, sbar, mrbar, pbar, cbar, ubar, sigma...
    n_subgrupo: int | None
    origen: str                     # descripción del periodo base
    k_base: int = 0                 # subgrupos usados para estimarlos
    excluidos: list[int] = field(default_factory=list)

    def resumen(self) -> pd.DataFrame:
        """Tabla de dos columnas con los parámetros congelados.

        La columna de valores se formatea como texto porque mezcla números con
        descripciones; una columna de tipo mixto no es serializable y obligaría
        a repararla en cada renderizado.
        """
        def _n(x):
            return f"{float(x):,.6g}".replace(",", " ")

        filas = [{"campo": "tipo de carta", "valor": self.tipo},
                 {"campo": "línea central", "valor": _n(self.centro)},
                 {"campo": "tamaño de subgrupo",
                  "valor": str(self.n_subgrupo) if self.n_subgrupo else "-"},
                 {"campo": "puntos de la base", "valor": str(self.k_base)},
                 {"campo": "excluidos de la base",
                  "valor": ", ".join(str(i + 1) for i in self.excluidos) or "ninguno"}]
        for k, v in self.parametros.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                filas.append({"campo": k, "valor": _n(v)})
        filas.append({"campo": "origen", "valor": self.origen})
        return pd.DataFrame(filas)


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
    excluidos: list[int] = field(default_factory=list)   # graficados, fuera del cálculo
    limites: "LimitesControl | None" = None              # para congelar y reutilizar
    origen_limites: str = ""                             # Fase I propia o Fase II heredada

    @property
    def fuera_de_control(self) -> list[int]:
        """Índices (base 0) con al menos una señal de cualquier regla."""
        return sorted({i for idxs in self.violaciones.values() for i in idxs})

    @property
    def senales_nuevas(self) -> list[int]:
        """Señales que NO corresponden a puntos ya excluidos por causa conocida."""
        return [i for i in self.fuera_de_control if i not in set(self.excluidos)]

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
        df["excluido"] = df.index.isin(self.excluidos)
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


def _mascara(n: int, excluir: Sequence[int] | None) -> np.ndarray:
    """Máscara booleana de los puntos que SÍ entran en el cálculo de límites."""
    m = np.ones(n, dtype=bool)
    if excluir:
        idx = np.asarray(sorted(set(int(i) for i in excluir)), dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"Índices a excluir fuera de rango 0..{n-1}: {idx.tolist()}")
        m[idx] = False
    if m.sum() < 2:
        raise ValueError("Tras excluir puntos quedan menos de 2 válidos para estimar límites.")
    return m


def _origen(limites, k_usados: int, excluidos: list[int]) -> str:
    if limites is not None:
        return (f"FASE II: límites congelados de {limites.origen}. "
                "Los datos mostrados se vigilan contra esos límites, no los recalculan.")
    base = f"FASE I: límites estimados con los {k_usados} puntos base mostrados"
    if excluidos:
        base += (f", excluyendo {len(excluidos)} punto(s) con causa asignable identificada "
                 f"(posiciones {', '.join(str(i+1) for i in excluidos)})")
    return base + "."


# ------------------------------------------------- cartas de variables -----
def xbar_r(datos, limites: LimitesControl | None = None,
           excluir: Sequence[int] | None = None) -> tuple[ChartResult, ChartResult]:
    """Carta Xbar-R. sigma_hat = Rbar/d2(n). Devuelve (carta Xbar, carta R).

    Se evalúa PRIMERO la carta R: si la dispersión no es estable, los límites
    de la carta de medias se calculan con un Rbar contaminado y no son válidos.

    limites : límites congelados de una Fase I. Si se entregan, NO se recalculan
              a partir de estos datos: se vigila contra ellos (Fase II).
    excluir : índices (base 0) de subgrupos con causa asignable ya identificada
              y corregida. Se grafican, pero no entran en el cálculo de límites.
    """
    m = _as_subgroup_matrix(datos)
    k, n = m.shape
    medias, rangos = m.mean(axis=1), m.max(axis=1) - m.min(axis=1)
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)

    if limites is not None:
        _verificar_tipo(limites, "xbar_r", n)
        xbarbar = limites.centro
        rbar = limites.parametros["rbar"]
        lim_r = limites.parametros["lim_r"]
    else:
        xbarbar = float(medias[msk].mean())
        rbar = float(rangos[msk].mean())
        lim_r = LimitesControl("r", rbar, {"rbar": rbar}, n,
                               _origen(None, int(msk.sum()), exc), int(msk.sum()), exc)
    sigma_hat = rbar / d2(n)
    origen = _origen(limites, int(msk.sum()), exc)

    r_chart = ChartResult(
        "Carta R", "Rango del subgrupo", rangos,
        np.full(k, rbar), np.full(k, d3_lower(n) * rbar), np.full(k, d4_upper(n) * rbar),
        sigma=None, n_subgrupo=np.full(k, n), excluidos=exc, limites=lim_r, origen_limites=origen,
        notas=[f"Rbar={rbar:.4f}; D3={d3_lower(n):.3f}, D4={d4_upper(n):.3f} para n={n}.",
               "Los límites de R son asimétricos: no se aplican las reglas de zona 5-8."],
    )
    r_chart.violaciones = _limit_rules(rangos, r_chart.lci, r_chart.lcs) | nelson_rules(rangos, r_chart.centro, None)

    lim_x = LimitesControl("xbar_r", xbarbar, {"rbar": rbar, "lim_r": lim_r}, n,
                           origen if limites is None else limites.origen,
                           int(msk.sum()), exc)
    x_chart = ChartResult(
        "Carta Xbar", "Media del subgrupo", medias,
        np.full(k, xbarbar), np.full(k, xbarbar - a2(n) * rbar), np.full(k, xbarbar + a2(n) * rbar),
        sigma=sigma_hat / np.sqrt(n), n_subgrupo=np.full(k, n), excluidos=exc,
        limites=lim_x, origen_limites=origen,
        notas=[f"Xbarbar={xbarbar:.4f}; A2={a2(n):.3f} para n={n}.",
               f"sigma del proceso estimada = Rbar/d2 = {sigma_hat:.4f}.",
               "No superponga límites de especificación sobre esta carta."],
    )
    x_chart.violaciones = nelson_rules(medias, x_chart.centro, x_chart.sigma)
    return x_chart, r_chart


def xbar_s(datos, limites: LimitesControl | None = None,
           excluir: Sequence[int] | None = None) -> tuple[ChartResult, ChartResult]:
    """Carta Xbar-S. sigma_hat = Sbar/c4(n). Preferible a Xbar-R con n >= 9."""
    m = _as_subgroup_matrix(datos)
    k, n = m.shape
    medias, sds = m.mean(axis=1), m.std(axis=1, ddof=1)
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)

    if limites is not None:
        _verificar_tipo(limites, "xbar_s", n)
        xbarbar = limites.centro
        sbar = limites.parametros["sbar"]
        lim_s = limites.parametros["lim_s"]
    else:
        xbarbar = float(medias[msk].mean())
        sbar = float(sds[msk].mean())
        lim_s = LimitesControl("s", sbar, {"sbar": sbar}, n,
                               _origen(None, int(msk.sum()), exc), int(msk.sum()), exc)
    sigma_hat = sbar / c4(n)
    origen = _origen(limites, int(msk.sum()), exc)

    s_chart = ChartResult(
        "Carta S", "Desviación estándar del subgrupo", sds,
        np.full(k, sbar), np.full(k, b3(n) * sbar), np.full(k, b4(n) * sbar),
        sigma=None, n_subgrupo=np.full(k, n), excluidos=exc, limites=lim_s, origen_limites=origen,
        notas=[f"Sbar={sbar:.4f}; B3={b3(n):.3f}, B4={b4(n):.3f} para n={n}."],
    )
    s_chart.violaciones = _limit_rules(sds, s_chart.lci, s_chart.lcs) | nelson_rules(sds, s_chart.centro, None)

    lim_x = LimitesControl("xbar_s", xbarbar, {"sbar": sbar, "lim_s": lim_s}, n,
                           origen if limites is None else limites.origen,
                           int(msk.sum()), exc)
    x_chart = ChartResult(
        "Carta Xbar (S)", "Media del subgrupo", medias,
        np.full(k, xbarbar), np.full(k, xbarbar - a3(n) * sbar), np.full(k, xbarbar + a3(n) * sbar),
        sigma=sigma_hat / np.sqrt(n), n_subgrupo=np.full(k, n), excluidos=exc,
        limites=lim_x, origen_limites=origen,
        notas=[f"Xbarbar={xbarbar:.4f}; A3={a3(n):.3f} para n={n}.",
               f"sigma del proceso estimada = Sbar/c4 = {sigma_hat:.4f}."],
    )
    x_chart.violaciones = nelson_rules(medias, x_chart.centro, x_chart.sigma)
    return x_chart, s_chart


def i_mr(x, limites: LimitesControl | None = None,
         excluir: Sequence[int] | None = None) -> tuple[ChartResult, ChartResult]:
    """Carta de individuales y rango móvil. sigma_hat = MRbar/1.128.

    Se usa cuando no hay subgrupos racionales (una medición por periodo). Es
    sensible a la no normalidad, más que la carta Xbar, porque no se beneficia
    del teorema central del límite.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    k = s.size
    if k < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    mr = np.abs(np.diff(s))
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)
    # Un rango móvil se descarta si cualquiera de sus dos observaciones se excluye
    msk_mr = msk[1:] & msk[:-1]
    if msk_mr.sum() < 1:
        raise ValueError("Tras excluir puntos no queda ningún rango móvil válido.")

    if limites is not None:
        _verificar_tipo(limites, "i", None)
        xbar = limites.centro
        mrbar = limites.parametros["mrbar"]
        lim_mr = limites.parametros["lim_mr"]
    else:
        xbar = float(s[msk].mean())
        mrbar = float(mr[msk_mr].mean())
        lim_mr = LimitesControl("mr", mrbar, {"mrbar": mrbar}, None,
                                _origen(None, int(msk.sum()), exc), int(msk.sum()), exc)
    sigma_hat = mrbar / d2(2)
    origen = _origen(limites, int(msk.sum()), exc)

    lim_i = LimitesControl("i", xbar, {"mrbar": mrbar, "lim_mr": lim_mr, "sigma": sigma_hat}, None,
                           origen if limites is None else limites.origen, int(msk.sum()), exc)
    i_chart = ChartResult(
        "Carta I (individuales)", "Observación individual", s,
        np.full(k, xbar), np.full(k, xbar - 3 * sigma_hat), np.full(k, xbar + 3 * sigma_hat),
        sigma=sigma_hat, excluidos=exc, limites=lim_i, origen_limites=origen,
        notas=[f"Xbar={xbar:.4f}; MRbar={mrbar:.4f}; sigma=MRbar/1.128={sigma_hat:.4f}.",
               "Más sensible a la no normalidad que la carta Xbar: verifique la distribución."],
    )
    i_chart.violaciones = nelson_rules(s, i_chart.centro, sigma_hat)

    mr_chart = ChartResult(
        "Carta MR (rango móvil)", "Rango móvil (orden 2)", mr,
        np.full(k - 1, mrbar), np.full(k - 1, d3_lower(2) * mrbar), np.full(k - 1, d4_upper(2) * mrbar),
        sigma=None, excluidos=[i for i in exc if i < k - 1], limites=lim_mr, origen_limites=origen,
        notas=["Los rangos móviles están autocorrelacionados por construcción: "
               "las reglas de rachas sobre la carta MR producen falsas alarmas. "
               "Use principalmente la regla 1."],
    )
    mr_chart.violaciones = _limit_rules(mr, mr_chart.lci, mr_chart.lcs)
    return i_chart, mr_chart


# ------------------------------------------- cartas de memoria (pequeños corrimientos)
def ewma(x, lam: float = 0.2, L: float = 3.0, mu0: float | None = None,
         sigma: float | None = None, limites: LimitesControl | None = None,
         excluir: Sequence[int] | None = None) -> ChartResult:
    """Carta EWMA (media móvil exponencialmente ponderada).

    La carta de Shewhart solo mira el punto actual, por lo que es lenta para
    detectar corrimientos PEQUEÑOS y sostenidos (del orden de 1 sigma), que son
    los habituales en un proceso que se degrada poco a poco. La EWMA acumula
    memoria de las observaciones anteriores y los detecta mucho antes.

        z_i = lambda*x_i + (1-lambda)*z_{i-1},   z_0 = mu0

    Los límites se ensanchan con i y convergen:

        mu0 +/- L*sigma*sqrt( lambda/(2-lambda) * (1-(1-lambda)^(2i)) )

    lam pequeño (0.05-0.2) da más memoria y detecta corrimientos menores pero
    responde más lento; lam -> 1 reproduce la carta de individuales.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    k = s.size
    if k < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    if not 0 < lam <= 1:
        raise ValueError("lambda debe estar en (0, 1].")
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)

    if limites is not None:
        _verificar_tipo(limites, "ewma", None)
        mu = limites.centro
        sig = limites.parametros["sigma"]
    else:
        mu = float(s[msk].mean()) if mu0 is None else float(mu0)
        if sigma is None:
            mr = np.abs(np.diff(s))
            msk_mr = msk[1:] & msk[:-1]
            sig = float(mr[msk_mr].mean()) / d2(2)
        else:
            sig = float(sigma)
    if sig <= 0:
        raise ValueError("La sigma estimada es cero: datos constantes.")

    z = np.empty(k)
    previo = mu
    for i in range(k):
        previo = lam * s[i] + (1 - lam) * previo
        z[i] = previo
    i_arr = np.arange(1, k + 1)
    ancho = L * sig * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i_arr)))
    origen = _origen(limites, int(msk.sum()), exc)

    res = ChartResult(
        f"Carta EWMA (lambda={lam:g}, L={L:g})", "EWMA de la observación", z,
        np.full(k, mu), mu - ancho, mu + ancho, sigma=None, excluidos=exc,
        limites=LimitesControl("ewma", mu, {"sigma": sig, "lam": lam, "L": L}, None,
                               origen if limites is None else limites.origen,
                               int(msk.sum()), exc),
        origen_limites=origen,
        notas=[f"mu0={mu:.4f}; sigma={sig:.4f} (de MRbar/1.128 salvo que se indique otra).",
               "Los puntos EWMA están autocorrelacionados por construcción: NO se aplican "
               "las reglas de rachas, solo el cruce de límites.",
               "Los límites se ensanchan en los primeros puntos y luego se estabilizan."],
    )
    res.violaciones = _limit_rules(z, res.lci, res.lcs)
    return res


@dataclass
class CusumResult:
    """Suma acumulada tabular de dos lados."""
    c_mas: np.ndarray
    c_menos: np.ndarray
    H: float
    K: float
    mu0: float
    sigma: float
    k_sigma: float
    h_sigma: float
    senales_mas: list[int]
    senales_menos: list[int]
    excluidos: list[int]
    origen_limites: str
    reiniciado: bool = False
    notas: list[str] = field(default_factory=list)

    @property
    def fuera_de_control(self) -> list[int]:
        return sorted(set(self.senales_mas) | set(self.senales_menos))

    @staticmethod
    def _excursiones(idxs: list[int]) -> list[int]:
        """Primer punto de cada tramo consecutivo por encima de H.

        Sin reinicio, una sola excursión del CUSUM deja varios puntos seguidos
        sobre el umbral. Contarlos uno a uno sobreestima el número de eventos:
        lo que importa es cuántas VECES se cruzó, no cuántos puntos quedaron
        arriba.
        """
        salida, previo = [], None
        for i in sorted(idxs):
            if previo is None or i != previo + 1:
                salida.append(i)
            previo = i
        return salida

    @property
    def excursiones_mas(self) -> list[int]:
        return self._excursiones(self.senales_mas)

    @property
    def excursiones_menos(self) -> list[int]:
        return self._excursiones(self.senales_menos)

    @property
    def excursiones(self) -> list[int]:
        return sorted(set(self.excursiones_mas) | set(self.excursiones_menos))

    @property
    def en_control(self) -> bool:
        return not self.fuera_de_control

    def to_frame(self) -> pd.DataFrame:
        n = len(self.c_mas)
        return pd.DataFrame({
            "observacion": np.arange(1, n + 1),
            "C_mas": self.c_mas, "C_menos": self.c_menos,
            "H": np.full(n, self.H),
            "senal_superior": np.isin(np.arange(n), self.senales_mas),
            "senal_inferior": np.isin(np.arange(n), self.senales_menos),
            "excluido": np.isin(np.arange(n), self.excluidos),
        })


def cusum(x, k_sigma: float = 0.5, h_sigma: float = 4.0, mu0: float | None = None,
          sigma: float | None = None, excluir: Sequence[int] | None = None,
          reiniciar_tras_senal: bool = False) -> CusumResult:
    """Carta CUSUM tabular de dos lados.

    Acumula las desviaciones respecto del objetivo, descontando una holgura K.
    Detecta corrimientos sostenidos antes que Shewhart.

        C+_i = max(0, x_i - (mu0 + K) + C+_{i-1})
        C-_i = max(0, (mu0 - K) - x_i + C-_{i-1})

    con K = k_sigma * sigma (holgura) y H = h_sigma * sigma (umbral de decisión).
    La combinación habitual k=0.5 y h=4 o 5 está optimizada para detectar un
    corrimiento de 1 sigma manteniendo una tasa de falsas alarmas comparable a
    la de una carta de Shewhart de 3 sigma.

    reiniciar_tras_senal : con False (por defecto) la acumulación continúa tras
        cruzar H, que es como la dibujan Minitab y la mayoría de los textos; una
        misma excursión deja entonces varios puntos sobre el umbral, por lo que
        conviene contar `excursiones` y no `senales`. Con True, C+ y C- vuelven
        a cero después de cada señal, que es lo que corresponde en operación
        cuando la señal se investiga y el proceso se ajusta.
    """
    s = pd.Series(x, dtype="float64").dropna().to_numpy()
    n = s.size
    if n < 2:
        raise ValueError("Se requieren al menos 2 observaciones.")
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(n, excluir)

    mu = float(s[msk].mean()) if mu0 is None else float(mu0)
    if sigma is None:
        mr = np.abs(np.diff(s))
        msk_mr = msk[1:] & msk[:-1]
        sig = float(mr[msk_mr].mean()) / d2(2)
    else:
        sig = float(sigma)
    if sig <= 0:
        raise ValueError("La sigma estimada es cero: datos constantes.")

    K, H = k_sigma * sig, h_sigma * sig
    cp, cm = np.zeros(n), np.zeros(n)
    ant_p = ant_m = 0.0
    for i in range(n):
        cp[i] = max(0.0, s[i] - (mu + K) + ant_p)
        cm[i] = max(0.0, (mu - K) - s[i] + ant_m)
        if reiniciar_tras_senal and (cp[i] > H or cm[i] > H):
            ant_p = ant_m = 0.0
        else:
            ant_p, ant_m = cp[i], cm[i]

    return CusumResult(
        cp, cm, H, K, mu, sig, k_sigma, h_sigma,
        np.where(cp > H)[0].tolist(), np.where(cm > H)[0].tolist(), exc,
        _origen(None, int(msk.sum()), exc), reiniciar_tras_senal,
        notas=[f"mu0={mu:.4f}; sigma={sig:.4f}; K={K:.4f}; H={H:.4f}.",
               "Una señal indica QUE el proceso se corrió, y la pendiente de la "
               "acumulación indica CUÁNDO empezó: el punto donde C+ o C- despegó de cero.",
               "CUSUM detecta corrimientos pequeños antes que Shewhart, pero reacciona "
               "más lento ante un salto grande y aislado. Ambas se complementan."],
    )


# ------------------------------------------------- cartas de atributos -----
def p_chart(defectuosos, n, limites: LimitesControl | None = None,
            excluir: Sequence[int] | None = None) -> ChartResult:
    """Carta p: proporción de unidades defectuosas. Modelo binomial.

    Límites variables si n cambia:  pbar +/- 3*sqrt(pbar(1-pbar)/n_i)
    pbar = total defectuosos / total inspeccionados  (NO el promedio de las p_i).
    """
    d = np.asarray(defectuosos, dtype=float)
    nn = np.asarray(n, dtype=float) * np.ones_like(d)
    if np.any(d > nn) or np.any(d < 0):
        raise ValueError("Los defectuosos deben estar entre 0 y n.")
    if np.any(nn <= 0):
        raise ValueError("El tamaño de muestra debe ser positivo.")
    p = d / nn
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(d.size, excluir)

    if limites is not None:
        _verificar_tipo(limites, "p", None)
        pbar = limites.centro
    else:
        pbar = float(d[msk].sum() / nn[msk].sum())
    se = np.sqrt(pbar * (1 - pbar) / nn)
    origen = _origen(limites, int(msk.sum()), exc)

    res = ChartResult(
        "Carta p", "Proporción defectuosa", p,
        np.full(p.size, pbar), np.maximum(0.0, pbar - 3 * se), np.minimum(1.0, pbar + 3 * se),
        sigma=None, n_subgrupo=nn, excluidos=exc, origen_limites=origen,
        limites=LimitesControl("p", pbar, {"pbar": pbar}, None,
                               origen if limites is None else limites.origen,
                               int(msk.sum()), exc),
        notas=[f"pbar = {pbar:.5f} = {int(d[msk].sum())}/{int(nn[msk].sum())}.",
               "Supuestos: unidades independientes y probabilidad de defecto constante. "
               "Si la variabilidad observada excede la binomial (sobredispersión), "
               "los límites son demasiado estrechos y aparecen falsas alarmas."],
    )
    if np.any(nn * pbar < 5):
        res.notas.append("Aviso: n*pbar < 5 en algún subgrupo; la aproximación normal "
                         "de los límites 3-sigma es pobre.")
    res.violaciones = _limit_rules(p, res.lci, res.lcs) | nelson_rules(p, res.centro, None)
    return res


def np_chart(defectuosos, n: int, limites: LimitesControl | None = None,
             excluir: Sequence[int] | None = None) -> ChartResult:
    """Carta np: número de defectuosos, con n CONSTANTE."""
    d = np.asarray(defectuosos, dtype=float)
    k = d.size
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)
    if limites is not None:
        _verificar_tipo(limites, "np", n)
        pbar = limites.parametros["pbar"]
    else:
        pbar = float(d[msk].sum() / (n * msk.sum()))
    npbar = n * pbar
    se = np.sqrt(npbar * (1 - pbar))
    origen = _origen(limites, int(msk.sum()), exc)
    res = ChartResult(
        "Carta np", "Número de defectuosos", d,
        np.full(k, npbar), np.full(k, max(0.0, npbar - 3 * se)), np.full(k, npbar + 3 * se),
        sigma=None, n_subgrupo=np.full(k, n), excluidos=exc, origen_limites=origen,
        limites=LimitesControl("np", npbar, {"pbar": pbar}, n,
                               origen if limites is None else limites.origen,
                               int(msk.sum()), exc),
        notas=[f"pbar={pbar:.5f}; n*pbar={npbar:.3f}.",
               "Requiere tamaño de muestra constante. Si n varía, use la carta p."],
    )
    res.violaciones = _limit_rules(d, res.lci, res.lcs) | nelson_rules(d, res.centro, None)
    return res


def c_chart(defectos, limites: LimitesControl | None = None,
            excluir: Sequence[int] | None = None) -> ChartResult:
    """Carta c: número de defectos por unidad de inspección CONSTANTE. Poisson."""
    c = np.asarray(defectos, dtype=float)
    k = c.size
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(k, excluir)
    if limites is not None:
        _verificar_tipo(limites, "c", None)
        cbar = limites.centro
    else:
        cbar = float(c[msk].mean())
    se = np.sqrt(cbar)
    origen = _origen(limites, int(msk.sum()), exc)
    res = ChartResult(
        "Carta c", "Número de defectos", c,
        np.full(k, cbar), np.full(k, max(0.0, cbar - 3 * se)), np.full(k, cbar + 3 * se),
        sigma=None, excluidos=exc, origen_limites=origen,
        limites=LimitesControl("c", cbar, {"cbar": cbar}, None,
                               origen if limites is None else limites.origen,
                               int(msk.sum()), exc),
        notas=[f"cbar={cbar:.4f}; sigma=sqrt(cbar)={se:.4f}.",
               "Supuesto Poisson: media = varianza, defectos independientes y área de "
               "oportunidad constante. Si la varianza excede la media, hay "
               "sobredispersión y el modelo no aplica."],
    )
    if cbar < 5:
        res.notas.append("Aviso: cbar < 5; la Poisson es marcadamente asimétrica y los "
                         "límites 3-sigma simétricos son aproximados.")
    res.violaciones = _limit_rules(c, res.lci, res.lcs) | nelson_rules(c, res.centro, None)
    return res


def u_chart(defectos, unidades, limites: LimitesControl | None = None,
            excluir: Sequence[int] | None = None) -> ChartResult:
    """Carta u: defectos por unidad, con área de oportunidad VARIABLE.

    ubar = total defectos / total unidades;  límites: ubar +/- 3*sqrt(ubar/n_i)
    """
    c = np.asarray(defectos, dtype=float)
    nn = np.asarray(unidades, dtype=float) * np.ones_like(c)
    if np.any(nn <= 0):
        raise ValueError("El número de unidades debe ser positivo.")
    u = c / nn
    exc = sorted(set(int(i) for i in (excluir or [])))
    msk = _mascara(c.size, excluir)
    if limites is not None:
        _verificar_tipo(limites, "u", None)
        ubar = limites.centro
    else:
        ubar = float(c[msk].sum() / nn[msk].sum())
    se = np.sqrt(ubar / nn)
    origen = _origen(limites, int(msk.sum()), exc)
    res = ChartResult(
        "Carta u", "Defectos por unidad", u,
        np.full(u.size, ubar), np.maximum(0.0, ubar - 3 * se), ubar + 3 * se,
        sigma=None, n_subgrupo=nn, excluidos=exc, origen_limites=origen,
        limites=LimitesControl("u", ubar, {"ubar": ubar}, None,
                               origen if limites is None else limites.origen,
                               int(msk.sum()), exc),
        notas=[f"ubar = {ubar:.5f} = {c[msk].sum():.0f}/{nn[msk].sum():.0f}.",
               "Límites variables cuando el área de oportunidad cambia entre subgrupos."],
    )
    res.violaciones = _limit_rules(u, res.lci, res.lcs) | nelson_rules(u, res.centro, None)
    return res


def _limit_rules(x, lci, lcs) -> dict[int, list[int]]:
    """Regla 1 para cartas con límites asimétricos o variables."""
    x = np.asarray(x, dtype=float)
    idx = np.where((x > np.asarray(lcs) + 1e-12) | (x < np.asarray(lci) - 1e-12))[0].tolist()
    return {1: idx} if idx else {}


def _verificar_tipo(limites: LimitesControl, esperado: str, n: int | None) -> None:
    if limites.tipo != esperado:
        raise ValueError(
            f"Los límites congelados son de una carta '{limites.tipo}' y se están "
            f"aplicando a una carta '{esperado}'. No son intercambiables.")
    if n is not None and limites.n_subgrupo is not None and limites.n_subgrupo != n:
        raise ValueError(
            f"Los límites se estimaron con subgrupos de tamaño {limites.n_subgrupo} y "
            f"ahora el tamaño es {n}. Los factores de carta dependen de n: vuelva a "
            "estimar la Fase I con el mismo tamaño de subgrupo.")
