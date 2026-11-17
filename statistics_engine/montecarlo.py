"""Simulación de Monte Carlo: propagación de incertidumbre y análisis de sensibilidad.

POR QUÉ ESTE MÓDULO EXISTE
--------------------------
Un cálculo determinístico entrega una cifra puntual: VAN = 420.000. Esa cifra
hereda la incertidumbre de cada uno de sus insumos, pero la oculta. Monte Carlo
hace lo contrario: declara explícitamente qué se sabe y qué no de cada entrada,
propaga esa incertidumbre a través del modelo y entrega una DISTRIBUCIÓN de
resultados, con la que se puede responder lo que de verdad importa para decidir:
¿cuál es la probabilidad de que esta alternativa destruya valor?

QUÉ NO HACE
-----------
Monte Carlo no crea información. Si las distribuciones de entrada se eligen mal,
la distribución de salida estará igual de mal, solo que con apariencia de rigor.
Tres limitaciones que conviene tener presentes:

1. Supone INDEPENDENCIA entre las entradas salvo que se declare correlación.
   En la práctica muchas variables se mueven juntas (precio y costo, ley y
   recuperación), y ignorarlo subestima la dispersión de la salida.
2. Las colas de la distribución de salida dependen mucho de la forma elegida
   para las entradas, que suele ser la parte peor conocida.
3. El error de muestreo baja como 1/sqrt(n): cuadruplicar las simulaciones solo
   reduce a la mitad el error. El módulo reporta ese error para que se vea.

El diseño de las distribuciones sigue el de `blast_stochastic_simulator`
(fija / normal / triangular / uniforme), ampliado con lognormal y PERT y
vectorizado: aquí se muestrean las n simulaciones de una vez en lugar de
recorrer un bucle, lo que lo hace unas cien veces más rápido.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUCIONES = ("fija", "normal", "triangular", "uniforme", "lognormal", "pert")


def crear_generador(semilla: int | None = None) -> np.random.Generator:
    """Generador reproducible. Con la misma semilla, los mismos resultados."""
    return np.random.default_rng(seed=semilla)


@dataclass
class Incierto:
    """Una entrada del modelo, con o sin incertidumbre declarada.

    fija        : {"valor"}                    sin incertidumbre
    normal      : {"media", "sd"}              simétrica, colas infinitas
    triangular  : {"minimo", "moda", "maximo"} para juicio experto: lo mínimo,
                  lo más probable y lo máximo
    uniforme    : {"minimo", "maximo"}         cuando solo se conoce el rango
    lognormal   : {"media", "sd"}              de la variable (no de su log);
                  positiva y asimétrica a la derecha, útil para costos y tiempos
    pert        : {"minimo", "moda", "maximo", "lambda"} versión suavizada de la
                  triangular, que pondera más la moda (lambda = 4 por defecto)
    """
    nombre: str
    distribucion: str
    parametros: dict
    unidad: str = ""
    minimo_fisico: float | None = None   # recorte inferior (p. ej. no negativos)
    maximo_fisico: float | None = None

    def __post_init__(self):
        if self.distribucion not in DISTRIBUCIONES:
            raise ValueError(f"Distribución '{self.distribucion}' desconocida. "
                             f"Opciones: {', '.join(DISTRIBUCIONES)}.")
        self._validar()

    def _validar(self) -> None:
        p, d = self.parametros, self.distribucion
        def exige(*claves):
            faltan = [c for c in claves if c not in p]
            if faltan:
                raise ValueError(f"'{self.nombre}' ({d}) requiere {faltan}.")
        if d == "fija":
            exige("valor")
        elif d in ("normal", "lognormal"):
            exige("media", "sd")
            if p["sd"] < 0:
                raise ValueError(f"'{self.nombre}': la desviación estándar no puede ser negativa.")
            if d == "lognormal" and p["media"] <= 0:
                raise ValueError(f"'{self.nombre}': la lognormal requiere media positiva.")
        elif d == "uniforme":
            exige("minimo", "maximo")
            if p["minimo"] >= p["maximo"]:
                raise ValueError(f"'{self.nombre}': el mínimo debe ser menor que el máximo.")
        else:  # triangular y pert
            exige("minimo", "moda", "maximo")
            if not (p["minimo"] <= p["moda"] <= p["maximo"]):
                raise ValueError(f"'{self.nombre}': debe cumplirse mínimo <= moda <= máximo.")
            if p["minimo"] == p["maximo"]:
                raise ValueError(f"'{self.nombre}': el mínimo y el máximo no pueden coincidir.")

    def muestrear(self, n: int, rng: np.random.Generator) -> np.ndarray:
        p, d = self.parametros, self.distribucion
        if d == "fija":
            x = np.full(n, float(p["valor"]))
        elif d == "normal":
            x = rng.normal(p["media"], p["sd"], n)
        elif d == "uniforme":
            x = rng.uniform(p["minimo"], p["maximo"], n)
        elif d == "triangular":
            x = rng.triangular(p["minimo"], p["moda"], p["maximo"], n)
        elif d == "lognormal":
            # Se parametriza por la media y la desviación de la VARIABLE, no de
            # su logaritmo, que es lo que el usuario puede estimar.
            m, s = float(p["media"]), float(p["sd"])
            if s == 0:
                x = np.full(n, m)
            else:
                sigma2 = np.log1p((s / m) ** 2)
                x = rng.lognormal(np.log(m) - sigma2 / 2, np.sqrt(sigma2), n)
        else:  # pert
            a, m, b = float(p["minimo"]), float(p["moda"]), float(p["maximo"])
            lam = float(p.get("lambda", 4.0))
            media = (a + lam * m + b) / (lam + 2)
            if np.isclose(media, m):
                alfa = beta = (lam + 2) / 2
            else:
                alfa = ((media - a) * (2 * m - a - b)) / ((m - media) * (b - a))
                beta = alfa * (b - media) / (media - a)
            alfa, beta = max(alfa, 1e-6), max(beta, 1e-6)
            x = a + rng.beta(alfa, beta, n) * (b - a)
        if self.minimo_fisico is not None:
            x = np.maximum(x, self.minimo_fisico)
        if self.maximo_fisico is not None:
            x = np.minimum(x, self.maximo_fisico)
        return x

    def descripcion(self) -> str:
        p = self.parametros
        u = f" {self.unidad}" if self.unidad else ""
        if self.distribucion == "fija":
            return f"fija en {p['valor']:g}{u}"
        if self.distribucion in ("normal", "lognormal"):
            return f"{self.distribucion}, media {p['media']:g}{u}, sd {p['sd']:g}"
        if self.distribucion == "uniforme":
            return f"uniforme entre {p['minimo']:g} y {p['maximo']:g}{u}"
        return (f"{self.distribucion}: mínimo {p['minimo']:g}, moda {p['moda']:g}, "
                f"máximo {p['maximo']:g}{u}")


@dataclass
class ResultadoMC:
    muestras: pd.DataFrame          # una columna por entrada
    salida: np.ndarray
    nombre_salida: str
    unidad_salida: str
    n: int
    semilla: int | None
    entradas: list[Incierto] = field(default_factory=list)

    def resumen(self) -> dict:
        s = self.salida
        q = np.percentile(s, [5, 10, 25, 50, 75, 90, 95])
        sd = float(s.std(ddof=1))
        return {
            "n_simulaciones": self.n, "media": float(s.mean()), "desv_std": sd,
            "error_std_media": sd / np.sqrt(self.n),
            "minimo": float(s.min()), "p5": q[0], "p10": q[1], "p25": q[2],
            "mediana": q[3], "p75": q[4], "p90": q[5], "p95": q[6],
            "maximo": float(s.max()),
            "coef_variacion": sd / abs(float(s.mean())) if s.mean() != 0 else np.nan,
        }

    def tabla_resumen(self) -> pd.DataFrame:
        r = self.resumen()
        return pd.DataFrame({"indicador": list(r), "valor": list(r.values())})

    def probabilidad_mayor_que(self, umbral: float) -> float:
        """P(salida > umbral), estimada como la fracción de simulaciones."""
        return float((self.salida > umbral).mean())

    def probabilidad_menor_que(self, umbral: float) -> float:
        return float((self.salida < umbral).mean())

    def intervalo(self, confianza: float = 0.90) -> tuple[float, float]:
        """Intervalo central que contiene esa fracción de los resultados."""
        if not 0 < confianza < 1:
            raise ValueError("La confianza debe estar entre 0 y 1.")
        a = (1 - confianza) / 2
        lo, hi = np.percentile(self.salida, [100 * a, 100 * (1 - a)])
        return float(lo), float(hi)

    def error_montecarlo(self) -> float:
        """Error estándar de la media por muestreo. Baja como 1/sqrt(n)."""
        return float(self.salida.std(ddof=1) / np.sqrt(self.n))

    def sensibilidad(self) -> pd.DataFrame:
        """Tornado: correlación de rangos entre cada entrada y la salida.

        Se usa Spearman y no Pearson porque el modelo puede ser no lineal: lo
        que interesa es si la salida sube o baja de forma consistente con la
        entrada, no si lo hace en línea recta. La contribución normaliza los
        cuadrados para que sumen 100%, lo que facilita leer cuál entrada domina.

        ATENCIÓN: esto mide la influencia de la INCERTIDUMBRE de cada entrada,
        no la importancia física de la variable. Una variable muy influyente
        pero bien conocida (declarada como fija o con sd pequeña) aparecerá
        abajo en el tornado, y eso es correcto: reducir su incertidumbre no
        reduciría la del resultado.
        """
        filas = []
        for col in self.muestras.columns:
            x = self.muestras[col].to_numpy()
            if np.allclose(x, x[0]):          # entrada fija: no aporta incertidumbre
                filas.append({"entrada": col, "correlacion_spearman": 0.0,
                              "contribucion_pct": 0.0, "fija": True})
                continue
            r = float(stats.spearmanr(x, self.salida).statistic)
            filas.append({"entrada": col, "correlacion_spearman": r,
                          "contribucion_pct": r ** 2, "fija": False})
        df = pd.DataFrame(filas)
        total = df["contribucion_pct"].sum()
        df["contribucion_pct"] = 100 * df["contribucion_pct"] / total if total > 0 else 0.0
        return df.sort_values("contribucion_pct", ascending=False).reset_index(drop=True)

    def convergencia(self, pasos: int = 60) -> pd.DataFrame:
        """Media acumulada frente al número de simulaciones.

        Sirve para ver si n es suficiente: cuando la curva deja de moverse, se
        alcanzó la precisión que el muestreo puede dar.
        """
        ns = np.unique(np.linspace(20, self.n, pasos).astype(int))
        acum = np.cumsum(self.salida)
        medias = acum[ns - 1] / ns
        sd = self.salida.std(ddof=1)
        return pd.DataFrame({"n": ns, "media_acumulada": medias,
                             "error_std": sd / np.sqrt(ns)})


def simular(entradas: Sequence[Incierto], modelo: Callable[..., np.ndarray],
            n: int = 10_000, semilla: int | None = None,
            nombre_salida: str = "resultado", unidad_salida: str = "") -> ResultadoMC:
    """Propaga la incertidumbre de las entradas a través del modelo.

    modelo : función que recibe las muestras como argumentos con nombre (cada
             uno un arreglo de longitud n) y devuelve un arreglo de longitud n.
             Debe estar vectorizada; si no lo está, envuélvala con np.vectorize.
    """
    if n < 10:
        raise ValueError("Se requieren al menos 10 simulaciones.")
    nombres = [e.nombre for e in entradas]
    if len(set(nombres)) != len(nombres):
        raise ValueError("Hay nombres de entrada repetidos.")
    rng = crear_generador(semilla)
    muestras = pd.DataFrame({e.nombre: e.muestrear(n, rng) for e in entradas})
    salida = np.asarray(modelo(**{c: muestras[c].to_numpy() for c in muestras}), dtype=float)
    if salida.shape != (n,):
        raise ValueError(f"El modelo devolvió {salida.shape} y se esperaba ({n},). "
                         "Verifique que esté vectorizado.")
    return ResultadoMC(muestras, salida, nombre_salida, unidad_salida, n, semilla, list(entradas))


# --------------------------------------------------- aplicación económica --
def van_montecarlo(inversion: Incierto, beneficio_anual: Incierto, tasa: Incierto,
                   horizonte: int, n: int = 10_000, semilla: int | None = None) -> ResultadoMC:
    """Distribución del valor actual neto con flujos y tasa inciertos.

        VAN = sum_{t=1..T} B / (1+r)^t  -  I

    El beneficio anual se trata como constante dentro de cada simulación pero
    incierto entre simulaciones. Es un supuesto fuerte: en realidad el beneficio
    varía año a año, y modelarlo como constante SUBESTIMA la dispersión del VAN.
    """
    if horizonte < 1:
        raise ValueError("El horizonte debe ser de al menos 1 año.")

    def modelo(inversion, beneficio_anual, tasa):
        r = np.maximum(tasa, -0.99)
        t = np.arange(1, horizonte + 1)[:, None]
        factor = np.sum(1.0 / (1.0 + r[None, :]) ** t, axis=0)
        return beneficio_anual * factor - inversion

    return simular([inversion, beneficio_anual, tasa], modelo, n, semilla,
                   nombre_salida="VAN", unidad_salida="")
