"""Plantillas CSV para aplicar la aplicación a un proceso real.

Cada carta de control necesita una estructura de datos distinta. Estas
plantillas traen las columnas exactas que la aplicación espera y unas pocas
filas de ejemplo con el formato correcto, para que el usuario las reemplace
por sus propios registros y vuelva a subir el archivo.

La columna de FECHA es opcional pero recomendable: con ella las cartas usan un
eje temporal real y una señal puede relacionarse con lo que ocurrió ese turno
en la operación, que es lo que convierte la carta en una herramienta de
investigación y no solo en un gráfico.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

HOY = date(2026, 1, 5)


def _fechas(n: int, cada_horas: int = 12) -> list[str]:
    base = pd.Timestamp(HOY)
    return [(base + pd.Timedelta(hours=cada_horas * i)).strftime("%Y-%m-%d %H:%M")
            for i in range(n)]


PLANTILLAS: dict[str, dict] = {}


def _reg(clave, titulo, cartas, descripcion, columnas, df):
    PLANTILLAS[clave] = {"titulo": titulo, "cartas": cartas,
                         "descripcion": descripcion, "columnas": columnas, "datos": df}


# ---------------------------------------------- variables con subgrupos ---
_n = 24
_rng = np.random.default_rng(2026)
_reg(
    "variables",
    "Mediciones continuas",
    "Xbar-R, Xbar-S, I-MR, EWMA, CUSUM, capacidad del proceso",
    "Una fila por MEDICIÓN, no por subgrupo. La aplicación forma los subgrupos "
    "agrupando filas consecutivas, opcionalmente dentro de cada nivel de un factor "
    "(turno, equipo, línea). Añada las columnas de factor que necesite: aparecerán "
    "como opciones de estratificación y de desagregación.",
    {"fecha": "opcional. Fecha y hora de la medición (AAAA-MM-DD HH:MM)",
     "turno": "opcional. Factor de estratificación; puede añadir otros",
     "equipo": "opcional. Otro factor de estratificación",
     "medicion": "OBLIGATORIA. El valor medido, numérico"},
    pd.DataFrame({
        "fecha": _fechas(_n),
        "turno": np.tile(["Dia", "Noche"], _n // 2),
        "equipo": _rng.choice(["EQ-01", "EQ-02", "EQ-03"], _n),
        "medicion": np.round(_rng.normal(33.5, 2.1, _n), 3),
    }),
)

# ------------------------------------------------- atributos: p y np ------
_k = 20
_n_insp = _rng.integers(120, 190, _k)
_reg(
    "atributos_defectuosos",
    "Unidades defectuosas",
    "Carta p (n variable) y carta np (n constante)",
    "Una fila por SUBGRUPO de inspección: cuántas unidades se revisaron y cuántas "
    "resultaron defectuosas. Una unidad es defectuosa o no lo es; si una unidad puede "
    "tener varios defectos, use la plantilla de conteo de defectos.",
    {"fecha": "opcional. Fecha del subgrupo de inspección",
     "turno": "opcional. Factor de estratificación",
     "n_inspeccionadas": "OBLIGATORIA. Unidades revisadas en el subgrupo (entero positivo)",
     "defectuosas": "OBLIGATORIA. Unidades defectuosas (entero, entre 0 y n_inspeccionadas)"},
    pd.DataFrame({
        "fecha": _fechas(_k),
        "turno": np.tile(["Dia", "Noche"], _k // 2),
        "n_inspeccionadas": _n_insp,
        "defectuosas": _rng.binomial(_n_insp, 0.045),
    }),
)

# -------------------------------------------------- atributos: c y u ------
_unid = _rng.integers(8, 15, _k)
_reg(
    "atributos_defectos",
    "Conteo de defectos",
    "Carta c (área constante) y carta u (área variable)",
    "Una fila por SUBGRUPO: cuántas unidades de inspección se revisaron y cuántos "
    "defectos se contaron en total. Aquí una misma unidad puede aportar varios defectos. "
    "Para la carta c el número de unidades debe ser constante; si varía, use la carta u.",
    {"fecha": "opcional. Fecha del subgrupo",
     "turno": "opcional. Factor de estratificación",
     "unidades_inspeccion": "OBLIGATORIA. Unidades o área inspeccionada (positivo)",
     "defectos": "OBLIGATORIA. Defectos contados en total (entero >= 0)"},
    pd.DataFrame({
        "fecha": _fechas(_k),
        "turno": np.tile(["Dia", "Noche"], _k // 2),
        "unidades_inspeccion": _unid,
        "defectos": _rng.poisson(0.55 * _unid),
    }),
)

# ------------------------------------------------------------ Gage R&R ----
_filas = [{"pieza": f"P{i+1:02d}", "operador": f"Op{j+1}", "replica": r + 1,
           "medicion": round(float(100 + 4 * np.sin(i) + 0.6 * j + _rng.normal(0, 0.9)), 3)}
          for i in range(10) for j in range(3) for r in range(3)]
_reg(
    "gage_rr",
    "Estudio del sistema de medición",
    "Gage R&R cruzado por ANOVA",
    "Diseño CRUZADO y BALANCEADO obligatorio: cada operador mide CADA pieza el mismo "
    "número de veces. Lo habitual son 10 piezas x 3 operadores x 3 réplicas = 90 filas. "
    "Las piezas deben cubrir el rango real de variación del proceso; si se eligen piezas "
    "parecidas entre sí, el %R&R sale artificialmente alto aunque el instrumento sea bueno.",
    {"pieza": "OBLIGATORIA. Identificador de la pieza o ítem medido",
     "operador": "OBLIGATORIA. Identificador del operador",
     "replica": "opcional. Número de repetición",
     "medicion": "OBLIGATORIA. El valor medido, numérico"},
    pd.DataFrame(_filas),
)


def obtener(clave: str) -> dict:
    if clave not in PLANTILLAS:
        raise KeyError(f"Plantilla desconocida: {clave}. Disponibles: {sorted(PLANTILLAS)}")
    return PLANTILLAS[clave]


def tabla_columnas(clave: str) -> pd.DataFrame:
    p = obtener(clave)
    return pd.DataFrame({"columna": list(p["columnas"]),
                         "qué debe contener": list(p["columnas"].values())})


def validar(df: pd.DataFrame, clave: str) -> tuple[bool, list[str]]:
    """Comprueba que un archivo cargado sirve para esa familia de cartas."""
    p = obtener(clave)
    obligatorias = [c for c, d in p["columnas"].items() if d.startswith("OBLIGATORIA")]
    faltan = [c for c in obligatorias if c not in df.columns]
    msgs = []
    if faltan:
        msgs.append(f"Faltan columnas obligatorias: {', '.join(faltan)}. "
                    f"Descargue la plantilla '{p['titulo']}' para ver el formato esperado.")
        return False, msgs
    for c in obligatorias:
        if c in ("pieza", "operador"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            msgs.append(f"La columna '{c}' no es numérica. Revise el separador decimal "
                        "del archivo (coma o punto).")
            return False, msgs
    if clave == "atributos_defectuosos":
        if (df["defectuosas"] > df["n_inspeccionadas"]).any():
            msgs.append("Hay subgrupos con más defectuosas que inspeccionadas.")
            return False, msgs
        if (df["n_inspeccionadas"] <= 0).any():
            msgs.append("Hay subgrupos con n_inspeccionadas no positiva.")
            return False, msgs
    if clave == "atributos_defectos":
        if (df["unidades_inspeccion"] <= 0).any():
            msgs.append("Hay subgrupos con unidades_inspeccion no positiva.")
            return False, msgs
        if (df["defectos"] < 0).any():
            msgs.append("Hay subgrupos con defectos negativos.")
            return False, msgs
    if clave == "gage_rr":
        conteo = df.groupby(["pieza", "operador"]).size()
        if conteo.nunique() != 1:
            msgs.append("El diseño no está balanceado: cada combinación pieza x operador "
                        "debe tener el mismo número de réplicas.")
            return False, msgs
        if conteo.iloc[0] < 2:
            msgs.append("Se requieren al menos 2 réplicas por combinación pieza x operador.")
            return False, msgs
    msgs.append(f"Archivo válido para {p['cartas']}. {len(df)} filas.")
    return True, msgs
