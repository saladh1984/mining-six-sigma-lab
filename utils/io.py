"""Entrada y salida de datos: exportación a CSV y carga de archivos del usuario."""
from __future__ import annotations

import io
from dataclasses import is_dataclass, asdict

import pandas as pd


def a_csv(df: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a CSV UTF-8 con BOM, para que Excel lo abra bien."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=";", decimal=",", float_format="%.6f")
    return buf.getvalue().encode("utf-8-sig")


def resultado_a_frame(obj, nombre: str = "resultado") -> pd.DataFrame:
    """Convierte un resultado (dataclass o dict) en una tabla de dos columnas."""
    if is_dataclass(obj):
        d = asdict(obj)
    elif isinstance(obj, dict):
        d = obj
    else:
        raise TypeError(f"No se puede convertir {type(obj)} a tabla.")
    filas = []
    for k, v in d.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, (list, tuple)):
            v = " | ".join(str(x) for x in v)
        filas.append({"indicador": k, "valor": v})
    df = pd.DataFrame(filas)
    # La columna de valores mezcla numeros, textos y nulos: se normaliza a texto
    # porque una columna de tipo mixto no es serializable a Arrow y Streamlit
    # tendria que repararla en cada renderizado.
    if "valor" in df.columns:
        df["valor"] = df["valor"].map(
            lambda x: "-" if x is None else (f"{x:,.6g}".replace(",", " ")
                                             if isinstance(x, (int, float)) and not isinstance(x, bool)
                                             else str(x)))
    return df


def validar_csv(archivo, columnas_requeridas: list[str] | None = None
                ) -> tuple[pd.DataFrame | None, list[str]]:
    """Lee un CSV subido por el usuario y devuelve (datos, mensajes de problema).

    Acepta separador coma o punto y coma y decimal punto o coma. Si el archivo
    no se puede interpretar, devuelve (None, mensajes) en vez de fallar.
    """
    mensajes: list[str] = []
    df = None
    for sep, dec in ((";", ","), (",", "."), (";", "."), (",", ",")):
        try:
            archivo.seek(0)
            cand = pd.read_csv(archivo, sep=sep, decimal=dec)
            if cand.shape[1] > 1:
                df = cand
                mensajes.append(f"Archivo leido con separador '{sep}' y decimal '{dec}'.")
                break
        except Exception:
            continue
    if df is None:
        return None, ["No se pudo interpretar el archivo. Debe ser un CSV con separador "
                      "coma o punto y coma."]

    if df.empty:
        return None, mensajes + ["El archivo no contiene filas."]

    num = df.select_dtypes("number").columns.tolist()
    if not num:
        mensajes.append("Aviso: no se detecto ninguna columna numérica. Revise el separador "
                        "decimal del archivo.")
    else:
        mensajes.append(f"{df.shape[0]} filas y {df.shape[1]} columnas "
                        f"({len(num)} numéricas: {', '.join(num[:8])}"
                        f"{' ...' if len(num) > 8 else ''}).")

    if columnas_requeridas:
        faltan = [c for c in columnas_requeridas if c not in df.columns]
        if faltan:
            mensajes.append(f"Aviso: faltan las columnas esperadas {faltan}. Podra usar el "
                            "archivo igualmente, pero deberá seleccionar las variables a mano.")

    nulos = int(df.isna().sum().sum())
    if nulos:
        mensajes.append(f"Aviso: el archivo tiene {nulos} valores faltantes. Cada análisis los "
                        "excluye por su cuenta y reporta el n efectivo utilizado.")
    return df, mensajes
