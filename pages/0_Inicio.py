"""Pagina de inicio: contexto del caso y tablero de indicadores."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import graficos as g
from components.tema import CSS, ESTADO, fila_tiles, nota, num, registrar_tema, tile
from data.cases import NOMBRES
from statistics_engine.capability import capability
from statistics_engine.descriptive import describe
from statistics_engine.control_charts import i_mr
from data.generators import subgrupos_racionales
from utils.estado import barra_lateral, datos_actuales, usando_datos_propios
from utils.io import a_csv

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()

# ------------------------------------------------------------------ encabezado
st.title("MINING SIX SIGMA LAB")
st.caption("Laboratorio de herramientas estadísticas y mejora de procesos mineros · "
           "Recorrido completo DMAIC con calculos verificables")

st.markdown(nota(
    "<b>Sobre los datos.</b> Los conjuntos de datos de esta aplicación son <b>simulados</b> a "
    "partir de un modelo explicito documentado en <code>data/generators.py</code>. "
    "No son mediciones de ninguna faena real y no constituyen evidencia sobre ninguna operación. "
    "Lo que si es real es el <b>calculo</b>: todos los estadísticos, índices y gráficos se "
    "computan íntegramente sobre los datos presentes, sin valores fijados a mano. "
    "Puede reemplazarlos por sus propios datos con el cargador de CSV de la barra lateral."
), unsafe_allow_html=True)

if usando_datos_propios():
    st.success("Está trabajando con **datos propios cargados desde CSV**, no con los simulados.")

# ------------------------------------------------------------------ KPI
st.subheader(f"{NOMBRES[caso.id]}")
st.markdown(f"**Area:** {caso.area}")

resp = caso.respuesta if caso.respuesta in df.columns else df.select_dtypes("number").columns[0]
v = caso.var(resp) if resp in [x.nombre for x in caso.variables] else None
unidad = v.unidad if v else ""
d = describe(df[resp])

tiles = [
    tile("Observaciones", f"{d.n:,}".replace(",", " "), sub="registros en el conjunto actual"),
    tile(f"Media de {v.etiqueta if v else resp}", num(d.media), unidad,
         sub=f"mediana {num(d.mediana)} {unidad}"),
    tile("Desviación estándar", num(d.desv_std), unidad,
         sub=f"rango {num(d.minimo)} a {num(d.maximo)} {unidad}"),
]
if d.coef_variacion is not None:
    cv = 100 * d.coef_variacion
    est = "bueno" if cv < 10 else "alerta" if cv < 20 else "serio" if cv < 30 else "critico"
    etiq = {"bueno": "BAJA", "alerta": "MODERADA", "serio": "ALTA", "critico": "MUY ALTA"}[est]
    tiles.append(tile("Coeficiente de variación", num(cv, 1), "%", sub="dispersión relativa",
                      estado=est, etiqueta_estado=f"DISPERSIÓN {etiq}"))

cap = None
if v is not None and (v.lie is not None or v.lse is not None):
    try:
        sub, _ = subgrupos_racionales(df, resp, caso.factores[-1] if caso.factores else None, 5)
        cap = capability(df[resp], lie=v.lie, lse=v.lse, objetivo=v.objetivo, subgrupos=sub)
        est = ("bueno" if cap.cpk >= 1.33 else "alerta" if cap.cpk >= 1.0 else "critico")
        etiq = {"bueno": "CAPAZ", "alerta": "MARGINAL", "critico": "NO CAPAZ"}[est]
        tiles.append(tile("Cpk (corto plazo)", num(cap.cpk), sub=f"Ppk {num(cap.ppk)}",
                          estado=est, etiqueta_estado=etiq))
        tiles.append(tile("Fuera de especificación", num(cap.ppm_observado / 1e4, 2), "%",
                          sub=f"{num(cap.ppm_observado, 0)} PPM, conteo directo"))
    except (ValueError, KeyError):
        cap = None

try:
    carta_i, _ = i_mr(df[resp])
    n_sen = len(carta_i.fuera_de_control)
    tiles.append(tile("Señales de causa especial", str(n_sen),
                      sub=f"sobre {len(carta_i.puntos)} observaciones (carta I)",
                      estado="bueno" if n_sen == 0 else "serio",
                      etiqueta_estado="ESTABLE" if n_sen == 0 else "INVESTIGAR"))
except ValueError:
    pass

st.markdown(fila_tiles(tiles), unsafe_allow_html=True)

if cap is not None and cap.cpk < 1.0:
    st.warning(
        f"**Lectura del tablero.** Con Cpk = {num(cap.cpk)} el proceso no es capaz de cumplir la "
        f"especificación de forma consistente: {num(cap.ppm_observado/1e4, 1)}% de las "
        "observaciones quedan fuera. Antes de interpretar este índice como una predicción, "
        "verifique en el módulo **Controlar** que el proceso este en control estadístico: "
        "la capacidad solo es predictiva sobre procesos estables."
    )

# ------------------------------------------------------------------ contexto
izq, der = st.columns([1.35, 1])
with izq:
    st.markdown("#### Contexto operacional")
    st.write(caso.contexto)
    st.markdown("#### Problema identificado")
    st.write(caso.problema)
    st.markdown("#### Objetivo del proyecto")
    st.write(caso.objetivo)
with der:
    st.markdown("#### Indicadores afectados")
    for i in caso.indicadores:
        st.markdown(f"- {i}")
    st.markdown("#### Alcance del proceso")
    st.markdown(f"**Inicio:** {caso.alcance_inicio}")
    st.markdown(f"**Fin:** {caso.alcance_fin}")
    st.markdown("#### Impacto operacional")
    st.write(caso.impacto)

# ------------------------------------------------------------------ vista rapida
st.divider()
st.markdown("#### Comportamiento de la variable crítica")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(g.histograma(df[resp], f"Distribución de {resp}", unidad,
                                 lie=v.lie if v else None, lse=v.lse if v else None,
                                 objetivo=v.objetivo if v else None),
                    width="stretch")
with c2:
    factor = caso.factores[0] if caso.factores and caso.factores[0] in df.columns else None
    if factor:
        st.plotly_chart(g.boxplot(df, resp, factor, f"{resp} por {factor}", unidad),
                        width="stretch")
    else:
        st.plotly_chart(g.boxplot(df, resp, None, f"{resp}", unidad), width="stretch")

# ------------------------------------------------------------------ ruta DMAIC
st.divider()
st.markdown("#### Ruta DMAIC y herramientas disponibles para este caso")
cols = st.columns(5)
for col, (etapa, herramientas) in zip(cols, caso.herramientas.items()):
    with col:
        st.markdown(f"**{etapa}**")
        for h in herramientas:
            st.markdown(f"<span style='font-size:13px'>· {h}</span>", unsafe_allow_html=True)
st.caption("Use el menu lateral para recorrer las cinco etapas. Cada herramienta incluye "
           "su explicación conceptual, su formula, sus supuestos, sus limitaciones y un boton "
           "de interpretación que redacta la lectura a partir de los valores realmente calculados.")

# ------------------------------------------------------------------ datos
st.divider()
with st.expander("Ver y exportar el conjunto de datos completo", expanded=False):
    st.caption(caso.nota_datos)
    st.dataframe(df, width="stretch", height=330)
    st.download_button("Descargar datos en CSV", a_csv(df),
                       file_name=f"{caso.id}_datos.csv", mime="text/csv")
    st.markdown("**Resumen descriptivo de todas las variables numéricas**")
    from statistics_engine.descriptive import describe_frame
    res = describe_frame(df)
    st.dataframe(res, width="stretch")
    st.download_button("Descargar resumen descriptivo en CSV", a_csv(res),
                       file_name=f"{caso.id}_descriptivo.csv", mime="text/csv")
