"""Etapa DEFINIR: Project Charter, SIPOC, VOC, CTQ y Pareto preliminar."""
from __future__ import annotations

import pandas as pd
import streamlit as st


from components import graficos as g
from components import interpretacion as I
from components.tema import CSS, fila_tiles, nota, num, registrar_tema, tile
from components.ui import cabecera, ficha, interpretar, tabla
from data.generators import hoja_verificacion
from statistics_engine.pareto import pareto as calcular_pareto, matriz_causa_efecto
from utils.estado import barra_lateral

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()

cabecera("Definir el problema y el alcance", "ETAPA 1 — DEFINIR",
         "El objetivo de esta etapa es acotar: que problema se resuelve, para quien, con que "
         "alcance y con que criterio de exito. Un proyecto mal definido no se arregla con "
         "estadística después. Aquí no se calcula casi nada; se decide.")

t_charter, t_sipoc, t_voc, t_pareto = st.tabs(
    ["Project Charter", "SIPOC", "VOC y CTQ", "Pareto preliminar"])

# ------------------------------------------------------------ Project Charter
with t_charter:
    st.markdown("#### Project Charter")
    st.caption("Documento de una pagina que autoriza el proyecto y fija su alcance. "
               "Si no cabe en una pagina, el proyecto está mal acotado.")

    st.markdown(fila_tiles([
        tile("Area", caso.area.split(" - ")[0], sub=caso.area),
        tile("Variable crítica", caso.respuesta,
             sub=f"unidad: {caso.var(caso.respuesta).unidad or 'adimensional'}"),
        tile("Indicadores afectados", str(len(caso.indicadores)),
             sub=", ".join(caso.indicadores[:2]) + " ..."),
    ]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1. Problema")
        st.write(caso.problema)
        st.markdown("##### 2. Objetivo (específico y medible)")
        st.write(caso.objetivo)
        st.markdown("##### 3. Alcance")
        st.markdown(f"**Inicio del proceso:** {caso.alcance_inicio}")
        st.markdown(f"**Fin del proceso:** {caso.alcance_fin}")
        st.markdown(f"**Fuera de alcance:** {caso.fuera_de_alcance}")
    with c2:
        st.markdown("##### 4. Impacto operacional")
        st.write(caso.impacto)
        st.markdown("##### 5. Beneficio estimado")
        st.write(caso.beneficio_estimado)
        st.markdown("##### 6. Indicadores")
        for i in caso.indicadores:
            st.markdown(f"- {i}")
        st.markdown("##### 7. Equipo")
        for e in caso.equipo:
            st.markdown(f"- {e}")

    st.markdown(nota(
        "<b>Criterio de calidad del charter.</b> El objetivo debe ser verificable con una "
        "medición: 'reducir la desviación estándar en 25%' es verificable; 'mejorar la "
        "eficiencia' no lo es. El beneficio estimado es una hipótesis a validar en la etapa "
        "Mejorar, no un compromiso."), unsafe_allow_html=True)

    charter = pd.DataFrame({
        "Elemento": ["Problema", "Objetivo", "Inicio del alcance", "Fin del alcance",
                     "Fuera de alcance", "Impacto", "Beneficio estimado",
                     "Indicadores", "Equipo"],
        "Contenido": [caso.problema, caso.objetivo, caso.alcance_inicio, caso.alcance_fin,
                      caso.fuera_de_alcance, caso.impacto, caso.beneficio_estimado,
                      " | ".join(caso.indicadores), " | ".join(caso.equipo)],
    })
    tabla(charter, f"{caso.id}_charter", "Charter exportable")

# ------------------------------------------------------------ SIPOC
with t_sipoc:
    st.markdown("#### SIPOC")
    st.caption("Mapa de alto nivel: Proveedores, Entradas, Proceso, Salidas y Clientes. "
               "Sirve para fijar las fronteras del proceso y detectar a quien afecta el problema.")
    cols = st.columns(5)
    for col, (k, items) in zip(cols, caso.sipoc.items()):
        with col:
            st.markdown(f"**{k}**")
            for it in items:
                st.markdown(f"<div style='font-size:13px;padding:3px 0'>· {it}</div>",
                            unsafe_allow_html=True)
    largo = max(len(v) for v in caso.sipoc.values())
    sipoc_df = pd.DataFrame({k: v + [""] * (largo - len(v)) for k, v in caso.sipoc.items()})
    st.divider()
    tabla(sipoc_df, f"{caso.id}_sipoc", "SIPOC exportable")
    st.markdown(nota(
        "<b>Por que importa el orden.</b> El SIPOC se construye de derecha a izquierda: primero "
        "el cliente y lo que necesita, después las salidas, y solo al final las entradas y los "
        "proveedores. Construirlo al reves lleva a optimizar partes del proceso que al cliente "
        "no le importan."), unsafe_allow_html=True)

# ------------------------------------------------------------ VOC / CTQ
with t_voc:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Voz del cliente (VOC)")
        st.caption("Lo que el cliente dice, en sus palabras, antes de traducirlo.")
        for v in caso.voc:
            with st.container(border=True):
                st.markdown(f"**{v['cliente']}**")
                st.markdown(f"*\"{v['voz']}\"*")
                st.caption(f"Necesidad: {v['necesidad']}  ·  Se traduce en: {v['ctq']}")
    with c2:
        st.markdown("#### Crítico para la calidad (CTQ)")
        st.caption("Traducción de la voz del cliente a variables medibles con meta y "
                   "especificación.")
        ctq_df = pd.DataFrame(caso.ctq)
        st.dataframe(ctq_df, width="stretch", hide_index=True)
        st.markdown(nota(
            "<b>Especificación y meta no son lo mismo.</b> La especificación es el límite que "
            "define el defecto (lo fija el cliente o el requisito técnico); la meta es donde se "
            "quiere operar, con margen respecto de ese límite. Y ninguna de las dos es un límite "
            "de control, que se calcula a partir del proceso."), unsafe_allow_html=True)
    st.divider()
    tabla(pd.DataFrame(caso.voc), f"{caso.id}_voc", "VOC exportable")
    tabla(pd.DataFrame(caso.ctq), f"{caso.id}_ctq", "CTQ exportable")

# ------------------------------------------------------------ Pareto
with t_pareto:
    st.markdown("#### Pareto preliminar de problemas operacionales")
    ficha("pareto")
    hoja = hoja_verificacion()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        unidad_pareto = st.radio("Unidad de análisis",
                                 ["Minutos perdidos (impacto)", "Número de eventos (frecuencia)"],
                                 help="Ordenar por frecuencia y por impacto suele dar rankings "
                                      "distintos. Para decidir, el relevante es el de impacto.")
    with c2:
        filtro_turno = st.selectbox("Filtrar por turno", ["Todos"] + sorted(hoja["turno"].unique()))
    with c3:
        corte = st.slider("Corte de pocos vitales (%)", 50, 95, 80, 5)

    h = hoja if filtro_turno == "Todos" else hoja[hoja["turno"] == filtro_turno]
    col_valor = "minutos_perdidos" if unidad_pareto.startswith("Minutos") else "eventos"
    etiqueta = "Minutos perdidos" if col_valor == "minutos_perdidos" else "Eventos"
    res = calcular_pareto(h["categoria"], h[col_valor], corte=corte, etiqueta_valor=etiqueta)

    st.plotly_chart(g.pareto(res["tabla"], etiqueta, corte), width="stretch")

    st.markdown(fila_tiles([
        tile("Total", num(res["total"], 0), sub=etiqueta.lower()),
        tile("Pocos vitales", f"{res['n_vitales']} de {res['n_categorias']}",
             sub=f"cubren {num(res['pct_cubierto_por_vitales'],1)}% del total",
             estado="bueno" if res["concentrado"] else "alerta",
             etiqueta_estado="CONCENTRADO" if res["concentrado"] else "DISTRIBUIDO"),
        tile("Primera categoría", res["tabla"].iloc[0]["categoria"],
             sub=f"{num(res['tabla'].iloc[0]['pct'],1)}% del total"),
    ]), unsafe_allow_html=True)

    tabla(res["tabla"], f"{caso.id}_pareto", "Tabla de Pareto")
    interpretar("pareto_definir", lambda: I.pareto(res))

    st.divider()
    st.markdown("#### Hoja de verificación")
    st.caption("Registro estructurado de eventos por categoría y turno. Es la fuente del Pareto: "
               "sin un sistema de registro consistente, el Pareto ordena ruido.")
    tabla(hoja, f"{caso.id}_hoja_verificacion", altura=340)

    st.divider()
    st.markdown("#### Comparación: frecuencia frente a impacto")
    st.caption("Las dos tablas ordenan las mismas categorías con criterios distintos. "
               "Si el orden cambia, decidir por frecuencia lleva a atacar el problema equivocado.")
    pf = calcular_pareto(hoja["categoria"], hoja["eventos"], etiqueta_valor="Eventos")["tabla"]
    pi = calcular_pareto(hoja["categoria"], hoja["minutos_perdidos"],
                         etiqueta_valor="Minutos")["tabla"]
    comp = pd.DataFrame({
        "Rango": range(1, len(pf) + 1),
        "Por frecuencia": pf["categoria"].values,
        "Eventos": pf["valor"].values.round(0),
        "Por impacto": pi["categoria"].values,
        "Minutos perdidos": pi["valor"].values.round(1),
    })
    cambia = (pf["categoria"].values != pi["categoria"].values).any()
    if cambia:
        st.warning("**El orden cambia según la unidad de análisis.** La categoría más frecuente "
                   f"es *{pf.iloc[0]['categoria']}*, pero la de mayor impacto es "
                   f"*{pi.iloc[0]['categoria']}*. La decisión debe basarse en el impacto.")
    tabla(comp, f"{caso.id}_pareto_comparacion", descargar=False)
