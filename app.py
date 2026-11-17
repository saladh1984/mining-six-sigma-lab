"""MINING SIX SIGMA LAB
Laboratorio de herramientas estadísticas y mejora de procesos mineros.

Punto de entrada de la aplicación. Ejecutar con:

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Mining Six Sigma Lab", page_icon="assets/favicon.svg",
                   layout="wide", initial_sidebar_state="expanded")

PAGINAS = [
    st.Page("pages/0_Inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("pages/1_Definir.py", title="1 · Definir", icon=":material/flag:"),
    st.Page("pages/2_Medir.py", title="2 · Medir", icon=":material/straighten:"),
    st.Page("pages/3_Analizar.py", title="3 · Analizar", icon=":material/query_stats:"),
    st.Page("pages/4_Mejorar.py", title="4 · Mejorar", icon=":material/trending_up:"),
    st.Page("pages/5_Controlar.py", title="5 · Controlar", icon=":material/monitoring:"),
]

st.navigation(PAGINAS).run()
