"""Componentes de interfaz reutilizables en todas las paginas."""
from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from utils.io import a_csv
from .didactica import obtener as obtener_ficha
from .tema import nota


def ficha(clave: str, expandido: bool = False) -> None:
    """Bloque didáctico de tres secciones + supuestos y limitaciones."""
    f = obtener_ficha(clave)
    with st.expander(f"Guía: {f.titulo}", expanded=expandido):
        t1, t2, t3, t4 = st.tabs(["Que es?", "Como funciona?", "Como se interpreta?",
                                  "Supuestos y limitaciones"])
        with t1:
            st.write(f.que_es)
        with t2:
            st.write(f.como_funciona)
            if f.formulas:
                st.markdown("**Formulas**")
                for latex, desc in f.formulas:
                    st.latex(latex)
                    st.caption(desc)
            if f.procedimiento:
                st.markdown("**Procedimiento**")
                for i, p in enumerate(f.procedimiento, 1):
                    st.markdown(f"{i}. {p}")
        with t3:
            st.write(f.como_se_interpreta)
        with t4:
            if f.supuestos:
                st.markdown("**Supuestos**")
                for s in f.supuestos:
                    st.markdown(f"- {s}")
            if f.limitaciones:
                st.markdown("**Limitaciones**")
                for l in f.limitaciones:
                    st.markdown(f"- {l}")


def interpretar(clave: str, generador: Callable[[], list[str]],
                etiqueta: str = "Interpretar resultados") -> None:
    """Boton que redacta la lectura a partir de los valores ya calculados."""
    if st.button(etiqueta, key=f"btn_{clave}", type="primary"):
        try:
            parrafos = generador()
        except Exception as e:  # el analisis puede no ser aplicable a estos datos
            st.error(f"No se pudo generar la interpretación: {e}")
            return
        with st.container(border=True):
            st.markdown("##### Interpretación de los resultados calculados")
            for p in parrafos:
                st.markdown(p)
            st.caption("Texto generado a partir de los valores calculados sobre los datos "
                       "actuales. No contiene conclusiones predefinidas.")


# Etiquetas de presentación para los nombres internos de columna. Se aplican
# solo al mostrar y exportar: las claves de los datos no cambian, de modo que
# ningún análisis depende de esta traducción.
ETIQUETAS = {
    "senal": "señal", "senal_superior": "señal superior",
    "senal_inferior": "señal inferior", "C_mas": "C+", "C_menos": "C−",
    "observacion": "observación", "excluido": "excluido",
    "media_acumulada": "media acumulada", "contribucion_pct": "contribución (%)",
    "correlacion_spearman": "correlación de Spearman", "entrada": "entrada",
    "fija": "sin incertidumbre", "categoria": "categoría", "minimo": "mínimo", "maximo": "máximo",
    "mediana": "mediana", "desv_std": "desv. estándar", "coef_variacion": "coef. de variación",
    "error_std_media": "error estándar de la media", "asimetria": "asimetría",
    "curtosis_exceso": "curtosis en exceso", "moda_metodo": "método de la moda",
    "n_missing": "faltantes", "pct_acumulado": "% acumulado", "pct": "%",
    "pct_contribucion": "% de contribución", "p_valor": "p-valor", "r2": "R²",
    "R2_bivariado": "R² bivariado", "error_std": "error estándar",
    "termino": "término", "efecto": "efecto", "coeficiente": "coeficiente",
    "ecuacion": "ecuación", "especificacion": "especificación",
    "critico_porque": "por qué es crítico", "ctq": "CTQ", "medida": "medida",
    "objetivo": "objetivo", "cliente": "cliente", "voz": "voz", "necesidad": "necesidad",
    "causa": "causa", "puntaje": "puntaje", "categoria_": "categoría",
    "minutos_perdidos": "minutos perdidos", "eventos": "eventos", "turno": "turno",
    "n_inspeccionadas": "n inspeccionadas", "defectuosas": "defectuosas",
    "unidades_inspeccion": "unidades de inspección", "defectos": "defectos",
    "n_inspeccionadas_const": "n inspeccionadas (constante)",
    "defectuosas_const": "defectuosas (n constante)",
    "defectos_area_constante": "defectos (área constante)",
    "variable": "variable", "valor": "valor", "subgrupo": "subgrupo",
    "replica": "réplica", "corrida": "corrida", "orden_ejecucion": "orden de ejecución",
    "posicion": "posición", "media": "media", "varianza": "varianza",
    "simple": "simple", "indicador": "indicador", "contenido": "contenido",
    "Elemento": "Elemento", "Contenido": "Contenido", "cambio_%": "cambio (%)",
    "cumplimiento_%": "cumplimiento (%)", "CV_%": "CV (%)", "rank": "orden",
    "vital": "pocos vitales", "n": "n", "gl": "gl", "SC": "SC", "CM": "CM",
    "F": "F", "p": "p", "pieza": "pieza", "operador": "operador",
    "pse_degenerada": "PSE degenerada", "supera_ME": "supera ME", "supera_SME": "supera SME",
    "abs_ef": "|efecto|", "orden": "orden", "contraste": "contraste",
}


def _etiquetar(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas solo para mostrar; no altera los datos subyacentes."""
    ren = {c: ETIQUETAS[c] for c in df.columns if isinstance(c, str) and c in ETIQUETAS
           and ETIQUETAS[c] != c}
    cols_regla = {c: c.replace("regla_", "regla ") for c in df.columns
                  if isinstance(c, str) and c.startswith("regla_")}
    ren.update(cols_regla)
    return df.rename(columns=ren) if ren else df


def tabla(df: pd.DataFrame, nombre: str, titulo: str = "", altura: int | None = None,
          descargar: bool = True) -> None:
    """Tabla con boton de exportación. Cumple la regla de relieve de la paleta:
    toda figura tiene una vista tabular de respaldo."""
    if titulo:
        st.markdown(f"**{titulo}**")
    vista = _etiquetar(df)
    kw = {"width": "stretch"}
    if altura is not None:
        kw["height"] = altura
    st.dataframe(vista, **kw)
    if descargar:
        st.download_button(f"Descargar {nombre} en CSV", a_csv(vista),
                           file_name=f"{nombre}.csv", mime="text/csv",
                           key=f"dl_{nombre}_{titulo[:18]}")


def advertencias(lista: list[str], titulo: str = "Verificación de supuestos") -> None:
    if not lista:
        return
    with st.expander(f"{titulo} ({len(lista)})", expanded=True):
        for a in lista:
            st.markdown(f"- {a}")


def requisito(cumple: bool, texto_ok: str, texto_falla: str) -> bool:
    """Muestra el estado de un prerrequisito metodológico y lo devuelve."""
    (st.success if cumple else st.error)(texto_ok if cumple else texto_falla)
    return cumple


def cabecera(titulo: str, etapa: str, descripcion: str) -> None:
    st.markdown(f"### {etapa} · {titulo}")
    st.markdown(nota(descripcion), unsafe_allow_html=True)


def selector_variable(df: pd.DataFrame, caso, clave: str, etiqueta: str = "Variable a analizar",
                      solo_numericas: bool = True, defecto: str | None = None) -> str:
    cols = (df.select_dtypes("number").columns.tolist() if solo_numericas
            else df.columns.tolist())
    pref = defecto or caso.respuesta
    idx = cols.index(pref) if pref in cols else 0
    return st.selectbox(etiqueta, cols, index=idx, key=clave)


def unidad_de(caso, variable: str) -> str:
    try:
        return caso.var(variable).unidad
    except KeyError:
        return ""
