"""Estado compartido entre paginas de Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.cases import CASOS, NOMBRES, Caso, obtener

CLAVE_CASO = "caso_id"
CLAVE_SEMILLA = "semilla"
CLAVE_N = "n_obs"
CLAVE_DATOS_USUARIO = "datos_usuario"
CLAVE_FUENTE = "fuente_datos"


def inicializar() -> None:
    st.session_state.setdefault(CLAVE_CASO, "caso1")
    st.session_state.setdefault(CLAVE_SEMILLA, 2024)
    st.session_state.setdefault(CLAVE_N, 300)
    st.session_state.setdefault(CLAVE_DATOS_USUARIO, None)
    st.session_state.setdefault(CLAVE_FUENTE, "Datos simulados")


def caso_actual() -> Caso:
    return obtener(st.session_state[CLAVE_CASO])


@st.cache_data(show_spinner=False)
def _generar(case_id: str, n: int, semilla: int, mejora: bool) -> pd.DataFrame:
    return obtener(case_id).datos(n=n, semilla=semilla, mejora=mejora)


def datos_actuales(mejora: bool = False) -> pd.DataFrame:
    """Datos vigentes: los del usuario si cargo un CSV, o los simulados."""
    if (st.session_state[CLAVE_FUENTE] == "Archivo CSV propio"
            and st.session_state[CLAVE_DATOS_USUARIO] is not None and not mejora):
        return st.session_state[CLAVE_DATOS_USUARIO]
    return _generar(st.session_state[CLAVE_CASO], st.session_state[CLAVE_N],
                    st.session_state[CLAVE_SEMILLA], mejora)


def usando_datos_propios() -> bool:
    return (st.session_state[CLAVE_FUENTE] == "Archivo CSV propio"
            and st.session_state[CLAVE_DATOS_USUARIO] is not None)


def numericas(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes("number").columns.tolist()


def categoricas(df: pd.DataFrame, max_niveles: int = 12) -> list[str]:
    out = []
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            if df[c].nunique() <= max_niveles:
                out.append(c)
    return out


def barra_lateral() -> Caso:
    """Selector de caso y de fuente de datos, comun a todas las paginas."""
    from utils.io import validar_csv

    inicializar()
    with st.sidebar:
        st.markdown("### Caso de estudio")
        cid = st.selectbox("Operación minera", list(CASOS),
                           format_func=lambda k: NOMBRES[k],
                           index=list(CASOS).index(st.session_state[CLAVE_CASO]),
                           key="sel_caso")
        if cid != st.session_state[CLAVE_CASO]:
            st.session_state[CLAVE_CASO] = cid
            st.session_state[CLAVE_DATOS_USUARIO] = None
            st.session_state[CLAVE_FUENTE] = "Datos simulados"
            st.rerun()

        st.markdown("### Fuente de datos")
        fuente = st.radio("Origen", ["Datos simulados", "Archivo CSV propio"],
                          index=0 if st.session_state[CLAVE_FUENTE] == "Datos simulados" else 1,
                          label_visibility="collapsed", key="sel_fuente")
        st.session_state[CLAVE_FUENTE] = fuente

        if fuente == "Datos simulados":
            st.session_state[CLAVE_N] = st.slider(
                "Número de observaciones", 60, 1000, st.session_state[CLAVE_N], 20,
                help="Más observaciones reducen el error aleatorio de las estimaciones, "
                     "no el sesgo.")
            st.session_state[CLAVE_SEMILLA] = st.number_input(
                "Semilla aleatoria", 1, 999999, st.session_state[CLAVE_SEMILLA], 1,
                help="La misma semilla reproduce exactamente el mismo conjunto de datos.")
        else:
            archivo = st.file_uploader("Suba un CSV", type=["csv"], key="up_csv")
            if archivo is not None:
                df, msgs = validar_csv(archivo)
                for m in msgs:
                    (st.warning if m.startswith("Aviso") or m.startswith("No se") else st.caption)(m)
                if df is not None:
                    st.session_state[CLAVE_DATOS_USUARIO] = df
                    st.success(f"Datos cargados: {df.shape[0]} x {df.shape[1]}.")
            elif st.session_state[CLAVE_DATOS_USUARIO] is None:
                st.info("Sin archivo cargado todavía: se siguen usando los datos simulados.")

        st.divider()
        st.caption("**Los datos simulados no son mediciones de faena.** Provienen de un modelo "
                   "documentado en `data/generators.py`. Los estadísticos si se calculan "
                   "íntegramente sobre los datos.")
    return caso_actual()
