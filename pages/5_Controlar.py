"""Etapa CONTROLAR: Fase I y Fase II, cartas de control, memoria y plan de control."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import graficos as g
from components import interpretacion as I
from components.tema import CSS, fila_tiles, nota, num, registrar_tema, tile
from components.ui import (advertencias, cabecera, ficha, interpretar, selector_variable,
                           tabla, unidad_de)
from data.generators import datos_atributos, subgrupos_racionales
from statistics_engine.capability import capability
from statistics_engine.control_charts import (NELSON_DESC, c_chart, cusum, ewma, i_mr,
                                              nelson_rules, np_chart, p_chart, u_chart,
                                              xbar_r, xbar_s)
from statistics_engine.descriptive import describe
from utils import plantillas
from utils.estado import barra_lateral, categoricas, datos_actuales
from utils.io import a_csv, validar_csv

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()
st.session_state.setdefault("limites_congelados", {})

cabecera("Sostener la mejora en el tiempo", "ETAPA 5 — CONTROLAR",
         "El objetivo es detectar cuándo el proceso cambia, para actuar sobre la causa y no "
         "sobre el síntoma. Una carta de control separa la variación por causas comunes "
         "(inherente, estable) de la variación por causas especiales (externa, corregible). "
         "Los límites de control se calculan a partir del proceso; no son límites de "
         "especificación.")

# ---------------------------------------------------------- utilidades ----
def _columna_fecha(d: pd.DataFrame) -> str | None:
    """Primera columna interpretable como fecha, si existe."""
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            return c
        if d[c].dtype == object and c.lower() in ("fecha", "fecha_hora", "timestamp", "date"):
            try:
                pd.to_datetime(d[c], errors="raise")
                return c
            except (ValueError, TypeError):
                continue
    return None


def _clave(variable: str, tipo: str) -> str:
    return f"{caso.id}|{variable}|{tipo}"


def _guardar(variable: str, tipo: str, limites, contexto: str) -> None:
    st.session_state["limites_congelados"][_clave(variable, tipo)] = {
        "limites": limites, "contexto": contexto, "variable": variable, "tipo": tipo}


def _recuperar(variable: str, tipo: str):
    return st.session_state["limites_congelados"].get(_clave(variable, tipo))


with st.expander("Plantillas CSV para aplicar esto a un proceso real", expanded=False):
    st.caption("Cada familia de cartas necesita una estructura de datos distinta. Descargue la "
               "plantilla, reemplace las filas de ejemplo por sus registros y súbala en la "
               "barra lateral (o, para atributos y Gage R&R, en su propia pestaña).")
    cols = st.columns(len(plantillas.PLANTILLAS))
    for col, (clave, p) in zip(cols, plantillas.PLANTILLAS.items()):
        with col:
            st.markdown(f"**{p['titulo']}**")
            st.caption(p["cartas"])
            st.download_button(f"Descargar plantilla", a_csv(p["datos"]),
                               file_name=f"plantilla_{clave}.csv", mime="text/csv",
                               key=f"pl_{clave}")
    sel = st.selectbox("Ver el detalle de las columnas de", list(plantillas.PLANTILLAS),
                       format_func=lambda k: plantillas.PLANTILLAS[k]["titulo"])
    st.write(plantillas.obtener(sel)["descripcion"])
    st.dataframe(plantillas.tabla_columnas(sel), width="stretch", hide_index=True)

t_var, t_mem, t_atr, t_ind, t_plan = st.tabs(
    ["A · Cartas de variables", "B · Cartas con memoria (EWMA y CUSUM)",
     "C · Cartas de atributos", "D · Indicadores", "E · Plan de control"])

# ====================================================== A. VARIABLES
with t_var:
    ficha("fases")
    ficha("cartas")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tipo = st.selectbox("Tipo de carta", ["Xbar-R", "Xbar-S", "I-MR"], key="cc_tipo",
                            help="Xbar-R con subgrupos pequeños (n<=10); Xbar-S con n>=9; "
                                 "I-MR cuando solo hay una observación por periodo.")
    with c2:
        var = selector_variable(df, caso, "cc_var")
    with c3:
        n_sub = st.slider("Tamaño de subgrupo", 2, 10, 5, key="cc_n",
                          disabled=(tipo == "I-MR"))
    with c4:
        cats = categoricas(df)
        estrato = st.selectbox("Subgrupos homogéneos dentro de",
                               ["Observaciones consecutivas"] + cats, key="cc_est",
                               disabled=(tipo == "I-MR"))
        estrato = None if estrato == "Observaciones consecutivas" else estrato

    u = unidad_de(caso, var)
    col_fecha = _columna_fecha(df)

    # --- construir el estadístico y su eje
    try:
        if tipo == "I-MR":
            serie = df[var].dropna()
            idx_filas = df.index.get_indexer(serie.index)[:, None]
            desc_sub = "una observación por punto (sin subgrupos)"
            k_total = len(serie)
        else:
            sub, desc_sub, idx_filas = subgrupos_racionales(
                df, var, estrato, n_sub, devolver_indices=True)
            k_total = sub.shape[0]
    except ValueError as e:
        st.error(f"No se pudo formar los subgrupos: {e}")
        st.stop()

    eje_x, etiqueta_x = None, "Subgrupo / observación"
    if col_fecha:
        usar_fecha = st.checkbox(f"Usar '{col_fecha}' como eje temporal", value=True,
                                 key="cc_fecha",
                                 help="Con un eje temporal real, una señal puede relacionarse "
                                      "con lo que ocurrió ese turno en la operación.")
        if usar_fecha:
            fechas = pd.to_datetime(df[col_fecha], errors="coerce").to_numpy()
            eje_x = fechas[idx_filas[:, 0]]
            etiqueta_x = col_fecha

    st.caption(f"Criterio de subagrupación: {desc_sub} ({k_total} puntos).")

    FASE_I = "Fase I — establecer la línea base"
    FASE_II = "Fase II — vigilar con límites congelados"
    modo = st.radio("Modo de trabajo", [FASE_I, FASE_II], horizontal=True, key="cc_modo")

    guardado = _recuperar(var, tipo)
    reglas_sel = st.multiselect(
        "Reglas de patrones no aleatorios a aplicar", list(NELSON_DESC),
        default=[1, 2, 3, 5, 6],
        format_func=lambda r: f"Regla {r}: {NELSON_DESC[r]}", key="cc_reglas",
        help="Activar las ocho reglas aumenta la detección pero también las falsas alarmas: "
             "la longitud media de racha en control cae de ~370 a ~91 subgrupos.")
    if len(reglas_sel) >= 8:
        st.warning("Con las ocho reglas activas, aproximadamente 1 de cada 91 subgrupos genera "
                   "una señal aunque el proceso esté perfectamente estable.")

    # ---------------------------------------------------------- FASE I ----
    if modo == FASE_I:
        cf1, cf2 = st.columns([1, 2])
        with cf1:
            k_base = st.slider("Puntos del periodo base", min(10, k_total), k_total,
                               k_total, key="cc_kbase",
                               help="La referencia habitual son 20 a 25 subgrupos. Con menos, "
                                    "los límites tienen mucha incertidumbre.")
        with cf2:
            excluir = st.multiselect(
                "Excluir del cálculo (causa asignable identificada y corregida)",
                list(range(k_base)), format_func=lambda i: f"Punto {i+1}", key="cc_exc",
                help="Se siguen graficando, con un anillo gris, pero no contaminan los "
                     "límites. Excluir un punto solo porque incomoda invalida el análisis.")

        datos_base = (df[var].dropna().iloc[:k_base] if tipo == "I-MR" else sub[:k_base])
        try:
            if tipo == "I-MR":
                principal, secundaria = i_mr(datos_base, excluir=excluir)
            elif tipo == "Xbar-R":
                principal, secundaria = xbar_r(datos_base, excluir=excluir)
            else:
                principal, secundaria = xbar_s(datos_base, excluir=excluir)
        except ValueError as e:
            st.error(f"No se pudo construir la carta: {e}")
            st.stop()
        principal.violaciones = nelson_rules(principal.puntos, principal.centro,
                                             principal.sigma, reglas_sel)
        eje_base = eje_x[:len(principal.puntos)] if eje_x is not None else None

        st.warning("**Orden de lectura correcto: primero la carta de dispersión.** Si el rango o "
                   "la desviación estándar no son estables, los límites de la carta de medias se "
                   "calcularon con un Rbar (o Sbar) contaminado y no son válidos.")
        st.plotly_chart(g.carta_control(
            secundaria, x=(eje_base[:len(secundaria.puntos)] if eje_base is not None else None),
            etiqueta_x=etiqueta_x), width="stretch")
        st.plotly_chart(g.carta_control(principal, x=eje_base, etiqueta_x=etiqueta_x),
                        width="stretch")

        nuevas = principal.senales_nuevas
        estable = not nuevas and secundaria.en_control
        st.markdown(fila_tiles([
            tile("Puntos en la base", str(k_base), sub=f"{len(excluir)} excluidos"),
            tile("Línea central", num(principal.centro[0], 4), u),
            tile("LCI", num(principal.lci[0], 4), u, sub="límite de CONTROL"),
            tile("LCS", num(principal.lcs[0], 4), u, sub="límite de CONTROL"),
            tile("Señales sin explicar", str(len(nuevas)),
                 estado="bueno" if estable else "critico",
                 etiqueta_estado="BASE ESTABLE" if estable else "AÚN NO ESTABLE"),
        ]), unsafe_allow_html=True)

        if estable:
            st.success("El periodo base no muestra señales sin explicar: es una línea base "
                       "válida. Congele los límites para pasar a vigilancia.")
        else:
            st.error(
                f"Quedan {len(nuevas)} señal(es) sin explicar en los puntos "
                f"{', '.join(str(i+1) for i in nuevas[:15])}. **Investíguelas antes de "
                "congelar.** Si identifica y corrige la causa de alguna, exclúyala arriba y "
                "los límites se recalcularán. Si una señal no tiene causa identificable, NO la "
                "excluya: es parte del comportamiento del proceso.")

        if st.button("Congelar estos límites como línea base", type="primary", key="cc_freeze"):
            _guardar(var, tipo, principal.limites,
                     f"{k_base} puntos base, {len(excluir)} excluidos, "
                     f"criterio: {desc_sub}")
            st.success(f"Límites congelados para **{var}** en carta **{tipo}**. "
                       "Cambie a Fase II para vigilar contra ellos.")
            st.rerun()

        advertencias(principal.notas, "Notas de la carta")
        interpretar("carta_fase1", lambda: I.carta(principal, var))
        tabla(principal.to_frame().round(5), f"{caso.id}_{var}_fase1",
              f"Datos de la {principal.nombre}", altura=300)

    # --------------------------------------------------------- FASE II ----
    else:
        if guardado is None:
            st.error(
                f"**No hay límites congelados para {var} en carta {tipo}.** La Fase II vigila "
                "contra una línea base; sin ella no hay nada contra qué comparar. Vuelva a "
                "Fase I, verifique que el periodo base esté estable y pulse «Congelar»."
            )
            st.stop()

        st.info(f"**Vigilando contra la línea base:** {guardado['contexto']}")
        datos_todo = (df[var].dropna() if tipo == "I-MR" else sub)
        try:
            if tipo == "I-MR":
                principal, secundaria = i_mr(datos_todo, limites=guardado["limites"])
            elif tipo == "Xbar-R":
                principal, secundaria = xbar_r(datos_todo, limites=guardado["limites"])
            else:
                principal, secundaria = xbar_s(datos_todo, limites=guardado["limites"])
        except ValueError as e:
            st.error(f"Los límites congelados no aplican a estos datos: {e}")
            st.stop()
        principal.violaciones = nelson_rules(principal.puntos, principal.centro,
                                             principal.sigma, reglas_sel)

        st.plotly_chart(g.carta_control(
            secundaria, x=(eje_x[:len(secundaria.puntos)] if eje_x is not None else None),
            etiqueta_x=etiqueta_x), width="stretch")
        st.plotly_chart(g.carta_control(principal, x=eje_x, etiqueta_x=etiqueta_x),
                        width="stretch")

        n_sen = len(principal.fuera_de_control)
        st.markdown(fila_tiles([
            tile("Línea central (congelada)", num(principal.centro[0], 4), u),
            tile("LCI", num(principal.lci[0], 4), u),
            tile("LCS", num(principal.lcs[0], 4), u),
            tile("Puntos vigilados", str(len(principal.puntos))),
            tile("Señales", str(n_sen), estado="bueno" if n_sen == 0 else "critico",
                 etiqueta_estado="EN CONTROL" if n_sen == 0 else "FUERA DE CONTROL"),
        ]), unsafe_allow_html=True)

        # Comparación con el error de recalcular
        try:
            if tipo == "I-MR":
                recalc, _ = i_mr(datos_todo)
            elif tipo == "Xbar-R":
                recalc, _ = xbar_r(datos_todo)
            else:
                recalc, _ = xbar_s(datos_todo)
            recalc.violaciones = nelson_rules(recalc.puntos, recalc.centro,
                                              recalc.sigma, reglas_sel)
            dif = len(principal.fuera_de_control) - len(recalc.fuera_de_control)
            if dif > 0:
                st.warning(
                    f"**Por qué no se recalculan los límites.** Con los límites congelados se "
                    f"detectan {len(principal.fuera_de_control)} señales. Si se recalcularan con "
                    f"estos mismos datos, la línea central pasaría de {num(principal.centro[0],4)} "
                    f"a {num(recalc.centro[0],4)} y solo se detectarían "
                    f"{len(recalc.fuera_de_control)}: **{dif} señales quedarían ocultas** porque "
                    "el proceso habría arrastrado sus propios límites.")
        except ValueError:
            pass

        if st.button("Descongelar y volver a estimar la línea base", key="cc_unfreeze"):
            st.session_state["limites_congelados"].pop(_clave(var, tipo), None)
            st.info("Límites descongelados. Recalcular solo es legítimo ante un cambio "
                    "deliberado y documentado del proceso.")
            st.rerun()

        st.dataframe(guardado["limites"].resumen(), width="stretch", hide_index=True)
        advertencias(principal.notas, "Notas de la carta")
        interpretar("carta_fase2", lambda: I.carta(principal, var))
        tabla(principal.to_frame().round(5), f"{caso.id}_{var}_fase2",
              f"Datos de la {principal.nombre}", altura=300)

    v = caso.var(var) if var in [x.nombre for x in caso.variables] else None
    if v and (v.lie is not None or v.lse is not None):
        lim = [f"LIE = {num(v.lie)}" for _ in [1] if v.lie is not None]
        lim += [f"LSE = {num(v.lse)}" for _ in [1] if v.lse is not None]
        st.info(
            f"**Especificación de esta variable: {', '.join(lim)} {u}.** Deliberadamente no se "
            "dibuja sobre la carta de medias: la media de subgrupos dispersa menos que las "
            "observaciones individuales (su desviación estándar es sigma/raíz(n)), de modo que "
            "compararlas con una especificación individual da una impresión falsamente "
            "optimista. La comparación correcta es el análisis de capacidad del módulo Medir.")

# ====================================================== B. MEMORIA
with t_mem:
    ficha("memoria")
    st.markdown(nota(
        "<b>Cuándo usar estas cartas.</b> La de Shewhart solo mira el punto actual y es lenta "
        "ante un corrimiento pequeño y sostenido, que es la forma típica en que un proceso se "
        "degrada. EWMA y CUSUM acumulan memoria y lo detectan mucho antes. A cambio, reaccionan "
        "más lento ante un salto grande y aislado, de modo que se complementan con Shewhart "
        "en lugar de sustituirla."), unsafe_allow_html=True)

    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        var_m = selector_variable(df, caso, "mem_var")
        carta_m = st.radio("Carta", ["EWMA", "CUSUM"], horizontal=True, key="mem_tipo")
    with cm2:
        if carta_m == "EWMA":
            lam = st.slider("lambda (memoria)", 0.05, 1.0, 0.20, 0.05, key="mem_lam",
                            help="Menor lambda = más memoria = detecta corrimientos más "
                                 "pequeños pero responde más lento. lambda=1 reproduce la "
                                 "carta de individuales.")
            L = st.slider("L (ancho de los límites, en sigmas)", 2.0, 3.5, 3.0, 0.1, key="mem_L")
        else:
            k_s = st.slider("k (holgura, en sigmas)", 0.25, 1.5, 0.50, 0.05, key="mem_k",
                            help="k = delta/2 está optimizado para detectar un corrimiento "
                                 "de delta sigmas. k=0.5 apunta a 1 sigma.")
            h_s = st.slider("h (umbral de decisión, en sigmas)", 2.0, 8.0, 4.0, 0.5, key="mem_h")
    with cm3:
        k_base_m = st.slider("Observaciones del periodo base", 10,
                             max(11, int(df[var_m].notna().sum())),
                             min(40, int(df[var_m].notna().sum())), key="mem_base",
                             help="mu0 y sigma se estiman SOLO con estas observaciones. Si se "
                                  "estiman con la serie ya corrida, ambas cartas quedan ciegas "
                                  "al corrimiento.")
        reinicio = (st.checkbox("Reiniciar tras señal", value=False, key="mem_reset",
                                help="Con reinicio, C+ y C- vuelven a cero tras señalar, que es "
                                     "lo que corresponde cuando la señal se investiga y el "
                                     "proceso se ajusta.")
                    if carta_m == "CUSUM" else False)

    serie_m = df[var_m].dropna()
    um = unidad_de(caso, var_m)
    base_m = serie_m.iloc[:k_base_m]
    try:
        mu0 = float(base_m.mean())
        i_base, _ = i_mr(base_m)
        sigma0 = i_base.limites.parametros["sigma"]
        st.caption(f"Periodo base: primeras {k_base_m} observaciones. "
                   f"mu0 = {num(mu0, 4)} {um}, sigma = {num(sigma0, 4)} {um} "
                   "(estimada por MRbar/1.128 sobre la base).")

        if carta_m == "EWMA":
            res_m = ewma(serie_m, lam=lam, L=L, mu0=mu0, sigma=sigma0)
            st.plotly_chart(g.carta_control(res_m, mostrar_zonas=False), width="stretch")
            n_sen = len(res_m.fuera_de_control)
            st.markdown(fila_tiles([
                tile("lambda", f"{lam:g}", sub="peso de la observación actual"),
                tile("mu0", num(mu0, 4), um, sub="objetivo de la línea base"),
                tile("Ancho asintótico", num(L * sigma0 * np.sqrt(lam / (2 - lam)), 4), um),
                tile("Señales", str(n_sen), estado="bueno" if n_sen == 0 else "critico",
                     etiqueta_estado="EN CONTROL" if n_sen == 0 else "CORRIMIENTO DETECTADO"),
            ]), unsafe_allow_html=True)
            advertencias(res_m.notas, "Notas de la carta EWMA")
            interpretar("ewma", lambda: I.carta(res_m, var_m))
            tabla(res_m.to_frame().round(5), f"{caso.id}_{var_m}_ewma", altura=300)
        else:
            res_m = cusum(serie_m, k_sigma=k_s, h_sigma=h_s, mu0=mu0, sigma=sigma0,
                          reiniciar_tras_senal=reinicio)
            st.plotly_chart(g.carta_cusum(res_m), width="stretch")
            st.markdown(fila_tiles([
                tile("K (holgura)", num(res_m.K, 4), um, sub=f"{k_s:g} sigma"),
                tile("H (umbral)", num(res_m.H, 4), um, sub=f"{h_s:g} sigma"),
                tile("Excursiones al alza", str(len(res_m.excursiones_mas)),
                     estado="bueno" if not res_m.excursiones_mas else "critico",
                     etiqueta_estado="SIN SEÑAL" if not res_m.excursiones_mas else "CORRIMIENTO"),
                tile("Excursiones a la baja", str(len(res_m.excursiones_menos)),
                     estado="bueno" if not res_m.excursiones_menos else "critico",
                     etiqueta_estado="SIN SEÑAL" if not res_m.excursiones_menos else "CORRIMIENTO"),
            ]), unsafe_allow_html=True)
            advertencias(res_m.notas, "Notas de la carta CUSUM")
            interpretar("cusum", lambda: I.cusum(res_m))
            tabla(res_m.to_frame().round(5), f"{caso.id}_{var_m}_cusum", altura=300)

        st.divider()
        st.markdown("##### Comparación con la carta de Shewhart sobre los mismos datos")
        sh, _ = i_mr(serie_m, limites=i_base.limites)
        ew = ewma(serie_m, lam=0.2, L=3.0, mu0=mu0, sigma=sigma0)
        cu = cusum(serie_m, k_sigma=0.5, h_sigma=4.0, mu0=mu0, sigma=sigma0)
        post = lambda idxs: [i for i in idxs if i >= k_base_m]
        filas = []
        for nombre, idxs in (("Shewhart (carta I, regla 1)", sh.violaciones.get(1, [])),
                             ("EWMA (lambda=0.2, L=3)", ew.fuera_de_control),
                             ("CUSUM (k=0.5, h=4)", cu.excursiones)):
            p = post(idxs)
            filas.append({"carta": nombre, "señales tras la base": len(p),
                          "primera señal (observación)": (min(p) + 1) if p else None})
        tabla(pd.DataFrame(filas), f"{caso.id}_{var_m}_comparacion_cartas",
              "Las tres sobre el mismo periodo base", descargar=False)
        st.caption("Si EWMA o CUSUM señalan antes que Shewhart, hay un corrimiento pequeño y "
                   "sostenido. Si solo señala Shewhart, se trata de un salto grande y aislado.")
    except ValueError as e:
        st.error(f"No se pudo construir la carta: {e}")

# ====================================================== C. ATRIBUTOS
with t_atr:
    ficha("cartas")
    fuente_a = st.radio("Datos de conteo", ["Ejemplo simulado", "Mi archivo CSV"],
                        horizontal=True, key="at_fuente")

    datos_a, listo = None, False
    if fuente_a == "Mi archivo CSV":
        familia = st.radio("Tipo de conteo",
                           ["Unidades defectuosas (cartas p y np)",
                            "Defectos contados (cartas c y u)"], key="at_familia")
        clave_pl = ("atributos_defectuosos" if familia.startswith("Unidades")
                    else "atributos_defectos")
        ca1, ca2 = st.columns([1, 2])
        with ca1:
            st.download_button("Descargar la plantilla correspondiente",
                               a_csv(plantillas.obtener(clave_pl)["datos"]),
                               file_name=f"plantilla_{clave_pl}.csv", mime="text/csv",
                               key="at_pl")
        with ca2:
            st.dataframe(plantillas.tabla_columnas(clave_pl), width="stretch", hide_index=True)
        archivo = st.file_uploader("Suba su archivo de conteos", type=["csv"], key="at_up")
        if archivo is not None:
            cand, msgs = validar_csv(archivo)
            for m in msgs:
                st.caption(m)
            if cand is not None:
                ok, vmsgs = plantillas.validar(cand, clave_pl)
                for m in vmsgs:
                    (st.success if ok else st.error)(m)
                if ok:
                    datos_a, listo = cand, True
        else:
            st.info("Sin archivo cargado todavía.")
    else:
        clave_pl = None
        st.markdown(nota(
            "<b>Por qué el ejemplo usa su propio conjunto.</b> Las cartas de atributos "
            "requieren CONTEOS (unidades defectuosas o defectos), no mediciones continuas, y "
            "esas columnas no existen en los datos de proceso del caso. Se simula un registro "
            "de inspección por turno con una causa especial inyectada en el subgrupo 22, para "
            "que la carta tenga algo real que detectar. Con «Mi archivo CSV» puede aplicarlo a "
            "sus propios registros."), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            k = st.slider("Número de subgrupos", 15, 60, 30, key="at_k")
        with c2:
            p_real = st.slider("Proporción defectuosa del proceso", 0.005, 0.20, 0.045, 0.005,
                               key="at_p")
        datos_a, listo = datos_atributos(k=k, p_real=p_real), True

    if listo:
        propio = fuente_a == "Mi archivo CSV"
        if propio:
            opciones = (["p (proporción defectuosa)", "np (número de defectuosos)"]
                        if clave_pl == "atributos_defectuosos"
                        else ["c (defectos, área constante)", "u (defectos por unidad)"])
        else:
            opciones = ["p (proporción defectuosa)", "np (número de defectuosos)",
                        "c (defectos, área constante)", "u (defectos por unidad)"]
        carta_a = st.selectbox("Carta", opciones, key="at_tipo")
        FASE_I_AT = "Fase I — establecer la línea base"
        FASE_II_AT = "Fase II — vigilar con límites congelados"
        modo_a = st.radio("Modo", [FASE_I_AT, FASE_II_AT], horizontal=True, key="at_modo")

        col_def = "defectuosas" if propio else "defectuosas"
        col_n = "n_inspeccionadas" if propio else "n_inspeccionadas"
        try:
            if carta_a.startswith("p "):
                args = ((datos_a[col_def], datos_a[col_n]) if propio
                        else (datos_a["defectuosas"], datos_a["n_inspeccionadas"]))
                fn, tipo_a = p_chart, "p"
            elif carta_a.startswith("np"):
                if propio:
                    if datos_a[col_n].nunique() != 1:
                        st.error("La carta np requiere tamaño de muestra CONSTANTE y su archivo "
                                 "tiene n variable. Use la carta p.")
                        st.stop()
                    args = (datos_a[col_def], int(datos_a[col_n].iloc[0]))
                else:
                    args = (datos_a["defectuosas_const"],
                            int(datos_a["n_inspeccionadas_const"].iloc[0]))
                fn, tipo_a = np_chart, "np"
            elif carta_a.startswith("c "):
                if propio:
                    if datos_a["unidades_inspeccion"].nunique() != 1:
                        st.warning("Su archivo tiene área de inspección variable: la carta c "
                                   "supone área constante. La carta u es la apropiada.")
                    args = (datos_a["defectos"],)
                else:
                    args = (datos_a["defectos_area_constante"],)
                fn, tipo_a = c_chart, "c"
            else:
                args = ((datos_a["defectos"], datos_a["unidades_inspeccion"]) if propio
                        else (datos_a["defectos"], datos_a["unidades_inspeccion"]))
                fn, tipo_a = u_chart, "u"

            clave_at = f"atributos_{tipo_a}"
            guardado_a = _recuperar(clave_at, tipo_a)
            if modo_a == FASE_I_AT:
                exc_a = st.multiselect(
                    "Excluir del cálculo (causa asignable identificada)",
                    list(range(len(datos_a))), format_func=lambda i: f"Subgrupo {i+1}",
                    key="at_exc")
                res_a = fn(*args, excluir=exc_a)
            else:
                if guardado_a is None:
                    st.error(f"No hay límites congelados para la carta {tipo_a}. "
                             "Establézcalos primero en Fase I.")
                    st.stop()
                st.info(f"**Vigilando contra:** {guardado_a['contexto']}")
                res_a = fn(*args, limites=guardado_a["limites"])

            eje_a = None
            if "fecha" in datos_a.columns:
                try:
                    eje_a = pd.to_datetime(datos_a["fecha"], errors="coerce").to_numpy()
                except (ValueError, TypeError):
                    eje_a = None
            st.plotly_chart(g.carta_control(res_a, x=eje_a,
                                            etiqueta_x="fecha" if eje_a is not None else "Subgrupo"),
                            width="stretch")

            n_sen = len(res_a.fuera_de_control)
            st.markdown(fila_tiles([
                tile("Línea central", num(res_a.centro[0], 5)),
                tile("Subgrupos", str(len(res_a.puntos))),
                tile("Señales", str(n_sen), estado="bueno" if n_sen == 0 else "critico",
                     etiqueta_estado="EN CONTROL" if n_sen == 0 else "FUERA DE CONTROL"),
                tile("Límites", "constantes" if len(set(np.round(res_a.lcs, 9))) == 1
                     else "variables",
                     sub="dependen del tamaño de cada subgrupo"
                     if len(set(np.round(res_a.lcs, 9))) > 1 else "n constante"),
            ]), unsafe_allow_html=True)

            if modo_a == FASE_I_AT and st.button("Congelar estos límites",
                                                         type="primary", key="at_freeze"):
                _guardar(clave_at, tipo_a, res_a.limites,
                         f"{len(res_a.puntos)} subgrupos base, "
                         f"{len(res_a.excluidos)} excluidos"
                         + (" (datos propios)" if propio else " (ejemplo simulado)"))
                st.success("Límites congelados.")
                st.rerun()

            st.info("Las cartas de atributos tienen estadísticos **asimétricos**, por lo que las "
                    "reglas de zona (5 a 8), que suponen simetría y normalidad, no se aplican. "
                    "Solo se usan las reglas 1 a 4.")
            advertencias(res_a.notas, "Notas y supuestos del modelo")
            interpretar("carta_atributos", lambda: I.carta(res_a))
            cc1, cc2 = st.columns(2)
            with cc1:
                tabla(datos_a, f"{caso.id}_atributos_datos", "Registro de inspección", altura=300)
            with cc2:
                tabla(res_a.to_frame().round(6), f"{caso.id}_atributos_carta",
                      "Carta exportable", altura=300)
        except (ValueError, KeyError) as e:
            st.error(f"No se pudo construir la carta: {e}")

# ====================================================== D. INDICADORES
with t_ind:
    st.markdown("#### Indicadores de seguimiento")
    st.caption("Tablero de control del proceso con desagregación por factor operacional. "
               "Todos los valores se calculan sobre los datos vigentes.")
    var_i = selector_variable(df, caso, "ind_var")
    ui = unidad_de(caso, var_i)
    vi = caso.var(var_i) if var_i in [x.nombre for x in caso.variables] else None
    d = describe(df[var_i])

    cumple = None
    if vi and (vi.lie is not None or vi.lse is not None):
        dentro = pd.Series(True, index=df.index)
        if vi.lie is not None:
            dentro &= df[var_i] >= vi.lie
        if vi.lse is not None:
            dentro &= df[var_i] <= vi.lse
        cumple = 100 * dentro.mean()

    tiles = [tile("Media del proceso", num(d.media, 3), ui),
             tile("Desviación estándar", num(d.desv_std, 3), ui),
             tile("CV", num(100 * d.coef_variacion, 1) if d.coef_variacion else "-", "%")]
    if cumple is not None:
        est = "bueno" if cumple >= 99 else "alerta" if cumple >= 95 else "critico"
        tiles.append(tile("Cumplimiento de especificación", num(cumple, 2), "%",
                          sub=f"{num(100-cumple,2)}% de defectos", estado=est,
                          etiqueta_estado={"bueno": "CONFORME", "alerta": "ATENCIÓN",
                                           "critico": "NO CONFORME"}[est]))
    if "productividad_tph" in df.columns:
        tiles.append(tile("Productividad media", num(df["productividad_tph"].mean(), 1), "t/h"))
    try:
        ci, _ = i_mr(df[var_i])
        tiles.append(tile("Señales de causa especial", str(len(ci.fuera_de_control)),
                          sub=f"sobre {len(ci.puntos)} observaciones",
                          estado="bueno" if ci.en_control else "serio",
                          etiqueta_estado="ESTABLE" if ci.en_control else "INVESTIGAR"))
    except ValueError:
        pass
    st.markdown(fila_tiles(tiles), unsafe_allow_html=True)

    cats = categoricas(df)
    if cats:
        st.markdown("##### Desagregación por factor")
        fac = st.selectbox("Agrupar por", cats, key="ind_fac")
        agg = df.groupby(fac, observed=True)[var_i].agg(
            n="count", media="mean", mediana="median",
            desv_std=lambda z: z.std(ddof=1), minimo="min", maximo="max").reset_index()
        agg["CV_%"] = (100 * agg["desv_std"] / agg["media"]).round(2)
        if vi and (vi.lie is not None or vi.lse is not None):
            def _cumple(z):
                ok = pd.Series(True, index=z.index)
                if vi.lie is not None:
                    ok &= z >= vi.lie
                if vi.lse is not None:
                    ok &= z <= vi.lse
                return 100 * ok.mean()
            agg["cumplimiento_%"] = df.groupby(fac, observed=True)[var_i].apply(
                _cumple).round(2).values
        ci1, ci2 = st.columns([1.1, 1])
        with ci1:
            st.plotly_chart(g.boxplot(df, var_i, fac, f"{var_i} por {fac}", ui), width="stretch")
        with ci2:
            tabla(agg.round(4), f"{caso.id}_indicadores_{fac}", f"Indicadores por {fac}")
        peor = agg.loc[agg["desv_std"].idxmax()]
        mejor = agg.loc[agg["desv_std"].idxmin()]
        st.info(f"El nivel **{peor[fac]}** presenta la mayor dispersión "
                f"(s = {num(peor['desv_std'],3)} {ui}) y **{mejor[fac]}** la menor "
                f"(s = {num(mejor['desv_std'],3)} {ui}). Una diferencia sistemática entre "
                f"niveles de *{fac}* es variación ENTRE subgrupos: aparece como señal en la "
                "carta de medias y hace que Cp supere a Pp.")

    st.markdown("##### Evolución temporal")
    st.plotly_chart(g.serie_temporal(df.assign(_i=range(1, len(df) + 1)), "_i", var_i,
                                     cats[0] if cats else None,
                                     f"Evolución de {var_i} en orden de registro", ui),
                    width="stretch")

# ====================================================== E. PLAN DE CONTROL
with t_plan:
    ficha("plan_control")
    st.markdown("#### Plan de control del proceso")
    st.caption("Edite la tabla según su operación. Es el entregable que sostiene la mejora una "
               "vez disuelto el equipo del proyecto.")

    criticas = [v for v in caso.variables if v.rol in ("respuesta", "componente")][:6]
    filas = []
    for v in criticas:
        lim = []
        if v.lie is not None:
            lim.append(f"LIE {v.lie}")
        if v.lse is not None:
            lim.append(f"LSE {v.lse}")
        filas.append({
            "Variable controlada": f"{v.etiqueta} ({v.unidad})" if v.unidad else v.etiqueta,
            "Método de medición": "Sistema de despacho / registro automático"
            if "min" in v.unidad else "Instrumento de planta / laboratorio",
            "Frecuencia": "Cada ciclo" if "min" in v.unidad else "Por turno",
            "Tamaño de subgrupo": 5,
            "Responsable": "Jefe de Turno",
            "Tipo de límite": "Especificación" if lim else "Control (calculado)",
            "Límites": ", ".join(lim) if lim else "Calculados de la carta",
            "Herramienta de control": "Carta Xbar-R" if "min" in v.unidad else "Carta I-MR",
            "Acción correctiva": "Detener, registrar la causa, escalar a Ingeniería de Procesos "
                                 "si se repite en dos turnos consecutivos",
        })
    plan = st.data_editor(pd.DataFrame(filas), width="stretch", hide_index=True,
                          num_rows="dynamic", key="plan_editor",
                          column_config={
                              "Frecuencia": st.column_config.SelectboxColumn(
                                  options=["Cada ciclo", "Por hora", "Por turno", "Diaria",
                                           "Semanal"]),
                              "Tipo de límite": st.column_config.SelectboxColumn(
                                  options=["Control (calculado)", "Especificación", "Ambos"]),
                          })
    tabla(plan, f"{caso.id}_plan_control", descargar=True)

    congelados = st.session_state["limites_congelados"]
    if congelados:
        st.markdown("##### Líneas base congeladas en esta sesión")
        st.caption("Estos son los límites que el plan de control debe citar. Expórtelos junto "
                   "con el plan: sin ellos, nadie puede reproducir la vigilancia.")
        resumen = []
        for clave, info in congelados.items():
            resumen.append({"variable": info["variable"], "carta": info["tipo"],
                            "línea central": round(info["limites"].centro, 6),
                            "puntos base": info["limites"].k_base,
                            "excluidos": len(info["limites"].excluidos),
                            "contexto": info["contexto"]})
        tabla(pd.DataFrame(resumen), f"{caso.id}_lineas_base", descargar=True)
    else:
        st.info("Todavía no ha congelado ninguna línea base. Hágalo en la pestaña A o C: un plan "
                "de control que no cita límites concretos no es verificable.")

    st.markdown(nota(
        "<b>Dos errores frecuentes en los planes de control.</b> El primero es confundir límites "
        "de control con límites de especificación: los primeros se calculan del proceso y sirven "
        "para detectar cambios; los segundos vienen del cliente y definen el defecto. Un proceso "
        "puede estar en control y aun así incumplir la especificación, y al revés. El segundo es "
        "dejar la acción correctiva sin escribir: si no está definida de antemano, ante una "
        "señal cada turno improvisará una respuesta distinta."), unsafe_allow_html=True)

    st.markdown("##### Criterios de reacción ante señales")
    reaccion = pd.DataFrame([
        {"Señal": f"Regla {r}", "Descripción": NELSON_DESC[r],
         "Interpretación habitual": inter, "Acción inmediata": acc}
        for r, inter, acc in [
            (1, "Cambio brusco o evento puntual (falla, error de registro, condición anómala)",
             "Detener, verificar el dato, investigar el evento del subgrupo señalado"),
            (2, "Corrimiento sostenido del nivel medio del proceso",
             "Revisar cambios recientes: personal, equipo, material, procedimiento"),
            (3, "Deriva progresiva (desgaste, fatiga, agotamiento de insumo)",
             "Revisar variables que cambian de forma continua en el tiempo"),
            (5, "Corrimiento moderado del nivel medio",
             "Verificar si coincide con un cambio de turno o de condición operacional"),
            (6, "Corrimiento pequeño pero sostenido",
             "Monitorear y confirmar con los subgrupos siguientes antes de actuar"),
            (7, "Estratificación: los subgrupos mezclan fuentes distintas",
             "Revisar el criterio de subagrupación: probablemente no es racional"),
            (8, "Mezcla: dos poblaciones distintas en el mismo flujo",
             "Separar las fuentes y graficarlas por separado"),
        ]])
    st.dataframe(reaccion, width="stretch", hide_index=True)
    st.download_button("Descargar criterios de reacción en CSV", a_csv(reaccion),
                       file_name=f"{caso.id}_criterios_reaccion.csv", mime="text/csv")

    st.markdown("##### Revisión del propio plan")
    st.markdown(
        "- Recalcular los límites de control solo cuando exista una **razón documentada**: un "
        "cambio deliberado del proceso, no una racha incómoda.\n"
        "- Revisar el plan al menos cada seis meses, y cada vez que cambie el equipo, el "
        "material o el procedimiento.\n"
        "- Revalidar el sistema de medición (Gage R&R) cuando se cambie el instrumento o el "
        "criterio de medición.\n"
        "- Verificar que las acciones correctivas efectivamente se ejecutaron: un plan que nadie "
        "aplica no controla nada.")
