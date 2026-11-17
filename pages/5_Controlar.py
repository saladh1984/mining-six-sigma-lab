"""Etapa CONTROLAR: cartas de control, indicadores de seguimiento y plan de control."""
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
from statistics_engine.control_charts import (NELSON_DESC, c_chart, i_mr, nelson_rules,
                                              np_chart, p_chart, u_chart, xbar_r, xbar_s)
from statistics_engine.descriptive import describe
from utils.estado import barra_lateral, categoricas, datos_actuales
from utils.io import a_csv

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()

cabecera("Sostener la mejora en el tiempo", "ETAPA 5 — CONTROLAR",
         "El objetivo es detectar cuando el proceso cambia, para actuar sobre la causa y no "
         "sobre el síntoma. Una carta de control separa la variación por causas comunes "
         "(inherente, estable) de la variación por causas especiales (externa, corregible). "
         "Los límites de control se calculan a partir del proceso; no son límites de "
         "especificación.")

t_var, t_atr, t_ind, t_plan = st.tabs(
    ["A · Cartas de variables", "B · Cartas de atributos",
     "C · Indicadores de seguimiento", "D · Plan de control"])

# ====================================================== A. VARIABLES
with t_var:
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
    reglas_sel = st.multiselect(
        "Reglas de patrones no aleatorios a aplicar", list(NELSON_DESC),
        default=[1, 2, 3, 5, 6],
        format_func=lambda r: f"Regla {r}: {NELSON_DESC[r]}", key="cc_reglas",
        help="Activar las ocho reglas aumenta la detección pero también las falsas alarmas: "
             "la longitud media de racha en control cae de ~370 a ~91 subgrupos.")
    if len(reglas_sel) >= 8:
        st.warning("Con las ocho reglas activas, aproximadamente 1 de cada 91 subgrupos genera "
                   "una señal aunque el proceso este perfectamente estable.")

    try:
        if tipo == "I-MR":
            principal, secundaria = i_mr(df[var])
            desc_sub = "una observación por punto (sin subgrupos)"
        else:
            sub, desc_sub = subgrupos_racionales(df, var, estrato, n_sub)
            principal, secundaria = (xbar_r(sub) if tipo == "Xbar-R" else xbar_s(sub))
        st.caption(f"Criterio de subagrupación: {desc_sub}.")

        # Reaplicar solo las reglas seleccionadas a la carta principal
        principal.violaciones = nelson_rules(principal.puntos, principal.centro,
                                             principal.sigma, reglas_sel)

        st.warning(
            "**Orden de lectura correcto: primero la carta de dispersión.** Si el rango o la "
            "desviación estándar no son estables, los límites de la carta de medias se "
            "calcularon con un Rbar (o Sbar) contaminado y no son válidos.")

        st.plotly_chart(g.carta_control(secundaria), width="stretch")
        if secundaria.en_control:
            st.success(f"La {secundaria.nombre} no muestra señales: la dispersión del proceso es "
                       "estable, de modo que los límites de la carta de medias son válidos.")
        else:
            st.error(f"La {secundaria.nombre} muestra "
                     f"{len(secundaria.fuera_de_control)} señal(es): **la dispersión no es "
                     "estable**. Investigue esas causas antes de interpretar la carta de medias.")

        st.plotly_chart(g.carta_control(principal), width="stretch")
        n_sen = len(principal.fuera_de_control)
        st.markdown(fila_tiles([
            tile("Línea central", num(principal.centro[0], 4), u),
            tile("LCI", num(principal.lci[0], 4), u, sub="límite de CONTROL, no de especificación"),
            tile("LCS", num(principal.lcs[0], 4), u, sub="límite de CONTROL, no de especificación"),
            tile("Puntos graficados", str(len(principal.puntos))),
            tile("Señales detectadas", str(n_sen),
                 estado="bueno" if n_sen == 0 else "critico",
                 etiqueta_estado="EN CONTROL" if n_sen == 0 else "FUERA DE CONTROL"),
        ]), unsafe_allow_html=True)

        v = caso.var(var) if var in [x.nombre for x in caso.variables] else None
        if v and (v.lie is not None or v.lse is not None):
            lim = []
            if v.lie is not None:
                lim.append(f"LIE = {num(v.lie)}")
            if v.lse is not None:
                lim.append(f"LSE = {num(v.lse)}")
            st.info(
                f"**Especificación de esta variable: {', '.join(lim)} {u}.** Deliberadamente no "
                "se dibuja sobre la carta de medias: la media de subgrupos dispersa menos que las "
                "observaciones individuales (su desviación estándar es sigma/raiz(n)), de modo "
                "que comparar medias contra una especificación individual da una impresión "
                "falsamente optimista. La comparación correcta con la especificación es el "
                "análisis de capacidad del módulo Medir.")

        advertencias(principal.notas, "Notas de la carta")
        interpretar("carta_principal", lambda: I.carta(principal, var))

        cta, ctb = st.columns(2)
        with cta:
            tabla(principal.to_frame().round(5), f"{caso.id}_{var}_carta_principal",
                  f"Datos de la {principal.nombre}", altura=320)
        with ctb:
            tabla(secundaria.to_frame().round(5), f"{caso.id}_{var}_carta_dispersion",
                  f"Datos de la {secundaria.nombre}", altura=320)

        st.divider()
        st.markdown("##### Capacidad condicionada a la estabilidad")
        if principal.en_control and secundaria.en_control:
            if v and (v.lie is not None or v.lse is not None) and tipo != "I-MR":
                cap = capability(df[var], lie=v.lie, lse=v.lse, objetivo=v.objetivo, subgrupos=sub)
                st.success("El proceso está en control estadístico: los índices de capacidad SI "
                           "son interpretables como predicción del comportamiento futuro.")
                st.markdown(fila_tiles([
                    tile("Cp", num(cap.cp) if cap.cp else "n/a"),
                    tile("Cpk", num(cap.cpk)),
                    tile("Pp", num(cap.pp) if cap.pp else "n/a"),
                    tile("Ppk", num(cap.ppk)),
                ]), unsafe_allow_html=True)
            else:
                st.info("El proceso está en control. Defina límites de especificación en el "
                        "módulo Medir para evaluar la capacidad.")
        else:
            st.error("**El proceso NO está en control estadístico.** Calcular índices de "
                     "capacidad sobre estos datos produce números sin valor predictivo: el "
                     "proceso no tiene una distribución estable que proyectar. Primero hay que "
                     "identificar y eliminar las causas especiales.")
    except ValueError as e:
        st.error(f"No se pudo construir la carta: {e}")

# ====================================================== B. ATRIBUTOS
with t_atr:
    ficha("cartas")
    st.markdown(nota(
        "<b>Por que estas cartas usan su propio conjunto de datos.</b> Las cartas de atributos "
        "requieren CONTEOS (unidades defectuosas o defectos), no mediciones continuas. Se simula "
        "un registro de inspección por turno con una causa especial inyectada en el subgrupo 22, "
        "para que la carta tenga algo real que detectar."), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        carta_a = st.selectbox("Carta", ["p (proporción defectuosa)", "np (número de defectuosos)",
                                         "c (defectos por unidad constante)",
                                         "u (defectos por unidad variable)"], key="at_tipo")
    with c2:
        k = st.slider("Número de subgrupos", 15, 60, 30, key="at_k")
    with c3:
        p_real = st.slider("Proporción defectuosa del proceso", 0.005, 0.20, 0.045, 0.005,
                           key="at_p")

    da = datos_atributos(k=k, p_real=p_real)
    if carta_a.startswith("p "):
        res = p_chart(da["defectuosas"], da["n_inspeccionadas"])
        datos_carta = da[["subgrupo", "turno", "n_inspeccionadas", "defectuosas"]]
        explic = ("Modelo binomial con tamaño de muestra VARIABLE: los límites se estrechan "
                  "cuando n crece, porque una proporción estimada con más unidades es más precisa.")
    elif carta_a.startswith("np"):
        res = np_chart(da["defectuosas_const"], int(da["n_inspeccionadas_const"].iloc[0]))
        datos_carta = da[["subgrupo", "turno", "n_inspeccionadas_const", "defectuosas_const"]]
        explic = ("Modelo binomial con tamaño de muestra CONSTANTE. Si n variara, habría que "
                  "usar la carta p.")
    elif carta_a.startswith("c "):
        res = c_chart(da["defectos_area_constante"])
        datos_carta = da[["subgrupo", "turno", "defectos_area_constante"]]
        explic = ("Modelo Poisson con area de oportunidad CONSTANTE. Supone media igual a "
                  "varianza: si la varianza observada la excede, hay sobredispersión y el "
                  "modelo no aplica.")
    else:
        res = u_chart(da["defectos"], da["unidades_inspeccion"])
        datos_carta = da[["subgrupo", "turno", "unidades_inspeccion", "defectos"]]
        explic = ("Modelo Poisson con area de oportunidad VARIABLE: los límites cambian entre "
                  "subgrupos según cuántas unidades se inspeccionaron.")

    st.caption(explic)
    st.plotly_chart(g.carta_control(res), width="stretch")
    n_sen = len(res.fuera_de_control)
    st.markdown(fila_tiles([
        tile("Línea central", num(res.centro[0], 5)),
        tile("Subgrupos", str(len(res.puntos))),
        tile("Señales", str(n_sen), estado="bueno" if n_sen == 0 else "crítico",
             etiqueta_estado="EN CONTROL" if n_sen == 0 else "FUERA DE CONTROL"),
        tile("Límites", "constantes" if len(set(np.round(res.lcs, 9))) == 1 else "variables",
             sub="dependen del tamaño de cada subgrupo" if
             len(set(np.round(res.lcs, 9))) > 1 else "tamaño de muestra constante"),
    ]), unsafe_allow_html=True)
    st.info("Las cartas de atributos tienen estadísticos **asimétricos**, por lo que las reglas "
            "de zona (5 a 8), que suponen simetría y normalidad, no se aplican. Solo se usan "
            "las reglas 1 a 4.")
    advertencias(res.notas, "Notas y supuestos del modelo")
    interpretar("carta_atributos", lambda: I.carta(res))
    ca1, ca2 = st.columns(2)
    with ca1:
        tabla(datos_carta, f"{caso.id}_atributos_datos", "Registro de inspección", altura=300)
    with ca2:
        tabla(res.to_frame().round(6), f"{caso.id}_atributos_carta", "Carta exportable", altura=300)

# ====================================================== C. INDICADORES
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

    tiles = [
        tile("Media del proceso", num(d.media, 3), ui),
        tile("Desviación estándar", num(d.desv_std, 3), ui),
        tile("CV", num(100 * d.coef_variacion, 1) if d.coef_variacion else "-", "%"),
    ]
    if cumple is not None:
        est = "bueno" if cumple >= 99 else "alerta" if cumple >= 95 else "critico"
        tiles.append(tile("Cumplimiento de especificación", num(cumple, 2), "%",
                          sub=f"{num(100-cumple,2)}% de defectos", estado=est,
                          etiqueta_estado={"bueno": "CONFORME", "alerta": "ATENCION",
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
            st.plotly_chart(g.boxplot(df, var_i, fac, f"{var_i} por {fac}", ui),
                            width="stretch")
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
    orden = df.columns[0]
    st.plotly_chart(g.serie_temporal(df.assign(_i=range(1, len(df) + 1)), "_i", var_i,
                                     cats[0] if cats else None,
                                     f"Evolución de {var_i} en orden de registro", ui),
                    width="stretch")

# ====================================================== D. PLAN DE CONTROL
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
            "Tipo de límite": "Especificacion" if lim else "Control (calculado)",
            "Limites": ", ".join(lim) if lim else "Calculados de la carta",
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
                                  options=["Control (calculado)", "Especificacion", "Ambos"]),
                          })
    tabla(plan, f"{caso.id}_plan_control", descargar=True)

    st.markdown(nota(
        "<b>Dos errores frecuentes en los planes de control.</b> El primero es confundir límites "
        "de control con límites de especificación: los primeros se calculan del proceso y sirven "
        "para detectar cambios; los segundos vienen del cliente y definen el defecto. Un proceso "
        "puede estar en control y aún así incumplir la especificación, y al reves. El segundo es "
        "dejar la acción correctiva sin escribir: si no está definida de antemano, ante una "
        "señal cada turno improvisara una respuesta distinta."), unsafe_allow_html=True)

    st.markdown("##### Criterios de reacción ante señales")
    reaccion = pd.DataFrame([
        {"Senal": f"Regla {r}", "Descripcion": NELSON_DESC[r],
         "Interpretación habitual": inter, "Acción inmediata": acc}
        for r, inter, acc in [
            (1, "Cambio brusco o evento puntual (falla, error de registro, condición anomala)",
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
        "cambio deliberado del proceso, no una racha incomoda.\n"
        "- Revisar el plan al menos cada seis meses, y cada vez que cambie el equipo, el "
        "material o el procedimiento.\n"
        "- Revalidar el sistema de medición (Gage R&R) cuando se cambie el instrumento o el "
        "criterio de medición.\n"
        "- Verificar que las acciones correctivas efectivamente se ejecutaron: un plan que nadie "
        "aplica no controla nada.")
