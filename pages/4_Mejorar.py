"""Etapa MEJORAR: diseño de experimentos, simulación de escenarios y priorización."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from components import graficos as g
from components import interpretacion as I
from components.tema import (CSS, ESTADO, SERIE, SUPERFICIE, TINTA_2, TINTA_MUTE,
                             fila_tiles, nota, num, registrar_tema, tile)
from components.ui import (advertencias, cabecera, ficha, interpretar, selector_variable,
                           tabla, unidad_de)
from data.generators import caso1_ciclo_camiones, datos_doe_carguio, datos_doe_molienda
from statistics_engine.descriptive import describe
from statistics_engine.doe import (factorial_analysis, interaction_plot_data,
                                   main_effect_plot_data)
from statistics_engine.inference import two_sample_t
from statistics_engine.montecarlo import Incierto, van_montecarlo
from utils.estado import barra_lateral, datos_actuales
from utils.io import a_csv

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()

cabecera("Diseñar y evaluar las mejoras", "ETAPA 4 — MEJORAR",
         "El objetivo es generar y probar soluciones. El diseño de experimentos es la única "
         "herramienta de todo el recorrido que permite afirmaciones causales, porque manipula "
         "los factores en vez de observarlos. La simulación extiende esas conclusiones a "
         "escenarios completos, y la priorización decide cual implementar primero.")

t_doe, t_sim, t_pri = st.tabs(
    ["A · Diseño de experimentos (DOE)", "B · Simulación de escenarios",
     "C · Priorización y costo-beneficio"])

# ====================================================== A. DOE
with t_doe:
    ficha("doe")
    diseno = st.radio(
        "Experimento",
        ["2^2 · Carguio: capacidad de balde x posición del camion",
         "2^3 · Flotación: pH x dosis de colector x P80 de molienda"],
        key="doe_sel")

    if diseno.startswith("2^2"):
        c1, c2, c3 = st.columns(3)
        with c1:
            reps = st.slider("Replicas por corrida", 1, 10, 3, key="doe_rep")
        with c2:
            semilla = st.number_input("Semilla", 1, 999999, 2024, key="doe_sem")
        with c3:
            alfa = st.select_slider("alfa", [0.01, 0.05, 0.10], value=0.05, key="doe_alfa")
        datos = datos_doe_carguio(replicas=reps, semilla=int(semilla))
        factores, resp, unidad = ["A", "B"], "t_carguio_min", "min"
        nombres = {"A": "Capacidad de balde (28 vs 42 m3)",
                   "B": "Posición del camion (simple vs doble)"}
        st.caption("**Factor A:** capacidad de balde, 28 m3 (bajo) frente a 42 m3 (alto). "
                   "**Factor B:** posición del camion, carga por un solo lado (bajo) frente a "
                   "carga por ambos lados (alto). **Respuesta:** tiempo de carguio en minutos.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            reps = st.slider("Replicas por corrida", 1, 6, 1, key="doe3_rep")
        with c2:
            semilla = st.number_input("Semilla", 1, 999999, 2024, key="doe3_sem")
        alfa = 0.05
        datos = datos_doe_molienda(replicas=reps, semilla=int(semilla))
        factores, resp, unidad = ["A", "B", "C"], "recuperacion_pct", "%"
        nombres = {"A": "pH de flotación (10.2 vs 11.0)",
                   "B": "Dosis de colector (24 vs 34 g/t)",
                   "C": "P80 de molienda (160 vs 205 um)"}
        st.caption("Diseño 2^3. Con una sola replica no existe estimación independiente del "
                   "error y se recurre al método de Lenth, que tiene poca potencia. "
                   "Suba el número de replicas y observe como cambian las conclusiones.")

    r = factorial_analysis(datos, factores, resp, alfa=alfa)

    st.markdown("##### Matriz experimental")
    cm1, cm2 = st.columns([1.1, 1])
    with cm1:
        cols_show = (["corrida", "orden_ejecucion"] if "orden_ejecucion" in datos.columns
                     else ["corrida"]) + factores + \
                    [c for c in datos.columns if c.endswith("_real") or c in
                     ("capacidad_balde_m3", "posicion", "ph", "colector_g_t", "p80_um")] + \
                    ["replica", resp]
        cols_show = [c for c in dict.fromkeys(cols_show) if c in datos.columns]
        tabla(datos[cols_show], f"{caso.id}_doe_matriz", altura=300)
    with cm2:
        st.markdown("**Medias por combinación**")
        st.dataframe(r["medias_por_corrida"].round(4), width="stretch", hide_index=True)
        st.markdown("**Factores**")
        for k, v in nombres.items():
            st.markdown(f"- **{k}**: {v}")

    st.markdown("##### Efectos estimados")
    tiles = []
    for row in r["efectos"].sort_values("efecto", key=np.abs, ascending=False).head(4).itertuples():
        sig = row.termino in r.get("significativos", [])
        tiles.append(tile(row.termino, num(row.efecto, 3), unidad,
                          sub=f"{num(row.pct_contribucion,1)}% de la SC",
                          estado="bueno" if sig else None,
                          etiqueta_estado="SIGNIFICATIVO" if sig else ""))
    st.markdown(fila_tiles(tiles), unsafe_allow_html=True)
    st.code(r["ecuacion_codificada"], language="text")

    cd1, cd2 = st.columns(2)
    with cd1:
        st.plotly_chart(g.efectos_principales(main_effect_plot_data(datos, factores, resp),
                                              f"{resp} ({unidad})"), width="stretch")
    with cd2:
        pares = [(factores[i], factores[j]) for i in range(len(factores))
                 for j in range(i + 1, len(factores))]
        par = st.selectbox("Interacción a visualizar", pares,
                           format_func=lambda p: f"{p[0]} x {p[1]}", key="doe_par")
        st.plotly_chart(g.interaccion(interaction_plot_data(datos, par[0], par[1], resp),
                                      par[0], par[1], f"{resp} ({unidad})"),
                        width="stretch")

    lim = r.get("margen_error_ME")
    st.plotly_chart(g.pareto_efectos(r["efectos"], lim,
                                     f"Margen de Lenth ({num(lim,3)})" if lim else ""),
                    width="stretch")

    if "tabla_anova" in r:
        tabla(r["tabla_anova"].round(5), f"{caso.id}_doe_anova", "Tabla ANOVA del diseño")
    if "tabla_lenth" in r:
        tabla(r["tabla_lenth"].round(5), f"{caso.id}_doe_lenth", "Análisis de Lenth")

    advertencias(r["advertencias"], "Advertencias del diseño")
    interpretar("doe", lambda: I.doe(r, unidad))

    st.divider()
    st.markdown("##### Combinación optima según el experimento")
    medias = r["medias_por_corrida"]
    minimizar = st.radio("Objetivo", ["Minimizar la respuesta", "Maximizar la respuesta"],
                         index=0 if resp.startswith("t_") else 1, horizontal=True, key="doe_obj")
    mejor = (medias.loc[medias["media"].idxmin()] if minimizar.startswith("Minimizar")
             else medias.loc[medias["media"].idxmax()])
    combo = ", ".join(f"{f} = {'alto (+1)' if mejor[f] == 1 else 'bajo (-1)'}" for f in factores)
    st.success(f"La combinación observada con mejor resultado es **{combo}**, con media "
               f"**{num(mejor['media'], 3)} {unidad}** (n = {int(mejor['n'])}). "
               f"La peor combinación da "
               f"{num(medias['media'].max() if minimizar.startswith('Minimizar') else medias['media'].min(), 3)}"
               f" {unidad}: la diferencia entre ambas es la ganancia máxima alcanzable dentro "
               "del rango experimentado.")
    st.caption("Esta conclusión es válida solo dentro de los niveles ensayados. Un diseño 2^k no "
               "detecta curvatura, de modo que el optimo podría estar en un nivel intermedio "
               "no probado o fuera del rango; para explorarlo se requieren puntos centrales o "
               "un diseño de superficie de respuesta.")

# ====================================================== B. SIMULACION
with t_sim:
    ficha("simulacion")
    if caso.id != "caso1":
        st.info("El simulador paramétrico detallado está construido sobre el modelo de carguio y "
                "transporte (Caso 1). Para los demas casos se dispone de la comparación "
                "antes/después del escenario de mejora, que aparece más abajo.")

    if caso.id == "caso1":
        st.markdown("##### Parámetros operacionales")
        st.caption("Modifique los parámetros del proceso y observe su efecto sobre el ciclo, "
                   "la productividad y la capacidad de la flota. Cada escenario se recalcula "
                   "íntegramente: no hay resultados precargados.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_espera = st.slider("Factor de espera en pala", 0.1, 1.5, 1.0, 0.05,
                                 help="1.0 = situación actual. Representa el efecto de un "
                                      "sistema de despacho asistido sobre la cola en pala.")
            n_camiones = st.slider("Camiones en la flota", 2, 12, 6)
        with c2:
            mejora_via = st.slider("Mejora del estado de via (puntos del índice)", 0.0, 2.0, 0.0, 0.25,
                                   help="Desplaza el índice de condición de la carpeta de rodado.")
            horas = st.slider("Horas efectivas por día", 8, 22, 20)
        with c3:
            f_carguio = st.slider("Factor de tiempo de carguio", 0.6, 1.3, 1.0, 0.05,
                                  help="Efecto de un balde de mayor capacidad: menos pases.")
            semilla_s = st.number_input("Semilla del escenario", 1, 999999, 2024, key="sim_sem")
        with c4:
            n_sim = st.slider("Ciclos a simular", 100, 2000, 600, 100)
            meta_lse = st.number_input("Meta de ciclo (LSE, min)", value=38.0, format="%.1f")

        base = caso1_ciclo_camiones(n=n_sim, semilla=int(semilla_s))
        esc = base.copy()
        # Se recalcula el ciclo aplicando los factores a los componentes correspondientes
        esc["t_espera_pala_min"] = (base["t_espera_pala_min"] * f_espera).round(3)
        esc["t_carguio_min"] = (base["t_carguio_min"] * f_carguio).round(3)
        if mejora_via > 0:
            # Mejor via implica mayor velocidad: el tiempo de recorrido baja proporcionalmente
            deficit_base = (5 - base["estado_via"]).clip(lower=0)
            deficit_nuevo = (5 - (base["estado_via"] + mejora_via).clip(upper=5)).clip(lower=0)
            v_car_b = (22.0 - 0.95 * base["pendiente_pct"] - 0.80 * deficit_base).clip(lower=6)
            v_car_n = (22.0 - 0.95 * base["pendiente_pct"] - 0.80 * deficit_nuevo).clip(lower=6)
            v_vac_b = (32.0 - 0.70 * base["pendiente_pct"] - 0.60 * deficit_base).clip(lower=8)
            v_vac_n = (32.0 - 0.70 * base["pendiente_pct"] - 0.60 * deficit_nuevo).clip(lower=8)
            esc["t_transporte_min"] = (base["t_transporte_min"] * v_car_b / v_car_n).round(3)
            esc["t_retorno_min"] = (base["t_retorno_min"] * v_vac_b / v_vac_n).round(3)
        comp = ["t_carguio_min", "t_transporte_min", "t_descarga_min",
                "t_retorno_min", "t_espera_pala_min"]
        esc["ciclo_total_min"] = esc[comp].sum(axis=1).round(3)
        esc["productividad_tph"] = (esc["carga_ton"] / (esc["ciclo_total_min"] / 60)).round(2)

        db, de = describe(base["ciclo_total_min"]), describe(esc["ciclo_total_min"])
        cap_b = n_camiones * (60 / db.media) * base["carga_ton"].mean() * horas
        cap_e = n_camiones * (60 / de.media) * esc["carga_ton"].mean() * horas
        fuera_b = 100 * (base["ciclo_total_min"] > meta_lse).mean()
        fuera_e = 100 * (esc["ciclo_total_min"] > meta_lse).mean()

        st.markdown("##### Resultado del escenario")
        st.markdown(fila_tiles([
            tile("Ciclo medio", num(de.media, 2), "min",
                 sub=f"base {num(db.media,2)} · cambio {num(de.media-db.media,2)}",
                 estado="bueno" if de.media < db.media else "alerta",
                 etiqueta_estado=("MEJORA" if de.media < db.media else
                                  "SIN CAMBIO" if abs(de.media-db.media) < 1e-9 else "EMPEORA")),
            tile("Desv. estándar del ciclo", num(de.desv_std, 2), "min",
                 sub=f"base {num(db.desv_std,2)} · cambio {num(de.desv_std-db.desv_std,2)}"),
            tile("Productividad media", num(esc["productividad_tph"].mean(), 1), "t/h",
                 sub=f"base {num(base['productividad_tph'].mean(),1)} t/h"),
            tile("Capacidad de flota", num(cap_e, 0), "t/dia",
                 sub=f"base {num(cap_b,0)} · {num(cap_e-cap_b,0)} t/día adicionales",
                 estado="bueno" if cap_e > cap_b else None,
                 etiqueta_estado="GANANCIA" if cap_e > cap_b else ""),
            tile("Ciclos sobre la meta", num(fuera_e, 1), "%",
                 sub=f"base {num(fuera_b,1)}%",
                 estado="bueno" if fuera_e < fuera_b else "alerta",
                 etiqueta_estado="MENOS INCUMPLIMIENTO" if fuera_e < fuera_b else ""),
        ]), unsafe_allow_html=True)

        cs1, cs2 = st.columns(2)
        with cs1:
            st.plotly_chart(g.comparacion_escenarios(
                base["ciclo_total_min"], esc["ciclo_total_min"], "Línea base", "Escenario",
                "Distribución del tiempo de ciclo", "min"), width="stretch")
        with cs2:
            comp_df = pd.DataFrame({
                "Componente": ["Carguio", "Transporte cargado", "Descarga", "Retorno",
                               "Espera en pala"],
                "Base (min)": [base[c].mean() for c in comp],
                "Escenario (min)": [esc[c].mean() for c in comp],
            })
            comp_df["Cambio (min)"] = comp_df["Escenario (min)"] - comp_df["Base (min)"]
            comp_df["Cambio (%)"] = 100 * comp_df["Cambio (min)"] / comp_df["Base (min)"]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=comp_df["Componente"], x=comp_df["Base (min)"],
                                 orientation="h", name="Línea base",
                                 marker=dict(color=SERIE[0], line=dict(color=SUPERFICIE, width=2)),
                                 hovertemplate="%{y}<br>base %{x:.2f} min<extra></extra>"))
            fig.add_trace(go.Bar(y=comp_df["Componente"], x=comp_df["Escenario (min)"],
                                 orientation="h", name="Escenario",
                                 marker=dict(color=SERIE[1], line=dict(color=SUPERFICIE, width=2)),
                                 hovertemplate="%{y}<br>escenario %{x:.2f} min<extra></extra>"))
            fig.update_layout(title="Composición del ciclo por componente", height=420,
                              barmode="group", bargap=0.28, bargroupgap=0.08,
                              xaxis_title="Minutos promedio")
            st.plotly_chart(fig, width="stretch")
        tabla(comp_df.round(3), f"{caso.id}_simulacion_componentes",
              "Descomposición del ciclo", descargar=True)

        t_sim_res = two_sample_t(base["ciclo_total_min"], esc["ciclo_total_min"])
        interpretar("simulacion", lambda: I.comparacion(
            base["ciclo_total_min"], esc["ciclo_total_min"], "Tiempo de ciclo", "min",
            t_sim_res if abs(f_espera - 1) + abs(f_carguio - 1) + mejora_via > 0 else None))

    st.divider()
    st.markdown("##### Comparación antes / después del escenario de mejora del caso")
    st.caption("Escenario de mejora definido en el generador del caso: para carguio y transporte "
               "combina despacho asistido y mantenimiento de vias.")
    base_c = datos_actuales(mejora=False)
    mej_c = datos_actuales(mejora=True)
    var_cmp = selector_variable(base_c, caso, "cmp_var", "Variable a comparar")
    ucmp = unidad_de(caso, var_cmp)
    n = min(len(base_c), len(mej_c))
    t_cmp = two_sample_t(base_c[var_cmp].iloc[:n], mej_c[var_cmp].iloc[:n])
    da, dd = describe(base_c[var_cmp]), describe(mej_c[var_cmp])
    st.markdown(fila_tiles([
        tile("Media antes", num(da.media, 3), ucmp, sub=f"s = {num(da.desv_std,3)}"),
        tile("Media después", num(dd.media, 3), ucmp, sub=f"s = {num(dd.desv_std,3)}"),
        tile("Cambio en la media", num(dd.media - da.media, 3), ucmp,
             sub=f"{num(100*(dd.media-da.media)/da.media,1)}%"),
        tile("Cambio en la dispersión", num(dd.desv_std - da.desv_std, 3), ucmp,
             sub=f"{num(100*(dd.desv_std-da.desv_std)/da.desv_std,1)}%"),
        tile("p-valor de la diferencia", num(t_cmp.p_valor, 5),
             estado="bueno" if t_cmp.rechaza_h0 else "alerta",
             etiqueta_estado="DIFERENCIA DETECTADA" if t_cmp.rechaza_h0 else "NO CONCLUYENTE"),
    ]), unsafe_allow_html=True)
    st.plotly_chart(g.comparacion_escenarios(base_c[var_cmp], mej_c[var_cmp],
                                             "Antes de la mejora", "Después de la mejora",
                                             f"Comparación de {var_cmp}", ucmp),
                    width="stretch")
    interpretar("cmp_caso", lambda: I.comparacion(base_c[var_cmp], mej_c[var_cmp],
                                                  var_cmp, ucmp, t_cmp))

# ====================================================== C. PRIORIZACION
with t_pri:
    ficha("priorizacion")
    st.markdown("##### Alternativas de mejora")
    st.caption("Edite la tabla: impacto y esfuerzo en escala 1 a 10, costo e ingreso anual en "
               "unidades monetarias. Los indicadores se recalculan sobre lo que usted ingrese.")

    alternativas_base = pd.DataFrame([
        {"Alternativa": "Despacho asistido por tiempos de ciclo por ruta", "Impacto": 9,
         "Esfuerzo": 6, "Inversion": 180000.0, "Beneficio_anual": 420000.0, "Riesgo": "Medio"},
        {"Alternativa": "Programa de mantenimiento de vias", "Impacto": 7,
         "Esfuerzo": 4, "Inversion": 95000.0, "Beneficio_anual": 260000.0, "Riesgo": "Bajo"},
        {"Alternativa": "Estandarizar posicionamiento de carga doble", "Impacto": 5,
         "Esfuerzo": 2, "Inversion": 12000.0, "Beneficio_anual": 85000.0, "Riesgo": "Bajo"},
        {"Alternativa": "Balde de mayor capacidad en la pala principal", "Impacto": 8,
         "Esfuerzo": 9, "Inversion": 950000.0, "Beneficio_anual": 480000.0, "Riesgo": "Alto"},
        {"Alternativa": "Capacitación de operadores en conducción eficiente", "Impacto": 4,
         "Esfuerzo": 3, "Inversion": 28000.0, "Beneficio_anual": 70000.0, "Riesgo": "Bajo"},
    ])
    alt = st.data_editor(alternativas_base, width="stretch", hide_index=True,
                         num_rows="dynamic", key="alt_editor",
                         column_config={
                             "Impacto": st.column_config.NumberColumn(min_value=1, max_value=10),
                             "Esfuerzo": st.column_config.NumberColumn(min_value=1, max_value=10),
                             "Riesgo": st.column_config.SelectboxColumn(
                                 options=["Bajo", "Medio", "Alto"]),
                         })

    c1, c2 = st.columns(2)
    with c1:
        tasa = st.slider("Tasa de descuento anual (%)", 0.0, 30.0, 12.0, 0.5)
    with c2:
        horizonte = st.slider("Horizonte de evaluación (años)", 1, 10, 5)

    a = alt.dropna(subset=["Alternativa"]).copy()
    if a.empty:
        st.warning("Agregue al menos una alternativa.")
    else:
        a["Prioridad (I/E)"] = (a["Impacto"] / a["Esfuerzo"]).round(3)
        r = tasa / 100
        a["VAN"] = [round(sum(row.Beneficio_anual / (1 + r) ** t for t in range(1, horizonte + 1))
                          - row.Inversion, 0) for row in a.itertuples()]
        a["Payback (años)"] = (a["Inversion"] / a["Beneficio_anual"]).round(2)
        a["Cuadrante"] = np.where(
            (a["Impacto"] >= 6) & (a["Esfuerzo"] < 6), "1. Ganancia rápida",
            np.where((a["Impacto"] >= 6) & (a["Esfuerzo"] >= 6), "2. Proyecto mayor",
                     np.where((a["Impacto"] < 6) & (a["Esfuerzo"] < 6), "3. Relleno",
                              "4. Descartar")))

        cp1, cp2 = st.columns([1.2, 1])
        with cp1:
            fig = go.Figure()
            fig.add_vline(x=5.5, line=dict(color=TINTA_MUTE, width=1.5, dash="dot"))
            fig.add_hline(y=5.5, line=dict(color=TINTA_MUTE, width=1.5, dash="dot"))
            for etiqueta, xx, yy in (("Ganancia rápida", 3, 9.4), ("Proyecto mayor", 8.4, 9.4),
                                     ("Relleno", 3, 1.4), ("Descartar", 8.4, 1.4)):
                fig.add_annotation(x=xx, y=yy, text=etiqueta, showarrow=False,
                                   font=dict(size=11, color=TINTA_MUTE))
            fig.add_trace(go.Scatter(
                x=a["Esfuerzo"], y=a["Impacto"], mode="markers+text",
                text=[t[:26] for t in a["Alternativa"]], textposition="bottom center",
                textfont=dict(size=10, color=TINTA_2),
                marker=dict(size=np.clip(a["VAN"] / max(abs(a["VAN"]).max(), 1) * 28 + 12, 12, 42),
                            color=SERIE[0], opacity=0.8,
                            line=dict(color=SUPERFICIE, width=2)),
                customdata=np.stack([a["VAN"], a["Payback (años)"]], axis=-1),
                name="Alternativas",
                hovertemplate="%{text}<br>Esfuerzo %{x} · Impacto %{y}"
                              "<br>VAN %{customdata[0]:,.0f}"
                              "<br>Payback %{customdata[1]:.2f} años<extra></extra>"))
            fig.update_layout(title="Matriz esfuerzo-impacto (tamaño proporcional al VAN)",
                              height=460, xaxis_title="Esfuerzo requerido (1-10)",
                              yaxis_title="Impacto esperado (1-10)",
                              xaxis=dict(range=[0, 11]), yaxis=dict(range=[0, 11]),
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with cp2:
            st.markdown("**Ranking por prioridad impacto/esfuerzo**")
            st.dataframe(a.sort_values("Prioridad (I/E)", ascending=False)[
                ["Alternativa", "Prioridad (I/E)", "Cuadrante"]].round(3),
                width="stretch", hide_index=True)
            st.markdown("**Ranking económico**")
            st.dataframe(a.sort_values("VAN", ascending=False)[
                ["Alternativa", "VAN", "Payback (años)", "Riesgo"]],
                width="stretch", hide_index=True)

        tabla(a.round(3), f"{caso.id}_priorizacion", "Análisis completo exportable")

        rapidas = a[a["Cuadrante"] == "1. Ganancia rápida"]["Alternativa"].tolist()
        descartar = a[a["Cuadrante"] == "4. Descartar"]["Alternativa"].tolist()
        mejor_van = a.loc[a["VAN"].idxmax()]
        mejor_ie = a.loc[a["Prioridad (I/E)"].idxmax()]
        if rapidas:
            st.success(f"**Ganancias rápidas** (alto impacto, bajo esfuerzo): {', '.join(rapidas)}. "
                       "Se ejecutan primero porque liberan beneficio sin competir por presupuesto.")
        if descartar:
            st.info(f"**Descartar explícitamente**: {', '.join(descartar)}. Dejar constancia del "
                    "descarte evita que la alternativa reaparezca sin información nueva.")
        if mejor_van["Alternativa"] != mejor_ie["Alternativa"]:
            st.warning(
                f"**Los dos criterios discrepan.** Por impacto/esfuerzo la primera es "
                f"*{mejor_ie['Alternativa']}*; por VAN es *{mejor_van['Alternativa']}* "
                f"(VAN {num(mejor_van['VAN'],0)}, payback {num(mejor_van['Payback (años)'],2)} "
                f"años, riesgo {mejor_van['Riesgo']}). La discrepancia es información: el VAN "
                "pondera magnitud económica y el índice impacto/esfuerzo pondera facilidad de "
                "ejecución. Decida explicitando cual criterio domina y por que.")
        else:
            st.success(f"Ambos criterios coinciden en **{mejor_van['Alternativa']}** como "
                       "primera prioridad.")

        st.markdown(nota(
            "<b>El VAN de arriba es una cifra puntual y eso es engañoso.</b> Los beneficios "
            "anuales se derivan de la simulación del proceso, la inversión es una estimación y "
            "la tasa de descuento es una convención. Las tres son inciertas, y un VAN calculado "
            "con un solo valor de cada una no dice nada sobre el riesgo de la decisión. "
            "La sección siguiente propaga esa incertidumbre."), unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Monte Carlo: el VAN como distribución, no como cifra")
        ficha("montecarlo")

        alt_sel = st.selectbox("Alternativa a evaluar", a["Alternativa"].tolist(), key="mc_alt")
        fila = a[a["Alternativa"] == alt_sel].iloc[0]
        inv0, ben0 = float(fila["Inversion"]), float(fila["Beneficio_anual"])

        st.caption("Declare lo que sabe y lo que no de cada entrada. Con juicio experto, la "
                   "triangular y la PERT (mínimo, más probable, máximo) suelen ser las formas "
                   "más honestas: no exigen suponer simetría ni colas infinitas.")
        cmc1, cmc2, cmc3 = st.columns(3)
        with cmc1:
            st.markdown("**Inversión**")
            d_inv = st.selectbox("Distribución", ["pert", "triangular", "uniforme", "normal",
                                                  "lognormal", "fija"], key="mc_dinv")
            if d_inv in ("pert", "triangular"):
                i_min = st.number_input("Mínimo", value=round(inv0 * 0.85), key="mc_imin")
                i_mod = st.number_input("Más probable", value=round(inv0), key="mc_imod")
                i_max = st.number_input("Máximo", value=round(inv0 * 1.45), key="mc_imax")
                par_inv = {"minimo": i_min, "moda": i_mod, "maximo": i_max}
            elif d_inv == "uniforme":
                par_inv = {"minimo": st.number_input("Mínimo", value=round(inv0 * 0.85),
                                                     key="mc_imin2"),
                           "maximo": st.number_input("Máximo", value=round(inv0 * 1.45),
                                                     key="mc_imax2")}
            elif d_inv == "fija":
                par_inv = {"valor": st.number_input("Valor", value=round(inv0), key="mc_ival")}
            else:
                par_inv = {"media": st.number_input("Media", value=round(inv0), key="mc_imed"),
                           "sd": st.number_input("Desv. estándar", value=round(inv0 * 0.15),
                                                 key="mc_isd")}
        with cmc2:
            st.markdown("**Beneficio anual**")
            d_ben = st.selectbox("Distribución", ["triangular", "pert", "uniforme", "normal",
                                                  "lognormal", "fija"], key="mc_dben")
            if d_ben in ("pert", "triangular"):
                b_min = st.number_input("Mínimo ", value=round(ben0 * 0.55), key="mc_bmin")
                b_mod = st.number_input("Más probable ", value=round(ben0), key="mc_bmod")
                b_max = st.number_input("Máximo ", value=round(ben0 * 1.25), key="mc_bmax")
                par_ben = {"minimo": b_min, "moda": b_mod, "maximo": b_max}
            elif d_ben == "uniforme":
                par_ben = {"minimo": st.number_input("Mínimo ", value=round(ben0 * 0.55),
                                                     key="mc_bmin2"),
                           "maximo": st.number_input("Máximo ", value=round(ben0 * 1.25),
                                                     key="mc_bmax2")}
            elif d_ben == "fija":
                par_ben = {"valor": st.number_input("Valor ", value=round(ben0), key="mc_bval")}
            else:
                par_ben = {"media": st.number_input("Media ", value=round(ben0), key="mc_bmed"),
                           "sd": st.number_input("Desv. estándar ", value=round(ben0 * 0.25),
                                                 key="mc_bsd")}
        with cmc3:
            st.markdown("**Tasa de descuento**")
            d_tas = st.selectbox("Distribución", ["uniforme", "triangular", "fija"],
                                 key="mc_dtas")
            if d_tas == "uniforme":
                par_tas = {"minimo": st.number_input("Mínima", value=max(tasa / 100 - 0.04, 0.0),
                                                     format="%.4f", key="mc_tmin"),
                           "maximo": st.number_input("Máxima", value=tasa / 100 + 0.04,
                                                     format="%.4f", key="mc_tmax")}
            elif d_tas == "triangular":
                par_tas = {"minimo": st.number_input("Mínima", value=max(tasa / 100 - 0.04, 0.0),
                                                     format="%.4f", key="mc_tmin2"),
                           "moda": st.number_input("Más probable", value=tasa / 100,
                                                   format="%.4f", key="mc_tmod"),
                           "maximo": st.number_input("Máxima", value=tasa / 100 + 0.04,
                                                     format="%.4f", key="mc_tmax2")}
            else:
                par_tas = {"valor": st.number_input("Valor", value=tasa / 100, format="%.4f",
                                                    key="mc_tval")}
            n_sim = st.select_slider("Simulaciones", [1_000, 5_000, 10_000, 50_000, 100_000],
                                     value=10_000, key="mc_n")
            semilla_mc = st.number_input("Semilla", 1, 999_999, 2024, key="mc_sem")

        try:
            inc_inv = Incierto("inversion", d_inv, par_inv, "USD", minimo_fisico=0.0)
            inc_ben = Incierto("beneficio_anual", d_ben, par_ben, "USD/año")
            inc_tas = Incierto("tasa", d_tas, par_tas)
            rmc = van_montecarlo(inc_inv, inc_ben, inc_tas, horizonte,
                                 n=int(n_sim), semilla=int(semilla_mc))
            rr = rmc.resumen()
            lo90, hi90 = rmc.intervalo(0.90)
            p_neg = rmc.probabilidad_menor_que(0.0)
            est = "bueno" if p_neg < 0.05 else "alerta" if p_neg < 0.25 else "critico"
            st.markdown(fila_tiles([
                tile("VAN esperado", num(rr["media"], 0), sub=f"mediana {num(rr['mediana'], 0)}"),
                tile("Intervalo 90%", f"{num(lo90, 0)} a {num(hi90, 0)}",
                     sub="percentiles 5 y 95"),
                tile("P(VAN < 0)", num(100 * p_neg, 1), "%", sub="probabilidad de destruir valor",
                     estado=est, etiqueta_estado={"bueno": "RIESGO BAJO",
                                                  "alerta": "RIESGO APRECIABLE",
                                                  "critico": "RIESGO ALTO"}[est]),
                tile("Desviación estándar", num(rr["desv_std"], 0)),
                tile("Error de muestreo", num(rmc.error_montecarlo(), 0),
                     sub=f"sobre {int(n_sim):,} simulaciones".replace(",", " ")),
            ]), unsafe_allow_html=True)

            cg1, cg2 = st.columns([1.3, 1])
            with cg1:
                st.plotly_chart(g.distribucion_montecarlo(rmc, umbral=0.0,
                                                          etiqueta_umbral="VAN = 0"),
                                width="stretch")
            with cg2:
                st.plotly_chart(g.tornado(rmc.sensibilidad(), "el VAN"), width="stretch")
            st.plotly_chart(g.convergencia_montecarlo(rmc.convergencia(), "el VAN"),
                            width="stretch")

            cd1, cd2 = st.columns(2)
            with cd1:
                tabla(rmc.tabla_resumen().round(4), f"{caso.id}_mc_resumen",
                      "Resumen de la distribución", descargar=True)
            with cd2:
                tabla(rmc.sensibilidad().round(4), f"{caso.id}_mc_sensibilidad",
                      "Contribución a la incertidumbre", descargar=True)
            st.caption("Entradas declaradas: "
                       + " · ".join(f"**{e.nombre}**: {e.descripcion()}"
                                    for e in (inc_inv, inc_ben, inc_tas)))
            interpretar("montecarlo", lambda: I.montecarlo(rmc, umbral=0.0,
                                                           etiqueta_umbral="el punto de equilibrio"))

            st.divider()
            st.markdown("##### Comparación de alternativas con la misma incertidumbre relativa")
            st.caption("Se aplican a cada alternativa las mismas formas de distribución y las "
                       "mismas proporciones de incertidumbre definidas arriba, para que la "
                       "comparación sea entre alternativas y no entre supuestos.")
            comparables = {}
            for row in a.head(3).itertuples():
                i0, b0 = float(row.Inversion), float(row.Beneficio_anual)
                comparables[row.Alternativa] = van_montecarlo(
                    Incierto("inversion", "pert",
                             {"minimo": i0 * 0.85, "moda": i0, "maximo": i0 * 1.45},
                             minimo_fisico=0.0),
                    Incierto("beneficio_anual", "triangular",
                             {"minimo": b0 * 0.55, "moda": b0, "maximo": b0 * 1.25}),
                    inc_tas, horizonte, n=int(min(n_sim, 20_000)), semilla=int(semilla_mc))
            st.plotly_chart(g.comparar_van(comparables), width="stretch")
            comp_mc = pd.DataFrame([
                {"Alternativa": k,
                 "VAN esperado": round(v.resumen()["media"], 0),
                 "VAN mediana": round(v.resumen()["mediana"], 0),
                 "P5": round(v.intervalo(0.90)[0], 0),
                 "P95": round(v.intervalo(0.90)[1], 0),
                 "P(VAN<0) %": round(100 * v.probabilidad_menor_que(0), 1)}
                for k, v in comparables.items()])
            tabla(comp_mc, f"{caso.id}_mc_comparacion", descargar=True)
            mejor_media = comp_mc.loc[comp_mc["VAN esperado"].idxmax(), "Alternativa"]
            menor_riesgo = comp_mc.loc[comp_mc["P(VAN<0) %"].idxmin(), "Alternativa"]
            if mejor_media != menor_riesgo:
                st.warning(
                    f"**El mayor VAN esperado y el menor riesgo no coinciden.** Por valor "
                    f"esperado conviene *{mejor_media}*; por probabilidad de destruir valor, "
                    f"*{menor_riesgo}*. Esa discrepancia es precisamente lo que una cifra "
                    "puntual ocultaba. La elección depende de cuánta pérdida es tolerable, "
                    "que es una decisión de la organización y no del modelo.")
            else:
                st.success(f"**{mejor_media}** domina en ambos criterios: mayor VAN esperado y "
                           "menor probabilidad de destruir valor.")
        except ValueError as e:
            st.error(f"No se pudo ejecutar la simulación: {e}")
