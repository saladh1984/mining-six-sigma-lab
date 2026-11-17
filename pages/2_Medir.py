"""Etapa MEDIR: descriptiva, capacidad del proceso, sistema de medición y muestreo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


from components import graficos as g
from components import interpretacion as I
from components.tema import CSS, fila_tiles, nota, num, registrar_tema, tile
from components.ui import (advertencias, cabecera, ficha, interpretar, selector_variable,
                           tabla, unidad_de)
from data.generators import datos_gage_rr, subgrupos_racionales
from statistics_engine.capability import capability, intervalo_confianza_cpk
from statistics_engine.control_charts import i_mr
from statistics_engine.descriptive import boxplot_stats, describe, describe_frame, percentiles
from statistics_engine.gagerr import gage_rr_anova
from statistics_engine.inference import normality
from utils.estado import barra_lateral, categoricas, datos_actuales
from utils.io import a_csv, resultado_a_frame

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()

cabecera("Caracterizar el proceso con datos", "ETAPA 2 — MEDIR",
         "El objetivo es establecer la línea base: cuánto vale la variable crítica, cuánto varía, "
         "si cumple la especificación y si el sistema de medición es confiable. "
         "Sin una línea base medida, no hay forma de demostrar después que algo mejoro.")

t_desc, t_cap, t_msa, t_mue = st.tabs(
    ["A · Estadística descriptiva", "B · Capacidad del proceso",
     "C · Sistema de medición (Gage R&R)", "D · Muestreo"])

# ====================================================== A. DESCRIPTIVA
with t_desc:
    ficha("descriptiva")
    c1, c2 = st.columns([1, 1])
    with c1:
        var = selector_variable(df, caso, "medir_var")
    with c2:
        cats = categoricas(df)
        por = st.selectbox("Estratificar por (opcional)", ["Sin estratificar"] + cats,
                           key="medir_por",
                           help="Estratificar revela si el comportamiento global oculta grupos "
                                "con comportamientos distintos.")
        por = None if por == "Sin estratificar" else por
    u = unidad_de(caso, var)
    d = describe(df[var])

    st.markdown(fila_tiles([
        tile("n", str(d.n), sub=f"{d.n_missing} faltantes excluidos"),
        tile("Media", num(d.media), u),
        tile("Mediana", num(d.mediana), u),
        tile("Desv. estándar", num(d.desv_std), u),
        tile("CV", num(100 * d.coef_variacion, 1) if d.coef_variacion else "-", "%"),
        tile("IQR", num(d.iqr), u, sub=f"Q1 {num(d.q1)} · Q3 {num(d.q3)}"),
    ]), unsafe_allow_html=True)

    resumen = pd.DataFrame({
        "Medida": ["n", "Faltantes", "Media", "Mediana", "Moda", "Método de la moda",
                   "Desviación estándar", "Varianza", "Coeficiente de variación",
                   "Minimo", "Maximo", "Rango", "Q1", "Q2 (mediana)", "Q3", "IQR",
                   "Asimetria", "Curtosis en exceso", "Error estándar de la media"],
        # Se formatea a texto porque la columna mezcla numeros con la descripcion
        # del metodo de la moda; una columna de tipo mixto no es serializable a Arrow.
        "Valor": [str(d.n), str(d.n_missing), num(d.media, 4), num(d.mediana, 4),
                  num(d.moda, 4) if d.moda is not None else "sin moda", d.moda_metodo,
                  num(d.desv_std, 4), num(d.varianza, 4),
                  num(d.coef_variacion, 4) if d.coef_variacion is not None else "-",
                  num(d.minimo, 4), num(d.maximo, 4), num(d.rango, 4), num(d.q1, 4),
                  num(d.q2, 4), num(d.q3, 4), num(d.iqr, 4), num(d.asimetria, 4),
                  num(d.curtosis_exceso, 4), num(d.error_std_media, 4)],
    })

    cg1, cg2 = st.columns(2)
    with cg1:
        bins = st.slider("Intervalos del histograma", 0, 60, 0, 5,
                         help="0 = selección automática de Plotly.")
        v = caso.var(var) if var in [x.nombre for x in caso.variables] else None
        st.plotly_chart(g.histograma(df[var], f"Histograma de {var}", u, bins,
                                     lie=v.lie if v else None, lse=v.lse if v else None,
                                     objetivo=v.objetivo if v else None),
                        width="stretch")
        st.plotly_chart(g.densidad(df[var], f"Densidad de {var}", u,
                                   df[por] if por else None), width="stretch")
    with cg2:
        st.plotly_chart(g.boxplot(df, var, por, f"Diagrama de cajas de {var}", u),
                        width="stretch")
        num_cols = [c for c in df.select_dtypes("number").columns if c != var]
        if num_cols:
            xv = st.selectbox("Dispersión frente a", num_cols, key="medir_disp")
            from statistics_engine.regression import scatter_fit
            st.plotly_chart(g.dispersion(df[xv], df[var], scatter_fit(df[xv], df[var]),
                                         xv, var, bandas=False), width="stretch")

    cb1, cb2 = st.columns(2)
    with cb1:
        tabla(resumen, f"{caso.id}_{var}_descriptivo", "Resumen descriptivo")
    with cb2:
        tabla(percentiles(df[var]), f"{caso.id}_{var}_percentiles", "Percentiles")
        b = boxplot_stats(df[var])
        st.caption(f"Regla de Tukey (k = {b['k']}): límites [{num(b['limite_inferior_tukey'])}; "
                   f"{num(b['limite_superior_tukey'])}] {u}. "
                   f"**{b['n_outliers']} observación(es) fuera**. Un punto fuera no es "
                   "necesariamente un error: es un candidato a investigación.")

    if por:
        st.markdown("**Resumen por estrato**")
        g_res = df.groupby(por, observed=True)[var].agg(
            n="count", media="mean", mediana="median",
            desv_std=lambda z: z.std(ddof=1), minimo="min", maximo="max").reset_index()
        g_res["CV_%"] = 100 * g_res["desv_std"] / g_res["media"]
        tabla(g_res.round(4), f"{caso.id}_{var}_por_{por}", descargar=True)

    interpretar("descriptiva", lambda: I.descriptiva(d, var, u))

    st.divider()
    st.markdown("#### Verificación de normalidad")
    st.caption("Varios métodos posteriores (capacidad, pruebas t, cartas de individuales) "
               "suponen normalidad. Conviene verificarla aquí, no al final.")
    norm = normality(df[var])
    cn1, cn2 = st.columns([1, 1])
    with cn1:
        st.plotly_chart(g.qq(norm), width="stretch")
    with cn2:
        st.markdown(fila_tiles([
            tile("Shapiro-Wilk p", num(norm["shapiro"]["p_valor"], 5),
                 estado="critico" if norm["shapiro"]["rechaza_normalidad"] else "bueno",
                 etiqueta_estado="NO NORMAL" if norm["shapiro"]["rechaza_normalidad"]
                 else "COMPATIBLE"),
            tile("Asimetría", num(norm["asimetria"]), sub="0 en una normal exacta"),
            tile("Curtosis exceso", num(norm["curtosis_exceso"]), sub="0 en una normal exacta"),
        ]), unsafe_allow_html=True)
        interpretar("normalidad_medir", lambda: I.normalidad(norm, var))

    st.divider()
    with st.expander("Resumen descriptivo de todas las variables numéricas"):
        tabla(describe_frame(df).round(4), f"{caso.id}_descriptivo_general", altura=340)

# ====================================================== B. CAPACIDAD
with t_cap:
    ficha("capacidad")
    v = caso.var(caso.respuesta) if caso.respuesta in [x.nombre for x in caso.variables] else None
    var_c = selector_variable(df, caso, "cap_var", "Variable a evaluar")
    vc = caso.var(var_c) if var_c in [x.nombre for x in caso.variables] else None
    uc = unidad_de(caso, var_c)

    st.markdown("##### Límites de especificación")
    st.caption("Los límites de especificación los define el cliente o el requisito técnico. "
               "NO se calculan a partir de los datos y NO son límites de control.")
    if vc and vc.origen_especificacion:
        st.info(f"**Origen de la especificación por defecto:** {vc.origen_especificacion}")

    cc1, cc2, cc3, cc4 = st.columns(4)
    rango = float(df[var_c].max() - df[var_c].min())
    with cc1:
        usar_lie = st.checkbox("Definir LIE", value=bool(vc and vc.lie is not None), key="cap_ulie")
        lie = st.number_input("LIE", value=float(vc.lie) if vc and vc.lie is not None
                              else float(df[var_c].min() - 0.1 * rango),
                              disabled=not usar_lie, key="cap_lie", format="%.4f")
    with cc2:
        usar_lse = st.checkbox("Definir LSE", value=bool(vc and vc.lse is not None), key="cap_ulse")
        lse = st.number_input("LSE", value=float(vc.lse) if vc and vc.lse is not None
                              else float(df[var_c].max() + 0.1 * rango),
                              disabled=not usar_lse, key="cap_lse", format="%.4f")
    with cc3:
        usar_obj = st.checkbox("Definir objetivo", value=bool(vc and vc.objetivo is not None),
                               key="cap_uobj")
        obj = st.number_input("Objetivo", value=float(vc.objetivo) if vc and vc.objetivo is not None
                              else float(df[var_c].mean()), disabled=not usar_obj,
                              key="cap_obj", format="%.4f")
    with cc4:
        n_sub = st.slider("Tamaño de subgrupo", 2, 10, 5, key="cap_n")
        cats = categoricas(df)
        estrato = st.selectbox("Subgrupos homogéneos dentro de", ["Observaciones consecutivas"] + cats,
                               index=(1 + cats.index(caso.factores[-1]))
                               if caso.factores and caso.factores[-1] in cats else 0,
                               key="cap_estrato",
                               help="Un subgrupo RACIONAL agrupa observaciones tomadas en "
                                    "condiciones homogéneas. Cambiar este criterio cambia "
                                    "sigma dentro de subgrupo, y con ella Cp y Cpk.")
        estrato = None if estrato == "Observaciones consecutivas" else estrato

    if not usar_lie and not usar_lse:
        st.error("Debe definir al menos un límite de especificación para evaluar la capacidad.")
    else:
        try:
            sub, desc_sub = subgrupos_racionales(df, var_c, estrato, n_sub)
            res = capability(df[var_c], lie=lie if usar_lie else None,
                             lse=lse if usar_lse else None,
                             objetivo=obj if usar_obj else None, subgrupos=sub)
            st.caption(f"Criterio de subagrupación: {desc_sub} ({sub.shape[0]} subgrupos).")

            # Prerrequisito de estabilidad
            carta, _ = i_mr(df[var_c])
            estable = carta.en_control
            if estable:
                st.success("**Prerrequisito de estabilidad verificado**: la carta de "
                           "individuales no muestra señales de causa especial. Los índices de "
                           "capacidad son interpretables como predicción.")
            else:
                st.error(
                    f"**Prerrequisito de estabilidad NO cumplido**: la carta de individuales "
                    f"detecta {len(carta.fuera_de_control)} señal(es) de causa especial. "
                    "Los índices que siguen se calculan igualmente, pero **no predicen el "
                    "comportamiento futuro**: primero hay que estabilizar el proceso. "
                    "Revise el módulo Controlar.")
            if norm_rechaza := normality(df[var_c])["shapiro"]["rechaza_normalidad"]:
                st.warning("**Supuesto de normalidad rechazado** (Shapiro-Wilk). La conversión de "
                           "sigmas a PPM supone normalidad: el PPM esperado será aproximado. "
                           "Compare siempre contra el PPM observado, que es un conteo directo.")

            tiles = []
            for nombre, val, ayuda in (("Cp", res.cp, "potencial, corto plazo"),
                                       ("Cpk", res.cpk, "real, corto plazo"),
                                       ("Pp", res.pp, "potencial, largo plazo"),
                                       ("Ppk", res.ppk, "desempeño, largo plazo")):
                if val is None:
                    tiles.append(tile(nombre, "n/a", sub="especificación unilateral"))
                else:
                    est = "bueno" if val >= 1.33 else "alerta" if val >= 1.0 else "critico"
                    etq = {"bueno": "CAPAZ", "alerta": "MARGINAL", "critico": "NO CAPAZ"}[est]
                    tiles.append(tile(nombre, num(val), sub=ayuda, estado=est,
                                      etiqueta_estado=etq))
            st.markdown(fila_tiles(tiles), unsafe_allow_html=True)

            st.markdown(fila_tiles([
                tile("Sigma corto plazo", num(res.sigma_within, 4), uc, sub=res.metodo_sigma_within),
                tile("Sigma total", num(res.sigma_overall, 4), uc, sub="desviación estándar global"),
                tile("PPM observado", num(res.ppm_observado, 0), sub="conteo directo, sin supuestos"),
                tile("PPM esperado", num(res.ppm_esperado_within, 0), sub="bajo normalidad"),
                tile("Nivel sigma (Z)", num(res.z_bench_within) if res.z_bench_within else "-",
                     sub="benchmark de corto plazo"),
            ]), unsafe_allow_html=True)

            st.plotly_chart(g.capacidad(df[var_c], res, f"Capacidad de {var_c}", uc),
                            width="stretch")

            if res.cpk is not None:
                lo, hi = intervalo_confianza_cpk(res.cpk, res.n)
                st.caption(f"Intervalo de confianza al 95% para Cpk (aproximación de Bissell, "
                           f"válida bajo normalidad): **[{num(lo)}; {num(hi)}]**. Con "
                           f"n = {res.n}, la incertidumbre del índice es de "
                           f"+/- {num((hi-lo)/2)}.")

            advertencias(res.advertencias)
            tabla(resultado_a_frame(res), f"{caso.id}_{var_c}_capacidad", "Resultados exportables")
            interpretar("capacidad", lambda: I.capacidad(res, var_c, uc))

            st.divider()
            st.markdown("##### Efecto del criterio de subagrupación sobre Cp y Pp")
            st.caption("Cp usa sigma DENTRO de subgrupo y Pp usa sigma TOTAL. Cambiar el criterio "
                       "de subagrupación cambia el primero pero no el segundo. Si Cp es mucho "
                       "mayor que Pp, la variación entre subgrupos domina y el proceso no es estable.")
            filas = []
            for e in [None] + categoricas(df):
                try:
                    s_, d_ = subgrupos_racionales(df, var_c, e, n_sub)
                    r_ = capability(df[var_c], lie=lie if usar_lie else None,
                                    lse=lse if usar_lse else None, subgrupos=s_)
                    filas.append({"Criterio": e or "Consecutivos", "k subgrupos": s_.shape[0],
                                  "sigma dentro": round(r_.sigma_within, 4),
                                  "sigma total": round(r_.sigma_overall, 4),
                                  "Cp": None if r_.cp is None else round(r_.cp, 4),
                                  "Cpk": round(r_.cpk, 4),
                                  "Pp": None if r_.pp is None else round(r_.pp, 4),
                                  "Ppk": round(r_.ppk, 4),
                                  "Cp/Pp": None if (r_.cp is None or not r_.pp)
                                  else round(r_.cp / r_.pp, 3)})
                except (ValueError, KeyError):
                    continue
            if filas:
                tabla(pd.DataFrame(filas), f"{caso.id}_{var_c}_subagrupacion", descargar=False)
        except ValueError as e:
            st.error(f"No se pudo calcular la capacidad: {e}")

# ====================================================== C. GAGE R&R
with t_msa:
    ficha("gage")
    st.markdown(nota(
        "<b>Por que este estudio usa su propio conjunto de datos.</b> El Gage R&R exige un "
        "diseño CRUZADO Y BALANCEADO: cada operador mide cada pieza varias veces. Ese diseño no "
        "existe en los datos operacionales corrientes, así que no se puede calcular sobre ellos. "
        "Aquí se simula un estudio con ese diseño; usted puede modificar sus parámetros y "
        "observar como cambia el veredicto."), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_p = st.slider("Piezas", 5, 20, 10, key="g_p")
        n_o = st.slider("Operadores", 2, 4, 3, key="g_o")
    with c2:
        n_r = st.slider("Replicas", 2, 5, 3, key="g_r")
        s_pieza = st.slider("Sigma entre piezas (proceso)", 0.5, 12.0, 4.0, 0.5, key="g_sp")
    with c3:
        s_rep = st.slider("Sigma de repetibilidad (equipo)", 0.1, 6.0, 0.9, 0.1, key="g_sr")
        sesgo = st.slider("Diferencia máxima entre operadores", 0.0, 5.0, 0.7, 0.1, key="g_se")
    with c4:
        usar_tol = st.checkbox("Comparar contra tolerancia", value=False, key="g_ut")
        tol = st.number_input("Amplitud de tolerancia", value=24.0, disabled=not usar_tol,
                              key="g_tol")

    ses = tuple(np.linspace(-sesgo / 2, sesgo / 2, n_o))
    dg = datos_gage_rr(piezas=n_p, operadores=n_o, replicas=n_r, sigma_pieza=s_pieza,
                       sigma_repetibilidad=s_rep, sesgos_operador=ses,
                       etiqueta=caso.gage_etiqueta)
    try:
        rg = gage_rr_anova(dg, "pieza", "operador", caso.gage_etiqueta,
                           tolerancia=tol if usar_tol else None)
        est = ("bueno" if rg["pct_rr_estudio"] < 10 else
               "alerta" if rg["pct_rr_estudio"] <= 30 else "critico")
        etq = {"bueno": "ACEPTABLE", "alerta": "CONDICIONAL", "critico": "INACEPTABLE"}[est]
        st.markdown(fila_tiles([
            tile("%R&R (variación de estudio)", num(rg["pct_rr_estudio"], 1), "%",
                 sub="criterio AIAG: <10% aceptable", estado=est, etiqueta_estado=etq),
            tile("%Contribución", num(rg["pct_rr_contribucion"], 1), "%",
                 sub="sobre varianzas, no comparable con el anterior"),
            tile("ndc", str(rg["ndc"]), sub="categorías distinguibles; se requiere >= 5",
                 estado="bueno" if rg["ndc"] and rg["ndc"] >= 5 else "critico",
                 etiqueta_estado="SUFICIENTE" if rg["ndc"] and rg["ndc"] >= 5 else "INSUFICIENTE"),
            tile("Mediciones", str(rg["N"]), sub=f"{n_p} x {n_o} x {n_r}"),
        ]), unsafe_allow_html=True)

        cga, cgb = st.columns([1.1, 1])
        with cga:
            st.plotly_chart(g.componentes_variacion(rg["componentes"]), width="stretch")
        with cgb:
            st.markdown("**Tabla ANOVA del estudio**")
            st.dataframe(rg["tabla_anova"].round(5), width="stretch", hide_index=True)
            st.markdown("**Componentes de variación**")
            st.dataframe(rg["componentes"].round(4), width="stretch", hide_index=True)

        st.download_button("Descargar componentes en CSV", a_csv(rg["componentes"]),
                           file_name=f"{caso.id}_gagerr.csv", mime="text/csv")
        advertencias(rg["notas"], "Notas metodológicas")
        interpretar("gage", lambda: I.gage(rg))

        with st.expander("Ver los datos del estudio"):
            tabla(dg, f"{caso.id}_gage_datos", altura=320)
    except ValueError as e:
        st.error(f"No se pudo ejecutar el Gage R&R: {e}")

# ====================================================== D. MUESTREO
with t_mue:
    ficha("muestreo")
    var_m = selector_variable(df, caso, "mue_var", "Variable de referencia")
    um = unidad_de(caso, var_m)
    s_obs = float(df[var_m].std(ddof=1))
    media_obs = float(df[var_m].mean())

    c1, c2, c3 = st.columns(3)
    with c1:
        sigma_sup = st.number_input(f"Sigma supuesta ({um})", value=round(s_obs, 4),
                                    min_value=0.0001, format="%.4f",
                                    help="Por defecto, la desviación estándar observada en los "
                                         "datos actuales. En la práctica proviene de un piloto "
                                         "o de datos historicos.")
    with c2:
        margen = st.number_input(f"Margen de error deseado ({um})",
                                 value=round(max(s_obs / 8, 0.01), 4), min_value=0.0001,
                                 format="%.4f")
    with c3:
        conf = st.select_slider("Confianza", [0.80, 0.90, 0.95, 0.99], value=0.95)

    from scipy import stats as _st
    z = float(_st.norm.isf((1 - conf) / 2))
    n_req = int(np.ceil((z * sigma_sup / margen) ** 2))
    n_actual = int(df[var_m].notna().sum())
    margen_actual = z * sigma_sup / np.sqrt(n_actual)

    st.markdown(fila_tiles([
        tile("Tamaño requerido", f"{n_req:,}".replace(",", " "),
             sub=f"para margen +/- {num(margen)} {um} al {int(conf*100)}%"),
        tile("Observaciones disponibles", f"{n_actual:,}".replace(",", " "),
             estado="bueno" if n_actual >= n_req else "alerta",
             etiqueta_estado="SUFICIENTE" if n_actual >= n_req else "INSUFICIENTE"),
        tile("Margen real alcanzable", num(margen_actual), um,
             sub=f"con las {n_actual} observaciones actuales"),
        tile("IC de la media", f"[{num(media_obs - margen_actual)}; "
                               f"{num(media_obs + margen_actual)}]", um,
             sub=f"al {int(conf*100)}% de confianza"),
    ]), unsafe_allow_html=True)

    if n_actual < n_req:
        st.warning(
            f"Con {n_actual} observaciones el margen real es +/- {num(margen_actual)} {um}, mayor "
            f"que el deseado de +/- {num(margen)} {um}. **Conclusiones más finas que "
            f"{num(margen_actual)} {um} no están respaldadas por estos datos.** Para alcanzar "
            f"el margen deseado se requieren {n_req} observaciones.")
    else:
        st.success(f"Las {n_actual} observaciones disponibles superan las {n_req} requeridas: "
                   f"el margen alcanzado es +/- {num(margen_actual)} {um}.")

    st.markdown("##### Curva de tamaño de muestra")
    ns = np.unique(np.round(np.logspace(np.log10(10), np.log10(max(n_req * 2, 500)), 70)).astype(int))
    curva = pd.DataFrame({"n": ns, "margen_de_error": z * sigma_sup / np.sqrt(ns)})
    import plotly.graph_objects as go
    from components.tema import ESTADO, SERIE, SUPERFICIE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curva["n"], y=curva["margen_de_error"], mode="lines",
                             line=dict(color=SERIE[0], width=2), name="Margen de error",
                             hovertemplate="n=%{x}<br>margen +/- %{y:.4f}<extra></extra>"))
    fig.add_hline(y=margen, line=dict(color=ESTADO["bueno"], width=2, dash="dash"),
                  annotation_text="Margen deseado", annotation_font=dict(size=11))
    fig.add_vline(x=n_actual, line=dict(color=ESTADO["serio"], width=2, dash="dot"),
                  annotation_text="n disponible", annotation_font=dict(size=11))
    fig.update_layout(title="Margen de error en función del tamaño de muestra", height=380,
                      xaxis_title="Tamaño de muestra (n)", yaxis_title=f"Margen de error ({um})",
                      xaxis_type="log")
    st.plotly_chart(fig, width="stretch")
    tabla(curva.round(5), f"{caso.id}_curva_muestreo", descargar=False)

    st.markdown(nota(
        "<b>El tamaño no corrige el sesgo.</b> Aumentar n reduce el error ALEATORIO. Si el "
        "muestreo es sistemáticamente sesgado (solo el turno de día, solo un equipo, solo los "
        "ciclos que el sistema registro correctamente), ningun tamaño de muestra corrige esa "
        "distorsión: solo la hace más precisa."), unsafe_allow_html=True)
