"""Etapa ANALIZAR: correlación, regresión, pruebas de hipótesis y análisis causal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


from components import graficos as g
from components import interpretacion as I
from components.tema import CSS, fila_tiles, nota, num, registrar_tema, tile
from components.ui import (advertencias, cabecera, ficha, interpretar, selector_variable,
                           tabla, unidad_de)
from statistics_engine.inference import levene, normality, one_way_anova, paired_t, two_sample_t
from statistics_engine.pareto import matriz_causa_efecto, pareto as calcular_pareto
from statistics_engine.regression import correlation_matrix, ols, scatter_fit
from utils.estado import barra_lateral, categoricas, datos_actuales, numericas
from utils.io import a_csv

registrar_tema()
st.markdown(CSS, unsafe_allow_html=True)
caso = barra_lateral()
df = datos_actuales()

cabecera("Identificar las causas de la variabilidad", "ETAPA 3 — ANALIZAR",
         "El objetivo es pasar de 'el proceso varía' a 'el proceso varía por estas razones'. "
         "Las herramientas cuantitativas miden asociación; las cualitativas proponen mecanismos. "
         "Ninguna de las dos por si sola establece causalidad: eso requiere un experimento, "
         "que es el contenido de la etapa Mejorar.")

t_cor, t_reg, t_hip, t_cau = st.tabs(
    ["A · Correlación", "B · Regresión", "C · Pruebas de hipótesis", "D · Análisis causal"])

num_cols = numericas(df)
cat_cols = categoricas(df)

# ====================================================== A. CORRELACION
with t_cor:
    ficha("correlacion")
    c1, c2 = st.columns([1, 2])
    with c1:
        metodo = st.radio("Método", ["pearson", "spearman"], key="cor_met",
                          format_func=lambda m: "Pearson (asociación lineal)" if m == "pearson"
                          else "Spearman (asociación monotona, sobre rangos)")
        resp = selector_variable(df, caso, "cor_resp", "Variable respuesta")
    with c2:
        defecto = list(dict.fromkeys(
            v for v in ([caso.respuesta] + caso.predictores + caso.componentes)
            if v in num_cols))
        sel = st.multiselect("Variables a incluir en la matriz", num_cols,
                             default=defecto or num_cols[:6], key="cor_vars")
        sel = list(dict.fromkeys(sel))

    if len(sel) < 2:
        st.warning("Seleccione al menos dos variables.")
    else:
        res = correlation_matrix(df, metodo, sel)
        cm1, cm2 = st.columns([1.3, 1])
        with cm1:
            st.plotly_chart(g.matriz_correlacion(res["r"], metodo), width="stretch")
        with cm2:
            if resp in sel:
                s = res["r"][resp].drop(labels=[resp]).sort_values(key=np.abs, ascending=False)
                rank = pd.DataFrame({
                    "variable": s.index, "r": s.values, "R2_bivariado": s.values ** 2,
                    "p_valor": [res["p"].loc[v, resp] for v in s.index]})
                tabla(rank.round(5), f"{caso.id}_correlaciones_{resp}",
                      f"Correlaciones con {resp}", altura=330)
            st.markdown(nota(
                "<b>Pearson frente a Spearman.</b> Si Spearman es claramente mayor que Pearson, "
                "la relación es monotona pero no lineal, y una regresión lineal la subestimará. "
                "Compare ambos métodos cambiando el selector."), unsafe_allow_html=True)

        if resp in sel:
            st.divider()
            otras = [c for c in sel if c != resp]
            xv = st.selectbox("Examinar la dispersión frente a", otras, key="cor_scatter")
            fit = scatter_fit(df[xv], df[resp])
            cs1, cs2 = st.columns([1.4, 1])
            with cs1:
                st.plotly_chart(g.dispersion(df[xv], df[resp], fit, xv, resp),
                                width="stretch")
            with cs2:
                st.markdown(fila_tiles([
                    tile("r", num(fit["r"], 4)),
                    tile("R2", num(fit["r2"], 4), sub="varianza explicada por la recta"),
                    tile("Pendiente", num(fit["pendiente"], 4),
                         sub=f"+/- {num(fit['error_std_pendiente'], 4)}"),
                    tile("p-valor", num(fit["p_valor"], 5)),
                ]), unsafe_allow_html=True)
                st.markdown(f"Ecuación ajustada: `{fit['ecuacion']}`")
                st.caption("La banda interna es el IC de la media condicional; la externa es el "
                           "IC de predicción de una observación individual, siempre más ancha.")
            interpretar("correlacion", lambda: I.correlacion(res, resp))

        st.warning(
            "**Correlación no es causalidad.** Una asociación fuerte es compatible con causalidad "
            "directa, causalidad inversa, una variable de confusión que afecte a ambas, o "
            "coincidencia muestral. En estos datos, por ejemplo, distancia y pendiente dependen "
            "ambas de la fase del rajo: parte de la asociación de cada una con el tiempo de ciclo "
            "es atribuible a la fase y no a ellas mismas.")
        tabla(res["r"].round(5).reset_index().rename(columns={"index": "variable"}),
              f"{caso.id}_matriz_correlacion_{metodo}", "Matriz de correlación exportable")

# ====================================================== B. REGRESION
with t_reg:
    ficha("regresion")
    c1, c2 = st.columns([1, 2])
    with c1:
        y = selector_variable(df, caso, "reg_y", "Variable respuesta (y)")
    with c2:
        cand = [c for c in num_cols if c != y]
        pred_def = [p for p in caso.predictores if p in cand]
        xs = st.multiselect("Predictores (x)", cand, default=pred_def or cand[:3], key="reg_x")

    if not xs:
        st.warning("Seleccione al menos un predictor.")
    else:
        try:
            r = ols(df, y, xs)
            uy = unidad_de(caso, y)
            st.markdown(fila_tiles([
                tile("R2", num(r["r2"], 4), sub="varianza explicada"),
                tile("R2 ajustado", num(r["r2_ajustado"], 4), sub=f"penaliza {r['k']} predictores"),
                tile("Error residual", num(r["s_residual"], 4), uy,
                     sub="dispersión típica alrededor de la predicción"),
                tile("F global", num(r["f"], 1), sub=f"p = {num(r['f_p_valor'], 5)}",
                     estado="bueno" if r["f_p_valor"] < 0.05 else "alerta",
                     etiqueta_estado="SIGNIFICATIVO" if r["f_p_valor"] < 0.05 else "NO SIGNIF."),
                tile("n", str(r["n"]), sub=f"{num(r['n']/max(r['k'],1),1)} obs. por predictor"),
            ]), unsafe_allow_html=True)

            st.markdown("##### Modelo ajustado")
            st.code(r["ecuacion"], language="text")
            tabla(r["coeficientes"].round(6), f"{caso.id}_regresion_coeficientes",
                  "Coeficientes estimados", descargar=True)

            st.markdown("##### Diagnóstico de residuos")
            st.plotly_chart(g.residuos(r), width="stretch")
            dg = r["diagnostico"]
            tiles_d = [
                tile("Durbin-Watson", num(dg["durbin_watson"], 3),
                     sub="2 = sin autocorrelación",
                     estado="bueno" if 1.5 <= dg["durbin_watson"] <= 2.5 else "alerta",
                     etiqueta_estado="OK" if 1.5 <= dg["durbin_watson"] <= 2.5 else "REVISAR"),
                tile("Shapiro de residuos", num(dg["shapiro_residuos"]["p_valor"], 5),
                     sub="normalidad de los errores"),
            ]
            if "p_valor" in dg.get("breusch_pagan", {}):
                bp = dg["breusch_pagan"]["p_valor"]
                tiles_d.append(tile("Breusch-Pagan", num(bp, 5), sub="varianza constante",
                                    estado="bueno" if bp >= 0.05 else "alerta",
                                    etiqueta_estado="HOMOCEDASTICO" if bp >= 0.05
                                    else "HETEROCEDASTICO"))
            if "vif" in dg:
                vmax = float(dg["vif"]["VIF"].max())
                tiles_d.append(tile("VIF máximo", num(vmax, 2), sub="multicolinealidad",
                                    estado="bueno" if vmax < 5 else "alerta" if vmax < 10
                                    else "critico",
                                    etiqueta_estado="BAJA" if vmax < 5 else "MODERADA"
                                    if vmax < 10 else "SEVERA"))
            st.markdown(fila_tiles(tiles_d), unsafe_allow_html=True)
            if "vif" in dg:
                tabla(dg["vif"].round(3), f"{caso.id}_vif", "Factores de inflación de varianza",
                      descargar=False)

            advertencias(r["advertencias"], "Diagnóstico y advertencias")
            interpretar("regresion", lambda: I.regresion(r, uy))

            st.divider()
            st.markdown("##### Comparación: regresión simple frente a múltiple")
            st.caption("Omitir un predictor correlacionado con los demas sesga los coeficientes "
                       "restantes (sesgo por variable omitida). Si el coeficiente cambia mucho al "
                       "pasar de simple a múltiple, las variables estaban confundidas.")
            filas = []
            for x in xs:
                simple = ols(df, y, [x])["coeficientes"].set_index("termino").loc[x, "coeficiente"]
                multi = r["coeficientes"].set_index("termino").loc[x, "coeficiente"]
                filas.append({"predictor": x, "coef. en regresión simple": round(simple, 5),
                              "coef. en regresión múltiple": round(multi, 5),
                              "cambio": round(multi - simple, 5),
                              "cambio_%": round(100 * (multi - simple) / simple, 1)
                              if simple else np.nan})
            comp = pd.DataFrame(filas)
            tabla(comp, f"{caso.id}_sesgo_variable_omitida", descargar=False)
            grandes = comp.loc[comp["cambio_%"].abs() > 25, "predictor"].tolist()
            if grandes:
                st.warning(f"Los coeficientes de **{', '.join(grandes)}** cambian más de 25% al "
                           "pasar de la regresión simple a la múltiple: hay confusión entre "
                           "predictores y los resultados bivariados son enganosos.")

            st.divider()
            st.markdown("##### Predicción con el modelo")
            cols_p = st.columns(min(len(xs), 4))
            valores = {}
            for i, x in enumerate(xs):
                with cols_p[i % len(cols_p)]:
                    valores[x] = st.number_input(
                        f"{x}", value=float(df[x].mean()),
                        min_value=float(df[x].min()), max_value=float(df[x].max()),
                        key=f"pred_{x}", format="%.4f")
            import statsmodels.api as sm
            Xn = pd.DataFrame([valores])
            Xn = sm.add_constant(Xn, has_constant="add")[r["modelo"].params.index]
            pr = r["modelo"].get_prediction(Xn).summary_frame(alpha=0.05)
            st.markdown(fila_tiles([
                tile(f"{y} previsto", num(pr['mean'].iloc[0], 3), uy),
                tile("IC 95% de la media", f"[{num(pr['mean_ci_lower'].iloc[0],3)}; "
                                           f"{num(pr['mean_ci_upper'].iloc[0],3)}]", uy),
                tile("IC 95% de predicción", f"[{num(pr['obs_ci_lower'].iloc[0],3)}; "
                                             f"{num(pr['obs_ci_upper'].iloc[0],3)}]", uy,
                     sub="para una observación individual"),
            ]), unsafe_allow_html=True)
            st.caption("Solo se permiten valores dentro del rango observado de cada predictor: "
                       "extrapolar fuera de ese rango no está respaldado por los datos.")
        except ValueError as e:
            st.error(f"No se pudo ajustar el modelo: {e}")

# ====================================================== C. HIPOTESIS
with t_hip:
    ficha("hipotesis")
    prueba_sel = st.radio("Prueba a ejecutar",
                          ["Prueba t para dos muestras independientes",
                           "Prueba t pareada", "ANOVA de un factor"],
                          horizontal=True, key="hip_tipo")
    alfa = st.select_slider("Nivel de significancia (alfa)", [0.01, 0.05, 0.10], value=0.05)

    # ---- t dos muestras
    if prueba_sel.startswith("Prueba t para dos"):
        c1, c2, c3 = st.columns(3)
        with c1:
            yv = selector_variable(df, caso, "t2_y", "Variable a comparar")
        with c2:
            if not cat_cols:
                st.error("No hay variables categoricas disponibles para formar los grupos.")
                st.stop()
            fac = st.selectbox("Factor de agrupación", cat_cols, key="t2_f")
        niveles = sorted(df[fac].astype(str).unique())
        with c3:
            if len(niveles) < 2:
                st.error("El factor debe tener al menos dos niveles.")
                st.stop()
            na = st.selectbox("Grupo 1", niveles, index=0, key="t2_a")
            nb = st.selectbox("Grupo 2", niveles, index=1, key="t2_b")
        if na == nb:
            st.warning("Seleccione dos grupos distintos.")
        else:
            a = df.loc[df[fac].astype(str) == na, yv]
            b = df.loc[df[fac].astype(str) == nb, yv]
            uy = unidad_de(caso, yv)
            st.markdown("##### Verificación previa de supuestos")
            s1, s2, s3 = st.columns(3)
            n1 = normality(a); n2 = normality(b); lv = levene(a, b, alfa=alfa)
            with s1:
                st.metric(f"Normalidad {na} (Shapiro p)", num(n1["shapiro"]["p_valor"], 5))
            with s2:
                st.metric(f"Normalidad {nb} (Shapiro p)", num(n2["shapiro"]["p_valor"], 5))
            with s3:
                st.metric("Levene (igualdad de varianzas, p)", num(lv.p_valor, 5))
            if n1["shapiro"]["rechaza_normalidad"] or n2["shapiro"]["rechaza_normalidad"]:
                st.info(f"La normalidad se rechaza en al menos un grupo. Con n = {len(a)} y "
                        f"{len(b)}, la prueba t sigue siendo razonablemente robusta por el "
                        "teorema central del límite, pero conviene contrastar con una prueba "
                        "no parametrica si las muestras fueran pequeñas.")
            t = two_sample_t(a, b, alfa=alfa)
            st.markdown(fila_tiles([
                tile(f"Media {na}", num(t.detalles["media_1"], 3), uy, sub=f"n = {t.detalles['n_1']}"),
                tile(f"Media {nb}", num(t.detalles["media_2"], 3), uy, sub=f"n = {t.detalles['n_2']}"),
                tile("Diferencia", num(t.efecto_valor, 3), uy,
                     sub=f"IC [{num(t.ic_inferior,3)}; {num(t.ic_superior,3)}]"),
                tile("p-valor", num(t.p_valor, 5),
                     estado="critico" if t.rechaza_h0 else "bueno",
                     etiqueta_estado="SE RECHAZA H0" if t.rechaza_h0 else "NO SE RECHAZA H0"),
                tile("d de Cohen", num(t.detalles["d_cohen"], 3), sub="tamaño del efecto"),
            ]), unsafe_allow_html=True)
            st.plotly_chart(g.boxplot(df[df[fac].astype(str).isin([na, nb])], yv, fac,
                                      f"{yv} por {fac}", uy), width="stretch")
            interpretar("t2", lambda: I.prueba(t, na, nb, uy))
            tabla(pd.DataFrame([t.to_dict()]).T.reset_index()
                  .rename(columns={"index": "campo", 0: "valor"}),
                  f"{caso.id}_prueba_t", descargar=True)

    # ---- t pareada
    elif prueba_sel == "Prueba t pareada":
        st.markdown(nota(
            "<b>Cuando usar la prueba pareada.</b> Solo si cada observación de 'antes' y "
            "'después' corresponde a la MISMA unidad experimental: el mismo camion, la misma "
            "ruta, el mismo pozo. Si las muestras son independientes, esta prueba es invalida y "
            "exagera la significancia. Aquí se compara el escenario base con el escenario "
            "mejorado emparejando por índice de observación, lo que es legitimo porque ambos "
            "conjuntos se generan con la misma semilla y comparten las mismas condiciones."),
            unsafe_allow_html=True)
        yv = selector_variable(df, caso, "tp_y", "Variable a comparar")
        from utils.estado import datos_actuales as _da
        base = _da(mejora=False)[yv]
        mej = _da(mejora=True)[yv]
        n = min(len(base), len(mej))
        t = paired_t(base.iloc[:n], mej.iloc[:n], alfa=alfa)
        uy = unidad_de(caso, yv)
        st.markdown(fila_tiles([
            tile("Media antes", num(t.detalles["media_antes"], 3), uy),
            tile("Media después", num(t.detalles["media_despues"], 3), uy),
            tile("Diferencia media", num(t.efecto_valor, 3), uy,
                 sub=f"IC [{num(t.ic_inferior,3)}; {num(t.ic_superior,3)}]"),
            tile("p-valor", num(t.p_valor, 5),
                 estado="critico" if t.rechaza_h0 else "bueno",
                 etiqueta_estado="SE RECHAZA H0" if t.rechaza_h0 else "NO SE RECHAZA H0"),
            tile("Pares", str(t.detalles["n_pares"])),
        ]), unsafe_allow_html=True)
        st.plotly_chart(g.comparacion_escenarios(base.iloc[:n], mej.iloc[:n], "Antes", "Despues",
                                                 f"Distribución de {yv}: antes y después", uy),
                        width="stretch")
        interpretar("tp", lambda: I.prueba(t, "antes", "despues", uy))

    # ---- ANOVA
    else:
        c1, c2 = st.columns(2)
        with c1:
            yv = selector_variable(df, caso, "an_y", "Variable respuesta")
        with c2:
            if not cat_cols:
                st.error("No hay variables categoricas disponibles.")
                st.stop()
            fac = st.selectbox("Factor", cat_cols, key="an_f")
        uy = unidad_de(caso, yv)
        try:
            r = one_way_anova(df, yv, fac, alfa=alfa)
            st.markdown(fila_tiles([
                tile("F", num(r["F"], 3), sub=f"gl {r['gl_entre']}, {r['gl_dentro']}"),
                tile("p-valor", num(r["p_valor"], 5),
                     estado="critico" if r["rechaza_h0"] else "bueno",
                     etiqueta_estado="SE RECHAZA H0" if r["rechaza_h0"] else "NO SE RECHAZA H0"),
                tile("eta cuadrado", num(r["eta_cuadrado"], 4),
                     sub=f"{num(100*r['eta_cuadrado'],1)}% de la variabilidad"),
                tile("omega cuadrado", num(r["omega_cuadrado"], 4), sub="versión menos sesgada"),
                tile("Levene p", num(r["levene"]["p_valor"], 5),
                     estado="bueno" if not r["levene"]["rechaza_h0"] else "alerta",
                     etiqueta_estado="VARIANZAS OK" if not r["levene"]["rechaza_h0"]
                     else "VARIANZAS DESIGUALES"),
            ]), unsafe_allow_html=True)
            ca1, ca2 = st.columns([1.2, 1])
            with ca1:
                st.plotly_chart(g.boxplot(df, yv, fac, f"{yv} por {fac}", uy),
                                width="stretch")
            with ca2:
                tabla(r["tabla_anova"].round(5), f"{caso.id}_anova", "Tabla ANOVA", descargar=False)
                tabla(r["resumen_grupos"].round(4), f"{caso.id}_anova_grupos",
                      "Resumen por grupo", descargar=False)
            st.markdown("**Comparaciones por pares (Tukey HSD)**")
            st.caption("Tukey controla el error tipo I del conjunto de comparaciones; hacer "
                       "múltiples pruebas t por separado lo inflaría.")
            st.dataframe(r["tukey"], width="stretch", hide_index=True)
            advertencias(r["supuestos"], "Supuestos del ANOVA")
            st.info(r["interpretacion_p"])
            interpretar("anova", lambda: I.anova(r, yv, fac, uy))
            st.download_button("Descargar tabla ANOVA en CSV", a_csv(r["tabla_anova"]),
                               file_name=f"{caso.id}_anova.csv", mime="text/csv")
        except ValueError as e:
            st.error(f"No se pudo ejecutar el ANOVA: {e}")

# ====================================================== D. CAUSAL
with t_cau:
    ficha("ishikawa")
    sub1, sub2, sub3 = st.tabs(["Ishikawa (6M)", "Cinco porqués", "Matriz causa-efecto"])

    with sub1:
        st.markdown("#### Diagrama de Ishikawa")
        st.caption(f"Efecto analizado: **{caso.problema}**")
        cols = st.columns(3)
        for i, (categoria, causas) in enumerate(caso.ishikawa.items()):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{categoria}**")
                    for c in causas:
                        st.markdown(f"<div style='font-size:13px;padding:2px 0'>· {c}</div>",
                                    unsafe_allow_html=True)
        isk = pd.DataFrame([{"categoria": k, "causa": c}
                            for k, v in caso.ishikawa.items() for c in v])
        tabla(isk, f"{caso.id}_ishikawa", descargar=True)
        st.warning("El Ishikawa produce **hipótesis**, no hallazgos. Cada causa listada debe "
                   "verificarse con datos: las pestanas de correlación, regresión y pruebas de "
                   "hipótesis existen para eso.")

    with sub2:
        st.markdown("#### Cinco porqués")
        cp = caso.cinco_porques
        st.markdown(f"**Problema de partida:** {cp['problema']}")
        for i, p in enumerate(cp["porques"], 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {p['pregunta']}**")
                st.markdown(p["respuesta"])
        st.success(f"**Causa raiz propuesta:** {cp['causa_raiz']}")
        st.warning(cp["advertencia"])
        tabla(pd.DataFrame(cp["porques"]), f"{caso.id}_cinco_porques", descargar=True)

    with sub3:
        st.markdown("#### Matriz causa-efecto")
        st.caption("Pondera cada causa potencial contra las salidas críticas. Los pesos los fija "
                   "el cliente (importancia de cada salida) y las calificaciones el equipo "
                   "(0 sin relación, 1 débil, 3 moderada, 9 fuerte).")
        salidas = [c["ctq"] for c in caso.ctq]
        causas = sorted({c for v in caso.ishikawa.values() for c in v})[:12]
        st.markdown("**Pesos de las salidas (importancia para el cliente, 1 a 10)**")
        cols_w = st.columns(len(salidas))
        pesos = []
        for i, (col, s) in enumerate(zip(cols_w, salidas)):
            with col:
                pesos.append(st.slider(s, 1, 10, 10 - i if 10 - i > 0 else 1, key=f"ce_w{i}"))
        st.markdown("**Calificación de cada causa frente a cada salida**")
        calif = {}
        for j, c in enumerate(causas):
            cols_c = st.columns([2] + [1] * len(salidas))
            cols_c[0].markdown(f"<div style='font-size:13px;padding-top:8px'>{c}</div>",
                               unsafe_allow_html=True)
            fila = []
            for i, s in enumerate(salidas):
                with cols_c[i + 1]:
                    fila.append(st.selectbox(s, [0, 1, 3, 9], index=1,
                                             key=f"ce_{j}_{i}", label_visibility="collapsed"))
            calif[c] = fila
        rce = matriz_causa_efecto(causas, salidas, pesos, calif)
        st.divider()
        tabla(rce["tabla"].round(2), f"{caso.id}_causa_efecto", "Priorización resultante")
        st.caption(rce["nota"])
        top3 = rce["tabla"].head(3)["causa"].tolist()
        st.info(f"**Causas mejor puntuadas:** {', '.join(top3)}. Estas son las candidatas a "
                "verificar con datos y, si se confirman, a manipular en el diseño de "
                "experimentos de la etapa Mejorar.")
