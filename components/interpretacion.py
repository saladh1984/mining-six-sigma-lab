"""Generación de interpretaciones a partir de resultados YA CALCULADOS.

Regla de diseño: cada función recibe el objeto de resultado del motor
estadístico y devuelve texto construido exclusivamente con esos números. No
hay plantillas con conclusiones predefinidas ni afirmaciones que no se
deriven de un valor presente en el resultado. Si un dato no está disponible,
se dice que no está disponible en vez de rellenarlo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tema import num


def _vt(x, dec=3):
    return num(x, dec)


# ------------------------------------------------------------ descriptiva
def descriptiva(d, etiqueta: str, unidad: str = "") -> list[str]:
    u = f" {unidad}" if unidad else ""
    out = [
        f"Sobre **{d.n} observaciones** de *{etiqueta}*, la media es **{_vt(d.media)}{u}** "
        f"y la mediana **{_vt(d.mediana)}{u}**. La desviación estándar es **{_vt(d.desv_std)}{u}**, "
        f"con un rango de {_vt(d.minimo)} a {_vt(d.maximo)}{u}."
    ]
    if d.coef_variacion is not None:
        cv = 100 * d.coef_variacion
        nivel = ("baja" if cv < 10 else "moderada" if cv < 20 else
                 "alta" if cv < 30 else "muy alta")
        out.append(
            f"El coeficiente de variación es **{_vt(cv, 1)}%**, una dispersión relativa "
            f"**{nivel}**: por cada 100 unidades de valor medio hay {_vt(cv, 1)} unidades "
            "de dispersión típica."
        )
    sep = abs(d.media - d.mediana)
    ref = d.desv_std if d.desv_std > 0 else 1.0
    if sep / ref > 0.15:
        direccion = "por encima" if d.media > d.mediana else "por debajo"
        out.append(
            f"La media esta {direccion} de la mediana en {_vt(sep)}{u} "
            f"({_vt(100 * sep / ref, 1)}% de una desviación estándar), lo que indica "
            "**asimetría**: la media deja de ser un buen resumen del valor típico y conviene "
            "reportar también la mediana."
        )
    else:
        out.append(
            f"Media y mediana difieren solo en {_vt(sep)}{u}, de modo que la distribución es "
            "**aproximadamente simetrica** en su parte central."
        )
    if abs(d.asimetria) > 0.5:
        lado = "derecha (cola larga hacia valores altos)" if d.asimetria > 0 else \
               "izquierda (cola larga hacia valores bajos)"
        out.append(f"El coeficiente de asimetría es **{_vt(d.asimetria)}**: la distribución se "
                   f"extiende hacia la {lado}.")
    if abs(d.curtosis_exceso) > 1:
        tipo = ("colas más pesadas que la normal: hay más valores extremos de los esperados"
                if d.curtosis_exceso > 0 else
                "colas más livianas que la normal: menos valores extremos de los esperados")
        out.append(f"La curtosis en exceso es **{_vt(d.curtosis_exceso)}**, es decir {tipo}.")
    out.append(
        f"El 50% central de los datos se ubica entre {_vt(d.q1)} y {_vt(d.q3)}{u} "
        f"(IQR = {_vt(d.iqr)}{u}). El error estándar de la media es {_vt(d.error_std_media)}{u}: "
        f"con estos datos la media poblacional se estima con esa precisión típica."
    )
    return out


# ------------------------------------------------------------- capacidad
def capacidad(r, etiqueta: str, unidad: str = "") -> list[str]:
    u = f" {unidad}" if unidad else ""
    out = []
    bilateral = r.lie is not None and r.lse is not None
    spec = (f"LIE = {_vt(r.lie)}{u} y LSE = {_vt(r.lse)}{u}" if bilateral else
            (f"LSE = {_vt(r.lse)}{u} (especificación unilateral superior)" if r.lse is not None
             else f"LIE = {_vt(r.lie)}{u} (especificación unilateral inferior)"))
    out.append(
        f"Con {r.n} observaciones de *{etiqueta}*, media {_vt(r.media)}{u} y especificación {spec}. "
        f"Sigma de corto plazo (dentro de subgrupo) = **{_vt(r.sigma_within, 4)}**, estimada por "
        f"{r.metodo_sigma_within}. Sigma total (largo plazo) = **{_vt(r.sigma_overall, 4)}**."
    )
    if r.cp is not None:
        out.append(
            f"**Cp = {_vt(r.cp)}**: la ventana de especificación mide {_vt(r.cp * 6)} sigmas de "
            "ancho. Cp mide solo si el proceso CABE en la ventana, sin considerar si está centrado."
        )
    else:
        out.append("**Cp y Pp no están definidos** porque la especificación es unilateral: no hay "
                   "una ventana de ancho finito que medir. Solo Cpk y Ppk son interpretables aquí.")

    if r.cpk is not None:
        juicio = ("**capaz** según la convención habitual (Cpk >= 1.33)" if r.cpk >= 1.33 else
                  "**marginalmente capaz** (1.00 <= Cpk < 1.33): el margen es estrecho y pequeños "
                  "corrimientos generan material fuera de especificación" if r.cpk >= 1.0 else
                  "**no capaz** (Cpk < 1.00): la dispersión natural del proceso excede la "
                  "ventana de especificación, de modo que se producen resultados fuera de meta "
                  "de forma estructural, no por casualidad")
        out.append(f"**Cpk = {_vt(r.cpk)}** y **Ppk = {_vt(r.ppk)}**. El proceso es {juicio}.")

    if r.cp is not None and r.cpk is not None and r.cp > 0:
        brecha = r.cp - r.cpk
        if brecha > 0.1:
            out.append(
                f"Cp ({_vt(r.cp)}) supera a Cpk ({_vt(r.cpk)}) en {_vt(brecha)}: el proceso esta "
                "**descentrado**. Solo con recentrarlo, sin reducir su dispersión, el Cpk podría "
                f"llegar hasta {_vt(r.cp)}. Centrar suele ser más barato que reducir variabilidad."
            )
        else:
            out.append(f"Cp y Cpk son casi iguales ({_vt(r.cp)} frente a {_vt(r.cpk)}): el proceso "
                       "esta **centrado**, y la única via de mejora es reducir su dispersión.")

    if r.cp is not None and r.pp is not None and r.pp > 0:
        razon = r.cp / r.pp
        if razon > 1.33:
            out.append(
                f"La razón Cp/Pp = **{_vt(razon)}** es alta: la variación ENTRE subgrupos domina "
                "sobre la variación DENTRO de subgrupo. Esto es síntoma de causas especiales "
                "(deriva, diferencias de turno, equipo u operador). **El proceso probablemente no "
                "está en control estadístico y estos índices no son predictivos todavía.**"
            )
        else:
            out.append(f"La razón Cp/Pp = **{_vt(razon)}** es cercana a 1: la variación dentro y "
                       "entre subgrupos es similar, lo que es coherente con un proceso estable.")

    out.append(
        f"Fuera de especificación: **{_vt(r.ppm_observado, 0)} PPM observados** (conteo directo "
        f"sobre los datos, sin supuestos) frente a {_vt(r.ppm_esperado_within, 0)} PPM esperados "
        "bajo normalidad con sigma de corto plazo. Si ambas cifras difieren mucho, el supuesto de "
        "normalidad no se sostiene y el PPM esperado no es confiable."
    )
    if r.z_bench_within is not None:
        out.append(f"El nivel sigma del proceso (Z benchmark de corto plazo) es "
                   f"**{_vt(r.z_bench_within)}**.")
    return out


# --------------------------------------------------------------- Gage R&R
def gage(r) -> list[str]:
    c = r["componentes"].set_index("Fuente")
    out = [
        f"Estudio cruzado de {r['n_piezas']} piezas x {r['n_operadores']} operadores x "
        f"{r['n_replicas']} replicas ({r['N']} mediciones). Modelo ajustado: {r['modelo']}.",
        f"El sistema de medición aporta **{_vt(r['pct_rr_estudio'], 1)}% de la variación de "
        f"estudio** y **{_vt(r['pct_rr_contribucion'], 1)}% de la varianza total**. "
        f"{r['veredicto']}",
        f"Descomposición: repetibilidad (equipo) "
        f"{_vt(c.loc['  Repetibilidad (equipo)', '%VariacionEstudio'], 1)}%, "
        f"reproducibilidad (operadores) "
        f"{_vt(c.loc['  Reproducibilidad', '%VariacionEstudio'], 1)}%.",
    ]
    rep = c.loc["  Repetibilidad (equipo)", "Varianza"]
    rpr = c.loc["  Reproducibilidad", "Varianza"]
    if rep > rpr * 1.5:
        out.append("La **repetibilidad domina**: el problema está en el equipo o el método de "
                   "medición, no en las personas. Actuar sobre el instrumento, su resolución o "
                   "el procedimiento.")
    elif rpr > rep * 1.5:
        out.append("La **reproducibilidad domina**: los operadores miden distinto entre si. "
                   "Actuar sobre capacitación, criterios de medición y estandarización, no sobre "
                   "el instrumento.")
    else:
        out.append("Repetibilidad y reproducibilidad son de magnitud comparable: conviene actuar "
                   "sobre ambos frentes.")
    if r["interaccion_significativa"]:
        out.append(f"La interacción pieza x operador es significativa (p = "
                   f"{_vt(r['p_interaccion'], 4)}): algunos operadores miden distinto SEGÚN la "
                   "pieza, lo que suele indicar criterios de medición no estandarizados.")
    ndc = r["ndc"]
    out.append(
        f"**ndc = {ndc}**: el sistema distingue {ndc} categorías dentro del rango de las piezas. "
        + ("Cumple el mínimo de 5 exigido por AIAG." if ndc and ndc >= 5 else
           "**No alcanza el mínimo de 5** exigido por AIAG: el sistema no resuelve diferencias "
           "útiles entre piezas y no sirve para control estadístico de este proceso.")
    )
    return out


# ------------------------------------------------------------- hipotesis
def prueba(t, nombre_a: str = "grupo 1", nombre_b: str = "grupo 2", unidad: str = "") -> list[str]:
    u = f" {unidad}" if unidad else ""
    out = [
        f"**{t.nombre}.** H0: {t.hipotesis_nula}. Estadístico = {_vt(t.estadistico)}"
        + (f", gl = {_vt(t.gl, 1)}" if t.gl is not None else "")
        + f", p = **{_vt(t.p_valor, 5)}** (alfa = {t.alfa})."
    ]
    if t.rechaza_h0:
        out.append(
            f"Como p < {t.alfa}, **se rechaza H0**: los datos son poco compatibles con la "
            "hipótesis de que no hay diferencia. Esto NO significa que la probabilidad de que H0 "
            f"sea cierta sea {_vt(t.p_valor, 5)}: el p-valor es la probabilidad de observar un "
            "resultado al menos tan extremo COMO ESTE si H0 fuera cierta."
        )
    else:
        out.append(
            f"Como p >= {t.alfa}, **no se rechaza H0**. Esto NO prueba que no exista diferencia: "
            "puede que la haya y que la muestra no tenga potencia suficiente para detectarla. "
            "El intervalo de confianza indica que tamaños de diferencia siguen siendo compatibles "
            "con los datos."
        )
    if t.efecto_valor is not None and t.ic_inferior is not None:
        out.append(
            f"La diferencia estimada es **{_vt(t.efecto_valor)}{u}**, con intervalo de confianza "
            f"al {int((1-t.alfa)*100)}% de **[{_vt(t.ic_inferior)}; {_vt(t.ic_superior)}]{u}**. "
            "Este intervalo, y no el p-valor, es lo que debe usarse para decidir si la diferencia "
            "es operacionalmente relevante."
        )
    d = t.detalles.get("d_cohen")
    if d is not None:
        mag = ("despreciable" if abs(d) < 0.2 else "pequeño" if abs(d) < 0.5 else
               "mediano" if abs(d) < 0.8 else "grande")
        out.append(f"Tamaño del efecto (d de Cohen) = **{_vt(d)}**, un efecto {mag} en la escala "
                   "convencional. A diferencia del p-valor, no depende del tamaño muestral.")
    lev = t.detalles.get("levene_p")
    if lev is not None:
        out.append(
            f"Supuesto de varianzas iguales: Levene p = {_vt(lev, 4)}, por lo que se "
            + ("asumieron varianzas iguales y se uso la versión agrupada."
               if t.detalles.get("varianzas_iguales_asumidas") else
               "rechaza la igualdad de varianzas y se uso la corrección de Welch.")
        )
    return out


def anova(r, variable: str, factor: str, unidad: str = "") -> list[str]:
    u = f" {unidad}" if unidad else ""
    out = [
        f"ANOVA de un factor para *{variable}* según *{factor}* ({len(r['niveles'])} niveles: "
        f"{', '.join(r['niveles'])}). F({r['gl_entre']}, {r['gl_dentro']}) = "
        f"**{_vt(r['F'])}**, p = **{_vt(r['p_valor'], 5)}**."
    ]
    if r["rechaza_h0"]:
        out.append(f"Se rechaza la igualdad de medias: al menos un nivel de *{factor}* difiere. "
                   "ANOVA no indica cual; para eso se usa la comparación de Tukey, que controla "
                   "el error tipo I del conjunto de comparaciones.")
        tk = r["tukey"]
        sig = [f"{row.group1} vs {row.group2} (diferencia {_vt(row.meandiff)}{u}, "
               f"IC [{_vt(row.lower)}; {_vt(row.upper)}])"
               for row in tk.itertuples() if row.reject]
        if sig:
            out.append("Pares con diferencia significativa: " + "; ".join(sig) + ".")
        else:
            out.append("Ninguna comparación por pares resulta significativa tras el ajuste de "
                       "Tukey: la prueba global detecta diferencia pero no se atribuye a un par "
                       "concreto, situación posible cuando el efecto se reparte entre varios niveles.")
    else:
        out.append("No se rechaza la igualdad de medias: los datos no aportan evidencia "
                   f"suficiente de que *{factor}* afecte el nivel medio de *{variable}*.")
    e2 = r["eta_cuadrado"]
    out.append(
        f"**eta cuadrado = {_vt(e2, 4)}**: el factor *{factor}* explica el {_vt(100*e2, 1)}% de la "
        f"variabilidad total de *{variable}*; el {_vt(100*(1-e2), 1)}% restante corresponde a "
        "variación dentro de los niveles. Un p pequeño con eta cuadrado bajo indica un efecto "
        "real pero de poca importancia práctica."
    )
    lev = r["levene"]
    if lev["rechaza_h0"]:
        out.append(f"**Advertencia de supuesto**: la prueba de Levene rechaza la homogeneidad de "
                   f"varianzas (p = {_vt(lev['p_valor'], 4)}). El ANOVA clasico pierde validez; "
                   "conviene usar la versión de Welch o transformar la respuesta.")
    return out


def normalidad(r, etiqueta: str) -> list[str]:
    sh = r["shapiro"]
    out = [
        f"Prueba de Shapiro-Wilk sobre *{etiqueta}* (n = {r['n']}): W = "
        f"{_vt(sh['estadistico'], 4)}, p = **{_vt(sh['p_valor'], 5)}**. "
        + ("Se **rechaza** la normalidad." if sh["rechaza_normalidad"]
           else "**No se rechaza** la normalidad.")
    ]
    ad = r["anderson_darling"]
    out.append(f"Anderson-Darling: A2 = {_vt(ad['estadistico'], 4)} frente al valor crítico al 5% "
               f"de {_vt(ad['valor_critico_5pct'], 4)}, por lo que "
               + ("también rechaza" if ad["rechaza_normalidad"] else "tampoco rechaza")
               + " la normalidad.")
    out.append(f"Asimetría = {_vt(r['asimetria'])} y curtosis en exceso = "
               f"{_vt(r['curtosis_exceso'])} (ambas valen 0 en una normal exacta).")
    if "advertencia" in r:
        out.append(f"**{r['advertencia']}**")
    out.append("La decisión no debe tomarse solo con el p-valor: el gráfico Q-Q muestra DONDE se "
               "aparta la distribución, que es lo que determina si el apartamiento importa para "
               "el análisis que se quiere hacer.")
    if sh["rechaza_normalidad"]:
        out.append("Consecuencia práctica: los índices de capacidad y el PPM esperado suponen "
                   "normalidad y serán aproximados. Las pruebas t y el ANOVA son razonablemente "
                   "robustos con muestras grandes gracias al teorema central del límite.")
    return out


# ------------------------------------------------------------- regresion
def regresion(r, unidad_y: str = "") -> list[str]:
    u = f" {unidad_y}" if unidad_y else ""
    coef = r["coeficientes"].set_index("termino")
    out = [
        f"Modelo ajustado sobre {r['n']} observaciones con {r['k']} predictor(es):\n\n"
        f"`{r['ecuacion']}`",
        f"**R2 = {_vt(r['r2'], 4)}** y **R2 ajustado = {_vt(r['r2_ajustado'], 4)}**: el modelo "
        f"explica el {_vt(100*r['r2'], 1)}% de la variabilidad de la respuesta. El error estándar "
        f"residual es {_vt(r['s_residual'])}{u}, que es la dispersión típica de las observaciones "
        "alrededor de la predicción.",
        f"Prueba F global: F = {_vt(r['f'])}, p = {_vt(r['f_p_valor'], 5)}, "
        + ("por lo que el conjunto de predictores aporta información significativa."
           if r["f_p_valor"] < 0.05 else
           "por lo que el conjunto de predictores no aporta información significativa."),
    ]
    sig, nosig = [], []
    for termino in coef.index:
        if termino == "const":
            continue
        b, p = coef.loc[termino, "coeficiente"], coef.loc[termino, "p_valor"]
        lo, hi = coef.loc[termino, "IC_inf"], coef.loc[termino, "IC_sup"]
        texto = (f"*{termino}*: {_vt(b, 4)}{u} por unidad (IC [{_vt(lo, 4)}; {_vt(hi, 4)}], "
                 f"p = {_vt(p, 5)})")
        (sig if p < 0.05 else nosig).append(texto)
    if sig:
        out.append("Predictores con coeficiente significativo al 5%: " + "; ".join(sig) + ".")
    if nosig:
        out.append("Predictores sin evidencia de efecto al 5%: " + "; ".join(nosig) +
                   ". Su intervalo de confianza incluye el cero.")
    out.append(
        "Cada coeficiente es un efecto **parcial**: el cambio esperado en la respuesta por unidad "
        "del predictor MANTENIENDO CONSTANTES los demas. Esa condición puede no ser "
        "operacionalmente alcanzable, y el modelo describe asociación, no causalidad."
    )
    for a in r["advertencias"]:
        out.append(f"**Diagnóstico**: {a}")
    return out


def correlacion(res, respuesta: str) -> list[str]:
    r = res["r"]
    if respuesta not in r.index:
        return [f"La variable {respuesta} no está en la matriz."]
    s = r[respuesta].drop(labels=[respuesta]).sort_values(key=np.abs, ascending=False)
    p = res["p"][respuesta]
    out = [f"Correlaciones de **{respuesta}** con el resto de variables "
           f"(método {res['metodo']}, n = {res['n']}), ordenadas por magnitud:"]
    for v in s.index[:6]:
        rv, pv = s[v], p[v]
        fuerza = ("muy fuerte" if abs(rv) >= 0.9 else "fuerte" if abs(rv) >= 0.7 else
                  "moderada" if abs(rv) >= 0.4 else "debil" if abs(rv) >= 0.2 else "muy débil")
        signo = "positiva" if rv > 0 else "negativa"
        out.append(f"- **{v}**: r = {_vt(rv, 3)} (asociación {signo} {fuerza}, "
                   f"R2 bivariado = {_vt(rv**2, 3)}, p = {_vt(pv, 5)})")
    mayor = s.index[0]
    out.append(
        f"La asociación más fuerte es con **{mayor}** (r = {_vt(s[mayor], 3)}), que por si sola "
        f"explicaría el {_vt(100*s[mayor]**2, 1)}% de la varianza de {respuesta} en un modelo "
        "bivariado. Esto es una asociación observada: no implica que actuar sobre esa variable "
        "produzca el cambio, porque puede haber variables de confusión que afecten a ambas."
    )
    return out


# ------------------------------------------------------------------- DOE
def doe(r, unidad: str = "") -> list[str]:
    u = f" {unidad}" if unidad else ""
    e = r["efectos"].sort_values("efecto", key=np.abs, ascending=False)
    out = [
        f"Diseño 2^{r['k']} con {r['n_corridas']} corridas y {r['replicas']} replica(s) "
        f"({r['N']} observaciones). Método de evaluación: {r['metodo']}. "
        f"Respuesta media global: {_vt(r['media_global'])}{u}.",
        f"Modelo en variables codificadas:\n\n`{r['ecuacion_codificada']}`",
    ]
    lineas = []
    for row in e.itertuples():
        marca = "**significativo**" if row.termino in r.get("significativos", []) else "no significativo"
        tipo = "efecto principal" if row.orden == 1 else f"interacción de orden {row.orden}"
        lineas.append(f"- *{row.termino}* ({tipo}): efecto = {_vt(row.efecto)}{u}, "
                      f"{_vt(row.pct_contribucion, 1)}% de la suma de cuadrados, {marca}")
    out.append("Efectos estimados, ordenados por magnitud:\n" + "\n".join(lineas))

    dominante = e.iloc[0]
    out.append(
        f"El efecto dominante es **{dominante.termino}** con {_vt(dominante.efecto)}{u}: pasar "
        f"ese factor de su nivel bajo al alto "
        f"{'reduce' if dominante.efecto < 0 else 'aumenta'} la respuesta en "
        f"{_vt(abs(dominante.efecto))}{u} en promedio."
    )
    inter = [t for t in r.get("significativos", []) if ":" in t]
    if inter:
        out.append(
            f"Hay interacción significativa ({', '.join(inter)}): el efecto de un factor "
            "**depende del nivel del otro**, de modo que los efectos principales involucrados no "
            "pueden interpretarse por separado. La combinación optima debe leerse en el gráfico "
            "de interacción, no sumando efectos principales."
        )
    else:
        out.append("No se detecta interacción significativa: los efectos de los factores son "
                   "aproximadamente aditivos y pueden optimizarse por separado.")
    if "r2" in r:
        out.append(f"El modelo explica el {_vt(100*r['r2'], 1)}% de la variabilidad; el error "
                   f"experimental puro tiene desviación estándar {_vt(r['s_error'])}{u} con "
                   f"{r['gl_error']} grados de libertad.")
    if "pse_lenth" in r:
        out.append(f"Sin replicas, el error se estima con el pseudo-error de Lenth "
                   f"(PSE = {_vt(r['pse_lenth'], 4)}); el margen de error es "
                   f"{_vt(r['margen_error_ME'], 4)}{u}. Los efectos que lo superan se consideran "
                   "activos. No hay p-valores exactos.")
    for a in r["advertencias"]:
        out.append(f"**Advertencia**: {a}")
    return out


# ------------------------------------------------------------- cartas SPC
def carta(res, etiqueta: str = "") -> list[str]:
    n = len(res.puntos)
    fuera = res.fuera_de_control
    out = [
        f"**{res.nombre}** sobre {n} puntos. Línea central = {_vt(res.centro[0], 4)}; "
        + (f"límites de control [{_vt(res.lci[0], 4)}; {_vt(res.lcs[0], 4)}]."
           if len(set(np.round(res.lcs, 9))) == 1 else
           "límites de control variables (dependen del tamaño de cada subgrupo).")
    ]
    if res.sigma:
        out.append(f"Desviación estándar del estadístico graficado: {_vt(res.sigma, 5)}. "
                   "Los límites están a 3 de esas desviaciones de la línea central.")
    if getattr(res, "origen_limites", ""):
        out.append(f"**Origen de los límites.** {res.origen_limites}")
    if getattr(res, "excluidos", []):
        out.append(
            f"Se excluyeron del cálculo {len(res.excluidos)} punto(s) con causa asignable "
            f"identificada (posiciones {', '.join(str(i+1) for i in res.excluidos)}). "
            "Siguen graficados para dejar constancia, pero no contaminan los límites. "
            "Excluir un punto solo es legítimo si la causa se identificó y se corrigió: "
            "hacerlo porque el punto incomoda invalida el análisis.")
    if not fuera:
        out.append(
            "**No hay señales de causa especial**: ningun punto cae fuera de los límites y no se "
            "detectan patrones no aleatorios. El proceso se comporta de forma estable y su "
            "variación es atribuible a causas comunes. En este estado, ajustar el proceso ante "
            "variaciones individuales lo empeoraría (sobrecorrección)."
        )
    else:
        from statistics_engine.control_charts import NELSON_DESC
        out.append(
            f"**Se detectan {len(fuera)} punto(s) con señal** de un total de {n} "
            f"({_vt(100*len(fuera)/n, 1)}%): subgrupos "
            f"{', '.join(str(i+1) for i in fuera[:20])}"
            + (" ..." if len(fuera) > 20 else "") + "."
        )
        for regla, idxs in sorted(res.violaciones.items()):
            out.append(f"- Regla {regla} ({NELSON_DESC[regla]}): {len(idxs)} punto(s), "
                       f"en las posiciones {', '.join(str(i+1) for i in idxs[:12])}"
                       + (" ..." if len(idxs) > 12 else ""))
        out.append(
            "Cada señal debe **investigarse** para identificar la causa asignable antes de "
            "recalcular los límites. Un proceso con señales no está en control estadístico, por "
            "lo que **los índices de capacidad calculados sobre estos datos no son predictivos**."
        )
    for nt in res.notas:
        out.append(f"*{nt}*")
    return out


# --------------------------------------------------------------- Pareto
def pareto(r) -> list[str]:
    t = r["tabla"]
    out = [r["lectura"]]
    top = t.head(3)
    detalle = "; ".join(f"**{row.categoria}** ({_vt(row.valor, 1)} = {_vt(row.pct, 1)}%)"
                        for row in top.itertuples())
    out.append(f"Las categorías de mayor peso son: {detalle}.")
    if r["concentrado"]:
        out.append(
            f"La distribución está concentrada: actuar sobre {r['n_vitales']} de "
            f"{r['n_categorias']} categorías cubre el {_vt(r['pct_cubierto_por_vitales'], 1)}% del "
            f"total de {r['etiqueta_valor'].lower()}. Ahí debe ir el esfuerzo de mejora."
        )
    out.append(
        "Atención a la unidad de análisis: ordenar por FRECUENCIA de eventos y ordenar por "
        "IMPACTO (minutos, toneladas, costo) puede producir rankings distintos, y el relevante "
        "para decidir es el de impacto."
    )
    return out


# ------------------------------------------------- comparacion de escenarios
def comparacion(antes, despues, etiqueta: str, unidad: str = "",
                prueba_t=None) -> list[str]:
    a, d = pd.Series(antes).dropna(), pd.Series(despues).dropna()
    u = f" {unidad}" if unidad else ""
    dm = d.mean() - a.mean()
    ds = d.std(ddof=1) - a.std(ddof=1)
    out = [
        f"*{etiqueta}* pasa de media {_vt(a.mean())}{u} (s = {_vt(a.std(ddof=1))}) a "
        f"{_vt(d.mean())}{u} (s = {_vt(d.std(ddof=1))}).",
        f"Cambio en la media: **{_vt(dm)}{u}** ({_vt(100*dm/a.mean(), 1)}%). "
        f"Cambio en la desviación estándar: **{_vt(ds)}{u}** "
        f"({_vt(100*ds/a.std(ddof=1), 1)}%).",
    ]
    if abs(ds / a.std(ddof=1)) > abs(dm / a.mean()):
        out.append("La reducción de **dispersión** es proporcionalmente mayor que la de la media. "
                   "Para la planificación esto suele valer más: un proceso más predecible permite "
                   "operar con menos holgura aunque su promedio cambie poco.")
    if prueba_t is not None:
        out += prueba(prueba_t, unidad=unidad)
    out.append("Esta comparación proviene de un **modelo de simulación**, no de una prueba en "
               "terreno: es una consecuencia lógica de los supuestos del modelo y debe validarse "
               "con un piloto antes de comprometer el beneficio.")
    return out


# ---------------------------------------------------------------- CUSUM --
def cusum(res) -> list[str]:
    n = len(res.c_mas)
    out = [
        f"**CUSUM tabular de dos lados** sobre {n} observaciones. Objetivo mu0 = "
        f"{_vt(res.mu0, 4)}, sigma = {_vt(res.sigma, 4)}, holgura K = {_vt(res.K, 4)} "
        f"({_vt(res.k_sigma, 2)} sigma) y umbral H = {_vt(res.H, 4)} "
        f"({_vt(res.h_sigma, 2)} sigma).",
        f"{res.origen_limites}",
    ]
    exc_mas, exc_menos = res.excursiones_mas, res.excursiones_menos
    if not res.fuera_de_control:
        out.append(
            "**No hay señales**: la suma acumulada nunca superó el umbral en ninguno de los "
            "dos lados. No hay evidencia de un corrimiento sostenido del nivel del proceso "
            "respecto del objetivo."
        )
    else:
        if exc_mas:
            inicio = min(exc_mas) + 1
            out.append(
                f"**Corrimiento AL ALZA detectado.** {len(exc_mas)} excursión(es) sobre H; la "
                f"primera señal aparece en la observación {inicio}. El proceso está operando "
                f"por encima del objetivo de forma sostenida."
            )
        if exc_menos:
            inicio = min(exc_menos) + 1
            out.append(
                f"**Corrimiento A LA BAJA detectado.** {len(exc_menos)} excursión(es) sobre H; "
                f"la primera señal aparece en la observación {inicio}."
            )
        out.append(
            f"Se contaron {len(res.excursiones)} excursión(es) distinta(s) y no "
            f"{len(res.fuera_de_control)} puntos sueltos: sin reinicio, una sola excursión deja "
            "varios puntos consecutivos por encima del umbral, de modo que contar puntos "
            "sobreestima el número de eventos."
        )
        # Dónde empezó realmente: último punto en que la acumulación valía cero
        import numpy as _np
        lado = res.c_mas if exc_mas else res.c_menos
        primera = (min(exc_mas) if exc_mas else min(exc_menos))
        ceros = _np.where(lado[:primera + 1] == 0)[0]
        if ceros.size:
            out.append(
                f"La acumulación despegó de cero en la observación {int(ceros[-1]) + 2}: ese es "
                "el punto donde el corrimiento probablemente comenzó, varias observaciones antes "
                "de que la carta lo señalara. Esa estimación del inicio es la ventaja práctica "
                "del CUSUM frente a Shewhart."
            )
    for nt in res.notas:
        out.append(f"*{nt}*")
    return out


# ---------------------------------------------------------- Monte Carlo --
def montecarlo(res, umbral: float | None = None, etiqueta_umbral: str = "el umbral",
               mayor_es_mejor: bool = True) -> list[str]:
    r = res.resumen()
    lo, hi = res.intervalo(0.90)
    u = f" {res.unidad_salida}" if res.unidad_salida else ""
    n_txt = f"{res.n:,}".replace(",", " ")
    out = [
        f"**{n_txt} simulaciones** de *{res.nombre_salida}* (semilla {res.semilla}). "
        f"Media {_vt(r['media'], 2)}{u}, mediana {_vt(r['mediana'], 2)}{u}, desviación "
        f"estándar {_vt(r['desv_std'], 2)}{u}.",
        f"El 90% de los resultados cae entre **{_vt(lo, 2)}** y **{_vt(hi, 2)}**{u} "
        f"(percentiles 5 y 95). El rango completo simulado va de {_vt(r['minimo'], 2)} a "
        f"{_vt(r['maximo'], 2)}{u}.",
    ]
    sesgo = r["media"] - r["mediana"]
    if r["desv_std"] > 0 and abs(sesgo) / r["desv_std"] > 0.1:
        lado = "derecha" if sesgo > 0 else "izquierda"
        out.append(
            f"La media supera a la mediana en {_vt(sesgo, 2)}{u}" if sesgo > 0 else
            f"La mediana supera a la media en {_vt(-sesgo, 2)}{u}"
        )
        out[-1] += (f": la distribución es asimétrica hacia la {lado}. Con distribuciones "
                    "asimétricas la media no es el valor típico; para decidir conviene mirar "
                    "la mediana y los percentiles.")
    if umbral is not None:
        p_sup = res.probabilidad_mayor_que(umbral)
        p_inf = 1 - p_sup
        favorable, adverso = (p_sup, p_inf) if mayor_es_mejor else (p_inf, p_sup)
        out.append(
            f"**Probabilidad de superar {etiqueta_umbral} ({_vt(umbral, 2)}{u}): "
            f"{p_sup:.1%}.** Probabilidad de quedar por debajo: {p_inf:.1%}."
        )
        if adverso < 0.05:
            out.append(f"El resultado adverso tiene una probabilidad de {adverso:.1%}: la "
                       "decisión es robusta frente a la incertidumbre declarada.")
        elif adverso < 0.25:
            out.append(f"El resultado adverso tiene una probabilidad de {adverso:.1%}: hay "
                       "riesgo apreciable. Conviene dimensionar qué pérdida implicaría y si "
                       "es tolerable, no solo mirar el valor esperado.")
        else:
            out.append(f"El resultado adverso tiene una probabilidad de {adverso:.1%}: **el "
                       "valor esperado favorable esconde un riesgo alto.** Decidir solo por la "
                       "media sería un error; considere reducir la incertidumbre antes de "
                       "comprometerse, o diseñar la alternativa para limitar la pérdida.")

    err = res.error_montecarlo()
    rel = 100 * err / abs(r["media"]) if r["media"] else float("nan")
    out.append(
        f"Error de muestreo de la media: {_vt(err, 3)}{u}"
        + (f" ({_vt(rel, 2)}% de la media)" if rel == rel else "")
        + f". Baja como 1/raíz(n), así que cuadruplicar las {n_txt} simulaciones solo lo "
          "reduciría a la mitad."
    )

    s = res.sensibilidad()
    activas = s[~s["fija"]]
    if not activas.empty:
        dom = activas.iloc[0]
        out.append(
            f"**Dónde está la incertidumbre.** *{dom['entrada']}* explica el "
            f"{_vt(dom['contribucion_pct'], 1)}% de la varianza del resultado "
            f"(correlación de rangos {_vt(dom['correlacion_spearman'], 3)}). "
            "Reducir la incertidumbre de esa entrada, con un piloto o una medición mejor, "
            "es lo que más estrecharía la distribución del resultado."
        )
        resto = activas.iloc[1:]
        if not resto.empty:
            detalle = "; ".join(f"*{row.entrada}* {row.contribucion_pct:.1f}%"
                                for row in resto.itertuples())
            out.append(f"Las demás entradas contribuyen: {detalle}.")
    fijas = s[s["fija"]]["entrada"].tolist()
    if fijas:
        out.append(f"Declaradas sin incertidumbre y por tanto fuera del tornado: "
                   f"{', '.join(fijas)}. Si alguna de ellas en realidad varía, esta "
                   "simulación subestima la dispersión del resultado.")
    out.append(
        "Recordatorio: Monte Carlo propaga la incertidumbre que usted declaró, no la que "
        "existe. Si las distribuciones de entrada están mal elegidas, estos números están "
        "igual de mal, solo que con apariencia de rigor. Además se supone independencia "
        "entre las entradas: si en realidad se mueven juntas, la dispersión real es mayor."
    )
    return out
