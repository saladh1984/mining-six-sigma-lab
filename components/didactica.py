"""Catalogo didáctico: cada herramienta en tres secciones + supuestos y limitaciones.

Estructura exigida por la especificación (punto 12):
  1. Que es?          explicación conceptual breve
  2. Como funciona?   formula, variables y procedimiento
  3. Como se interpreta?  lectura del resultado aplicado al ejemplo minero

Se agregan dos secciones que la práctica estadística exige y que la
especificación pide en el punto 11: SUPUESTOS y LIMITACIONES.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ficha:
    titulo: str
    que_es: str
    como_funciona: str
    formulas: list[tuple[str, str]]      # (latex, descripcion de variables)
    procedimiento: list[str]
    como_se_interpreta: str
    supuestos: list[str] = field(default_factory=list)
    limitaciones: list[str] = field(default_factory=list)


FICHAS: dict[str, Ficha] = {}


def _f(clave: str, **kw) -> None:
    FICHAS[clave] = Ficha(**kw)


# ------------------------------------------------------------- MEDIR -----
_f("descriptiva",
   titulo="Estadística descriptiva",
   que_es="Conjunto de medidas que resumen un conjunto de datos en pocos números: "
          "donde está centrado (tendencia central), cuánto se dispersa (variabilidad) "
          "y que forma tiene (asimetría y curtosis). Es el punto de partida obligatorio "
          "de cualquier análisis: describir antes de modelar.",
   como_funciona="Cada medida se calcula directamente sobre las n observaciones. La "
                 "varianza y la desviación estándar usan n-1 en el denominador (corrección "
                 "de Bessel) porque estimamos la media con los mismos datos, lo que consume "
                 "un grado de libertad.",
   formulas=[
       (r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i",
        "media muestral; x_i es la observación i, n el número de observaciones"),
       (r"s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2",
        "varianza muestral; s es la desviación estándar, su raiz cuadrada"),
       (r"CV = \frac{s}{\bar{x}}",
        "coeficiente de variación: dispersión relativa, adimensional"),
       (r"IQR = Q_3 - Q_1",
        "rango intercuartilico: amplitud del 50% central de los datos"),
   ],
   procedimiento=[
       "Eliminar o declarar los valores faltantes.",
       "Calcular la media y la mediana; si difieren mucho, la distribución es asimétrica.",
       "Calcular s y el CV para comparar dispersión entre variables de distinta unidad.",
       "Obtener cuartiles e IQR, y aplicar la regla de Tukey para identificar candidatos a outlier.",
       "Contrastar el histograma con el boxplot antes de concluir.",
   ],
   como_se_interpreta="En el ciclo de camiones, una media de 33.5 min con s = 7.7 min da "
                      "CV = 23%: por cada 100 minutos de ciclo medio hay 23 minutos de "
                      "dispersión típica. Ese CV es la cifra que justifica el proyecto, porque "
                      "es la dispersión, y no la media, lo que obliga a sobredimensionar la flota. "
                      "Si media y mediana difieren, la distribución está sesgada y la media deja "
                      "de ser un buen resumen del comportamiento típico.",
   supuestos=["Ninguno distribucional: son descripciones, no inferencias.",
              "Las observaciones deben provenir del mismo proceso; si se mezclan turnos o "
              "equipos con comportamientos distintos, el resumen global oculta esa mezcla."],
   limitaciones=["La media es sensible a valores extremos; la mediana no.",
                 "La moda no está definida de forma natural en variables continuas: aquí se "
                 "reporta sobre datos redondeados y el método queda declarado.",
                 "Un resumen descriptivo no dice nada sobre estabilidad en el tiempo: para eso "
                 "se necesita una carta de control."])

_f("capacidad",
   titulo="Análisis de capacidad del proceso",
   que_es="Compara la VOZ DEL PROCESO (su dispersión natural) con la VOZ DEL CLIENTE "
          "(los límites de especificación) para cuantificar si el proceso es capaz de "
          "cumplir el requisito. Cp y Cpk miden capacidad POTENCIAL de corto plazo; "
          "Pp y Ppk miden DESEMPEÑO real de largo plazo.",
   como_funciona="Cp y Cpk usan sigma DENTRO de subgrupo, estimada con Rbar/d2 (o con el "
                 "rango movil si hay datos individuales): representa la variación de causas "
                 "comunes. Pp y Ppk usan la desviación estándar TOTAL, que incluye además la "
                 "variación entre subgrupos. Cp mide solo la amplitud; Cpk además penaliza el "
                 "descentrado.",
   formulas=[
       (r"C_p = \frac{LSE - LIE}{6\,\hat{\sigma}_{dentro}}",
        "capacidad potencial: cuántas veces cabe la dispersión natural en la ventana"),
       (r"C_{pk} = \min\left(\frac{LSE-\bar{x}}{3\hat{\sigma}_{dentro}},\ "
        r"\frac{\bar{x}-LIE}{3\hat{\sigma}_{dentro}}\right)",
        "capacidad real: toma el lado más desfavorable respecto del centro"),
       (r"P_p = \frac{LSE - LIE}{6\,s}\qquad P_{pk} = \min\left(\frac{LSE-\bar{x}}{3s},\ "
        r"\frac{\bar{x}-LIE}{3s}\right)",
        "desempeño de largo plazo, con s = desviación estándar total"),
       (r"\hat{\sigma}_{dentro} = \frac{\bar{R}}{d_2(n)} \quad\text{o}\quad "
        r"\frac{\overline{MR}}{1.128}",
        "sigma de corto plazo desde subgrupos o desde rangos moviles"),
   ],
   procedimiento=[
       "PRIMERO: verificar que el proceso este en control estadístico con una carta de control.",
       "Verificar la normalidad de los datos (Shapiro-Wilk y gráfico Q-Q).",
       "Definir LIE y LSE desde el requisito del cliente, NUNCA desde los datos.",
       "Estimar sigma dentro de subgrupo y sigma total.",
       "Calcular los cuatro índices y compararlos entre si.",
   ],
   como_se_interpreta="Para el ciclo de camiones la especificación es unilateral (LSE = 38 min), "
                      "así que Cp no está definido y Cpk se reduce a Cpu. Un Cpk menor que 1 "
                      "significa que la dispersión natural del proceso no cabe dentro de la "
                      "ventana: hay ciclos fuera de meta de forma estructural, no por mala suerte. "
                      "Si Cp es bastante mayor que Pp, el proceso tiene potencial pero no es "
                      "estable: existe variación entre subgrupos (turnos, fases, operadores) que "
                      "primero hay que eliminar. Convenciones habituales: Cpk >= 1.33 capaz, "
                      "1.00-1.33 marginal, < 1.00 no capaz.",
   supuestos=["Proceso en control estadístico. Sin estabilidad, el índice no predice nada.",
              "Datos aproximadamente normales: la conversión de sigmas a PPM lo requiere.",
              "Límites de especificación externos al proceso y correctamente definidos.",
              "Muestra suficiente: AIAG sugiere 100 observaciones o más para Pp/Ppk."],
   limitaciones=["Con datos sesgados (los tiempos de ciclo suelen serlo) el PPM estimado puede "
                 "errar por ordenes de magnitud.",
                 "Un Cpk alto sobre un proceso inestable es un número sin significado predictivo.",
                 "Los índices son estimaciones: con n pequeño su intervalo de confianza es amplio.",
                 "Cp no detecta descentrado; Cpk no detecta si el proceso está desplazándose."])

_f("gage",
   titulo="Gage R&R (análisis del sistema de medición)",
   que_es="Cuantifica que parte de la variabilidad observada proviene del SISTEMA DE MEDICIÓN "
          "y no del proceso. Si el instrumento aporta demasiado ruido, buena parte de lo que "
          "parece variabilidad del proceso es error de medición, y cualquier mejora se "
          "evaluara sobre datos poco confiables.",
   como_funciona="Se descompone la varianza total mediante un ANOVA de dos factores con "
                 "interacción: pieza, operador y su interacción. La repetibilidad es la varianza "
                 "del error (mismo operador, misma pieza); la reproducibilidad es la varianza "
                 "atribuible al operador y a su interacción con la pieza.",
   formulas=[
       (r"\sigma^2_{total} = \sigma^2_{pieza} + \sigma^2_{repetibilidad} + \sigma^2_{reproducibilidad}",
        "descomposición de la varianza observada"),
       (r"\sigma^2_{R\&R} = \sigma^2_{repetibilidad} + \sigma^2_{reproducibilidad}",
        "variación total del sistema de medición"),
       (r"\%VE = 100 \times \frac{\sigma_{R\&R}}{\sigma_{total}}",
        "porcentaje de variación de estudio: se calcula sobre DESVIACIONES"),
       (r"ndc = 1.41 \times \frac{\sigma_{pieza}}{\sigma_{R\&R}}",
        "número de categorías distinguibles por el sistema de medición"),
   ],
   procedimiento=[
       "Diseñar un estudio CRUZADO: cada operador mide cada pieza, 2 o 3 veces.",
       "Elegir 10 piezas que cubran el RANGO REAL del proceso, no piezas similares.",
       "Aleatorizar el orden y medir a ciegas respecto de los valores anteriores.",
       "Ejecutar el ANOVA y obtener los componentes de varianza.",
       "Comparar %VE contra los criterios AIAG y verificar ndc >= 5.",
   ],
   como_se_interpreta="Criterios AIAG sobre %VE: menos de 10% el sistema es aceptable; entre 10% "
                      "y 30% es condicional según la criticidad; sobre 30% es inaceptable y debe "
                      "corregirse ANTES de analizar el proceso. Para el ciclo de camiones, el "
                      "'instrumento' es el criterio con que el sistema de despacho marca el inicio "
                      "y el fin del ciclo: si distintos registros discrepan, parte de la "
                      "variabilidad que atribuimos a la operación es en realidad ruido de medición.",
   supuestos=["Diseño cruzado y balanceado: todas las combinaciones pieza x operador con igual "
              "número de replicas.",
              "Las piezas seleccionadas representan el rango de variación real del proceso.",
              "El proceso de medición es estable durante el estudio."],
   limitaciones=["%VE depende de la variación de las piezas elegidas: piezas muy parecidas "
                 "inflan artificialmente el %R&R aunque el instrumento sea bueno.",
                 "Evalua PRECISIÓN, no EXACTITUD. El sesgo, la linealidad y la estabilidad "
                 "requieren un patron de referencia trazable.",
                 "%Contribución (sobre varianzas) y %VE (sobre desviaciones) no son comparables: "
                 "el criterio 10/30% aplica al segundo."])

_f("pareto",
   titulo="Diagrama de Pareto",
   que_es="Gráfico de barras ordenadas de mayor a menor con una línea de porcentaje acumulado. "
          "Sirve para separar los 'pocos vitales' de los 'muchos triviales' y decidir donde "
          "concentrar el esfuerzo de mejora.",
   como_funciona="Se agrupan los eventos por categoría, se ordenan por magnitud descendente y se "
                 "acumula el porcentaje. En esta aplicación las barras se expresan en porcentaje "
                 "del total, igual que la línea acumulada, para que ambas compartan un solo eje.",
   formulas=[
       (r"p_i = \frac{v_i}{\sum_j v_j} \times 100",
        "porcentaje de la categoría i sobre el total"),
       (r"P_k = \sum_{i=1}^{k} p_i",
        "porcentaje acumulado hasta la categoría k, con las categorías ya ordenadas"),
   ],
   procedimiento=[
       "Definir la unidad de análisis: frecuencia de eventos o IMPACTO (minutos, toneladas, costo).",
       "Agrupar y sumar por categoría.",
       "Ordenar de mayor a menor y acumular.",
       "Identificar las categorías que cubren el corte elegido (habitualmente 80%).",
   ],
   como_se_interpreta="Si tres de ocho categorías concentran el 86% de los minutos perdidos, el "
                      "esfuerzo debe ir a esas tres. Atención a la unidad: ordenar por FRECUENCIA "
                      "y ordenar por IMPACTO suele dar rankings distintos, y el relevante para "
                      "decidir es el de impacto. Una falla mecánica puede ocurrir pocas veces y "
                      "aún así dominar los minutos perdidos.",
   supuestos=["Las categorías deben ser mutuamente excluyentes y colectivamente exhaustivas.",
              "Los valores deben ser aditivos y no negativos."],
   limitaciones=["El principio 80/20 es una regularidad empirica, no una ley: hay procesos con "
                 "distribución plana y en ese caso no existe un grupo dominante.",
                 "Es descriptivo: muestra donde esta el problema, no por qué ocurre.",
                 "Una mala definición de categorías puede esconder la causa real en 'Otros'."])

_f("muestreo",
   titulo="Muestreo de datos",
   que_es="Procedimiento para seleccionar un subconjunto de observaciones que represente al "
          "proceso completo. El tamaño y el método de la muestra determinan la precisión de "
          "todo lo que se concluya después.",
   como_funciona="Para estimar una media con un margen de error E y confianza 1-alfa, el tamaño "
                 "requerido depende de la variabilidad del proceso. Para una carta de control, "
                 "el muestreo debe producir SUBGRUPOS RACIONALES: observaciones tomadas bajo "
                 "condiciones lo más homogéneas posible.",
   formulas=[
       (r"n = \left(\frac{z_{\alpha/2}\,\sigma}{E}\right)^2",
        "tamaño de muestra para estimar una media con margen E"),
       (r"IC_{1-\alpha} = \bar{x} \pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt{n}}",
        "intervalo de confianza para la media"),
   ],
   procedimiento=[
       "Definir que se quiere estimar y con que precisión.",
       "Estimar sigma con un estudio piloto o con datos historicos.",
       "Calcular n y verificar que sea factible operacionalmente.",
       "Elegir el método: aleatorio simple, sistemático o estratificado por turno/equipo.",
       "Para cartas de control, formar subgrupos racionales consecutivos.",
   ],
   como_se_interpreta="Con s = 7.7 min, estimar el ciclo medio con un margen de +/- 1 min al 95% "
                      "requiere n = (1.96 x 7.7 / 1)^2 = 228 ciclos. Si solo se dispone de 50, el "
                      "margen real será de +/- 2.1 min, y conclusiones más finas que eso no están "
                      "respaldadas por los datos.",
   supuestos=["Muestreo aleatorio o, al menos, no sistemáticamente sesgado.",
              "Observaciones independientes entre si."],
   limitaciones=["Un muestreo por conveniencia (solo el turno de día, solo un camion) produce "
                 "estimaciones sesgadas que ningun tamaño de muestra corrige.",
                 "Aumentar n reduce el error aleatorio, nunca el sesgo sistemático."])

# ----------------------------------------------------------- ANALIZAR ----
_f("correlacion",
   titulo="Correlación",
   que_es="Mide la fuerza y dirección de la asociación entre dos variables. Pearson mide "
          "asociación LINEAL; Spearman mide asociación MONOTONA sobre los rangos y es robusta "
          "a valores extremos y a relaciones curvas pero crecientes.",
   como_funciona="Pearson normaliza la covarianza por el producto de las desviaciones estándar, "
                 "obteniendo un valor entre -1 y +1. Spearman aplica la misma formula a los "
                 "RANGOS de los datos en vez de a los valores.",
   formulas=[
       (r"r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}"
        r"{\sqrt{\sum (x_i-\bar{x})^2}\sqrt{\sum (y_i-\bar{y})^2}}",
        "coeficiente de Pearson; r = +1 asociación lineal positiva perfecta"),
       (r"r_s = r_{\text{sobre los rangos de } x,\, y}",
        "coeficiente de Spearman: Pearson aplicado a los rangos"),
       (r"R^2 = r^2",
        "proporción de la varianza de y explicada linealmente por x (solo bivariado)"),
   ],
   procedimiento=[
       "Graficar primero el diagrama de dispersión: un r alto puede provenir de un solo outlier.",
       "Calcular Pearson si la relación parece lineal; Spearman si es monotona pero curva.",
       "Revisar el p-valor junto con la magnitud: con n grande, r = 0.1 puede ser significativo "
       "y a la vez irrelevante.",
   ],
   como_se_interpreta="Si la distancia de transporte correlaciona 0.91 con el tiempo de ciclo, "
                      "la distancia explica aproximadamente el 83% de su varianza (R2 = 0.83). "
                      "Pero correlación NO es causalidad: en estos datos la distancia y la "
                      "pendiente dependen ambas de la fase del rajo, de modo que parte de la "
                      "asociación observada entre pendiente y ciclo es atribuible a la fase, no "
                      "a la pendiente en si.",
   supuestos=["Pearson: relación lineal y, para el p-valor, normalidad bivariada aproximada.",
              "Observaciones independientes."],
   limitaciones=["No distingue causalidad, causalidad inversa ni variable de confusión.",
                 "Pearson no detecta relaciones no lineales: una parabola puede dar r = 0.",
                 "Es muy sensible a outliers y al rango de los datos observados."])

_f("regresion",
   titulo="Regresión lineal (simple y múltiple)",
   que_es="Modela la relación entre una variable respuesta y uno o más predictores, estimando "
          "cuánto cambia la respuesta por unidad de cada predictor manteniendo los demas "
          "constantes. Permite cuantificar y predecir, no solo describir.",
   como_funciona="Mínimos cuadrados ordinarios elige los coeficientes que minimizan la suma de "
                 "los residuos al cuadrado. Cada coeficiente es un efecto PARCIAL: el cambio "
                 "esperado en y por unidad de x_j con los demas predictores fijos.",
   formulas=[
       (r"y_i = \beta_0 + \beta_1 x_{1i} + \dots + \beta_k x_{ki} + \varepsilon_i",
        "modelo; beta_j son los coeficientes, epsilon el error aleatorio"),
       (r"\hat{\beta} = (X^TX)^{-1}X^Ty",
        "estimador de mínimos cuadrados en forma matricial"),
       (r"R^2 = 1 - \frac{SC_{residual}}{SC_{total}} \qquad "
        r"R^2_{aj} = 1 - \frac{SC_{res}/(n-k-1)}{SC_{tot}/(n-1)}",
        "bondad de ajuste; el ajustado penaliza el número de predictores k"),
       (r"VIF_j = \frac{1}{1-R_j^2}",
        "factor de inflación de varianza: multicolinealidad del predictor j"),
   ],
   procedimiento=[
       "Explorar primero las correlaciones y los diagramas de dispersión.",
       "Ajustar el modelo y revisar la significancia conjunta (prueba F) antes que los coeficientes.",
       "Diagnosticar los residuos: linealidad, varianza constante, normalidad, independencia.",
       "Revisar VIF para detectar multicolinealidad.",
       "Identificar observaciones influyentes con la distancia de Cook.",
   ],
   como_se_interpreta="Un coeficiente de 2.8 min/km para la distancia significa que, a igual "
                      "pendiente, estado de via y carga, cada kilometro adicional agrega 2.8 min "
                      "al ciclo. El R2 ajustado indica que fracción de la variabilidad queda "
                      "explicada; el resto es variación no capturada por el modelo. Comparar la "
                      "regresión simple con la múltiple revela el sesgo por variable omitida: si "
                      "el coeficiente de la distancia cambia al agregar la pendiente, es porque "
                      "ambas estaban confundidas.",
   supuestos=["Linealidad en los parámetros.",
              "Errores con media cero y varianza constante (homocedasticidad).",
              "Errores independientes (crítico en series temporales).",
              "Errores aproximadamente normales, para que los IC y p-valores sean exactos.",
              "Predictores sin multicolinealidad severa."],
   limitaciones=["Estima asociación condicional, no causalidad.",
                 "'Manteniendo lo demas constante' puede ser operacionalmente imposible: no se "
                 "puede cambiar la distancia sin cambiar la fase del rajo.",
                 "Extrapolar fuera del rango observado no está respaldado por los datos.",
                 "Con multicolinealidad alta los coeficientes individuales son inestables aunque "
                 "el R2 global sea bueno."])

_f("hipotesis",
   titulo="Pruebas de hipótesis",
   que_es="Procedimiento para decidir si la evidencia muestral es suficiente para rechazar una "
          "afirmación de partida (la hipótesis nula H0), controlando la probabilidad de "
          "rechazarla por azar cuando es cierta (error tipo I, alfa).",
   como_funciona="Se calcula un estadístico que mide cuan lejos están los datos de lo esperado "
                 "bajo H0, y se obtiene el p-valor: la probabilidad de observar un resultado al "
                 "menos tan extremo COMO EL OBSERVADO SI H0 FUERA CIERTA. Si p < alfa, se rechaza H0.",
   formulas=[
       (r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}",
        "prueba t de Welch para dos muestras independientes"),
       (r"t = \frac{\bar{d}}{s_d/\sqrt{n}}",
        "prueba t pareada, sobre las diferencias d_i dentro de cada par"),
       (r"F = \frac{CM_{entre}}{CM_{dentro}}",
        "ANOVA de un factor: razón de variabilidad entre grupos y dentro de grupos"),
       (r"d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}",
        "d de Cohen: tamaño del efecto, independiente de n"),
   ],
   procedimiento=[
       "Formular H0 y H1 ANTES de ver los datos.",
       "Verificar los supuestos: normalidad (Shapiro-Wilk) y homogeneidad de varianzas (Levene).",
       "Elegir la prueba correcta: pareada solo si las observaciones están emparejadas.",
       "Calcular el estadístico, el p-valor, el intervalo de confianza y el tamaño del efecto.",
       "Decidir con el IC y el efecto, no solo con el p-valor.",
   ],
   como_se_interpreta="Si la prueba t de día contra noche da p = 0.003, los datos son poco "
                      "compatibles con la hipótesis de que ambos turnos tienen el mismo ciclo "
                      "medio. El p-valor NO es la probabilidad de que H0 sea cierta ni la "
                      "probabilidad de equivocarse. Lo que decide la acción es el intervalo de "
                      "confianza: una diferencia de 0.8 min con IC [0.3; 1.3] es estadísticamente "
                      "detectable pero operacionalmente menor; una de 4 min cambia la planificación.",
   supuestos=["Observaciones independientes (el supuesto más importante y el menos verificable "
              "con los datos: depende del diseño del muestreo).",
              "Normalidad de los grupos, o n suficiente para que opere el teorema central del límite.",
              "ANOVA clasico: homogeneidad de varianzas entre grupos.",
              "Prueba pareada: cada par corresponde a la MISMA unidad experimental."],
   limitaciones=["Un p pequeño con n grande puede reflejar una diferencia irrelevante.",
                 "Un p grande no prueba que no haya diferencia: puede faltar potencia.",
                 "Hacer muchas pruebas infla el error tipo I; por eso ANOVA usa Tukey en vez de "
                 "múltiples pruebas t.",
                 "ANOVA indica que al menos un grupo difiere, no cual."])

_f("ishikawa",
   titulo="Diagrama de Ishikawa y cinco porqués",
   que_es="Herramientas de razonamiento causal estructurado. El Ishikawa (espina de pescado) "
          "organiza las causas potenciales en categorías (las 6M); los cinco porqués encadenan "
          "preguntas para pasar del síntoma a la causa raiz.",
   como_funciona="El Ishikawa no calcula: estructura el conocimiento del equipo para que no se "
                 "omitan familias enteras de causas. Los cinco porqués recorren la cadena causal "
                 "hacia atras hasta llegar a una causa accionable.",
   formulas=[],
   procedimiento=[
       "Definir el efecto (el problema) con precisión y en terminos medibles.",
       "Recorrer las 6M: Maquinaria, Método, Mano de obra, Material, Medio ambiente, Medición.",
       "Generar causas sin filtrarlas inicialmente.",
       "Priorizar las causas candidatas con una matriz causa-efecto.",
       "VERIFICAR las causas priorizadas con datos: ahí entran la regresión y el ANOVA.",
   ],
   como_se_interpreta="El resultado es una lista de HIPÓTESIS, no un hallazgo. Si el equipo "
                      "concluye que la causa raiz es la ausencia de caracterización estadística "
                      "del ciclo por ruta, esa afirmación debe contrastarse: si fuera cierta, la "
                      "asignación de despacho debería explicar una parte medible de la "
                      "variabilidad, y eso se verifica con un ANOVA o una regresión.",
   supuestos=["Participación de quienes conocen el proceso en detalle."],
   limitaciones=["Produce hipótesis, no evidencia: distintos equipos llegan a cadenas distintas.",
                 "Los cinco porqués asumen una cadena causal única y lineal; los sistemas reales "
                 "suelen tener causas múltiples e interactuantes.",
                 "Es vulnerable al sesgo de confirmación del equipo."])

# ----------------------------------------------------------- MEJORAR ----
_f("doe",
   titulo="Diseño de experimentos factorial 2^k",
   que_es="Método para estudiar simultaneamente el efecto de varios factores sobre una respuesta, "
          "probando todas las combinaciones de dos niveles (bajo y alto). A diferencia del "
          "análisis observacional, el DOE MANIPULA los factores, lo que permite afirmaciones "
          "causales dentro del rango experimentado.",
   como_funciona="Cada factor se codifica en -1 (bajo) y +1 (alto). El efecto principal de un "
                 "factor es la diferencia entre la respuesta media en su nivel alto y en su nivel "
                 "bajo. Las columnas del diseño son ortogonales, por lo que cada efecto se estima "
                 "de forma independiente de los demas.",
   formulas=[
       (r"\text{Efecto}_A = \bar{y}_{A+} - \bar{y}_{A-}",
        "efecto principal: cambio medio en la respuesta al pasar de bajo a alto"),
       (r"\beta_A = \frac{\text{Efecto}_A}{2}",
        "coeficiente del modelo en variables codificadas"),
       (r"SC_{efecto} = \frac{N\,(\text{Efecto})^2}{4}",
        "suma de cuadrados del efecto, con N el número total de corridas"),
       (r"PSE = 1.5 \times \text{mediana}\{|efecto_i| : |efecto_i| < 2.5 s_0\}",
        "pseudo-error de Lenth, para diseños sin replicas"),
   ],
   procedimiento=[
       "Seleccionar los factores y definir niveles bajo y alto operacionalmente alcanzables.",
       "Construir la matriz experimental completa (2^k corridas).",
       "ALEATORIZAR el orden de ejecución: es lo que protege de factores que derivan en el tiempo.",
       "Replicar si es posible: sin replicas no hay estimación independiente del error.",
       "Calcular efectos, interacciones y su significancia; graficar efectos principales e interacción.",
   ],
   como_se_interpreta="Un efecto de -1.60 min para la capacidad del balde significa que pasar de "
                      "28 a 42 m3 reduce el tiempo de carguio en 1.60 min en promedio. Una "
                      "interacción A:B de +0.35 significa que el beneficio de cambiar la posición "
                      "del camion es MENOR cuando el balde es grande: los dos factores no se suman "
                      "de forma independiente. Cuando hay interacción relevante, los efectos "
                      "principales no deben interpretarse por separado. En el gráfico de "
                      "interacción, líneas paralelas indican ausencia de interacción.",
   supuestos=["Aleatorización del orden de ejecución.",
              "Errores independientes, de varianza constante y aproximadamente normales.",
              "Los niveles elegidos son alcanzables y representativos."],
   limitaciones=["Solo detecta efectos LINEALES entre los dos niveles: no detecta curvatura. "
                 "Para eso se requieren puntos centrales o un diseño de superficie de respuesta.",
                 "Las conclusiones valen solo dentro del rango experimentado.",
                 "Sin replicas, el método de Lenth supone que la mayoría de los efectos son nulos "
                 "y tiene poca potencia (con k=3 solo hay ~2.3 grados de libertad efectivos).",
                 "El número de corridas crece como 2^k: con muchos factores se requieren diseños "
                 "fraccionados, que confunden efectos entre si."])

_f("simulacion",
   titulo="Simulación de escenarios",
   que_es="Evaluación del efecto de cambios en los parámetros operacionales sobre los indicadores "
          "de salida, usando el modelo del proceso. Permite comparar alternativas antes de "
          "implementarlas en terreno.",
   como_funciona="Se modifican los parámetros del generador del proceso (espera en pala, estado "
                 "de via, capacidad de balde) y se recalculan los indicadores sobre el conjunto "
                 "de datos resultante. La comparación antes/después usa las mismas herramientas "
                 "estadísticas que el resto de la aplicación.",
   formulas=[
       (r"\text{Productividad} = \frac{\text{carga}}{\text{ciclo}/60}",
        "toneladas por hora por camion"),
       (r"\text{Capacidad de flota} = N_{camiones} \times \frac{60}{T_{ciclo}} \times \text{carga} \times H",
        "toneladas por periodo, con H horas efectivas"),
   ],
   procedimiento=[
       "Definir el escenario base con los parámetros actuales.",
       "Modificar un parámetro a la vez para aislar su efecto.",
       "Recalcular los indicadores y comparar las distribuciones completas, no solo las medias.",
       "Verificar la diferencia con una prueba estadística.",
   ],
   como_se_interpreta="Si reducir la espera en pala desplaza el ciclo medio de 33.5 a 30.7 min, "
                      "la ganancia es de 2.8 min por ciclo. Traducido a capacidad: con 6 camiones "
                      "y 20 h efectivas, son aproximadamente 1.800 t/día adicionales. Importa "
                      "tanto o más la reducción de la DISPERSIÓN: un proceso más predecible "
                      "permite planificar con menos holgura.",
   supuestos=["El modelo del proceso representa adecuadamente la realidad.",
              "Los parámetros modificados son efectivamente controlables."],
   limitaciones=["Una simulación no es evidencia empirica: es una consecuencia lógica del modelo. "
                 "Si el modelo está mal especificado, el resultado también lo esta.",
                 "No captura efectos de segundo orden no modelados (congestion emergente, "
                 "respuestas de los operadores al cambio).",
                 "El beneficio proyectado debe validarse con una prueba piloto antes de comprometerlo."])

_f("priorizacion",
   titulo="Matrices de priorización y análisis costo-beneficio",
   que_es="Herramientas para ordenar alternativas de mejora según su impacto esperado, el "
          "esfuerzo requerido y su retorno económico, de modo que la decisión sea explicita y "
          "trazable en vez de intuitiva.",
   como_funciona="La matriz esfuerzo-impacto ubica cada alternativa en un plano y define cuatro "
                 "cuadrantes. El análisis costo-beneficio descuenta los flujos para comparar "
                 "alternativas con perfiles temporales distintos.",
   formulas=[
       (r"\text{Prioridad} = \frac{\text{Impacto}}{\text{Esfuerzo}}",
        "razón simple de priorización"),
       (r"VAN = \sum_{t=0}^{T} \frac{F_t}{(1+r)^t}",
        "valor actual neto; F_t es el flujo del periodo t y r la tasa de descuento"),
       (r"\text{Payback} = \frac{\text{Inversión}}{\text{Beneficio anual}}",
        "periodo simple de recuperación"),
   ],
   procedimiento=[
       "Listar las alternativas surgidas del análisis causal y del DOE.",
       "Estimar impacto y esfuerzo con criterios explicitos y una escala comun.",
       "Ubicarlas en la matriz e identificar las ganancias rápidas (alto impacto, bajo esfuerzo).",
       "Cuantificar económicamente las finalistas.",
       "Decidir considerando también el riesgo y la reversibilidad.",
   ],
   como_se_interpreta="Las alternativas de alto impacto y bajo esfuerzo se ejecutan primero. Las "
                      "de alto impacto y alto esfuerzo se convierten en proyectos con "
                      "financiamiento propio. Las de bajo impacto y alto esfuerzo se descartan "
                      "explícitamente, y dejar constancia de ese descarte es parte del valor de "
                      "la herramienta.",
   supuestos=["Las estimaciones de impacto provienen del análisis estadístico, no de intuición."],
   limitaciones=["Es una priorización basada en juicio experto, no evidencia estadística.",
                 "Las estimaciones de impacto arrastran la incertidumbre del análisis que las "
                 "origina: conviene presentarlas como rango, no como cifra puntual."])

# --------------------------------------------------------- CONTROLAR ----
_f("cartas",
   titulo="Cartas de control",
   que_es="Gráfico de un estadístico del proceso en el tiempo con límites calculados a partir de "
          "su propia variabilidad. Distingue la variación por CAUSAS COMUNES (inherente al "
          "proceso, estable y predecible) de la variación por CAUSAS ESPECIALES (externa, "
          "identificable y corregible).",
   como_funciona="La línea central es el promedio del estadístico y los límites se situan a tres "
                 "desviaciones estándar del estadístico graficado. Un proceso estable produce "
                 "puntos dentro de los límites y sin patrones; cualquier señal indica que algo "
                 "cambio y debe investigarse.",
   formulas=[
       (r"LC \pm 3\hat{\sigma}_{estadístico}",
        "forma general de los límites de control"),
       (r"\bar{\bar{x}} \pm A_2 \bar{R}",
        "carta Xbar-R; A2 depende del tamaño de subgrupo n"),
       (r"\bar{\bar{x}} \pm A_3 \bar{S} \qquad LCS_S = B_4\bar{S},\ LCI_S = B_3\bar{S}",
        "carta Xbar-S, preferible con n >= 9"),
       (r"\bar{x} \pm 3\frac{\overline{MR}}{1.128}",
        "carta de individuales (I-MR), cuando n = 1"),
       (r"\bar{p} \pm 3\sqrt{\frac{\bar{p}(1-\bar{p})}{n_i}} \qquad "
        r"\bar{c} \pm 3\sqrt{\bar{c}} \qquad \bar{u} \pm 3\sqrt{\bar{u}/n_i}",
        "cartas de atributos: p (binomial), c y u (Poisson)"),
   ],
   procedimiento=[
       "Elegir la carta según el tipo de dato: continuo con subgrupos (Xbar-R, Xbar-S), continuo "
       "individual (I-MR), proporción de defectuosos (p, np) o conteo de defectos (c, u).",
       "Formar SUBGRUPOS RACIONALES: homogéneos por dentro, de modo que la variación interna "
       "refleje solo causas comunes.",
       "Evaluar PRIMERO la carta de dispersión (R o S): si la dispersión es inestable, los "
       "límites de la carta de medias se calculan con un Rbar contaminado.",
       "Aplicar las reglas de patrones no aleatorios.",
       "Investigar cada señal antes de recalcular límites.",
   ],
   como_se_interpreta="Un punto fuera de los límites, nueve puntos del mismo lado o seis en "
                      "tendencia indican que el proceso cambio: hay que buscar la causa, no "
                      "ajustar el proceso. Ajustar un proceso estable en respuesta a variación "
                      "comun lo empeora (es la trampa de la sobrecorrección). LOS LÍMITES DE "
                      "CONTROL NO SON LÍMITES DE ESPECIFICACIÓN: los primeros describen lo que el "
                      "proceso hace, los segundos lo que el cliente exige. Nunca se dibujan "
                      "especificaciones sobre una carta de medias, porque la media de subgrupos "
                      "dispersa menos que las observaciones individuales y la comparación sería "
                      "falsamente optimista.",
   supuestos=["Subgrupos racionales correctamente formados.",
              "Observaciones independientes; la autocorrelación produce falsas alarmas.",
              "Carta I-MR: sensible a la no normalidad, más que la carta Xbar.",
              "Carta p y np: modelo binomial (probabilidad constante, unidades independientes).",
              "Cartas c y u: modelo Poisson (media igual a varianza)."],
   limitaciones=["Las reglas de zona (5 a 8) suponen límites simetricos y aproximadamente "
                 "normales: no aplican a las cartas R, S, p, np, c ni u.",
                 "Activar las ocho reglas a la vez aumenta las falsas alarmas: la longitud media "
                 "de racha en control cae de ~370 a ~91 subgrupos.",
                 "Una carta detecta cambios, no explica su causa.",
                 "Con sobredispersión (varianza mayor que la del modelo binomial o Poisson) los "
                 "límites quedan demasiado estrechos."])

_f("plan_control",
   titulo="Plan de control",
   que_es="Documento que fija, para cada variable crítica, que se mide, como, con que frecuencia, "
          "quien es responsable, cuales son los límites y que acción correctiva se toma cuando se "
          "sale de ellos. Es el mecanismo que sostiene la mejora una vez que el equipo del "
          "proyecto se disuelve.",
   como_funciona="Traduce los hallazgos de las etapas anteriores en una rutina operacional "
                 "asignada a personas concretas, con criterios de reacción definidos de antemano.",
   formulas=[],
   procedimiento=[
       "Listar las variables críticas identificadas en Analizar y confirmadas en Mejorar.",
       "Definir método y frecuencia de medición coherentes con la velocidad del proceso.",
       "Asignar un responsable nominal por variable.",
       "Distinguir con claridad los límites de control de los de especificación.",
       "Escribir la acción correctiva ANTES de necesitarla, incluyendo a quien escalar.",
       "Definir la frecuencia de revisión del propio plan.",
   ],
   como_se_interpreta="Un plan de control sin responsable nominal o sin acción correctiva escrita "
                      "no es un plan: es una lista de deseos. La prueba de que funciona es que, "
                      "ante una señal, el operador sabe exactamente que hacer sin consultar.",
   supuestos=["El sistema de medición fue validado (Gage R&R) antes de incorporarse al plan."],
   limitaciones=["Un plan de control no mejora el proceso: solo evita que la mejora se pierda.",
                 "Si la frecuencia de medición es menor que la velocidad a la que cambia el "
                 "proceso, las señales llegan tarde."])


def obtener(clave: str) -> Ficha:
    if clave not in FICHAS:
        raise KeyError(f"Ficha desconocida: {clave}. Disponibles: {sorted(FICHAS)}")
    return FICHAS[clave]


# ------------------------------------------- CONTROLAR: fases y memoria --
_f("fases",
   titulo="Fase I y Fase II: estudio base y vigilancia",
   que_es="Las dos etapas en que se divide el trabajo con cartas de control. En la FASE I "
          "se reúne un periodo base, se calculan límites provisionales, se investigan y "
          "eliminan las causas especiales y se recalculan hasta obtener un periodo estable. "
          "En la FASE II esos límites se CONGELAN y los datos nuevos se vigilan contra ellos. "
          "Es la diferencia entre caracterizar un proceso y monitorearlo.",
   como_funciona="En Fase I los límites son una estimación que se depura: cada punto con causa "
                 "asignable identificada y corregida se excluye del cálculo (pero se sigue "
                 "graficando) y los límites se recalculan sin él. Cuando ya no quedan señales "
                 "inexplicadas, el periodo es la línea base. En Fase II no se recalcula nada: "
                 "los parámetros guardados (Xbarbar y Rbar, o pbar, o cbar) generan los límites "
                 "contra los que se comparan las observaciones nuevas.",
   formulas=[
       (r"\text{Fase I: } \bar{\bar{x}},\ \bar{R} \ \text{ estimados con los } k \text{ subgrupos base}",
        "parámetros que quedan guardados"),
       (r"\text{Fase II: } LC = \bar{\bar{x}},\quad LCS/LCI = \bar{\bar{x}} \pm A_2 \bar{R}",
        "límites fijos; los datos nuevos no los modifican"),
   ],
   procedimiento=[
       "Reunir un periodo base: la referencia habitual es de 20 a 25 subgrupos.",
       "Calcular límites provisionales y revisar PRIMERO la carta de dispersión.",
       "Investigar cada señal. Si se identifica una causa asignable y se corrige, excluir "
       "ese subgrupo del cálculo y recalcular.",
       "Si una señal no tiene causa identificable, NO excluirla: es parte del proceso.",
       "Cuando el periodo base quede estable, congelar los límites.",
       "Vigilar los datos nuevos contra esos límites, sin recalcularlos.",
       "Recalcular solo ante un cambio deliberado y documentado del proceso.",
   ],
   como_se_interpreta="El efecto es grande y poco intuitivo. Si un proceso se corre 3 sigma y se "
                      "recalculan los límites con los datos nuevos, la línea central se muda con "
                      "el proceso, los límites se ensanchan y la carta no señala NADA: el "
                      "corrimiento queda perfectamente oculto. Con los límites congelados, los "
                      "mismos datos producen señal en todos los subgrupos. Recalcular es el error "
                      "más frecuente y el más silencioso en el uso real del control estadístico.",
   supuestos=["El periodo base es representativo de la operación normal del proceso.",
              "Las causas asignables excluidas fueron efectivamente identificadas y corregidas; "
              "excluir un punto solo porque incomoda invalida todo el análisis."],
   limitaciones=["Con menos de 20 subgrupos base los límites tienen mucha incertidumbre y "
                 "producen falsas alarmas en Fase II.",
                 "Si el proceso cambia de forma legítima (nuevo equipo, nueva mezcla), los "
                 "límites congelados dejan de aplicar y hay que rehacer la Fase I.",
                 "Los límites de una carta no sirven para otra, ni para un tamaño de subgrupo "
                 "distinto: los factores de carta dependen de n."])

_f("memoria",
   titulo="Cartas con memoria: EWMA y CUSUM",
   que_es="Cartas diseñadas para detectar corrimientos PEQUEÑOS y sostenidos, del orden de "
          "1 sigma, que la carta de Shewhart detecta muy tarde o no detecta. Shewhart solo mira "
          "el punto actual; EWMA y CUSUM acumulan información de los puntos anteriores, y por "
          "eso notan antes una deriva lenta.",
   como_funciona="La EWMA promedia las observaciones dando más peso a las recientes y menos a "
                 "las antiguas, con decaimiento geométrico. El CUSUM acumula las desviaciones "
                 "respecto del objetivo descontando una holgura K, de modo que el ruido no se "
                 "acumula pero un sesgo sistemático sí.",
   formulas=[
       (r"z_i = \lambda x_i + (1-\lambda) z_{i-1}, \qquad z_0 = \mu_0",
        "EWMA; lambda entre 0.05 y 0.3. Menor lambda = más memoria"),
       (r"LC_i = \mu_0 \pm L\,\sigma\sqrt{\frac{\lambda}{2-\lambda}\left[1-(1-\lambda)^{2i}\right]}",
        "límites EWMA: se ensanchan al principio y luego convergen"),
       (r"C^+_i = \max\left(0,\ x_i - (\mu_0 + K) + C^+_{i-1}\right)",
        "CUSUM de lado superior; K = k·sigma es la holgura (k = 0.5 habitual)"),
       (r"C^-_i = \max\left(0,\ (\mu_0 - K) - x_i + C^-_{i-1}\right)",
        "CUSUM de lado inferior; se señala cuando C+ o C- superan H = h·sigma (h = 4 o 5)"),
   ],
   procedimiento=[
       "Establecer mu0 y sigma en una Fase I limpia. Si se estiman con datos ya corridos, "
       "ambas cartas quedan ciegas al corrimiento.",
       "Elegir lambda (EWMA) o k y h (CUSUM) según el tamaño del corrimiento que importa "
       "detectar: k = delta/2 está optimizado para un corrimiento de delta sigmas.",
       "Graficar y vigilar el cruce de los límites.",
       "En CUSUM, leer dónde la acumulación despegó de cero: ahí empezó el corrimiento.",
   ],
   como_se_interpreta="Ante un corrimiento de 1 sigma, y midiéndolo por simulación en esta misma "
                      "aplicación: la carta de Shewhart lo detecta en cerca del 66% de los casos "
                      "dentro de 60 observaciones, con una mediana de 17 observaciones de "
                      "retraso; la EWMA y el CUSUM lo detectan siempre, con medianas de 9 y 7 "
                      "observaciones. A cambio, Shewhart sigue siendo mejor ante un salto grande "
                      "y aislado. Lo razonable es usarlas juntas.",
   supuestos=["mu0 y sigma provienen de un periodo base estable.",
              "Observaciones independientes. La autocorrelación afecta más a estas cartas que a "
              "Shewhart, porque acumulan.",
              "Aproximadamente normales, aunque ambas son razonablemente robustas."],
   limitaciones=["Los puntos de ambas cartas están autocorrelacionados por construcción: NO se "
                 "aplican las reglas de rachas, solo el cruce de límites.",
                 "Reaccionan más lento que Shewhart ante un salto grande y aislado.",
                 "El CUSUM sin reinicio deja varios puntos seguidos sobre el umbral tras una "
                 "sola excursión: hay que contar excursiones, no puntos.",
                 "No dicen cuál es la causa, solo que el nivel cambió."])

_f("montecarlo",
   titulo="Simulación de Monte Carlo y análisis de sensibilidad",
   que_es="Método para propagar la incertidumbre de las entradas de un modelo hasta su "
          "resultado. En vez de una cifra puntual, entrega una distribución: con ella se "
          "responde qué tan probable es cada desenlace, que es lo que una decisión bajo riesgo "
          "necesita saber.",
   como_funciona="Cada entrada incierta se declara como una distribución. Se extrae una muestra "
                 "aleatoria de cada una, se evalúa el modelo con esa combinación y se repite "
                 "miles de veces. El conjunto de resultados aproxima la distribución de la "
                 "salida. El análisis de sensibilidad correlaciona cada entrada con la salida "
                 "para ver cuál incertidumbre domina.",
   formulas=[
       (r"Y = f(X_1, X_2, \dots, X_k), \qquad X_j \sim F_j",
        "el modelo se evalúa n veces con muestras de cada distribución"),
       (r"\hat{P}(Y > y_0) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[y_i > y_0]",
        "probabilidad de superar un umbral: fracción de simulaciones que lo hacen"),
       (r"EE(\bar{Y}) = \frac{s_Y}{\sqrt{n}}",
        "error de muestreo: baja como 1/sqrt(n), no como 1/n"),
       (r"\text{Contribución}_j = \frac{\rho_{s,j}^2}{\sum_m \rho_{s,m}^2} \times 100",
        "tornado: cuadrado de la correlación de rangos, normalizado"),
   ],
   procedimiento=[
       "Identificar qué entradas son inciertas y cuáles se conocen bien.",
       "Elegir una distribución para cada una. Con juicio experto, la triangular o la PERT "
       "(mínimo, más probable, máximo) suelen ser las más honestas.",
       "Fijar una semilla para que el resultado sea reproducible.",
       "Ejecutar y verificar la convergencia: si la media acumulada aún se mueve, subir n.",
       "Leer la distribución completa, no solo la media: percentiles, intervalo y probabilidad "
       "de cruzar el umbral que importa.",
       "Revisar el tornado para saber dónde conviene invertir en reducir incertidumbre.",
   ],
   como_se_interpreta="Un VAN medio de 420.000 no dice nada sobre el riesgo. Si la simulación "
                      "entrega un intervalo del 90% entre -80.000 y 910.000 con una probabilidad "
                      "del 18% de VAN negativo, la decisión cambia por completo. El tornado "
                      "indica además dónde actuar: si el beneficio anual explica el 86% de la "
                      "incertidumbre del VAN, un estudio piloto que acote ese beneficio vale "
                      "más que cualquier refinamiento del costo de inversión.",
   supuestos=["Las distribuciones declaradas representan razonablemente lo que se sabe.",
              "Las entradas son independientes salvo que se declare correlación.",
              "El modelo determinístico es correcto; Monte Carlo propaga incertidumbre, "
              "no corrige un modelo mal especificado."],
   limitaciones=["No crea información: con distribuciones mal elegidas, el resultado está igual "
                 "de mal pero con apariencia de rigor.",
                 "Ignorar la correlación entre entradas SUBESTIMA la dispersión de la salida.",
                 "Las colas del resultado dependen mucho de la forma supuesta para las entradas, "
                 "que es justamente la parte peor conocida.",
                 "El error de muestreo baja como 1/sqrt(n): cuadruplicar las simulaciones solo "
                 "reduce el error a la mitad.",
                 "El tornado mide la influencia de la INCERTIDUMBRE de cada entrada, no su "
                 "importancia física: una variable decisiva pero bien conocida aparece abajo."])
