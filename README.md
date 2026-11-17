# MINING SIX SIGMA LAB

**Laboratorio de herramientas estadísticas y mejora de procesos mineros.**

Aplicación web interactiva que recorre las cinco etapas DMAIC sobre cuatro casos de
operación minera, con todos los cálculos estadísticos implementados y verificados.

---

## Advertencia metodológica (léala antes de usar cualquier resultado)

Los conjuntos de datos de esta aplicación son **simulados**. Se generan a partir de un
modelo explícito cuyos parámetros están documentados en `data/generators.py` y se
muestran en la interfaz. En consecuencia:

- Los **estadísticos son reales**: media, varianza, Cp, Cpk, límites de control,
  coeficientes de regresión, componentes de varianza y p-valores se computan
  íntegramente sobre los datos presentes. No hay valores fijados a mano ni gráficos
  decorativos.
- Los **datos no son evidencia** sobre ninguna faena. Las magnitudes se eligieron para
  ser plausibles en una operación a rajo abierto, pero no provienen de mediciones de
  campo.
- Como el modelo generador es conocido, cada análisis **puede verificarse**: la
  regresión múltiple debe recuperar aproximadamente los coeficientes definidos en el
  generador. Varias pruebas automáticas comprueban precisamente eso.

Puede sustituir los datos simulados por los suyos con el cargador de CSV de la barra
lateral.

---

## Instalación y ejecución

Requiere **Python 3.11 o superior**.

```bash
git clone <URL-DEL-REPOSITORIO>
```

```bash
cd mining-six-sigma-lab && python -m venv .venv && source .venv/bin/activate
```

En Windows (PowerShell), active el entorno con `.venv\Scripts\Activate.ps1`.

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`.

### Verificar que los cálculos son correctos

```bash
pytest -q
```

Son 215 pruebas que contrastan los resultados contra tablas publicadas, contra SciPy y
statsmodels, y contra conjuntos de datos con parámetros conocidos.

---

## Estructura del proyecto

```
app.py                      Punto de entrada y navegación
pages/
  0_Inicio.py               Tablero de indicadores y contexto del caso
  1_Definir.py              Charter, SIPOC, VOC, CTQ, Pareto preliminar
  2_Medir.py                Descriptiva, capacidad, Gage R&R, muestreo
  3_Analizar.py             Correlación, regresión, pruebas de hipótesis, análisis causal
  4_Mejorar.py              DOE, simulación de escenarios, priorización
  5_Controlar.py            Cartas de control, indicadores, plan de control
components/
  tema.py                   Paleta validada, plantilla de Plotly, tarjetas KPI
  graficos.py               Figuras de Plotly (reciben resultados ya calculados)
  didactica.py              Catálogo de fichas: qué es, cómo funciona, cómo se interpreta
  interpretacion.py         Redacción de lecturas a partir de los valores calculados
  ui.py                     Componentes de interfaz reutilizables
statistics_engine/          Motor de cálculo, sin dependencia de la interfaz
  montecarlo.py             Propagación de incertidumbre, tornado y convergencia
  constants.py              d2, d3, c4 y factores A2, A3, B3, B4, D3, D4
  descriptive.py            Estadística descriptiva, percentiles, regla de Tukey
  capability.py             Cp, Cpk, Pp, Ppk, Cpm, Z, PPM, IC de Cpk
  control_charts.py         Xbar-R, Xbar-S, I-MR, p, np, c, u, EWMA, CUSUM,
                            reglas de Nelson y límites congelados de Fase I
  inference.py              Pruebas t, ANOVA, Tukey, Levene, normalidad
  regression.py             Correlación, MCO, VIF, Breusch-Pagan, Durbin-Watson, Cook
  doe.py                    Factoriales 2^k, efectos, interacciones, ANOVA, Lenth
  gagerr.py                 Gage R&R cruzado por ANOVA, ndc, criterios AIAG
  pareto.py                 Pareto y matriz causa-efecto
data/
  generators.py             Generadores simulados con modelo documentado
  cases.py                  Registro de los cuatro casos con metadatos DMAIC
utils/
  io.py                     Exportación a CSV y validación de archivos cargados
  plantillas.py             Plantillas CSV descargables por familia de carta
  estado.py                 Estado compartido entre páginas
tests/                      215 pruebas del motor estadístico
assets/                     Recursos estáticos
```

La carpeta del motor se llama `statistics_engine` y no `statistics` de forma
deliberada: un paquete llamado `statistics` en la raíz del proyecto ensombrecería el
módulo homónimo de la biblioteca estándar de Python, que SciPy y statsmodels importan.

---

## Casos de estudio

| Caso | Área | Problema | Variable crítica |
|---|---|---|---|
| 1 | Carguío y transporte | Variabilidad del tiempo de ciclo de camiones | `ciclo_total_min` (LSE 38 min) |
| 2 | Perforación y voladura | Desviación de pozos y variabilidad del P80 | `p80_fragmentacion_cm` (LSE 28 cm) |
| 3 | Planta concentradora | Variabilidad de la recuperación metalúrgica | `recuperacion_pct` (LIE 85 %) |
| 4 | Chancado | Variabilidad del producto y disponibilidad | `p80_producto_mm` (32–44 mm) |

Cada caso incluye contexto operacional, problema, Project Charter, SIPOC, VOC, CTQ,
diagrama de Ishikawa, cinco porqués, conjunto de datos simulado con variables de entrada
y salida, límites de especificación con su origen declarado, y la lista de herramientas
DMAIC recomendadas.

Los módulos Medir, Analizar y Controlar son **genéricos sobre el registro de casos**:
funcionan con cualquiera de los cuatro y con datos propios cargados desde CSV.

---

## Herramientas implementadas

**Definir** — Project Charter · SIPOC · Voz del cliente · CTQ · Pareto preliminar ·
Hoja de verificación · Comparación frecuencia frente a impacto

**Medir** — Media, mediana, moda, varianza, desviación estándar, coeficiente de
variación, cuartiles, percentiles, asimetría, curtosis · Histograma, densidad, cajas,
dispersión, Q-Q · Cp, Cpk, Pp, Ppk, Cpm, Z, PPM observado y esperado, IC de Cpk ·
Gage R&R cruzado por ANOVA con %Contribución, %Variación de estudio, ndc y P/T ·
Pruebas de normalidad (Shapiro-Wilk, Anderson-Darling, D'Agostino) · Cálculo de tamaño
de muestra

**Analizar** — Matriz de correlación de Pearson y de Spearman con p-valores ·
Dispersión con recta ajustada, R² y bandas de confianza y de predicción · Regresión
lineal simple y múltiple con coeficientes, IC, R², R² ajustado, análisis de residuos,
VIF, Breusch-Pagan, Durbin-Watson y distancia de Cook · Demostración del sesgo por
variable omitida · Predicción con intervalos · Prueba t de dos muestras (agrupada y
Welch), t pareada, ANOVA de un factor con Tukey HSD, eta² y omega² · Ishikawa ·
Cinco porqués · Matriz causa-efecto ponderada

**Mejorar** — Diseño factorial 2² y 2³ con matriz experimental aleatorizada, efectos
principales, interacciones, ANOVA con error puro y método de Lenth para diseños sin
réplicas · Gráficos de efectos principales, de interacción y Pareto de efectos ·
Simulador paramétrico de escenarios operacionales · Comparación antes/después con
prueba estadística · Matriz esfuerzo-impacto · Análisis costo-beneficio con VAN y
período de recuperación · **Monte Carlo sobre el VAN**: distribuciones fija, normal,
triangular, uniforme, lognormal y PERT, probabilidad de destruir valor, tornado de
sensibilidad, curva de convergencia y comparación de alternativas por riesgo

**Controlar** — **Fase I y Fase II con límites congelados** · **Exclusión de puntos con
causa asignable** y recálculo · Cartas Xbar-R, Xbar-S, I-MR, p, np, c y u con línea
central, límites calculados y las ocho reglas de Nelson seleccionables · **Cartas con
memoria EWMA y CUSUM** para corrimientos pequeños y sostenidos · Eje temporal con fechas
reales · Formación de subgrupos racionales estratificados · Indicadores de seguimiento
con desagregación por factor · Plan de control editable que cita las líneas base
congeladas · Criterios de reacción ante cada tipo de señal

Cada herramienta presenta tres secciones didácticas (**¿Qué es?**, **¿Cómo funciona?**
con fórmulas y procedimiento, **¿Cómo se interpreta?** aplicado al ejemplo minero), más
sus **supuestos** y **limitaciones**. El botón **Interpretar resultados** redacta la
lectura a partir de los valores realmente calculados sobre los datos vigentes.

---

## Criterios estadísticos que la aplicación hace cumplir

No son advertencias decorativas: están implementadas en el código y condicionan lo que
la interfaz muestra.

1. **Límites de control frente a límites de especificación.** Los primeros se calculan
   del proceso; los segundos los fija el cliente. La aplicación nunca dibuja
   especificaciones sobre una carta de medias y explica por qué en cada carta.
2. **Capacidad potencial frente a desempeño.** Cp y Cpk usan sigma dentro de subgrupo;
   Pp y Ppk usan sigma total. Cuando la razón Cp/Pp supera 1,33 la aplicación advierte
   que la variación entre subgrupos domina y el proceso no es estable.
3. **La capacidad exige estabilidad.** Antes de mostrar los índices se evalúa una carta
   de individuales y, si hay señales, se declara explícitamente que los índices no son
   predictivos.
4. **Correlación no es causalidad.** Se advierte en el módulo de análisis y se
   demuestra el sesgo por variable omitida comparando regresión simple y múltiple.
5. **Verificación de supuestos antes de cada prueba.** Shapiro-Wilk y Levene se
   ejecutan y se reportan antes de la prueba t y del ANOVA, y la elección entre la
   versión agrupada y la de Welch se decide con ese resultado.
6. **El p-valor no es la probabilidad de que H₀ sea cierta.** Cada prueba reporta
   además el intervalo de confianza y el tamaño del efecto, y la interpretación
   generada lo señala explícitamente.
7. **Variación dentro y entre subgrupos.** La formación de subgrupos racionales es
   configurable y la aplicación muestra una tabla comparativa de cómo cambia Cp según
   el criterio de estratificación elegido.
8. **Las reglas de zona no aplican a cartas asimétricas.** En R, S, p, np, c y u solo
   se evalúan las reglas 1 a 4, y se advierte que activar las ocho reglas eleva la tasa
   de falsas alarmas.
9. **Fase I y Fase II están separadas.** Los límites se estiman en un periodo base y se
   congelan; la aplicación impide pasar a vigilancia sin una línea base y muestra cuántas
   señales se ocultarían si se recalcularan los límites con los datos nuevos.
10. **Excluir un punto exige causa identificada.** Los puntos excluidos se siguen
    graficando, marcados, y la interpretación generada recuerda que excluir un punto
    porque incomoda invalida el análisis.
11. **Los límites congelados no son intercambiables.** Aplicar los de una carta a otra, o
    a un tamaño de subgrupo distinto, se rechaza con un error explícito.
12. **En EWMA y CUSUM no se aplican las reglas de rachas**, porque sus puntos están
    autocorrelacionados por construcción; y en CUSUM se cuentan excursiones, no puntos
    sobre el umbral.

---

## Decisiones de diseño de la visualización

- Un solo eje por gráfico. El diagrama de Pareto expresa las barras en porcentaje del
  total para que la línea acumulada comparta escala, en lugar de recurrir a un segundo
  eje.
- Paleta categórica de tres posiciones (`#2a78d6`, `#eb6834`, `#1baf7a`), validada para
  deficiencia de visión cromática sobre fondo claro con separación mínima ΔE 9,2 y
  ΔE 24,0 en visión normal, considerando todos los pares. Más de tres series se agrupan
  en «Otros».
- Los colores de estado (bueno, alerta, crítico) están reservados y nunca se usan como
  serie. Siempre van acompañados de icono y texto, nunca color solo.
- Los puntos fuera de control se marcan con color **y** símbolo distinto.
- Toda figura tiene su tabla de datos y su exportación a CSV.

---

## ¿Sirve para un proceso real o solo para aprender?

Para ambas cosas, pero conviene saber dónde está cada límite.

**Lo que puede aplicar a su operación hoy.** Descargue la plantilla CSV que corresponda a
su tipo de dato (están en la pestaña Controlar), reemplace las filas de ejemplo por sus
registros y súbala. Con eso funcionan sobre sus datos: la estadística descriptiva, el
análisis de capacidad, las cartas de variables con Fase I y Fase II, las cartas con
memoria, las cartas de atributos, el Gage R&R, la correlación, la regresión, las pruebas
de hipótesis y el ANOVA. Todos los resultados se exportan a CSV.

**Lo que conviene hacer bien para que sirva.** Establezca una línea base en Fase I con
20 a 25 subgrupos, investigue y explique cada señal antes de congelar los límites, y a
partir de ahí trabaje en Fase II. Si recalcula los límites con cada lote nuevo, un
proceso que se corre arrastra sus propios límites y el corrimiento queda oculto: la
aplicación le muestra explícitamente cuántas señales se perderían si lo hiciera.

**Lo que esta aplicación no es.** No tiene persistencia: al cerrar la pestaña se pierden
las líneas base congeladas y el plan de control editado, así que exporte lo que quiera
conservar. No se conecta a ningún sistema de planta ni actualiza nada en tiempo real.
No reemplaza un sistema de información de proceso: es una herramienta de análisis y de
aprendizaje con la que se puede trabajar sobre datos reales, no un historiador de datos.

## Limitaciones conocidas

- El Gage R&R y las cartas de atributos traen un ejemplo simulado propio, porque exigen
  diseños (cruzado balanceado y conteos, respectivamente) que no existen en los datos de
  proceso continuos. Con sus plantillas CSV sí operan sobre datos reales.
- No hay persistencia entre sesiones: las líneas base congeladas viven en la sesión del
  navegador. Expórtelas antes de cerrar.
- El simulador paramétrico detallado está construido sobre el modelo del Caso 1. Los
  demás casos disponen de la comparación antes/después de su escenario de mejora.
- El análisis de capacidad supone normalidad. Con datos sesgados, el PPM esperado puede
  errar por órdenes de magnitud; la aplicación lo advierte y muestra siempre el PPM
  observado, que es un conteo directo.
- Los diseños 2^k solo detectan efectos lineales entre los dos niveles ensayados. No
  detectan curvatura.
- El Monte Carlo del VAN supone independencia entre las entradas y un beneficio anual
  constante dentro de cada simulación. Ambos supuestos SUBESTIMAN la dispersión real del
  resultado, y la aplicación lo advierte en cada ejecución.

---

## Referencias

- Montgomery, D. C. (2013). *Introduction to Statistical Quality Control*, 7.ª ed.
  Wiley. Factores de carta, reglas de patrones y análisis de capacidad.
- Nelson, L. S. (1984). The Shewhart control chart: tests for special causes.
  *Journal of Quality Technology*, 16(4).
- AIAG (2010). *Measurement Systems Analysis (MSA)*, 4.ª ed. Criterios de Gage R&R.
- Lenth, R. V. (1989). Quick and easy analysis of unreplicated factorials.
  *Technometrics*, 31(4).
- Bissell, A. F. (1990). How reliable is your capability index?
  *Applied Statistics*, 39(3). Intervalo de confianza de Cpk.
- Roberts, S. W. (1959). Control chart tests based on geometric moving averages.
  *Technometrics*, 1(3). Carta EWMA.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1). CUSUM.
- Vose, D. (2008). *Risk Analysis: A Quantitative Guide*, 3.ª ed. Wiley.
  Distribución PERT y análisis de sensibilidad en simulación de Monte Carlo.
