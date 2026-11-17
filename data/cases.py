"""Registro de casos mineros: metadatos DMAIC + datos simulados.

Cada caso declara sus variables, sus roles analiticos y sus límites de
especificación, de modo que los módulos Medir, Analizar y Controlar funcionan
de forma generica sobre cualquier caso seleccionado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import generators as gen


@dataclass(frozen=True)
class Variable:
    nombre: str
    etiqueta: str
    unidad: str
    rol: str                      # respuesta | componente | predictor | factor | atributo
    descripcion: str
    lie: float | None = None      # limite inferior de ESPECIFICACION (del cliente)
    lse: float | None = None      # limite superior de ESPECIFICACION
    objetivo: float | None = None
    origen_especificacion: str = ""


@dataclass
class Caso:
    id: str
    nombre: str
    area: str
    contexto: str
    problema: str
    objetivo: str
    alcance_inicio: str
    alcance_fin: str
    fuera_de_alcance: str
    impacto: str
    beneficio_estimado: str
    indicadores: list[str]
    equipo: list[str]
    variables: list[Variable]
    respuesta: str
    predictores: list[str]
    componentes: list[str]
    factores: list[str]
    sipoc: dict
    voc: list[dict]
    ctq: list[dict]
    ishikawa: dict
    cinco_porques: dict
    herramientas: dict
    generador: Callable[..., pd.DataFrame]
    gage_etiqueta: str = "medicion"
    nota_datos: str = ""

    def datos(self, n: int | None = None, semilla: int = 2024, mejora: bool = False) -> pd.DataFrame:
        kw = {"semilla": semilla, "mejora": mejora}
        if n is not None:
            kw["n"] = n
        return self.generador(**kw)

    def var(self, nombre: str) -> Variable:
        for v in self.variables:
            if v.nombre == nombre:
                return v
        raise KeyError(nombre)

    @property
    def numericas(self) -> list[str]:
        return [v.nombre for v in self.variables if v.rol in ("respuesta", "componente", "predictor")]

    @property
    def categoricas(self) -> list[str]:
        return [v.nombre for v in self.variables if v.rol == "factor"]


# =====================================================================
# CASO 1
# =====================================================================
CASO_1 = Caso(
    id="caso1",
    nombre="Variabilidad del tiempo de ciclo de camiones",
    area="Carguio y transporte - Mina rajo abierto",
    contexto=(
        "Operación a rajo abierto con una flota de 6 camiones de extracción (CAEX) de "
        "220 t nominales, atendidos por palas hidráulicas en tres fases activas. El ciclo "
        "de transporte se compone de carguio, transporte cargado, descarga en chancador "
        "primario, retorno en vacio y espera en pala. El area de Planificación Mina reporta "
        "que el cumplimiento del plan de movimiento se ve afectado no tanto por un ciclo "
        "medio alto como por su DISPERSIÓN, que impide dimensionar la flota con confianza."
    ),
    problema=(
        "El tiempo de ciclo de los camiones presenta una variabilidad elevada "
        "(desviación estándar cercana a 7.7 min sobre una media de 33.5 min, CV ~ 23%). "
        "Esta dispersión obliga a mantener holgura en la asignación de equipos y degrada "
        "la productividad efectiva de la flota."
    ),
    objetivo=(
        "Reducir la desviación estándar del tiempo de ciclo en al menos 25% y la media "
        "en al menos 2 minutos en un plazo de 4 meses, sin aumentar el número de equipos."
    ),
    alcance_inicio="Posicionamiento del camion bajo la pala para iniciar el carguio.",
    alcance_fin="Retorno del camion al punto de encolamiento de la pala.",
    fuera_de_alcance=(
        "Mantenimiento mayor de la flota, disenio del rajo, secuenciamiento de fases y "
        "operación aguas abajo del chancador primario."
    ),
    impacto=(
        "Con 6 camiones operando 20 h/día, reducir 2 min el ciclo medio libera "
        "aproximadamente 1.800 t/día de capacidad de transporte sin inversión en equipos."
    ),
    beneficio_estimado="Entre 1.500 y 2.200 t/día de capacidad liberada (estimación a validar en la etapa Mejorar).",
    indicadores=["Tiempo de ciclo total (min)", "Productividad (t/h)",
                 "Tiempo de espera en pala (min)", "Utilización efectiva de la flota (%)"],
    equipo=["Champion: Gerente de Operaciones Mina", "Black Belt: Ingeniería de Procesos",
            "Green Belt: Jefe de Turno Mina", "Miembros: Despacho, Mantenimiento de Vias, Planificación"],
    variables=[
        Variable("ciclo_total_min", "Tiempo de ciclo total", "min", "respuesta",
                 "Suma de carguio, transporte cargado, descarga, retorno y espera en pala.",
                 lse=38.0, objetivo=30.0,
                 origen_especificacion="Meta de planificación mina: 38 min como ciclo máximo "
                 "compatible con el plan de movimiento. Es un límite UNILATERAL: un ciclo más "
                 "corto nunca es un defecto."),
        Variable("t_carguio_min", "Tiempo de carguio", "min", "componente",
                 "Desde el posicionamiento bajo la pala hasta el último pase de balde."),
        Variable("t_transporte_min", "Tiempo de transporte cargado", "min", "componente",
                 "Desde la salida de la pala hasta la llegada al punto de descarga."),
        Variable("t_descarga_min", "Tiempo de descarga", "min", "componente",
                 "Maniobra y vaciado en el chancador primario o botadero."),
        Variable("t_retorno_min", "Tiempo de retorno en vacio", "min", "componente",
                 "Desde la descarga hasta el regreso a la cola de la pala."),
        Variable("t_espera_pala_min", "Tiempo de espera en pala", "min", "componente",
                 "Tiempo en cola antes de iniciar el carguio. Es tiempo sin valor agregado."),
        Variable("distancia_km", "Distancia de transporte", "km", "predictor",
                 "Longitud de la ruta cargada desde la pala hasta el punto de descarga."),
        Variable("pendiente_pct", "Pendiente de la ruta", "%", "predictor",
                 "Pendiente media ponderada de la rampa de subida."),
        Variable("estado_via", "Estado de la via", "índice 1-5", "predictor",
                 "Índice de condición de la carpeta de rodado (5 = optimo)."),
        Variable("carga_ton", "Carga transportada", "t", "predictor",
                 "Tonelaje efectivo registrado por el sistema de pesaje a bordo."),
        Variable("productividad_tph", "Productividad instantanea", "t/h", "respuesta",
                 "Carga transportada dividida por la duración del ciclo.", lie=380.0,
                 origen_especificacion="Meta de productividad del plan: 380 t/h por camion."),
        Variable("turno", "Turno", "", "factor", "Turno de día o de noche."),
        Variable("camion", "Camion", "", "factor", "Identificador del equipo CAEX."),
        Variable("operador", "Operador", "", "factor", "Identificador del operador."),
        Variable("fase", "Fase del rajo", "", "factor", "Fase de explotación activa."),
    ],
    respuesta="ciclo_total_min",
    predictores=["distancia_km", "pendiente_pct", "estado_via", "carga_ton", "t_espera_pala_min"],
    componentes=["t_carguio_min", "t_transporte_min", "t_descarga_min", "t_retorno_min", "t_espera_pala_min"],
    factores=["turno", "camion", "operador", "fase"],
    sipoc={
        "Proveedores": ["Pala hidráulica / cargador frontal", "Mantenimiento de vias",
                        "Despacho (sistema de asignación)", "Mantenimiento mina", "Abastecimiento de combustible"],
        "Entradas": ["Material tronado en el banco", "Camion disponible y operativo",
                     "Ruta habilitada", "Asignación de despacho", "Operador acreditado"],
        "Proceso": ["Encolar en pala", "Posicionar camion", "Cargar (3-5 pases)",
                    "Transportar cargado", "Descargar en chancador", "Retornar en vacio"],
        "Salidas": ["Material entregado en chancador (t)", "Registro de ciclo (min)",
                    "Consumo de combustible", "Horas de operación del equipo"],
        "Clientes": ["Planta concentradora", "Planificación mina", "Gerencia de Operaciones"],
    },
    voc=[
        {"cliente": "Planta concentradora", "voz": "Necesitamos alimentación estable al chancador; "
         "las interrupciones nos obligan a bajar la carga del molino.",
         "necesidad": "Flujo de material predecible", "ctq": "Variabilidad del ciclo (sigma)"},
        {"cliente": "Planificación mina", "voz": "No puedo comprometer el plan mensual si el "
         "rendimiento por camion cambia 20% entre turnos.",
         "necesidad": "Rendimiento predecible por equipo", "ctq": "Productividad (t/h) y su dispersión"},
        {"cliente": "Gerencia de Operaciones", "voz": "Antes de aprobar un camion adicional quiero "
         "saber cuanta capacidad estamos perdiendo con la flota actual.",
         "necesidad": "Uso eficiente del activo existente", "ctq": "Tiempo de espera en pala"},
    ],
    ctq=[
        {"ctq": "Tiempo de ciclo total", "medida": "min por ciclo", "objetivo": "<= 30 min",
         "especificacion": "LSE = 38 min", "critico_porque": "Determina los viajes por turno y el cumplimiento del plan."},
        {"ctq": "Dispersión del ciclo", "medida": "desviación estándar (min)", "objetivo": "<= 5.8 min",
         "especificacion": "Reducción de 25% sobre la línea base", "critico_porque": "La dispersión obliga a holgura en la flota."},
        {"ctq": "Tiempo de espera en pala", "medida": "min por ciclo", "objetivo": "<= 2.0 min",
         "especificacion": "Sin especificación formal del cliente", "critico_porque": "Es tiempo sin valor agregado, 100% desperdicio."},
        {"ctq": "Productividad por camion", "medida": "t/h", "objetivo": ">= 420 t/h",
         "especificacion": "LIE = 380 t/h", "critico_porque": "Traduce el ciclo en capacidad de movimiento."},
    ],
    ishikawa={
        "Maquinaria": ["Potencia disponible del camion", "Desgaste de neumaticos",
                       "Capacidad del balde de la pala", "Disponibilidad mecánica de la pala"],
        "Metodo": ["Secuencia de asignación de despacho", "Posicionamiento del camion (simple/doble)",
                   "Politica de relevo de turno", "Reglas de prioridad en la cola"],
        "Mano de obra": ["Experiencia del operador", "Velocidad de conducción adoptada",
                         "Fatiga en turno de noche", "Calidad del relevo"],
        "Material": ["Granulometría del material tronado", "Humedad y adherencia en el balde",
                     "Densidad del material", "Presencia de sobretamanos"],
        "Medio ambiente": ["Visibilidad (noche, polvo, niebla)", "Lluvia y estado de la carpeta",
                           "Temperatura", "Congestion en rutas compartidas"],
        "Medicion": ["Exactitud del GPS de despacho", "Criterio de inicio y fin de ciclo",
                     "Calibración del pesaje a bordo", "Registros manuales de demora"],
    },
    cinco_porques={
        "problema": "El tiempo de ciclo de los camiones tiene una dispersión de 7.7 min.",
        "porques": [
            {"pregunta": "Por que el ciclo varía tanto?",
             "respuesta": "Porque el tiempo de espera en pala varía entre 0.3 y 11 min."},
            {"pregunta": "Por que la espera en pala varía tanto?",
             "respuesta": "Porque la asignación de camiones a palas no equilibra la carga entre fases."},
            {"pregunta": "Por que la asignación no equilibra la carga?",
             "respuesta": "Porque el despacho opera con reglas fijas por fase y no con el estado real de la cola."},
            {"pregunta": "Por que opera con reglas fijas?",
             "respuesta": "Porque el sistema de despacho no incorpora el tiempo de ciclo esperado por ruta."},
            {"pregunta": "Por que no lo incorpora?",
             "respuesta": "Porque los tiempos por ruta nunca se caracterizaron estadísticamente."},
        ],
        "causa_raiz": ("Ausencia de una caracterización estadística del ciclo por ruta que "
                       "alimente las reglas de asignación del despacho."),
        "advertencia": ("Los cinco porqués son una herramienta de razonamiento causal basada en "
                        "juicio experto. Producen una HIPÓTESIS que debe verificarse con datos, "
                        "no una conclusión. Distintos equipos llegan a cadenas distintas."),
    },
    herramientas={
        "Definir": ["Project Charter", "SIPOC", "VOC", "CTQ", "Pareto preliminar"],
        "Medir": ["Estadística descriptiva", "Histograma / Boxplot / Densidad",
                  "Capacidad (Cpk unilateral sobre LSE=38)", "Gage R&R del sistema de despacho",
                  "Hoja de verificación", "Muestreo"],
        "Analizar": ["Matriz de correlación", "Regresión múltiple", "ANOVA por turno y fase",
                     "Prueba t día vs noche", "Ishikawa", "Cinco porqués", "Matriz causa-efecto"],
        "Mejorar": ["DOE 2^2 (balde x posición)", "Simulación de escenarios",
                    "Matriz esfuerzo-impacto", "Costo-beneficio", "Comparación antes/después"],
        "Controlar": ["Carta Xbar-R por turno", "Carta I-MR", "Carta p de ciclos fuera de meta",
                      "Plan de control", "Indicadores de seguimiento"],
    },
    generador=gen.caso1_ciclo_camiones,
    gage_etiqueta="tiempo_ciclo_min",
    nota_datos=("300 ciclos simulados. El modelo generador está documentado en "
                "data/generators.py: la regresión múltiple debe recuperar aproximadamente "
                "esos coeficientes, lo que permite verificar que el análisis es correcto."),
)

# =====================================================================
# CASOS 2, 3 y 4
# =====================================================================
CASO_2 = Caso(
    id="caso2",
    nombre="Variabilidad en perforación y fragmentación",
    area="Perforación y voladura - Mina rajo abierto",
    contexto=("Malla de perforación de 12.5 m de profundidad media con equipos DTH y rotativos. "
              "La fragmentación resultante (P80) condiciona el rendimiento del chancado y la "
              "molienda aguas abajo."),
    problema=("El diametro de perforación se desvía del nominal de 165 mm a medida que avanza el "
              "desgaste de la broca, y la desviación de pozos degrada el P80, que presenta alta "
              "variabilidad entre disparos."),
    objetivo="Reducir la dispersión del P80 en 20% y mantener el diametro dentro de 165 +/- 4 mm.",
    alcance_inicio="Marcación de la malla de perforación.",
    alcance_fin="Medición de fragmentación P80 post voladura.",
    fuera_de_alcance="Diseño geomecanico del banco y selección del explosivo.",
    impacto="Un P80 menor y más estable reduce el consumo de energía en chancado y molienda.",
    beneficio_estimado="Reducción estimada de 3-5% en consumo específico de energía de molienda.",
    indicadores=["P80 (cm)", "Diametro de perforación (mm)", "Desviación de pozo (m)"],
    equipo=["Champion: Superintendente Mina", "Black Belt: Ingeniería de Perforación y Voladura",
            "Miembros: Operadores de perforadora, Topografía, Geotecnia"],
    variables=[
        Variable("p80_fragmentacion_cm", "P80 de fragmentación", "cm", "respuesta",
                 "Tamaño bajo el cual pasa el 80% del material tronado.", lse=28.0, objetivo=22.0,
                 origen_especificacion="Requisito del chancador primario: P80 <= 28 cm."),
        Variable("diametro_mm", "Diametro de perforación", "mm", "respuesta",
                 "Diametro efectivo del pozo perforado.", lie=161.0, lse=169.0, objetivo=165.0,
                 origen_especificacion="Diseño de voladura: 165 mm nominal, tolerancia +/- 4 mm."),
        Variable("desviacion_pozo_m", "Desviación del pozo", "m", "respuesta",
                 "Desviación del eje del pozo respecto de la vertical en el fondo.", lse=0.45,
                 origen_especificacion="Criterio geotécnico: máximo 0.45 m."),
        Variable("desgaste_broca_pct", "Desgaste de la broca", "%", "predictor", "Porcentaje de vida útil consumida."),
        Variable("profundidad_m", "Profundidad del pozo", "m", "predictor", "Profundidad perforada."),
        Variable("factor_carga_kg_m3", "Factor de carga", "kg/m3", "predictor", "Explosivo por volumen de roca."),
        Variable("espaciamiento_m", "Espaciamiento", "m", "predictor", "Distancia entre pozos de la malla."),
        Variable("banco", "Banco", "", "factor", "Banco de explotación."),
        Variable("equipo", "Equipo de perforación", "", "factor", "Perforadora utilizada."),
        Variable("turno", "Turno", "", "factor", "Turno de operación."),
    ],
    respuesta="p80_fragmentacion_cm",
    predictores=["desgaste_broca_pct", "profundidad_m", "factor_carga_kg_m3", "espaciamiento_m", "desviacion_pozo_m"],
    componentes=["diametro_mm", "desviacion_pozo_m", "p80_fragmentacion_cm"],
    factores=["banco", "equipo", "turno"],
    sipoc={
        "Proveedores": ["Topografia", "Abastecimiento de explosivos", "Mantenimiento de perforadoras"],
        "Entradas": ["Diseño de malla", "Broca y barras", "Agente de voladura", "Modelo geotécnico"],
        "Proceso": ["Marcar malla", "Perforar pozos", "Medir desviación", "Cargar explosivo", "Tronar", "Medir P80"],
        "Salidas": ["Material tronado", "Registro de P80", "Reporte de desviaciones"],
        "Clientes": ["Carguio y transporte", "Chancado", "Geotecnia"],
    },
    voc=[{"cliente": "Chancado", "voz": "Los sobretamanos nos detienen la línea.",
          "necesidad": "Fragmentación uniforme", "ctq": "P80 y cola gruesa"},
         {"cliente": "Carguio", "voz": "Con material mal tronado el balde no llena.",
          "necesidad": "Material suelto y uniforme", "ctq": "P80"}],
    ctq=[{"ctq": "P80 de fragmentación", "medida": "cm", "objetivo": "<= 22 cm",
          "especificacion": "LSE = 28 cm", "critico_porque": "Determina el rendimiento del chancado."},
         {"ctq": "Diametro de perforación", "medida": "mm", "objetivo": "165 mm",
          "especificacion": "161 - 169 mm", "critico_porque": "Afecta la carga explosiva por pozo."}],
    ishikawa={
        "Maquinaria": ["Desgaste de la broca", "Rigidez de la sarta", "Potencia de la perforadora"],
        "Metodo": ["Criterio de cambio de broca", "Velocidad de penetración", "Control de verticalidad"],
        "Mano de obra": ["Experiencia del operador", "Inspección de la broca"],
        "Material": ["Dureza de la roca", "Presencia de estructuras", "Humedad"],
        "Medio ambiente": ["Topografía del banco", "Agua en el pozo"],
        "Medicion": ["Método de medición de P80 (fotografico)", "Calibración del inclinometro"],
    },
    cinco_porques={
        "problema": "El P80 varía entre disparos más de lo tolerable.",
        "porques": [
            {"pregunta": "Por que varía el P80?", "respuesta": "Porque la distribución de energía explosiva no es uniforme."},
            {"pregunta": "Por que no es uniforme?", "respuesta": "Porque los pozos se desvían y cambia el burden efectivo."},
            {"pregunta": "Por que se desvían?", "respuesta": "Porque la broca desgastada reduce el control de verticalidad."},
            {"pregunta": "Por que se perfora con broca desgastada?", "respuesta": "Porque el cambio se decide por horas y no por condición."},
            {"pregunta": "Por que se decide por horas?", "respuesta": "Porque no se mide el efecto del desgaste sobre la desviación."},
        ],
        "causa_raiz": "Politica de cambio de broca basada en tiempo y no en condición medida.",
        "advertencia": "Hipótesis a verificar con datos, no conclusión.",
    },
    herramientas={
        "Definir": ["Project Charter", "SIPOC", "VOC", "CTQ"],
        "Medir": ["Descriptiva", "Capacidad bilateral del diametro", "Capacidad unilateral del P80"],
        "Analizar": ["Correlacion", "Regresión múltiple", "ANOVA por equipo"],
        "Mejorar": ["DOE", "Simulación de escenarios", "Costo-beneficio"],
        "Controlar": ["Carta Xbar-S del diametro", "Carta I-MR del P80", "Plan de control"],
    },
    generador=gen.caso2_perforacion_voladura,
    gage_etiqueta="diametro_mm",
    nota_datos="240 pozos simulados con modelo documentado en data/generators.py.",
)

CASO_3 = Caso(
    id="caso3",
    nombre="Variabilidad de la recuperación metalúrgica",
    area="Planta concentradora - Flotación",
    contexto=("Planta de flotación de cobre con dos líneas paralelas. La recuperación "
              "metalúrgica depende de la ley de alimentación, la granulometría de molienda, "
              "el pH y la dosificación de reactivos."),
    problema=("La recuperación metalúrgica presenta una dispersión que impide cumplir de forma "
              "consistente la meta de 85%, y la ley de concentrado oscila fuera del rango "
              "comercial acordado."),
    objetivo="Elevar la recuperación media por encima de 85% y reducir su desviación estándar en 20%.",
    alcance_inicio="Alimentación al molino SAG.",
    alcance_fin="Concentrado final espesado.",
    fuera_de_alcance="Chancado primario y transporte de concentrado.",
    impacto="Un punto porcentual de recuperación representa varios millones de USD al año.",
    beneficio_estimado="1.0 - 1.5 puntos de recuperación.",
    indicadores=["Recuperación (%)", "Ley de concentrado (%)", "P80 de molienda (um)"],
    equipo=["Champion: Gerente de Planta", "Black Belt: Metalurgia", "Miembros: Operadores de flotación, Laboratorio"],
    variables=[
        Variable("recuperacion_pct", "Recuperación metalúrgica", "%", "respuesta",
                 "Porcentaje de cobre recuperado en el concentrado.", lie=85.0, objetivo=88.0,
                 origen_especificacion="Meta del plan metalúrgico: mínimo 85%. Límite UNILATERAL inferior."),
        Variable("ley_concentrado_pct", "Ley de concentrado", "%", "respuesta",
                 "Contenido de cobre en el concentrado final.", lie=24.0, lse=30.0, objetivo=27.0,
                 origen_especificacion="Contrato de venta: 24-30% Cu; fuera de rango hay penalización."),
        Variable("ley_alimentacion_pct", "Ley de alimentación", "%", "predictor", "Cobre en la alimentación."),
        Variable("p80_molienda_um", "P80 de molienda", "um", "predictor", "Granulometría de alimentación a flotación."),
        Variable("ph_flotacion", "pH de flotación", "", "predictor", "pH en el circuito rougher."),
        Variable("dosis_colector_g_t", "Dosis de colector", "g/t", "predictor", "Reactivo colector dosificado."),
        Variable("solidos_pct", "Porcentaje de solidos", "%", "predictor", "Solidos en la pulpa de alimentación."),
        Variable("turno", "Turno", "", "factor", "Turno A, B o C."),
        Variable("linea", "Linea", "", "factor", "Línea de flotación."),
    ],
    respuesta="recuperacion_pct",
    predictores=["ley_alimentacion_pct", "p80_molienda_um", "ph_flotacion", "dosis_colector_g_t", "solidos_pct"],
    componentes=["recuperacion_pct", "ley_concentrado_pct"],
    factores=["turno", "linea"],
    sipoc={
        "Proveedores": ["Mina", "Chancado", "Abastecimiento de reactivos", "Laboratorio químico"],
        "Entradas": ["Mineral chancado", "Agua de proceso", "Reactivos (colector, espumante, cal)", "Energia"],
        "Proceso": ["Moler", "Clasificar", "Acondicionar", "Flotar rougher", "Limpiar", "Espesar"],
        "Salidas": ["Concentrado de cobre", "Relave", "Reporte metalúrgico"],
        "Clientes": ["Fundición / cliente comercial", "Gerencia de Operaciones", "Medio ambiente (relaves)"],
    },
    voc=[{"cliente": "Cliente comercial", "voz": "Las leyes fuera de rango activan penalizaciones.",
          "necesidad": "Concentrado dentro de especificación", "ctq": "Ley de concentrado"},
         {"cliente": "Gerencia", "voz": "Necesitamos recuperación estable para proyectar ingresos.",
          "necesidad": "Recuperación predecible", "ctq": "Recuperación y su dispersión"}],
    ctq=[{"ctq": "Recuperación metalúrgica", "medida": "%", "objetivo": ">= 88%",
          "especificacion": "LIE = 85%", "critico_porque": "Impacto económico directo."},
         {"ctq": "Ley de concentrado", "medida": "%", "objetivo": "27%",
          "especificacion": "24 - 30%", "critico_porque": "Penalizaciones comerciales."}],
    ishikawa={
        "Maquinaria": ["Desgaste de revestimientos del molino", "Estado de rotores de celdas", "Bombas de pulpa"],
        "Metodo": ["Estrategia de dosificación", "Control de nivel de espuma", "Tiempo de residencia"],
        "Mano de obra": ["Criterio del operador para ajustar reactivos", "Turno"],
        "Material": ["Mineralogia", "Ley de alimentación", "Presencia de arcillas", "Oxidacion"],
        "Medio ambiente": ["Temperatura del agua", "Calidad del agua recirculada"],
        "Medicion": ["Exactitud del analizador en línea", "Muestreo del relave", "Laboratorio"],
    },
    cinco_porques={
        "problema": "La recuperación no alcanza la meta de 85% de forma consistente.",
        "porques": [
            {"pregunta": "Por que no alcanza la meta?", "respuesta": "Porque la granulometría de molienda varía."},
            {"pregunta": "Por que varía la granulometría?", "respuesta": "Porque la dureza de la alimentación cambia entre fases."},
            {"pregunta": "Por que no se compensa?", "respuesta": "Porque el control de molienda no usa información de dureza."},
            {"pregunta": "Por que no la usa?", "respuesta": "Porque la dureza no se mide en línea."},
            {"pregunta": "Por que no se mide?", "respuesta": "Porque no se ha caracterizado su efecto sobre la recuperación."},
        ],
        "causa_raiz": "Control de molienda sin realimentación de la dureza de alimentación.",
        "advertencia": "Hipótesis a verificar con datos.",
    },
    herramientas={
        "Definir": ["Project Charter", "SIPOC", "VOC", "CTQ"],
        "Medir": ["Descriptiva", "Capacidad unilateral (recuperación)", "Capacidad bilateral (ley)"],
        "Analizar": ["Correlacion", "Regresión múltiple", "ANOVA por turno y línea"],
        "Mejorar": ["DOE 2^3 (pH x colector x P80)", "Simulacion", "Costo-beneficio"],
        "Controlar": ["Carta Xbar-R", "Carta I-MR", "Plan de control"],
    },
    generador=gen.caso3_planta_concentradora,
    gage_etiqueta="ley_concentrado_pct",
    nota_datos="260 muestras simuladas con modelo documentado en data/generators.py.",
)

CASO_4 = Caso(
    id="caso4",
    nombre="Variabilidad del producto y disponibilidad en chancado",
    area="Chancado - Planta",
    contexto=("Circuito de chancado primario y secundario que alimenta al acopio. El tamaño del "
              "producto (P80) depende del setting (CSS), el desgaste de revestimientos y la "
              "dureza del mineral."),
    problema=("El P80 del producto de chancado presenta variabilidad que desestabiliza la molienda, "
              "y la disponibilidad operacional cae por debajo de la meta por paradas no programadas."),
    objetivo="Reducir la dispersión del P80 en 25% y elevar la disponibilidad por encima de 92%.",
    alcance_inicio="Descarga de camiones en el chancador primario.",
    alcance_fin="Entrega de producto al acopio.",
    fuera_de_alcance="Molienda y flotación.",
    impacto="Un P80 estable permite subir la carga del molino SAG sin riesgo de sobrecarga.",
    beneficio_estimado="2-4% de aumento en el throughput de molienda.",
    indicadores=["P80 del producto (mm)", "Disponibilidad (%)", "Consumo de energía (kWh/t)"],
    equipo=["Champion: Superintendente de Planta", "Black Belt: Confiabilidad", "Miembros: Mantenimiento, Operaciones"],
    variables=[
        Variable("p80_producto_mm", "P80 del producto", "mm", "respuesta",
                 "Tamaño bajo el cual pasa el 80% del producto de chancado.",
                 lie=32.0, lse=44.0, objetivo=38.0,
                 origen_especificacion="Ventana de alimentación del molino SAG: 32-44 mm."),
        Variable("disponibilidad_pct", "Disponibilidad operacional", "%", "respuesta",
                 "Porcentaje del tiempo de turno con el equipo disponible.", lie=92.0, objetivo=96.0,
                 origen_especificacion="Meta de confiabilidad: mínimo 92%."),
        Variable("consumo_energia_kwh_t", "Consumo específico de energía", "kWh/t", "respuesta",
                 "Energía consumida por tonelada procesada.", lse=1.15,
                 origen_especificacion="Presupuesto energético: máximo 1.15 kWh/t."),
        Variable("css_mm", "CSS (setting)", "mm", "predictor", "Closed Side Setting del chancador."),
        Variable("desgaste_revestimiento_pct", "Desgaste de revestimiento", "%", "predictor", "Vida útil consumida."),
        Variable("humedad_pct", "Humedad del mineral", "%", "predictor", "Contenido de humedad."),
        Variable("alimentacion_tph", "Tasa de alimentación", "t/h", "predictor", "Tonelaje horario alimentado."),
        Variable("dureza_indice", "Índice de dureza", "", "predictor", "Índice de trabajo del mineral."),
        Variable("horas_paro", "Horas de paro", "h", "componente", "Horas de detención en el turno."),
        Variable("turno", "Turno", "", "factor", "Turno de día o de noche."),
        Variable("chancador", "Chancador", "", "factor", "Equipo primario o secundario."),
    ],
    respuesta="p80_producto_mm",
    predictores=["css_mm", "desgaste_revestimiento_pct", "humedad_pct", "alimentacion_tph", "dureza_indice"],
    componentes=["p80_producto_mm", "disponibilidad_pct", "consumo_energia_kwh_t"],
    factores=["turno", "chancador"],
    sipoc={
        "Proveedores": ["Carguio y transporte", "Mantenimiento", "Energia"],
        "Entradas": ["Mineral tronado", "Revestimientos", "Energía electrica", "Plan de mantenimiento"],
        "Proceso": ["Recibir", "Chancar primario", "Harnear", "Chancar secundario", "Transportar a acopio"],
        "Salidas": ["Producto chancado", "Registro de disponibilidad", "Consumo energético"],
        "Clientes": ["Molienda", "Planificación de planta"],
    },
    voc=[{"cliente": "Molienda", "voz": "Si el P80 sube, el SAG se sobrecarga.",
          "necesidad": "Producto dentro de ventana", "ctq": "P80 del producto"},
         {"cliente": "Planificacion", "voz": "Las paradas no programadas rompen el plan.",
          "necesidad": "Disponibilidad predecible", "ctq": "Disponibilidad"}],
    ctq=[{"ctq": "P80 del producto", "medida": "mm", "objetivo": "38 mm",
          "especificacion": "32 - 44 mm", "critico_porque": "Define la carga admisible del SAG."},
         {"ctq": "Disponibilidad", "medida": "%", "objetivo": ">= 96%",
          "especificacion": "LIE = 92%", "critico_porque": "Determina el throughput del circuito."}],
    ishikawa={
        "Maquinaria": ["Desgaste de revestimientos", "Estado del sistema hidraulico", "Correas transportadoras"],
        "Metodo": ["Criterio de ajuste del CSS", "Plan de cambio de revestimientos", "Control de nivel de tolva"],
        "Mano de obra": ["Criterio del operador", "Tiempo de respuesta a atollos"],
        "Material": ["Dureza", "Humedad y arcillas", "Sobretamanos de la voladura"],
        "Medio ambiente": ["Lluvia", "Polvo"],
        "Medicion": ["Medición del CSS", "Analizador granulometrico en correa"],
    },
    cinco_porques={
        "problema": "El P80 del producto de chancado es inestable.",
        "porques": [
            {"pregunta": "Por que es inestable?", "respuesta": "Porque el CSS efectivo cambia durante la campaña."},
            {"pregunta": "Por que cambia el CSS?", "respuesta": "Porque el revestimiento se desgasta."},
            {"pregunta": "Por que no se compensa el desgaste?", "respuesta": "Porque el ajuste se hace en frecuencia fija."},
            {"pregunta": "Por que en frecuencia fija?", "respuesta": "Porque no hay medición continua del P80."},
            {"pregunta": "Por que no hay medición continua?", "respuesta": "Porque no se cuantifico la perdida asociada."},
        ],
        "causa_raiz": "Ajuste del CSS desacoplado del desgaste real del revestimiento.",
        "advertencia": "Hipótesis a verificar con datos.",
    },
    herramientas={
        "Definir": ["Project Charter", "SIPOC", "VOC", "CTQ"],
        "Medir": ["Descriptiva", "Capacidad bilateral del P80", "Capacidad unilateral de disponibilidad"],
        "Analizar": ["Correlacion", "Regresión múltiple", "ANOVA por chancador"],
        "Mejorar": ["DOE", "Simulacion", "Matriz esfuerzo-impacto"],
        "Controlar": ["Carta I-MR", "Carta u de atollos", "Plan de control"],
    },
    generador=gen.caso4_chancado,
    gage_etiqueta="p80_producto_mm",
    nota_datos="280 turnos simulados con modelo documentado en data/generators.py.",
)

CASOS: dict[str, Caso] = {c.id: c for c in (CASO_1, CASO_2, CASO_3, CASO_4)}
NOMBRES = {c.id: f"Caso {i+1}. {c.nombre}" for i, c in enumerate(CASOS.values())}


def obtener(case_id: str) -> Caso:
    if case_id not in CASOS:
        raise KeyError(f"Caso desconocido: {case_id}. Disponibles: {list(CASOS)}")
    return CASOS[case_id]
