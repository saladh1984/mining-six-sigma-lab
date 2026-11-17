"""Generadores de datos sinteticos para los casos mineros.

ADVERTENCIA METODOLÓGICA (leala antes de usar cualquier resultado)
-------------------------------------------------------------------
Los datos de esta aplicación son SIMULADOS. Se generan a partir de un modelo
explicito cuyos parámetros están escritos en este archivo y se muestran en la
interfaz. Por lo tanto:

  * Los ESTADÍSTICOS que calcula la aplicación son reales: se computan sobre
    los datos, sin valores fijados a mano.
  * Los DATOS no son evidencia sobre ninguna faena. Las magnitudes se eligieron
    para ser plausibles en una operación a rajo abierto, pero no provienen de
    mediciones de campo.
  * Como el modelo generador es conocido, cada análisis puede verificarse: la
    regresión debe recuperar aproximadamente los coeficientes aquí definidos.
    Esa es precisamente la utilidad didáctica de simular.

Toda generación usa numpy.random.default_rng(semilla): el mismo número produce
exactamente el mismo conjunto de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ====================================================================
# CASO 1 - CARGUIO Y TRANSPORTE: tiempo de ciclo de camiones
# ====================================================================
# Modelo generador (minutos), "verdad" que los analisis deben recuperar:
#   t_carguio   = 2.60 + 0.0180*carga_ton + 0.45*(turno=Noche) + e,  e~N(0,0.55)
#   v_cargado   = 22.0 - 0.95*pendiente_pct - 0.80*(5-estado_via) km/h
#   t_transporte= 60*distancia_km / v_cargado + e
#   t_descarga  = 1.35 + 0.30*(turno=Noche) + e,  e~N(0,0.25)
#   v_vacio     = 32.0 - 0.70*pendiente_pct - 0.60*(5-estado_via) km/h
#   t_retorno   = 60*distancia_km / v_vacio + e
#   t_espera_pala ~ Gamma(forma=2.0, escala=1.5) + 0.8*(flota saturada)
#   ciclo_total = suma de los cinco componentes
CASO1_PARAMS = {
    "carguio_base": 2.60, "carguio_por_ton": 0.0180, "carguio_noche": 0.45, "carguio_sd": 0.55,
    "v_cargado_base": 22.0, "v_cargado_pendiente": -0.95, "v_cargado_via": -0.80,
    "v_vacio_base": 32.0, "v_vacio_pendiente": -0.70, "v_vacio_via": -0.60,
    "descarga_base": 1.35, "descarga_noche": 0.30, "descarga_sd": 0.25,
    "espera_forma": 2.0, "espera_escala": 1.50,
    "transporte_sd": 0.80, "retorno_sd": 0.70,
}


def caso1_ciclo_camiones(n: int = 300, semilla: int = 2024,
                         mejora: bool = False, params: dict | None = None) -> pd.DataFrame:
    """Ciclos de camiones de extracción (CAEX) en un rajo abierto.

    mejora=True aplica la intervención evaluada en la etapa Mejorar:
    despacho asistido (reduce la espera en pala) y mantenimiento de vias
    (mejora el estado de la via). Sirve para la comparación antes/después.
    """
    p = {**CASO1_PARAMS, **(params or {})}
    rng = np.random.default_rng(semilla + (1000 if mejora else 0))

    turno = rng.choice(["Dia", "Noche"], n, p=[0.52, 0.48])
    camion = rng.choice([f"CAEX-{i:02d}" for i in range(1, 7)], n)
    operador = rng.choice([f"OP-{i}" for i in range(1, 9)], n)
    fase = rng.choice(["Fase 3", "Fase 4", "Fase 5"], n, p=[0.35, 0.40, 0.25])

    # La distancia depende de la fase del rajo: fases mas profundas, rutas mas largas
    base_dist = {"Fase 3": 2.3, "Fase 4": 3.1, "Fase 5": 4.2}
    distancia = np.array([rng.normal(base_dist[f], 0.35) for f in fase]).clip(1.2, 6.5)
    # La pendiente tambien depende de la fase: confusion realista con la distancia
    base_pend = {"Fase 3": 6.5, "Fase 4": 7.8, "Fase 5": 9.0}
    pendiente = np.array([rng.normal(base_pend[f], 1.1) for f in fase]).clip(3.0, 12.0)

    estado_via = rng.choice([2, 3, 4, 5], n, p=([0.25, 0.35, 0.28, 0.12] if not mejora
                                                else [0.06, 0.22, 0.42, 0.30]))
    carga = rng.normal(218, 11, n).clip(170, 245)

    es_noche = (turno == "Noche").astype(float)
    deficit_via = 5 - estado_via

    t_carguio = (p["carguio_base"] + p["carguio_por_ton"] * carga
                 + p["carguio_noche"] * es_noche + rng.normal(0, p["carguio_sd"], n))

    v_car = (p["v_cargado_base"] + p["v_cargado_pendiente"] * pendiente
             + p["v_cargado_via"] * deficit_via).clip(6, None)
    t_transporte = 60 * distancia / v_car + rng.normal(0, p["transporte_sd"], n)

    t_descarga = (p["descarga_base"] + p["descarga_noche"] * es_noche
                  + rng.normal(0, p["descarga_sd"], n))

    v_vac = (p["v_vacio_base"] + p["v_vacio_pendiente"] * pendiente
             + p["v_vacio_via"] * deficit_via).clip(8, None)
    t_retorno = 60 * distancia / v_vac + rng.normal(0, p["retorno_sd"], n)

    escala = p["espera_escala"] * (0.45 if mejora else 1.0)
    t_espera = rng.gamma(p["espera_forma"], escala, n)
    # Congestion: en el turno de dia operan mas camiones por pala
    t_espera = t_espera + np.where(turno == "Dia", rng.gamma(1.4, 0.55 * (0.5 if mejora else 1.0), n), 0.0)

    comp = np.column_stack([t_carguio, t_transporte, t_descarga, t_retorno, t_espera]).clip(0.05, None)
    df = pd.DataFrame({
        "ciclo": np.arange(1, n + 1),
        "turno": turno, "camion": camion, "operador": operador, "fase": fase,
        "distancia_km": distancia.round(3),
        "pendiente_pct": pendiente.round(2),
        "estado_via": estado_via,
        "carga_ton": carga.round(1),
        "t_carguio_min": comp[:, 0].round(3),
        "t_transporte_min": comp[:, 1].round(3),
        "t_descarga_min": comp[:, 2].round(3),
        "t_retorno_min": comp[:, 3].round(3),
        "t_espera_pala_min": comp[:, 4].round(3),
    })
    df["ciclo_total_min"] = comp.sum(axis=1).round(3)
    df["productividad_tph"] = (df["carga_ton"] / (df["ciclo_total_min"] / 60)).round(2)
    df["escenario"] = "Después de la mejora" if mejora else "Línea base"
    return df


# ====================================================================
# CASO 2 - PERFORACION Y VOLADURA
# ====================================================================
# diametro_mm  ~ N(165 + desgaste_broca*0.45, 1.4)      especificacion 165 +/- 4 mm
# desviacion_m ~ |N(0, 0.09 + 0.012*profundidad)|       maximo admisible 0.45 m
# p80_cm       = 32 - 1.9*factor_carga + 0.8*espaciamiento + 2.2*desviacion + e
def caso2_perforacion_voladura(n: int = 240, semilla: int = 2024, mejora: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(semilla + 7 + (1000 if mejora else 0))
    banco = rng.choice(["B-3080", "B-3065", "B-3050"], n)
    equipo = rng.choice(["DTH-01", "DTH-02", "ROT-01"], n)
    turno = rng.choice(["Dia", "Noche"], n)
    desgaste = rng.uniform(0, 100, n) * (0.55 if mejora else 1.0)
    profundidad = rng.normal(12.5, 0.9, n).clip(9, 16)
    factor_carga = rng.normal(0.72 if not mejora else 0.80, 0.07, n).clip(0.45, 1.05)
    espaciamiento = rng.normal(6.2, 0.35, n).clip(5.0, 7.5)

    diametro = 165 + 0.0045 * desgaste * 10 + rng.normal(0, 1.4 if not mejora else 0.95, n)
    desviacion = np.abs(rng.normal(0, 0.09 + 0.012 * profundidad)) * (0.7 if mejora else 1.0)
    p80 = (32 - 1.9 * (factor_carga * 10) / 10 * 10 + 0.8 * espaciamiento
           + 2.2 * desviacion * 10 + rng.normal(0, 1.6, n))
    p80 = (p80 - p80.mean() + (24.5 if not mejora else 20.5)).clip(8, None)

    return pd.DataFrame({
        "pozo": np.arange(1, n + 1), "banco": banco, "equipo": equipo, "turno": turno,
        "desgaste_broca_pct": desgaste.round(1),
        "profundidad_m": profundidad.round(2),
        "factor_carga_kg_m3": factor_carga.round(3),
        "espaciamiento_m": espaciamiento.round(2),
        "diametro_mm": diametro.round(2),
        "desviacion_pozo_m": desviacion.round(3),
        "p80_fragmentacion_cm": p80.round(2),
        "escenario": "Después de la mejora" if mejora else "Línea base",
    })


# ====================================================================
# CASO 3 - PLANTA CONCENTRADORA
# ====================================================================
# recuperacion_% = 68 + 7.5*ley_alim - 0.28*(p80_molienda-180)/10 + 1.4*(pH-10.5)
#                  + 0.9*dosis_colector - 0.045*(solidos-32)^2 + e
def caso3_planta_concentradora(n: int = 260, semilla: int = 2024, mejora: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(semilla + 13 + (1000 if mejora else 0))
    turno = rng.choice(["A", "B", "C"], n)
    linea = rng.choice(["Línea 1", "Línea 2"], n)
    ley_alim = rng.normal(0.68, 0.11, n).clip(0.30, 1.10)
    p80_mol = rng.normal(180 if not mejora else 168, 14 if not mejora else 9, n).clip(130, 235)
    ph = rng.normal(10.5, 0.35 if not mejora else 0.18, n).clip(9.3, 11.8)
    colector = rng.normal(28, 3.5, n).clip(18, 40)
    solidos = rng.normal(32, 2.2, n).clip(25, 40)

    rec = (68 + 7.5 * ley_alim - 0.28 * (p80_mol - 180) / 10 + 1.4 * (ph - 10.5)
           + 0.9 * (colector - 28) / 10 - 0.045 * (solidos - 32) ** 2
           + rng.normal(0, 1.6, n)).clip(55, 96)
    ley_conc = (26 + 4.2 * (ley_alim - 0.68) + 0.045 * (rec - 85)
                + rng.normal(0, 0.95, n)).clip(18, 34)
    return pd.DataFrame({
        "muestra": np.arange(1, n + 1), "turno": turno, "linea": linea,
        "ley_alimentacion_pct": ley_alim.round(3),
        "p80_molienda_um": p80_mol.round(1),
        "ph_flotacion": ph.round(2),
        "dosis_colector_g_t": colector.round(1),
        "solidos_pct": solidos.round(2),
        "recuperacion_pct": rec.round(2),
        "ley_concentrado_pct": ley_conc.round(2),
        "escenario": "Después de la mejora" if mejora else "Línea base",
    })


# ====================================================================
# CASO 4 - CHANCADO
# ====================================================================
def caso4_chancado(n: int = 280, semilla: int = 2024, mejora: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(semilla + 21 + (1000 if mejora else 0))
    turno = rng.choice(["Dia", "Noche"], n)
    chancador = rng.choice(["CH-Primario", "CH-Secundario"], n)
    css = rng.normal(38, 2.4 if not mejora else 1.3, n).clip(30, 48)
    desgaste = rng.uniform(0, 100, n) * (0.6 if mejora else 1.0)
    humedad = rng.normal(3.2, 0.9, n).clip(0.8, 6.5)
    alimentacion = rng.normal(1850, 160, n).clip(1300, 2300)
    dureza = rng.normal(14.5, 1.8, n).clip(9, 20)

    p80 = (0.92 * css + 0.035 * desgaste + 0.45 * dureza - 6.0
           + rng.normal(0, 1.5 if not mejora else 1.0, n)).clip(15, None)
    energia = (0.62 + 0.018 * dureza + 0.0009 * alimentacion / 10
               + 0.004 * humedad + rng.normal(0, 0.05, n)).clip(0.4, None)
    horas_paro = np.where(rng.random(n) < (0.18 if not mejora else 0.09),
                          rng.gamma(2.0, 0.9, n), 0.0)
    disponibilidad = (100 * (12 - horas_paro) / 12).clip(0, 100)
    return pd.DataFrame({
        "turno_id": np.arange(1, n + 1), "turno": turno, "chancador": chancador,
        "css_mm": css.round(2), "desgaste_revestimiento_pct": desgaste.round(1),
        "humedad_pct": humedad.round(2), "alimentacion_tph": alimentacion.round(0),
        "dureza_indice": dureza.round(2),
        "p80_producto_mm": p80.round(2),
        "consumo_energia_kwh_t": energia.round(3),
        "horas_paro": horas_paro.round(2),
        "disponibilidad_pct": disponibilidad.round(2),
        "escenario": "Después de la mejora" if mejora else "Línea base",
    })


# ====================================================================
# DATOS DE ATRIBUTOS (cartas p, np, c, u)
# ====================================================================
def datos_atributos(k: int = 30, semilla: int = 2024, p_real: float = 0.045,
                    n_constante: int = 150, n_variable: bool = True) -> pd.DataFrame:
    """Inspecciones por turno: unidades defectuosas y defectos contados.

    defectuosos ~ Binomial(n_i, p_real)        -> cartas p y np
    defectos    ~ Poisson(lambda * unidades_i) -> cartas c y u
    En el subgrupo 22 se inyecta una causa especial (p sube a 2.6 veces) para
    que la carta tenga algo que detectar.
    """
    rng = np.random.default_rng(semilla + 77)
    n_i = (rng.integers(110, 190, k) if n_variable else np.full(k, n_constante))
    p_i = np.full(k, p_real)
    p_i[21] = p_real * 2.6
    defectuosos = rng.binomial(n_i, p_i)
    unidades = rng.integers(8, 15, k)
    lam = 0.55
    defectos = rng.poisson(lam * unidades)
    defectos_c = rng.poisson(6.5, k)
    return pd.DataFrame({
        "subgrupo": np.arange(1, k + 1),
        "turno": np.tile(["Día", "Noche"], k)[:k],
        "n_inspeccionadas": n_i,
        "defectuosas": defectuosos,
        "n_inspeccionadas_const": np.full(k, n_constante),
        "defectuosas_const": rng.binomial(n_constante, p_i),
        "unidades_inspeccion": unidades,
        "defectos": defectos,
        "defectos_area_constante": defectos_c,
    })


# ====================================================================
# GAGE R&R
# ====================================================================
def datos_gage_rr(piezas: int = 10, operadores: int = 3, replicas: int = 3,
                  semilla: int = 2024, sigma_pieza: float = 4.0,
                  sigma_repetibilidad: float = 0.9,
                  sesgos_operador: tuple[float, ...] = (0.0, 0.7, -0.45),
                  media: float = 100.0, etiqueta: str = "medicion") -> pd.DataFrame:
    """Estudio Gage R&R cruzado y balanceado con componentes de varianza conocidos.

    Diseño: cada operador mide CADA pieza 'replicas' veces. Este diseño es
    obligatorio; un Gage R&R no se puede calcular sobre datos operacionales
    corrientes.
    """
    rng = np.random.default_rng(semilla + 31)
    valores_pieza = rng.normal(media, sigma_pieza, piezas)
    ses = list(sesgos_operador)[:operadores]
    while len(ses) < operadores:
        ses.append(rng.normal(0, 0.5))
    filas = []
    for i in range(piezas):
        for j in range(operadores):
            for k in range(replicas):
                filas.append({
                    "pieza": f"P{i+1:02d}", "operador": f"Op{j+1}", "replica": k + 1,
                    etiqueta: round(float(valores_pieza[i] + ses[j]
                                          + rng.normal(0, sigma_repetibilidad)), 3),
                })
    return pd.DataFrame(filas)


# ====================================================================
# DOE 2^k
# ====================================================================
def datos_doe_carguio(replicas: int = 3, semilla: int = 2024) -> pd.DataFrame:
    """2^2: capacidad de balde (A) y posición del camion (B) sobre t_carguio.

    Verdad del generador (minutos de carguio):
        media global                      = 6.20
        efecto A (balde 28 -> 42 m3)      = -1.60  (balde mayor, menos pases)
        efecto B (posición simple->doble) = -0.70
        interacción A:B                   = +0.35  (con balde grande, cambiar la
                                                    posición rinde menos)
        error experimental sd             =  0.22
    """
    from statistics_engine.doe import design_matrix
    d = design_matrix({"A": (28, 42), "B": (0, 1)}, replicas=replicas,
                      aleatorizar=True, semilla=semilla)
    rng = np.random.default_rng(semilla + 41)
    d["t_carguio_min"] = (6.20 - 0.80 * d["A"] - 0.35 * d["B"] + 0.175 * d["A"] * d["B"]
                          + rng.normal(0, 0.22, len(d))).round(3)
    d = d.rename(columns={"A_real": "capacidad_balde_m3", "B_real": "posicion_codigo"})
    d["posicion"] = np.where(d["B"] == -1, "Simple (un lado)", "Doble (ambos lados)")
    return d


def datos_doe_molienda(replicas: int = 1, semilla: int = 2024) -> pd.DataFrame:
    """2^3 sin replicas sobre recuperación metalúrgica (%).

    Verdad (efectos, no coeficientes): media 84.0; A(pH) +0.50; B(colector) +2.00;
    C(p80) -4.40; B:C +0.40; resto nulos; ruido sd 0.18.

    Los efectos se eligieron ESCASOS y de magnitudes muy distintas a propósito: el
    método de Lenth, único disponible sin replicas, supone que la mayoría de los
    efectos son nulos y tiene muy poca potencia con k=3 (solo 7 efectos, gl ~ 2.3).
    Con cuatro efectos grandes simultaneos, Lenth no detecta ninguno.
    """
    from statistics_engine.doe import design_matrix
    d = design_matrix({"A": (10.2, 11.0), "B": (24, 34), "C": (160, 205)},
                      replicas=replicas, aleatorizar=True, semilla=semilla)
    rng = np.random.default_rng(semilla + 43)
    d["recuperacion_pct"] = (84.0 + 0.25 * d["A"] + 1.00 * d["B"] - 2.20 * d["C"]
                             + 0.20 * d["B"] * d["C"]
                             + rng.normal(0, 0.18, len(d))).round(3)
    return d.rename(columns={"A_real": "ph", "B_real": "colector_g_t", "C_real": "p80_um"})


def hoja_verificacion(semilla: int = 2024) -> pd.DataFrame:
    """Hoja de verificación: conteo de eventos de demora por categoría y turno."""
    rng = np.random.default_rng(semilla + 53)
    categorias = ["Espera en pala", "Congestion en ruta", "Via en mal estado",
                  "Falla mecánica", "Carga de combustible", "Cambio de turno",
                  "Espera en chancador", "Otros"]
    tasas = [42, 26, 18, 9, 7, 6, 5, 2]
    filas = []
    for cat, tasa in zip(categorias, tasas):
        for turno in ("Dia", "Noche"):
            filas.append({"categoria": cat, "turno": turno,
                          "eventos": int(rng.poisson(tasa * (1.15 if turno == "Dia" else 0.85))),
                          "minutos_perdidos": 0})
    df = pd.DataFrame(filas)
    dur = {"Espera en pala": 4.2, "Congestion en ruta": 3.1, "Via en mal estado": 2.4,
           "Falla mecánica": 28.0, "Carga de combustible": 12.0, "Cambio de turno": 9.0,
           "Espera en chancador": 3.8, "Otros": 5.0}
    df["minutos_perdidos"] = [round(r.eventos * dur[r.categoria] * rng.uniform(0.85, 1.15), 1)
                              for r in df.itertuples()]
    return df


def subgrupos_desde_serie(serie, n: int = 5) -> np.ndarray:
    """Reagrupa una serie en subgrupos racionales consecutivos de tamaño n.

    Un subgrupo racional debe contener observaciones tomadas bajo condiciones
    tan homogéneas como sea posible, de modo que la variación DENTRO del
    subgrupo represente solo causas comunes. Agrupar ciclos consecutivos del
    mismo turno cumple razonablemente ese criterio.
    """
    x = np.asarray(pd.Series(serie).dropna(), dtype=float)
    k = x.size // n
    if k < 2:
        raise ValueError(f"Datos insuficientes para formar 2 subgrupos de tamaño {n}.")
    return x[: k * n].reshape(k, n)


def subgrupos_racionales(df: pd.DataFrame, variable: str, por: str | None = None,
                         n: int = 5, devolver_indices: bool = False):
    """Forma subgrupos racionales, opcionalmente homogéneos dentro de un factor.

    Un subgrupo es RACIONAL cuando sus observaciones se tomaron bajo condiciones
    tan parecidas como sea posible, de modo que la variación DENTRO del subgrupo
    represente solo causas comunes y las diferencias entre condiciones aparezcan
    como variación ENTRE subgrupos.

    por=None  : subgrupos de observaciones consecutivas. Si el conjunto mezcla
                turnos o equipos, esa variación queda DENTRO del subgrupo, infla
                sigma_within y hace que Cp y Pp se parezcan artificialmente.
    por="turno": se agrupa primero por turno y se forman subgrupos dentro de cada
                uno. Entonces la diferencia entre turnos aparece como variación
                entre subgrupos, que es lo que la carta de medias debe detectar.

    devolver_indices : si es True, devuelve además una matriz con las posiciones
                de fila originales de cada observación, necesaria para poner un
                eje temporal o para rastrear una señal hasta el registro que la
                produjo.

    Devuelve (matriz, descripción) o (matriz, descripción, índices).
    """
    serie = df[variable]
    validos = serie.dropna().index.to_numpy()
    if por is None:
        x = serie.loc[validos].to_numpy(dtype=float)
        k = x.size // n
        if k < 2:
            raise ValueError(
                f"Datos insuficientes para formar 2 subgrupos de tamaño {n}.")
        m = x[: k * n].reshape(k, n)
        idx = df.index.get_indexer(validos[: k * n]).reshape(k, n)
        desc = f"{n} observaciones consecutivas por subgrupo, sin estratificar"
        return (m, desc, idx) if devolver_indices else (m, desc)

    bloques, bloques_idx = [], []
    for _, g in df.loc[validos].groupby(por, observed=True, sort=True):
        pos = g.index.to_numpy()
        x = g[variable].to_numpy(dtype=float)
        k = x.size // n
        if k >= 1:
            bloques.append(x[: k * n].reshape(k, n))
            bloques_idx.append(df.index.get_indexer(pos[: k * n]).reshape(k, n))
    if not bloques or sum(b.shape[0] for b in bloques) < 2:
        raise ValueError(
            f"Datos insuficientes para formar al menos 2 subgrupos de tamaño {n} "
            f"dentro de los niveles de '{por}'.")
    m = np.vstack(bloques)
    desc = f"{n} observaciones por subgrupo, homogéneas dentro de '{por}'"
    if devolver_indices:
        return m, desc, np.vstack(bloques_idx)
    return m, desc
