"""Toda tabla que la interfaz muestra debe ser serializable a Arrow.

Streamlit convierte cada DataFrame a una tabla de Arrow para enviarlo al
navegador. Una columna de tipo MIXTO (números y texto en la misma columna) no
es convertible: Streamlit la repara sola en cada renderizado y deja una traza
de error en el registro del servidor. No rompe la página, por eso no lo detecta
una prueba que solo cuente excepciones, pero es un defecto real y silencioso.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from data.cases import CASOS
from data.generators import datos_atributos, datos_doe_carguio, datos_gage_rr, subgrupos_racionales
from statistics_engine.capability import capability
from statistics_engine.control_charts import c_chart, cusum, ewma, i_mr, p_chart, u_chart, xbar_r
from statistics_engine.descriptive import describe_frame, percentiles
from statistics_engine.doe import factorial_analysis
from statistics_engine.gagerr import gage_rr_anova
from statistics_engine.inference import one_way_anova
from statistics_engine.montecarlo import Incierto, van_montecarlo
from statistics_engine.pareto import matriz_causa_efecto, pareto
from statistics_engine.regression import correlation_matrix, ols
from utils import plantillas
from utils.io import resultado_a_frame


def _arrow(df: pd.DataFrame, nombre: str) -> None:
    assert isinstance(df, pd.DataFrame), nombre
    try:
        pa.Table.from_pandas(df)
    except Exception as e:  # pragma: no cover - el mensaje importa al fallar
        pytest.fail(f"{nombre} no es serializable a Arrow: {e}")


@pytest.fixture(scope="module")
def datos():
    return CASOS["caso1"].datos(n=200, semilla=3)


def test_tablas_de_cartas_de_variables(datos):
    sub, _ = subgrupos_racionales(datos, "ciclo_total_min", None, 5)
    x, r = xbar_r(sub)
    _arrow(x.to_frame(), "xbar.to_frame")
    _arrow(r.to_frame(), "r.to_frame")
    _arrow(x.limites.resumen(), "limites.resumen")
    xe, _ = xbar_r(sub, excluir=[0, 3])
    _arrow(xe.to_frame(), "xbar excluyendo.to_frame")
    _arrow(xe.limites.resumen(), "limites con excluidos.resumen")


def test_tablas_de_cartas_de_memoria(datos):
    serie = datos["ciclo_total_min"]
    _arrow(ewma(serie).to_frame(), "ewma.to_frame")
    _arrow(ewma(serie).limites.resumen(), "ewma limites.resumen")
    _arrow(cusum(serie).to_frame(), "cusum.to_frame")
    i, mr = i_mr(serie)
    _arrow(i.to_frame(), "i.to_frame")
    _arrow(mr.to_frame(), "mr.to_frame")


def test_tablas_de_atributos():
    d = datos_atributos(k=25)
    _arrow(d, "datos_atributos")
    _arrow(p_chart(d["defectuosas"], d["n_inspeccionadas"]).to_frame(), "p.to_frame")
    _arrow(c_chart(d["defectos_area_constante"]).to_frame(), "c.to_frame")
    _arrow(u_chart(d["defectos"], d["unidades_inspeccion"]).to_frame(), "u.to_frame")


def test_tablas_de_capacidad_y_descriptiva(datos):
    _arrow(describe_frame(datos), "describe_frame")
    _arrow(percentiles(datos["ciclo_total_min"]), "percentiles")
    r = capability(datos["ciclo_total_min"], lse=38)
    _arrow(resultado_a_frame(r), "capacidad resultado_a_frame")


def test_tablas_de_analisis(datos):
    cols = ["ciclo_total_min", "distancia_km", "pendiente_pct"]
    c = correlation_matrix(datos, "pearson", cols)
    _arrow(c["r"].reset_index(), "matriz de correlacion")
    o = ols(datos, "ciclo_total_min", ["distancia_km", "pendiente_pct"])
    _arrow(o["coeficientes"], "coeficientes de regresion")
    _arrow(o["diagnostico"]["vif"], "vif")
    a = one_way_anova(datos, "ciclo_total_min", "turno")
    _arrow(a["tabla_anova"], "tabla anova")
    _arrow(a["resumen_grupos"], "resumen por grupo")
    _arrow(a["tukey"], "tukey")


def test_tablas_de_doe_y_gage():
    d = datos_doe_carguio(replicas=3)
    r = factorial_analysis(d, ["A", "B"], "t_carguio_min")
    _arrow(r["efectos"], "efectos DOE")
    _arrow(r["tabla_anova"], "anova DOE")
    _arrow(r["medias_por_corrida"], "medias por corrida")
    sl = factorial_analysis(datos_doe_carguio(replicas=1), ["A", "B"], "t_carguio_min")
    _arrow(sl["tabla_lenth"], "tabla de Lenth")
    g = gage_rr_anova(datos_gage_rr(), "pieza", "operador", "medicion", tolerancia=30)
    _arrow(g["tabla_anova"], "anova gage")
    _arrow(g["componentes"], "componentes gage")


def test_tablas_de_pareto_y_priorizacion():
    p = pareto(["A", "B", "C"], [50, 30, 20], etiqueta_valor="Minutos")
    _arrow(p["tabla"], "tabla de pareto")
    ce = matriz_causa_efecto(["c1", "c2"], ["s1", "s2"], [10, 5], {"c1": [9, 1], "c2": [3, 9]})
    _arrow(ce["tabla"], "matriz causa-efecto")


def test_tablas_de_montecarlo():
    r = van_montecarlo(
        Incierto("inversion", "pert", {"minimo": 90, "moda": 100, "maximo": 140}),
        Incierto("beneficio_anual", "triangular", {"minimo": 15, "moda": 28, "maximo": 40}),
        Incierto("tasa", "fija", {"valor": 0.12}), 5, n=1_000, semilla=4)
    _arrow(r.tabla_resumen(), "mc tabla_resumen")
    _arrow(r.sensibilidad(), "mc sensibilidad")
    _arrow(r.convergencia(), "mc convergencia")
    _arrow(r.muestras, "mc muestras")


@pytest.mark.parametrize("clave", list(plantillas.PLANTILLAS))
def test_tablas_de_plantillas(clave):
    _arrow(plantillas.obtener(clave)["datos"], f"plantilla {clave}")
    _arrow(plantillas.tabla_columnas(clave), f"columnas de {clave}")


@pytest.mark.parametrize("cid", list(CASOS))
def test_tablas_de_metadatos_de_cada_caso(cid):
    c = CASOS[cid]
    _arrow(c.datos(), f"datos de {cid}")
    _arrow(pd.DataFrame(c.voc), f"voc de {cid}")
    _arrow(pd.DataFrame(c.ctq), f"ctq de {cid}")
    _arrow(pd.DataFrame(c.cinco_porques["porques"]), f"cinco porques de {cid}")
    largo = max(len(v) for v in c.sipoc.values())
    _arrow(pd.DataFrame({k: v + [""] * (largo - len(v)) for k, v in c.sipoc.items()}),
           f"sipoc de {cid}")
    _arrow(pd.DataFrame([{"categoria": k, "causa": x}
                         for k, v in c.ishikawa.items() for x in v]), f"ishikawa de {cid}")
