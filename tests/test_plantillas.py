"""Plantillas CSV: estructura, validación y vuelta completa por la lectura real."""
import io

import pandas as pd
import pytest

from statistics_engine.control_charts import c_chart, np_chart, p_chart, u_chart, xbar_r
from statistics_engine.gagerr import gage_rr_anova
from utils import plantillas
from utils.io import a_csv, validar_csv


@pytest.mark.parametrize("clave", list(plantillas.PLANTILLAS))
def test_cada_plantilla_se_valida_a_si_misma(clave):
    p = plantillas.obtener(clave)
    ok, msgs = plantillas.validar(p["datos"], clave)
    assert ok, msgs
    assert p["titulo"] and p["cartas"] and p["descripcion"]
    assert p["columnas"]


@pytest.mark.parametrize("clave", list(plantillas.PLANTILLAS))
def test_vuelta_completa_por_csv(clave):
    """Exportar la plantilla y volver a leerla debe seguir validando: es el
    camino real que recorre el usuario."""
    p = plantillas.obtener(clave)
    df, msgs = validar_csv(io.BytesIO(a_csv(p["datos"])))
    assert df is not None, msgs
    ok, vmsgs = plantillas.validar(df, clave)
    assert ok, vmsgs
    assert len(df) == len(p["datos"])


@pytest.mark.parametrize("clave", list(plantillas.PLANTILLAS))
def test_las_columnas_obligatorias_estan_declaradas_y_presentes(clave):
    p = plantillas.obtener(clave)
    oblig = [c for c, d in p["columnas"].items() if d.startswith("OBLIGATORIA")]
    assert oblig, "toda plantilla debe declarar al menos una columna obligatoria"
    for c in oblig:
        assert c in p["datos"].columns


def test_falta_una_columna_obligatoria():
    d = plantillas.obtener("variables")["datos"].drop(columns=["medicion"])
    ok, msgs = plantillas.validar(d, "variables")
    assert not ok and "medicion" in msgs[0]


def test_columna_no_numerica():
    d = plantillas.obtener("variables")["datos"].copy()
    d["medicion"] = d["medicion"].astype(str) + " min"
    ok, msgs = plantillas.validar(d, "variables")
    assert not ok and "numérica" in msgs[0]


def test_defectuosas_mayores_que_inspeccionadas():
    d = plantillas.obtener("atributos_defectuosos")["datos"].copy()
    d.loc[0, "defectuosas"] = d.loc[0, "n_inspeccionadas"] + 1
    ok, msgs = plantillas.validar(d, "atributos_defectuosos")
    assert not ok and "defectuosas" in msgs[0]


def test_unidades_no_positivas():
    d = plantillas.obtener("atributos_defectos")["datos"].copy()
    d.loc[0, "unidades_inspeccion"] = 0
    ok, msgs = plantillas.validar(d, "atributos_defectos")
    assert not ok


def test_gage_desbalanceado_se_rechaza():
    d = plantillas.obtener("gage_rr")["datos"].iloc[:-1]
    ok, msgs = plantillas.validar(d, "gage_rr")
    assert not ok and "balanceado" in msgs[0]


def test_plantilla_desconocida():
    with pytest.raises(KeyError):
        plantillas.obtener("inexistente")


# ------------------------------------ las plantillas alimentan las cartas --
def test_la_plantilla_de_variables_produce_una_carta():
    from data.generators import subgrupos_racionales
    d = plantillas.obtener("variables")["datos"]
    sub, _ = subgrupos_racionales(d, "medicion", None, 4)
    x, r = xbar_r(sub)
    assert len(x.puntos) == sub.shape[0]
    assert x.limites is not None


def test_la_plantilla_de_defectuosos_produce_cartas_p_y_np():
    d = plantillas.obtener("atributos_defectuosos")["datos"]
    rp = p_chart(d["defectuosas"], d["n_inspeccionadas"])
    assert len(rp.puntos) == len(d)
    assert 0 <= rp.centro[0] <= 1


def test_la_plantilla_de_defectos_produce_cartas_c_y_u():
    d = plantillas.obtener("atributos_defectos")["datos"]
    rc = c_chart(d["defectos"])
    ru = u_chart(d["defectos"], d["unidades_inspeccion"])
    assert len(rc.puntos) == len(ru.puntos) == len(d)
    assert ru.centro[0] == pytest.approx(d["defectos"].sum() / d["unidades_inspeccion"].sum())


def test_la_plantilla_de_gage_produce_el_estudio():
    d = plantillas.obtener("gage_rr")["datos"]
    r = gage_rr_anova(d, "pieza", "operador", "medicion")
    assert r["n_piezas"] == 10 and r["n_operadores"] == 3 and r["n_replicas"] == 3
    assert 0 <= r["pct_rr_estudio"] <= 100


def test_tabla_de_columnas_es_legible():
    t = plantillas.tabla_columnas("variables")
    assert list(t.columns) == ["columna", "qué debe contener"]
    assert len(t) == len(plantillas.obtener("variables")["columnas"])
