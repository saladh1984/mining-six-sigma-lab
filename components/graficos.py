"""Figuras de Plotly. Cada función recibe RESULTADOS YA CALCULADOS.

Ninguna figura calcula estadística por su cuenta ni fija valores: todo lo que
se dibuja proviene del motor en statistics_engine/. Así, si un número aparece
en un gráfico, existe también en la tabla y en la exportación a CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .tema import (DIVERGENTE, EJE, ESTADO, REJILLA, SERIE, SUPERFICIE, TINTA,
                   TINTA_2, TINTA_MUTE, color_serie)

ALTURA = 420


def _vline(fig, x, color, texto, dash="dash", y_ref="paper"):
    fig.add_vline(x=x, line=dict(color=color, width=2, dash=dash),
                  annotation_text=texto, annotation_position="top",
                  annotation_font=dict(size=11, color=color))


def _hline(fig, y, color, texto, dash="dash", width=2):
    fig.add_hline(y=y, line=dict(color=color, width=width, dash=dash),
                  annotation_text=texto, annotation_position="right",
                  annotation_font=dict(size=11, color=color))


# ====================================================== MEDIR: distribucion
def histograma(x, titulo: str, unidad: str = "", bins: int = 0,
               lie=None, lse=None, objetivo=None, curva_normal: bool = True) -> go.Figure:
    s = pd.Series(x).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=bins or None, histnorm="probability density",
        marker=dict(color=SERIE[0], line=dict(color=SUPERFICIE, width=2)),
        opacity=0.85, name="Datos observados",
        hovertemplate="Intervalo: %{x}<br>Densidad: %{y:.4f}<extra></extra>"))
    if curva_normal and s.size > 2 and s.std(ddof=1) > 0:
        xs = np.linspace(s.min(), s.max(), 240)
        fig.add_trace(go.Scatter(
            x=xs, y=stats.norm.pdf(xs, s.mean(), s.std(ddof=1)),
            mode="lines", line=dict(color=TINTA_2, width=2),
            name="Normal ajustada (referencia)",
            hovertemplate="x=%{x:.3f}<br>densidad=%{y:.4f}<extra></extra>"))
    if lie is not None:
        _vline(fig, lie, ESTADO["critico"], "LIE")
    if lse is not None:
        _vline(fig, lse, ESTADO["critico"], "LSE")
    if objetivo is not None:
        _vline(fig, objetivo, ESTADO["bueno"], "Objetivo", dash="dot")
    fig.update_layout(title=titulo, height=ALTURA, bargap=0.02,
                      xaxis_title=unidad or "Valor", yaxis_title="Densidad")
    return fig


def densidad(x, titulo: str, unidad: str = "", por: pd.Series | None = None) -> go.Figure:
    """Densidad por kernel gaussiano. Hasta 3 grupos; más se agrupan en 'Otros'."""
    fig = go.Figure()
    s = pd.Series(x).dropna()
    if por is None:
        grupos = [("Todos", s)]
    else:
        g = pd.DataFrame({"v": pd.Series(x), "g": pd.Series(por).astype(str)}).dropna()
        top = g["g"].value_counts().index[:3]
        g["g"] = np.where(g["g"].isin(top), g["g"], "Otros")
        grupos = [(k, d["v"]) for k, d in g.groupby("g", observed=True)]
    for i, (nombre, v) in enumerate(grupos):
        if v.size < 3 or v.std(ddof=1) == 0:
            continue
        k = stats.gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), 240)
        fig.add_trace(go.Scatter(x=xs, y=k(xs), mode="lines", name=str(nombre),
                                 line=dict(color=color_serie(i), width=2),
                                 hovertemplate=f"{nombre}<br>x=%{{x:.3f}}<extra></extra>"))
    fig.update_layout(title=titulo, height=ALTURA,
                      xaxis_title=unidad or "Valor", yaxis_title="Densidad estimada")
    return fig


def boxplot(df: pd.DataFrame, variable: str, por: str | None = None,
            titulo: str = "", unidad: str = "") -> go.Figure:
    fig = go.Figure()
    if por is None:
        fig.add_trace(go.Box(y=df[variable], name=variable, boxpoints="suspectedoutliers",
                             marker=dict(color=SERIE[0], size=6),
                             line=dict(color=SERIE[0], width=2), fillcolor="rgba(42,120,214,0.14)"))
    else:
        cats = df[por].astype(str).value_counts().index[:3].tolist()
        otros = [c for c in df[por].astype(str).unique() if c not in cats]
        for i, c in enumerate(cats):
            v = df.loc[df[por].astype(str) == c, variable]
            fig.add_trace(go.Box(y=v, name=str(c), boxpoints="suspectedoutliers",
                                 marker=dict(color=color_serie(i), size=6),
                                 line=dict(color=color_serie(i), width=2)))
        if otros:
            v = df.loc[df[por].astype(str).isin(otros), variable]
            fig.add_trace(go.Box(y=v, name=f"Otros ({len(otros)})", boxpoints="suspectedoutliers",
                                 marker=dict(color=TINTA_MUTE, size=6),
                                 line=dict(color=TINTA_MUTE, width=2)))
    fig.update_layout(title=titulo or f"Diagrama de cajas - {variable}", height=ALTURA,
                      yaxis_title=unidad or variable, showlegend=por is not None)
    return fig


def capacidad(x, res, titulo: str = "", unidad: str = "") -> go.Figure:
    """Histograma con las dos curvas normales: corto plazo (Cp) y total (Pp)."""
    s = pd.Series(x).dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, histnorm="probability density", nbinsx=30,
                               marker=dict(color="#9ec5f4", line=dict(color=SUPERFICIE, width=2)),
                               name="Datos observados", opacity=0.9,
                               hovertemplate="Intervalo: %{x}<br>Densidad: %{y:.4f}<extra></extra>"))
    lo = min(s.min(), res.lie if res.lie is not None else s.min())
    hi = max(s.max(), res.lse if res.lse is not None else s.max())
    pad = 0.08 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 300)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, res.media, res.sigma_within),
                             mode="lines", line=dict(color=SERIE[0], width=2),
                             name="Corto plazo - dentro de subgrupo (Cp, Cpk)",
                             hovertemplate="x=%{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, res.media, res.sigma_overall),
                             mode="lines", line=dict(color=SERIE[1], width=2, dash="dot"),
                             name="Largo plazo - total (Pp, Ppk)",
                             hovertemplate="x=%{x:.3f}<extra></extra>"))
    if res.lie is not None:
        _vline(fig, res.lie, ESTADO["critico"], "LIE")
    if res.lse is not None:
        _vline(fig, res.lse, ESTADO["critico"], "LSE")
    if res.objetivo is not None:
        _vline(fig, res.objetivo, ESTADO["bueno"], "Objetivo", dash="dot")
    _vline(fig, res.media, TINTA_2, "Media", dash="solid")
    fig.update_layout(title=titulo or "Análisis de capacidad del proceso", height=460,
                      xaxis_title=unidad or "Valor", yaxis_title="Densidad", bargap=0.02)
    return fig


def qq(resultado_normalidad: dict, titulo: str = "Gráfico Q-Q (normalidad)") -> go.Figure:
    q = resultado_normalidad["qq"]
    t, o = np.array(q["teoricos"]), np.array(q["observados"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=q["intercepto"] + q["pendiente"] * t, mode="lines",
                             line=dict(color=TINTA_2, width=2), name="Referencia normal",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=o, mode="markers", name="Observaciones",
                             marker=dict(color=SERIE[0], size=8,
                                         line=dict(color=SUPERFICIE, width=2)),
                             hovertemplate="Cuantil teórico %{x:.3f}<br>Observado %{y:.3f}<extra></extra>"))
    fig.update_layout(title=titulo, height=ALTURA,
                      xaxis_title="Cuantiles teóricos de la normal",
                      yaxis_title="Cuantiles observados")
    return fig


# ======================================================== ANALIZAR
def dispersion(xv, yv, fit: dict, xlab: str, ylab: str, bandas: bool = True) -> go.Figure:
    fig = go.Figure()
    if bandas:
        fig.add_trace(go.Scatter(
            x=np.r_[fit["x_linea"], fit["x_linea"][::-1]],
            y=np.r_[fit["ic_pred_sup"], fit["ic_pred_inf"][::-1]],
            fill="toself", fillcolor="rgba(42,120,214,0.07)", line=dict(width=0),
            name="IC 95% de predicción", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=np.r_[fit["x_linea"], fit["x_linea"][::-1]],
            y=np.r_[fit["ic_media_sup"], fit["ic_media_inf"][::-1]],
            fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0),
            name="IC 95% de la media", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", name="Observaciones",
                             marker=dict(color=SERIE[0], size=8, opacity=0.75,
                                         line=dict(color=SUPERFICIE, width=2)),
                             hovertemplate=f"{xlab}: %{{x:.3f}}<br>{ylab}: %{{y:.3f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fit["x_linea"], y=fit["y_linea"], mode="lines",
                             line=dict(color=ESTADO["critico"], width=2),
                             name=f"Ajuste MCO (R2 = {fit['r2']:.3f})", hoverinfo="skip"))
    fig.update_layout(title=f"{ylab} frente a {xlab}", height=ALTURA,
                      xaxis_title=xlab, yaxis_title=ylab)
    return fig


def matriz_correlacion(r: pd.DataFrame, metodo: str) -> go.Figure:
    """Mapa de calor divergente: polaridad (-1 .. +1) con gris en el cero."""
    escala = [[0.0, DIVERGENTE[6]], [0.17, DIVERGENTE[5]], [0.33, DIVERGENTE[4]],
              [0.5, DIVERGENTE[3]], [0.67, DIVERGENTE[2]], [0.83, DIVERGENTE[1]],
              [1.0, DIVERGENTE[0]]]
    fig = go.Figure(go.Heatmap(
        z=r.values, x=list(r.columns), y=list(r.index), zmin=-1, zmax=1,
        colorscale=escala, zmid=0, xgap=2, ygap=2,
        colorbar=dict(title=dict(text="r", side="right"), tickvals=[-1, -0.5, 0, 0.5, 1],
                      outlinewidth=0, thickness=14),
        text=np.round(r.values, 2), texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.4f}<extra></extra>"))
    fig.update_layout(title=f"Matriz de correlación ({metodo})",
                      height=max(420, 60 + 42 * len(r)),
                      xaxis=dict(showgrid=False, tickangle=-35),
                      yaxis=dict(showgrid=False, autorange="reversed"))
    return fig


def residuos(res: dict) -> go.Figure:
    """Cuatro paneles de diagnóstico de residuos en una sola figura."""
    from plotly.subplots import make_subplots
    aj, re_ = res["ajustados"], res["residuos"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Residuos frente a valores ajustados", "Q-Q de los residuos",
        "Histograma de residuos", "Residuos en orden de observación"))
    m = dict(color=SERIE[0], size=7, opacity=0.75, line=dict(color=SUPERFICIE, width=1.5))
    fig.add_trace(go.Scatter(x=aj, y=re_, mode="markers", marker=m, showlegend=False,
                             hovertemplate="ajustado %{x:.3f}<br>residuo %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line=dict(color=TINTA_2, width=2, dash="dash"), row=1, col=1)
    (osm, osr), (sl, ic, _) = stats.probplot(re_, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=ic + sl * osm, mode="lines",
                             line=dict(color=TINTA_2, width=2), showlegend=False, hoverinfo="skip"), 1, 2)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=m, showlegend=False,
                             hovertemplate="teórico %{x:.3f}<br>observado %{y:.3f}<extra></extra>"), 1, 2)
    fig.add_trace(go.Histogram(x=re_, nbinsx=25, showlegend=False,
                               marker=dict(color=SERIE[0], line=dict(color=SUPERFICIE, width=2)),
                               hovertemplate="residuo %{x}<br>n=%{y}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(re_) + 1), y=re_, mode="lines+markers",
                             line=dict(color=SERIE[0], width=1.5),
                             marker=dict(color=SERIE[0], size=5), showlegend=False,
                             hovertemplate="obs %{x}<br>residuo %{y:.3f}<extra></extra>"), 2, 2)
    fig.add_hline(y=0, line=dict(color=TINTA_2, width=2, dash="dash"), row=2, col=2)
    fig.update_annotations(font=dict(size=12, color=TINTA))
    fig.update_layout(height=640, title="Diagnóstico de residuos", bargap=0.03)
    return fig


# =========================================================== MEJORAR: DOE
def efectos_principales(datos: pd.DataFrame, respuesta_lab: str) -> go.Figure:
    from plotly.subplots import make_subplots
    factores = datos["factor"].unique().tolist()
    fig = make_subplots(rows=1, cols=len(factores), shared_yaxes=True,
                        subplot_titles=[f"Factor {f}" for f in factores])
    media = datos["media"].mean()
    for i, f in enumerate(factores, start=1):
        d = datos[datos["factor"] == f].sort_values("codigo")
        fig.add_trace(go.Scatter(x=d["nivel"], y=d["media"], mode="lines+markers+text",
                                 text=[f"{v:.2f}" for v in d["media"]], textposition="top center",
                                 textfont=dict(size=11, color=TINTA_2),
                                 line=dict(color=SERIE[0], width=2),
                                 marker=dict(color=SERIE[0], size=10,
                                             line=dict(color=SUPERFICIE, width=2)),
                                 showlegend=False,
                                 hovertemplate=f"{f} %{{x}}<br>media %{{y:.4f}}<extra></extra>"), 1, i)
        fig.add_hline(y=media, line=dict(color=TINTA_MUTE, width=1.5, dash="dot"), row=1, col=i)
    fig.update_annotations(font=dict(size=12, color=TINTA))
    fig.update_yaxes(title_text=respuesta_lab, row=1, col=1)
    fig.update_layout(height=400, title="Gráfico de efectos principales "
                                        "(línea punteada = media global)")
    return fig


def interaccion(datos: pd.DataFrame, f1: str, f2: str, respuesta_lab: str) -> go.Figure:
    fig = go.Figure()
    for i, (nivel, d) in enumerate(datos.groupby(f2, observed=True)):
        d = d.sort_values(f1)
        fig.add_trace(go.Scatter(x=d[f1], y=d["media"], mode="lines+markers+text",
                                 name=f"{f2} = {nivel}",
                                 text=[f"{v:.2f}" for v in d["media"]], textposition="top center",
                                 textfont=dict(size=11, color=TINTA_2),
                                 line=dict(color=color_serie(i), width=2),
                                 marker=dict(color=color_serie(i), size=10,
                                             line=dict(color=SUPERFICIE, width=2)),
                                 hovertemplate=f"{f1} %{{x}}<br>media %{{y:.4f}}<extra></extra>"))
    fig.update_layout(title=f"Gráfico de interacción {f1} x {f2} "
                            "(líneas paralelas = sin interacción)",
                      height=ALTURA, xaxis_title=f"Factor {f1}", yaxis_title=respuesta_lab)
    return fig


def pareto_efectos(efectos: pd.DataFrame, limite: float | None = None,
                   etiqueta_limite: str = "") -> go.Figure:
    d = efectos.assign(abs_ef=efectos["efecto"].abs()).sort_values("abs_ef")
    colores = [SERIE[0] if o == 1 else SERIE[1] for o in d["orden"]]
    fig = go.Figure(go.Bar(
        x=d["abs_ef"], y=d["termino"], orientation="h", marker=dict(color=colores),
        text=[f"{v:.3f}" for v in d["efecto"]], textposition="outside",
        textfont=dict(size=11, color=TINTA_2), name="Magnitud del efecto",
        hovertemplate="%{y}<br>efecto = %{text}<extra></extra>"))
    if limite:
        fig.add_vline(x=limite, line=dict(color=ESTADO["critico"], width=2, dash="dash"),
                      annotation_text=etiqueta_limite or "Limite",
                      annotation_font=dict(size=11, color=ESTADO["critico"]))
    fig.update_layout(title="Pareto de efectos (magnitud absoluta)",
                      height=max(320, 70 + 38 * len(d)),
                      xaxis_title="|Efecto|", yaxis_title="",
                      showlegend=False)
    return fig


# ============================================================ PARETO
def pareto(tabla: pd.DataFrame, etiqueta_valor: str, corte: float = 80.0) -> go.Figure:
    """Pareto en UN SOLO EJE: las barras van en porcentaje del total, igual que
    la línea acumulada, de modo que ambas comparten escala. Los valores
    absolutos están en el hover y en la tabla de datos."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tabla["categoria"], y=tabla["pct"],
        marker=dict(color=[SERIE[0] if v else TINTA_MUTE for v in tabla["vital"]],
                    line=dict(color=SUPERFICIE, width=2)),
        name=f"% del total ({etiqueta_valor})",
        customdata=tabla["valor"],
        text=[f"{v:.1f}%" for v in tabla["pct"]], textposition="outside",
        textfont=dict(size=11, color=TINTA_2),
        hovertemplate="%{x}<br>%{y:.1f}% del total<br>"
                      f"{etiqueta_valor}: %{{customdata:,.1f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=tabla["categoria"], y=tabla["pct_acumulado"], mode="lines+markers",
        line=dict(color=ESTADO["serio"], width=2),
        marker=dict(color=ESTADO["serio"], size=9, line=dict(color=SUPERFICIE, width=2)),
        name="% acumulado",
        hovertemplate="%{x}<br>acumulado %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=corte, line=dict(color=TINTA_MUTE, width=1.5, dash="dot"),
                  annotation_text=f"Corte {corte:.0f}%", annotation_position="right",
                  annotation_font=dict(size=11, color=TINTA_MUTE))
    fig.update_layout(title=f"Diagrama de Pareto - {etiqueta_valor}", height=ALTURA,
                      yaxis_title="Porcentaje del total (%)", xaxis_title="",
                      yaxis=dict(range=[0, 108]), xaxis=dict(tickangle=-25))
    return fig


# =========================================================== CONTROLAR
def carta_control(res, titulo: str = "", mostrar_zonas: bool = True) -> go.Figure:
    """Carta de control. Los puntos con señal se marcan con color Y simbolo,
    nunca con color solo."""
    n = len(res.puntos)
    x = np.arange(1, n + 1)
    fuera = set(res.fuera_de_control)
    fig = go.Figure()

    if mostrar_zonas and res.sigma:
        cl = res.centro[0]
        for k, alpha in ((1, 0.04), (2, 0.04)):
            fig.add_hrect(y0=cl + (k - 1) * res.sigma, y1=cl + k * res.sigma,
                          fillcolor=f"rgba(42,120,214,{alpha})", line_width=0, layer="below")
            fig.add_hrect(y0=cl - k * res.sigma, y1=cl - (k - 1) * res.sigma,
                          fillcolor=f"rgba(42,120,214,{alpha})", line_width=0, layer="below")

    fig.add_trace(go.Scatter(x=x, y=res.lcs, mode="lines", name="LCS (límite de control superior)",
                             line=dict(color=ESTADO["critico"], width=2, dash="dash"),
                             hovertemplate="LCS = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=res.centro, mode="lines", name="LC (línea central)",
                             line=dict(color=ESTADO["bueno"], width=2),
                             hovertemplate="LC = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=res.lci, mode="lines", name="LCI (límite de control inferior)",
                             line=dict(color=ESTADO["critico"], width=2, dash="dash"),
                             hovertemplate="LCI = %{y:.4f}<extra></extra>"))

    reglas_por_punto = {}
    for r, idxs in res.violaciones.items():
        for i in idxs:
            reglas_por_punto.setdefault(i, []).append(r)
    texto = ["sin señal" if i not in reglas_por_punto else
             "reglas " + ", ".join(map(str, sorted(reglas_por_punto[i]))) for i in range(n)]

    normales = [i for i in range(n) if i not in fuera]
    fig.add_trace(go.Scatter(
        x=x[normales], y=np.asarray(res.puntos)[normales], mode="lines+markers",
        name="Dentro de control", line=dict(color=SERIE[0], width=1.6),
        marker=dict(color=SERIE[0], size=8, line=dict(color=SUPERFICIE, width=2)),
        customdata=[texto[i] for i in normales],
        hovertemplate="Subgrupo %{x}<br>Valor %{y:.4f}<br>%{customdata}<extra></extra>"))
    if fuera:
        f = sorted(fuera)
        fig.add_trace(go.Scatter(
            x=x[f], y=np.asarray(res.puntos)[f], mode="markers",
            name="Señal de causa especial",
            marker=dict(color=ESTADO["critico"], size=13, symbol="x-thin",
                        line=dict(color=ESTADO["critico"], width=3)),
            customdata=[texto[i] for i in f],
            hovertemplate="Subgrupo %{x}<br>Valor %{y:.4f}<br><b>%{customdata}</b><extra></extra>"))

    fig.update_layout(title=titulo or res.nombre, height=ALTURA,
                      xaxis_title="Subgrupo / observación", yaxis_title=res.etiqueta_y)
    return fig


def componentes_variacion(componentes: pd.DataFrame) -> go.Figure:
    """Barras de %Contribución y %Variación de estudio del Gage R&R."""
    d = componentes[~componentes["Fuente"].str.startswith(("  ", "Variación total"))].copy()
    d = pd.concat([d, componentes[componentes["Fuente"].str.startswith("  ")]], ignore_index=True)
    d["Fuente"] = d["Fuente"].str.strip()
    # Etiquetas cortas en dos lineas: los nombres completos se cortaban contra el
    # borde inferior del grafico. El nombre integro queda en el hover y en la tabla.
    cortas = {
        "Gage R&R total": "Gage R&R<br>total",
        "Repetibilidad (equipo)": "Repetibilidad<br>(equipo)",
        "Reproducibilidad": "Reproduci-<br>bilidad",
    }
    d["etiqueta"] = [cortas.get(f, f.replace(" de ", "<br>de ").replace(" x ", "<br>x "))
                     for f in d["Fuente"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["etiqueta"], y=d["%Contribucion"], name="% de contribución (varianzas)",
                         marker=dict(color=SERIE[0], line=dict(color=SUPERFICIE, width=2)),
                         text=[f"{v:.1f}%" for v in d["%Contribucion"]], textposition="outside",
                         textfont=dict(size=11, color=TINTA_2), customdata=d["Fuente"],
                         hovertemplate="%{customdata}<br>%{y:.2f}% de la varianza total<extra></extra>"))
    fig.add_trace(go.Bar(x=d["etiqueta"], y=d["%VariacionEstudio"],
                         name="% de variación de estudio (desviaciones)",
                         marker=dict(color=SERIE[1], line=dict(color=SUPERFICIE, width=2)),
                         text=[f"{v:.1f}%" for v in d["%VariacionEstudio"]], textposition="outside",
                         textfont=dict(size=11, color=TINTA_2), customdata=d["Fuente"],
                         hovertemplate="%{customdata}<br>%{y:.2f}% de la desviación total<extra></extra>"))
    fig.add_hline(y=30, line=dict(color=ESTADO["critico"], width=2, dash="dash"),
                  annotation_text="30% inaceptable", annotation_position="right",
                  annotation_font=dict(size=11, color=ESTADO["critico"]))
    fig.add_hline(y=10, line=dict(color=ESTADO["bueno"], width=2, dash="dot"),
                  annotation_text="10% aceptable", annotation_position="right",
                  annotation_font=dict(size=11, color=ESTADO["bueno"]))
    fig.update_layout(title="Componentes de variación del sistema de medición",
                      height=470, barmode="group", bargap=0.25, bargroupgap=0.08,
                      margin=dict(b=96), yaxis_title="Porcentaje (%)",
                      xaxis=dict(tickangle=0, tickfont=dict(size=10)))
    return fig


def serie_temporal(df: pd.DataFrame, x: str, y: str, color: str | None = None,
                   titulo: str = "", unidad: str = "") -> go.Figure:
    fig = go.Figure()
    if color is None:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=y,
                                 line=dict(color=SERIE[0], width=1.8),
                                 marker=dict(color=SERIE[0], size=6),
                                 hovertemplate=f"{x} %{{x}}<br>{y} %{{y:.3f}}<extra></extra>"))
    else:
        cats = df[color].astype(str).value_counts().index[:3].tolist()
        for i, c in enumerate(cats):
            d = df[df[color].astype(str) == c]
            fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", name=str(c),
                                     marker=dict(color=color_serie(i), size=7, opacity=0.8,
                                                 line=dict(color=SUPERFICIE, width=1.5)),
                                     hovertemplate=f"{c}<br>{x} %{{x}}<br>%{{y:.3f}}<extra></extra>"))
    fig.update_layout(title=titulo or f"{y} en el tiempo", height=ALTURA,
                      xaxis_title=x, yaxis_title=unidad or y)
    return fig


def comparacion_escenarios(antes, despues, etiqueta_antes: str, etiqueta_despues: str,
                           titulo: str, unidad: str = "") -> go.Figure:
    """Densidades superpuestas de dos escenarios, con sus medias marcadas."""
    fig = go.Figure()
    for i, (v, nom) in enumerate(((pd.Series(antes).dropna(), etiqueta_antes),
                                  (pd.Series(despues).dropna(), etiqueta_despues))):
        k = stats.gaussian_kde(v)
        xs = np.linspace(min(pd.Series(antes).min(), pd.Series(despues).min()),
                         max(pd.Series(antes).max(), pd.Series(despues).max()), 300)
        fig.add_trace(go.Scatter(x=xs, y=k(xs), mode="lines", name=nom,
                                 line=dict(color=color_serie(i), width=2),
                                 fill="tozeroy",
                                 fillcolor=("rgba(42,120,214,0.12)" if i == 0
                                            else "rgba(235,104,52,0.12)"),
                                 hovertemplate=f"{nom}<br>x=%{{x:.3f}}<extra></extra>"))
        fig.add_vline(x=float(v.mean()), line=dict(color=color_serie(i), width=2, dash="dash"),
                      annotation_text=f"media {nom}: {v.mean():.2f}",
                      annotation_position="top", annotation_font=dict(size=11, color=color_serie(i)))
    fig.update_layout(title=titulo, height=ALTURA,
                      xaxis_title=unidad or "Valor", yaxis_title="Densidad estimada")
    return fig
