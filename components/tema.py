"""Tema visual y componentes de presentación.

Paleta validada con el validador de paletas (modo claro, superficie #fcfcfb,
modo --pairs all): separación CVD mínima dE 9.2, visión normal dE 24.0.
La serie aqua queda bajo 3:1 de contraste contra el fondo, por lo que toda
vista que la use acompaña el gráfico con su tabla de datos (regla de relieve).

Reglas de graficado que se respetan en todo el módulo de visualización:
  * Un solo eje por gráfico. El Pareto expresa las barras en PORCENTAJE para
    que la línea acumulada comparta escala, en vez de usar un segundo eje.
  * Los colores categoricos se asignan en orden fijo, nunca ciclados, y se
    limitan a tres series simultaneas.
  * Los colores de estado (bueno/alerta/crítico) nunca se usan como serie.
  * El texto va siempre en tinta, nunca en el color de la serie.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------- color ----
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorica: 3 slots maximo
SERIE_NOMBRES = ["azul", "naranja", "aqua"]

ESTADO = {"bueno": "#0ca30c", "alerta": "#fab219", "serio": "#ec835a", "critico": "#d03b3b"}

DIVERGENTE = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#e8a0a0", "#d03b3b", "#7a1f1f"]
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SUPERFICIE = "#fcfcfb"
PLANO = "#f9f9f7"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

FUENTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def registrar_tema() -> None:
    """Registra la plantilla de Plotly usada por toda la aplicación."""
    pio.templates["mssl"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FUENTE, size=13, color=TINTA_2),
            title=dict(font=dict(size=15, color=TINTA), x=0, xanchor="left"),
            paper_bgcolor=SUPERFICIE,
            plot_bgcolor=SUPERFICIE,
            colorway=SERIE,
            margin=dict(l=62, r=24, t=52, b=52),
            hoverlabel=dict(font=dict(family=FUENTE, size=12), bgcolor="#ffffff",
                            bordercolor=EJE),
            hovermode="closest",
            xaxis=dict(gridcolor=REJILLA, zerolinecolor=EJE, linecolor=EJE,
                       tickfont=dict(color=TINTA_MUTE, size=11),
                       title=dict(font=dict(color=TINTA_2, size=12)), showline=True),
            yaxis=dict(gridcolor=REJILLA, zerolinecolor=EJE, linecolor=EJE,
                       tickfont=dict(color=TINTA_MUTE, size=11),
                       title=dict(font=dict(color=TINTA_2, size=12)), showline=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=12, color=TINTA_2), bgcolor="rgba(0,0,0,0)"),
        )
    )
    pio.templates.default = "mssl"


def color_serie(i: int) -> str:
    """Color categorico en orden fijo. Más de 3 series: agrupar en 'Otros'."""
    if i >= len(SERIE):
        return TINTA_MUTE
    return SERIE[i]


CSS = f"""
<style>
  .stApp {{ background: {PLANO}; }}
  html, body, [class*="css"] {{ font-family: {FUENTE}; }}
  .mssl-tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0 18px 0; }}
  .mssl-tile {{
      flex: 1 1 165px; background: {SUPERFICIE}; border: 1px solid rgba(11,11,11,0.10);
      border-radius: 10px; padding: 14px 16px; min-width: 150px;
  }}
  .mssl-tile .lab {{
      font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
      color: {TINTA_MUTE}; margin-bottom: 6px; line-height: 1.3;
  }}
  .mssl-tile .val {{ font-size: 26px; font-weight: 600; color: {TINTA}; line-height: 1.1; }}
  .mssl-tile .uni {{ font-size: 13px; font-weight: 400; color: {TINTA_2}; margin-left: 3px; }}
  .mssl-tile .sub {{ font-size: 12px; color: {TINTA_2}; margin-top: 5px; line-height: 1.35; }}
  .mssl-badge {{
      display: inline-flex; align-items: center; gap: 5px; font-size: 11px;
      font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-top: 6px;
  }}
  .mssl-nota {{
      background: {SUPERFICIE}; border-left: 3px solid {EJE}; border-radius: 0 8px 8px 0;
      padding: 10px 14px; font-size: 13px; color: {TINTA_2}; margin: 8px 0;
      line-height: 1.5;
  }}
  .mssl-formula {{
      background: #ffffff; border: 1px solid {REJILLA}; border-radius: 8px;
      padding: 10px 14px; margin: 6px 0;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 24px; }}
  section[data-testid="stSidebar"] {{ background: {SUPERFICIE}; }}
</style>
"""

_ICONO = {"bueno": "●", "alerta": "▲", "serio": "▲", "critico": "✕", "neutro": "–"}


def tile(label: str, valor: str, unidad: str = "", sub: str = "",
         estado: str | None = None, etiqueta_estado: str = "") -> str:
    """HTML de una tarjeta KPI. El estado siempre lleva icono y texto, nunca color solo."""
    badge = ""
    if estado:
        c = ESTADO.get(estado, TINTA_MUTE)
        fondo = {"bueno": "#e8f6e8", "alerta": "#fdf3dc", "serio": "#fdeee7",
                 "critico": "#fbe9e9"}.get(estado, "#f0efec")
        badge = (f'<div class="mssl-badge" style="background:{fondo};color:{c};">'
                 f'{_ICONO.get(estado, "–")} {etiqueta_estado or estado.upper()}</div>')
    uni = f'<span class="uni">{unidad}</span>' if unidad else ""
    sb = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="mssl-tile"><div class="lab">{label}</div>'
            f'<div class="val">{valor}{uni}</div>{sb}{badge}</div>')


def fila_tiles(tiles: list[str]) -> str:
    return f'<div class="mssl-tiles">{"".join(tiles)}</div>'


def nota(texto: str) -> str:
    return f'<div class="mssl-nota">{texto}</div>'


def num(x, dec: int = 2) -> str:
    """Formato numérico consistente; devuelve '-' para valores no disponibles."""
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "-"
    if abs(v) >= 1e6 or (abs(v) < 1e-4 and v != 0):
        return f"{v:.3e}"
    return f"{v:,.{dec}f}".replace(",", " ")
