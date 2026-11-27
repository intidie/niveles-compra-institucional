"""UI web (NiceGUI + TradingView Lightweight Charts en Analisis, Apache ECharts en Valor Justo; modo oscuro) para el screener de niveles de compra institucionales.

Lanzar:  python app_ui.py        (puerto 8080; variables APP_UI_PORT y APP_UI_SHOW=0 para no abrir navegador)

Solo CONSUME la logica existente (``main.analizar_ticker``, ``main.leer_watchlist``, ``main.ordenar_ranking``,
``main.asignar_capital`` y las utilidades de datos de ``src.reporting.charts``); no modifica src/data, src/levels
ni src/validation. ``import app_ui`` no descarga nada ni levanta el servidor.

Mapeo de toggles -> nombres reales en ``zonas[i]["familias"]`` / ``["tipos"]``
(ver ``src/validation/backtest_ranges.py::FAMILIAS`` y ``main.analizar_ticker``):

    Toggle UI              familia real ("familias")   tipos reales ("tipos")
    ---------------------  --------------------------  -------------------------------------
    HVN / Volume Profile   "volumen"                   POC, VAL, VAH, HVN
    Order Blocks           "ob"                        OB
    AVWAP                  "avwap"                     AVWAP
    Swing Lows macro       "swing"                     SWING
    LogReg                 "logreg"                    LOG
    DCF / P/E              "valoracion" (pseudo)       PE_*, EVFCF_*, DCF_*  (de res["valoracion"])

Las zonas calculadas por ``analizar_ticker`` NO traen anclas de valoracion (el backtest las excluye por no ser
point-in-time); el toggle DCF/P/E las superpone desde ``res["valoracion"]`` como lineas de referencia y las cuenta
como una familia extra en las zonas que las contienen (misma tolerancia +-2 % que usa main.py).

Modo oscuro: TODOS los colores viven en el bloque PALETA (mas abajo). Los graficos ECharts de Valor Justo
(``opciones_historico``, ``opciones_barras``), el tema que recibe el JS del grafico de Analisis (``tema_grafico``), el CSS
inyectado (``css_tema``) y ``ui.colors`` los toman de ahi; un test impide literales de color fuera de ese bloque y
comprueba contraste WCAG >= 4.5:1 en los pares texto/fondo.

Pestana "Valor Justo": bateria de 15 modelos de ``src/levels/fair_value.py`` (implementacion propia "estilo
InvestingPro"; no replica sus cifras) en dos ESCENARIOS (Medio y Pesimista). Recalcula en memoria con
``fv.calcular_escenarios`` sobre ``res["entradas_valoracion"]`` (sin SEC ni yfinance) y es aditiva: no toca las zonas ni
los pisos de valoracion. Estructura: 4 tarjetas KPI, grafico historico con bandas +-10/20/30 % (``fv.serie_valor_justo_historica``,
fuera del hilo de eventos y cacheada por huella) y, en acordeones colapsados, barras por modelo, controles y supuestos.
Con "Mostrar Valor Justo" dibuja en Analisis SOLO la linea del Escenario Medio (``valor_justo_medio`` es el unico punto de acceso).

Local vs nube (``userdata.en_nube()`` = variable PORT): local sin restricciones (10 anios, cache SQLite, escaneo completo);
nube = visor ligero (``periodo="3y"``, serie de 3 anios, un solo escaneo a la vez con ``MAX_TICKERS_ESCANEO_NUBE`` tickers y
``gc.collect()`` tras cada consulta).

Grafico de Analisis: ``static/vendor/lightweight-charts.standalone.production.js`` (v5, Apache-2.0, servido en local
con ``app.add_static_files``) y ``static/charts/grafico_analisis.js``; Python solo prepara JSON (``datos_grafico``,
``overlays_grafico``) y lo envia con ``run_javascript`` (``Vista._grafico_js``). Sin CDN: funciona offline y en el .exe.

Pestana "Modelo": ``contenido_modelo()`` devuelve la explicacion como datos (lista de secciones) leyendo pesos y
parametros del codigo en tiempo de llamada; ``renderizar_modelo()`` solo la dibuja.
"""
from __future__ import annotations

import asyncio
import copy
import gc
import inspect
import json
import math
import os
import re
import sys
from collections import OrderedDict
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError
from string import Template
from typing import Any, Iterable

if sys.stdout is None or sys.stderr is None:  # exe congelado sin consola (console=False): print/uvicorn fallarian
    _nul = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or _nul
    sys.stderr = sys.stderr or _nul
os.environ.setdefault("MPLBACKEND", "Agg")  # charts.py importa pyplot; el servidor no necesita ventana
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from nicegui import app  # noqa: E402  (importar no abre el servidor; solo registra /health)

import main as nucleo  # noqa: E402  (analizar_ticker, leer_watchlist, ordenar_ranking, asignar_capital)
from src.levels import fair_value as fv  # noqa: E402  (valor justo: calculo puro y en memoria, sin red)
from src.reporting.charts import (ETIQUETA_DIST_RSUP, dist_a_r_sup_pct, marcadores_form4,  # noqa: E402
                                  perfil_histograma, trimestres_acumulacion)
from src.reporting import userdata  # noqa: E402

RAIZ = userdata.raiz_recursos()  # raiz del repo o sys._MEIPASS si esta congelado (PyInstaller)
RUTA_WATCHLIST = userdata.recurso("watchlist.csv")  # lista de EJEMPLO, solo lectura
RUTA_METODOLOGIA = userdata.recurso("docs", "METODOLOGIA.md")
URL_STATIC = "/static-niveles"   # ruta publica de la carpeta static/ (LW offline, sin CDN)
DIR_STATIC = userdata.recurso("static")   # recursos estaticos empaquetados (build.spec los copia): LW offline, sin CDN
if os.path.isdir(DIR_STATIC):
    app.add_static_files(URL_STATIC, DIR_STATIC)
MAX_CACHE_NUBE = 3  # tickers con resultado completo (df incluido) retenidos en la nube (512 MB)
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
VJ_RANGO_EJE = (0.3, 3.0)   # el valor justo fuera de [0.3x, 3x] el precio no entra en el rango del eje Y (se anota "fuera de rango")
TOLERANCIA_VALORACION = (0.98, 1.02)  # igual que main.analizar_ticker (anclas de valoracion en zona)

# ===================================== PALETA (unica fuente de colores) =====================================
# Fondos (de mas oscuro a mas claro)
BG_PAGINA = "#0d1117"      # body / area principal
BG_PAPEL = "#0d1117"       # fondo opaco de los graficos ECharts (el area de trazado usa BG_PANEL)
BG_PANEL = "#131722"       # area de trazado de los graficos
BG_DRAWER = "#10141c"      # panel lateral
BG_TARJETA = "#161b22"     # tarjetas, expansiones, cabeceras de tabla
BG_POPUP = "#1f2735"       # menus y tooltips (Quasar y ECharts)
BG_ETIQUETA = "rgba(13,17,23,0.82)"   # fondo de las etiquetas de bandas/lineas sobre el grafico
# Bordes y rejilla
BORDE = "#3a4352"
GRILLA = "#232936"
# Texto
TXT = "#e6edf3"
BLANCO = "#ffffff"
MUTED = "#a8b3c1"          # texto secundario: >= 4.5:1 sobre todos los fondos de arriba
# Acentos y estados
ACENTO = "#1f6feb"         # fondo de botones/switches (texto blanco >= 4.5:1)
ACENTO_TXT = "#79b8ff"     # acento como TEXTO sobre fondo oscuro (foco, tab activa, item seleccionado)
OK_TXT, AVISO_TXT, ERROR_TXT = "#56d364", "#ffc83d", "#ff8b81"
OK_BG, ERROR_BG, INFO_BG, AVISO_BG = "#1a7f37", "#b42318", "#1f6feb", "#f2c037"  # ui.colors / notify
TXT_SOBRE_AVISO = "#1a1a1a"  # el amarillo de aviso lleva texto oscuro; el resto lleva blanco
HOVER_FILA = "rgba(121,184,255,0.10)"
SEL_FILA = "rgba(121,184,255,0.20)"
ACTIVO_ITEM = "rgba(121,184,255,0.14)"
# Series del grafico
VELA_ALZA, VELA_BAJA = "#2ea043", "#f85149"
COLOR_13F, COLOR_FORM4, COLOR_VALORACION, COLOR_PRECIO = "#a371f7", "#3fb950", "#e3b341", "#f0f6fc"
COLOR_HVN, COLOR_BARRA_VP = "#f59e0b", "#404a5a"
COLORES_NIVEL = {1: "#2ea043", 2: "#00b4d8", 3: "#f59e0b", 4: "#a371f7"}
COLOR_FAIR_VALUE = "#ff7eb6"            # linea del valor justo (distinta de bandas N1-N4, doradas y precio)
VJ_ACTIVO, VJ_INACTIVO = "#58a6ff", "#6e7b8f"   # barras de modelos usados / desactivados (el outlier usa AVISO_TXT)
VJ_PESIMISTA = "#f0883e"                # Escenario Pesimista: barras y linea del piso fundamental
VJ_CIAN, VJ_MAGENTA = "#22d3ee", "#e879f9"   # bandas del grafico historico: +10/20/30 % (cian) y -10/-20/-30 % (magenta)
# =================================== FIN PALETA ===================================

PALETA: dict[str, str] = {k: v for k, v in dict(globals()).items()
                          if k.isupper() and isinstance(v, str) and re.match(r"^(#|rgba?\()", v)}
# Colores Quasar para ui.colors (notificaciones, botones, errores de campo); el texto de los estados es blanco
# salvo en el aviso amarillo (TXT_SOBRE_AVISO). Ver test_colores_quasar_contraste.
COLORES_QUASAR = dict(primary=ACENTO, secondary=ACENTO, accent=COLOR_13F, dark=BG_TARJETA, dark_page=BG_PAGINA,
                      positive=OK_BG, negative=ERROR_BG, info=INFO_BG, warning=AVISO_BG)

_CSS_TEMA = Template("""
html{color-scheme:dark}
html,body,body.body--dark,.q-layout,.q-page-container,.q-page{background:$BG_PAGINA;color:$TXT}
.q-drawer,.q-drawer--dark{background:$BG_DRAWER !important;color:$TXT}
.body--dark .q-card{background:$BG_TARJETA;color:$TXT;border:1px solid $BORDE;box-shadow:none}
.txt-muted{color:$MUTED !important}
.txt-aviso{color:$AVISO_TXT !important}
.txt-error{color:$ERROR_TXT !important}
.txt-ok{color:$OK_TXT !important}
.ayuda-icono{color:$MUTED;cursor:help}
.ayuda-icono:hover,.ayuda-icono:focus{color:$TXT}
.q-menu{background:$BG_POPUP;color:$TXT;border:1px solid $BORDE}
.q-tooltip{background:$BG_POPUP;color:$TXT;border:1px solid $BORDE;font-size:12.5px;line-height:1.4;max-width:300px;
  white-space:normal}
.q-dialog__inner>.q-card{background:$BG_POPUP;color:$TXT}
.q-notification{font-size:14px}
.q-notification__actions,.q-notification__actions .q-btn{color:inherit !important}
.q-item{color:$TXT}
.q-item--active{color:$ACENTO_TXT;background:$ACTIVO_ITEM}
.q-item__label--caption{color:$MUTED}
.q-field--focused .q-field__label{color:$ACENTO_TXT}
.q-field--error .q-field__label,.q-field--error .q-field__bottom,.q-field--error .q-field__marginal{color:$ERROR_TXT !important}
.q-field--error .q-field__control:after{border-color:$ERROR_TXT !important}
.q-field--dark .q-field__native,.q-field--dark .q-field__prefix,.q-field--dark .q-field__suffix{color:$TXT}
.q-tab{color:$MUTED}
.q-tab--active{color:$TXT}
.q-tab__indicator{color:$ACENTO_TXT}
.q-table{color:$TXT}
.q-table thead th{color:$TXT;font-weight:600;background:$BG_TARJETA}
.q-table tbody td{color:$TXT}
.q-table tbody tr:hover td{background:$HOVER_FILA}
.q-table tbody tr.selected td{background:$SEL_FILA}
.q-table__bottom,.q-table__control,.q-table__bottom .q-field__native,.q-table__bottom--nodata{color:$MUTED}
.q-linear-progress__track{opacity:.35}
.modelo-sec{background:$BG_TARJETA;border:1px solid $BORDE;border-radius:6px}
.modelo-sec .q-expansion-item__container>.q-item{color:$TXT;font-weight:600}
.modelo-sec .q-item__section--side>.q-icon{color:$ACENTO_TXT}
.nicegui-markdown,.nicegui-markdown p,.nicegui-markdown li{color:$TXT;line-height:1.55}
.nicegui-markdown strong{color:$BLANCO}
.nicegui-markdown a{color:$ACENTO_TXT}
.nicegui-markdown code{background:$BG_PAGINA;color:$ACENTO_TXT;padding:1px 4px;border-radius:3px}
.nicegui-markdown table{border-collapse:collapse;width:100%;font-size:.92rem;margin:4px 0}
.nicegui-markdown th,.nicegui-markdown td{border:1px solid $BORDE;padding:6px 10px;text-align:left;vertical-align:top}
.nicegui-markdown th{background:$BG_DRAWER;color:$TXT}
.ejemplo-box{border-left:3px solid $ACENTO_TXT;background:$BG_PAGINA;padding:8px 12px;border-radius:0 4px 4px 0}
.aviso-box{border-left:4px solid $AVISO_TXT;background:$BG_PAGINA;padding:8px 12px;border-radius:0 4px 4px 0}
.aviso-card{border:1px solid $AVISO_TXT !important;border-left-width:5px !important}
.vj-badge{display:inline-block;padding:2px 12px;border-radius:12px;font-weight:700;font-size:1.05rem}
.vj-baja{background:$OK_BG;color:$BLANCO}
.vj-media{background:$AVISO_BG;color:$TXT_SOBRE_AVISO}
.vj-alta{background:$ERROR_BG;color:$BLANCO}
.vj-dx-infra{background:$OK_BG;color:$BLANCO}
.vj-dx-linea{background:$INFO_BG;color:$BLANCO}
.vj-dx-sobre{background:$ERROR_BG;color:$BLANCO}
.vj-dx-debil{background:$BG_POPUP;color:$MUTED;border:1px dashed $MUTED}
.vj-kpi-valor{font-size:1.45rem;font-weight:600;line-height:1.2}
.vj-hist{height:460px}
.vj-fila{border-bottom:1px solid $GRILLA}
""")


def css_tema() -> str:
    """CSS inyectado: fondo del body y popups de Quasar (.q-menu, .q-tooltip, .q-dialog, .q-notification), todo desde PALETA."""
    return _CSS_TEMA.substitute(PALETA)


HEAD_TEMA = '<meta name="color-scheme" content="dark"><meta name="theme-color" content="%s">' % BG_PAGINA

# --- Toggles de familias -------------------------------------------------------------------------------------
AYUDA_TOGGLE_SUFIJO = " Apagarla la quita del conteo y del gráfico; los límites de las zonas no se recalculan."
TOGGLES: dict[str, dict[str, Any]] = {
    "HVN": {"label": "HVN / Volume Profile", "familias": ("volumen",), "tipos": ("POC", "VAL", "VAH", "HVN"),
            "ayuda": "Zonas donde más volumen se negoció en el perfil de volumen de varios años (POC, VAL, VAH y HVN)."},
    "OB": {"label": "Order Blocks", "familias": ("ob",), "tipos": ("OB",),
           "ayuda": "Velas semanales que dieron origen a una subida fuerte y que el precio aún no ha perforado."},
    "AVWAP": {"label": "AVWAP", "familias": ("avwap",), "tipos": ("AVWAP",),
              "ayuda": "Costo medio (ponderado por volumen) de quienes compraron desde un mínimo clave."},
    "SWING": {"label": "Swing Lows macro", "familias": ("swing",), "tipos": ("SWING",),
              "ayuda": "Mínimos semanales importantes: puntos donde el precio ya rebotó antes."},
    "LOGREG": {"label": "LogReg", "familias": ("logreg",), "tipos": ("LOG",),
               "ayuda": "Bandas bajas de la tendencia de largo plazo (regresión logarítmica)."},
    "VALORACION": {"label": "DCF / P/E", "familias": ("valoracion",), "tipos": ("PE", "EVFCF", "DCF"),
                   "ayuda": "Precios de valoración (P/E, EV/FCF, DCF): líneas doradas que cuentan como una familia extra "
                            "donde caen dentro de una zona."},
}
# Ayuda contextual del panel lateral (tooltips legibles en oscuro; ver .q-tooltip en css_tema)
AYUDAS: dict[str, str] = {
    "ticker": "Símbolo bursátil (NVDA, BRK-B). El menú sugiere tu watchlist; también puedes escribir otro.",
    "capital": "Dinero POR TICKER a repartir entre sus mejores zonas (máximo 3). En Watchlist cada activo recibe el "
               "capital completo; no se reparte entre tickers. Cambiarlo no recalcula nada.",
    **{k: t["ayuda"] + AYUDA_TOGGLE_SUFIJO for k, t in TOGGLES.items()},
    "backtest": "Simula órdenes límite de los últimos años para medir Fill rate, Bounce y MAE. Es lento; sin él esas "
                "columnas salen n/d. Requiere Recalcular.",
    "sec": "Consulta SEC EDGAR (Form 4 y 13F) y STOCK Act. Sin esto no hay marcadores de insiders ni 13F y la "
           "cobertura de la Confiabilidad baja. Requiere Recalcular.",
    "valor_justo": "Dibuja en el gráfico de Análisis una línea con el valor justo consolidado de los modelos activos "
                   "(pestaña Valor Justo). Es una estimación con supuestos, no un precio objetivo.",
    "recalcular": "Descarga y calcula todo de nuevo para el ticker. Solo hace falta con un ticker nuevo o al cambiar "
                  "Backtest/SEC; capital y familias se actualizan solos.",
}
# Copia local de backtest_ranges.FAMILIAS (tipo -> familia); un test verifica que siga alineada.
TIPO_A_FAMILIA = {"POC": "volumen", "VAL": "volumen", "VAH": "volumen", "HVN": "volumen", "AVWAP": "avwap",
                  "OB": "ob", "SWING": "swing", "LOG": "logreg"}


def familias_activas(toggles: Iterable[str]) -> set[str]:
    """Nombres reales de familia habilitados por los toggles marcados."""
    out: set[str] = set()
    for t in toggles:
        out.update(TOGGLES.get(t, {}).get("familias", ()))
    return out


def _num(v: Any) -> float | None:
    """float finito o None (para mostrar 'n/d')."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --- Logica pura: valoracion, filtrado de zonas, resumen ---------------------------------------------------
def anclas_valoracion(res: dict) -> list[dict]:
    """Anclas P/E, EV/FCF y DCF (precio por unidad) desde ``res["valoracion"]``. Lista vacia si no hay."""
    val = res.get("valoracion") or {}
    out: list[dict] = []
    fuentes = (("pe", "pisos", "PE"), ("ev_fcf", "pisos", "EVFCF"), ("dcf", "precios_mos", "DCF"))
    for clave, sub, prefijo in fuentes:
        modelo = val.get(clave)
        if not isinstance(modelo, dict):
            continue
        for k, v in (modelo.get(sub) or {}).items():
            p = _num(v)
            if p and p > 0:
                out.append({"tipo": f"{prefijo}_{k}", "familia": "valoracion", "precio": p})
    return out


def filtrar_zonas(zonas: list[dict], activas: set[str], anclas_val: list[dict] | None = None,
                  capital: float | None = None) -> list[dict]:
    """Zonas visibles segun las familias activas (copias; no muta la entrada).

    Cada zona conserva solo las familias/tipos activos y se descarta si no queda ninguna. Los limites
    [r_inf, r_sup] no cambian (provienen del cluster completo). Se reordena como main.py (mas familias,
    mas anclas, mas cercana), se renumera N1..Nk y, si se pasa ``capital``, se reasigna con
    ``main.asignar_capital`` para que acciones/capital reflejen solo lo mostrado.
    """
    anclas_val = anclas_val if "valoracion" in activas else []
    lo_t, hi_t = TOLERANCIA_VALORACION
    out: list[dict] = []
    for z in zonas:
        fams = [f for f in z.get("familias", []) if f in activas]
        tipos = [t for t in z.get("tipos", []) if TIPO_A_FAMILIA.get(t, t) in activas]
        n_filtradas = len(fams) != len(z.get("familias", []))
        val_en_zona = [a["tipo"] for a in (anclas_val or []) if z["r_inf"] * lo_t <= a["precio"] <= z["r_sup"] * hi_t]
        if val_en_zona:
            fams.append("valoracion")
            tipos += sorted(val_en_zona)
            n_filtradas = True
        if not fams:
            continue
        nz = copy.deepcopy(z)
        nz.update(familias=fams, tipos=tipos, n_familias=len(fams), fuente="+".join(tipos))
        if n_filtradas:
            nz["n_anclas"] = len(tipos)
        out.append(nz)
    out.sort(key=lambda z: (-z["n_familias"], -z.get("n_anclas", 0), -z.get("dist_pct", 0.0)))
    for i, z in enumerate(out, 1):
        z["nivel"] = i
    if capital is not None:
        nucleo.asignar_capital(out, float(capital))
    return out


def precio_promedio_ponderado(zonas: list[dict]) -> float | None:
    """Precio promedio de entrada ponderado por capital: sum(capital) / sum(acciones).

    Cada tramo se ejecuta con orden limite en ``r_sup`` (mismo supuesto que main.asignar_capital y el backtest),
    asi que equivale al promedio de r_sup ponderado por el capital asignado (media armonica ponderada).
    """
    cap = sum(z.get("capital_asignado", 0.0) for z in zonas)
    acc = sum(z.get("acciones", 0.0) for z in zonas)
    return float(cap / acc) if cap > 0 and acc > 0 else None


def metricas_auditoria(backtest_resumen: dict | None) -> dict[str, float | None]:
    """Bounce / fill / MAE60 (en %) del backtest walk-forward del ticker; None si no existe.

    Son metricas del rango primario del ticker (src/validation/backtest_ranges.resumen), no por zona.
    """
    bt = backtest_resumen or {}
    def pct(clave: str) -> float | None:
        v = _num(bt.get(clave))
        return None if v is None else v * 100.0
    return {"bounce": pct("bounce"), "fill_rate": pct("fill_rate"), "mae": pct("mae60_mediana")}


def distancia_zona(z: dict, precio_actual: float | None = None) -> float | None:
    """Distancia % de ``r_sup`` (limite GTC) al precio actual: la UNICA referencia de distancia (igual que main.py).

    Con precio se recalcula con la formula compartida (``charts.dist_a_r_sup_pct``); sin precio se usa el
    ``dist_pct`` que ya trae la zona (calculado por main.py con esa misma referencia).
    """
    d = dist_a_r_sup_pct(z["r_sup"], precio_actual) if precio_actual else None
    return d if d is not None else _num(z.get("dist_pct"))


def ancho_zona_pct(z: dict, precio_actual: float | None = None) -> float | None:
    """Ancho (r_sup - r_inf) en % del precio actual (o del centro de la zona si no hay precio)."""
    try:
        ref = precio_actual if precio_actual and precio_actual > 0 else float(z.get("p_centro") or
                                                                             (z["r_inf"] + z["r_sup"]) / 2)
        return (float(z["r_sup"]) - float(z["r_inf"])) / ref * 100.0 if ref > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def filas_tabla(zonas: list[dict], backtest_resumen: dict | None = None,
                precio_actual: float | None = None) -> list[dict]:
    """Filas para la tabla de zonas (valores numericos para ordenar; None -> 'n/d' en la plantilla)."""
    aud = metricas_auditoria(backtest_resumen)
    filas = []
    for z in zonas:
        dist = distancia_zona(z, precio_actual)
        filas.append({
            "id": int(z.get("nivel", len(filas) + 1)), "nivel": f"N{z.get('nivel', len(filas) + 1)}",
            "r_inf": float(z["r_inf"]), "r_sup": float(z["r_sup"]),
            "p_centro": float(z.get("p_centro", (z["r_inf"] + z["r_sup"]) / 2)),
            "ancho_pct": ancho_zona_pct(z, precio_actual), "dist_pct": dist, "familias": ", ".join(z.get("familias", [])),
            "bounce": aud["bounce"], "fill_rate": aud["fill_rate"], "mae": aud["mae"],
            "capital": float(z.get("capital_asignado", 0.0)), "acciones": float(z.get("acciones", 0.0)),
            "entrada": float(z["r_sup"]) if z.get("acciones", 0.0) > 0 else None})
    return filas


def nombres_sin_dato(res: dict) -> list[str]:
    """Etiquetas legibles de los componentes de la Confiabilidad excluidos por falta de dato."""
    return [ETIQUETAS_COMPONENTES.get(k, k) for k in (res.get("componentes_sin_dato")
                                                      or nucleo.componentes_sin_dato(res.get("confiabilidad")))]


def texto_alcance_backtest(res: dict) -> str:
    """Pie de la tabla de zonas: de donde salen Bounce / Fill / MAE (rango primario N1, anios, fills) y su limite."""
    if not res.get("backtest_resumen"):
        return "Sin backtest: Bounce / Fill rate / MAE 60d no disponibles (n/d)."
    n_f, n_o = res.get("n_fills"), res.get("n_ordenes")
    muestra = "" if n_f is None else f" ({n_f} fills de {n_o if n_o is not None else 'n/d'} órdenes)"
    alcance = res.get("backtest_alcance") or nucleo.alcance_backtest(res.get("anios_backtest"))
    return (f"Bounce / Fill rate / MAE 60d: {alcance}{muestra}; se repiten igual en cada fila. "
            f"Entrada = límite en r_sup; precio prom. pond. = capital total / acciones totales.")


def resumen_tarjetas(res: dict, zonas: list[dict]) -> dict[str, Any]:
    """Datos de las tarjetas superiores: precio, scores, cobertura, avisos y precio promedio ponderado.

    ``avisos`` junta los de la UI (earnings, sin SEC, sin backtest) con ``res["avisos"]`` de ``main.analizar_ticker``
    (muestra pequena, gestores 13F que no respondieron, STOCK Act no disponible); ``avisos_datos`` es este ultimo grupo.
    """
    conf = res.get("confiabilidad") or {}
    earn = res.get("earnings") or {}
    avisos: list[str] = []
    if earn.get("alerta_activa"):
        avisos.append(earn.get("mensaje", "Earnings proximos"))
    if res.get("form4") is None and res.get("r13f") is None:
        avisos.append("Sin datos SEC (Form 4 / 13F): marcadores omitidos.")
    if not res.get("backtest_resumen"):
        avisos.append("Sin backtest: Bounce/Fill/MAE no disponibles (n/d).")
    avisos_datos = [str(a) for a in (res.get("avisos") or [])]
    return {"precio": _num(res.get("precio_actual")), "confluencia": _num(res.get("confluencia_score")),
            "confiabilidad": _num(conf.get("indice")), "cobertura": _num(conf.get("cobertura")),
            "sin_dato": nombres_sin_dato(res), "alcance_backtest": texto_alcance_backtest(res),
            "score_ranking": _num(res.get("score_ranking")), "earnings": earn.get("mensaje"),
            "earnings_alerta": bool(earn.get("alerta_activa")), "errores": list(res.get("errores") or []),
            "avisos": avisos + avisos_datos, "avisos_datos": avisos_datos,
            "precio_promedio": precio_promedio_ponderado(zonas),
            "capital_total": float(sum(z.get("capital_asignado", 0.0) for z in zonas)),
            "n_zonas": len(zonas)}


def filas_ranking(resultados: list[dict]) -> tuple[list[dict], list[str]]:
    """(filas ordenadas por ``main.ordenar_ranking``, tickers sin datos). Una fila por activo con precio."""
    validos = [r for r in resultados if r.get("precio_actual") is not None]
    fallidos = [r["ticker"] for r in resultados if r.get("precio_actual") is None]
    filas = []
    for i, r in enumerate(nucleo.ordenar_ranking(validos), 1):
        z = r["zonas"][0] if r.get("zonas") else None
        conf = r.get("confiabilidad") or {}
        cob = _num(conf.get("cobertura"))
        filas.append({
            "id": r["ticker"], "pos": i, "ticker": r["ticker"], "precio": _num(r.get("precio_actual")),
            "score": _num(r.get("score_ranking")), "confluencia": _num(r.get("confluencia_score")),
            "confiabilidad": _num(conf.get("indice")),
            "cobertura": None if cob is None else cob * 100.0, "sin_dato": ", ".join(nombres_sin_dato(r)),
            "mejor_zona": f"${z['r_inf']:.2f} - ${z['r_sup']:.2f}" if z else "n/d",
            "dist_pct": distancia_zona(z, r.get("precio_actual")) if z else None,
            "upside": _num((r.get("valor_justo") or {}).get("upside_pct")),
            "upside_nota": nota_soporte((r.get("valor_justo") or {}).get("incertidumbre"),
                                        (r.get("valor_justo") or {}).get("n_usados")) if r.get("valor_justo") else None})
    return filas, fallidos


def nota_ranking() -> str:
    """Nota visible de la pestana Watchlist: el score ordena, no predice; el peso de la confluencia se calcula al vuelo."""
    w = nucleo.peso_confluencia_ranking()
    return (f"Score de ranking = promedio de confluencia y confiabilidad: es una ayuda de organización, no una señal. "
            f"La confluencia domina el score (≈{_es(w * 100, 1)} %) y la evidencia propia no respalda que más familias "
            f"mejoren el retorno; scores con cobertura distinta no son comparables.")


def coberturas_distintas(filas: list[dict]) -> bool:
    """True si las filas del ranking tienen coberturas distintas (>= 1 punto de diferencia): sus scores no se comparan."""
    cobs = [f["cobertura"] for f in filas if f.get("cobertura") is not None]
    return bool(cobs) and max(cobs) - min(cobs) >= 1.0


# --- Grafico de Analisis (TradingView Lightweight Charts): funciones puras que devuelven estructuras JSON ---------
# El dibujo vive en static/charts/grafico_analisis.js; aqui solo se PREPARAN los datos (sin NiceGUI ni red).
ANIOS_VELAS_LOCAL, ANIOS_VELAS_NUBE = 10, 3      # velas enviadas al navegador (nube: menos memoria y ancho de banda)
BINS_PERFIL_LOCAL, BINS_PERFIL_NUBE = 70, 40     # densidad del Volume Profile
HVN_FRACCION = 0.7                               # bin HVN si su volumen >= 70 % del maximo (igual que charts.py)
MAX_ETIQUETA_ZONA = 82
VERSION_LW = "5.3.0"
VERSION_JS = "1"                                 # sube al cambiar grafico_analisis.js (rompe la cache del navegador)
SCRIPTS_GRAFICO = (f"{URL_STATIC}/vendor/lightweight-charts.standalone.production.js?v={VERSION_LW}",
                   f"{URL_STATIC}/charts/grafico_analisis.js?v={VERSION_JS}")
ATRIBUCION_LW = "Gráfico: TradingView Lightweight Charts™ (Apache-2.0)"
URL_TRADINGVIEW = "https://www.tradingview.com/"
MSG_SIN_DATOS = "Sin datos de mercado (calcula un ticker valido o revisa la red)"
COLUMNAS_OHLCV = {"Open", "High", "Low", "Close", "Volume"}


def anios_velas_defecto(nube: bool | None = None) -> int:
    """Anios de velas por defecto: 3 en la nube, 10 en local (``anios_velas`` de ``datos_grafico`` lo sobreescribe)."""
    return ANIOS_VELAS_NUBE if (userdata.en_nube() if nube is None else nube) else ANIOS_VELAS_LOCAL


def tema_grafico() -> dict[str, Any]:
    """Colores del grafico (todos de PALETA) para el JS: el archivo .js no lleva literales de color."""
    p = PALETA
    return {"bg_pagina": p["BG_PAGINA"], "bg_panel": p["BG_PANEL"], "bg_popup": p["BG_POPUP"],
            "bg_etiqueta": p["BG_ETIQUETA"], "borde": p["BORDE"], "grilla": p["GRILLA"], "txt": p["TXT"],
            "muted": p["MUTED"], "blanco": p["BLANCO"], "vela_alza": p["VELA_ALZA"], "vela_baja": p["VELA_BAJA"],
            "color_13f": p["COLOR_13F"], "color_form4": p["COLOR_FORM4"], "color_valoracion": p["COLOR_VALORACION"],
            "color_precio": p["COLOR_PRECIO"], "color_hvn": p["COLOR_HVN"], "color_barra_vp": p["COLOR_BARRA_VP"],
            "color_fair_value": p["COLOR_FAIR_VALUE"], "acento_txt": p["ACENTO_TXT"],
            "niveles": {str(k): v for k, v in COLORES_NIVEL.items()}}


def _r(v: Any, dec: int = 4) -> float | None:
    """Float redondeado y finito (None si NaN/inf/no numerico): el JSON no admite NaN."""
    x = _num(v)
    return None if x is None else round(x, dec)


def _naive_idx(idx: Any) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(idx)
    return i.tz_localize(None) if i.tz is not None else i


def _segundos(idx: Any) -> list[int]:
    """Indice de fechas -> segundos UNIX (UTC, sin zona horaria); el grafico usa tiempo de negocio por barra."""
    i = _naive_idx(idx)
    return [int(x) for x in ((i - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))]


def df_valido(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and COLUMNAS_OHLCV.issubset(df.columns)


def limpiar_velas(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV con indice sin zona horaria, ordenado, sin tiempos duplicados ni filas con OHLC no finito o <= 0."""
    d = df.copy()
    d.index = _naive_idx(d.index)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    ohlc = d[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    ok = ohlc.map(lambda v: math.isfinite(v) and v > 0).all(axis=1)
    return d[ok.values]


def recortar_velas(df: pd.DataFrame, anios: int | None) -> pd.DataFrame:
    """Ultimos ``anios`` anios del historial (None/0 = todo)."""
    if not anios or df.empty:
        return df
    return df[df.index >= df.index[-1] - pd.DateOffset(years=int(anios))]


def velas_json(d: pd.DataFrame) -> dict[str, list]:
    """Velas OHLC en columnas (``t`` en segundos UNIX): JSON compacto y rapido de recorrer (~2500 filas = ~100 kB)."""
    return {"t": _segundos(d.index), **{k: [round(float(x), 4) for x in d[c].values]
                                       for k, c in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"))}}


def sma_json(d: pd.DataFrame, n: int = 200) -> list[float | None] | None:
    """SMA(n) alineada con las velas (None en las primeras n-1); None si no hay n velas."""
    if len(d) < n:
        return None
    return [_r(v, 2) for v in d["Close"].rolling(n).mean().values]


def perfil_json(df: pd.DataFrame, n_bins: int = BINS_PERFIL_LOCAL) -> dict | None:
    """Volume Profile: ``p`` = centros de bin (``lo + paso*(i+0.5)``), ``v`` = volumen suavizado, ``paso`` = ancho del bin,
    ``hvn`` = indices de los bins >= 70 % del maximo. None si no hay perfil calculable."""
    centros, vol, paso = perfil_histograma(df, n_bins=n_bins)
    if not len(centros) or not paso:
        return None
    vmax = float(vol.max())
    return {"p": [round(float(c), 4) for c in centros], "v": [int(round(float(v))) for v in vol],
            "paso": round(float(paso), 6),
            "hvn": [i for i, v in enumerate(vol) if vmax > 0 and v >= HVN_FRACCION * vmax]}


def _barra_en_o_antes(barras: list[int], t: int) -> int:
    """Segundos de la ultima barra <= t (la primera si t es anterior a todas)."""
    import bisect
    i = bisect.bisect_right(barras, t) - 1
    return barras[max(i, 0)]


def _t_seg(ts: Any) -> int:
    t = pd.Timestamp(ts)
    t = t.tz_localize(None) if t.tzinfo else t
    return int((t.normalize() - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))


def marcadores_form4_json(form4: pd.DataFrame | None, d: pd.DataFrame, barras: list[int] | None = None) -> list[dict]:
    """Compras P de insiders: una marca por barra (varias compras el mismo dia se agrupan en ``n``), ordenadas por fecha y
    sin tiempos duplicados. ``t`` = barra de negocio, ``p`` = precio medio, ``f`` = fecha ISO, ``n`` = compras."""
    f4 = marcadores_form4(form4, d)
    if f4.empty:
        return []
    barras = barras or _segundos(d.index)
    grupos: dict[int, list[float]] = {}
    for fecha, precio in zip(f4["fecha"], f4["precio"]):
        p = _num(precio)
        if p is not None and p > 0:
            grupos.setdefault(_barra_en_o_antes(barras, _t_seg(fecha)), []).append(p)
    return [{"t": t, "p": round(sum(ps) / len(ps), 4), "f": pd.Timestamp(t, unit="s").strftime("%Y-%m-%d"), "n": len(ps)}
            for t, ps in sorted(grupos.items())]


def marcadores_13f_json(r13f: pd.DataFrame | None, d: pd.DataFrame, barras: list[int] | None = None) -> list[dict]:
    """Trimestres de acumulacion 13F: ``t`` = ultima barra <= cierre del trimestre, ``t0`` = barra de inicio (banda de 3
    meses), ``p`` = cierre de esa barra, ``f`` = fin de trimestre ISO. Ordenados y sin ``t`` duplicado."""
    qs = trimestres_acumulacion(r13f, d)
    if not qs:
        return []
    barras = barras or _segundos(d.index)
    cierres = dict(zip(barras, (float(x) for x in d["Close"].values)))
    out: dict[int, dict] = {}
    for q in qs:
        t = _barra_en_o_antes(barras, _t_seg(q))
        t0 = _barra_en_o_antes(barras, _t_seg(q - pd.DateOffset(months=3)))
        out.setdefault(t, {"t": t, "t0": t0, "p": round(cierres[t], 4), "f": pd.Timestamp(q).strftime("%Y-%m-%d")})
    return [out[t] for t in sorted(out)]


def datos_grafico(df: pd.DataFrame | None, ticker: str = "-", form4: pd.DataFrame | None = None,
                  r13f: pd.DataFrame | None = None, anios_velas: int | None = None, nube: bool | None = None) -> dict:
    """Parte PESADA del grafico (se reenvia solo al cambiar de ticker/datos): velas, SMA 200, Volume Profile y marcadores.

    ``anios_velas`` es el unico parametro de cuanto historial se dibuja (None = 10 en local, 3 en la nube); el Volume
    Profile se calcula sobre los ultimos 5 anios del historial completo, con menos bins en la nube. Sin datos validos:
    ``{"vacio": mensaje}``.
    """
    if not df_valido(df):
        return {"vacio": MSG_SIN_DATOS}
    completo = limpiar_velas(df)
    if completo.empty:
        return {"vacio": MSG_SIN_DATOS}
    nube = userdata.en_nube() if nube is None else nube
    anios = anios_velas if anios_velas is not None else anios_velas_defecto(nube)
    d = recortar_velas(completo, anios)
    sma = sma_json(d)
    velas = velas_json(d)
    barras = velas["t"]
    return {"ticker": ticker, "anios": anios, **velas, "tiene_sma": sma is not None, "sma": sma,
            "vp": perfil_json(completo, BINS_PERFIL_NUBE if nube else BINS_PERFIL_LOCAL),
            "form4": marcadores_form4_json(form4, d, barras), "q13": marcadores_13f_json(r13f, d, barras),
            "rango": [round(float(d["Low"].min()), 4), round(float(d["High"].max()), 4)]}


def etiqueta_zona(z: dict, precio: float) -> str:
    """``N1: $lo-$hi | dist. a r_sup +x.x% | fuente`` (dist. siempre al limite r_sup; maximo 82 caracteres)."""
    lo, hi = float(z["r_inf"]), float(z["r_sup"])
    dist = dist_a_r_sup_pct(hi, precio)
    txt = (f"N{int(z.get('nivel', 1))}: ${lo:.2f}-${hi:.2f} | {ETIQUETA_DIST_RSUP} "
           f"{'n/d' if dist is None else f'{dist:+.1f}%'} | {z.get('fuente', '')}")
    return txt if len(txt) <= MAX_ETIQUETA_ZONA else txt[:MAX_ETIQUETA_ZONA - 3] + "..."


def bandas_zonas(zonas: list[dict] | None, precio: float, nivel_seleccionado: int | None = None) -> list[dict]:
    """Zonas N1-N4 -> bandas ``{nivel, lo, hi, sel, etiqueta}`` (``sel`` = fila seleccionada en la tabla). Descarta las
    zonas con limites no numericos o invertidos."""
    out = []
    for z in zonas or []:
        lo, hi = _num(z.get("r_inf")), _num(z.get("r_sup"))
        if lo is None or hi is None or hi < lo:
            continue
        nivel = int(z.get("nivel", 1))
        out.append({"nivel": nivel, "lo": round(lo, 4), "hi": round(hi, 4), "sel": nivel_seleccionado == nivel,
                    "etiqueta": etiqueta_zona(z, precio)})
    return out


def anclas_visibles(anclas_val: list[dict] | None, rango: tuple[float, float] | list[float] | None) -> list[dict]:
    """Anclas de valoracion dentro de [0.95 x minimo, 1.04 x maximo] del historial dibujado (igual criterio que antes)."""
    if not anclas_val or not rango:
        return []
    out = []
    for a in anclas_val:
        p = _num(a.get("precio"))
        if p is not None and rango[0] * 0.95 <= p <= rango[1] * 1.04:
            out.append({"tipo": str(a.get("tipo", "")), "precio": round(p, 4)})
    return out


def valor_justo_medio(vj: dict | None) -> dict | None:
    """UNICO punto de acceso al valor justo que se dibuja en Analisis: el del Escenario MEDIO.

    Hoy el agregado trae ``fair_value`` directo; cuando el motor devuelva escenarios (``{"medio": {...}, "pesimista":
    {...}}``) se toma ``medio`` (heredando ``incertidumbre``/``n_usados`` del padre si faltan). Adaptar aqui y solo aqui.
    """
    if not isinstance(vj, dict):
        return None
    medio = vj.get("medio")
    if isinstance(medio, dict):
        return {**{k: vj.get(k) for k in ("incertidumbre", "n_usados")}, **medio}
    return vj


def linea_valor_justo(vj: dict | None, precio: float) -> dict | None:
    """Linea del valor justo medio: ``{precio, etiqueta, fuera_de_rango}``; None sin valor justo valido.

    Etiqueta ``Fair Value Medio: $XXX (+XX% Upside)`` (+ `` · incert. Alta`` / `` · n=2`` con soporte debil). Fuera de
    [0.3x, 3x] el precio no se dibuja (no deforma la escala): solo el rotulo con la flecha y ``fuera_de_rango``.
    """
    m = valor_justo_medio(vj)
    v = _num((m or {}).get("fair_value"))
    if v is None or v <= 0:
        return None
    etiqueta = etiqueta_fair_value(v, m.get("upside_pct"), nota_soporte(m.get("incertidumbre"), m.get("n_usados")),
                                   nombre="Fair Value Medio")
    fuera = None
    if not (VJ_RANGO_EJE[0] * precio <= v <= VJ_RANGO_EJE[1] * precio):
        fuera = "arriba" if v > precio else "abajo"
        etiqueta += f" {'▲' if fuera == 'arriba' else '▼'} fuera de rango del eje"
    return {"precio": round(v, 4), "etiqueta": etiqueta, "fuera_de_rango": fuera}


def overlays_grafico(datos: dict | None, ticker: str, zonas: list[dict] | None, precio_actual: float | None,
                     anclas_val: list[dict] | None = None, nivel_seleccionado: int | None = None,
                     subtitulo: str | None = None, valor_justo: dict | None = None) -> dict:
    """Parte LIGERA del grafico (cambia con toggles, seleccion de fila, capital y valor justo): bandas, anclas, precio,
    linea de valor justo medio y titulo. ``extras`` = precios que la escala debe poder incluir (zonas, valor justo, precio,
    anclas). Con ``datos`` vacio/None devuelve solo el titulo."""
    base = f"{ticker} - Mapa de niveles institucionales" if ticker and ticker != "-" else "Mapa de niveles institucionales"
    titulo = base + (f"  |  {subtitulo}" if subtitulo else "")
    if not datos or datos.get("vacio") or not datos.get("rango"):
        return {"titulo": titulo, "zonas": [], "anclas": [], "precio": None, "vj": None, "extras": []}
    px = _num(precio_actual)
    if px is None:
        px = float(datos["c"][-1])
    bandas = bandas_zonas(zonas, px, nivel_seleccionado)
    anclas = anclas_visibles(anclas_val, datos["rango"])
    vj = linea_valor_justo(valor_justo, px)
    extras = [v for b in bandas for v in (b["lo"], b["hi"])] + [px] + [a["precio"] for a in anclas]
    if vj and not vj["fuera_de_rango"]:
        extras.append(vj["precio"])
    return {"titulo": titulo, "zonas": bandas, "anclas": anclas, "precio": round(px, 4), "vj": vj, "extras": extras}


def huella_datos(ticker: str | None, df: Any, form4: Any, r13f: Any, anios: int | None) -> tuple:
    """Clave barata de los datos pesados del grafico: si no cambia, no se reenvian las velas al navegador."""
    if not df_valido(df):
        return (ticker, None)

    def n(x: Any) -> int:
        return len(x) if isinstance(x, pd.DataFrame) else -1
    return (ticker, len(df), str(df.index[0]), str(df.index[-1]), round(float(df["Close"].iloc[-1]), 4),
            round(float(pd.to_numeric(df["Close"], errors="coerce").sum()), 2), n(form4), n(r13f), anios)


def js_actualizar(html_id: str, tema: dict | None = None, datos: dict | None = None, overlays: dict | None = None) -> str:
    """Sentencia JS ``NivelesGrafico.actualizar(id, {...})`` con el payload en JSON compacto (sin NaN)."""
    msg = {k: v for k, v in (("tema", tema), ("datos", datos), ("overlays", overlays)) if v is not None}
    return (f"NivelesGrafico.actualizar({json.dumps(html_id)},"
            f"{json.dumps(msg, separators=(',', ':'), allow_nan=False, ensure_ascii=False)});")


# --- Valor justo: logica pura (sin NiceGUI ni red) ---------------------------------------------------------
VJ_DEBOUNCE_S = 0.25   # espera tras mover un slider antes de recalcular (evita recalcular en cada pixel)
VJ_SLIDERS = {"wacc": (4.0, 16.0, 0.1), "g": (0.0, 5.0, 0.1), "mos": (0.0, 60.0, 1.0)}   # (min, max, paso) en %
ESTADOS_MODELO = ("activo", "outlier", "desactivado", "no_aplica")
NOMBRES_ESTADO = {"activo": "Activo", "outlier": "Outlier", "desactivado": "Desactivado por usuario",
                  "no_aplica": "No aplica"}
# Diagnostico de valoracion (tarjeta 4). Umbral NOMBRADO y alineado con las bandas del grafico historico: el precio esta a
# +10 % o mas por encima del Valor Justo Medio (banda +10 %) => Sobrevalorada; a -10 % o mas por debajo (banda -10 %) =>
# Infravalorada; entre ambas => En linea. Es una etiqueta de lectura rapida, no una recomendacion.
UMBRAL_DIAGNOSTICO = 0.10
DIAGNOSTICOS = {"infra": "Infravalorada", "linea": "En línea", "sobre": "Sobrevalorada"}
# Nombres de las series del grafico historico (el tooltip JS las busca por nombre)
SERIE_PRECIO, SERIE_VJ = "Precio de mercado", "Valor justo medio"
SERIE_PROY, SERIE_PISO = "Proyección 12M (ilustrativa)", "Piso fundamental (Pesimista)"
BANDAS_ORDEN = ("-30%", "-20%", "-10%", "0%", "+10%", "+20%", "+30%")   # de abajo arriba; "0%" = el propio valor justo
MAX_SERIES_LOCAL, MAX_SERIES_NUBE = 8, 1          # series historicas retenidas por huella (nube: solo la ultima)
MAX_TICKERS_ESCANEO_NUBE = 5                      # tickers por escaneo en la nube (visor ligero; local sin limite)
PERIODO_LOCAL, PERIODO_NUBE = "10y", "3y"         # historial que se pide a yfinance


def periodo_consulta(nube: bool | None = None) -> str:
    """``periodo`` de ``analizar_ticker``: 10y en local (sin restricciones), 3y en la nube (visor ligero)."""
    return PERIODO_NUBE if (userdata.en_nube() if nube is None else nube) else PERIODO_LOCAL


def liberar_memoria() -> None:
    """``gc.collect()`` en la nube tras cada consulta o recalculo (512 MB); en local no hace falta."""
    if userdata.en_nube():
        gc.collect()


def con_alfa(color: str, alfa: float) -> str:
    """``#rrggbb`` + opacidad -> ``rgba(r,g,b,a)`` (las bandas del grafico derivan de PALETA, sin literales aparte)."""
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa:.2f})"


def nota_soporte(incertidumbre: str | None, n_usados: int | None) -> str | None:
    """Nota de soporte debil del consolidado: ``n=2`` con menos de ``MIN_MODELOS_CONFIABLE`` modelos usados, ``incert. Alta``
    si la incertidumbre es Alta; None si el soporte es suficiente (o no se conoce)."""
    if n_usados is not None and n_usados < fv.MIN_MODELOS_CONFIABLE:
        return f"n={n_usados}"
    return "incert. Alta" if incertidumbre == "Alta" else None


def etiqueta_fair_value(valor: float, upside_pct: float | None, nota: str | None = None,
                        nombre: str = "Fair Value") -> str:
    """Etiqueta de la linea: ``Fair Value: $XXX (+XX% Upside)`` (o ``-XX%``) y, con soporte debil, `` · incert. Alta`` (o
    `` · n=2``); sin precio actual no hay porcentaje. ``nombre`` = ``Fair Value Medio`` en el grafico de Analisis."""
    base = (f"{nombre}: ${valor:,.2f}" if _num(upside_pct) is None
            else f"{nombre}: ${valor:,.2f} ({upside_pct:+.0f}% Upside)")
    return base + (f" · {nota}" if nota else "")


def diagnostico_valoracion(precio: float | None, valor_justo: float | None, incertidumbre: str | None = None,
                           n_usados: int | None = None) -> dict:
    """Diagnostico de la tarjeta 4: ``{clave, texto, premium_pct, debil}``.

    ``premium_pct`` = precio / valor justo - 1 (en %): >= ``UMBRAL_DIAGNOSTICO`` => "Sobrevalorada", <= -``UMBRAL_DIAGNOSTICO``
    => "Infravalorada", si no "En línea". ``debil`` = nota de soporte (``n=2`` / ``incert. Alta``) cuando el consolidado
    se apoya en pocos modelos o los modelos divergen: la tarjeta se atenua y la muestra. Sin datos: clave None."""
    px, v = _num(precio), _num(valor_justo)
    if px is None or v is None or v <= 0 or px <= 0:
        return {"clave": None, "texto": "n/d", "premium_pct": None, "debil": None}
    prem = round(px / v - 1.0, 9)      # sin ruido de coma flotante en el borde exacto (+-10 %)
    clave = "sobre" if prem >= UMBRAL_DIAGNOSTICO else "infra" if prem <= -UMBRAL_DIAGNOSTICO else "linea"
    return {"clave": clave, "texto": DIAGNOSTICOS[clave], "premium_pct": prem * 100.0,
            "debil": nota_soporte(incertidumbre, n_usados)}


def estado_modelo(modelo: dict, agregado: dict, activo_usuario: bool = True) -> tuple[str, str]:
    """(clave, texto) del estado de un modelo: no_aplica > desactivado > outlier > activo (el texto lleva el motivo).

    Acepta el modelo de ``calcular_escenarios()["modelos"]`` (``aplicable``/``motivo`` + ``aplicable_medio``/``_pesimista``)
    o el de un escenario (``motivo_exclusion``). ``agregado`` = el del Escenario Medio (outliers). Si el modelo solo aplica
    en el Pesimista (p. ej. EPV con ingresos que crecen mas de 20 %), sigue "activo" y el texto lo aclara."""
    if not modelo.get("aplicable"):
        return "no_aplica", f"No aplica: {modelo.get('motivo_exclusion') or modelo.get('motivo') or 'sin datos'}"
    if not activo_usuario:
        return "desactivado", NOMBRES_ESTADO["desactivado"]
    if modelo["id"] in (agregado or {}).get("modelos_outlier", []):
        return "outlier", "Outlier: excluido del promedio (filtro IQR)"
    if modelo.get("aplicable_medio") is False and modelo.get("aplicable_pesimista"):
        return "activo", f"Activo solo en el Pesimista (Medio: {modelo.get('motivo_medio') or 'no aplica'})"
    return "activo", NOMBRES_ESTADO["activo"]


def ids_por_familia(familia: str) -> list[str]:
    return [i for i, _, f in fv.MODELOS if f == familia]


def valor_justo_de_entradas(entradas: Any, params: "fv.ParametrosValoracion | None" = None,
                            activos: dict[str, bool] | None = None, precio: float | None = None) -> dict | None:
    """Agregado del Escenario MEDIO (con ``modelos``) desde las entradas livianas; None si no hay entradas. Sin red."""
    if not entradas:
        return None
    e = entradas if isinstance(entradas, fv.EntradasValoracion) else fv.EntradasValoracion.desde_dict(entradas)
    p = params or fv.ParametrosValoracion()
    ids = None if activos is None else [i for i, v in activos.items() if v]
    return fv.valor_justo(e, p, ids, precio if precio is not None else e.precio)


def resumen_valor_justo(res: dict, params: "fv.ParametrosValoracion | None" = None) -> dict | None:
    """Resumen liviano (fair_value Medio, upside, incertidumbre) para el ranking de la Watchlist; None sin entradas."""
    agg = valor_justo_de_entradas(res.get("entradas_valoracion"), params, None, res.get("precio_actual"))
    if agg is None:
        return None
    return {k: agg.get(k) for k in ("fair_value", "upside_pct", "incertidumbre", "n_usados", "cv")}


SERIE_RATIO_MAX = 10.0   # un punto historico con valor justo > 10x (o < 1/10) del precio de ese dia se trata como dato corrupto


def sanear_serie(serie: dict | None) -> tuple[dict | None, int]:
    """(serie sin puntos absurdos, cuantos se descartaron). Un valor justo mas de ``SERIE_RATIO_MAX`` veces mayor (o menor)
    que el precio del mismo dia casi siempre es un dato por accion erroneo en un reporte antiguo (p. ej. KO 2017: ~24 M USD):
    deformaria toda la escala. Esos puntos pasan a None (la curva y sus bandas se interrumpen) y se avisa en las notas; el
    motor (``fair_value``) no se toca. La serie original no se modifica."""
    if not serie or not serie.get("fechas"):
        return serie, 0
    vj, px = serie.get("valor_justo") or [], serie.get("precio") or []
    malos = {i for i, (v, p) in enumerate(zip(vj, px))
             if v is not None and p and (v > SERIE_RATIO_MAX * p or v < p / SERIE_RATIO_MAX)}
    if not malos:
        return serie, 0

    def limpia(lista: list) -> list:
        return [None if i in malos else v for i, v in enumerate(lista)]
    out = {**serie, "valor_justo": limpia(vj), "bandas": {k: limpia(v) for k, v in (serie.get("bandas") or {}).items()},
           "notas": list(serie.get("notas") or []) + [
               f"{len(malos)} punto(s) de la serie se omiten del gráfico: el valor justo difería más de {SERIE_RATIO_MAX:.0f}x "
               "del precio de ese día (probable dato por acción erróneo en un reporte antiguo de SEC EDGAR)."]}
    return out, len(malos)


def calcular_serie(entradas: Any, df: Any, params: "fv.ParametrosValoracion", activos: list[str], anios: int | None) -> dict:
    """Serie historica del Valor Justo Medio (``fv.serie_valor_justo_historica``) tolerante a fallos: nunca lanza.

    Pensada para ``asyncio.to_thread``: frio ~0.25 s, caliente (cache SQLite local) < 20 ms."""
    try:
        return fv.serie_valor_justo_historica(entradas, df, params, modelos_activos=activos, anios=anios)
    except Exception as e:  # noqa: BLE001
        return {"fechas": [], "valor_justo": [], "precio": [], "bandas": {}, "proyeccion": {}, "ancla": None,
                "notas": [f"No se pudo construir la serie histórica: {type(e).__name__}: {e}"], "error": True}


# --- Opciones ECharts (funciones puras: JSON sin NiceGUI; los colores salen de PALETA) -----------------------------------
def _estilo_base() -> dict:
    return {"backgroundColor": BG_PANEL, "textStyle": {"color": TXT, "fontFamily": "Roboto, 'Segoe UI', sans-serif"},
            "animation": False}


def _js_tooltip_historico() -> str:
    """Formatter JS del tooltip: Fecha, Precio, Valor Justo Medio y % Sobre/Infravalorado (precio vs valor justo EN ESA fecha)."""
    cfg = {"px": SERIE_PRECIO, "vj": SERIE_VJ, "pr": SERIE_PROY, "ok": OK_TXT, "mal": ERROR_TXT, "muted": MUTED}
    return ("""(ps) => {
  const C = __C__;
  const pick = (n) => ps.find((p) => p.seriesName === n);
  const val = (p) => (p && Array.isArray(p.value) && p.value[1] != null) ? Number(p.value[1]) : null;
  const usd = (v) => v == null ? 'n/d' : '$' + v.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
  const px = val(pick(C.px)), vj = val(pick(C.vj)), pr = val(pick(C.pr));
  const ref = vj != null ? vj : pr;
  const fecha = String(ps[0].value[0]).slice(0, 10);
  let h = '<b>Fecha:</b> ' + fecha + '<br/><b>Precio:</b> ' + usd(px)
        + '<br/><b>Valor Justo Medio' + (vj == null && pr != null ? ' (proyección ilustrativa)' : '') + ':</b> ' + usd(ref);
  if (px != null && ref != null && ref > 0) {
    const d = (px / ref - 1) * 100;
    const t = d > 0 ? 'Sobrevalorado' : (d < 0 ? 'Infravalorado' : 'En línea');
    const c = d > 0 ? C.mal : (d < 0 ? C.ok : C.muted);
    h += '<br/><b>% Sobre/Infravalorado:</b> <span style="color:' + c + ';font-weight:600">' + (d > 0 ? '+' : '')
       + d.toFixed(1) + ' % (' + t + ')</span>';
  }
  return h;
}""").replace("__C__", json.dumps(cfg, ensure_ascii=False))


def _degradado(color: str, a_arriba: float, a_abajo: float) -> dict:
    return {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
            "colorStops": [{"offset": 0, "color": con_alfa(color, a_arriba)},
                           {"offset": 1, "color": con_alfa(color, a_abajo)}]}


def opciones_historico(serie: dict | None, esc: dict | None = None, mensaje: str | None = None) -> dict:
    """Opciones ECharts del grafico historico estilo GuruFocus.

    Series (SIEMPRE las mismas 10, para que ``set_option`` fusione sin recrear el grafico): 7 lineas apiladas
    ``-30% .. +30%`` (las diferencias entre bandas forman las areas con degradado: magenta para -10/-20/-30 %, cian para
    +10/+20/+30 %), la curva del valor justo Medio (continua), la proyeccion 12M (punteada, ILUSTRATIVA) y el precio de
    mercado. El Escenario Pesimista actual es una linea horizontal (``markLine``). Con ``serie`` vacia/None solo muestra
    ``mensaje``. Las etiquetas porcentuales van en el borde derecho (``endLabel``)."""
    serie = serie or {}
    fechas = list(serie.get("fechas") or [])
    proy = serie.get("proyeccion") or {}
    pf = list(proy.get("fechas") or [])
    solape = bool(pf and fechas and pf[0] == fechas[-1])
    union = fechas + (pf[1:] if solape else pf)
    n_h, n_u = len(fechas), len(fechas) + (len(pf) - (1 if solape else 0))
    desfase = n_h - 1 if solape else n_h        # indice de la union donde empieza la proyeccion

    def r2(v: Any) -> float | None:
        v = _num(v)
        return None if v is None else round(v, 2)

    def completar(hist: list | None, proj: list | None) -> list[float | None]:
        out: list[float | None] = [r2(v) for v in (hist or [])] + [None] * (n_u - n_h)
        for j, v in enumerate(proj or []):
            k = desfase + j
            if n_h <= k < n_u:
                out[k] = r2(v)
        return out

    def par(vals: list) -> list:
        return [[f, v] for f, v in zip(union, vals)]

    bandas, bandas_p = serie.get("bandas") or {}, proy.get("bandas") or {}
    niveles: dict[str, list[float | None]] = {}
    for k in BANDAS_ORDEN:
        if k == "0%":
            niveles[k] = completar(serie.get("valor_justo"), proy.get("valor_justo"))
        else:
            niveles[k] = completar(bandas.get(k), bandas_p.get(k))
    colores = {"-30%": VJ_MAGENTA, "-20%": VJ_MAGENTA, "-10%": VJ_MAGENTA, "0%": VJ_MAGENTA,
               "+10%": VJ_CIAN, "+20%": VJ_CIAN, "+30%": VJ_CIAN}
    alfas = {"-20%": (0.28, 0.50), "-10%": (0.16, 0.32), "0%": (0.05, 0.16),
             "+10%": (0.16, 0.05), "+20%": (0.32, 0.16), "+30%": (0.50, 0.28)}
    series: list[dict] = []
    previo: list[float | None] | None = None
    for k in BANDAS_ORDEN:
        abs_ = niveles[k]
        if previo is None:
            datos = abs_
        else:
            datos = [None if (a is None or b is None) else round(a - b, 4) for a, b in zip(abs_, previo)]
        previo = abs_
        s: dict[str, Any] = {"name": k, "type": "line", "stack": "bandas", "symbol": "none", "smooth": False, "silent": True,
                             "z": 1, "emphasis": {"disabled": True}, "data": par(datos) if n_u else [],
                             "lineStyle": {"width": 0 if k == "0%" else 1, "color": con_alfa(colores[k], 0.85)}}
        if k != "-30%":
            s["areaStyle"] = {"color": _degradado(colores[k], *alfas[k])}
        if k != "0%":
            s["endLabel"] = {"show": True, "formatter": k, "color": colores[k], "fontSize": 11, "distance": 4}
        series.append(s)
    piso = _num(((esc or {}).get("pesimista") or {}).get("fair_value"))
    marca = {"symbol": "none", "silent": True, "animation": False, "data": []}
    if piso is not None:
        marca["data"] = [{"yAxis": round(piso, 2)}]
        marca["lineStyle"] = {"color": VJ_PESIMISTA, "type": "dashed", "width": 1.5}
        marca["label"] = {"show": True, "formatter": f"{SERIE_PISO} ${piso:,.2f}", "color": VJ_PESIMISTA,
                          "position": "insideStartTop", "fontSize": 11}
    vj_h = completar(serie.get("valor_justo"), None)
    vj_p = [None] * desfase + [r2(v) for v in (proy.get("valor_justo") or [])]
    vj_p = (vj_p + [None] * n_u)[:n_u] if n_u else []
    px_h = completar(serie.get("precio"), None)
    series += [
        {"name": SERIE_VJ, "type": "line", "symbol": "none", "smooth": False, "z": 10, "data": par(vj_h) if n_u else [],
         "lineStyle": {"width": 2.5, "color": COLOR_FAIR_VALUE}, "itemStyle": {"color": COLOR_FAIR_VALUE},
         "markLine": marca},
        {"name": SERIE_PROY, "type": "line", "symbol": "none", "smooth": False, "z": 10, "data": par(vj_p) if n_u else [],
         "lineStyle": {"width": 3, "type": "dotted", "color": COLOR_FAIR_VALUE}, "itemStyle": {"color": COLOR_FAIR_VALUE}},
        {"name": SERIE_PRECIO, "type": "line", "symbol": "none", "smooth": False, "z": 9, "data": par(px_h) if n_u else [],
         "lineStyle": {"width": 1.6, "color": COLOR_PRECIO}, "itemStyle": {"color": COLOR_PRECIO}}]
    eje = {"axisLine": {"lineStyle": {"color": BORDE}}, "axisLabel": {"color": MUTED}}
    opc = {**_estilo_base(),
           "title": {"text": mensaje or "", "left": "center", "top": "middle",
                     "textStyle": {"color": MUTED, "fontSize": 14, "fontWeight": "normal"}},
           "grid": {"left": 64, "right": 78, "top": 44, "bottom": 74},
           "legend": {"data": [SERIE_VJ, SERIE_PROY, SERIE_PRECIO], "top": 6, "textStyle": {"color": TXT, "fontSize": 12},
                      "icon": "roundRect"},
           "tooltip": {"trigger": "axis", "backgroundColor": BG_POPUP, "borderColor": BORDE, "textStyle": {"color": TXT, "fontSize": 12},
                       "axisPointer": {"type": "line", "lineStyle": {"color": MUTED}}, ":formatter": _js_tooltip_historico()},
           "xAxis": {"type": "time", **eje, "splitLine": {"show": False}},
           "yAxis": {"type": "value", "scale": True, **eje, "splitLine": {"lineStyle": {"color": GRILLA}},
                     "axisLabel": {"color": MUTED, ":formatter": "(v) => '$' + Math.round(v).toLocaleString('en-US')"}},
           "dataZoom": [{"type": "inside", "xAxisIndex": 0, "filterMode": "filter"},
                        {"type": "slider", "xAxisIndex": 0, "height": 22, "bottom": 10, "borderColor": BORDE,
                         "backgroundColor": BG_PAGINA, "fillerColor": con_alfa(ACENTO_TXT, 0.18),
                         "textStyle": {"color": MUTED}, "handleStyle": {"color": ACENTO_TXT, "borderColor": ACENTO_TXT},
                         "dataBackground": {"lineStyle": {"color": BORDE}, "areaStyle": {"color": con_alfa(BORDE, 0.35)}}}],
           "series": series}
    return opc


def opciones_barras(esc: dict | None, precio: float | None = None, activos: dict[str, bool] | None = None,
                    mensaje: str | None = None) -> dict:
    """Opciones ECharts de las barras horizontales por modelo: DOS series (Escenario Medio y Pesimista) por modelo, lineas
    verticales del valor justo Medio, del piso Pesimista y del precio actual. Los desactivados salen atenuados y gris, los
    outliers en amarillo, los que no aplican sin barra (la etiqueta lo dice) y los de tope activo llevan ``· tope``."""
    esc = esc or {}
    modelos = list(esc.get("modelos") or [])
    med, pes = esc.get("medio") or {}, esc.get("pesimista") or {}
    activos = activos or {}
    px = _num(precio)
    fm, fp = _num(med.get("fair_value")), _num(pes.get("fair_value"))
    valores = [v for v in (px, fm, fp) if v]
    for m in modelos:
        for k, agg in (("precio_medio", med), ("precio_pesimista", pes)):
            v = _num(m.get(k))
            if v and m["id"] not in (agg.get("modelos_outlier") or []):
                valores.append(v)
    x_max = max(valores) * 1.25 if valores else 1.0
    nombres: list[str] = []
    datos = {"medio": [], "pesimista": []}
    for m in modelos:
        activo = bool(activos.get(m["id"], m.get("activo", True)))
        nombre = m["nombre"] + ("" if m.get("aplicable") else " · no aplica") + (" · desactivado" if m.get("aplicable") and not activo else "")
        nombres.append(nombre)
        for clave, agg, col in (("medio", med, VJ_ACTIVO), ("pesimista", pes, VJ_PESIMISTA)):
            v = _num(m.get(f"precio_{clave}"))
            if v is None or not m.get(f"aplicable_{clave}"):
                datos[clave].append({"value": None})
                continue
            outlier = m["id"] in (agg.get("modelos_outlier") or [])
            color = VJ_INACTIVO if not activo else (AVISO_TXT if outlier else col)
            topes = m.get(f"topes_{clave}") or []
            txt = f"${v:,.2f}" + (" ▶" if v > x_max else "") + (" · outlier" if outlier and activo else "") \
                + (" · tope" if topes and activo and not outlier else "")
            datos[clave].append({"value": round(min(v, x_max), 2), "itemStyle": {"color": color, "opacity": 1.0 if activo else 0.45},
                                 "label": {"formatter": txt}})

    def linea(valor: float | None, nombre: str, color: str, tipo: str, k: int) -> list[dict]:
        if valor is None:
            return []
        return [{"xAxis": round(valor, 2), "lineStyle": {"color": color, "type": tipo, "width": 2},
                 "label": {"show": True, "formatter": f"{nombre} ${valor:,.2f}", "position": "start", "color": color,
                           "fontSize": 11, "offset": [0, -16 * k]}}]
    marca_data = (linea(fm, "Valor justo Medio", COLOR_FAIR_VALUE, "dashed", 0) + linea(fp, "Piso Pesimista", VJ_PESIMISTA, "dashed", 1)
                  + linea(px, "Precio", COLOR_PRECIO, "solid", 2))
    marca = {"symbol": "none", "silent": True, "animation": False, "data": marca_data}
    rotulo = {"show": True, "position": "right", "color": TXT, "fontSize": 11}
    series = [
        {"name": "Escenario Medio", "type": "bar", "barMaxWidth": 14, "itemStyle": {"color": VJ_ACTIVO}, "label": rotulo,
         "data": datos["medio"], "markLine": marca},
        {"name": "Escenario Pesimista", "type": "bar", "barMaxWidth": 14, "itemStyle": {"color": VJ_PESIMISTA}, "label": rotulo,
         "data": datos["pesimista"]}]
    return {**_estilo_base(),
            "title": {"text": mensaje or "", "left": "center", "top": "middle",
                      "textStyle": {"color": MUTED, "fontSize": 14, "fontWeight": "normal"}},
            "grid": {"left": 190, "right": 90, "top": 64, "bottom": 28},
            "legend": {"data": ["Escenario Medio", "Escenario Pesimista"], "top": 4, "textStyle": {"color": TXT, "fontSize": 12}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "backgroundColor": BG_POPUP, "borderColor": BORDE,
                        "textStyle": {"color": TXT, "fontSize": 12},
                        ":valueFormatter": "(v) => v == null ? 'No aplica' : '$' + Number(v).toFixed(2)"},
            "xAxis": {"type": "value", "min": 0, "max": round(x_max, 2), "axisLine": {"lineStyle": {"color": BORDE}},
                      "axisLabel": {"color": MUTED, ":formatter": "(v) => '$' + Math.round(v).toLocaleString('en-US')"},
                      "splitLine": {"lineStyle": {"color": GRILLA}}},
            "yAxis": {"type": "category", "inverse": True, "data": nombres, "axisLine": {"lineStyle": {"color": BORDE}},
                      "axisLabel": {"color": TXT, "fontSize": 11, "width": 180, "overflow": "truncate"}},
            "series": series}


def alto_barras(n_modelos: int) -> int:
    """Alto en px del grafico de barras segun el numero de modelos (2 barras por modelo)."""
    return max(420, 40 * n_modelos + 110)


# --- Texto explicativo de los modelos (pestana Valor Justo, acordeon colapsable) ---------------------------------------
def _expl_modelos() -> dict[str, str]:
    t = fv.TOPE_MARGEN
    return {
        "dcf_5y_fcf": "Flujo de caja libre (FCF) proyectado 5 años + valor terminal en el año 5.",
        "dcf_10y_fcf": f"FCF proyectado {fv.N_ANIOS_DCF} años en 2 etapas (crece a {_pct(fv.FRACCION_CRECIMIENTO, 0)} del "
                       f"crecimiento fundamental los primeros {fv.ANIOS_ETAPA1} años y decae hasta g) + valor terminal.",
        "dcf_5y_ebitda": "Igual que el DCF 5 años pero parte del EBITDA convertido a flujo con su conversión histórica.",
        "dcf_10y_ebitda": f"DCF de {fv.N_ANIOS_DCF} años desde el EBITDA; el margen converge a la mediana histórica, con tope "
                          f"de {_pct(t['medio'], 0)} (Medio) / {_pct(t['pesimista'], 0)} (Pesimista) de los ingresos.",
        "dcf_5y_revenue": "Proyecta ingresos 5 años y los convierte en FCF con el margen histórico (con tope).",
        "dcf_10y_revenue": f"Proyecta ingresos {fv.N_ANIOS_DCF} años; el margen FCF converge a la mediana histórica "
                           f"(tope {_pct(t['medio'], 0)} / {_pct(t['pesimista'], 0)}).",
        "pe": "Ganancia por acción (EPS) x P/E objetivo (mediana histórica propia recortada, mezclada con el sector).",
        "ev_ebitda": "EBITDA x EV/EBITDA objetivo, menos la deuda neta, por acción.",
        "ev_revenue": "Ingresos x EV/Ingresos objetivo, menos la deuda neta, por acción.",
        "pb": "Valor contable por acción x P/B objetivo.",
        "p_fcf": "FCF por acción x P/FCF objetivo.",
        "ddm": "Modelo de dividendos de Gordon: dividendo siguiente / (ke - g).",
        "epv": "Valor actual de las ganancias normalizadas SIN crecimiento (Greenwald); es un piso de crecimiento cero.",
        "graham": f"Raíz de {_es(fv.GRAHAM_FACTOR, 1)} x EPS x valor contable por acción.",
        "lynch": "PEG = 1: EPS x crecimiento del EPS (%), con tope de crecimiento (25 % Medio / 15 % Pesimista)."}


def explicacion_modelos(modelos: list[dict] | None = None) -> dict:
    """Contenido del acordeon «Explicación de los modelos y familias» como datos: ``familias`` (que mide cada una),
    ``modelos`` (una linea por modelo) y ``no_aplican`` (modelo + motivo REAL del motor para el ticker actual)."""
    rg = fv.RANGOS_MULTIPLOS
    familias = [
        ("DCF (flujos descontados)", f"Valora el negocio por los flujos de caja que generará, descontados al WACC. "
                                     f"Depende de tus supuestos de futuro (crecimiento, márgenes, WACC, g)."),
        ("Múltiplos", f"Compara con lo que el mercado pagó antes por la misma empresa (mediana histórica propia, "
                      f"{_pct(fv.PESO_MULTIPLO_PROPIO, 0)}) y por su sector ({_pct(1 - fv.PESO_MULTIPLO_PROPIO, 0)}), dentro de "
                      f"rangos fijos (P/E {_es(rg['pe'][0], 0)}-{_es(rg['pe'][1], 0)})."),
        ("Valor / Dividendos", "Reglas clásicas (EPV, Graham, Lynch) y el modelo de dividendos: valoran por lo que gana "
                               "hoy o paga, con sesgos propios; en empresas de alto crecimiento suelen dar por debajo del precio.")]
    expl = _expl_modelos()
    por_id = {m["id"]: m for m in (modelos or [])}
    lista = [{"id": i, "nombre": n, "familia": fam, "linea": expl.get(i, "")} for i, n, fam in fv.MODELOS]
    no = [{"id": m["id"], "nombre": m["nombre"], "motivo": m.get("motivo") or m.get("motivo_exclusion") or "sin datos"}
          for m in por_id.values() if not m.get("aplicable")]
    return {"familias": familias, "modelos": lista, "no_aplican": no}

def texto_ayuda_historico() -> list[str]:
    """Texto del acordeon «? Cómo interpretar esta gráfica» (las cifras salen de las constantes del motor)."""
    return [
        "Qué muestra: el Valor Justo del Escenario Medio a lo largo del tiempo (línea rosa), el precio de mercado (línea blanca) "
        "y seis bandas alrededor del valor justo: ±10 %, ±20 % y ±30 %.",
        "Bandas de sobrevaloración (cian, +10 a +30 %): cuando el precio entra en ellas cotiza por encima del valor justo; "
        "cuanto más arriba, mayor la sobrevaloración según el modelo.",
        "Bandas de margen de seguridad (magenta, −10 a −30 %): el precio por debajo del valor justo; cuanto más abajo, mayor el "
        "colchón frente a que los supuestos fallen. La línea discontinua naranja es el Piso fundamental (Escenario Pesimista actual).",
        f"Es una reconstrucción con el modelo ACTUAL, no un dato histórico publicado. Causal: cada punto usa solo reportes "
        f"ya públicos ({fv.REZAGO_PUBLICACION_DIAS} días de rezago) y se mantiene entre reportes con una transición suave de "
        f"{fv.TRANSICION_DIAS} días (sin mirar al futuro). Ancla final: el último punto es el valor justo de hoy y es lo único "
        f"que no es point-in-time (usa el último balance y el crecimiento forward de yfinance).",
        "Los parámetros de mercado (Rf ^TNX, ERP, beta, sector) y los sliders son los vigentes en todas las fechas: no son "
        "point-in-time.",
        f"La línea punteada a 12 meses es ILUSTRATIVA, no una predicción: hace crecer el valor justo actual a una tasa fija "
        f"({_pct(fv.PROYECCION_AMORTIGUACION, 0)} del crecimiento de etapa 1 + {_pct(1 - fv.PROYECCION_AMORTIGUACION, 0)} de g∞, "
        f"tope {_pct(fv.PROYECCION_TOPE, 0)}).",
        "Rueda del ratón: zoom; deslizador inferior: rango; el cursor muestra fecha, precio, valor justo medio y % "
        "sobre/infravalorado en esa fecha."]


def texto_slider(clave: str) -> str:
    """Descripcion breve del impacto de cada slider (rangos y reglas leidos de las constantes del motor)."""
    if clave == "wacc":
        return (f"Tasa de descuento del Escenario Medio (automática: ^TNX + beta de Blume x ERP, acotada a "
                f"{_pct(fv.WACC_MIN, 1)}-{_pct(fv.WACC_MAX, 1)}). Más alta: los DCF y el EPV valen menos. Sustituye al WACC "
                f"Medio; el Pesimista se deriva (Medio + {fv.WACC_PESIMISTA_SPREAD * 1e4:.0f} pb, mínimo {_pct(fv.WACC_PESIMISTA_MIN, 1)}).")
    if clave == "g":
        return (f"Crecimiento perpetuo tras el año {fv.N_ANIOS_DCF}. Más alto: mayor valor terminal (domina el DCF). Máximo = "
                f"WACC - {_es(fv.DIF_MIN_WACC_G * 100, 1)} pp. Fuera de {_pct(fv.G_RANGO_AVISO[0], 1)}-{_pct(fv.G_RANGO_AVISO[1], 1)} "
                f"se avisa. El Pesimista usa el menor entre {_pct(fv.G_TERMINAL['pesimista'], 1)} y este valor.")
    return ("Descuento voluntario sobre el valor justo Medio: solo cambia el precio con margen de seguridad de la tarjeta; "
            "no mueve el valor justo.")




# --- Ejecucion en segundo plano ----------------------------------------------------------------------------
def validar_ticker(texto: str | None) -> str | None:
    """Ticker normalizado (mayusculas) o None si no cumple el formato bursatil basico."""
    t = (texto or "").strip().upper()
    return t if TICKER_RE.match(t) else None


def validar_capital(v: Any) -> bool:
    """Capital valido: numero finito > 0."""
    f = _num(v)
    return f is not None and f > 0


@app.get("/health")
def health() -> dict:
    """Health check (Render/UptimeRobot): no construye interfaz ni carga modulos pesados."""
    return {"status": "ok"}


def limitar_tickers_escaneo(tickers: list[str], nube: bool | None = None) -> tuple[list[str], int]:
    """(tickers a escanear, cuantos se omiten): local sin limite; nube = los primeros ``MAX_TICKERS_ESCANEO_NUBE``."""
    nube = userdata.en_nube() if nube is None else nube
    if not nube or len(tickers) <= MAX_TICKERS_ESCANEO_NUBE:
        return list(tickers), 0
    return list(tickers[:MAX_TICKERS_ESCANEO_NUBE]), len(tickers) - MAX_TICKERS_ESCANEO_NUBE


async def calcular_ticker(ticker: str, capital: float, backtest: bool = True, sec: bool = True,
                          valor_justo: bool = True) -> dict:
    """Corre ``main.analizar_ticker`` fuera del hilo de eventos (proceso aparte; hilo si el pool no existe).

    El backtest es Python puro con GIL, por eso en local se prefiere ``run.cpu_bound``. En la nube (512 MB) un pool de
    procesos duplicaria la memoria: se usa un solo hilo (``asyncio.to_thread``). ``periodo`` = ``periodo_consulta()``: 10y en
    local, 3y en la nube (visor ligero).
    """
    kw = dict(periodo=periodo_consulta(), backtest=backtest, sec=sec, verbose=False)
    if not valor_justo:  # solo se pasa si se desactiva (por defecto analizar_ticker ya extrae las entradas)
        kw["valor_justo"] = False
    if userdata.en_nube():
        return await asyncio.to_thread(nucleo.analizar_ticker, ticker, capital, **kw)
    from nicegui import run
    try:
        return await run.cpu_bound(nucleo.analizar_ticker, ticker, capital, **kw)
    except (RuntimeError, BrokenProcessPool, PicklingError, OSError):
        return await asyncio.to_thread(nucleo.analizar_ticker, ticker, capital, **kw)


def cargar_watchlist(ruta: str = RUTA_WATCHLIST) -> list[str]:
    """Tickers de la lista de EJEMPLO watchlist.csv ([] si falta o esta vacio)."""
    try:
        return nucleo.leer_watchlist(ruta)
    except (OSError, UnicodeDecodeError):
        return []


def cargar_watchlist_usuario(directorio: str | None = None) -> list[str]:
    """Tickers de la watchlist personalizada del usuario ([] si aun no existe)."""
    return userdata.tickers(userdata.leer(directorio))


def resumen_liviano(r: dict) -> dict:
    """Resultado sin los objetos pesados (df, form4, r13f): lo unico que necesita el ranking."""
    return {k: v for k, v in r.items() if k not in ("df", "form4", "r13f")}


class CacheLRU(OrderedDict):
    """dict con tope opcional (LRU): ``maximo=None`` = sin limite (local)."""

    def __init__(self, maximo: int | None = None) -> None:
        super().__init__()
        self.maximo = maximo

    def __setitem__(self, k, v) -> None:
        super().__setitem__(k, v)
        self.move_to_end(k)
        while self.maximo is not None and len(self) > self.maximo:
            self.popitem(last=False)


# --- Pestana "Modelo": contenido explicativo como DATOS (testeable sin red ni NiceGUI) ---------------------------
FECHA_AUDITORIA = "2026-11-26"  # fecha de los datos de docs/METODOLOGIA.md
# Cifras de la auditoria (docs/METODOLOGIA.md, formato del documento). Un test comprueba que cada una siga en el .md.
CIFRAS_AUDITORIA: dict[str, str] = {
    "n_calibracion": "236", "spearman_bounce": "-0.084", "spearman_ret60": "+0.154",
    "tercil_bajo_bounce": "65.0 %", "tercil_medio_bounce": "61.0 %", "tercil_alto_bounce": "57.0 %",
    "tercil_bajo_ret60": "4.5 %", "tercil_medio_ret60": "5.4 %", "tercil_alto_ret60": "10.6 %",
    "mae60_tercil_bajo": "-9.1 %", "mae60_tercil_alto": "-10.0 %",
    "n_ordenes": "682", "n_fills": "237", "fill_pool": "34.8 %", "bounce_pool": "61.2 %", "absorcion_pool": "28.7 %",
    "mediana252_fill": "23.3 %", "mediana252_base": "22.5 %", "media_pp60": "+1.1", "media_pp252": "+3.6",
    "mae60_mediana": "-9.1 %", "mae60_base": "-6.6 %", "mae252_mediana": "-15.9 %", "mae252_base": "-11.9 %",
    "solape_rodante": "25.0 %", "solape_control": "29.8 %", "solape_estricto": "6.2 %", "n_trimestres_13f": "16",
    "fill_32_pp": "-3.7 pp", "fam3_antes": "40.3 %", "fam3_ahora": "9.4 %",
    "abs_ret_con": "+9.5 %", "abs_ret_sin": "+4.5 %", "abs_p": "0.06",
}
ETIQUETAS_COMPONENTES = {"respaldo_13f_form4": "Respaldo 13F / Form 4", "stock_act": "STOCK Act (Congreso)",
                         "absorcion": "Absorción", "backtest": "Backtest", "confluencia": "Confluencia"}
def normalizacion_componentes(p: dict[str, Any]) -> dict[str, str]:
    """Como se pasa cada componente a 0-100, en palabras simples. Los numeros de la formula (50/30/20, 60/40, 15 pts,
    1 M USD, +15 %, -25 %) estan en scoring.py y los fija test_modelo_normalizacion_consistente_con_scoring."""
    return {
        "respaldo_13f_form4": "La mitad mide qué fracción de los gestores 13F ACTIVOS con posición en la acción la "
                              "aumentó (los pasivos de índice no cuentan); la otra mitad mide las compras de insiders "
                              "(Form 4, código P) en USD, y llega al máximo con 1 millón de USD comprados. Una mitad sin "
                              "dato se excluye y el componente usa solo la otra; si faltan ambas, el componente se excluye.",
        "stock_act": "Toma los puntos (0 a 15) del flujo de compras de miembros del Congreso de EE. UU. (ley STOCK Act) "
                     "y los pasa a escala 0-100: 15 puntos equivalen a 100. Si la fuente no responde, el componente se "
                     "excluye (no vale 0 ni un valor neutro).",
        "absorcion": f"Es el porcentaje de órdenes ejecutadas en las que, en las {p['velas_abs']} velas desde el toque "
                     f"(la del toque y las {p['ventana_abs']} siguientes), apareció una vela de rechazo dentro de la zona (cierre en el tercio superior y volumen mayor a "
                     f"{_es(p['vol_mult'], 2)} veces su media de 20 días). Una tasa de 26 % equivale a 26 puntos.",
        "backtest": "Mezcla tres cosas: 50 % el bounce rate, 30 % el retorno mediano a 60 ruedas (llega al máximo con "
                    "+15 %) y 20 % el control de la MAE (0 % de caída da el máximo; -25 % o peor da cero).",
        "confluencia": "60 % por el número de familias que coinciden en la zona (1 familia da 0 y 4 familias dan el "
                       "máximo de esa parte) y 40 % por las anclas de valoración dentro de la zona (2 o más dan el "
                       "máximo). Si la valoración no se pudo consultar, esa parte se excluye y el score usa solo las "
                       "familias, reescalado a 0-100; si se consultó y ninguna ancla cae en la zona, la parte vale 0."}


AVISO_LEGAL = ("**No es asesoría financiera.** Esta herramienta es de investigación y educación: no recomienda "
               "comprar ni vender nada. Invertir implica riesgo de pérdida, y los resultados pasados no garantizan "
               "resultados futuros.")


def _es(x: float | str, dec: int = 1) -> str:
    """Numero con coma decimal (12.5 -> '12,5'); acepta texto con formato del documento ('22.6 %')."""
    if isinstance(x, str):
        return re.sub(r"(\d)\.(\d)", r"\1,\2", x)
    return f"{x:.{dec}f}".replace(".", ",")


def _pct(frac: float, dec: int = 1) -> str:
    return f"{_es(frac * 100, dec)} %"


def _c(clave: str) -> str:
    """Cifra de auditoria con coma decimal."""
    return _es(CIFRAS_AUDITORIA[clave])


def gestores_13f() -> dict[str, list[str]]:
    """Gestores 13F que consulta el screener, leidos de ``src/data/sec_edgar.py`` (no se copian a mano).

    ``activos`` cuentan para la acumulacion; ``pasivos`` (Vanguard / State Street, flujos de indice) se consultan y
    se muestran pero no cuentan. Listas vacias si el modulo no se puede importar.
    """
    try:
        from src.data import sec_edgar
        pasivos = set(getattr(sec_edgar, "GESTORES_PASIVOS", ()))
        nombres = list(sec_edgar.GESTORES_13F)
    except Exception:  # noqa: BLE001
        return {"activos": [], "pasivos": []}
    return {"activos": [g for g in nombres if g not in pasivos], "pasivos": [g for g in nombres if g in pasivos]}


def _lista_nombres(nombres: list[str]) -> str:
    return ", ".join(nombres) if nombres else "(lista no disponible)"


def _parametros() -> dict[str, Any]:
    """Parametros reales del modelo leidos del codigo (el texto de la pestana no los copia a mano)."""
    from src.levels import valuation_floors as vf
    from src.levels import volume_liquidity as vl
    from src.validation import backtest_ranges as br
    return {
        "dist_min": br.DIST_MIN, "dist_max": br.DIST_MAX, "vida_orden": br.VIDA_ORDEN,
        "bounce_pct": br.BOUNCE_PCT, "bounce_ventana": br.BOUNCE_VENTANA, "bounce_inval": br.BOUNCE_INVALIDA,
        "vol_mult": br.VOL_MULT_ABS, "ventana_abs": br.VENTANA_ABS,
        "velas_abs": br.VENTANA_ABS + 1,  # range(k, k + VENTANA_ABS + 1): la vela del toque cuenta
        "ancho_cluster_atr": br.ANCHO_MAX_CLUSTER_ATR, "ancho_cluster_pct": br.ANCHO_MAX_CLUSTER_PCT,
        "nucleo_operativo": br.NUCLEO_OPERATIVO,
        "tickers_bt": list(br.TICKERS_DEFECTO), "anios_bt_ref": br.ANIOS_BACKTEST,
        "vp_anios": vl.VP_ANIOS, "value_area": vl.VALUE_AREA_PCT, "swing_semanas": vl.OB_SWING_SEMANAL,
        "correccion": vl.CORRECCION_MAYOR_PCT, "avwap_dias": vl.AVWAP_MIN_ESTRUCTURAL_DIAS,
        "percentiles": tuple(vf.PERCENTILES), "tasa_dcf": vf.TASA_DESCUENTO, "horizonte_dcf": vf.HORIZONTE_DCF,
        "margenes": tuple(vf.MARGENES_SEGURIDAD), "bandas_sd": tuple(vf.BANDAS_SD),
        "max_zonas": nucleo.MAX_ZONAS, "max_tramos": nucleo.MAX_TRAMOS_CAPITAL, "min_fills": nucleo.MIN_FILLS_AVISO,
        "gestores": gestores_13f(),
        "anios_bt_app": inspect.signature(nucleo.analizar_ticker).parameters["anios_backtest"].default}


def _ejemplo_capital(capital: float = 1000.0) -> dict[str, Any]:
    """Reparto de ejemplo calculado con las MISMAS funciones de la app (asignar_capital, precio promedio)."""
    zonas = [{"nivel": 1, "r_inf": 94.0, "r_sup": 96.0, "n_familias": 3},
             {"nivel": 2, "r_inf": 85.0, "r_sup": 88.0, "n_familias": 2},
             {"nivel": 3, "r_inf": 78.0, "r_sup": 80.0, "n_familias": 1}]
    nucleo.asignar_capital(zonas, capital)
    return {"capital": capital, "zonas": zonas, "promedio": precio_promedio_ponderado(zonas)}


def _tabla_evidencia() -> dict | None:
    """Evidencia empirica por componente, leida de pesos_calibrados.json (None si no existe)."""
    from src.validation import scoring
    try:
        with open(scoring.RUTA_PESOS, encoding="utf-8") as f:
            ev = json.load(f).get("evidencia") or {}
    except (OSError, ValueError):
        return None
    filas = []
    for k in scoring.COMPONENTES:
        e = ev.get(k)
        if e:
            z = e.get("z")
            filas.append([ETIQUETAS_COMPONENTES.get(k, k), e.get("detalle", ""), "n/d" if z is None else f"{z:+.2f}"])
    return {"columnas": ["Componente", "Qué se midió (texto original de la calibración)", "z / t"], "filas": filas} \
        if filas else None


# Funciones reales citadas por nombre en la seccion "La logica detras" (modulo, funcion). Un test comprueba que cada
# una exista y que el texto la mencione; si se renombra una en src/, el test avisa.
FUNCIONES_CITADAS: tuple[tuple[str, str], ...] = (
    ("src.data.market_data", "descargar_historial"),
    ("src.data.market_data", "verificar_proximidad_earnings"),
    ("src.data.sec_edgar", "obtener_13f_ticker"),
    ("src.data.sec_edgar", "compras_insiders"),
    ("src.data.stock_act", "consultar_flujo_congreso_stock_act"),
    ("src.data.stock_act", "sincronizar_repo"),
    ("src.levels.volume_liquidity", "order_blocks_semanales"),
    ("src.levels.valuation_floors", "calcular_pisos_valoracion"),
    ("src.validation.backtest_ranges", "calcular_zonas_pit"),
    ("src.validation.backtest_ranges", "backtest_ticker"),
    ("src.validation.scoring", "indice_desde_modulos"),
    ("main", "asignar_capital"),
)
# Ejemplo de semiconductores: TODAS estas cifras son inventadas y el texto las rotula como ILUSTRATIVAS.
EJ_SEMI = {"precio": 100.0, "tramos": ((90.0, 0.50), (84.0, 0.30), (78.0, 0.20)), "minimo": 76.0}


def _seccion_logica(p: dict[str, Any]) -> dict:
    """Seccion "La logica detras: como y por que (en simple)". Texto para no expertos; parametros y gestores se
    leen del codigo, las cifras de evidencia de ``CIFRAS_AUDITORIA`` (verificadas contra docs/METODOLOGIA.md)."""
    ges = p["gestores"]
    n_ges = len(ges["activos"]) + len(ges["pasivos"])
    pct0 = lambda f: _es(f * 100, 0)  # noqa: E731
    ej = EJ_SEMI
    t1, t2, t3 = ej["tramos"]
    prom = 1.0 / sum(w / pr for pr, w in ej["tramos"])  # precio promedio si se ejecutan los tres tramos
    mae_1 = (ej["minimo"] / t1[0] - 1) * 100
    mae_prom = (ej["minimo"] / prom - 1) * 100
    f = lambda k: _c(k)  # noqa: E731
    par: list[Any] = [
        "Esta sección explica el *por qué* del modelo con palabras simples. Separa tres cosas: **la idea** (la "
        "hipótesis), **cómo se calcula** (lo que hace el código) y **lo que está demostrado** (la evidencia propia, que "
        "es modesta). Mezclarlas es el error más común al leer herramientas como esta.",

        "### 1. La idea central",
        "Un fondo grande no puede comprar todo lo que quiere de golpe: si lo intentara, su propia compra empujaría el "
        "precio hacia arriba y pagaría más caro. Por eso suele acumular poco a poco, en zonas donde hay vendedores "
        "dispuestos (liquidez). Esas compras dejan **huellas**: mucho volumen en un rango de precios, rechazos "
        "repetidos del mismo piso, estructuras que el precio respeta.",
        {"ejemplo": "**Analogía:** imagina a alguien que necesita comprar 10 000 kilos de fruta en un mercado pequeño. Si "
                    "pide todo al primer puesto, el precio sube. Va pasando por varios puestos y deja marcas: puestos "
                    "donde se vendió mucho, precios que se repiten. El modelo no ve al comprador; busca esas marcas y "
                    "señala los lugares donde **varias marcas distintas coinciden**."},
        "Esa coincidencia es la *confluencia* (sección 3). La hipótesis es que una zona con varias huellas "
        "independientes es un lugar más razonable para dejar una orden que un precio cualquiera. Es una hipótesis, "
        "no una ley: más adelante verás qué dice la evidencia.",

        "### 2. Cómo lo logra, paso a paso",
        "Es el flujo real de `main.analizar_ticker`:",
        {"lista": [
            "**1. Datos.** Baja el historial de precios y volumen ajustado (`descargar_historial`, yfinance) y mira si "
            "hay resultados trimestrales (earnings) cerca (`verificar_proximidad_earnings`), porque una orden "
            "abierta durante un reporte puede saltarse el precio con un gap.",
            f"**2. Niveles por familia.** Calcula las anclas de cada familia: perfil de volumen de {p['vp_anios']} años, "
            f"Order Blocks semanales (`order_blocks_semanales`), AVWAP, mínimos semanales y regresión logarítmica. "
            f"Solo usan datos hasta hoy.",
            f"**3. Confluencia y clusters.** Agrupa las anclas cercanas en zonas (`calcular_zonas_pit`; ancho máximo "
            f"{_es(p['ancho_cluster_atr'], 0)} ATR semanales y {pct0(p['ancho_cluster_pct'])} % del precio) y las numera N1 a N{p['max_zonas']} por cuántas familias "
            f"distintas coinciden.",
            "**4. Valoración.** Compara el precio con los múltiplos históricos de la propia empresa (P/E, EV/FCF), un "
            "DCF conservador y la banda de regresión (`calcular_pisos_valoracion`). Cuenta cuántas de esas líneas "
            "caen dentro de la zona principal.",
            "**5. Respaldo de compras reales.** Revisa si insiders compraron con su dinero (Form 4, código P: "
            "`compras_insiders`), si fondos grandes aumentaron posición (13F: `obtener_13f_ticker`) y qué hizo el "
            "Congreso de EE. UU. (STOCK Act: `consultar_flujo_congreso_stock_act`).",
            f"**6. Backtest walk-forward.** Reconstruye, mes a mes y usando solo lo que se sabía en cada fecha, las "
            f"zonas del pasado y simula órdenes límite con vida de {p['vida_orden']} ruedas (`backtest_ticker`). Mide "
            f"cuántas se ejecutan, cuántas rebotan y cuánto cae el precio después de comprar.",
            "**7. Índice de confiabilidad.** Resume cinco componentes en un número de 0 a 100 "
            "(`indice_desde_modulos`). Un dato que no se pudo obtener se excluye, no cuenta como cero.",
            f"**8. Reparto de capital.** Reparte tu capital entre las {p['max_tramos']} mejores zonas, en proporción a "
            f"su número de familias (`asignar_capital`), con una orden límite en el techo de cada zona."]},

        "### 3. De dónde salen los datos",
        {"lista": [
            "**Precios, volumen y calendario de earnings: yfinance** (librería no oficial que consulta Yahoo Finance). "
            "Aporta el historial ajustado por dividendos y splits y el calendario de resultados. Se descarga cada vez que "
            "pulsas Recalcular; esta app guarda en memoria el último resultado de cada ticker hasta que la cierras. "
            "Si falla (sin red o ticker inexistente), no hay análisis de ese ticker y se muestra el error. Los "
            "ajustes por dividendo usan información posterior: una distorsión pequeña.",
            "**SEC EDGAR, vía la librería `edgartools`:** es la fuente primaria. Se usa para tres cosas (siguientes puntos), y su manejo de fallos es común:",
            f"**13F-HR** (`obtener_13f_ticker`): la tenencia trimestral de {n_ges} gestores concretos "
            f"({_lista_nombres(ges['activos'] + ges['pasivos'])}); cuentan solo los activos "
            f"({_lista_nombres(ges['activos'])}). El 13F se presenta hasta unos 45 días después del cierre del "
            "trimestre: es información vieja, y no trae el día ni el precio de la compra.",
            "**Form 4** (`compras_insiders`): operaciones de directivos y grandes accionistas; se filtran solo las "
            "de código P (compra en mercado abierto). La regla de la SEC es presentarlo en 2 días hábiles tras la "
            "operación (es la norma de la SEC; el código no lo verifica). Revisa como máximo los 60 filings más "
            "recientes del último año.",
            "**Estados financieros 10-K / 10-Q** (datos XBRL, `get_facts`): ingresos, EPS, flujo de caja, deuda, "
            "acciones; alimentan la valoración. Salen con un rezago de unos 40 a 60 días tras cada cierre (el código supone 45 para evitar mirar "
            "el futuro) y la fecha del último 10-K ancla un AVWAP. Si EDGAR falla, el EPS cae a yfinance (solo unos 5 trimestres: menos "
            "historia) y, para la fecha del 10-K, a una estimación.",
            "**Si EDGAR no responde:** un Form 4 fallido, un 13F sin respuesta de ningún gestor activo o una valoración "
            "sin consultar se tratan como **dato no disponible** (se excluyen y baja la cobertura), nunca como cero. "
            "EDGAR exige identificarse (variable `SEC_EDGAR_IDENTITY`).",
            "**STOCK Act (Congreso de EE. UU.):** se lee el repositorio público de GitHub "
            "`kadoa-org/congress-trading-monitor` (carpeta `public/data`, `sincronizar_repo`). Las operaciones de "
            "legisladores se reportan con retraso (la norma legal da hasta unos 45 días; el código no lo verifica) y "
            "con rangos de monto, no cifras exactas. Se guarda una copia local que se renueva si tiene más de 24 horas. Si no se puede descargar o "
            "leer, el componente se marca no disponible.",
            "**`smartmoneyconcepts`** (librería de terceros, sin conexión a internet): detecta los Order Blocks "
            "semanales no mitigados. Es un concepto popular sin respaldo académico formal.",
            "**Caché local:** el backtest de validación guarda precios y órdenes en `src/validation/_cache`, y STOCK Act "
            "guarda su repositorio en `src/data/_cache`. Las consultas a EDGAR de esta app no se guardan en disco."]},

        "### 4. Por qué debería funcionar (hipótesis) y qué muestra la evidencia propia",
        "**La lógica (hipótesis):** los grandes jugadores acumulan donde hay liquidez; esas zonas dejan huellas; "
        "cuando varias coinciden, son lugares donde el precio ya fue aceptado antes y donde hay compradores con "
        "historial. Además, anclar las zonas a estructuras de varios años (y no al cierre de ayer) evita que se muevan "
        "con el ruido diario.",
        "**Lo demostrado (auditoría propia, `docs/METODOLOGIA.md`):**",
        {"lista": [
            f"El índice de confiabilidad **no discrimina resultados en conjunto** (con las zonas estrechas el "
            f"panorama es matizado): correlación con el retorno a 60 ruedas {f('spearman_ret60')}, pero con el rebote "
            f"{f('spearman_bounce')} ({f('n_calibracion')} órdenes). Por terciles de score, el rebote baja "
            f"({f('tercil_bajo_bounce')}, {f('tercil_medio_bounce')}, {f('tercil_alto_bounce')}), la MAE a 60 ruedas "
            f"empeora ({f('mae60_tercil_bajo')} a {f('mae60_tercil_alto')}) y el retorno medio a 60 ruedas sube "
            f"({f('tercil_bajo_ret60')}, {f('tercil_medio_ret60')}, {f('tercil_alto_ret60')}). Esa subida es débil y "
            f"no ex-ante (ver el aviso sobre la absorción en Limitaciones). No es una probabilidad.",
            f"La **mediana a 252 ruedas** de comprar en el rango ({f('mediana252_fill')}) es casi idéntica a la línea base "
            f"({f('mediana252_base')}). Solo la media mejora (+{f('media_pp60')[1:]} puntos a 60 ruedas), y se concentra en "
            f"acciones que subieron en el período.",
            f"La **MAE es peor**: mediana a 60 ruedas {f('mae60_mediana')} contra {f('mae60_base')} de la base. Se compra "
            f"después de una caída y el precio sigue cediendo antes de rebotar.",
            f"Con los 13F: los rangos calculados solo con datos previos **no coinciden mejor que el azar** con las zonas "
            f"donde compraron los fondos ({f('solape_rodante')} contra {f('solape_control')} de un rango aleatorio) y la "
            f"muestra es muy pequeña ({f('n_trimestres_13f')} trimestres).",
            "Los pisos de valoración, Form 4 y STOCK Act **no se probaron** empíricamente (faltan series históricas "
            "*point-in-time*). Hay sesgo de supervivencia: la muestra eran acciones grandes que ganaron."]},
        {"aviso": "**Conclusión honesta.** *Lo que aporta:* disciplina, un plan de entradas escalonadas, riesgo definido "
                  "antes de comprar, trazabilidad con fuentes primarias (SEC) y niveles que no dependen del cierre "
                  "intradía. *Lo que NO se puede afirmar:* que las zonas predigan rebotes, que el índice anticipe "
                  "resultados ni que esto supere a comprar sin más. Es una herramienta para ordenar y planificar, no una "
                  "señal."},

        "### 5. Un ejemplo sencillo: semiconductores",
        {"ejemplo": f"**ILUSTRATIVO, NO REAL.** Todas las cifras de este ejemplo son inventadas para explicar; no son "
                    f"precios, datos ni zonas reales de NVDA ni de ningún otro ticker."},
        "Los semiconductores son un sector **cíclico y muy volátil**: pasan de épocas de demanda enorme a épocas de "
        "exceso de inventario, y las acciones se mueven mucho más que el mercado. Para una empresa como NVDA, "
        "*imagina* estos números:",
        {"lista": [
            f"Precio actual (inventado): **{_es(ej['precio'], 0)} USD**. El modelo marca tres zonas por debajo, con "
            f"techos en {_es(t1[0], 0)}, {_es(t2[0], 0)} y {_es(t3[0], 0)} USD (ejemplo inventado).",
            f"Con 1000 USD (inventados) en tramos de {pct0(t1[1])} %, {pct0(t2[1])} % y {pct0(t3[1])} %, tu precio "
            f"promedio si se ejecutan los tres sería de **{_es(prom, 1)} USD**.",
            f"Supón que el precio cae hasta {_es(ej['minimo'], 0)} USD antes de rebotar. Quien compró todo en el primer "
            f"techo ({_es(t1[0], 0)}) sufrió una MAE de **{_es(mae_1, 1)} %**; con el escalonamiento, medida contra el "
            f"precio promedio, fue de {_es(mae_prom, 1)} %."]},
        "**Por qué importan aquí la MAE y el escalonamiento.** En un sector volátil el precio suele pasar de largo "
        "por el primer nivel y seguir cayendo antes de rebotar (justo lo que mostró la auditoría: MAE peor que la "
        "base). Por eso el modelo no manda todo a un solo precio: reparte en varios tramos y te muestra la MAE de "
        "antemano, para que decidas cuánto caída puedes aguantar *antes* de comprar y no después.",
        "**¿Y si cambia el ciclo del sector?** Las zonas se calculan con la historia (por ejemplo, "
        f"{p['vp_anios']} años de volumen). Si el sector pasa de un ciclo a otro (de auge a recesión, o por un cambio "
        "tecnológico), esas zonas pueden dejar de ser relevantes: el precio las perfora y no rebota. El modelo "
        "no tiene ninguna lógica específica de sector ni de ciclo; no sabe en qué fase está la industria. Las zonas se "
        "recalculan al pulsar Recalcular y el backtest mide solo el pasado, que puede no repetirse.",

        "### 6. Limitaciones, en corto",
        {"lista": [
            "Los 13F llegan con hasta 45 días de retraso y sin día ni precio de compra; solo se consultan "
            f"{n_ges} gestores ({len(ges['activos'])} activos): es una muestra mínima de fondos, no el universo.",
            f"Sesgo de supervivencia: la auditoría usó {len(p['tickers_bt'])} acciones de gran capitalización de "
            f"{p['anios_bt_ref']} años mayormente alcistas. Las órdenes se solapan y las pruebas estadísticas salen "
            f"infladas.",
            "Los pisos de valoración (P/E, EV/FCF, DCF) no están auditados.",
            "No se modelan costos de transacción, impuestos ni liquidez.",
            "Sector cíclico o cambio de régimen: lo que funcionó (o no) en el pasado puede no repetirse.",
            "Los datos que faltan se excluyen del índice (baja la cobertura) y no se rellenan con valores neutros.",
            f"Con menos de {p['min_fills']} ejecuciones simuladas, las métricas del backtest de un ticker son ruido."]},
        {"aviso": "Material educativo; no es asesoría financiera."}]
    return {"id": "logica", "titulo": "2. La lógica detrás: cómo y por qué (en simple)", "icono": "psychology",
            "abierta": True, "parrafos": par}


def _ejemplo_valor_justo() -> dict[str, Any]:
    """Ejemplo ILUSTRATIVO calculado con el codigo real de ``fair_value`` (todas las cifras de entrada son inventadas)."""
    e = fv.EntradasValoracion(ticker="EJEMPLO", precio=100.0, acciones_diluidas=10.0, fcf_ltm=100.0, eps_ltm=8.0, bvps=60.0,
                              caja=0.0, deuda_total=0.0, tasa_impositiva=0.21, rf=fv.RF_DEFECTO, beta=1.0,
                              hist_fcf=[60.0, 70.0, 80.0, 90.0, 100.0], hist_ingresos=[500.0, 560.0, 620.0, 690.0, 760.0],
                              hist_eps=[5.0, 5.8, 6.6, 7.3, 8.0], multiplos_hist={"pe": [14.0, 15.0, 16.0, 15.0, 15.0]})
    p = fv.ParametrosValoracion(wacc=0.10)
    ids = ["dcf_10y_fcf", "pe", "graham"]
    esc = fv.calcular_escenarios(e, p, ids)
    ms = {m["id"]: m for m in esc["modelos"]}
    return {"entradas": e, "params": p, "esc": esc, "agg": esc["medio"], "agg_pes": esc["pesimista"],
            "dcf": ms["dcf_10y_fcf"]["precio_medio"], "dcf_pes": ms["dcf_10y_fcf"]["precio_pesimista"],
            "pe": ms["pe"]["precio_medio"], "pe_pes": ms["pe"]["precio_pesimista"], "graham": ms["graham"]["precio_medio"],
            "g1": esc["supuestos"]["medio"]["crecimiento_inicial_fcf"]}


def _seccion_valor_justo() -> dict:
    """Seccion "Valor justo" de la pestana Modelo: lenguaje simple; TODAS las constantes se leen de ``fair_value``."""
    ej = _ejemplo_valor_justo()
    e, p, agg, pes = ej["entradas"], ej["params"], ej["agg"], ej["agg_pes"]
    mos = fv.MARGEN_SEGURIDAD_DEFECTO
    n_fam = {f: sum(1 for m in fv.MODELOS if m[2] == f) for f in fv.FAMILIAS}
    rg = fv.RANGOS_MULTIPLOS
    tc, tm, gt, cm = fv.TOPE_CRECIMIENTO_ETAPA1, fv.TOPE_MARGEN, fv.G_TERMINAL, fv.COMPRESION_MULTIPLOS
    par: list[Any] = [
        {"aviso": fv.AVISO},
        "**Qué es.** Una estimación del precio «razonable» de una acción según lo que gana, lo que debe y lo que tiene en "
        "caja. No usa el gráfico de precios: usa los estados financieros (SEC EDGAR, 10-K y 10-Q). La app calcula "
        f"{len(fv.MODELOS)} modelos y los consolida en **un número por escenario**.",
        "### Dos escenarios: Medio y Pesimista (no hay optimista)",
        "Ambos son **supuestos de escenario** fijados de antemano, **no predicciones**: cambian la tasa de descuento, el "
        "crecimiento y los topes, pero no «adivinan» el futuro.",
        {"tabla": {"columnas": ["Supuesto", "Medio", "Pesimista"],
                   "filas": [
                       ["WACC (tasa de descuento)", f"automático, acotado a {_pct(fv.WACC_MIN)}-{_pct(fv.WACC_MAX)}",
                        f"Medio + {fv.WACC_PESIMISTA_SPREAD * 1e4:.0f} pb, mínimo {_pct(fv.WACC_PESIMISTA_MIN)}"],
                       ["Crecimiento perpetuo g∞", _pct(gt["medio"]), f"{_pct(gt['pesimista'])} (nunca mayor que el Medio)"],
                       [f"Tope de crecimiento (años 1-{fv.ANIOS_ETAPA1})", _pct(tc["medio"], 0), _pct(tc["pesimista"], 0)],
                       ["Tope del margen objetivo (sobre ingresos)", _pct(tm["medio"], 0), _pct(tm["pesimista"], 0)],
                       ["Múltiplos objetivo", "sin castigo", f"{_pct(cm['pesimista'], 0)} (compresión)"],
                       ["EPV (crecimiento cero)", f"se excluye si los ingresos crecen más de {_pct(fv.EPV_CRECIMIENTO_MAX, 0)} al año",
                        "se incluye como piso"],
                       ["Tope de crecimiento de Lynch", f"{_es(fv.LYNCH_TOPE_PCT_ESC['medio'], 0)} %",
                        f"{_es(fv.LYNCH_TOPE_PCT_ESC['pesimista'], 0)} %"]]}},
        "### El WACC: de dónde sale",
        f"**Tasa libre de riesgo (Rf):** el rendimiento del bono a 10 años de EE. UU. (^TNX de yfinance, válido entre "
        f"{_es(fv.RF_RANGO_PCT[0], 1)} % y {_es(fv.RF_RANGO_PCT[1], 0)} %; si falla se usa {_pct(fv.RF_DEFECTO, 2)}). **Prima de riesgo "
        f"(ERP):** {_pct(fv.ERP_DEFECTO, 1)} fija. **Beta de Blume:** {_es(fv.BLUME_PESO_RAW, 2)} x beta de yfinance + "
        f"{_es(1 - fv.BLUME_PESO_RAW, 2)} (la beta tiende a 1; se acota a {_es(fv.BETA_RAW_MIN, 0)}-{_es(fv.BETA_RAW_MAX, 0)} para "
        f"descartar datos basura). **Costo del patrimonio** ke = Rf + beta x ERP; **costo de la deuda** kd después de impuestos; "
        f"el WACC pondera ambos con los pesos reales de patrimonio y deuda. El Medio se acota a {_pct(fv.WACC_MIN)}-"
        f"{_pct(fv.WACC_MAX)}; el Pesimista es el Medio vigente + {fv.WACC_PESIMISTA_SPREAD * 1e4:.0f} pb con mínimo "
        f"{_pct(fv.WACC_PESIMISTA_MIN)}. Si mueves el slider de WACC, sustituye al Medio y el Pesimista se deriva de él.",
        "### Las familias de modelos, con un ejemplo corto",
        f"Empresa **inventada** con precio {_es(e.precio, 0)} USD, {_es(e.acciones_diluidas, 0)} acciones, flujo de caja "
        f"libre (FCF) de {_es(e.fcf_ltm, 0)} (creció de 60 a 100 en 5 años), ganancia por acción (EPS) de {_es(e.eps_ltm, 0)}, "
        f"valor contable por acción de {_es(e.bvps, 0)} y WACC de {_pct(p.wacc, 0)}:",
        {"lista": [
            f"**DCF** ({n_fam['DCF']} modelos: 5 y 10 años sobre FCF, EBITDA e ingresos). *Analogía:* cuánto pagarías hoy por "
            f"un negocio que te va a dar flujos de caja durante años, descontados porque un dólar de mañana vale menos que "
            f"uno de hoy. El DCF principal es de **{fv.N_ANIOS_DCF} años en 2 etapas** con convención de **mitad de año** (los "
            f"flujos llegan repartidos durante el año, no el 31 de diciembre): en los años 1 a {fv.ANIOS_ETAPA1} crece "
            f"{_pct(fv.FRACCION_CRECIMIENTO, 0)} del crecimiento fundamental (tope {_pct(tc['medio'], 0)} Medio / "
            f"{_pct(tc['pesimista'], 0)} Pesimista) y desde el año {fv.ANIOS_ETAPA1 + 1} decae linealmente hasta g∞ mientras los "
            f"márgenes convergen a su mediana histórica (tope {_pct(tm['medio'], 0)} / {_pct(tm['pesimista'], 0)}). Los ids "
            f"«5 años» son el mismo motor con el valor terminal en el año {fv.N_ANIOS_DCF_CORTO}. En el ejemplo "
            f"(crecimiento de etapa 1 de {_pct(ej['g1'])}): **{_es(ej['dcf'], 1)} USD** (Medio) y **{_es(ej['dcf_pes'], 1)} USD** "
            f"(Pesimista).",
            f"**Múltiplos** ({n_fam['Múltiplos']} modelos: P/E, EV/EBITDA, EV/Ingresos, P/B, P/FCF). *Analogía:* tasar una "
            f"casa por lo que se paga por casas parecidas. El múltiplo objetivo es la **mediana recortada** de los múltiplos "
            f"históricos propios, acotada a rangos fijos (P/E {_es(rg['pe'][0], 0)}-{_es(rg['pe'][1], 0)}, EV/EBITDA "
            f"{_es(rg['ev_ebitda'][0], 0)}-{_es(rg['ev_ebitda'][1], 0)}, EV/Ingresos {_es(rg['ev_revenue'][0], 1)}-"
            f"{_es(rg['ev_revenue'][1], 0)}, P/B {_es(rg['pb'][0], 1)}-{_es(rg['pb'][1], 0)}, P/FCF {_es(rg['p_fcf'][0], 0)}-"
            f"{_es(rg['p_fcf'][1], 0)}) y **mezclada** {_pct(fv.PESO_MULTIPLO_PROPIO, 0)} propio + "
            f"{_pct(1 - fv.PESO_MULTIPLO_PROPIO, 0)} con un benchmark sectorial. Ese benchmark es una **aproximación fijada a "
            f"mano** (no un dato de mercado en vivo), con fecha {fv.BENCHMARK_FECHA}. En el ejemplo, con P/E histórico "
            f"mediano de 15, el P/E da **{_es(ej['pe'], 1)} USD** (Medio) y **{_es(ej['pe_pes'], 1)} USD** (Pesimista, con "
            f"{_pct(cm['pesimista'], 0)} de compresión).",
            f"**Dividendos** ({n_fam['Dividendos']} modelo: DDM de Gordon). Valora la acción por los dividendos que paga. Solo "
            f"aplica a empresas que pagan un dividendo relevante (al menos {_pct(fv.DDM_RENDIMIENTO_MIN, 0)} de rendimiento).",
            f"**Valor** ({n_fam['Valor']} modelos: EPV de Greenwald, Graham y Lynch). Son reglas clásicas, pero NO son "
            f"«conservadoras» en general: tienen sesgos estructurales. En empresas con ROE alto y muchas recompras suelen dar "
            f"muy por debajo del precio (Graham y EPV subestiman negocios con poco capital contable; Lynch usa el crecimiento "
            f"histórico y ignora los dividendos); Graham no aplica con P/B mayor que {_es(fv.GRAHAM_PB_MAX, 0)}. Ejemplo de "
            f"Graham: raíz de {_es(fv.GRAHAM_FACTOR, 1)} x EPS x valor contable = **{_es(ej['graham'], 1)} USD**."]},
        "### El valor justo consolidado y el margen de seguridad",
        f"Se filtran los modelos activos y válidos, con {fv.MIN_MODELOS_IQR} o más se descartan los outliers (IQR) y el "
        f"valor justo es el **promedio que iguala el peso de cada familia**: un tercio DCF, un tercio Múltiplos y un tercio "
        f"Valor / Dividendos (si una familia no tiene modelos, se reparte entre las demás). **Ya no es el promedio simple** "
        f"de los modelos: así los 6 DCF no pesan el doble que los demás. La mediana y el promedio simple se muestran solo "
        f"como información complementaria. Los botones «Solo DCF» y «Solo Múltiplos» muestran visiones distintas del mismo "
        f"negocio y pueden diferir mucho.",
        {"ejemplo": f"Con esos tres números (DCF, P/E y Graham, una por familia) el Escenario Medio es "
                    f"**{_es(agg['fair_value'], 1)} USD** y el Pesimista **{_es(pes['fair_value'], 1)} USD**. El **margen de "
                    f"seguridad** es un descuento voluntario por prudencia (por defecto {_pct(mos, 0)}) sobre el Medio: "
                    f"{_es(agg['fair_value'], 1)} x (1 - {_es(mos, 2)}) = **{_es(agg['margin_of_safety_price'], 1)} USD**. "
                    f"Es el precio al que el modelo todavía deja ese colchón por si los supuestos fallan."},
        f"**Potencial (Upside):** (valor justo - precio) / precio. Positivo = el precio está por debajo del valor "
        f"estimado; negativo = por encima. No es una predicción de que el precio converja. La tarjeta «Diagnóstico» dice "
        f"«Sobrevalorada» si el precio está {_pct(UMBRAL_DIAGNOSTICO, 0)} o más por encima del valor justo Medio, "
        f"«Infravalorada» si está {_pct(UMBRAL_DIAGNOSTICO, 0)} o más por debajo y «En línea» entre ambos.",
        "### Incertidumbre (CV) y outliers",
        f"La **incertidumbre** mide cuánto discrepan los modelos entre sí: es el coeficiente de variación (CV = desviación "
        f"estándar / promedio de los modelos usados). **Baja** si el CV es menor que {_es(fv.CV_BAJO, 2)}, **Media** entre "
        f"{_es(fv.CV_BAJO, 2)} y {_es(fv.CV_ALTO, 2)} y **Alta** si supera {_es(fv.CV_ALTO, 2)} o si quedan menos de "
        f"{fv.MIN_MODELOS_CONFIABLE} modelos. Una incertidumbre alta no es un fallo: indica que los modelos no se ponen de "
        f"acuerdo y que el consolidado vale poco.",
        f"**Outliers:** con {fv.MIN_MODELOS_IQR} o más modelos válidos se descartan los que quedan fuera de "
        f"[Q1 - {_es(fv.IQR_K, 1)} x IQR, Q3 + {_es(fv.IQR_K, 1)} x IQR] (rango intercuartílico). Se marcan «Outlier». "
        f"Un outlier queda excluido mientras haya {fv.MIN_MODELOS_IQR} o más modelos válidos; con menos ya no se filtra. "
        f"Puedes desactivar cualquier modelo con su casilla (si desactivas otros hasta quedar con menos de "
        f"{fv.MIN_MODELOS_IQR}, el filtro deja de aplicarse y el outlier vuelve a contar).",
        "### El gráfico histórico con bandas",
        f"Es una **reconstrucción con el modelo actual**, no un dato publicado. Es **causal**: cada punto usa solo reportes "
        f"ya públicos ({fv.REZAGO_PUBLICACION_DIAS} días de rezago) y se mantiene entre reportes con una transición suave de "
        f"{fv.TRANSICION_DIAS} días (sin mirar al futuro). El último punto es el **ancla**: el valor justo de hoy, que usa el "
        f"último balance y el crecimiento forward de yfinance y es lo único que no es point-in-time. La tasa libre de riesgo, "
        f"la prima de riesgo, la beta, el sector y los sliders son los vigentes en todas las fechas (**no son point-in-time**). "
        f"La proyección a 12 meses (línea punteada) es **ilustrativa**, no una predicción.",
        "### No es lo mismo que los niveles de compra técnicos (y cómo se complementan)",
        "Los **niveles técnicos** (zonas N1-N4) responden *dónde* dejar órdenes límite según estructuras de precio y "
        "volumen. El **valor justo** responde *cuánto parece valer* el negocio según sus cifras. Son preguntas "
        "distintas y no se mezclan en ningún cálculo: el valor justo es aditivo y no mueve las zonas.",
        {"tabla": {"columnas": ["Si ocurre...", "Lectura posible"],
                   "filas": [["Una zona N1 cae por debajo del valor justo con margen",
                              "Coinciden la estructura de precio y la valoración: más respaldo para dejar la orden ahí."],
                             ["Una zona N1 cae por encima del valor justo",
                              "La estructura de precio y la valoración discrepan: conviene revisar los supuestos."],
                             ["El precio está muy por encima del valor justo",
                              "Las zonas técnicas pueden quedar lejos; no implica que el precio vaya a bajar."]]}},
        "### Limitaciones, en corto",
        {"lista": [
            f"**El Escenario Medio puede dar valores muy por encima del precio en negocios de hipercrecimiento.** Con un "
            f"tope de crecimiento de {_pct(tc['medio'], 0)} durante {fv.ANIOS_ETAPA1} años, márgenes de hasta {_pct(tm['medio'], 0)} y "
            f"g∞ de {_pct(gt['medio'])}, el modelo lo permite: en una medición con datos reales de NVDA (precio de unos 149 USD) "
            f"el Medio dio unos 220 USD y los DCF de 10 años entre 397 y 475 USD. Eso es lo que dicen los supuestos, no una "
            f"predicción.",
            f"**El Pesimista es un estrés duro, no un «peor caso realista»** (WACC mínimo {_pct(fv.WACC_PESIMISTA_MIN)}, "
            f"crecimiento topado a {_pct(tc['pesimista'], 0)}, múltiplos {_pct(cm['pesimista'], 0)}): en una empresa defensiva "
            f"como KO dio cerca de un 61 % por debajo del precio. Ambos escenarios son supuestos, no predicciones.",
            f"**Sensibilidad:** el DCF cambia mucho con pequeños cambios del WACC ({_pct(fv.WACC_MIN, 1)} a "
            f"{_pct(fv.WACC_MAX, 1)} en el cálculo automático) y de g∞ ({_pct(fv.CRECIMIENTO_TERMINAL_DEFECTO)} por defecto). "
            f"Por eso la pestaña tiene sliders para probarlo; g no puede superar WACC - {_es(fv.DIF_MIN_WACC_G * 100, 1)} pp.",
            "**El valor terminal domina el DCF:** la mayor parte del valor sale de los flujos después del último año "
            "proyectado, que es lo más incierto.",
            f"**Datos LTM (últimos 12 meses) con rezago:** vienen de 10-K/10-Q, que se publican semanas después del cierre. "
            f"El crecimiento fundamental mezcla {_pct(fv.PESO_CAGR, 0)} del CAGR histórico winsorizado (3 y 5 años), "
            f"{_pct(fv.PESO_MOMENTUM, 0)} de la variación del último año y {_pct(fv.PESO_FORWARD, 0)} del crecimiento esperado de "
            f"yfinance (este último no es point-in-time).",
            "**Sectores donde no aplica:** bancos y aseguradoras (EBITDA y FCF no significan lo mismo; solo se usan P/E, "
            "P/B, DDM, Graham y Lynch), empresas con pérdidas (varios modelos se marcan «No aplica» con su motivo) y "
            f"dividendos simbólicos (menos de {_pct(fv.DDM_RENDIMIENTO_MIN, 0)} de rendimiento).",
            f"**Los modelos no son independientes:** los 6 DCF comparten WACC, g y crecimiento, y Graham, Lynch, P/E y P/B "
            f"salen de EPS y valor contable. N modelos no equivalen a N confirmaciones; por eso la incertidumbre «Baja» "
            f"exige modelos de al menos {fv.MIN_FAMILIAS_BAJA} grupos de familias.",
            f"**Ciclos y commodities:** un flujo muy variable con subidas y bajadas se normaliza (media de los últimos "
            f"{fv.N_NORMALIZACION} años, crecimiento inicial = g) en vez de extrapolar un valle o un pico; también puede "
            f"confundir un hecho puntual (un pago de impuestos extraordinario) con un ciclo. Y con menos de "
            f"{fv.MIN_ANIOS_FCF} años de flujo positivo el DCF de FCF no aplica.",
            "**Sectores no cubiertos:** REITs y utilities no se tratan con FFO ni con su estructura regulada; bancos y "
            "aseguradoras usan solo los modelos de resultados.",
            "**Qué cuenta como flujo, caja y deuda:** el FCF suma la remuneración en acciones (SBC) como si fuera efectivo "
            "(del orden del 6 al 10 % del FCF en las grandes tecnológicas); la deuda excluye arrendamientos y la caja solo "
            "incluye inversiones corrientes (las no corrientes no cuentan); en el DCF de FCF se suman los intereses "
            "después de impuestos al FCF y se resta la deuda aparte.",
            "**Topes por construcción:** cuando un tope está activo (múltiplo, crecimiento, margen, beta o WACC acotados) el "
            "resultado de ese modelo depende del tope y no solo de los datos: la pestaña lo marca como «tope activo».",
            f"**Poco soporte:** con menos de {fv.MIN_MODELOS_CONFIABLE} modelos, o sin DCF, el consolidado tiene soporte "
            f"mínimo; la línea del gráfico y el diagnóstico lo indican («incert. Alta» o «n=2»).",
            f"**Local vs nube:** en tu equipo se usan hasta {fv.ANIOS_LOCAL} años de historia y un caché en disco; en la versión "
            f"web de presentación solo {fv.ANIOS_NUBE} años (visor ligero), con menos reportes para los múltiplos históricos.",
            "**Otros DCF del proyecto:** los pisos de valoración de los niveles técnicos usan un DCF más conservador (tope "
            "de crecimiento inicial y tasa fijos, ver la sección 3) que el de esta pestaña: conviven con supuestos "
            "distintos y no deben compararse como si fueran el mismo.",
            "**No es una predicción ni un precio objetivo**, y no se auditó con backtesting como los niveles técnicos."]}]
    return {"id": "valor_justo", "titulo": "10. Valor justo: qué es, cómo se calcula y cómo se complementa",
            "icono": "balance", "abierta": False, "parrafos": par}


def contenido_modelo() -> list[dict]:
    """Explicacion del modelo para no expertos, como lista de secciones.

    Cada seccion: ``{"id", "titulo", "icono", "abierta", "parrafos"}``; ``parrafos`` mezcla texto (markdown corto) con
    bloques ``{"lista": [...]}``, ``{"tabla": {"columnas", "filas"}}``, ``{"ejemplo": "..."}`` y ``{"aviso": "..."}``.
    Pesos y parametros se leen de ``scoring.cargar_pesos()`` y de las constantes de src/ al llamar (no se hardcodean).
    """
    from src.validation import scoring
    p = _parametros()
    pesos = scoring.cargar_pesos()
    prior = scoring.PESOS_PRIOR
    comps = list(scoring.COMPONENTES)
    norm = normalizacion_componentes(p)
    nombre =lambda k: ETIQUETAS_COMPONENTES.get(k, k)  # noqa: E731
    pct0 = lambda f: _es(f * 100, 0)  # noqa: E731

    # ---- ejemplos numericos calculados con el codigo real
    ej = _ejemplo_capital()
    z1, z2, z3 = ej["zonas"]
    tot_acc = sum(z["acciones"] for z in ej["zonas"])
    comp_ej = {"respaldo_13f_form4": 50.0, "stock_act": None, "absorcion": 26.0, "backtest": 60.0, "confluencia": 40.0}
    ind = scoring.calcular_indice_confiabilidad(comp_ej, pesos)
    filas_ej = []
    for k in comps:
        v = comp_ej.get(k)
        if v is None or k not in ind["detalle"]:
            filas_ej.append([nombre(k), "sin dato", "excluido", "-"])
            continue
        w = ind["detalle"][k]["peso_efectivo"]
        filas_ej.append([nombre(k), _es(v, 0), _pct(w), _es(v * w, 1)])
    falta = [k for k in comps if comp_ej.get(k) is None]
    peso_falta = sum(pesos.get(k, 0.0) for k in falta)
    nombres_falta = " y ".join(nombre(k) for k in falta)
    c_3_0, c_4_2, c_1_0 = scoring.s_confluencia(3, 0), scoring.s_confluencia(4, 2), scoring.s_confluencia(1, 0)
    rank_ej = nucleo.puntaje_ranking(80.0, 40.0)

    secciones: list[dict] = []

    # 1 ---------------------------------------------------------------------------------------------------
    secciones.append({"id": "que_es", "titulo": "1. Qué hace el modelo y qué es un nivel de compra", "icono": "lightbulb",
                      "abierta": True, "parrafos": [
        "**En una frase:** el modelo busca, por debajo del precio actual de una acción, las zonas donde en el pasado se "
        "acumuló mucho volumen y se concentraron compras grandes, y te propone dejar ahí órdenes límite de compra "
        "(órdenes que solo se ejecutan si el precio baja hasta el precio que tú fijaste).",
        "**¿Qué es un nivel de compra?** No es un precio mágico: es una *zona*, un rango de precios (por ejemplo, de 95 a "
        "98 USD) donde históricamente el precio fue aceptado y se negoció mucho. Que una zona exista no garantiza que "
        "el precio llegue hasta ella ni que rebote si llega.",
        "**Regla de invarianza:** ningún nivel depende del cierre de un día cualquiera. Cada zona se apoya en "
        "estructuras acumuladas (volumen de varios años, mínimos semanales importantes, costos medios ponderados por "
        "volumen, bloques de órdenes no perforados y valoraciones de la empresa). Por eso un día de subida o de "
        "bajada de 1 % no mueve las zonas de lugar.",
        f"El precio actual solo decide cuáles zonas quedan «por debajo»: se consideran las que están entre "
        f"{_es(p['dist_min'] * 100)} % y {pct0(p['dist_max'])} % por debajo del precio.",
        {"ejemplo": f"Ejemplo: con la acción a 100 USD, solo se consideran zonas ubicadas entre unos "
                    f"{_es(100 * (1 - p['dist_max']), 0)} y {_es(100 * (1 - p['dist_min']))} USD."}]})

    # 2 (La logica detras, en simple) ---------------------------------------------------------------------
    secciones.append(_seccion_logica(p))

    # 3 ---------------------------------------------------------------------------------------------------
    secciones.append({"id": "familias", "titulo": "3. Las familias de niveles y la confluencia", "icono": "layers",
                      "abierta": False, "parrafos": [
        f"El modelo usa {len(TOGGLES)} «familias» de señales. Cada una mira el mismo precio desde un ángulo distinto:",
        {"lista": [
            f"**HVN / Volume Profile** (familia «volumen»): mira en qué precios se negociaron más acciones durante "
            f"los últimos {p['vp_anios']} años. *Analogía:* el pasillo más transitado de un centro comercial; donde hubo "
            f"más gente hubo más acuerdo entre compradores y vendedores. Incluye el POC (el precio con más volumen), "
            f"el VAL y el VAH (los bordes de la zona que concentra el {pct0(p['value_area'])} % del volumen) y los HVN "
            f"(otros picos de volumen).",
            "**Order Blocks (OB):** velas semanales alcistas que dieron origen a un movimiento fuerte y que el precio "
            "todavía no ha perforado hacia abajo («no mitigadas»), por debajo del precio actual. *Analogía:* el "
            "escalón desde el que arrancó una subida; mientras nadie lo rompa, sigue «vigente». Es un concepto de "
            "*Smart Money* (librería de terceros) sin respaldo académico formal.",
            f"**AVWAP** (precio medio anclado): el precio medio ponderado por volumen desde una fecha clave: el mínimo "
            f"de los últimos 2 años y el mínimo de la última corrección de {pct0(p['correccion'])} % o más. "
            f"*Analogía:* el costo promedio de todos los que compraron desde esa fecha; si el precio vuelve a ese "
            f"costo, esos compradores quedan «en tablas».",
            f"**Swing Lows macro:** mínimos semanales que fueron los más bajos de ±{p['swing_semanas']} semanas "
            f"(unos 2,5 meses). *Analogía:* el fondo de una piscina: un punto donde el precio ya rebotó antes.",
            f"**LogReg** (regresión logarítmica): una línea de tendencia de largo plazo calculada sobre el logaritmo del "
            f"precio (así, subir 10 % pesa igual a 100 USD que a 1000 USD) y bandas a "
            f"-{_es(p['bandas_sd'][0])} y -{_es(p['bandas_sd'][1])} desviaciones estándar. *Analogía:* el carril de "
            f"una carretera; el precio suele ir cerca del centro y la banda baja marca cuándo se alejó mucho hacia abajo.",
            f"**DCF / P/E** (valoración): precios por debajo de los cuales la empresa luce barata respecto a sí misma: "
            f"su P/E y su EV/FCF históricos en los percentiles " + ", ".join(f"P{q}" for q in p["percentiles"]) +
            f", y un DCF conservador a {p['horizonte_dcf']} años con descuento de {pct0(p['tasa_dcf'])} % y margen de "
            f"seguridad de " + " y ".join(f"{pct0(m)} %" for m in p["margenes"]) + ". *Analogía:* el precio de "
            f"«oferta» de un negocio según lo que gana, con descuento extra por prudencia. Estas anclas no entran en el "
            f"cálculo de las zonas (no se pueden auditar sin «mirar el futuro»): se dibujan como líneas doradas y "
            f"cuentan como una familia extra donde caen dentro de una zona (±2 %)."]},
        {"ejemplo": f"**N1 a N{p['max_zonas']}:** las anclas cercanas se agrupan en una zona mientras su ancho no pase de "
                    f"{_es(p['ancho_cluster_atr'], 0)} veces el rango semanal típico de la acción (ATR) ni del "
                    f"{pct0(p['ancho_cluster_pct'])} % del precio. Se guardan como "
                    f"máximo {p['max_zonas']} zonas, numeradas por fuerza: N1 es la que tiene más familias coincidentes; "
                    "si empatan, la que tiene más anclas; si vuelven a empatar, la más cercana al precio. N1 es la más "
                    "respaldada, no necesariamente la más cercana."},
        f"**Por qué las zonas son ahora más estrechas.** Antes un nodo de volumen (HVN) o un Order Block entraba en la "
        f"zona con todo su ancho y a veces una sola ancla ocupaba más del 6 % del precio. Ahora cada ancla se acota a su "
        f"parte más relevante (un HVN, a su pico más o menos 0,5 ATR; un Order Block, a 1 ATR desde su techo), el ancho "
        f"máximo de una zona es {_es(p['ancho_cluster_atr'], 0)} ATR y {pct0(p['ancho_cluster_pct'])} % del precio, y "
        f"las anclas que quedan sobre el precio o que tienen menos de la mitad de su rango dentro de la banda de "
        f"búsqueda se descartan. También se calcula un «núcleo» aún más estrecho de cada zona, pero "
        + ("**está activo como rango operativo**." if p["nucleo_operativo"] else
           "**no se usa como rango operativo** (solo queda disponible en los datos)."),
        {"ejemplo": "**Ejemplo REAL con fecha 2026-11-25 (NVDA, solo ilustra el efecto; no es una recomendación):** "
                    "antes, N1 era 104,51 a 114,81 USD y aparecía además una AVWAP por encima del precio; ahora hay un "
                    "único N1 de 107,35 a 114,81 USD. La zona es más corta y no incluye anclas que el precio ya dejó "
                    "atrás."},
        {"aviso": f"**El costo, honestamente:** una zona más estrecha se toca menos. En la auditoría con 32 tickers el "
                  f"fill rate bajó {_c('fill_32_pp')}; en el pool de 6 no cambió ({_c('fill_pool')}). La calidad medida (retornos "
                  f"y MAE) no empeoró de forma distinguible del ruido, pero tampoco hay una mejora demostrada: la "
                  f"ventaja es precisión del rango, no mejor rendimiento."},
        f"**Menos familias coincidiendo.** Como las anclas son más angostas se solapan menos: las órdenes con 3 o más "
        f"familias pasaron de {_c('fam3_antes')} a {_c('fam3_ahora')}. Los valores altos de confluencia son ahora "
        f"**raros**; lo normal es una zona de 1 o 2 familias y a veces solo hay 1 o 2 zonas por activo. Eso cambia la "
        f"escala del score de confluencia, del reparto de capital por familias y del ranking de la Watchlist.",
        "**Confluencia** es cuántas familias *distintas* coinciden en la misma zona. Ejemplo: una zona de 95 a 98 USD "
        "donde coinciden Volume Profile, AVWAP y un Order Block (3 familias) tiene más respaldo estructural que una "
        "zona de 80 a 82 USD donde solo hay un mínimo semanal (1 familia). Ojo: más coincidencias significan más "
        "respaldo estructural, **no** más probabilidad de rebote (ver Limitaciones).",
        f"**Score de confluencia (0 a 100):** 60 % sale del número de familias (1 familia da 0 puntos y 4 familias dan "
        f"los 60) y 40 % de las anclas de valoración dentro de la zona (2 o más dan los 40). Ejemplos: 1 familia sin "
        f"valoración = {_es(c_1_0, 0)}; 3 familias sin valoración = {_es(c_3_0, 0)}; 4 familias con 2 anclas de "
        f"valoración = {_es(c_4_2, 0)}. «Sin valoración» significa que se consultó y ninguna ancla cae en la zona "
        f"(vale 0). Si la valoración no se pudo consultar (por ejemplo, sin red hacia la fuente de datos), esa parte "
        f"se excluye y el score se reescala con las familias; la app lo avisa («valoracion ... no disponible»)."]})

    # 3 ---------------------------------------------------------------------------------------------------
    secciones.append({"id": "grafico", "titulo": "4. Cómo leer el gráfico", "icono": "candlestick_chart",
                      "abierta": False, "parrafos": [
        {"lista": [
            "**Velas:** cada vela es un día. Verde = cerró más arriba de lo que abrió; roja = cerró más abajo. La línea "
            "punteada gris es la media móvil de 200 días (SMA 200D).",
            "**Bandas de colores horizontales:** son las zonas N1 (verde), N2 (celeste), N3 (naranja) y N4 (violeta). "
            "La etiqueta indica el rango en USD, la «dist. a r_sup» (distancia del techo de la zona, r_sup, que es el "
            "precio de tu orden límite, al precio actual) y las familias que coinciden. Si seleccionas una fila de la "
            "tabla, su banda se resalta.",
            "**Triángulos verdes ▲ (Form 4):** compras de directivos o insiders en mercado abierto (código P), en la "
            "fecha de la operación. Es dinero propio que ellos pusieron.",
            "**Rombos violeta ◆ (13F) y bandas verticales tenues:** trimestres en que al menos un gran fondo ACTIVO "
            "aumentó o abrió posición según su 13F (Vanguard y State Street, pasivos de índice, no cuentan). El 13F "
            "solo informa por trimestre: no dice el día ni el precio de la compra, así que "
            "el rombo se coloca sobre el cierre del trimestre. Sirven para comprobar a ojo si los grandes jugadores "
            "compraron cerca de tus zonas.",
            f"**Panel derecho (Volume Profile):** barras horizontales con el volumen negociado en cada precio durante "
            f"{p['vp_anios']} años, sobre el mismo eje de precios. Las barras naranjas son los HVN (70 % o más del "
            f"máximo); las grises, el resto.",
            "**Líneas doradas punteadas:** precios de valoración (P/E, EV/FCF y DCF; etiquetas como PE_P25 o DCF_MoS30). "
            "La línea blanca es el precio actual.",
            "**Al pasar el ratón** sobre el gráfico aparece un cuadro con la fecha, apertura, máximo, mínimo y cierre, y "
            "el valor de la SMA de ese día."]}]})

    # 4 ---------------------------------------------------------------------------------------------------
    secciones.append({"id": "panel", "titulo": "5. El panel lateral explicado", "icono": "tune",
                      "abierta": False, "parrafos": [
        f"**Ticker:** el símbolo de la acción (por ejemplo NVDA o BRK-B). El menú sugiere los de tu watchlist personalizada y "
        f"también puedes escribir otro.",
        f"**Capital USD:** el dinero que quieres repartir **por ticker**. Solo reciben capital las {p['max_tramos']} "
        f"mejores zonas, repartido en proporción a su número de familias, y cada tramo se compra con una orden límite "
        f"en el techo de la zona (r_sup). **En la pestaña Watchlist (y en `--watchlist`) cada ticker recibe el capital "
        f"completo; no se reparte entre tickers:** si se ejecutaran las órdenes de 10 activos harían falta 10 veces ese "
        f"capital. El presupuesto total lo decides tú.",
        {"ejemplo": f"Ejemplo con {_es(ej['capital'], 0)} USD y 3 zonas de 3, 2 y 1 familias (6 en total): la primera "
                    f"recibe 3/6 = {_es(z1['capital_asignado'], 2)} USD, la segunda 2/6 = "
                    f"{_es(z2['capital_asignado'], 2)} USD y la tercera 1/6 = {_es(z3['capital_asignado'], 2)} USD. "
                    "Las acciones salen de dividir cada monto por el techo de su zona:"},
        {"tabla": {"columnas": ["Zona", "Familias", "Capital (USD)", "Techo r_sup (USD)", "Acciones"],
                   "filas": [[f"N{z['nivel']}", str(z["n_familias"]), _es(z["capital_asignado"], 2), _es(z["r_sup"], 2),
                              _es(z["acciones"], 4)] for z in ej["zonas"]] +
                            [["Total", "6", _es(ej["capital"], 2), "-", _es(tot_acc, 4)]]}},
        f"El **precio promedio ponderado** es el capital total dividido entre las acciones totales: "
        f"{_es(ej['capital'], 0)} / {_es(tot_acc, 4)} = {_es(ej['promedio'], 2)} USD. Solo sería el precio real de "
        f"compra si se ejecutaran las tres órdenes; en la práctica a veces se ejecutan solo algunas. Cambiar el "
        f"capital reparte de nuevo al instante, sin recalcular.",
        "**Interruptores de familias:** cada uno enciende o apaga una familia en el gráfico y en la tabla.",
        {"lista": [f"**{t['label']}:** {t['ayuda']}" for t in TOGGLES.values()]},
        "**¿Qué pasa al apagar una familia?** Deja de contarse y de dibujarse. Una zona que solo tenía esa familia "
        "desaparece; una zona con varias pierde una (por ejemplo pasa de 3 a 2), las zonas se reordenan y renumeran "
        "(N1, N2...) y el capital se reparte otra vez. **Los límites de la zona no se recalculan:** siguen siendo los del "
        "grupo completo de anclas. Por eso no hace falta pulsar Recalcular.",
        "**Backtest (lento)** y **SEC: Form 4 / 13F:** deciden qué datos se piden al calcular. Sin backtest, Bounce, Fill "
        "y MAE salen n/d y la confiabilidad pierde los componentes Backtest y Absorción (ambos salen del backtest). "
        "Sin SEC no hay marcadores de insiders ni 13F, y la confiabilidad pierde los componentes de respaldo y STOCK "
        "Act. Los componentes que faltan se excluyen y el resto se reescala (ver «cobertura»). Sus cambios solo se "
        "aplican al pulsar Recalcular.",
        {"tabla": {"columnas": ["Botón Recalcular", "¿Cuándo?"],
                   "filas": [["Necesario", "La primera vez que analizas un ticker; al cambiar los interruptores "
                                           "Backtest o SEC; o para traer datos más recientes."],
                             ["No necesario", "Al cambiar el capital, al encender o apagar familias (DCF / P/E "
                                              "incluido), al seleccionar una fila o al volver a un ticker ya calculado "
                                              "en esta sesión (queda en memoria hasta cerrar la app)."]]}}]})

    # 5 ---------------------------------------------------------------------------------------------------
    secciones.append({"id": "metricas", "titulo": "6. Las métricas de la tabla", "icono": "table_chart",
                      "abierta": False, "parrafos": [
        {"lista": [
            f"**Fill Rate (tasa de ejecución):** de las órdenes límite colocadas, cuántas llegaron a ejecutarse. "
            f"Ejemplo: de 10 órdenes colocadas, 3 se ejecutaron = 30 %. Cada orden vive {p['vida_orden']} ruedas "
            f"(unos 3 meses) y se ejecuta si el mínimo del día toca el techo de la zona.",
            f"**Bounce Rate (tasa de rebote):** de las órdenes *ejecutadas*, cuántas subieron +{pct0(p['bounce_pct'])} % "
            f"sobre tu precio de entrada, dentro de {p['bounce_ventana']} ruedas desde la ejecución, *antes* de caer "
            f"{pct0(p['bounce_inval'])} % por debajo del piso de la zona (r_inf). Ejemplo: entras a 100 con piso de la "
            f"zona en 96; la meta es 105 y la invalidación es {_es(96 * (1 - p['bounce_inval']), 2)} "
            f"(96 menos {pct0(p['bounce_inval'])} %). Si en {p['bounce_ventana']} ruedas toca 105 antes que "
            f"{_es(96 * (1 - p['bounce_inval']), 2)}, cuenta como rebote. Si el mismo día toca ambos niveles, cuenta "
            f"como fallo.",
            "**MAE (máxima excursión adversa):** cuánto cayó el precio en tu contra después de comprar. Ejemplo: compré a "
            "100 y el mínimo posterior fue 90 = MAE de -10 %. La tabla muestra la *mediana* a 60 ruedas: la mitad de "
            "las compras simuladas cayó más que esa cifra.",
            "**Retornos a 60 / 120 / 252 ruedas:** cuánto ganó o perdió la compra pasado ese tiempo (60 ruedas = unos 3 "
            "meses; 252 = 1 año bursátil). Ejemplo: compro a 100 y 60 ruedas después el cierre es 106 = +6 %. Esta "
            "tabla no tiene columnas de retorno: el retorno mediano a 60 ruedas pesa dentro de la Confiabilidad y "
            "los tres horizontes se comentan en la auditoría (docs/METODOLOGIA.md).",
            "**Dist. a r_sup (%):** distancia del techo de la zona (r_sup, el precio de tu orden límite GTC) al precio "
            "actual. Ejemplo: precio 100 y techo 95 = -5 % (la zona está 5 % por debajo). Es la misma referencia en la "
            "tabla, el gráfico, la consola y el export.",
            "**Capital, Acciones y Precio entrada:** el capital asignado a la zona, cuántas acciones compra (capital / "
            "techo) y el precio de la orden límite (el techo de la zona).",
            "**Precio prom. pond.** (tarjeta superior): capital total / acciones totales, como en el ejemplo de la "
            "sección 5."]},
        {"aviso": f"Bounce, Fill rate y MAE 60d son del *rango primario* (N1) del ticker, no de cada fila, y salen de un "
                  f"backtest de {p['anios_bt_app']} años: se repiten igual en todas las filas y **no son comparables con "
                  f"el pool de la auditoría** (docs/METODOLOGIA.md: {len(p['tickers_bt'])} acciones, {p['anios_bt_ref']} años). Si el "
                  f"backtest tiene menos de {p['min_fills']} ejecuciones simuladas, la app avisa «muestra pequeña: N "
                  f"fills» y esas cifras son ruido."}]})

    # 6 ---------------------------------------------------------------------------------------------------
    tabla_pesos = {"columnas": ["Componente", "Peso a priori", "Peso calibrado (el que se usa)", "Cómo se normaliza a 0-100"],
                   "filas": [[nombre(k), _pct(prior[k]) if k in prior else "n/d",
                              _pct(pesos[k]) if k in pesos else "n/d", norm.get(k, "")]
                             for k in comps]}
    par_conf: list[Any] = [
        "**¿Qué es?** Un número de 0 a 100 que resume cuánto respaldo tiene la zona principal (flujos institucionales, "
        "comportamiento histórico y coincidencias). Es un *resumen estandarizado*, **no** una probabilidad de ganar.",
        "Es un promedio ponderado de cinco componentes. Estos son sus pesos reales (leídos ahora de "
        "`src/validation/pesos_calibrados.json`):",
        {"tabla": tabla_pesos},
        f"Los pesos parten de valores a priori y se mezclan {pct0(1 - scoring.MEZCLA_EMPIRICA)}/{pct0(scoring.MEZCLA_EMPIRICA)} "
        f"con una versión ajustada según la evidencia del backtest. Como esa evidencia no resultó favorable (ver "
        f"Limitaciones), los calibrados quedan cerca de los a priori.",
        f"**Cobertura:** es la parte del peso total que tiene datos. Si un componente no tiene dato (por ejemplo, sin "
        f"red hacia SEC o sin backtest) se **excluye** y los pesos de los demás se reescalan para volver a sumar "
        f"100 %. Cobertura = suma de pesos disponibles / suma de todos los pesos. Se muestra debajo de la tarjeta "
        f"«Confiabilidad inst.», junto con los componentes que quedaron excluidos («sin dato: ...»).",
        "**Un dato no disponible no es un cero ni un valor neutro.** Si una consulta falla (Form 4 sin respuesta, "
        "gestores 13F que no contestaron, STOCK Act sin conexión, valoración sin consultar), ese dato se excluye (el "
        "resto se reescala) y la cobertura baja; en cambio un dato obtenido que vale cero (0 USD de compras de "
        "insiders, 0 gestores activos con posición, ninguna ancla de valoración en la zona) sí es dato y puntúa cero. "
        "Si no respondió ningún gestor activo (aunque sí respondan los pasivos), el 13F cuenta como no disponible. "
        "Los gestores que no respondieron se listan en Avisos («13f: no respondieron: ...»).",
        f"**13F: solo cuentan los gestores activos.** El screener consulta {len(p['gestores']['activos']) + len(p['gestores']['pasivos'])} "
        f"gestores (leídos de `src/data/sec_edgar.py`). Activos, que cuentan para la acumulación: "
        f"{_lista_nombres(p['gestores']['activos'])}. Pasivos de índice, que se consultan y se muestran pero no cuentan: "
        f"{_lista_nombres(p['gestores']['pasivos'])}. Es una muestra muy pequeña de fondos, no el universo 13F.",
        {"ejemplo": f"Ejemplo con valores inventados: sin datos de {nombres_falta} (peso {_pct(peso_falta)}), la "
                    f"cobertura es {_pct(ind['cobertura'])}. Cada componente con dato entra con su peso efectivo "
                    f"(su peso dividido entre la suma de los pesos disponibles) y el índice es la suma de los "
                    f"aportes: **{_es(ind['indice'], 1)}**."},
        {"tabla": {"columnas": ["Componente", "Valor (0-100)", "Peso efectivo", "Aporte"], "filas": filas_ej}},
        f"**Score de ranking (pestaña Watchlist):** es el promedio simple de la confluencia (0-100) y la confiabilidad "
        f"(0-100); si falta una de las dos, se usa la que haya. Ejemplo: confluencia 80 y confiabilidad 40 dan un "
        f"score de {_es(rank_ej, 0)}. Sirve para ordenar la lista, no para predecir. Dos scores con cobertura distinta "
        f"(por ejemplo, uno con backtest y otro sin él) no son comparables.",
        "**Score de confluencia:** se explica en la sección 3 y es una de las dos mitades del score de ranking."]
    secciones.append({"id": "confiabilidad", "titulo": "7. La Confiabilidad (índice de 0 a 100)", "icono": "verified",
                      "abierta": False, "parrafos": par_conf})

    # 7 ---------------------------------------------------------------------------------------------------
    evid = _tabla_evidencia()
    w_rank = nucleo.peso_confluencia_ranking(pesos)
    ev_confl = next((f[1] for f in (evid or {}).get("filas", []) if f[0] == nombre("confluencia")), "")
    lim: list[Any] = [
        {"aviso": AVISO_LEGAL},
        {"lista": [
            f"**La Confiabilidad NO es una probabilidad ni ha mostrado poder predictivo sólido.** En la calibración "
            f"propia ({_c('n_calibracion')} órdenes ejecutadas) la correlación de Spearman entre el score y el retorno "
            f"a 60 ruedas fue {_c('spearman_ret60')} (débil y con t inflado por el solapamiento de órdenes), pero "
            f"con el bounce fue {_c('spearman_bounce')}: el bounce y la MAE a 60 ruedas empeoran al subir el score. "
            f"El tercio de score más alto tuvo un bounce de {_c('tercil_alto_bounce')} y el más bajo de "
            f"{_c('tercil_bajo_bounce')}; el retorno medio a 60 ruedas fue {_c('tercil_alto_ret60')} contra "
            f"{_c('tercil_bajo_ret60')}. El score no discrimina resultados en conjunto y su relación con el retorno "
            f"es débil y no robusta; los pesos calibrados no deben leerse como evidencia de poder predictivo.",
            f"**Límite abierto de la absorción (posible look-ahead):** el peso de la absorción "
            f"({_pct(pesos.get('absorcion', 0.0))}) sale de una calibración en la que la señal se mide en las 5 velas "
            f"posteriores al llenado, es decir *después* de entrar; no es una señal disponible al colocar la orden y "
            f"podría contener look-ahead. Comprobación del auditor midiendo desde el cierre de la 6.ª rueda: retorno "
            f"medio con absorción {_c('abs_ret_con')} contra {_c('abs_ret_sin')} sin ella (t≈1,9, p≈{_c('abs_p')}): "
            f"sugerente, no concluyente. La fórmula no se cambió.",
            f"**Confluencia rara:** solo {_c('fam3_ahora')} de las órdenes tienen 3 o más familias (antes "
            f"{_c('fam3_antes')}), así que el score de confluencia rara vez es alto y el reparto por familias y el "
            f"ranking cambiaron de escala.",
            f"**Retornos:** comprar en el rango mejora la *media* frente a comprar sin criterio (+{_c('media_pp60')[1:]} "
            f"puntos a 60 ruedas y +{_c('media_pp252')[1:]} a 252), pero la **mediana a 252 ruedas es casi idéntica a la "
            f"línea base** ({_c('mediana252_fill')} contra {_c('mediana252_base')}) y la **MAE es sistemáticamente "
            f"peor**: mediana a 60 ruedas {_c('mae60_mediana')} contra {_c('mae60_base')} de la base, y a 252 ruedas "
            f"{_c('mae252_mediana')} contra {_c('mae252_base')}. Es decir, se compra tras una caída y el precio sigue "
            f"cediendo antes de rebotar. OXY mostró mediana negativa a 60, 120 y 252 ruedas. La ventaja se concentra en "
            f"las medias y en META, GOOGL, AAPL y MSFT (tendencia alcista del período); no hay evidencia estadística "
            f"independiente de alfa por sí sola.",
            f"**Grandes fondos (13F):** los rangos calculados solo con datos previos **no coinciden mejor que el azar** "
            f"con las zonas donde compraron los fondos: solape rodante {_c('solape_rodante')} contra "
            f"{_c('solape_control')} de un rango aleatorio ({_c('n_trimestres_13f')} trimestres con compra neta) y "
            f"solape estricto {_c('solape_estricto')}. Los fondos compraron cerca del mercado y en tendencia; el "
            f"algoritmo propone zonas estructurales profundas. Además, el 13F no trae fecha de ejecución, llega con "
            f"hasta 45 días de retraso y no incluye posiciones cortas ni fuera de EE. UU.",
            f"**Las métricas de auditoría de esta app son del rango primario (N1) del ticker y se repiten en todas las "
            f"filas** (no son por zona). Además salen de un backtest de {p['anios_bt_app']} años de un solo ticker: "
            f"como máximo unas 60 órdenes mensuales que se solapan entre sí, una muestra pequeña y ruidosa, **no "
            f"comparable con el pool de la auditoría**. Con menos de {p['min_fills']} ejecuciones simuladas la app "
            f"avisa «muestra pequeña: N fills».",
            f"**El ranking de la Watchlist es una ayuda de organización, no una señal.** Su score está dominado por la "
            f"confluencia: pesa ≈{_es(w_rank * 100, 1)} % del score con cobertura completa (50 % directo más la parte que la "
            f"confluencia tiene dentro de la confiabilidad, {_pct(pesos.get('confluencia', 0.0))}; más si faltan otros "
            f"componentes). Y la evidencia propia no respalda que más familias coincidiendo mejoren el retorno"
            + (f" ({ev_confl})" if ev_confl else "") + ". Úsalo solo para decidir qué ticker revisar primero.",
            f"**Sesgo de la muestra:** la auditoría de referencia usó {len(p['tickers_bt'])} acciones de gran "
            f"capitalización ({', '.join(p['tickers_bt'])}) durante {p['anios_bt_ref']} años mayormente alcistas "
            f"({_c('n_ordenes')} órdenes válidas, {_c('n_fills')} ejecutadas): hay sesgo de supervivencia (se eligieron "
            f"ganadoras). Las órdenes mensuales de {p['vida_orden']} ruedas se solapan, así que no son observaciones "
            f"independientes y los estadísticos t están inflados.",
            "**Lo que no se midió:** los pisos de valoración (P/E, EV/FCF, DCF) y el AVWAP anclado al 10-K no se "
            "auditaron porque no son *point-in-time* (usan datos que no existían en cada fecha): su aporte empírico "
            "es desconocido. Form 4 y STOCK Act tampoco se probaron empíricamente por falta de series históricas.",
            f"**Parámetros fijados a priori:** el umbral de bounce (+{pct0(p['bounce_pct'])} % en {p['bounce_ventana']} "
            f"ruedas, invalidación en -{pct0(p['bounce_inval'])} %), la ventana de absorción ({p['velas_abs']} velas: la del toque y las {p['ventana_abs']} siguientes) y "
            f"las ponderaciones de normalización no fueron optimizados.",
            "**No modelado:** costos de transacción, impuestos y liquidez. Los precios están ajustados por dividendos y "
            "splits, y el ajuste por dividendos usa información posterior (distorsión pequeña). Sin red, SEC EDGAR y "
            "STOCK Act no están disponibles.",
            f"**Cifras de referencia** (pool de la auditoría): fill rate {_c('fill_pool')}, absorción "
            f"{_c('absorcion_pool')} y bounce {_c('bounce_pool')}; son promedios de seis acciones y no se pueden "
            f"extrapolar a un ticker concreto."]}]
    if evid:
        lim.append("**Lo que midió cada componente en la calibración** (es la evidencia detrás de los pesos de la "
                   "sección 7):")
        lim.append({"tabla": evid})
    lim.append(f"Fuente: `docs/METODOLOGIA.md` (cifras con datos al {FECHA_AUDITORIA}). Para reproducirlas: "
               f"`python -m src.validation.backtest_ranges`, `python -m src.validation.case_studies` y "
               f"`python -m src.validation.scoring`.")
    secciones.append({"id": "limitaciones", "titulo": "8. Limitaciones honestas", "icono": "report_problem",
                      "abierta": True, "parrafos": lim})
    secciones.append({"id": "watchlist", "titulo": "9. Tu watchlist: qué es y qué calcula el escaneo", "icono": "star",
                      "abierta": False, "parrafos": [
        "**Qué es:** la lista de acciones que sigues. Se arma con el botón de estrella de la pestaña Análisis o "
        "escribiendo tickers en la pestaña Watchlist (también puedes cargar la lista de ejemplo y quitar lo que no te "
        "interese).",
        "**Qué calcula el escaneo:** el botón Escanear corre el análisis completo SOLO de los tickers de tu lista "
        "(los marcados) y los ordena por Score de ranking. No calcula ningún otro activo. El backtest es opcional "
        "porque lo hace mucho más lento (y en la nube viene desactivado).",
        "**Capital:** el capital del panel lateral es POR TICKER; no se reparte entre los activos de la lista.",
        "**Dónde se guarda:** en un archivo de tu equipo (carpeta de datos de la aplicación), no en el repositorio. En "
        "un servidor en la nube (por ejemplo Render, plan gratuito) el disco es efímero: la lista personalizada no "
        "persiste entre reinicios."]})
    secciones.append(_seccion_valor_justo())
    return secciones


def _quitar_md(texto: str) -> str:
    return re.sub(r"[*`]", "", texto)


def texto_seccion(seccion: dict) -> str:
    """Texto plano de una seccion (titulo + todos los bloques), sin marcas markdown; util para tests y busquedas."""
    partes = [seccion["titulo"]]
    for b in seccion["parrafos"]:
        if isinstance(b, str):
            partes.append(b)
        elif "lista" in b:
            partes.extend(b["lista"])
        elif "tabla" in b:
            partes.append(" | ".join(b["tabla"]["columnas"]))
            partes.extend(" | ".join(f) for f in b["tabla"]["filas"])
        else:
            partes.append(next(iter(b.values())))
    return _quitar_md("\n".join(partes))


def texto_modelo(secciones: list[dict] | None = None) -> str:
    return "\n".join(texto_seccion(s) for s in (secciones if secciones is not None else contenido_modelo()))


def _esc_md(texto: str) -> str:
    """Escapa '_' entre letras/digitos (markdown2 lo toma por enfasis: r_sup -> r<em>sup) salvo dentro de `codigo`."""
    partes = texto.split("`")
    return "`".join(p if i % 2 else re.sub(r"(?<=\w)_(?=\w)", r"\\_", p) for i, p in enumerate(partes))


def _md_tabla(tabla: dict) -> str:
    esc = lambda s: _esc_md(str(s).replace("|", "\\|").replace("\n", " "))  # noqa: E731
    cab = "| " + " | ".join(esc(c) for c in tabla["columnas"]) + " |"
    sep = "|" + "|".join(" --- " for _ in tabla["columnas"]) + "|"
    return "\n".join([cab, sep, *("| " + " | ".join(esc(c) for c in f) + " |" for f in tabla["filas"])])


def renderizar_modelo(secciones: list[dict]) -> None:
    """Capa minima de dibujo de ``contenido_modelo()``: aviso fijo arriba y una expansion por seccion."""
    from nicegui import ui
    with ui.column().classes("w-full gap-3").style("max-width: 1000px"):
        with ui.card().classes("aviso-card w-full"):
            with ui.row().classes("items-center no-wrap gap-3"):
                ui.icon("warning_amber", size="md").classes("txt-aviso")
                ui.markdown(AVISO_LEGAL + " Esta pestaña solo explica el modelo; no calcula nada.")
        for sec in secciones:
            with ui.expansion(sec["titulo"], icon=sec.get("icono"), value=bool(sec.get("abierta"))) \
                    .classes("modelo-sec w-full").props(f"data-seccion={sec['id']}"):
                with ui.column().classes("w-full gap-2 q-pa-sm"):
                    for b in sec["parrafos"]:
                        if isinstance(b, str):
                            ui.markdown(_esc_md(b))
                        elif "lista" in b:
                            ui.markdown("\n".join(f"- {_esc_md(i)}" for i in b["lista"]))
                        elif "tabla" in b:
                            ui.markdown(_md_tabla(b["tabla"]))
                        elif "ejemplo" in b:
                            with ui.element("div").classes("ejemplo-box w-full"):
                                ui.markdown(_esc_md(b["ejemplo"]))
                        elif "aviso" in b:
                            with ui.element("div").classes("aviso-box w-full"):
                                ui.markdown(_esc_md(b["aviso"]))


# --- Capa de widgets (NiceGUI) -----------------------------------------------------------------------------
# ticker -> ultimo resultado de analizar_ticker (compartido entre pestanas); LRU de 3 en la nube (memoria)
CACHE: CacheLRU = CacheLRU(MAX_CACHE_NUBE if userdata.en_nube() else None)
CACHE_SERIES: CacheLRU = CacheLRU(MAX_SERIES_NUBE if userdata.en_nube() else MAX_SERIES_LOCAL)  # serie historica por huella
ESCANEO_NUBE = {"activo": False}  # un solo escaneo a la vez en la nube (compartido entre clientes)

_PLANTILLA_NUM = ("<q-td :props=\"props\">{{ props.value == null ? 'n/d' : %s }}</q-td>")


def _slot(expr: str) -> str:
    return _PLANTILLA_NUM % expr


TEXTO_WATCHLIST_INTRO = (
    "Tu watchlist es la lista de acciones que sigues. Añade las que te interesan (aquí o con la estrella en Análisis).",
    "El botón Escanear calcula SOLO las de tu lista (las marcadas) y las ordena por puntaje; no toca ninguna otra.",
    "Cada ticker recibe el capital completo del panel lateral. La lista se guarda en tu equipo (en la nube no persiste).",
)


def _cabecera_con_ayuda(texto: str) -> str:
    """Plantilla de cabecera de columna con tooltip (mantiene el orden por clic de Quasar)."""
    return f"<q-th :props=\"props\">{{{{ props.col.label }}}}<q-tooltip>{texto}</q-tooltip></q-th>"


# Celda de cobertura del ranking: porcentaje y, si hay componentes excluidos, un tooltip "Sin dato: ...".
_PLANTILLA_COBERTURA = ("<q-td :props=\"props\">{{ props.value == null ? 'n/d' : props.value.toFixed(0) + '%' }}"
                        "<q-tooltip v-if=\"props.row.sin_dato\">Sin dato: {{ props.row.sin_dato }}</q-tooltip></q-td>")


class Vista:
    """Estado y widgets de una pagina; los calculos pesados no bloquean el bucle de eventos."""

    def __init__(self) -> None:
        from nicegui import ui
        self.ui = ui
        self.items = userdata.leer()  # watchlist personalizada [{ticker, nota}]; el CSV del repo es solo un ejemplo
        self.watchlist = userdata.tickers(self.items)
        self.sel_wl: dict[str, bool] = {}  # casillas de la pestana Watchlist (solo se escanean las marcadas)
        # El select arranca mostrando el primer ticker de la watchlist: el estado interno debe coincidir con lo que
        # se ve (si no, la pagina muestra "NVDA" pero analiza/refresca "nada" hasta el primer cambio manual).
        self.ticker: str | None = validar_ticker(self.watchlist[0]) if self.watchlist else None
        self.nivel_sel: int | None = None
        self.activos: dict[str, bool] = {k: True for k in TOGGLES}
        self.ocupado = False
        self.detener = False
        # Valor justo: TODO el estado vive aqui (la entrada liviana res["entradas_valoracion"] se cachea; sin red)
        self.vj_params = fv.ParametrosValoracion()
        self.vj_activos: dict[str, bool] = {i: True for i in fv.IDS_MODELOS}
        self.vj_ticker: str | None = None           # ticker y dict de entradas (por identidad) ya cargados
        self.vj_dict: dict | None = None
        self.vj_gen = 0                             # sube cada vez que cambian las entradas (parte de la huella de la serie)
        self.vj_entradas: fv.EntradasValoracion | None = None
        self.vj_esc: dict | None = None             # ultimo ``fv.calcular_escenarios`` (Medio, Pesimista, modelos, supuestos)
        self.vj_agg: dict | None = None             # = vj_esc["medio"]: unica fuente de KPIs, casillas, barras y linea de Analisis
        self.vj_serie: dict | None = None           # ultima serie historica pintada (``fv.serie_valor_justo_historica``)
        self._vj_seq_serie = 0                      # version de la peticion de la serie (descarta respuestas atrasadas)
        self._vj_expl_firma: tuple | None = None    # firma de lo que muestra el acordeon de explicacion de modelos
        self.n_resize_vj = 0                        # cuantas veces se pidio ``resize()`` a los graficos ECharts (pruebas)
        self.vj_wacc_tocado = False
        self._vj_silencio = False                   # True mientras el codigo (no el usuario) mueve controles
        self._vj_seq = 0                            # numero de version del debounce de los sliders
        self.vj_rows: dict[str, dict] = {}
        # Grafico de Analisis (Lightweight Charts): ultimo payload preparado (testeable sin navegador)
        self.anios_velas: int | None = None         # None = 10 anios en local, 3 en la nube (unico parametro de historial)
        self.grafico_datos: dict | None = None      # velas, SMA, Volume Profile y marcadores (pesado; solo si cambian)
        self.grafico_overlays: dict | None = None   # zonas, anclas, precio, valor justo medio y titulo (ligero)
        self._grafico_huella: tuple | None = None   # clave de grafico_datos ya enviada al navegador
        self.ultimo_js_grafico: str = ""            # ultima sentencia JS enviada (diagnostico y pruebas)

    # ---- construccion
    def _ayuda(self, clave: str) -> None:
        """Icono de ayuda (info) con tooltip corto legible en oscuro."""
        ic = self.ui.icon("info", size="xs").classes("ayuda-icono").props(f"data-ayuda={clave} tabindex=0")
        ic.tooltip(AYUDAS[clave])

    def construir(self) -> None:
        ui = self.ui
        ui.dark_mode(True)
        ui.colors(**COLORES_QUASAR)
        ui.add_head_html(HEAD_TEMA)
        ui.add_head_html("".join(f'<script src="{u}"></script>' for u in SCRIPTS_GRAFICO))  # LW local: sin CDN
        ui.add_css(css_tema())
        self.barra = ui.linear_progress(show_value=False).props("indeterminate color=amber").classes("w-full")
        self.barra.visible = False
        with ui.left_drawer(fixed=True, bordered=True).props("width=300").classes("gap-2"):
            ui.label("Niveles institucionales").classes("text-h6")
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                self.sel_ticker = ui.select(
                    self.watchlist, value=self.ticker, label="Ticker", with_input=True, new_value_mode="add-unique", clearable=True,
                    on_change=lambda e: self._ticker_cambiado(e.value)).classes("grow")
                self._ayuda("ticker")
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                self.capital = ui.number("Capital USD", value=1000, min=0, format="%.2f", prefix="$",
                                         validation={"Debe ser > 0": validar_capital},
                                         on_change=lambda e: self.refrescar()).classes("grow")
                self._ayuda("capital")
            ui.separator()
            ui.label("Familias de niveles").classes("text-caption txt-muted")
            for clave, t in TOGGLES.items():
                with ui.row().classes("w-full items-center no-wrap justify-between"):
                    ui.switch(t["label"], value=True, on_change=lambda e, k=clave: self._toggle(k, e.value))
                    self._ayuda(clave)
            ui.separator()
            with ui.row().classes("w-full items-center no-wrap justify-between"):
                self.sw_vj = ui.switch("Mostrar Valor Justo", value=False, on_change=lambda e: self._toggle_vj(e.value))
                self._ayuda("valor_justo")
            with ui.row().classes("w-full items-center no-wrap justify-between"):
                self.sw_bt = ui.switch("Backtest (lento)", value=True)
                self._ayuda("backtest")
            with ui.row().classes("w-full items-center no-wrap justify-between"):
                self.sw_sec = ui.switch("SEC: Form 4 / 13F", value=True)
                self._ayuda("sec")
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                self.btn = ui.button("Recalcular", icon="refresh", on_click=self.recalcular).classes("grow")
                self._ayuda("recalcular")
        self.tabs = ui.tabs().classes("w-full")
        with self.tabs:
            t_an = ui.tab("analisis", label="Análisis")
            t_vj = ui.tab("valor_justo", label="Valor Justo")
            t_wl = ui.tab("watchlist", label="Watchlist")
            t_md = ui.tab("modelo", label="Modelo")
        self.tabs.on_value_change(lambda e: self._tab_cambiada(e.value))   # ECharts oculto mide 0 px: resize al mostrarlo
        with ui.tab_panels(self.tabs, value=t_an).classes("w-full bg-transparent"):
            with ui.tab_panel(t_an):
                self._panel_analisis()
            with ui.tab_panel(t_vj):
                self._panel_valor_justo()
            with ui.tab_panel(t_wl):
                self._panel_watchlist()
            with ui.tab_panel(t_md):
                renderizar_modelo(contenido_modelo())
        self.refrescar()

    def _panel_analisis(self) -> None:
        ui = self.ui
        with ui.row().classes("w-full items-center gap-2"):
            self.btn_wl = ui.button("Agregar a mi watchlist", icon="star_border", on_click=self.alternar_watchlist) \
                .props("outline color=amber data-accion=watchlist-toggle")
            self.btn_wl.tooltip("Guarda este ticker en tu lista personalizada (pestaña Watchlist); en la nube no persiste.")
            self._actualizar_btn_wl()
        with ui.row().classes("w-full gap-3 items-stretch no-wrap overflow-auto"):
            self.tarj = {}
            for clave, titulo in (("precio", "Precio actual"), ("confluencia", "Confluencia"),
                                  ("confiabilidad", "Confiabilidad inst."), ("promedio", "Precio prom. pond.")):
                with ui.card().classes("min-w-[150px] p-3"):
                    ui.label(titulo).classes("text-caption txt-muted")
                    self.tarj[clave] = ui.label("n/d").classes("text-h6")
                    if clave == "confiabilidad":
                        self.tarj_cob = ui.label("").classes("text-caption txt-muted")
                        self.tarj_cob.tooltip("Cobertura: parte del peso de la Confiabilidad que tenía dato. Los "
                                              "componentes sin dato se excluyen: no cuentan como 0 ni como neutro.")
                        self.tarj_excl = ui.label("").classes("text-caption txt-aviso") \
                            .style("max-width: 210px; white-space: normal")
            with ui.card().classes("grow p-3"):
                ui.label("Avisos").classes("text-caption txt-muted")
                self.avisos = ui.column().classes("gap-0")
        # Grafico principal: contenedor vacio; static/charts/grafico_analisis.js crea ahi el grafico Lightweight Charts
        self.grafico = ui.element("div").classes("w-full").style("height: clamp(540px, 78vh, 860px)") \
            .props("data-grafico=analisis")
        self.grafico_id = self.grafico.html_id
        with ui.row().classes("w-full items-center gap-1 no-wrap"):
            ui.label("Rueda: zoom | arrastrar: mover | arrastrar el eje de precios: escala | doble clic en el eje: autoajuste.") \
                .classes("text-caption txt-muted")
            ui.link(ATRIBUCION_LW, URL_TRADINGVIEW, new_tab=True).classes("text-caption").style("margin-left:auto") \
                .props("data-ng-atribucion=tradingview")
        self.lbl_bt = ui.label(texto_alcance_backtest({})).classes("text-caption txt-muted")
        self.tabla = ui.table(columns=[
            dict(name="nivel", label="Nivel", field="nivel", sortable=True, align="left"),
            dict(name="rango", label="Rango inf - sup", field="r_inf", sortable=True, align="left"),
            dict(name="ancho_pct", label="Ancho %", field="ancho_pct", sortable=True),
            dict(name="dist_pct", label=f"{ETIQUETA_DIST_RSUP} %".capitalize(), field="dist_pct", sortable=True),
            dict(name="familias", label="Familias", field="familias", sortable=True, align="left"),
            dict(name="bounce", label="Bounce", field="bounce", sortable=True),
            dict(name="fill_rate", label="Fill rate", field="fill_rate", sortable=True),
            dict(name="mae", label="MAE 60d", field="mae", sortable=True),
            dict(name="capital", label="Capital", field="capital", sortable=True),
            dict(name="acciones", label="Acciones", field="acciones", sortable=True),
            dict(name="entrada", label="Precio entrada", field="entrada", sortable=True)],
            rows=[], row_key="id", selection="single",
            on_select=self._fila_seleccionada).props("dense flat bordered dark no-data-label=\"Sin zonas: elige un "
                                                     "ticker y pulsa Recalcular\"").classes("w-full")
        self.tabla.add_slot("header-cell-dist_pct", _cabecera_con_ayuda(
            "Distancia % de r_sup (precio del límite GTC) al precio actual; negativo = por debajo."))
        self.tabla.add_slot("body-cell-rango", "<q-td :props=\"props\">${{ props.row.r_inf.toFixed(2) }} - "
                                               "${{ props.row.r_sup.toFixed(2) }}</q-td>")
        self.tabla.add_slot("header-cell-ancho_pct", _cabecera_con_ayuda(
            "Ancho de la zona (r_sup - r_inf) en % del precio actual. Tope del modelo: 5 % del precio."))
        self.tabla.add_slot("body-cell-ancho_pct", _slot("props.value.toFixed(1) + '%'"))
        self.tabla.add_slot("body-cell-dist_pct", _slot("((props.value > 0 ? '+' : '') + props.value.toFixed(1) + '%')"))
        for col in ("bounce", "fill_rate", "mae"):
            self.tabla.add_slot(f"body-cell-{col}", _slot("props.value.toFixed(1) + '%'"))
        self.tabla.add_slot("body-cell-capital", _slot("'$' + props.value.toFixed(2)"))
        self.tabla.add_slot("body-cell-acciones", _slot("props.value.toFixed(4)"))
        self.tabla.add_slot("body-cell-entrada", _slot("'$' + props.value.toFixed(2)"))

    def _panel_valor_justo(self) -> None:
        ui = self.ui
        with ui.element("div").classes("aviso-box w-full").props("data-nota=valor-justo"):
            ui.label(fv.AVISO).classes("text-caption")
            ui.label(fv.AVISO_ESCENARIOS).classes("text-caption")
        self.vj_msg = ui.label("").classes("txt-muted").props("data-vj=mensaje")
        # ---- Bloque 1: resumen ejecutivo (4 tarjetas KPI)
        self.vj_kpi, self.vj_kpi_sub = {}, {}
        with ui.row().classes("w-full gap-3 items-stretch no-wrap overflow-auto").props("data-bloque=kpis"):
            for clave, titulo in (("precio", "Precio actual"), ("fair", "Valor justo (Escenario Medio)"),
                                  ("piso", "Piso fundamental (Escenario Pesimista)"), ("diag", "Diagnóstico de valoración")):
                with ui.card().classes("grow min-w-[200px] p-3 gap-1").props(f"data-kpi={clave}"):
                    ui.label(titulo).classes("text-caption txt-muted")
                    self.vj_kpi[clave] = ui.label("n/d").classes("vj-kpi-valor")
                    if clave == "diag":
                        self.vj_kpi["incert"] = ui.label("").classes("text-caption")
                    self.vj_kpi_sub[clave] = ui.label("").classes("text-caption txt-muted").style("white-space: normal")
        self.vj_kpi["fair"].tooltip("Escenario Medio: supuestos de escenario fijados de antemano (WACC con ^TNX/Blume/ERP, "
                                    "DCF de 10 años en 2 etapas). No es un precio objetivo ni una predicción.")
        self.vj_kpi["piso"].tooltip("Escenario Pesimista: estrés duro (WACC mínimo "
                                    f"{_pct(fv.WACC_PESIMISTA_MIN, 1)}, crecimiento topado a "
                                    f"{_pct(fv.TOPE_CRECIMIENTO_ETAPA1['pesimista'], 0)}, múltiplos "
                                    f"{_pct(fv.COMPRESION_MULTIPLOS['pesimista'], 0)}). No es un «peor caso realista».")
        self.vj_kpi["diag"].tooltip(f"Sobrevalorada: el precio está {_pct(UMBRAL_DIAGNOSTICO, 0)} o más por encima del valor "
                                    f"justo Medio. Infravalorada: {_pct(UMBRAL_DIAGNOSTICO, 0)} o más por debajo. Entre ambos: "
                                    "En línea. Es una etiqueta de lectura rápida, no una recomendación.")
        self.vj_comp = ui.column().classes("gap-0 w-full").props("data-vj=complementarias")
        # ---- Bloque 2: grafico historico con bandas (ancho completo)
        with ui.card().classes("w-full p-3 gap-1").props("data-bloque=historico"):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label("Valor justo histórico con bandas de ±10 / 20 / 30 % (Escenario Medio)").classes("text-subtitle2")
                self.vj_cargando = ui.row().classes("items-center gap-1 no-wrap").props("data-vj=cargando")
                with self.vj_cargando:
                    ui.spinner(size="sm")
                    ui.label("Reconstruyendo la serie histórica…").classes("text-caption txt-muted")
                self.vj_cargando.visible = False
            self.vj_hist = ui.echart(opciones_historico(None, None, "Pulsa Recalcular para analizar un ticker")) \
                .classes("w-full vj-hist").style("height: 460px").props("data-grafico=vj-historico")
            self.vj_hist_nota = ui.label("").classes("text-caption txt-aviso").props("data-vj=hist-nota")
            with ui.expansion("? Cómo interpretar esta gráfica").props("dense data-exp=vj-ayuda").classes("w-full modelo-sec"):
                for linea in texto_ayuda_historico():
                    ui.label(linea).classes("text-caption")
                self.vj_hist_notas = ui.column().classes("gap-0 w-full").props("data-vj=hist-notas")
        # ---- Bloque 3: desglose y controles (acordeones colapsados por defecto)
        with ui.row().classes("w-full gap-4 items-start no-wrap").props("data-bloque=desglose"):
            with ui.column().classes("gap-2").style("flex: 7 1 0; min-width: 0"):
                with ui.expansion("Valor por modelo: Escenario Medio vs Pesimista", icon="bar_chart") \
                        .props("dense data-exp=vj-barras").classes("w-full modelo-sec") as exp_b:
                    self.vj_barras = ui.echart(opciones_barras(None, None, None, "Sin análisis")) \
                        .classes("w-full").style("height: 420px").props("data-grafico=vj-barras")
                with ui.expansion("Explicación de los modelos y familias", icon="menu_book") \
                        .props("dense data-exp=vj-explicacion").classes("w-full modelo-sec") as exp_e:
                    self.vj_expl = ui.column().classes("gap-1 w-full").props("data-vj=explicacion")
                with ui.expansion("Supuestos y topes del cálculo", icon="tune") \
                        .props("dense data-exp=vj-supuestos").classes("w-full modelo-sec") as exp_s:
                    self.vj_supuestos = ui.column().classes("gap-0 w-full").props("data-vj=supuestos")
            with ui.column().classes("gap-2").style("flex: 5 1 0; min-width: 0"):
                with ui.expansion("Activar / Desactivar modelos", icon="checklist") \
                        .props("dense data-exp=vj-modelos").classes("w-full modelo-sec") as exp_m:
                    with ui.row().classes("gap-1"):
                        ui.button("Todos", on_click=lambda: self._vj_seleccionar(list(fv.IDS_MODELOS))) \
                            .props("dense outline size=sm data-accion=vj-todos")
                        ui.button("Solo DCF", on_click=lambda: self._vj_seleccionar(ids_por_familia("DCF"))) \
                            .props("dense outline size=sm data-accion=vj-dcf")
                        ui.button("Solo Múltiplos", on_click=lambda: self._vj_seleccionar(ids_por_familia("Múltiplos"))) \
                            .props("dense outline size=sm data-accion=vj-multiplos")
                    ui.label("Los modelos desactivados se mantienen al cambiar de ticker. «Solo DCF» y «Solo Múltiplos» "
                             "muestran visiones distintas del mismo negocio y pueden diferir mucho; «Todos» los reactiva. "
                             "Los modelos no son independientes: N modelos no equivalen a N confirmaciones.") \
                        .classes("text-caption txt-muted").props("data-vj=nota-seleccion")
                    for id_, nombre, familia in fv.MODELOS:
                        with ui.column().classes("vj-fila w-full gap-0").props(f"data-modelo={id_}"):
                            with ui.row().classes("w-full items-center no-wrap justify-between"):
                                chk = ui.checkbox(f"{nombre}", value=True,
                                                  on_change=lambda e, k=id_: self._vj_toggle_modelo(k, e.value)).props("dense")
                                px = ui.label("n/d").classes("text-caption")
                            est = ui.label("").classes("text-caption txt-muted").style("white-space: normal")
                            tope = ui.label("").classes("text-caption txt-aviso").style("white-space: normal")
                        self.vj_rows[id_] = {"chk": chk, "precio": px, "estado": est, "tope": tope}
                with ui.expansion("Parámetros de sensibilidad", icon="tune") \
                        .props("dense data-exp=vj-parametros").classes("w-full modelo-sec") as exp_p:
                    ui.label("Mueve los sliders para recalcular en vivo; no consultan SEC ni yfinance.") \
                        .classes("text-caption txt-muted")
                    self.vj_sl, self.vj_sl_lbl = {}, {}
                    for clave, titulo in (("wacc", "WACC"), ("g", "Crecimiento terminal g"), ("mos", "Margen de seguridad")):
                        lo, hi, paso = VJ_SLIDERS[clave]
                        with ui.column().classes("w-full gap-0"):
                            self.vj_sl_lbl[clave] = ui.label(titulo).classes("text-caption")
                            self.vj_sl[clave] = ui.slider(min=lo, max=hi, step=paso, value=lo,
                                                          on_change=lambda e, k=clave: self._vj_slider_cambiado(k)) \
                                .props(f"label data-slider={clave}")
                            ui.label(texto_slider(clave)).classes("text-caption txt-muted").props(f"data-ayuda-slider={clave}")
                    self.vj_avisos = ui.column().classes("gap-0 w-full").props("data-vj=avisos")
                    ui.button("Restablecer supuestos", icon="restart_alt", on_click=self._vj_restablecer) \
                        .props("dense outline size=sm data-accion=vj-reset")
        for exp in (exp_b, exp_e, exp_s, exp_m, exp_p):   # el grafico de un acordeon cerrado mide 0: se redimensiona al abrir
            exp.on_value_change(lambda _e: self._vj_resize())
        self._vj_poner_sliders(automatico=True)   # sliders = parametros reales (sin ticker no hay entradas que lo hagan)
        self._vj_actualizar()

    # ---- valor justo: estado, calculo en memoria y repintado
    def _vj_dict_actual(self) -> dict | None:
        """Entradas del ticker actual: las de CACHE o, si el LRU de la nube lo expulso, las ya cargadas en la Vista."""
        d = (CACHE.get(self.ticker or "") or {}).get("entradas_valoracion")
        if d:
            return d
        return self.vj_dict if self.ticker is not None and self.ticker == self.vj_ticker else None

    def _vj_cambio_entradas(self) -> bool:
        """True si las entradas de valoracion del ticker actual no son las ya cargadas (otro ticker o nuevo Recalcular)."""
        return self._vj_dict_actual() is not self.vj_dict or self.ticker != self.vj_ticker

    def _vj_sincronizar(self) -> None:
        """Toma de CACHE las entradas del ticker actual; si cambiaron, reinicia el WACC a su valor automatico."""
        if not self._vj_cambio_entradas():
            return
        d = self._vj_dict_actual()
        self.vj_ticker, self.vj_dict, self.vj_gen = self.ticker, d, self.vj_gen + 1
        self.vj_entradas = fv.EntradasValoracion.desde_dict(d) if d else None
        self.vj_params = fv.ParametrosValoracion(margen_seguridad=self.vj_params.margen_seguridad,
                                                 g=self.vj_params.g)
        self.vj_wacc_tocado = False
        self._vj_poner_sliders(automatico=True)

    def _vj_calcular(self) -> dict | None:
        """Escenarios Medio y Pesimista (``fv.calcular_escenarios``: ~2-13 ms, en memoria y sin red); None sin entradas.

        ``vj_agg`` = el agregado del Escenario Medio (``fair_value``, ``modelos``...): UNICA fuente de los KPI, las casillas,
        las barras y la linea de Analisis, para que nunca se desincronicen."""
        self._vj_sincronizar()
        if self.vj_entradas is None:
            self.vj_esc = self.vj_agg = None
            return None
        res = CACHE.get(self.ticker or "") or {}
        px = _num(res.get("precio_actual")) or self.vj_entradas.precio
        ids = [i for i, v in self.vj_activos.items() if v]
        self.vj_esc = fv.calcular_escenarios(self.vj_entradas, self.vj_params, ids, self.vj_params.margen_seguridad, px)
        self.vj_agg = self.vj_esc["medio"]
        return self.vj_esc

    def _vj_poner_sliders(self, automatico: bool = False) -> None:
        """Mueve los sliders a los parametros actuales (sin disparar recalculos); ``automatico`` usa el WACC automatico."""
        if not hasattr(self, "vj_sl"):
            return
        self._vj_silencio = True
        try:
            if automatico or self.vj_params.wacc is None:
                auto = fv.calcular_tasas(self.vj_entradas, self.vj_params)["wacc_auto"] if self.vj_entradas else 0.09
                self.vj_sl["wacc"].value = round(auto * 100, 1)
            else:
                self.vj_sl["wacc"].value = round(self.vj_params.wacc * 100, 1)
            self.vj_sl["g"].value = round(self.vj_params.g * 100, 1)
            self.vj_sl["mos"].value = round(self.vj_params.margen_seguridad * 100)
            self._vj_limitar_g()
        finally:
            self._vj_silencio = False
        self._vj_etiquetas_sliders()

    def _vj_limitar_g(self) -> None:
        """El slider de g no pasa de (WACC - DIF_MIN_WACC_G): por encima los DCF/EPV/DDM dejan de aplicar."""
        w = float(self.vj_sl["wacc"].value)
        tope = round(min(VJ_SLIDERS["g"][1], max(VJ_SLIDERS["g"][0], w - fv.DIF_MIN_WACC_G * 100)), 1)
        self.vj_sl["g"].props(f"max={tope}")
        if float(self.vj_sl["g"].value) > tope:
            silencio, self._vj_silencio = self._vj_silencio, True
            try:
                self.vj_sl["g"].value = tope
            finally:
                self._vj_silencio = silencio

    def _vj_etiquetas_sliders(self) -> None:
        w, g, m = (self.vj_sl[k].value for k in ("wacc", "g", "mos"))
        self.vj_sl_lbl["wacc"].text = f"WACC (Escenario Medio): {_es(w, 1)} %" + ("" if self.vj_wacc_tocado else " (automático)")
        self.vj_sl_lbl["g"].text = f"Crecimiento terminal g: {_es(g, 1)} %"
        self.vj_sl_lbl["mos"].text = f"Margen de seguridad: {_es(m, 0)} %"

    def _vj_leer_sliders(self) -> None:
        self._vj_limitar_g()
        w, g, m = (float(self.vj_sl[k].value) for k in ("wacc", "g", "mos"))
        self.vj_params = self.vj_params.con(wacc=w / 100.0 if self.vj_wacc_tocado else None, g=g / 100.0,
                                            margen_seguridad=m / 100.0)
        self._vj_etiquetas_sliders()

    def _vj_slider_cambiado(self, clave: str) -> None:
        """Slider movido por el usuario: actualiza parametros y etiquetas ya; el recalculo espera ``VJ_DEBOUNCE_S``."""
        if self._vj_silencio:  # el cambio lo hizo el codigo (reset / cambio de ticker), no el usuario
            return
        if clave == "wacc":
            self.vj_wacc_tocado = True
        self._vj_leer_sliders()
        self._vj_seq += 1
        seq = self._vj_seq
        try:
            bucle = asyncio.get_running_loop()
        except RuntimeError:  # sin bucle de eventos (llamada sincrona, p. ej. tests): aplicar de inmediato
            self._vj_aplicar()
            return
        bucle.call_later(VJ_DEBOUNCE_S, self._vj_fin_debounce, seq)

    def _vj_fin_debounce(self, seq: int) -> None:
        if seq == self._vj_seq:  # solo el ultimo cambio de la rafaga recalcula
            self._vj_aplicar()

    def _vj_aplicar(self) -> None:
        """Recalcula escenarios y repinta la pestana (KPIs, barras, serie) y la linea de Analisis, todo coherente."""
        self._vj_actualizar()
        self._vj_refrescar_linea()
        liberar_memoria()

    def _vj_refrescar_linea(self) -> None:
        """Redibuja la linea del Escenario Medio en Analisis solo si el interruptor esta encendido."""
        if self.sw_vj.value:
            self.refrescar(solo_figura=True)

    def _vj_restablecer(self) -> None:
        self.vj_params = fv.ParametrosValoracion()
        self.vj_wacc_tocado = False
        self._vj_poner_sliders(automatico=True)
        self._vj_aplicar()

    def _vj_toggle_modelo(self, id_: str, valor: bool) -> None:
        if self._vj_silencio:
            return
        self.vj_activos[id_] = bool(valor)
        self._vj_aplicar()

    def _vj_seleccionar(self, ids: list[str]) -> None:
        """Botones Todos / Solo DCF / Solo Multiplos: deja activos exactamente ``ids``."""
        self.vj_activos = {i: i in ids for i in fv.IDS_MODELOS}
        self._vj_aplicar()

    def _toggle_vj(self, valor: bool) -> None:
        self.refrescar(solo_figura=True)

    def _vj_para_grafico(self) -> dict | None:
        """Escenarios para la linea del grafico de Analisis (``valor_justo_medio`` toma SOLO el Medio); None si el
        interruptor esta apagado o no hay entradas."""
        if not getattr(self, "sw_vj", None) or not self.sw_vj.value:
            return None
        return self._vj_calcular()

    # ---- redimensionado de los graficos ECharts
    def _vj_js_resize(self) -> None:
        self.n_resize_vj += 1
        for ch in (getattr(self, "vj_hist", None), getattr(self, "vj_barras", None)):
            if ch is None:
                continue
            try:
                ch.run_chart_method("resize")
            except Exception:  # noqa: BLE001  (sin cliente conectado, p. ej. pruebas unitarias)
                pass

    def _vj_resize(self, demora: float = 0.45) -> None:
        """Un grafico dentro de una pestana o acordeon oculto mide 0 px: se le pide ``resize()`` al mostrarlo (ya, y de nuevo
        al terminar la animacion). El redimensionado de la ventana lo atiende el ResizeObserver del propio componente."""
        self._vj_js_resize()
        try:
            self.ui.timer(demora, self._vj_js_resize, once=True)
        except Exception:  # noqa: BLE001  (sin bucle de eventos / contexto)
            pass

    def _tab_cambiada(self, valor: Any) -> None:
        if valor == "valor_justo":
            self._vj_resize()

    @staticmethod
    def _echart_set(chart: Any, opciones: dict) -> None:
        """Sustituye las opciones y las envia (``setOption`` fusiona: el grafico no se recrea)."""
        chart.options.clear()
        chart.options.update(opciones)
        chart.update()

    # ---- serie historica (fuera del hilo de eventos, cacheada por huella)
    def _vj_anios(self) -> int:
        return self.anios_velas if self.anios_velas else fv.anios_para_modo(fv.modo_por_defecto())

    def _vj_huella(self) -> tuple:
        p = self.vj_params
        return (self.ticker, self.vj_gen, p.wacc, p.g, p.rf, p.erp, p.pe_objetivo, p.ev_ebitda_objetivo, p.ev_revenue_objetivo,
                p.pb_objetivo, p.p_fcf_objetivo, tuple(i for i, v in self.vj_activos.items() if v), self._vj_anios())

    def _vj_cargando(self, activo: bool) -> None:
        self.vj_cargando.visible = activo

    def _vj_pedir_serie(self) -> None:
        """Pide la serie historica del Valor Justo Medio: cache por huella (ticker + parametros + modelos + anios) o calculo
        en un hilo (``asyncio.to_thread``) con indicador de carga; sin bucle de eventos (pruebas) se calcula en linea."""
        res = CACHE.get(self.ticker or "") or {}
        df = res.get("df")
        if self.vj_entradas is None or not df_valido(df):
            self._vj_seq_serie += 1
            self._vj_cargando(False)
            self._vj_pintar_serie(None, "Sin precios en memoria para la serie histórica: pulsa Recalcular")
            return
        h = self._vj_huella()
        CACHE_SERIES.maximo = MAX_SERIES_NUBE if userdata.en_nube() else MAX_SERIES_LOCAL
        previa = CACHE_SERIES.get(h)
        self._vj_seq_serie += 1
        seq = self._vj_seq_serie
        if previa is not None:
            CACHE_SERIES.move_to_end(h)
            self._vj_cargando(False)
            self._vj_pintar_serie(previa)
            return
        args = (self.vj_entradas, df, self.vj_params, [i for i, v in self.vj_activos.items() if v], self._vj_anios())
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._vj_guardar_serie(h, calcular_serie(*args), seq)
            return
        self._vj_cargando(True)
        from nicegui import background_tasks
        background_tasks.create(self._vj_serie_async(h, args, seq), name="vj-serie")

    async def _vj_serie_async(self, h: tuple, args: tuple, seq: int) -> None:
        serie = await asyncio.to_thread(calcular_serie, *args)
        del args
        self._vj_guardar_serie(h, serie, seq)

    def _vj_guardar_serie(self, h: tuple, serie: dict, seq: int) -> None:
        if not serie.get("error"):
            CACHE_SERIES[h] = serie
        if seq != self._vj_seq_serie:     # llego tarde: ya se pidio otra (otros parametros/ticker)
            return
        self._vj_cargando(False)
        try:
            self._vj_pintar_serie(serie)
        except (RuntimeError, KeyError):  # cliente ya cerrado
            pass
        liberar_memoria()

    def _vj_pintar_serie(self, serie: dict | None, mensaje: str | None = None) -> None:
        serie, _omitidos = sanear_serie(serie)
        self.vj_serie = serie
        ok = bool(serie and serie.get("fechas"))
        msg = mensaje or (None if ok else "Sin datos suficientes para reconstruir la serie histórica")
        self._echart_set(self.vj_hist, opciones_historico(serie if ok else None, self.vj_esc, msg))
        piso = _num(((self.vj_esc or {}).get("pesimista") or {}).get("fair_value"))
        bajo = [v for v in ((serie or {}).get("bandas") or {}).get("-30%", []) if v is not None]
        self.vj_hist_nota.text = (f"El Piso fundamental (${piso:,.2f}) queda por debajo del rango visible del gráfico."
                                  if ok and piso is not None and bajo and piso < min(bajo) else "")
        self.vj_hist_notas.clear()
        with self.vj_hist_notas:
            for n in (serie or {}).get("notas") or []:
                self.ui.label(n).classes("text-caption txt-muted")

    # ---- repintado de la pestana
    def _vj_actualizar(self) -> None:
        """Repinta toda la pestana Valor Justo (KPIs, casillas, barras, supuestos) desde la memoria y pide la serie."""
        if not hasattr(self, "vj_kpi"):
            return
        esc = self._vj_calcular()
        res = CACHE.get(self.ticker or "") or {}
        px = _num(res.get("precio_actual"))
        self._vj_silencio = True
        try:
            if esc is None:
                self.vj_msg.text = ("Pulsa Recalcular para analizar el ticker y obtener las entradas de valoración."
                                    if not res else
                                    "Este análisis no trae entradas de valoración (ver Avisos en Análisis); pulsa "
                                    "Recalcular.")
                for k in self.vj_kpi:
                    self.vj_kpi[k].text = "n/d" if k != "incert" else ""
                for k in self.vj_kpi_sub:
                    self.vj_kpi_sub[k].text = ""
                self.vj_kpi["diag"].classes(remove="vj-badge vj-dx-infra vj-dx-linea vj-dx-sobre vj-dx-debil")
                self.vj_kpi["incert"].classes(remove="vj-badge vj-baja vj-media vj-alta")
                for fila in self.vj_rows.values():
                    fila["chk"].set_enabled(False)
                    fila["precio"].text = "n/d"
                    fila["estado"].text = ""
                    fila["tope"].text = ""
                self._echart_set(self.vj_barras, opciones_barras(None, None, None, "Sin análisis"))
                self.vj_supuestos.clear()
                self.vj_comp.clear()
                self.vj_avisos.clear()
                self.vj_expl.clear()
                self._vj_expl_firma = None
                self._vj_seq_serie += 1
                self._vj_cargando(False)
                self._vj_pintar_serie(None, "Pulsa Recalcular para analizar un ticker")
                return
            self.vj_msg.text = ""
            med, pes = esc["medio"], esc["pesimista"]
            fair, up = med.get("fair_value"), med.get("upside_pct")
            self.vj_kpi["precio"].text = _fmt(px, "${:,.2f}")
            self.vj_kpi_sub["precio"].text = "Último cierre"
            self.vj_kpi["fair"].text = _fmt(fair, "${:,.2f}")
            self.vj_kpi_sub["fair"].text = ("" if fair is None else
                                            f"Con margen de seguridad {_es(self.vj_params.margen_seguridad * 100, 0)} %: "
                                            f"{_fmt(med.get('margin_of_safety_price'), '${:,.2f}')}")
            self.vj_kpi["piso"].text = _fmt(pes.get("fair_value"), "${:,.2f}")
            up_p = pes.get("upside_pct")
            self.vj_kpi_sub["piso"].text = ("" if pes.get("fair_value") is None else
                                            (f"{up_p:+.1f} % vs precio · " if up_p is not None else "")
                                            + "estrés duro, no un peor caso realista")
            dx = diagnostico_valoracion(px, fair, med.get("incertidumbre") if fair is not None else None, med.get("n_usados"))
            d = self.vj_kpi["diag"]
            d.classes(remove="vj-badge vj-dx-infra vj-dx-linea vj-dx-sobre vj-dx-debil")
            d.text = dx["texto"] + (f" · {dx['debil']}" if dx["debil"] else "")
            if dx["clave"]:   # soporte debil: badge atenuado (borde punteado), el diagnostico se lee igual en el texto
                d.classes("vj-badge " + ("vj-dx-debil" if dx["debil"] else f"vj-dx-{dx['clave']}"))
            d.props(f"data-diag={dx['clave'] or 'nd'}")
            niv = med.get("incertidumbre") if fair is not None else None
            self.vj_kpi["incert"].classes(remove="vj-badge vj-baja vj-media vj-alta")
            self.vj_kpi["incert"].text = f"Incertidumbre {niv}" if niv else ""
            if niv:
                self.vj_kpi["incert"].classes("vj-badge " + {"Baja": "vj-baja", "Media": "vj-media", "Alta": "vj-alta"}[niv])
            cv = med.get("cv")
            self.vj_kpi_sub["diag"].text = (med.get("motivo") or (
                f"Upside {_fmt(up, '{:+.1f}')} % · {med['n_usados']} modelos usados de {med['n_validos']} válidos"
                + ("" if cv is None else f" · CV {_es(cv, 2)}")
                + (f" · outliers: {len(med['modelos_outlier'])}" if med["modelos_outlier"] else "")))
            for m in esc["modelos"]:
                fila = self.vj_rows[m["id"]]
                activo = bool(self.vj_activos.get(m["id"], True))
                clave, texto = estado_modelo(m, med, activo)
                fila["chk"].set_enabled(bool(m["aplicable"]))
                fila["chk"].value = activo and bool(m["aplicable"])
                fila["precio"].text = ("n/d" if not m["aplicable"] else
                                       f"Medio {_fmt(m.get('precio_medio'), '${:,.2f}')} · Pesimista {_fmt(m.get('precio_pesimista'), '${:,.2f}')}")
                fila["estado"].text = texto
                fila["estado"].classes(remove="txt-muted txt-aviso txt-ok txt-error")
                fila["estado"].classes({"activo": "txt-ok", "outlier": "txt-aviso"}.get(clave, "txt-muted"))
                fila["estado"].props(f"data-estado={clave}")
                topes = m.get("topes_medio") or m.get("topes_pesimista") or []
                fila["tope"].text = ("tope activo: " + "; ".join(topes)) if topes and clave in ("activo", "outlier") else ""
            self._echart_set(self.vj_barras, opciones_barras(esc, px, self.vj_activos))
            self.vj_barras.style(f"height: {alto_barras(len(esc['modelos']))}px")
            self._vj_pintar_complementarias(med, px)
            self._vj_pintar_supuestos(esc)
            self._vj_pintar_explicacion(esc)
        finally:
            self._vj_silencio = False
        self._vj_pedir_serie()

    def _vj_pintar_explicacion(self, esc: dict) -> None:
        """Acordeon «Explicación de los modelos y familias»: se redibuja solo si cambia el conjunto de modelos que no aplican."""
        ex = explicacion_modelos(esc["modelos"])
        firma = tuple((n["id"], n["motivo"]) for n in ex["no_aplican"])
        if firma == self._vj_expl_firma:
            return
        self._vj_expl_firma = firma
        self.vj_expl.clear()
        with self.vj_expl:
            for titulo, texto in ex["familias"]:
                self.ui.label(f"{titulo}: {texto}").classes("text-caption")
            self.ui.label("Un modelo por línea:").classes("text-caption txt-muted")
            for m in ex["modelos"]:
                self.ui.label(f"• {m['nombre']}: {m['linea']}").classes("text-caption txt-muted")
            if ex["no_aplican"]:
                self.ui.label("Por qué algunos aparecen como «No aplica» en este ticker (motivo del motor):") \
                    .classes("text-caption").props("data-vj=no-aplica")
                for n in ex["no_aplican"]:
                    self.ui.label(f"• {n['nombre']}: {n['motivo']}").classes("text-caption txt-aviso")

    def _vj_pintar_complementarias(self, agg: dict, px: float | None) -> None:
        """Una linea compacta bajo las tarjetas: mediana, submedias por familia y promedio simple del Escenario Medio."""
        self.vj_comp.clear()
        c = agg.get("complementarias")
        if c is None:
            return
        pct = lambda v: "" if v is None else f" ({v:+.0f} %)"  # noqa: E731
        partes = [f"Mediana ${c['mediana']:,.2f}{pct(c['upside_mediana'])}"]
        partes += [f"{g} ${v:,.2f} (n={c['n_por_grupo'][g]})" for g, v in c["submedias"].items()]
        partes.append(f"Promedio simple ${c['promedio_simple']:,.2f}{pct(c['upside_promedio_simple'])}")
        fam = ", ".join(agg.get("familias_usadas") or []) or "n/d"
        with self.vj_comp:
            self.ui.label("Complementarias del Escenario Medio (el consolidado iguala el peso de las familias, 1/3 cada una, "
                          "tras filtrar outliers con IQR): " + " | ".join(partes)).classes("text-caption") \
                .props("data-vj=comp-cifras")
            self.ui.label(f"Los modelos no son independientes: los 6 DCF comparten WACC, g y crecimiento, y los de Valor "
                          f"comparten EPS y valor contable. Familias usadas: {fam}.").classes("text-caption txt-muted")

    def _vj_lineas_dcf_inverso(self) -> list[str]:
        """DCF inverso: WACC y crecimiento que iguala el DCF a 10 anios (FCF) con el precio."""
        res = CACHE.get(self.ticker or "") or {}
        di = fv.dcf_inverso(self.vj_entradas, self.vj_params, _num(res.get("precio_actual")) or self.vj_entradas.precio)
        if di["wacc_implicito"] is None and di["g0_implicito"] is None:
            return ["DCF inverso (DCF 10 años, FCF): sin solución en el rango razonable o el modelo no aplica."]
        partes = []
        if di["wacc_implicito"] is not None:
            partes.append(f"WACC implícito {_es(di['wacc_implicito'] * 100, 2)} % (usado {_es(di['wacc_usado'] * 100, 2)} %)")
        if di["g0_implicito"] is not None:
            partes.append(f"crecimiento inicial implícito {_es(di['g0_implicito'] * 100, 1)} % (usado "
                          f"{_es(di['g0_usado'] * 100, 1)} %)")
        menos_riesgo = (di["wacc_implicito"] is not None and di["wacc_implicito"] < di["wacc_usado"]) or (
            di["g0_implicito"] is not None and di["g0_implicito"] > di["g0_usado"])
        cierre = ("el mercado descuenta menos riesgo o más crecimiento que estos supuestos" if menos_riesgo else
                  "el mercado descuenta más riesgo o menos crecimiento que estos supuestos")
        return ["DCF inverso (el que iguala el DCF a 10 años de FCF con el precio actual): " + "; ".join(partes)
                + f". Lectura: {cierre}."]

    def _vj_pintar_supuestos(self, esc: dict) -> None:
        sm, sp = esc["supuestos"]["medio"], esc["supuestos"]["pesimista"]
        self.vj_supuestos.clear()
        self.vj_avisos.clear()
        mult = sm["multiplos"]
        pct1 = lambda v: "n/d" if v is None else _es(v * 100, 1) + " %"  # noqa: E731
        lineas = [
            f"WACC Medio {_es(sm['wacc'] * 100, 2)} % ({sm['wacc_origen']}; ke {_es(sm['ke'] * 100, 2)} %, kd "
            f"{_es(sm['kd'] * 100, 2)} %, beta {_es(sm['beta'], 2)} {sm['beta_origen']}, tasa libre de riesgo "
            f"{_es(sm['rf'] * 100, 2)} % [{sm['rf_origen']}], prima de riesgo {_es(sm['erp'] * 100, 1)} %)",
            f"WACC Pesimista {_es(sp['wacc'] * 100, 2)} % ({sp['wacc_origen']})",
            f"g∞ Medio {_es(sm['g'] * 100, 1)} % | g∞ Pesimista {_es(sp['g'] * 100, 1)} % | margen de seguridad "
            f"{_es(self.vj_params.margen_seguridad * 100, 0)} % | tasa impositiva {_es(sm['tasa_impositiva'] * 100, 1)} % "
            f"({sm['tasa_origen']})",
            f"Crecimiento de la etapa 1 (años 1-{fv.ANIOS_ETAPA1}, Medio / Pesimista): FCF {pct1(sm['crecimiento_inicial_fcf'])} / "
            f"{pct1(sp['crecimiento_inicial_fcf'])}, EBITDA {pct1(sm['crecimiento_inicial_ebitda'])} / "
            f"{pct1(sp['crecimiento_inicial_ebitda'])}, ingresos {pct1(sm['crecimiento_inicial_revenue'])} / "
            f"{pct1(sp['crecimiento_inicial_revenue'])} (topes {_pct(sm['tope_crecimiento'], 0)} / {_pct(sp['tope_crecimiento'], 0)}; "
            "luego decae hasta g∞)",
            "Múltiplos objetivo (Medio): " + ", ".join(
                f"{nombre} {'n/d' if mult[k]['valor'] is None else _es(mult[k]['valor'], 1) + 'x'}"
                for k, nombre in (("pe", "P/E"), ("ev_ebitda", "EV/EBITDA"), ("ev_revenue", "EV/Ingresos"),
                                  ("pb", "P/B"), ("p_fcf", "P/FCF")))
            + f"; el Pesimista los castiga {_pct(-fv.COMPRESION_MULTIPLOS['pesimista'], 0)}",
            f"Margen FCF objetivo (DCF por ingresos): Medio {pct1(sm['margen_fcf_objetivo'])} (tope {_pct(sm['tope_margen'], 0)}) "
            f"/ Pesimista {pct1(sp['margen_fcf_objetivo'])} (tope {_pct(sp['tope_margen'], 0)})"]
        if sm["topes_activos"]:
            lineas.append("Topes activos (el resultado depende de ellos por construcción): " + "; ".join(sm["topes_activos"]))
        if sm["flujos_ciclicos_normalizados"]:
            lineas.append("Flujo cíclico normalizado (base = media de los últimos "
                          f"{fv.N_NORMALIZACION} años, crecimiento inicial = g): " + ", ".join(sm["flujos_ciclicos_normalizados"]))
        if sm["ajuste_fcff"]:
            lineas.append(f"DCF de FCF: se suman los intereses después de impuestos ({_es(sm['ajuste_fcff'] / 1e6, 0)} M USD) "
                          "para pasar de FCF apalancado a FCFF (la deuda se resta aparte).")
        lineas.extend(self._vj_lineas_dcf_inverso())
        avisos = list(esc["supuestos"].get("avisos") or [])
        if not sm["g_valido"]:
            avisos.append(f"La diferencia WACC - g es menor que {_es(fv.DIF_MIN_WACC_G * 100, 1)} pp (g "
                          f"{_es(sm['g'] * 100, 1)} %, WACC {_es(sm['wacc'] * 100, 2)} %): los DCF, el EPV y el DDM no "
                          f"aplican con estos supuestos.")
        with self.vj_supuestos:
            for linea in lineas:
                self.ui.label(linea).classes("txt-muted text-caption")
            for a in avisos:
                self.ui.label("Aviso: " + a).classes("txt-aviso text-caption")
            self.ui.label("Los parámetros de mercado (Rf, ERP, beta, sector) y los sliders son los vigentes, no point-in-time. "
                          "Valor estimado por modelos con supuestos; no es un precio objetivo ni asesoría.") \
                .classes("text-caption")
        with self.vj_avisos:
            for a in avisos:
                self.ui.label("Aviso: " + a).classes("txt-aviso text-caption").props("data-vj=aviso-supuesto")

    def _panel_watchlist(self) -> None:
        ui = self.ui
        with ui.element("div").classes("aviso-box w-full").props("data-nota=watchlist-intro"):
            for linea in TEXTO_WATCHLIST_INTRO:
                ui.label(linea).classes("text-caption")
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            self.in_wl_ticker = ui.input("Añadir ticker", placeholder="NVDA").props("dense outlined dark") \
                .classes("w-40").on("keydown.enter", lambda _e: self.anadir_manual())
            self.in_wl_nota = ui.input("Nota (opcional)").props("dense outlined dark").classes("grow") \
                .on("keydown.enter", lambda _e: self.anadir_manual())
            self.btn_wl_add = ui.button("Añadir", icon="add", on_click=self.anadir_manual)
        with ui.row().classes("w-full items-center gap-2"):
            self.btn_wl_ejemplo = ui.button("Cargar lista de ejemplo", icon="playlist_add",
                                            on_click=self.cargar_ejemplo).props("flat")
            self.btn_wl_vaciar = ui.button("Vaciar", icon="delete_sweep", on_click=self.vaciar_lista) \
                .props("flat color=negative")
        self.cont_wl = ui.column().classes("w-full gap-0").props("data-lista=watchlist")
        self._pintar_lista()
        with ui.row().classes("items-center gap-4"):
            self.btn_scan = ui.button("Escanear mi watchlist", icon="radar", on_click=self.escanear)
            self.btn_stop = ui.button("Detener", icon="stop", on_click=self._detener_scan).props("flat")
            self.btn_stop.visible = False
            self.sw_bt_wl = ui.switch("Incluir backtest (lento)", value=not userdata.en_nube())
            self.sw_vj_wl = ui.switch("Incluir valor justo (Upside)", value=not userdata.en_nube())
        self._actualizar_btn_scan()
        # Notas visibles (no solo tooltips): el capital NO se reparte entre tickers y el score es solo orden.
        with ui.element("div").classes("aviso-box w-full").props("data-nota=watchlist"):
            self.nota_capital = ui.label(nucleo.NOTA_CAPITAL_BATCH).classes("text-caption")
            self.nota_ranking = ui.label(nota_ranking()).classes("text-caption")
        self.prog = ui.linear_progress(value=0, show_value=False).props("color=amber").classes("w-full")
        self.prog.visible = False
        self.lbl_scan = ui.label("").classes("text-caption txt-muted")
        self.tabla_wl = ui.table(columns=[
            dict(name="pos", label="#", field="pos", sortable=True, align="left"),
            dict(name="ticker", label="Ticker", field="ticker", sortable=True, align="left"),
            dict(name="precio", label="Precio", field="precio", sortable=True),
            dict(name="score", label="Score ranking", field="score", sortable=True),
            dict(name="confluencia", label="Confluencia", field="confluencia", sortable=True),
            dict(name="confiabilidad", label="Confiabilidad", field="confiabilidad", sortable=True),
            dict(name="cobertura", label="Cobertura", field="cobertura", sortable=True),
            dict(name="mejor_zona", label="Mejor zona", field="mejor_zona", align="left"),
            dict(name="dist_pct", label=f"{ETIQUETA_DIST_RSUP} %".capitalize(), field="dist_pct", sortable=True),
            dict(name="upside", label="Upside %", field="upside", sortable=True)],
            rows=[], row_key="id", pagination=0).props("dense flat bordered dark no-data-label=\"Pulsa Escanear "
                                                       "tu watchlist\"").classes("w-full cursor-pointer")
        self.tabla_wl.add_slot("body-cell-precio", _slot("'$' + props.value.toFixed(2)"))
        for col in ("score", "confluencia", "confiabilidad"):
            self.tabla_wl.add_slot(f"body-cell-{col}", _slot("props.value.toFixed(1)"))
        self.tabla_wl.add_slot("body-cell-dist_pct", _slot("((props.value > 0 ? '+' : '') + props.value.toFixed(1) + '%')"))
        self.tabla_wl.add_slot("body-cell-upside", (
            "<q-td :props=\"props\" :class=\"props.row.upside_nota ? 'txt-muted' : ''\">"
            "{{ props.value == null ? 'n/d' : ((props.value > 0 ? '+' : '') + props.value.toFixed(1) + '%') }}"
            "<span v-if=\"props.row.upside_nota\"> · {{ props.row.upside_nota }}</span></q-td>"))
        self.tabla_wl.add_slot("header-cell-upside", _cabecera_con_ayuda(
            "Potencial del valor justo consolidado (todos los modelos, supuestos por defecto) frente al precio. Es una "
            "estimación, no un precio objetivo; n/d si no se calculó (opcional, apagado en la nube)."))
        self.tabla_wl.add_slot("body-cell-cobertura", _PLANTILLA_COBERTURA)
        self.tabla_wl.add_slot("header-cell-cobertura", _cabecera_con_ayuda(
            "Parte del peso de la Confiabilidad que tenía dato (pasa el ratón por una celda para ver qué quedó sin "
            "dato). Scores con cobertura distinta no son comparables."))
        self.tabla_wl.add_slot("header-cell-dist_pct", _cabecera_con_ayuda(
            "Distancia % de r_sup (precio del límite GTC) de la mejor zona al precio actual."))
        self.tabla_wl.on("rowClick", self._click_ranking)
        self.ui.label("Click en una fila para cargar el ticker en la vista Analisis.").classes("text-caption txt-muted")

    # ---- watchlist personalizada
    def _guardar_lista(self, items: list[dict]) -> bool:
        """Persiste (atomico) y repinta todo lo que depende de la lista. False si no se pudo escribir."""
        items = userdata.normalizar(items)
        try:
            userdata.guardar(items)
        except OSError as e:
            self.ui.notify(f"No se pudo guardar tu watchlist: {e}", type="negative", multi_line=True)
            return False
        self.items = items
        self.watchlist = userdata.tickers(items)
        self.sel_wl = {t: self.sel_wl.get(t, True) for t in self.watchlist}
        if hasattr(self, "sel_ticker"):
            opciones = self.watchlist + ([self.ticker] if self.ticker and self.ticker not in self.watchlist else [])
            self.sel_ticker.set_options(opciones)
        if hasattr(self, "cont_wl"):
            self._pintar_lista()
            self._actualizar_btn_scan()
        if hasattr(self, "btn_wl"):
            self._actualizar_btn_wl()
        return True

    def _pintar_lista(self) -> None:
        ui = self.ui
        self.sel_wl = {t: self.sel_wl.get(t, True) for t in self.watchlist}
        self.cont_wl.clear()
        with self.cont_wl:
            if not self.items:
                ui.label("Tu watchlist está vacía. Añade un ticker, usa el botón de estrella en Análisis o carga la "
                         "lista de ejemplo.").classes("txt-muted text-caption")
            for it in self.items:
                t = it["ticker"]
                with ui.row().classes("w-full items-center no-wrap gap-1").props(f"data-ticker={t}"):
                    ui.checkbox(t, value=self.sel_wl.get(t, True),
                                on_change=lambda e, k=t: self._marcar(k, e.value)).props("dense")
                    ui.label(it["nota"]).classes("grow txt-muted text-caption")
                    ui.button(icon="close", on_click=lambda _e, k=t: self.quitar_ticker(k)).props(
                        f"flat dense round size=sm aria-label=Quitar-{t}")

    def _marcar(self, t: str, valor: bool) -> None:
        self.sel_wl[t] = bool(valor)
        self._actualizar_btn_scan()

    def _seleccionados(self) -> list[str]:
        return [t for t in self.watchlist if self.sel_wl.get(t, True)]

    def _actualizar_btn_scan(self) -> None:
        if not hasattr(self, "btn_scan"):
            return
        n, total = len(self._seleccionados()), len(self.watchlist)
        self.btn_scan.text = ("Escanear mi watchlist" if n == total else "Escanear seleccionados") + f" ({n})"

    def _actualizar_btn_wl(self) -> None:
        t = self.ticker
        esta = bool(t) and t in self.watchlist
        self.btn_wl.text = ("Agregar a mi watchlist" if not t else
                            f"Quitar {t} de mi watchlist" if esta else f"Agregar {t} a mi watchlist")
        self.btn_wl.props(f"icon={'star' if esta else 'star_border'}")
        self.btn_wl.set_enabled(bool(t))

    def alternar_watchlist(self) -> None:
        t = validar_ticker(self.ticker)
        if t is None:
            self.ui.notify("Elige un ticker válido primero (p. ej. NVDA o BRK-B)", type="warning")
            return
        if t in self.watchlist:
            nueva, _ = userdata.quitar(self.items, t)
            if self._guardar_lista(nueva):
                self.ui.notify(f"{t} quitado de tu watchlist", type="info")
        else:
            nueva, ok = userdata.anadir(self.items, t)
            if ok and self._guardar_lista(nueva):
                self.ui.notify(f"{t} agregado a tu watchlist", type="positive")

    def anadir_manual(self) -> None:
        t = validar_ticker(self.in_wl_ticker.value)
        if t is None:
            self.ui.notify("Ticker inválido (letras/números, p. ej. NVDA o BRK-B)", type="warning")
            return
        nueva, ok = userdata.anadir(self.items, t, self.in_wl_nota.value)
        if not ok:
            self.ui.notify(f"{t} ya está en tu watchlist", type="info")
        elif self._guardar_lista(nueva):
            self.in_wl_ticker.value = ""
            self.in_wl_nota.value = ""
            self.ui.notify(f"{t} agregado a tu watchlist", type="positive")

    def quitar_ticker(self, t: str) -> None:
        nueva, ok = userdata.quitar(self.items, t)
        if ok and self._guardar_lista(nueva):
            self.ui.notify(f"{t} quitado de tu watchlist", type="info")

    def cargar_ejemplo(self) -> None:
        ejemplo = userdata.leer_csv(RUTA_WATCHLIST)
        if not ejemplo:
            self.ui.notify("No se encontró la lista de ejemplo (watchlist.csv)", type="warning")
            return
        nueva, n = userdata.fusionar(self.items, ejemplo)
        if self._guardar_lista(nueva):
            self.ui.notify(f"Lista de ejemplo cargada: {n} ticker(s) nuevo(s)", type="positive" if n else "info")

    def vaciar_lista(self) -> None:
        if self._guardar_lista([]):
            self.ui.notify("Watchlist vaciada", type="info")

    # ---- eventos
    def _ticker_cambiado(self, valor: str | None) -> None:
        self.ticker = validar_ticker(valor)
        self.nivel_sel = None
        self.refrescar()

    def _toggle(self, clave: str, valor: bool) -> None:
        self.activos[clave] = bool(valor)
        self.refrescar()

    def _fila_seleccionada(self, e) -> None:
        self.nivel_sel = int(e.selection[0]["id"]) if e.selection else None
        self.refrescar(solo_figura=True)

    def _click_ranking(self, e) -> None:
        fila = e.args[1] if isinstance(e.args, (list, tuple)) and len(e.args) > 1 else None
        t = (fila or {}).get("ticker")
        if not t:
            return
        if t not in self.sel_ticker.options:
            self.sel_ticker.set_options([*self.sel_ticker.options, t])
        self.sel_ticker.value = t  # dispara _ticker_cambiado y muestra el resultado cacheado
        self.tabs.set_value("analisis")

    def _detener_scan(self) -> None:
        self.detener = True

    def _capital(self) -> float | None:
        v = _num(self.capital.value)
        return v if validar_capital(v) else None

    # ---- acciones pesadas
    def _ocupar(self, activo: bool) -> None:
        self.ocupado = activo
        self.barra.visible = activo
        if activo:
            self.btn.props(add="loading")
        else:
            self.btn.props(remove="loading")
        self.btn.set_enabled(not activo)
        self.btn_scan.set_enabled(not activo)

    async def recalcular(self) -> None:
        ui = self.ui
        if self.ocupado:
            return
        t, cap = validar_ticker(self.sel_ticker.value), self._capital()
        if t is None:
            ui.notify("Ticker invalido (use letras/numeros, p. ej. NVDA o BRK-B)", type="warning")
            return
        if cap is None:
            ui.notify("El capital debe ser un numero mayor que 0", type="warning")
            return
        self.ticker, self.nivel_sel = t, None
        self._ocupar(True)
        try:
            res = await calcular_ticker(t, cap, self.sw_bt.value, self.sw_sec.value)
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Fallo inesperado al calcular {t}: {type(e).__name__}: {e}", type="negative", multi_line=True)
            return
        finally:
            self._ocupar(False)
        if res.get("precio_actual") is None:
            msg = "; ".join(res.get("errores") or ["sin datos"])
            ui.notify(f"{t}: sin datos de mercado (ticker invalido o sin red). {msg}", type="negative",
                      multi_line=True, timeout=8000)
            return
        CACHE[t] = res
        n_err, n_av = len(res.get("errores") or []), len(res.get("avisos") or [])
        del res  # la cache ya guarda el resultado; no retener una referencia mas (memoria en la nube)
        if n_err:
            ui.notify(f"{t}: {n_err} etapa(s) con fallo (ver Avisos).", type="warning")
        elif n_av:
            ui.notify(f"{t}: analisis completo con {n_av} aviso(s) de datos (ver Avisos).", type="warning")
        else:
            ui.notify(f"{t}: analisis completo.", type="positive")
        self.refrescar()
        liberar_memoria()  # nube (512 MB): liberar los DataFrames temporales del calculo

    async def escanear(self) -> None:
        ui = self.ui
        if self.ocupado:
            return
        tickers = self._seleccionados()
        if not tickers:
            ui.notify("Tu watchlist está vacía o sin tickers marcados: añade alguno o carga la lista de ejemplo.",
                      type="warning")
            return
        cap = self._capital() or 1000.0
        nube = userdata.en_nube()
        if nube and ESCANEO_NUBE["activo"]:   # visor ligero: un solo escaneo a la vez (memoria de 512 MB compartida)
            ui.notify("Ya hay un escaneo en curso en este servidor: espera a que termine.", type="warning")
            return
        tickers, omitidos = limitar_tickers_escaneo(tickers, nube)
        if omitidos:
            ui.notify(f"En la nube solo se escanean {MAX_TICKERS_ESCANEO_NUBE} tickers por vez: se omiten {omitidos}. "
                      "Para listas grandes usa la versión local.", type="warning", multi_line=True)
        if nube:
            ESCANEO_NUBE["activo"] = True
        self.detener = False
        self._ocupar(True)
        self.btn_stop.visible = True
        self.prog.visible = True
        self.prog.value = 0
        resultados: list[dict] = []
        try:
            for i, t in enumerate(tickers, 1):
                if self.detener:
                    break
                self.lbl_scan.text = f"Analizando {t} ({i}/{len(tickers)})..."
                try:
                    r = await calcular_ticker(t, cap, self.sw_bt_wl.value, True, self.sw_vj_wl.value)
                except Exception as e:  # noqa: BLE001  (un fallo individual no aborta el escaneo)
                    r = {"ticker": t, "precio_actual": None, "zonas": [], "errores": [f"{type(e).__name__}: {e}"]}
                if r.get("precio_actual") is not None:
                    previo = CACHE.get(t) or {}
                    if previo.get("entradas_valoracion") and not r.get("entradas_valoracion"):
                        r["entradas_valoracion"] = previo["entradas_valoracion"]  # el escaneo (sin valor justo) no las borra
                    CACHE[t] = r  # LRU en la nube: solo los ultimos 3 conservan el df completo
                liviano = resumen_liviano(r)  # el ranking solo necesita el resumen, no df/form4/r13f
                if self.sw_vj_wl.value:
                    liviano["valor_justo"] = resumen_valor_justo(r)  # mismas entradas que la pestana Valor Justo
                liviano.pop("entradas_valoracion", None)  # ya resumidas: no retenerlas en la lista del ranking
                resultados.append(liviano)
                del r, liviano
                gc.collect()  # secuencial + liberar memoria tras cada ticker (512 MB en la nube)
                self.prog.value = i / len(tickers)
        finally:
            ESCANEO_NUBE["activo"] = False
            self._ocupar(False)
            self.btn_stop.visible = False
            self.prog.visible = False
        filas, fallidos = filas_ranking(resultados)
        self.tabla_wl.rows = filas
        self.tabla_wl.update()
        self.lbl_scan.text = (f"{len(filas)} activos ranqueados" + (f" | sin datos: {', '.join(fallidos)}" if fallidos else "")
                              + (f" | nube: máximo {MAX_TICKERS_ESCANEO_NUBE} por escaneo, {omitidos} omitido(s)" if omitidos else "")
                              + (" | cobertura distinta entre activos: sus scores no son comparables"
                                 if coberturas_distintas(filas) else ""))
        if fallidos:
            ui.notify(f"Sin datos: {', '.join(fallidos)}", type="warning")

    # ---- render (sin recomputar)
    def refrescar(self, solo_figura: bool = False) -> None:
        if not hasattr(self, "tabla"):
            return
        self._actualizar_btn_wl()
        res = CACHE.get(self.ticker or "")
        capital = self._capital() or 1000.0
        if res is None:
            self._pintar_grafico(None, [], None, None)
            self.tabla.rows = []
            self.tabla.update()
            self._tarjetas(None, [])
            self._vj_actualizar()
            return
        act = familias_activas(k for k, v in self.activos.items() if v)
        av = anclas_valoracion(res)
        zonas = filtrar_zonas(res.get("zonas") or [], act, av, capital)
        c = res.get("confiabilidad") or {}
        cob = _num(c.get("cobertura"))
        sub = (f"Confluencia {_fmt(res.get('confluencia_score'))}/100 | Confiabilidad {_fmt(c.get('indice'))}/100"
               + ("" if cob is None else f" (cobertura {cob * 100:.0f} %)"))
        cambio = self._vj_cambio_entradas()  # ANTES de _vj_para_grafico, que sincroniza y lo dejaria en False
        vj = self._vj_para_grafico()
        self._pintar_grafico(res, zonas, av if "valoracion" in act else None, sub, vj)
        if solo_figura:
            return
        if cambio:
            self._vj_actualizar()  # entradas nuevas (otro ticker o nuevo Recalcular)
        filas = filas_tabla(zonas, res.get("backtest_resumen"), res.get("precio_actual"))
        self.tabla.rows = filas
        self.tabla.selected = [f for f in filas if f["id"] == self.nivel_sel]
        self.tabla.update()
        self._tarjetas(res, zonas)

    def _pintar_grafico(self, res: dict | None, zonas: list[dict], anclas: list[dict] | None, sub: str | None,
                        vj: dict | None = None) -> None:
        """Actualiza el grafico Lightweight Charts sin recrearlo: las velas/perfil/marcadores solo se reenvian si cambia el
        ticker o los datos; zonas, anclas, seleccion, precio y valor justo viajan siempre (payload pequeno)."""
        res = res or {}
        ticker = res.get("ticker") or self.ticker or "-"
        anios = self.anios_velas if self.anios_velas is not None else anios_velas_defecto()
        huella = huella_datos(ticker, res.get("df"), res.get("form4"), res.get("r13f"), anios)
        enviar = huella != self._grafico_huella or self.grafico_datos is None
        if enviar:
            self.grafico_datos = datos_grafico(res.get("df"), ticker, res.get("form4"), res.get("r13f"), anios)
            self._grafico_huella = huella
        self.grafico_overlays = overlays_grafico(self.grafico_datos, ticker, zonas, res.get("precio_actual"), anclas,
                                                 self.nivel_sel, sub, vj)
        self._grafico_js(js_actualizar(self.grafico_id, tema_grafico(), self.grafico_datos if enviar else None,
                                       self.grafico_overlays))

    def _grafico_js(self, codigo: str) -> None:
        """Unico punto que habla con el navegador (los tests lo sustituyen); se encola hasta que el cliente conecte."""
        self.ultimo_js_grafico = codigo
        try:
            self.grafico.client.run_javascript(codigo)
        except RuntimeError:   # sin bucle de eventos (pruebas unitarias sin servidor) o cliente ya eliminado: nada que pintar
            pass

    def _tarjetas(self, res: dict | None, zonas: list[dict]) -> None:
        self.avisos.clear()
        if res is None:
            for k in self.tarj:
                self.tarj[k].text = "n/d"
            self.tarj_cob.text = ""
            self.tarj_excl.text = ""
            self.lbl_bt.text = texto_alcance_backtest({})
            with self.avisos:
                self.ui.label(f"{self.ticker}: pulsa Recalcular para analizarlo." if self.ticker
                              else "Elige un ticker y pulsa Recalcular.").classes("txt-muted")
            return
        d = resumen_tarjetas(res, zonas)
        self.tarj["precio"].text = _fmt(d["precio"], "${:,.2f}")
        self.tarj["confluencia"].text = _fmt(d["confluencia"]) + "/100"
        self.tarj["confiabilidad"].text = _fmt(d["confiabilidad"]) + "/100"
        self.tarj_cob.text = "" if d["cobertura"] is None else f"cobertura {d['cobertura'] * 100:.0f} %"
        self.tarj_excl.text = f"sin dato: {', '.join(d['sin_dato'])}" if d["sin_dato"] else ""
        self.tarj["promedio"].text = _fmt(d["precio_promedio"], "${:,.2f}")
        self.lbl_bt.text = d["alcance_backtest"]
        with self.avisos:
            for a in d["avisos"]:
                fuerte = "ALERTA" in a or a in d["avisos_datos"]
                self.ui.label(a).classes("txt-aviso text-caption" if fuerte else "txt-muted text-caption")
            if d["earnings"] and not d["earnings_alerta"]:
                self.ui.label(d["earnings"]).classes("txt-muted text-caption")
            for e in d["errores"]:
                self.ui.label(e).classes("txt-error text-caption")
            if not (d["avisos"] or d["errores"] or d["earnings"]):
                self.ui.label("Sin avisos.").classes("txt-muted text-caption")


def _fmt(v: float | None, f: str = "{:.1f}") -> str:
    return "n/d" if v is None else f.format(v)


def pagina() -> None:
    """Pagina raiz: construye la UI de un cliente (se pasa como ``root`` a ``ui.run``)."""
    Vista().construir()


def config_ejecucion(env: dict | None = None, pywebview_ok: bool | None = None) -> dict:
    """Argumentos de ``ui.run`` segun el entorno (funcion pura, testeable).

    Nube (variable PORT): servidor 0.0.0.0, sin ventana nativa. Local: 127.0.0.1 y ventana nativa (pywebview) si esta
    disponible; si no, cae al navegador. ``APP_UI_NATIVE=0`` fuerza navegador; ``APP_UI_SHOW=0`` no lo abre.
    """
    env = os.environ if env is None else env
    nube = "PORT" in env
    nativo = not nube and env.get("APP_UI_NATIVE", "1") != "0"
    if nativo:
        if pywebview_ok is None:
            try:
                import webview  # noqa: F401
                pywebview_ok = True
            except Exception:  # noqa: BLE001  (ImportError o fallo del backend GUI)
                pywebview_ok = False
        if not pywebview_ok:
            print("[aviso] pywebview no esta disponible: se abre en el navegador (pip install pywebview para la "
                  "ventana nativa).")
            nativo = False
    cfg = dict(host="0.0.0.0" if nube else "127.0.0.1", port=int(env.get("PORT") or env.get("APP_UI_PORT") or 8080),
               native=nativo, reload=False,
               title="Niveles de Compra Institucional", dark=True)
    if nativo:
        cfg["window_size"] = (1440, 900)  # ojo: en ui.run, window_size ya activa native; solo se pasa si es nativo
    else:
        cfg["show"] = not nube and env.get("APP_UI_SHOW", "1") != "0"
    return cfg


def ejecutar() -> None:
    from nicegui import ui
    ui.run(pagina, **config_ejecucion())


if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing
    multiprocessing.freeze_support()  # NiceGUI nativo usa multiprocessing; imprescindible en el .exe de PyInstaller
    ejecutar()
