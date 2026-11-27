"""Auditoria walk-forward de rangos de compra [r_inf, r_sup] (sin look-ahead).

Metodo
------
* En cada fecha de rebalanceo t (cada ``PASO_REBALANCEO`` ruedas) el rango se calcula SOLO con datos <= t
  (``df.iloc[:i+1]``), reutilizando las anclas de ``src.levels.volume_liquidity`` (Volume Profile, AVWAP,
  Order Blocks semanales) mas swing lows semanales macro y bandas de regresion logaritmica
  (``src.levels.valuation_floors.bandas_log_regresion``). Los pisos de valoracion basados en XBRL
  (P/E, EV/FCF, DCF) NO se usan aqui porque ``_facts`` devuelve el historico completo de hoy (look-ahead).
* Las anclas se ensanchan (puntuales: +-0.25 ATR semanal), se agrupan en clusters (ancho <= ``ANCHO_MAX_CLUSTER_ATR``
  ATR semanal y, opcionalmente, <= ``ANCHO_MAX_CLUSTER_PCT`` del precio) y el rango primario es el cluster con mayor
  confluencia de familias de ancla bajo el precio (<= 30 % debajo). Variantes de precision (``ParamsZonas``): modo
  "ventana" (sin fusion en cadena), HVN/OB estrechados y rango operativo = nucleo del cluster (``lo_nucleo``/``hi_nucleo``;
  ``lo_amplio``/``hi_amplio`` conservan el cluster completo). Ver el A/B en docs/METODOLOGIA.md.
* Orden limite GTC en r_sup con vigencia ``VIDA_ORDEN`` ruedas. Llenado si Low <= r_sup; precio de entrada =
  min(Open del dia, r_sup) (modela gaps a la baja).
* Metricas: fill rate, absorcion (vela de rechazo vol > 1.35x SMA20 previa, cierre en el tercio superior,
  dentro del rango, ventana de 5 ruedas desde el toque), bounce rate (+5 % antes de perder r_inf - 3 % en 20
  ruedas), retornos forward 20/60/120/252 (media/mediana, censura por derecha) y MAE a 60 y 252 ruedas.
* Linea base incondicional: compra al cierre en cada fecha de rebalanceo (mismos horizontes).

Referencias: Steidlmayer (Market Profile), Shannon (AVWAP), Wilder (ATR), Graham/Damodaran (valoracion).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.levels import volume_liquidity as vl
from src.levels.valuation_floors import bandas_log_regresion

# --- Parametros (documentados en docs/METODOLOGIA.md) ----------------------------------------------
TICKERS_DEFECTO = ["AAPL", "MSFT", "META", "OXY", "GOOGL", "UNH"]
ANIOS_BACKTEST = 10
PASO_REBALANCEO = 21           # ~1 mes bursatil
VIDA_ORDEN = 63                # ~3 meses: vigencia de la orden GTC antes de recalcular
ANCHO_PUNTUAL_ATR = 0.25       # semiancho de anclas puntuales (POC, AVWAP, swing, log-reg) en ATR semanal
# --- Precision de las zonas (A/B walk-forward de 32 tickers x 10 anios; ver docs/METODOLOGIA.md, seccion de precision) ---
# Defaults = variante "G15" del A/B: la mediana del ancho del rango primario baja de 6.1 % a 4.0 % del precio, el p90 de
# 14.4 % a 5.0 %, con retornos a 60/120/252 ruedas y MAE dentro del ruido y un fill rate ~3.7 pp menor (36.2 % -> 32.5 %).
# Los valores historicos (ANCHO_MAX_CLUSTER_ATR=3.0, sin tope %, sin acotar anclas, HVN/OB completos, cualquier solape con
# la banda) se reproducen con ``params_zonas(**PARAMS_HISTORICOS)``.
ANCHO_MAX_CLUSTER_ATR = 2.0    # tope de ancho de un cluster (ATR semanal); historico: 3.0
ANCHO_MAX_CLUSTER_PCT = 0.05   # tope relativo adicional (fraccion del precio); None = sin tope relativo (historico)
MODO_CLUSTER = "cadena"        # "cadena" (fusion por solapamiento, ancho <= tope) | "ventana" (ventana de maxima confluencia)
ACOTAR_ANCLAS = True           # ninguna ancla supera el tope de ancho del cluster (se acota centrada en su pico/centro)
HVN_SEMIANCHO_ATR = 0.5        # HVN = pico del nodo +- 0.5 ATR semanal (dentro del FWHM); None = nodo completo (historico)
OB_ANCHO_MAX_ATR = 1.0         # ancho maximo de un Order Block (ATR semanal); None = completo (historico)
OB_DESDE_TECHO = True          # el OB acotado conserva su borde superior (demanda: el que toca primero el precio)
FRAC_MIN_EN_BANDA = 0.5        # descarta anclas con < 50 % de su rango dentro de la banda [-30 %, -0.5 %] (astillas); None = historico
NUCLEO_OPERATIVO = False       # True: lo/hi = nucleo del cluster (lo_amplio/hi_amplio conservan el cluster completo)
NUCLEO_FRAC_PESO = 0.6         # el nucleo reune >= 60 % del peso de las anclas del cluster ("mayoria ponderada")
NUCLEO_PAD_ATR = 0.1           # relleno del nucleo a cada lado de los centros extremos (ATR semanal)
NUCLEO_ANCHO_MIN_ATR = 0.3     # ancho minimo del nucleo (ATR semanal)
DIST_MIN, DIST_MAX = 0.005, 0.30
VOL_MULT_ABS = 1.35
TERCIO_SUP = 2.0 / 3.0
VENTANA_ABS = 5
BOUNCE_PCT, BOUNCE_VENTANA, BOUNCE_INVALIDA = 0.05, 20, 0.03
HORIZONTES = (20, 60, 120, 252)
MIN_BARRAS_WARMUP = 504        # 2 anios de historia antes del primer rango
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")

FAMILIAS = {"POC": "volumen", "VAL": "volumen", "VAH": "volumen", "HVN": "volumen",
            "AVWAP": "avwap", "OB": "ob", "SWING": "swing", "LOG": "logreg"}


# --- Anclas y clusters PIT ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamsZonas:
    """Parametros de construccion de zonas (todos sobreescribibles para el A/B; ver constantes del modulo).

    Los valores por defecto de ESTA clase son los historicos (previos al A/B de precision); los parametros vigentes
    (defaults de produccion) los devuelve ``params_zonas()`` a partir de las constantes del modulo."""
    ancho_puntual_atr: float = 0.25
    ancho_max_cluster_atr: float = 3.0
    ancho_max_cluster_pct: float | None = None
    modo: str = "cadena"                  # "cadena" (fusion por solapamiento) | "ventana" (ventana de maxima confluencia)
    acotar_anclas: bool = False           # True: ninguna ancla supera el tope de ancho (siempre activo en modo "ventana")
    hvn_semiancho_atr: float | None = None
    ob_ancho_max_atr: float | None = None
    ob_desde_techo: bool = False          # True: un OB acotado conserva su borde superior (el que toca primero el precio)
    frac_min_en_banda: float | None = None  # descarta anclas con < esta fraccion de su rango dentro de [-30 %, -0.5 %]
    nucleo: bool = False                  # True: lo/hi operativos = nucleo (lo_amplio/hi_amplio conservan el cluster)
    nucleo_frac_peso: float = 0.6
    nucleo_pad_atr: float = 0.1
    nucleo_ancho_min_atr: float = 0.3

    def ancho_max(self, px: float, atr: float) -> float:
        """Tope de ancho del cluster en precio: el menor entre el tope en ATR semanal y el tope relativo (% del precio)."""
        w = self.ancho_max_cluster_atr * atr
        return w if self.ancho_max_cluster_pct is None else min(w, self.ancho_max_cluster_pct * px)


PARAMS_HISTORICOS = dict(ancho_max_cluster_atr=3.0, ancho_max_cluster_pct=None, acotar_anclas=False,
                         hvn_semiancho_atr=None, ob_ancho_max_atr=None, ob_desde_techo=False,
                         frac_min_en_banda=None, nucleo=False, modo="cadena")


def params_zonas(**kw) -> ParamsZonas:
    """Parametros vigentes: valores por defecto = constantes del modulo (leidas en la llamada, asi que un
    ``monkeypatch`` de las constantes tambien funciona) salvo lo que se pase en ``kw``."""
    base = dict(ancho_puntual_atr=ANCHO_PUNTUAL_ATR, ancho_max_cluster_atr=ANCHO_MAX_CLUSTER_ATR,
                ancho_max_cluster_pct=ANCHO_MAX_CLUSTER_PCT, modo=MODO_CLUSTER, acotar_anclas=ACOTAR_ANCLAS,
                hvn_semiancho_atr=HVN_SEMIANCHO_ATR, ob_ancho_max_atr=OB_ANCHO_MAX_ATR, ob_desde_techo=OB_DESDE_TECHO,
                frac_min_en_banda=FRAC_MIN_EN_BANDA, nucleo=NUCLEO_OPERATIVO, nucleo_frac_peso=NUCLEO_FRAC_PESO,
                nucleo_pad_atr=NUCLEO_PAD_ATR, nucleo_ancho_min_atr=NUCLEO_ANCHO_MIN_ATR)
    base.update(kw)
    return ParamsZonas(**base)


def swing_lows_semanales(df: pd.DataFrame, n: int = vl.OB_SWING_SEMANAL) -> list[float]:
    """Pivotes bajistas semanales confirmados: el Low es minimo de +-n semanas (confirmado tras n semanas)."""
    w = vl.a_semanal(df)
    lows = w["Low"].values
    out = []
    for i in range(n, len(lows) - n):
        if lows[i] == lows[i - n:i + n + 1].min():
            out.append(float(lows[i]))
    return out


def anclas_pit(df: pd.DataFrame, params: ParamsZonas | None = None) -> tuple[list[dict], float, float]:
    """(anclas, precio, atr_semanal) usando exclusivamente ``df`` (datos hasta t).

    Cada ancla: tipo, familia, lo, hi (rango completo ya ensanchado si es puntual), ``centro`` (pico del HVN, punto de
    las puntuales, centro del OB) y ``puntual`` (True si su rango es solo el semiancho de ruido +-``ancho_puntual_atr``).
    ``clusters_pit`` usa ``centro``/``puntual`` para estrechar anclas sin recalcular el perfil de volumen.
    """
    p = params or params_zonas()
    d = vl._normalizar(df)
    px = float(d["Close"].iloc[-1])
    atr = vl.atr_estructural_semanal(d)
    vp = vl.volume_profile(d, atr=atr)
    puntual = p.ancho_puntual_atr * atr
    anclas: list[dict] = []

    def add(tipo, lo, hi=None, centro=None):
        es_punto = hi is None or hi == lo
        hi = lo if hi is None else hi
        c = (lo + hi) / 2.0 if centro is None else centro
        if es_punto:
            lo, hi = lo - puntual, hi + puntual
        anclas.append({"tipo": tipo, "familia": FAMILIAS[tipo], "lo": float(lo), "hi": float(hi),
                       "centro": float(c), "puntual": bool(es_punto)})

    add("POC", vp["poc"])
    add("VAL", vp["val"])
    add("VAH", vp["vah"])
    for h in vp["hvn"]:
        add("HVN", h["lo"], h["hi"], centro=h["precio"])
    for v in vl.avwap_multiciclo(d, None, None).values():  # sin ticker: no consulta EDGAR (sin red, sin look-ahead)
        add("AVWAP", v["avwap"])
    try:
        for o in vl.order_blocks_semanales(d, px):
            add("OB", o["bottom"], o["top"])
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"OB no disponible: {e}")
    for s in swing_lows_semanales(d):
        add("SWING", s)
    try:
        for v in bandas_log_regresion(d)["bandas"].values():
            if np.isfinite(v) and v > 0:
                add("LOG", v)
    except Exception:  # noqa: BLE001
        pass
    return anclas, px, atr


def _estrechar(a: dict, atr: float, p: ParamsZonas, tope_ancho: float | None) -> dict:
    """Copia del ancla con el rango estrechado segun ``p`` (siempre estructura acumulada, nunca el cierre del dia):
    puntual -> centro +- ``ancho_puntual_atr``; HVN -> pico +- ``hvn_semiancho_atr`` (dentro del nodo); OB -> como
    maximo ``ob_ancho_max_atr`` (conserva el borde superior si ``ob_desde_techo``, si no centrado); con
    ``tope_ancho`` ninguna ancla lo supera (centrada en su pico/centro; el OB desde su borde superior)."""
    b = dict(a)
    c = a.get("centro", (a["lo"] + a["hi"]) / 2.0)
    if a.get("puntual"):
        b["lo"], b["hi"] = c - p.ancho_puntual_atr * atr, c + p.ancho_puntual_atr * atr
    if a["tipo"] == "HVN" and p.hvn_semiancho_atr is not None:
        b["lo"], b["hi"] = max(b["lo"], c - p.hvn_semiancho_atr * atr), min(b["hi"], c + p.hvn_semiancho_atr * atr)
    techo = a["tipo"] == "OB" and p.ob_desde_techo
    if a["tipo"] == "OB" and p.ob_ancho_max_atr is not None and b["hi"] - b["lo"] > p.ob_ancho_max_atr * atr:
        if techo:
            b["lo"] = b["hi"] - p.ob_ancho_max_atr * atr
        else:
            m = (b["lo"] + b["hi"]) / 2.0
            b["lo"], b["hi"] = m - p.ob_ancho_max_atr * atr / 2.0, m + p.ob_ancho_max_atr * atr / 2.0
    if tope_ancho is not None and b["hi"] - b["lo"] > tope_ancho:
        if techo:
            b["lo"] = b["hi"] - tope_ancho
        else:
            m = min(max(c, b["lo"]), b["hi"])
            b["lo"], b["hi"] = max(b["lo"], m - tope_ancho / 2.0), min(b["hi"], m + tope_ancho / 2.0)
    b["centro"] = min(max(c, b["lo"]), b["hi"])
    b["peso"] = float(a.get("peso", 1.0))
    return b


def _agrupar_cadena(cand: list[dict], tope: float) -> list[list[dict]]:
    """Fusion por solapamiento en orden de ``lo`` mientras el ancho de la union <= ``tope`` (comportamiento historico)."""
    cand = sorted(cand, key=lambda a: a["lo"])  # estable: mismo orden que la version historica
    grupos, cur, lo, hi = [], None, 0.0, 0.0
    for a in cand:
        if cur is not None and a["lo"] <= hi and max(hi, a["hi"]) - lo <= tope:
            hi = max(hi, a["hi"])
            cur.append(a)
        else:
            if cur is not None:
                grupos.append(cur)
            cur, lo, hi = [a], a["lo"], a["hi"]
    if cur is not None:
        grupos.append(cur)
    return grupos


def _agrupar_ventana(cand: list[dict], tope: float) -> list[list[dict]]:
    """Sin fusion en cadena: cada cluster es la ventana de ancho <= ``tope`` que contiene la mayor confluencia
    (familias, peso) entre las anclas aun libres; las anclas contenidas salen del pool y se repite. Una ancla pertenece
    a un unico cluster y ningun cluster supera ``tope`` (las anclas se acotan antes a ``tope``)."""
    libres = sorted(cand, key=lambda a: (a["lo"], a["hi"], a["tipo"]))
    grupos = []
    eps = 1e-9
    while libres:
        mejor, mejor_clave = None, None
        for s in libres:
            g = [a for a in libres if a["lo"] >= s["lo"] - eps and a["hi"] <= s["lo"] + tope + eps]
            clave = (len({a["familia"] for a in g}), sum(a["peso"] for a in g), max(a["hi"] for a in g))
            if mejor_clave is None or clave > mejor_clave:
                mejor, mejor_clave = g, clave
        grupos.append(mejor)
        ids = {id(a) for a in mejor}
        libres = [a for a in libres if id(a) not in ids]
    return grupos


def _nucleo(comps: list[dict], atr: float, p: ParamsZonas, lo_amp: float, hi_amp: float) -> tuple[float, float]:
    """Nucleo del cluster: la sub-ventana de centros de ancla mas estrecha que reune >= ``nucleo_frac_peso`` del peso
    total, con un relleno de ``nucleo_pad_atr`` y un ancho minimo de ``nucleo_ancho_min_atr``, dentro del rango amplio."""
    pts = sorted((a["centro"], a["peso"]) for a in comps)
    total = sum(w for _, w in pts)
    need = p.nucleo_frac_peso * total - 1e-12
    mejor, mejor_clave = (pts[0][0], pts[-1][0]), None
    for i in range(len(pts)):
        acum = 0.0
        for j in range(i, len(pts)):
            acum += pts[j][1]
            if acum >= need:
                clave = (-(pts[j][0] - pts[i][0]), acum, pts[j][0])  # mas estrecho, mas peso, mas cercano al precio
                if mejor_clave is None or clave > mejor_clave:
                    mejor, mejor_clave = (pts[i][0], pts[j][0]), clave
                break
    lo, hi = mejor[0] - p.nucleo_pad_atr * atr, mejor[1] + p.nucleo_pad_atr * atr
    ancho = min(max(hi - lo, p.nucleo_ancho_min_atr * atr), hi_amp - lo_amp)
    lo = min(max((lo + hi) / 2.0 - ancho / 2.0, lo_amp), hi_amp - ancho)
    return float(lo), float(lo + ancho)


def clusters_pit(anclas: list[dict], px: float, atr: float, params: ParamsZonas | None = None) -> list[dict]:
    """Agrupa anclas bajo el precio en zonas de confluencia.

    Cada cluster conserva sus claves historicas (``lo``, ``hi``, ``anclas``, ``n_anclas``, ``n_familias``, ``familias``,
    ``tipos``, ``px``, ``dist_pct``, ``ancho_pct``). ``lo``/``hi`` son el rango OPERATIVO: el rango amplio del cluster
    (``lo_amplio``/``hi_amplio``) o, con ``params.nucleo``, su nucleo (``lo_nucleo``/``hi_nucleo``, siempre calculado).
    Trazabilidad: ``anclas`` (con su ``peso``), ``peso_total`` y ``anclas_nucleo`` (tipos cuyo centro cae en el nucleo).
    """
    p = params or params_zonas()
    tope = px * (1 - DIST_MIN)
    piso = px * (1 - DIST_MAX)
    w_max = p.ancho_max(px, atr)
    ventana = p.modo == "ventana"
    cand = []
    for a in anclas:
        b = _estrechar(a, atr, p, w_max if (ventana or p.acotar_anclas) else None)
        if b["hi"] > piso and b["lo"] < tope:
            if p.frac_min_en_banda is not None and b["hi"] > b["lo"] and                     (min(b["hi"], tope) - max(b["lo"], piso)) / (b["hi"] - b["lo"]) < p.frac_min_en_banda:
                continue  # ancla casi toda fuera de la banda de busqueda: recortarla dejaria una astilla sin significado
            b["hi"] = min(b["hi"], tope)
            if ventana:
                b["lo"] = max(b["lo"], piso)
            b["centro"] = min(max(b["centro"], b["lo"]), b["hi"])
            cand.append(b)
    grupos = _agrupar_ventana(cand, w_max) if ventana else _agrupar_cadena(cand, w_max)
    clusters = []
    for g in grupos:
        lo_a, hi_a = max(min(a["lo"] for a in g), piso), max(a["hi"] for a in g)
        fams = {a["familia"] for a in g}
        lo_n, hi_n = _nucleo(g, atr, p, lo_a, hi_a)
        lo, hi = (lo_n, hi_n) if p.nucleo else (lo_a, hi_a)
        clusters.append({"lo": lo, "hi": hi, "lo_amplio": lo_a, "hi_amplio": hi_a, "lo_nucleo": lo_n, "hi_nucleo": hi_n,
                         "anclas": g, "n_anclas": len(g), "n_familias": len(fams), "familias": sorted(fams),
                         "tipos": sorted({a["tipo"] for a in g}), "peso_total": float(sum(a["peso"] for a in g)),
                         "anclas_nucleo": sorted(a["tipo"] for a in g if lo_n <= a["centro"] <= hi_n), "px": px,
                         "dist_pct": (px - hi) / px, "ancho_pct": (hi - lo) / px,
                         "ancho_amplio_pct": (hi_a - lo_a) / px, "ancho_atr": (hi - lo) / atr})
    clusters.sort(key=lambda c: (c["lo"], c["hi"]))
    return clusters


def rango_primario(clusters: list[dict]) -> dict | None:
    """Cluster de mayor confluencia (familias, anclas); empate -> el mas cercano al precio."""
    if not clusters:
        return None
    return sorted(clusters, key=lambda c: (-c["n_familias"], -c["n_anclas"], c["dist_pct"]))[0]


def calcular_zonas_pit(df: pd.DataFrame, params: ParamsZonas | None = None) -> dict:
    """API publica: zonas PIT con datos hasta la ultima fila de ``df``. Devuelve clusters y rango primario."""
    p = params or params_zonas()
    anclas, px, atr = anclas_pit(df, p)
    cl = clusters_pit(anclas, px, atr, p)
    return {"px": px, "atr_semanal": atr, "clusters": cl, "primario": rango_primario(cl)}


# --- Evaluacion de una orden ----------------------------------------------------------------------------
def evaluar_orden(h: pd.DataFrame, i0: int, r_inf: float, r_sup: float, vol_sma_prev: np.ndarray) -> dict:
    """Simula la orden GTC colocada al cierre de la barra i0 (solo barras > i0)."""
    n = len(h)
    O, H, L, C, V = (h[c].values for c in ("Open", "High", "Low", "Close", "Volume"))
    fin = min(i0 + VIDA_ORDEN, n - 1)
    ventana_completa = (i0 + VIDA_ORDEN) <= n - 1
    k = next((j for j in range(i0 + 1, fin + 1) if L[j] <= r_sup), None)
    res = {"ventana_completa": ventana_completa, "llenada": k is not None}
    if k is None:
        return res
    entrada = min(O[k], r_sup)
    res.update(idx_fill=k, fecha_fill=h.index[k], entrada=float(entrada), dias_hasta_fill=k - i0,
               gap_bajo_rinf=bool(O[k] < r_inf), profunda=bool(L[k] <= r_inf))
    # Absorcion: vela de rechazo en rango, ventana de 5 ruedas desde el toque
    absorb = False
    for j in range(k, min(k + VENTANA_ABS, n - 1) + 1):
        rng = H[j] - L[j]
        if rng <= 0 or np.isnan(vol_sma_prev[j]):
            continue
        if (L[j] <= r_sup and H[j] >= r_inf and V[j] > VOL_MULT_ABS * vol_sma_prev[j]
                and (C[j] - L[j]) / rng >= TERCIO_SUP and C[j] >= r_inf):
            absorb = True
            break
    res["absorcion"] = absorb
    # Bounce (+5 % antes de invalidar en r_inf - 3 %) en 20 ruedas
    if k + BOUNCE_VENTANA <= n - 1:
        meta, inval, bounce = entrada * (1 + BOUNCE_PCT), r_inf * (1 - BOUNCE_INVALIDA), False
        for j in range(k, k + BOUNCE_VENTANA + 1):
            if L[j] < inval:
                break
            if H[j] >= meta:
                bounce = True
                break
        res["bounce"] = bounce
    for hz in HORIZONTES:
        if k + hz <= n - 1:
            res[f"ret{hz}"] = float(C[k + hz] / entrada - 1)
    for hz in (60, 252):
        if k + hz <= n - 1:
            res[f"mae{hz}"] = float(L[k:k + hz + 1].min() / entrada - 1)
    return res


# --- Backtest por ticker ---------------------------------------------------------------------------------
def _descargar(tk: str) -> pd.DataFrame:
    from src.data.market_data import descargar_historial
    os.makedirs(CACHE_DIR, exist_ok=True)
    ruta = os.path.join(CACHE_DIR, f"px_{tk}.pkl")
    if os.path.exists(ruta) and time.time() - os.path.getmtime(ruta) < 12 * 3600:
        return pd.read_pickle(ruta)
    _, df = descargar_historial(tk, "max")
    if df.empty:
        raise RuntimeError(f"sin datos de mercado para {tk}")
    df = vl._normalizar(df)
    df.to_pickle(ruta)
    return df


def backtest_ticker(tk: str, anios: int = ANIOS_BACKTEST, paso: int = PASO_REBALANCEO, df: pd.DataFrame | None = None,
                    verbose: bool = True, params: ParamsZonas | None = None,
                    anclas_cache: dict | None = None) -> pd.DataFrame:
    """Una fila por orden (llenada o no). Rangos estrictamente walk-forward.

    ``params``: parametros de zonas (por defecto los vigentes, ver ``params_zonas``). ``anclas_cache``: dict opcional
    {i0: (anclas, px, atr)} que se llena en la primera llamada y se reutiliza despues (las anclas no dependen de los
    parametros de ancho/cluster, solo ``clusters_pit`` los usa): permite comparar variantes A/B sin recalcular el perfil
    de volumen ni los Order Blocks. Cada entrada se calculo con ``df.iloc[:i0+1]`` (sin look-ahead)."""
    p_z = params or params_zonas()
    df = _descargar(tk) if df is None else vl._normalizar(df)
    n = len(df)
    inicio = max(MIN_BARRAS_WARMUP, n - int(anios * 252))
    vol_sma_prev = df["Volume"].rolling(20).mean().shift(1).values
    filas = []
    t0 = time.time()
    for i0 in range(inicio, n - 1, paso):
        try:
            if anclas_cache is not None and i0 in anclas_cache:
                anclas, px_t, atr_t = anclas_cache[i0]
            else:
                anclas, px_t, atr_t = anclas_pit(df.iloc[:i0 + 1], p_z)  # <- unico dato visible: <= t
                if anclas_cache is not None:
                    anclas_cache[i0] = (anclas, px_t, atr_t)
            cl_t = clusters_pit(anclas, px_t, atr_t, p_z)
            z = {"px": px_t, "atr_semanal": atr_t, "clusters": cl_t, "primario": rango_primario(cl_t)}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{tk} {df.index[i0].date()}: {e}")
            continue
        base = {"ticker": tk, "fecha": df.index[i0], "px": z["px"]}
        # linea base incondicional: compra al cierre de t
        for hz in HORIZONTES:
            if i0 + hz <= n - 1:
                base[f"base_ret{hz}"] = float(df["Close"].iloc[i0 + hz] / z["px"] - 1)
        for hz in (60, 252):
            if i0 + hz <= n - 1:
                base[f"base_mae{hz}"] = float(df["Low"].iloc[i0:i0 + hz + 1].min() / z["px"] - 1)
        p = z["primario"]
        if p is None:
            base["sin_rango"] = True
            filas.append(base)
            continue
        base.update(r_inf=p["lo"], r_sup=p["hi"], n_familias=p["n_familias"], n_anclas=p["n_anclas"],
                    familias="+".join(p["familias"]), tipos="+".join(p["tipos"]), dist_pct=p["dist_pct"],
                    ancho_pct=p["ancho_pct"], ancho_atr=p["ancho_atr"], lo_amplio=p["lo_amplio"],
                    hi_amplio=p["hi_amplio"], ancho_amplio_pct=p["ancho_amplio_pct"], atr_pct=z["atr_semanal"] / z["px"],
                    sin_rango=False)
        for f in ("volumen", "avwap", "ob", "swing", "logreg"):
            base[f"has_{f}"] = f in p["familias"]
        base.update(evaluar_orden(df, i0, p["lo"], p["hi"], vol_sma_prev))
        filas.append(base)
    if verbose:
        print(f"  {tk}: {len(filas)} ordenes en {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(filas)


def ejecutar_backtest(tickers=TICKERS_DEFECTO, anios: int = ANIOS_BACKTEST, usar_cache: bool = True) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = []
    for tk in tickers:
        ruta = os.path.join(CACHE_DIR, f"ordenes_{tk}.pkl")
        px_ruta = os.path.join(CACHE_DIR, f"px_{tk}.pkl")
        if usar_cache and os.path.exists(ruta) and os.path.exists(px_ruta) and \
                os.path.getmtime(ruta) > os.path.getmtime(px_ruta):
            out.append(pd.read_pickle(ruta))
            continue
        try:
            o = backtest_ticker(tk, anios)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: ERROR {e}")
            continue
        o.to_pickle(ruta)
        out.append(o)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --- Resumen ---------------------------------------------------------------------------------------------
def _col(d, c):
    return d[c].dropna() if c in d else pd.Series(dtype=float)


def resumen(o: pd.DataFrame) -> dict:
    v = o[(~o.get("sin_rango", False).astype(bool)) & o["ventana_completa"].astype(bool)] if len(o) else o
    llen = v[v["llenada"].astype(bool)]
    r = {"n_fechas": int(len(o)), "n_ordenes": int(len(v)), "n_fills": int(len(llen)),
         "fill_rate": len(llen) / len(v) if len(v) else np.nan,
         "fill_profundo": float(llen["profunda"].astype(bool).mean()) if len(llen) else np.nan,
         "dias_medios_fill": float(llen["dias_hasta_fill"].mean()) if len(llen) else np.nan,
         "absorcion": float(llen["absorcion"].astype(bool).mean()) if len(llen) else np.nan,
         "bounce": float(_col(llen, "bounce").astype(bool).mean()) if len(_col(llen, "bounce")) else np.nan}
    for hz in HORIZONTES:
        s, b = _col(llen, f"ret{hz}"), _col(o, f"base_ret{hz}")
        r[f"ret{hz}_media"], r[f"ret{hz}_mediana"], r[f"ret{hz}_n"] = s.mean(), s.median(), len(s)
        r[f"base{hz}_media"], r[f"base{hz}_mediana"] = b.mean(), b.median()
    for hz in (60, 252):
        s = _col(llen, f"mae{hz}")
        r[f"mae{hz}_media"], r[f"mae{hz}_mediana"] = s.mean(), s.median()
        r[f"base_mae{hz}_mediana"] = _col(o, f"base_mae{hz}").median()
    return r


def imprimir_resultados(ordenes: pd.DataFrame) -> pd.DataFrame:
    filas = {tk: resumen(g) for tk, g in ordenes.groupby("ticker")}
    filas["POOL"] = resumen(ordenes)
    t = pd.DataFrame(filas).T
    pct = lambda c: t[c].map(lambda x: "n/d" if pd.isna(x) else f"{x * 100:5.1f}%")  # noqa: E731
    print("\n=== TASAS DE LLENADO, ABSORCION Y REBOTE (rangos walk-forward, orden GTC en r_sup, vida 63 ruedas) ===")
    tab = pd.DataFrame({"ordenes": t["n_ordenes"].astype(int), "fills": t["n_fills"].astype(int),
                        "fill_rate": pct("fill_rate"), "fill_prof(<=r_inf)": pct("fill_profundo"),
                        "dias_a_fill": t["dias_medios_fill"].map(lambda x: f"{x:5.1f}"),
                        "absorcion": pct("absorcion"), "bounce(+5%/20d)": pct("bounce")})
    print(tab.to_string())
    for hz in HORIZONTES:
        print(f"\n--- Retorno forward {hz} ruedas desde el fill vs linea base (compra al cierre en cada rebalanceo) ---")
        tab = pd.DataFrame({"n": t[f"ret{hz}_n"].astype(int), "media_fill": pct(f"ret{hz}_media"),
                            "mediana_fill": pct(f"ret{hz}_mediana"), "media_base": pct(f"base{hz}_media"),
                            "mediana_base": pct(f"base{hz}_mediana")})
        print(tab.to_string())
    print("\n--- MAE (maxima excursion adversa) desde la entrada; base = idem desde el cierre de rebalanceo ---")
    tab = pd.DataFrame({"MAE60_media": pct("mae60_media"), "MAE60_mediana": pct("mae60_mediana"),
                        "base60_mediana": pct("base_mae60_mediana"), "MAE252_media": pct("mae252_media"),
                        "MAE252_mediana": pct("mae252_mediana"), "base252_mediana": pct("base_mae252_mediana")})
    print(tab.to_string())
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    tickers = [a.upper() for a in sys.argv[1:]] or TICKERS_DEFECTO
    print(f"Backtest walk-forward {ANIOS_BACKTEST}a, tickers={tickers}, paso={PASO_REBALANCEO}, vida={VIDA_ORDEN}")
    ordenes = ejecutar_backtest(tickers)
    if ordenes.empty:
        print("Sin resultados (sin datos de mercado).")
        return
    ordenes.to_pickle(os.path.join(CACHE_DIR, "ordenes_todas.pkl"))
    imprimir_resultados(ordenes)


if __name__ == "__main__":
    main()
