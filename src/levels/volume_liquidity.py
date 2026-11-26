"""Anclas de liquidez institucional: Volume Profile, Anchored VWAP multi-ciclo y Order Blocks semanales.

Referencias metodologicas
-------------------------
* Volume Profile (POC / HVN / VAL / VAH): Market Profile de J. Peter Steidlmayer (CBOT, 1985) y
  James Dalton, *Mind Over Markets* (1990/2013). El Value Area contiene el 70 % del volumen
  alrededor del POC (convencion de Dalton, ~1 desviacion estandar).
* Anchored VWAP: Brian Shannon, *Maximum Trading Gains With Anchored VWAP* (2023). El VWAP anclado a un
  evento estructural (minimo de ciclo, minimo de correccion mayor, reporte anual) es el costo medio
  ponderado por volumen de quienes compraron desde ese evento.
* Order Blocks no mitigados: conceptos Smart Money / ICT, implementados en la libreria
  ``smartmoneyconcepts`` (joshyattridge). Se usan velas semanales (1W) para evitar ruido intradia.
* ATR estructural: J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978).

Todas las funciones son puras respecto a ``df`` (OHLCV ajustado, indice fecha); la unica E/S de red
es la consulta de la fecha del ultimo 10-K en SEC EDGAR (con fallback documentado).
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- Parametros (cada uno justificado; no son niveles de precio) -------------------------------
ATR_PERIODO_SEMANAL = 14          # Wilder (1978): periodo canonico de 14 barras.
BIN_ATR_FRAC = 0.2                # Resolucion del perfil: 1 bin = 0.2 ATR semanal (resolucion, no nivel).
SUAVIZADO_BINS = 2.0              # Sigma gaussiana (en bins) para eliminar picos de ruido del histograma.
VALUE_AREA_PCT = 0.70             # Dalton: Value Area = 70 % del volumen.
HVN_PROMINENCIA_FRAC = 0.25       # Un HVN debe sobresalir >= 25 % del volumen del POC suavizado.
VP_ANIOS = 5                      # Ventana del perfil: 5 anios (requisito 3-5 anios).
AVWAP_MIN_ESTRUCTURAL_DIAS = 730  # Minimo estructural de 2 anios.
CORRECCION_MAYOR_PCT = 0.20       # Convencion bear market: caida >= 20 % desde maximo previo.
OB_SWING_SEMANAL = 10             # Pivotes semanales de +-10 semanas (~2.5 meses) = estructura macro.


# --- Utilidades ---------------------------------------------------------------------------------
def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con indice naive y columnas OHLCV numericas, sin NaN."""
    d = df.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    d = d[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    return d.sort_index()


def a_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1W (semana terminada en viernes) conservando OHLCV."""
    d = _normalizar(df)
    return d.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def atr_estructural_semanal(df: pd.DataFrame, periodo: int = ATR_PERIODO_SEMANAL) -> float:
    """ATR de Wilder sobre velas semanales; unidad de 'ruido estructural' para bins y clustering."""
    w = a_semanal(df)
    pc = w["Close"].shift(1)
    tr = pd.concat([w["High"] - w["Low"], (w["High"] - pc).abs(), (w["Low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / periodo, adjust=False, min_periods=periodo).mean()
    return float(atr.dropna().iloc[-1])


def _ancla(tipo, nombre, lo, hi=None, **meta):
    hi = lo if hi is None else hi
    return {"tipo": tipo, "clase": "liquidez", "nombre": nombre,
            "lo": float(min(lo, hi)), "hi": float(max(lo, hi)),
            "precio": float((lo + hi) / 2.0), "meta": meta}


# --- 1. Volume Profile --------------------------------------------------------------------------
def volume_profile(df: pd.DataFrame, anios: int = VP_ANIOS, atr: float | None = None) -> dict:
    """Perfil de volumen de ``anios`` anios: POC, HVN, VAL y VAH.

    El volumen de cada vela se reparte uniformemente entre los bins que cubre [Low, High]
    (no se usa solo el cierre, para no depender de un precio puntual: Regla de Invarianza).
    """
    d = _normalizar(df)
    d = d[d.index >= d.index[-1] - pd.DateOffset(years=anios)]
    atr = atr or atr_estructural_semanal(df)
    paso = BIN_ATR_FRAC * atr
    lo_g, hi_g = float(d["Low"].min()), float(d["High"].max())
    n = max(int(np.ceil((hi_g - lo_g) / paso)), 10)
    edges = lo_g + paso * np.arange(n + 1)
    hist = np.zeros(n)
    lows, highs, vols = d["Low"].values, d["High"].values, d["Volume"].values
    for l, h, v in zip(lows, highs, vols):
        i0 = min(int((l - lo_g) // paso), n - 1)
        i1 = min(int((h - lo_g) // paso), n - 1)
        hist[i0:i1 + 1] += v / (i1 - i0 + 1)
    suav = gaussian_filter1d(hist, SUAVIZADO_BINS)
    centros = (edges[:-1] + edges[1:]) / 2.0

    i_poc = int(np.argmax(suav))
    # Value Area (Dalton): expandir desde el POC hacia el lado con mas volumen hasta 70 %.
    total, acum = suav.sum(), suav[i_poc]
    a = b = i_poc
    while acum < VALUE_AREA_PCT * total and (a > 0 or b < n - 1):
        up = suav[b + 1] if b < n - 1 else -1.0
        dn = suav[a - 1] if a > 0 else -1.0
        if up >= dn:
            b += 1
            acum += suav[b]
        else:
            a -= 1
            acum += suav[a]
    val, vah = float(edges[a]), float(edges[b + 1])

    picos, props = find_peaks(suav, prominence=HVN_PROMINENCIA_FRAC * suav[i_poc])
    hvn = []
    for p, w_ in zip(picos, props["prominences"]):
        # Rango del HVN = ancho a media altura (FWHM) del pico suavizado.
        mitad = suav[p] / 2.0
        l = p
        while l > 0 and suav[l] > mitad:
            l -= 1
        r = p
        while r < n - 1 and suav[r] > mitad:
            r += 1
        hvn.append({"precio": float(centros[p]), "lo": float(edges[l]), "hi": float(edges[r + 1]),
                    "pct_volumen": float(suav[p] / total * 100.0), "prominencia": float(w_)})
    return {"poc": float(centros[i_poc]), "val": val, "vah": vah, "hvn": hvn,
            "bin": float(paso), "anios": anios, "atr_semanal": float(atr)}


# --- 2. Anchored VWAP ---------------------------------------------------------------------------
def anchored_vwap(df: pd.DataFrame, fecha_ancla) -> float:
    """AVWAP = sum(precio_tipico * volumen) / sum(volumen) desde ``fecha_ancla`` hasta hoy (Shannon)."""
    d = _normalizar(df)
    d = d[d.index >= pd.Timestamp(fecha_ancla)]
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    return float((tp * d["Volume"]).sum() / d["Volume"].sum())


def _minimo_ultima_correccion_mayor(d: pd.DataFrame):
    """Fecha del minimo de la ultima correccion >= 20 % (maximo -> minimo previo a nuevo maximo)."""
    pico = d["High"].cummax()
    dd = d["Low"] / pico - 1.0
    en_correccion = dd <= -CORRECCION_MAYOR_PCT
    if not en_correccion.any():
        return None
    # Episodio = tramo contiguo en que el precio no recupera el maximo previo.
    bajo_agua = d["High"] < pico
    grupo = (~bajo_agua).cumsum()
    ult_fin = en_correccion[en_correccion].index[-1]
    episodio = d[grupo == grupo.loc[ult_fin]]
    return episodio["Low"].idxmin()


def fecha_ultimo_10k(ticker_symbol: str, df: pd.DataFrame, info: dict | None = None):
    """(fecha, fuente). Fecha de presentacion del ultimo 10-K en SEC EDGAR.

    Fallback documentado si EDGAR falla: yfinance ``lastFiscalYearEnd`` + 60 dias (plazo maximo de
    presentacion del 10-K para large accelerated filers); si tampoco existe, el 31-dic previo + 60 dias.
    """
    try:
        from src.data.sec_edgar import _asegurar_identidad  # identidad SEC obligatoria
        from edgar import Company
        _asegurar_identidad()
        f = Company(ticker_symbol).get_filings(form="10-K").latest(1)
        f = f[0] if hasattr(f, "__getitem__") and not hasattr(f, "filing_date") else f
        return pd.Timestamp(f.filing_date), "SEC EDGAR 10-K filing_date"
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"EDGAR 10-K no disponible ({e}); usando fallback")
    fy = (info or {}).get("lastFiscalYearEnd")
    if fy:
        return pd.Timestamp(fy, unit="s") + pd.Timedelta(days=60), "FALLBACK yfinance lastFiscalYearEnd+60d"
    hoy = _normalizar(df).index[-1]
    dic31 = pd.Timestamp(year=hoy.year - (0 if hoy > pd.Timestamp(year=hoy.year, month=3, day=1) else 1),
                         month=12, day=31)
    return dic31 + pd.Timedelta(days=60), "FALLBACK 31-dic previo+60d"


def avwap_multiciclo(df: pd.DataFrame, ticker_symbol: str | None = None, info: dict | None = None) -> dict:
    """AVWAP anclado a: (1) minimo estructural de 2 anios, (2) minimo de la ultima correccion mayor
    (>=20 %), (3) fecha de presentacion del ultimo 10-K."""
    d = _normalizar(df)
    res = {}
    ventana = d[d.index >= d.index[-1] - pd.Timedelta(days=AVWAP_MIN_ESTRUCTURAL_DIAS)]
    f1 = ventana["Low"].idxmin()
    res["avwap_min_2a"] = {"fecha_ancla": f1, "avwap": anchored_vwap(d, f1), "fuente": "Minimo 2 anios (Low)"}
    f2 = _minimo_ultima_correccion_mayor(d)
    if f2 is not None:
        res["avwap_correccion"] = {"fecha_ancla": f2, "avwap": anchored_vwap(d, f2),
                                   "fuente": f"Minimo ultima correccion >= {CORRECCION_MAYOR_PCT:.0%}"}
    if ticker_symbol:
        f3, fuente = fecha_ultimo_10k(ticker_symbol, d, info)
        res["avwap_10k"] = {"fecha_ancla": f3, "avwap": anchored_vwap(d, f3), "fuente": fuente}
    return res


# --- 3. Order Blocks semanales no mitigados ------------------------------------------------------
def _importar_smc():
    # smartmoneyconcepts imprime un emoji al importar: puede romper consolas cp1252 en Windows.
    with contextlib.redirect_stdout(io.StringIO()):
        from smartmoneyconcepts import smc
    return smc


def order_blocks_semanales(df: pd.DataFrame, precio_actual: float | None = None,
                           swing_length: int = OB_SWING_SEMANAL) -> list[dict]:
    """OB alcistas semanales no mitigados cuyo Top esta por debajo del precio actual.

    API verificada (smartmoneyconcepts instalada): ``smc.swing_highs_lows(ohlc, swing_length)`` ->
    DataFrame[HighLow, Level]; ``smc.ob(ohlc, swing_hl, close_mitigation=False)`` ->
    DataFrame[OB, Top, Bottom, OBVolume, MitigatedIndex, Percentage]. Mitigado = el Low semanal
    perforo el Bottom (MitigatedIndex != 0). Se re-verifica de forma independiente contra los Low posteriores.
    """
    smc = _importar_smc()
    w = a_semanal(df)
    ohlc = w.rename(columns=str.lower)
    precio_actual = float(precio_actual if precio_actual is not None else w["Close"].iloc[-1])
    sw = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    ob = smc.ob(ohlc, sw, close_mitigation=False)
    out = []
    for i in np.flatnonzero((ob["OB"] == 1).values):
        top, bot = float(ob["Top"].iloc[i]), float(ob["Bottom"].iloc[i])
        if ob["MitigatedIndex"].iloc[i] != 0:
            continue  # smartmoneyconcepts lo marca mitigado
        if i + 1 < len(w) and w["Low"].iloc[i + 1:].min() < bot:
            continue  # verificacion independiente de mitigacion
        if top >= precio_actual:
            continue
        out.append({"fecha": w.index[i], "top": top, "bottom": bot,
                    "volumen": float(ob["OBVolume"].iloc[i]),
                    "fuerza": float(ob["Percentage"].iloc[i])})
    return sorted(out, key=lambda x: -x["top"])


# --- Orquestacion del modulo ---------------------------------------------------------------------
def calcular_anclas_liquidez(df: pd.DataFrame, ticker_symbol: str | None = None,
                             info: dict | None = None) -> dict:
    """Devuelve {'precio_actual', 'atr_semanal', 'volume_profile', 'avwap', 'order_blocks', 'anclas'}."""
    d = _normalizar(df)
    px = float(d["Close"].iloc[-1])
    atr = atr_estructural_semanal(d)
    vp = volume_profile(d, atr=atr)
    av = avwap_multiciclo(d, ticker_symbol, info)
    obs = order_blocks_semanales(d, px)

    anclas = [_ancla("POC", "Volume Profile POC", vp["poc"]),
              _ancla("VAL", "Volume Profile VAL", vp["val"]),
              _ancla("VAH", "Volume Profile VAH", vp["vah"])]
    for h in vp["hvn"]:
        anclas.append(_ancla("HVN", f"HVN {h['precio']:.2f} ({h['pct_volumen']:.2f}% vol)", h["lo"], h["hi"],
                             pct_volumen=h["pct_volumen"]))
    for k, v in av.items():
        anclas.append(_ancla("AVWAP", f"{k} (ancla {v['fecha_ancla']:%Y-%m-%d})", v["avwap"],
                             fecha_ancla=str(v["fecha_ancla"].date()), fuente=v["fuente"]))
    for o in obs:
        anclas.append(_ancla("OB", f"OB semanal {o['fecha']:%Y-%m-%d}", o["bottom"], o["top"],
                             fecha=str(o["fecha"].date()), fuerza=o["fuerza"]))
    return {"precio_actual": px, "atr_semanal": atr, "volume_profile": vp, "avwap": av,
            "order_blocks": obs, "anclas": anclas}
