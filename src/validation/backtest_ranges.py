"""Auditoria walk-forward de rangos de compra [r_inf, r_sup] (sin look-ahead).

Metodo
------
* En cada fecha de rebalanceo t (cada ``PASO_REBALANCEO`` ruedas) el rango se calcula SOLO con datos <= t
  (``df.iloc[:i+1]``), reutilizando las anclas de ``src.levels.volume_liquidity`` (Volume Profile, AVWAP,
  Order Blocks semanales) mas swing lows semanales macro y bandas de regresion logaritmica
  (``src.levels.valuation_floors.bandas_log_regresion``). Los pisos de valoracion basados en XBRL
  (P/E, EV/FCF, DCF) NO se usan aqui porque ``_facts`` devuelve el historico completo de hoy (look-ahead).
* Las anclas se ensanchan (puntuales: +-0.25 ATR semanal), se agrupan en clusters (ancho <= 3 ATR semanal) y
  el rango primario es el cluster con mayor confluencia de familias de ancla bajo el precio (<= 30 % debajo).
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
ANCHO_MAX_CLUSTER_ATR = 3.0
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
def swing_lows_semanales(df: pd.DataFrame, n: int = vl.OB_SWING_SEMANAL) -> list[float]:
    """Pivotes bajistas semanales confirmados: el Low es minimo de +-n semanas (confirmado tras n semanas)."""
    w = vl.a_semanal(df)
    lows = w["Low"].values
    out = []
    for i in range(n, len(lows) - n):
        if lows[i] == lows[i - n:i + n + 1].min():
            out.append(float(lows[i]))
    return out


def anclas_pit(df: pd.DataFrame) -> tuple[list[dict], float, float]:
    """(anclas, precio, atr_semanal) usando exclusivamente ``df`` (datos hasta t)."""
    d = vl._normalizar(df)
    px = float(d["Close"].iloc[-1])
    atr = vl.atr_estructural_semanal(d)
    vp = vl.volume_profile(d, atr=atr)
    puntual = ANCHO_PUNTUAL_ATR * atr
    anclas: list[dict] = []

    def add(tipo, lo, hi=None):
        hi = lo if hi is None else hi
        if hi == lo:
            lo, hi = lo - puntual, hi + puntual
        anclas.append({"tipo": tipo, "familia": FAMILIAS[tipo], "lo": float(lo), "hi": float(hi)})

    add("POC", vp["poc"])
    add("VAL", vp["val"])
    add("VAH", vp["vah"])
    for h in vp["hvn"]:
        add("HVN", h["lo"], h["hi"])
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


def clusters_pit(anclas: list[dict], px: float, atr: float) -> list[dict]:
    """Agrupa anclas bajo el precio en zonas de confluencia."""
    tope = px * (1 - DIST_MIN)
    piso = px * (1 - DIST_MAX)
    cand = [dict(a, hi=min(a["hi"], tope)) for a in anclas if a["hi"] > piso and a["lo"] < tope]
    cand.sort(key=lambda a: a["lo"])
    clusters, cur = [], None
    for a in cand:
        if cur and a["lo"] <= cur["hi"] and max(cur["hi"], a["hi"]) - cur["lo"] <= ANCHO_MAX_CLUSTER_ATR * atr:
            cur["hi"] = max(cur["hi"], a["hi"])
            cur["anclas"].append(a)
        else:
            if cur:
                clusters.append(cur)
            cur = {"lo": a["lo"], "hi": a["hi"], "anclas": [a]}
    if cur:
        clusters.append(cur)
    for c in clusters:
        c["lo"] = max(c["lo"], piso)
        fams = {a["familia"] for a in c["anclas"]}
        c.update(n_anclas=len(c["anclas"]), n_familias=len(fams), familias=sorted(fams),
                 tipos=sorted({a["tipo"] for a in c["anclas"]}), px=px,
                 dist_pct=(px - c["hi"]) / px, ancho_pct=(c["hi"] - c["lo"]) / px)
    return clusters


def rango_primario(clusters: list[dict]) -> dict | None:
    """Cluster de mayor confluencia (familias, anclas); empate -> el mas cercano al precio."""
    if not clusters:
        return None
    return sorted(clusters, key=lambda c: (-c["n_familias"], -c["n_anclas"], c["dist_pct"]))[0]


def calcular_zonas_pit(df: pd.DataFrame) -> dict:
    """API publica: zonas PIT con datos hasta la ultima fila de ``df``. Devuelve clusters y rango primario."""
    anclas, px, atr = anclas_pit(df)
    cl = clusters_pit(anclas, px, atr)
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
                    verbose: bool = True) -> pd.DataFrame:
    """Una fila por orden (llenada o no). Rangos estrictamente walk-forward."""
    df = _descargar(tk) if df is None else vl._normalizar(df)
    n = len(df)
    inicio = max(MIN_BARRAS_WARMUP, n - int(anios * 252))
    vol_sma_prev = df["Volume"].rolling(20).mean().shift(1).values
    filas = []
    t0 = time.time()
    for i0 in range(inicio, n - 1, paso):
        try:
            z = calcular_zonas_pit(df.iloc[:i0 + 1])  # <- unico dato visible: <= t
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
                    ancho_pct=p["ancho_pct"], sin_rango=False)
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
