"""Casos practicos 13F-HR (SEC EDGAR): contraste de los rangos PIT con las compras de grandes fondos.

Casos
-----
1. Berkshire Hathaway (CIK 1067983) en AAPL (Q1 2016 - Q1 2018) y OXY (2022).
2. Acumulacion institucional en caidas de calidad: META y MSFT (Q3-Q4 2022), opcional UNH y GOOGL (mismos
   trimestres). Se agrega el cambio de acciones de los gestores de referencia ``GESTORES_13F``
   (Vanguard y State Street son pasivos: se muestran pero NO cuentan para el flag de acumulacion, que usa
la variacion neta de gestores activos). Las acciones previas se ajustan por splits (p. ej. GOOGL 20:1, jul-2022).

Procedimiento Point-in-Time
---------------------------
Los 13F-HR se presentan hasta 45 dias despues del cierre del trimestre y NO traen fecha de ejecucion. Por tanto:
* La variacion de acciones trimestre contra trimestre (``SharesPrnAmount``) fija en que trimestres hubo compra neta.
* PROXY DE EJECUCION (limitacion explicita): el rango de precios del trimestre. Se usa el rango intercuartil
  ponderado por volumen del precio tipico diario ((H+L+C)/3) como zona probable de ejecucion, ademas del
  VWAP y del [Low, High] del trimestre.
* Los rangos del algoritmo se calculan con datos < inicio del trimestre (snapshot 0, estricto) y con
  snapshots al inicio de cada mes del trimestre (rodante), siempre con ``df`` truncado (sin look-ahead).
* Control: un rango aleatorio del mismo ancho, ubicado uniformemente entre -40 % y -0.5 % del precio de
  la fecha del snapshot, se solapa con el proxy con probabilidad ``p_control`` (Monte Carlo, semilla fija).

Sin red (EDGAR), el modulo degrada: evalua las ventanas documentadas con el proxy de precios y marca
``verificado_13f = False`` (no se afirma compra neta).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from src.levels import volume_liquidity as vl
from src.validation import backtest_ranges as br

CACHE_DIR = br.CACHE_DIR
RUTA_13F = os.path.join(CACHE_DIR, "posiciones_13f.json")
RUTA_CASOS = os.path.join(CACHE_DIR, "casos_resultados.json")

CUSIPS = {"AAPL": "037833100", "MSFT": "594918104", "META": "30303M102", "OXY": "674599105",
          "UNH": "91324P102", "GOOGL": "02079K305"}

# Gestores activos adicionales (CIK verificados contra EDGAR). Se anaden a GESTORES_13F solo aqui (dominio de validacion).
GESTORES_EXTRA = {"Pershing Square": 1336528, "Tiger Global": 1167483, "Coatue": 1135730, "Third Point": 1040273,
                  "Appaloosa": 1656456, "Soros Fund Mgmt": 1029160, "Viking Global": 1103804,
                  "Lone Pine": 1061165, "Baupost": 1061768}
PASIVOS = {"Vanguard Group", "State Street"}  # flujos de indice: no se cuentan como conviccion


def _gestores_all() -> dict:
    from src.data.sec_edgar import GESTORES_13F
    return {**GESTORES_13F, **GESTORES_EXTRA}


def _factor_split(tk: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> float:
    """Producto de splits con fecha en (desde, hasta]; ajusta acciones del trimestre previo a base actual."""
    try:
        import yfinance as yf
        sp = yf.Ticker(tk).splits
        sp.index = sp.index.tz_localize(None)
        return float(sp[(sp.index > desde) & (sp.index <= hasta)].prod()) if len(sp) else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


CASOS = [
    {"id": "BRK_AAPL", "titulo": "Berkshire Hathaway en AAPL (Q1 2016 - Q1 2018)", "ticker": "AAPL",
     "gestores": ["Berkshire Hathaway"], "trimestres": ["2016Q1", "2016Q2", "2016Q3", "2016Q4", "2017Q1",
                                                       "2017Q2", "2017Q3", "2017Q4", "2018Q1"]},
    {"id": "BRK_OXY", "titulo": "Berkshire Hathaway en OXY (2022)", "ticker": "OXY",
     "gestores": ["Berkshire Hathaway"], "trimestres": ["2022Q1", "2022Q2", "2022Q3", "2022Q4"]},
    {"id": "INST_META", "titulo": "Acumulacion institucional en META (Q3-Q4 2022)", "ticker": "META",
     "gestores": "TODOS", "trimestres": ["2022Q3", "2022Q4"]},
    {"id": "INST_MSFT", "titulo": "Acumulacion institucional en MSFT (Q3-Q4 2022)", "ticker": "MSFT",
     "gestores": "TODOS", "trimestres": ["2022Q3", "2022Q4"]},
    {"id": "INST_UNH", "titulo": "Acumulacion institucional en UNH (Q3-Q4 2022, opcional)", "ticker": "UNH",
     "gestores": "TODOS", "trimestres": ["2022Q3", "2022Q4"]},
    {"id": "INST_GOOGL", "titulo": "Acumulacion institucional en GOOGL (Q3-Q4 2022, opcional)", "ticker": "GOOGL",
     "gestores": "TODOS", "trimestres": ["2022Q3", "2022Q4"]},
]


def _prev(q: str) -> str:
    return str(pd.Period(q, "Q") - 1)


def _fin(q: str) -> pd.Timestamp:
    return pd.Period(q, "Q").end_time.normalize()


def _ini(q: str) -> pd.Timestamp:
    return pd.Period(q, "Q").start_time.normalize()


# --- 13F desde EDGAR (con cache y degradacion) -----------------------------------------------------------
def _posiciones_filing(obj, tickers) -> dict:
    it = obj.infotable
    out = {}
    if it is None or len(it) == 0:
        return {t: {"acciones": 0.0, "valor": 0.0} for t in tickers}
    for t in tickers:
        m = pd.Series(False, index=it.index)
        if "Ticker" in it.columns:
            m |= it["Ticker"].astype(str).str.upper() == t
        if "Cusip" in it.columns:
            m |= it["Cusip"].astype(str).str.upper() == CUSIPS[t]
        sel = it[m]
        out[t] = {"acciones": float(sel["SharesPrnAmount"].sum()), "valor": float(sel["Value"].sum())}
    return out


def descargar_13f(gestores: dict, periodos_por_gestor: dict, tickers, verbose=True) -> dict:
    """{gestor: {periodo: {ticker: {acciones, valor}} | None}}. Cache JSON; None = no disponible."""
    cache = json.load(open(RUTA_13F, encoding="utf-8")) if os.path.exists(RUTA_13F) else {}
    try:
        from src.data.sec_edgar import _asegurar_identidad
        from edgar import Company
        _asegurar_identidad()
    except Exception as e:  # noqa: BLE001
        print(f"  EDGAR no importable ({e}); degradando")
        return cache
    for nombre, periodos in periodos_por_gestor.items():
        c = cache.setdefault(nombre, {})
        pendientes = [p for p in sorted(periodos) if not (p in c and c[p] is not None and
                                                          all(t in c[p] for t in tickers))]
        if not pendientes:
            continue
        lista = None
        for intento in range(3):
            try:
                ini = (_ini(pendientes[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                fin = (_fin(pendientes[-1]) + pd.Timedelta(days=200)).strftime("%Y-%m-%d")
                lista = list(Company(gestores[nombre]).get_filings(form="13F-HR", filing_date=f"{ini}:{fin}"))
                break
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2)
        if lista is None:
            print(f"  {nombre}: EDGAR no disponible ({err}); periodos sin datos")
            for p in pendientes:
                c.setdefault(p, None)
            continue
        por_periodo = {}
        for f in lista:
            por_periodo.setdefault(str(pd.Period(pd.Timestamp(f.period_of_report), "Q")), f)  # 1ra (mas reciente)
        for p in pendientes:
            f = por_periodo.get(p)
            if f is None:
                c[p] = None
                if verbose:
                    print(f"  {nombre} {p}: sin 13F-HR en EDGAR")
                continue
            try:
                t0 = time.time()
                c[p] = _posiciones_filing(f.obj(), tickers)
                c[p]["_filing_date"] = str(f.filing_date)
                if verbose:
                    print(f"  {nombre} {p}: 13F-HR presentado {f.filing_date} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                c[p] = None
                print(f"  {nombre} {p}: error parseando 13F ({e})")
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(cache, open(RUTA_13F, "w", encoding="utf-8"), indent=1)
    return cache


def variacion(cache: dict, gestores: list[str], ticker: str, q: str) -> dict | None:
    """Cambio agregado de acciones en q, con acciones previas ajustadas por splits (None si no hay datos)."""
    det, tot, act = [], 0.0, 0.0
    fs = _factor_split(ticker, _fin(_prev(q)), _fin(q))
    for g in gestores:
        a, b = cache.get(g, {}).get(q), cache.get(g, {}).get(_prev(q))
        if a is None or b is None or ticker not in a or ticker not in b:
            continue
        prev = b[ticker]["acciones"] * fs
        d = a[ticker]["acciones"] - prev
        if a[ticker]["acciones"] or prev:
            det.append({"gestor": g, "acciones_prev": prev, "acciones": a[ticker]["acciones"], "delta": d,
                        "valor": a[ticker]["valor"], "presentado": a.get("_filing_date"), "pasivo": g in PASIVOS})
            tot += d
            act += 0.0 if g in PASIVOS else d
    if not det and not any(cache.get(g, {}).get(q) is not None for g in gestores):
        return None
    return {"delta_total": tot, "delta_activos": act, "factor_split": fs, "detalle": det}


# --- Proxy de ejecucion y PIT ----------------------------------------------------------------------------
def _cuantil_pond(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)])


def proxy_trimestre(df: pd.DataFrame, q: str) -> dict | None:
    d = df[(df.index >= _ini(q)) & (df.index <= _fin(q) + pd.Timedelta(days=1))]
    if len(d) < 20:
        return None
    tp = ((d["High"] + d["Low"] + d["Close"]) / 3).values
    w = d["Volume"].values
    return {"low": float(d["Low"].min()), "high": float(d["High"].max()),
            "vwap": float((tp * w).sum() / w.sum()), "p25": _cuantil_pond(tp, w, 0.25),
            "p75": _cuantil_pond(tp, w, 0.75)}


def _solapa(lo1, hi1, lo2, hi2) -> bool:
    return lo1 <= hi2 and lo2 <= hi1


def _p_control(px, ancho, p25, p75, n=4000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    techo, piso = px * (1 - br.DIST_MIN), px * 0.60
    hi = rng.uniform(piso + ancho, max(techo, piso + ancho + 1e-9), n)
    return float(np.mean((hi - ancho <= p75) & (hi >= p25)))


def evaluar_trimestre(df: pd.DataFrame, q: str) -> dict | None:
    pr = proxy_trimestre(df, q)
    if pr is None:
        return None
    snaps = []
    for m in range(3):
        ini_m = _ini(q) + pd.DateOffset(months=m)
        sub = df[df.index < ini_m]
        if len(sub) < br.MIN_BARRAS_WARMUP:
            continue
        z = br.calcular_zonas_pit(sub)
        p = z["primario"]
        if p is None:
            snaps.append({"mes": m, "primario": None})
            continue
        sn = {"mes": m, "px": z["px"], "r_inf": p["lo"], "r_sup": p["hi"], "tipos": p["tipos"],
              "n_familias": p["n_familias"], "n_clusters": len(z["clusters"]),
              "toque": bool(pr["low"] <= p["hi"]),
              "solapa_proxy": _solapa(p["lo"], p["hi"], pr["p25"], pr["p75"]),
              "solapa_rango_total": _solapa(p["lo"], p["hi"], pr["low"], pr["high"]),
              "solapa_algun_cluster": any(_solapa(c["lo"], c["hi"], pr["p25"], pr["p75"]) for c in z["clusters"]),
              "dist_vwap_pct": float(min(abs((c["lo"] + c["hi"]) / 2 - pr["vwap"]) for c in z["clusters"]) / pr["vwap"])
              if z["clusters"] else None,
              "p_control": _p_control(z["px"], p["hi"] - p["lo"], pr["p25"], pr["p75"])}
        snaps.append(sn)
    # retorno forward desde el VWAP del trimestre
    idx_fin = df.index.searchsorted(_fin(q) + pd.Timedelta(days=1))
    ret252 = None
    if idx_fin + 252 < len(df):
        ret252 = float(df["Close"].iloc[idx_fin + 252] / pr["vwap"] - 1)
    return {"trimestre": q, "proxy": pr, "snapshots": snaps, "ret252_desde_vwap": ret252}


# --- Orquestacion -----------------------------------------------------------------------------------------
def ejecutar_casos(casos=CASOS, verbose=True) -> dict:
    GESTORES_13F = _gestores_all()
    gest_todos = list(GESTORES_13F)
    periodos_por_gestor: dict[str, set] = {}
    tickers = sorted({c["ticker"] for c in casos})
    for c in casos:
        gs = gest_todos if c["gestores"] == "TODOS" else c["gestores"]
        for g in gs:
            s = periodos_por_gestor.setdefault(g, set())
            for q in c["trimestres"]:
                s.update([q, _prev(q)])
    if verbose:
        print("Descargando 13F-HR desde SEC EDGAR (con cache)...")
    cache = descargar_13f(GESTORES_13F, periodos_por_gestor, tickers, verbose)
    resultados = {}
    for c in casos:
        tk = c["ticker"]
        try:
            df = br._descargar(tk)
        except Exception as e:  # noqa: BLE001
            print(f"  {c['id']}: sin datos de mercado ({e})")
            continue
        gs = gest_todos if c["gestores"] == "TODOS" else c["gestores"]
        filas = []
        for q in c["trimestres"]:
            ev = evaluar_trimestre(df, q)
            if ev is None:
                continue
            v = variacion(cache, gs, tk, q)
            ev["verificado_13f"] = v is not None
            ev["delta_acciones"] = None if v is None else v["delta_total"]
            ev["detalle_13f"] = None if v is None else v["detalle"]
            ev["delta_activos"] = None if v is None else v["delta_activos"]
            ev["acumulacion"] = None if v is None else bool(v["delta_activos"] > 0)
            filas.append(ev)
        resultados[c["id"]] = {"titulo": c["titulo"], "ticker": tk, "gestores": gs, "trimestres": filas}
    json.dump(resultados, open(RUTA_CASOS, "w", encoding="utf-8"), indent=1, default=str)
    return resultados


def agregado(resultados: dict) -> dict:
    """Tasas sobre trimestres de acumulacion confirmada (o documentados si no hay 13F)."""
    filas = []
    for cid, r in resultados.items():
        for t in r["trimestres"]:
            if t["acumulacion"] is False:
                continue  # trimestre sin compra neta verificada
            for s in t["snapshots"]:
                if s.get("r_inf") is not None:
                    filas.append({"caso": cid, "q": t["trimestre"], "verif": t["verificado_13f"], **s})
    if not filas:
        return {"n": 0}
    d = pd.DataFrame(filas)
    s0 = d[d["mes"] == 0]
    q_roll = d.groupby(["caso", "q"]).agg(roll=("solapa_proxy", "any"), toque=("toque", "any"),
                                          ctrl=("p_control", lambda x: 1 - np.prod(1 - x)))
    return {"n_trimestres": int(len(s0)), "n_verificados_13f": int(s0["verif"].sum()),
            "solapa_estricto": float(s0["solapa_proxy"].mean()),
            "solapa_rodante": float(q_roll["roll"].mean()),
            "toque_estricto": float(s0["toque"].mean()),
            "solapa_algun_cluster": float(s0["solapa_algun_cluster"].mean()),
            "p_control_medio": float(s0["p_control"].mean()),
            "p_control_rodante": float(q_roll["ctrl"].mean()),
            "dist_vwap_mediana": float(s0["dist_vwap_pct"].dropna().median())}


def imprimir(resultados: dict):
    print("\n=== CASOS 13F-HR: rangos PIT vs. zona de ejecucion (proxy = IQR ponderado por volumen del trimestre) ===")
    print("NOTA: 13F no trae fecha de ejecucion; la 'ejecucion' es el proxy de precio del trimestre (ver docstring).")
    for cid, r in resultados.items():
        print(f"\n--- {r['titulo']} ---")
        print(f"{'Q':7}{'dAcc activos':>14}{'acum':>6}{'proxy P25-P75':>17}{'VWAP':>8}{'[Low-High]':>16}"
              f"{'rango PIT(Q-ini)':>20}{'toque':>6}{'solapa':>7}{'rodante':>8}{'ctrl':>6}{'ret252':>8}")
        for t in r["trimestres"]:
            pr, s0 = t["proxy"], t["snapshots"][0] if t["snapshots"] else {}
            da = "n/d" if t["delta_activos"] is None else f"{t['delta_activos'] / 1e6:+.2f}M"
            ac = {True: "SI", False: "no", None: "?"}[t["acumulacion"]]
            rod = any(s.get("solapa_proxy") for s in t["snapshots"])
            rango = f"{s0['r_inf']:.1f}-{s0['r_sup']:.1f}" if s0.get("r_inf") else "sin rango"
            r252 = "n/d" if t["ret252_desde_vwap"] is None else f"{t['ret252_desde_vwap'] * 100:+.0f}%"
            print(f"{t['trimestre']:7}{da:>14}{ac:>6}{pr['p25']:>8.1f}-{pr['p75']:<8.1f}{pr['vwap']:>8.1f}"
                  f"{pr['low']:>8.1f}-{pr['high']:<7.1f}{rango:>20}"
                  f"{'SI' if s0.get('toque') else 'no':>6}{'SI' if s0.get('solapa_proxy') else 'no':>7}"
                  f"{'SI' if rod else 'no':>8}{s0.get('p_control', float('nan')) * 100:>5.0f}%{r252:>8}")
            for dt in (t["detalle_13f"] or []):
                if dt["acciones"] or dt["acciones_prev"]:
                    print(f"         {dt['gestor']:<26} {dt['acciones_prev'] / 1e6:>9.2f}M -> {dt['acciones'] / 1e6:>9.2f}M "
                          f"(presentado {dt['presentado']})")
    ag = agregado(resultados)
    print("\n--- Agregado (trimestres con compra neta 13F; si EDGAR no disponible, ventanas documentadas) ---")
    for k, v in ag.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return ag


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    res = ejecutar_casos()
    if not res:
        print("Sin resultados.")
        return
    imprimir(res)


if __name__ == "__main__":
    main()
