"""CLI unificado: niveles de compra institucionales (data + levels + validation + reporting).

Modo individual:  python main.py --ticker NVDA --capital 1000
Modo batch:       python main.py --watchlist watchlist.csv --top 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

MAX_ZONAS = 4
MAX_TRAMOS_CAPITAL = 3


# --- CLI ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Screener de niveles de compra institucionales")
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--ticker", type=str, help="Modo individual: simbolo (p. ej. NVDA)")
    modo.add_argument("--watchlist", type=str, help="Modo batch: CSV con columna 'ticker'")
    p.add_argument("--capital", type=float, default=1000.0, help="Capital (USD) a distribuir en las zonas (def. 1000)")
    p.add_argument("--top", type=int, default=10, help="Batch: cuantos activos mostrar/exportar (def. 10)")
    p.add_argument("--periodo", type=str, default="10y", help="Historial yfinance (def. 10y)")
    p.add_argument("--anios-backtest", type=int, default=5, help="Anios del backtest walk-forward (def. 5)")
    p.add_argument("--sin-backtest", action="store_true", help="Omite el backtest (mucho mas rapido)")
    p.add_argument("--sin-sec", action="store_true", help="Omite Form 4 / 13F / STOCK Act (sin red SEC)")
    p.add_argument("--sin-grafico", action="store_true", help="No genera graficos")
    p.add_argument("--graficos-batch", action="store_true", help="Batch: tambien grafica el top")
    p.add_argument("--mostrar", action="store_true", help="Muestra el grafico en pantalla (plt.show)")
    p.add_argument("--salida", type=str, default=None, help="Directorio de salida (def. ./resultados o /content/resultados)")
    p.add_argument("--formato", choices=["csv", "json", "both"], default="both", help="Formato de exportacion")

    return p


def _positivo(args, parser):
    if args.capital <= 0:
        parser.error("--capital debe ser > 0")
    if args.top < 1:
        parser.error("--top debe ser >= 1")
    if args.anios_backtest < 1:
        parser.error("--anios-backtest debe ser >= 1")


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    _positivo(args, parser)
    if args.ticker is not None:
        args.ticker = args.ticker.strip().upper()
        if not args.ticker:
            parser.error("--ticker vacio")
    return args


def leer_watchlist(ruta: str) -> list[str]:
    """Tickers unicos (mayusculas) de un CSV: columna 'ticker' si existe, si no la primera. Ignora '#'."""
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        filas = [r for r in csv.reader(f) if r and r[0].strip() and not r[0].strip().startswith("#")]
    if not filas:
        return []
    cab = [c.strip().lower() for c in filas[0]]
    if "ticker" in cab:
        i, filas = cab.index("ticker"), filas[1:]
    else:
        i = 0
    vistos, out = set(), []
    for r in filas:
        if i < len(r):
            t = r[i].strip().upper()
            if t and t not in vistos:
                vistos.add(t)
                out.append(t)
    return out


# --- Analisis ----------------------------------------------------------------------------------------
def asignar_capital(zonas: list[dict], capital: float, max_tramos: int = MAX_TRAMOS_CAPITAL) -> list[dict]:
    """Reparte el capital entre las mejores zonas, ponderado por familias de confluencia. Limite en r_sup."""
    for z in zonas:
        z["capital_asignado"], z["acciones"] = 0.0, 0.0
    top = zonas[:max_tramos]
    peso = sum(z["n_familias"] for z in top)
    if not top or peso <= 0:
        return zonas
    for z in top:
        z["capital_asignado"] = float(capital * z["n_familias"] / peso)
        z["acciones"] = float(z["capital_asignado"] / z["r_sup"]) if z["r_sup"] > 0 else 0.0
    return zonas


def puntaje_ranking(confluencia: float | None, confiabilidad: float | None) -> float | None:
    """Promedio de confluencia actual y confiabilidad institucional (0-100); usa la disponible si falta una."""
    vals = [v for v in (confluencia, confiabilidad) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def ordenar_ranking(resultados: list[dict]) -> list[dict]:
    """Descendente por score_ranking; los sin score al final (orden estable)."""
    return sorted(resultados, key=lambda r: (r.get("score_ranking") is None, -(r.get("score_ranking") or 0.0)))


def analizar_ticker(ticker: str, capital: float, periodo: str = "10y", anios_backtest: int = 5,
                    backtest: bool = True, sec: bool = True, verbose: bool = True) -> dict:
    """Pipeline completo de un ticker. Nunca lanza: los fallos de cada etapa quedan en ``errores``."""
    res = {"ticker": ticker, "fecha": datetime.now().strftime("%Y-%m-%d"), "capital": capital,
           "precio_actual": None, "zonas": [], "errores": [], "df": None, "form4": None, "r13f": None}

    def err(etapa, e):
        res["errores"].append(f"{etapa}: {type(e).__name__}: {e}")
        if verbose:
            print(f"  [!] {ticker} {etapa}: {e}")

    def log(msg):
        if verbose:
            print(f"  {ticker}: {msg}", flush=True)

    # 1. Datos de mercado
    try:
        from src.data.market_data import descargar_historial, obtener_info, verificar_proximidad_earnings
        log("descargando historial")
        tk_obj, df = descargar_historial(ticker, periodo)
    except Exception as e:  # noqa: BLE001
        err("mercado", e)
        return res
    if df is None or df.empty:
        res["errores"].append("mercado: sin datos")
        return res
    res["df"] = df
    res["precio_actual"] = float(df["Close"].iloc[-1])
    info = obtener_info(tk_obj)
    try:
        res["earnings"] = verificar_proximidad_earnings(tk_obj)
    except Exception as e:  # noqa: BLE001
        err("earnings", e)

    # 2. Zonas (clusters de confluencia, solo datos hasta hoy)
    zonas_raw = {}
    try:
        from src.validation.backtest_ranges import calcular_zonas_pit
        zonas_raw = calcular_zonas_pit(df)
        res["atr_semanal"] = zonas_raw["atr_semanal"]
        cl = sorted(zonas_raw["clusters"], key=lambda c: (-c["n_familias"], -c["n_anclas"], c["dist_pct"]))
        for i, c in enumerate(cl[:MAX_ZONAS], 1):
            res["zonas"].append({"nivel": i, "r_inf": c["lo"], "r_sup": c["hi"], "p_centro": (c["lo"] + c["hi"]) / 2,
                                 "dist_pct": -c["dist_pct"] * 100.0, "n_familias": c["n_familias"],
                                 "n_anclas": c["n_anclas"], "familias": c["familias"], "tipos": c["tipos"],
                                 "fuente": "+".join(c["tipos"])})
    except Exception as e:  # noqa: BLE001
        err("zonas", e)
    prim = zonas_raw.get("primario")

    # 3. Pisos de valoracion (P/E propio, EV/FCF, DCF, log-reg) -> numero de anclas de valoracion en la zona
    n_val = 0
    try:
        from src.levels.valuation_floors import calcular_pisos_valoracion
        val = calcular_pisos_valoracion(ticker, tk_obj, df, info)
        res["valoracion"] = {k: v for k, v in val.items() if k != "anclas"}
        if prim:
            lo, hi = prim["lo"] * 0.98, prim["hi"] * 1.02
            n_val = sum(1 for a in val["anclas"] if lo <= a["precio"] <= hi)
    except Exception as e:  # noqa: BLE001
        err("valoracion", e)
    res["n_valoracion_en_zona"] = n_val

    # 4. SEC: Form 4 (P), 13F, STOCK Act
    respaldo, stock = None, None
    if sec:
        log("consultando SEC EDGAR / STOCK Act")
        try:
            from src.data.sec_edgar import compras_insiders
            f4, ins = compras_insiders(ticker)
            res["form4"], res["insiders"] = f4, ins
        except Exception as e:  # noqa: BLE001
            err("form4", e)
        try:
            from src.data.sec_edgar import obtener_13f_ticker
            r13 = obtener_13f_ticker(ticker)
            res["r13f"] = r13
            ch = pd.to_numeric(r13["cambio_pct"], errors="coerce") if len(r13) else pd.Series(dtype=float)
            nuevas = (pd.to_numeric(r13["acciones_previas"], errors="coerce").fillna(0) <= 0) if len(r13) else ch
            respaldo = {"n_con_posicion": int(len(r13)),
                        "n_acumulando": int(((ch > 0) | nuevas).sum()) if len(r13) else 0,
                        "compras_form4_usd": float((res.get("insiders") or {}).get("valor_usd", 0.0))}
            res["respaldo13f"] = respaldo
        except Exception as e:  # noqa: BLE001
            err("13f", e)
        try:
            from src.data.stock_act import consultar_flujo_congreso_stock_act
            stock = consultar_flujo_congreso_stock_act(ticker)
            res["stock_act"] = {k: v for k, v in stock.items() if k != "ultimas_tx"}
        except Exception as e:  # noqa: BLE001
            err("stock_act", e)

    # 5. Backtest walk-forward del ticker
    bt = {}
    if backtest and prim:
        try:
            from src.validation import backtest_ranges as br
            log(f"backtest walk-forward {anios_backtest}a (puede tardar)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ordenes = br.backtest_ticker(ticker, anios_backtest, df=df, verbose=False)
            bt = br.resumen(ordenes) if len(ordenes) else {}
            res["backtest_resumen"] = bt
        except Exception as e:  # noqa: BLE001
            err("backtest", e)

    # 6. Scoring
    try:
        from src.validation.scoring import s_confluencia, indice_desde_modulos
        n_fam = prim["n_familias"] if prim else None
        res["confluencia_score"] = s_confluencia(n_fam, n_val)
        res["confiabilidad"] = indice_desde_modulos(bt, respaldo, stock, None, n_fam, n_val)
        res["score_ranking"] = puntaje_ranking(res["confluencia_score"], res["confiabilidad"].get("indice"))
    except Exception as e:  # noqa: BLE001
        err("scoring", e)

    asignar_capital(res["zonas"], capital)
    return res


# --- Salida ------------------------------------------------------------------------------------------
def _fmt(v, f="{:.1f}", na="n/d"):
    return na if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def imprimir_individual(r: dict):
    print(f"\n=== {r['ticker']} | Precio ${_fmt(r['precio_actual'], '{:.2f}')} | Capital ${r['capital']:,.0f} ===")
    if not r["zonas"]:
        print("  Sin zonas de confluencia bajo el precio.")
    for z in r["zonas"]:
        print(f"  N{z['nivel']}: ${z['r_inf']:.2f} - ${z['r_sup']:.2f} ({z['dist_pct']:+.1f}%) | "
              f"{z['n_familias']} familias [{', '.join(z['familias'])}] | "
              f"${z['capital_asignado']:.0f} -> {z['acciones']:.3f} acc. @ ${z['r_sup']:.2f}")
    c = r.get("confiabilidad") or {}
    print(f"  Confluencia: {_fmt(r.get('confluencia_score'))}/100 | Confiabilidad institucional: "
          f"{_fmt(c.get('indice'))}/100 (cobertura {_fmt((c.get('cobertura') or 0) * 100, '{:.0f}')}%) | "
          f"Ranking: {_fmt(r.get('score_ranking'))}")
    if r["errores"]:
        print(f"  Advertencias: {len(r['errores'])} etapa(s) con fallo (ver export).")


def imprimir_ranking(rs: list[dict]):
    print(f"\n{'#':>2} {'Ticker':<7}{'Precio':>10}{'Zona 1':>20}{'Fam':>5}{'Confl.':>8}{'Confiab.':>9}{'Score':>7}")
    for i, r in enumerate(rs, 1):
        z = r["zonas"][0] if r["zonas"] else None
        zona = f"{z['r_inf']:.2f}-{z['r_sup']:.2f}" if z else "n/d"
        conf = (r.get("confiabilidad") or {}).get("indice")
        print(f"{i:>2} {r['ticker']:<7}{_fmt(r['precio_actual'], '{:.2f}'):>10}{zona:>20}"
              f"{(z['n_familias'] if z else 0):>5}{_fmt(r.get('confluencia_score')):>8}{_fmt(conf):>9}"
              f"{_fmt(r.get('score_ranking')):>7}")


def _graficar(r: dict, directorio: str, mostrar: bool):
    from src.reporting.charts import graficar_niveles
    from src.reporting.exporter import directorio_salida
    out = directorio_salida(directorio)
    ruta = os.path.join(out, f"{r['ticker']}_niveles.png")
    c = r.get("confiabilidad") or {}
    sub = f"Confluencia {_fmt(r.get('confluencia_score'))}/100 | Confiabilidad {_fmt(c.get('indice'))}/100"
    fig = graficar_niveles(r.get("df"), r["ticker"], r["zonas"], r.get("precio_actual"), r.get("form4"),
                           r.get("r13f"), sub, ruta, mostrar)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [ok] Grafico: {ruta}")
    return ruta


def _liviano(r: dict) -> dict:
    """Resultado sin objetos pesados (df de precios, Form 4 crudo) para exportar."""
    out = {k: v for k, v in r.items() if k not in ("df", "form4", "r13f")}
    out["form4_compras"] = None if r.get("form4") is None else r["form4"]
    out["respaldo13f_detalle"] = r.get("r13f")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    from src.reporting.exporter import exportar_resultados
    kw = dict(periodo=args.periodo, anios_backtest=args.anios_backtest, backtest=not args.sin_backtest,
              sec=not args.sin_sec)
    if args.ticker:
        r = analizar_ticker(args.ticker, args.capital, **kw)
        imprimir_individual(r)
        if not args.sin_grafico:
            try:
                _graficar(r, args.salida, args.mostrar)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] Grafico no generado: {e}")
        rutas = exportar_resultados([_liviano(r)], f"{r['ticker']}_niveles", args.salida, args.formato)
        print("  Exportado:", ", ".join(rutas.values()))
        return 0 if r["precio_actual"] is not None else 1

    try:
        tickers = leer_watchlist(args.watchlist)
    except OSError as e:
        print(f"No se pudo leer la watchlist: {e}", file=sys.stderr)
        return 2
    if not tickers:
        print("La watchlist esta vacia.", file=sys.stderr)
        return 2
    print(f"Escaneando {len(tickers)} activos...")
    resultados = []
    for t in tickers:
        resultados.append(analizar_ticker(t, args.capital, **kw))
    top = ordenar_ranking([r for r in resultados if r["precio_actual"] is not None])[:args.top]
    imprimir_ranking(top)
    fallidos = [r["ticker"] for r in resultados if r["precio_actual"] is None]
    if fallidos:
        print(f"\nSin datos: {', '.join(fallidos)}")
    if args.graficos_batch and not args.sin_grafico:
        for r in top:
            try:
                _graficar(r, args.salida, False)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] Grafico {r['ticker']}: {e}")
    rutas = exportar_resultados([_liviano(r) for r in top], "ranking_watchlist", args.salida, args.formato)
    print("\nExportado:", ", ".join(rutas.values()))
    return 0 if top else 1


if __name__ == "__main__":
    sys.exit(main())
