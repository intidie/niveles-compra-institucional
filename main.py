"""CLI unificado: niveles de compra institucionales (data + levels + validation + reporting).

Modo individual:  python main.py --ticker NVDA --capital 1000
Modo batch:       python main.py --watchlist watchlist.csv --top 10

Semantica de las cifras (consola, export y UI usan los mismos textos):

* ``--capital`` es POR TICKER: en modo batch cada activo recibe el capital completo, no se reparte entre tickers.
* ``dist_pct`` de cada zona es la distancia % de ``r_sup`` (el limite GTC) al precio actual; negativo = por debajo.
* Backtest (Fill / Bounce / MAE): solo del rango primario (N1), de ``anios_backtest`` anios (def. 5); no es comparable
  con el pool de docs/METODOLOGIA.md y con menos de ``MIN_FILLS_AVISO`` fills se avisa "muestra pequena".
* Un dato NO disponible (Form 4 / 13F / STOCK Act sin respuesta) se excluye de la confiabilidad y baja la cobertura;
  no cuenta como 0 ni como neutro. El 13F cuenta solo gestores activos (los pasivos de indice se ignoran).
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
MIN_FILLS_AVISO = 10  # menos ejecuciones simuladas que esto => las metricas del backtest son ruido (se avisa)

# Alcance de las cifras, un solo texto para consola / export / UI.
AYUDA_CAPITAL = ("USD a distribuir en las zonas: capital por ticker; en modo --watchlist no se reparte entre tickers "
                 "(cada activo recibe el capital completo) (def. 1000)")
NOTA_CAPITAL_BATCH = ("Capital por ticker: cada activo recibe el capital completo; no se reparte entre tickers "
                      "(si se llenaran las ordenes de N activos harian falta N veces el capital).")
NOMBRES_COMPONENTES = {"respaldo_13f_form4": "13F/Form 4", "stock_act": "STOCK Act", "absorcion": "absorcion",
                       "backtest": "backtest", "confluencia": "confluencia"}


# --- CLI ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Screener de niveles de compra institucionales")
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--ticker", type=str, help="Modo individual: simbolo (p. ej. NVDA)")
    modo.add_argument("--watchlist", type=str, help="Modo batch: CSV con columna 'ticker'")
    p.add_argument("--capital", type=float, default=1000.0, help=AYUDA_CAPITAL)
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


def peso_confluencia_ranking(pesos: dict | None = None) -> float:
    """Peso (0-1) de la confluencia en el score de ranking con cobertura completa.

    ranking = (confluencia + confiabilidad) / 2 y la confluencia es ademas un componente de la confiabilidad:
    ``0.5 + 0.5 * peso_confluencia`` (~55 % con los pesos calibrados; mas si faltan otros componentes).
    """
    from src.validation import scoring
    pesos = pesos or scoring.cargar_pesos()
    return 0.5 + 0.5 * float(pesos.get("confluencia", 0.0))


def alcance_backtest(anios: int | None) -> str:
    """Texto estandar del alcance de las metricas del backtest (consola, export y UI)."""
    return (f"rango primario (N1), backtest de {anios if anios is not None else 'n/d'} años; "
            f"no comparable con el pool de docs/METODOLOGIA.md")


def aviso_muestra_pequena(n_fills: int | None) -> str | None:
    """Aviso si hay menos de ``MIN_FILLS_AVISO`` ejecuciones simuladas (None si la muestra alcanza o no hay backtest)."""
    if n_fills is None or n_fills >= MIN_FILLS_AVISO:
        return None
    return (f"muestra pequeña: {n_fills} fills (< {MIN_FILLS_AVISO}); Bounce, Fill rate y MAE del backtest "
            f"no son concluyentes")


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Columna booleana tolerante (NaN/None -> False); todo False si la columna no existe."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].map(lambda v: bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False)


def info_consulta_13f(r13) -> tuple[list[str], int | None]:
    """(gestores fallidos, gestores consultados) que ``obtener_13f_ticker`` deja en ``df.attrs`` (contrato)."""
    attrs = getattr(r13, "attrs", None) or {}
    fallidos = [str(g) for g in (attrs.get("fallidos") or [])]
    try:
        consultados = None if attrs.get("consultados") is None else int(attrs["consultados"])
    except (TypeError, ValueError):
        consultados = None
    return fallidos, consultados


def respondieron_activos(r13) -> int | None:
    """``attrs['respondieron_activos']`` si ``obtener_13f_ticker`` lo informa (None si no existe: contrato antiguo)."""
    attrs = getattr(r13, "attrs", None) or {}
    try:
        return None if attrs.get("respondieron_activos") is None else int(attrs["respondieron_activos"])
    except (TypeError, ValueError):
        return None


def _gestores_pasivos() -> set[str]:
    try:
        from src.data.sec_edgar import GESTORES_PASIVOS
        return set(GESTORES_PASIVOS)
    except Exception:  # noqa: BLE001
        return set()


def avisos_13f(r13) -> list[str]:
    """Avisos de la consulta 13F: gestores que no respondieron y, si procede, que ningun activo respondio."""
    fallidos, consultados = info_consulta_13f(r13)
    out = []
    if fallidos:
        msg = "13f: no respondieron: " + ", ".join(fallidos)
        if consultados is not None and len(fallidos) >= consultados > 0:
            msg += " (ninguno respondio: dato 13F no disponible, no cuenta como 0)"
        out.append(msg)
    if respondieron_activos(r13) == 0:  # pueden haber respondido los pasivos: no es evidencia de nada
        pasivos = _gestores_pasivos()
        caidos = [g for g in fallidos if g not in pasivos]
        out.append("13f: no respondió ningún gestor activo (" + (", ".join(caidos) if caidos else "ninguno consultado")
                   + "); dato 13F no disponible, no cuenta como 0")
    return out


def resumen_13f(r13) -> tuple[int | None, int | None]:
    """(n_con_posicion, n_acumulando) contando SOLO gestores activos (``pasivo`` falso); None = dato no disponible.

    * Todos los gestores consultados fallaron (o df vacio con fallos y sin saber cuantos se consultaron): None, no 0.
    * ``attrs['respondieron_activos'] == 0`` (fallaron todos los activos aunque respondieran pasivos): None, no 0.
      Si ese attr no existe (contrato antiguo) se usa el comportamiento anterior.
    * Hubo gestores consultados con exito pero ninguno con posicion: 0 es un dato valido.
    * Acumula quien aumento la posicion (``cambio_pct`` > 0) o la abrio (``acciones_previas`` <= 0).
    """
    if r13 is None or not isinstance(r13, pd.DataFrame):
        return None, None
    if respondieron_activos(r13) == 0:
        return None, None
    fallidos, consultados = info_consulta_13f(r13)
    if consultados is not None:
        todos_fallaron = bool(fallidos) and consultados > 0 and len(fallidos) >= consultados
    else:  # sin saber cuantos se consultaron: un df vacio con fallos no prueba "0 gestores con posicion"
        todos_fallaron = bool(fallidos) and r13.empty
    if todos_fallaron:
        return None, None
    act = r13[~_bool_col(r13, "pasivo")] if len(r13) else r13
    if act.empty:
        return 0, 0
    cambio = pd.to_numeric(act["cambio_pct"] if "cambio_pct" in act else pd.Series(np.nan, index=act.index),
                           errors="coerce")
    # sin la columna no se puede afirmar que la posicion sea nueva (previas desconocidas => no cuenta como nueva)
    previas = pd.to_numeric(act["acciones_previas"], errors="coerce").fillna(0) if "acciones_previas" in act \
        else pd.Series(1.0, index=act.index)
    return int(len(act)), int(((cambio > 0) | (previas <= 0)).sum())


def construir_respaldo(insiders: dict | None, form4_ok: bool, r13, r13_ok: bool = True) -> dict:
    """Entrada de ``scoring.s_13f``: {n_con_posicion, n_acumulando, compras_form4_usd}.

    * Form 4: consulta fallida (``form4_ok`` falso) -> ``compras_form4_usd = None`` (no 0); con exito vale el USD
      comprado (0.0 es un dato valido: no hubo compras).
    * 13F: solo gestores activos (ver ``resumen_13f``); consulta fallida (``r13_ok`` falso) -> None.
    """
    n_pos, n_acum = resumen_13f(r13) if r13_ok else (None, None)
    compras = float((insiders or {}).get("valor_usd") or 0.0) if form4_ok else None
    return {"n_con_posicion": n_pos, "n_acumulando": n_acum, "compras_form4_usd": compras}


def n_valoracion_para_scoring(n_val: int | None) -> tuple[int | None, str | None]:
    """(``n_valoracion`` a pasar a ``scoring``, aviso) segun la valoracion se haya consultado o no.

    ``n_val`` es el numero de anclas de valoracion en la zona: 0 es un dato valido (consultada, ninguna ancla cae en la
    zona); ``None`` = la valoracion no se pudo consultar. Si ``scoring.s_confluencia`` admite None (la mitad de
    valoracion se excluye) se le pasa None; si todavia no lo admite (TypeError) se mantiene 0 para no perder todo el
    scoring, pero se AVISA de que esa mitad puntua 0 por falta de dato.
    """
    if n_val is not None:
        return n_val, None
    from src.validation.scoring import s_confluencia
    try:
        s_confluencia(1, None)
    except TypeError:
        return 0, ("valoracion (P/E, EV/FCF, DCF) no disponible: la parte de valoracion de la confluencia puntua 0 "
                   "(scoring aun no la excluye)")
    return None, ("valoracion (P/E, EV/FCF, DCF) no disponible: esa parte se excluye de la confluencia "
                  "(se reescala con las familias)")


def aviso_stock_act(stock: dict | None) -> str | None:
    """Aviso si STOCK Act no se pudo consultar (``disponible`` falso o ``pts`` None): se excluye, no es neutro."""
    if not isinstance(stock, dict) or (stock.get("pts") is not None and stock.get("disponible") is not False):
        return None
    motivo = stock.get("motivo") or stock.get("resumen") or "sin respuesta"
    return f"STOCK Act no disponible: {motivo} (excluido de la confiabilidad; no cuenta como neutro)"


def componentes_sin_dato(confiabilidad: dict | None) -> list[str]:
    """Claves de los componentes de la confiabilidad EXCLUIDOS por falta de dato (cuentan para la cobertura)."""
    conf = confiabilidad or {}
    if conf.get("excluidos") is not None:
        return list(conf["excluidos"])
    if "detalle" not in conf:
        return []
    from src.validation.scoring import COMPONENTES
    return [k for k in COMPONENTES if k not in conf["detalle"]]


def nombres_componentes(claves) -> str:
    return ", ".join(NOMBRES_COMPONENTES.get(k, k) for k in claves)


def _info_mercado(ticker: str, tk_obj, obtener_info) -> dict:
    """``info`` de yfinance. Local: con cache de 6 h en disco (menos peticiones); nube: directo, sin disco."""
    try:
        from src.reporting import userdata
        if not userdata.en_nube():
            from src.levels.fair_value import obtener_info_mercado
            info = obtener_info_mercado(ticker, tk_obj, fetch=lambda _t: obtener_info(tk_obj))
            if info:
                return info
    except Exception:  # noqa: BLE001  (el cache es opcional)
        pass
    return obtener_info(tk_obj)


def analizar_ticker(ticker: str, capital: float, periodo: str = "10y", anios_backtest: int = 5,
                    backtest: bool = True, sec: bool = True, verbose: bool = True, valor_justo: bool = True,
                    modo: str | None = None) -> dict:
    """Pipeline completo de un ticker. Nunca lanza: los fallos de cada etapa quedan en ``errores``.

    ``periodo``: historial de yfinance (10y por defecto; la UI en la nube pasa "3y"). ``modo`` ("local" | "nube" | None)
    fuerza el modo del valor justo (None = se decide por la variable PORT dentro de ``fair_value``). En local el ``info``
    de yfinance se cachea 6 h en disco (``fair_value.obtener_info_mercado``); en la nube no.

    ``valor_justo`` guarda en ``res["entradas_valoracion"]`` (dict liviano, sin DataFrames) los insumos de la bateria
    de modelos de ``src/levels/fair_value.py``; la UI recalcula el valor justo con ellos sin red. Es aditivo: no
    interviene en las zonas, la confluencia ni el score.
    """
    res = {"ticker": ticker, "fecha": datetime.now().strftime("%Y-%m-%d"), "capital": capital,
           "precio_actual": None, "zonas": [], "errores": [], "avisos": [], "df": None, "form4": None, "r13f": None,
           "r13f_fallidos": [], "r13f_consultados": None, "anios_backtest": None, "n_ordenes": None,
           "n_fills": None, "backtest_alcance": None, "componentes_sin_dato": []}

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
    info = _info_mercado(ticker, tk_obj, obtener_info)
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
        # dist_pct: UNICA referencia = r_sup (precio del limite GTC) vs precio actual; la misma que grafico y UI.
        from src.reporting.charts import dist_a_r_sup_pct
        cl = sorted(zonas_raw["clusters"], key=lambda c: (-c["n_familias"], -c["n_anclas"], c["dist_pct"]))
        for i, c in enumerate(cl[:MAX_ZONAS], 1):
            res["zonas"].append({"nivel": i, "r_inf": c["lo"], "r_sup": c["hi"], "p_centro": (c["lo"] + c["hi"]) / 2,
                                 "dist_pct": dist_a_r_sup_pct(c["hi"], res["precio_actual"]),
                                 "n_familias": c["n_familias"],
                                 "n_anclas": c["n_anclas"], "familias": c["familias"], "tipos": c["tipos"],
                                 "fuente": "+".join(c["tipos"])})
    except Exception as e:  # noqa: BLE001
        err("zonas", e)
    prim = zonas_raw.get("primario")

    # 3. Pisos de valoracion (P/E propio, EV/FCF, DCF, log-reg) -> numero de anclas de valoracion en la zona
    #    None = la valoracion no se pudo consultar (dato no disponible); 0 = consultada y ninguna ancla cae en la zona.
    n_val = None
    try:
        from src.levels.valuation_floors import calcular_pisos_valoracion
        val = calcular_pisos_valoracion(ticker, tk_obj, df, info)
        res["valoracion"] = {k: v for k, v in val.items() if k != "anclas"}
        n_val = 0
        if prim:
            lo, hi = prim["lo"] * 0.98, prim["hi"] * 1.02
            n_val = sum(1 for a in val["anclas"] if lo <= a["precio"] <= hi)
    except Exception as e:  # noqa: BLE001
        err("valoracion", e)
    res["n_valoracion_en_zona"] = n_val

    # 3b. Entradas del valor justo (15 modelos): se extraen UNA vez; no afectan a zonas ni a scores
    if valor_justo:
        try:
            from src.levels.fair_value import extraer_entradas
            kw_vj = {"modo": modo} if modo else {}     # sin ``modo`` el motor decide solo (PORT => nube)
            res["entradas_valoracion"] = extraer_entradas(ticker, tk_obj, df, info, **kw_vj).to_dict()
        except Exception as e:  # noqa: BLE001
            err("valor_justo", e)

    # 4. SEC: Form 4 (P), 13F, STOCK Act. Un dato que NO se pudo consultar es None (se excluye de la
    #    confiabilidad y baja la cobertura); un dato consultado que vale 0 es 0.
    respaldo, stock = None, None
    if sec:
        log("consultando SEC EDGAR / STOCK Act")
        form4_ok = r13_ok = True
        try:
            from src.data.sec_edgar import compras_insiders
            f4, ins = compras_insiders(ticker)
            res["form4"], res["insiders"] = f4, ins
        except Exception as e:  # noqa: BLE001
            form4_ok = False
            err("form4", e)
        r13 = None
        try:
            from src.data.sec_edgar import obtener_13f_ticker
            r13 = obtener_13f_ticker(ticker)
            res["r13f"] = r13
            fallidos, consultados = info_consulta_13f(r13)
            res["r13f_fallidos"], res["r13f_consultados"] = fallidos, consultados
            res["avisos"].extend(avisos_13f(r13))
        except Exception as e:  # noqa: BLE001
            r13_ok = False
            err("13f", e)
        respaldo = construir_respaldo(res.get("insiders"), form4_ok, r13, r13_ok)
        res["respaldo13f"] = respaldo
        try:
            from src.data.stock_act import consultar_flujo_congreso_stock_act
            stock = consultar_flujo_congreso_stock_act(ticker)
            res["stock_act"] = {k: v for k, v in stock.items() if k != "ultimas_tx"}
            aviso = aviso_stock_act(stock)
            if aviso:
                res["avisos"].append(aviso)
        except Exception as e:  # noqa: BLE001
            res["stock_act"] = {"pts": None, "disponible": False, "motivo": f"{type(e).__name__}: {e}"}
            err("stock_act", e)

    # 5. Backtest walk-forward del ticker (solo el rango primario N1, no cada zona)
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
            res["anios_backtest"] = anios_backtest
            res["n_ordenes"], res["n_fills"] = int(bt.get("n_ordenes", 0)), int(bt.get("n_fills", 0))
            res["backtest_alcance"] = alcance_backtest(anios_backtest)
            aviso = aviso_muestra_pequena(res["n_fills"])
            if aviso:
                res["avisos"].append(aviso)
        except Exception as e:  # noqa: BLE001
            err("backtest", e)

    # 6. Scoring
    try:
        from src.validation.scoring import s_confluencia, indice_desde_modulos
        n_fam = prim["n_familias"] if prim else None
        n_val_sc, aviso_val = n_valoracion_para_scoring(n_val)
        if aviso_val and n_fam is not None:
            res["avisos"].append(aviso_val)
        res["confluencia_score"] = s_confluencia(n_fam, n_val_sc)
        res["confiabilidad"] = indice_desde_modulos(bt, respaldo, stock, None, n_fam, n_val_sc)
        res["componentes_sin_dato"] = componentes_sin_dato(res["confiabilidad"])
        res["score_ranking"] = puntaje_ranking(res["confluencia_score"], res["confiabilidad"].get("indice"))
    except Exception as e:  # noqa: BLE001
        err("scoring", e)

    asignar_capital(res["zonas"], capital)
    return res


# --- Salida ------------------------------------------------------------------------------------------
def _fmt(v, f="{:.1f}", na="n/d"):
    return na if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def texto_confiabilidad(r: dict) -> str:
    """Linea de confluencia / confiabilidad con cobertura y componentes excluidos por falta de dato."""
    c = r.get("confiabilidad") or {}
    sin = r.get("componentes_sin_dato") or []
    cob = f"cobertura {_fmt((c.get('cobertura') or 0) * 100, '{:.0f}')}%"
    if sin:
        cob += f"; sin dato: {nombres_componentes(sin)}"
    return (f"Confluencia: {_fmt(r.get('confluencia_score'))}/100 | Confiabilidad institucional: "
            f"{_fmt(c.get('indice'))}/100 ({cob}) | Ranking: {_fmt(r.get('score_ranking'))}")


def texto_backtest(r: dict) -> str | None:
    """Linea del backtest con su alcance (rango primario N1, anios) o None si no se ejecuto."""
    bt = r.get("backtest_resumen")
    if not bt:
        return None
    pct = lambda v: _fmt(None if v is None else v * 100, "{:.0f}%")  # noqa: E731
    return (f"Backtest ({r.get('backtest_alcance') or alcance_backtest(r.get('anios_backtest'))}): "
            f"{_fmt(r.get('n_fills'), '{:d}')} fills de {_fmt(r.get('n_ordenes'), '{:d}')} ordenes | "
            f"fill {pct(bt.get('fill_rate'))} | bounce {pct(bt.get('bounce'))} | "
            f"MAE60 mediana {pct(bt.get('mae60_mediana'))}")


def imprimir_individual(r: dict):
    print(f"\n=== {r['ticker']} | Precio ${_fmt(r['precio_actual'], '{:.2f}')} | "
          f"Capital por ticker ${r['capital']:,.0f} ===")
    if not r["zonas"]:
        print("  Sin zonas de confluencia bajo el precio.")
    for z in r["zonas"]:
        print(f"  N{z['nivel']}: ${z['r_inf']:.2f} - ${z['r_sup']:.2f} (dist. a r_sup {_fmt(z.get('dist_pct'), '{:+.1f}')}%) | "
              f"{z['n_familias']} familias [{', '.join(z['familias'])}] | "
              f"${z['capital_asignado']:.0f} -> {z['acciones']:.3f} acc. @ ${z['r_sup']:.2f}")
    print(f"  {texto_confiabilidad(r)}")
    bt = texto_backtest(r)
    if bt:
        print(f"  {bt}")
    for a in r.get("avisos") or []:
        print(f"  [aviso] {a}")
    if r["errores"]:
        print(f"  Advertencias: {len(r['errores'])} etapa(s) con fallo (ver export).")


def imprimir_ranking(rs: list[dict], capital: float | None = None):
    if capital is not None:
        print(f"\n{NOTA_CAPITAL_BATCH} Capital por ticker: ${capital:,.0f}.")
    print(f"\n{'#':>2} {'Ticker':<7}{'Precio':>10}{'Zona 1':>20}{'Dist r_sup':>11}{'Fam':>5}{'Confl.':>8}{'Confiab.':>9}"
          f"{'Cob.':>6}{'Score':>7}")
    for i, r in enumerate(rs, 1):
        z = r["zonas"][0] if r["zonas"] else None
        zona = f"{z['r_inf']:.2f}-{z['r_sup']:.2f}" if z else "n/d"
        c = r.get("confiabilidad") or {}
        cob = "n/d" if c.get("cobertura") is None else f"{c['cobertura'] * 100:.0f}%"
        print(f"{i:>2} {r['ticker']:<7}{_fmt(r['precio_actual'], '{:.2f}'):>10}{zona:>20}"
              f"{_fmt(z.get('dist_pct') if z else None, '{:+.1f}%'):>11}"
              f"{(z['n_familias'] if z else 0):>5}{_fmt(r.get('confluencia_score')):>8}{_fmt(c.get('indice')):>9}"
              f"{cob:>6}{_fmt(r.get('score_ranking')):>7}")
    print("Score = promedio de confluencia y confiabilidad: ayuda para ordenar, no una senal; no comparar scores "
          "con cobertura distinta.")


def _graficar(r: dict, directorio: str, mostrar: bool):
    from src.reporting.charts import graficar_niveles
    from src.reporting.exporter import directorio_salida
    out = directorio_salida(directorio)
    ruta = os.path.join(out, f"{r['ticker']}_niveles.png")
    c = r.get("confiabilidad") or {}
    sub = (f"Confluencia {_fmt(r.get('confluencia_score'))}/100 | Confiabilidad {_fmt(c.get('indice'))}/100 "
           f"(cobertura {_fmt((c.get('cobertura') or 0) * 100, '{:.0f}')}%)")
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
    imprimir_ranking(top, args.capital)
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
