"""Pisos de valoracion: percentiles historicos propios (P/E, EV/FCF), DCF conservador y bandas log-regresion.

Referencias metodologicas
-------------------------
* Percentiles historicos propios: se compara el multiplo actual con la distribucion de la propia empresa
  (P10/P25/P40), cruzando precios historicos con EPS/FCF de XBRL (SEC EDGAR, fuente primaria).
  Base conceptual: Graham & Dodd, *Security Analysis*; Damodaran, *Investment Valuation* (multiplos relativos).
* DCF conservador: Benjamin Graham (*The Intelligent Investor*, margen de seguridad) y Aswath Damodaran
  (*The Dark Side of Valuation*, *Investment Valuation*): FCF a 5 anios, tasa de descuento 10 %,
  crecimiento con tope que decae linealmente hasta el crecimiento terminal, valor terminal de Gordon.
* Bandas de regresion logaritmica: ajuste ln(precio) = a + b*t (tendencia exponencial de largo plazo) y
  bandas a -1.5 / -2.0 desviaciones estandar de los residuos (log-regression channel; ver tambien
  Benoit Mandelbrot sobre escalado logaritmico de precios).

Fuentes y fallbacks (documentados)
----------------------------------
* EPS / OCF / CapEx / acciones / caja / deuda: ``Company.get_facts()`` (XBRL companyfacts) de edgartools.
* Si EDGAR falla o no devuelve EPS: fallback a ``ticker.quarterly_income_stmt`` de yfinance (solo ~5 trimestres:
  los percentiles quedan marcados como ``fuente = 'yfinance'``). Sin historial suficiente se omite el multiplo.
* Split-adjustment: los EPS XBRL antiguos pueden estar en base pre-split; se ajustan con ``ticker.splits``
  para los splits posteriores a la ultima presentacion que contiene el dato.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# --- Parametros ---------------------------------------------------------------------------------
PERCENTILES = (10, 25, 40)          # Requisito: P10, P25, P40.
PE_MIN, PE_MAX = 3.0, 150.0         # Recorte de outliers (misma convencion que el legado).
MIN_OBS_MULTIPLO = 252              # >= 1 anio de sesiones para estimar percentiles.
REZAGO_PUBLICACION_DIAS = 45        # 10-Q large accelerated filer: 40 d; 10-K: 60 d. Evita look-ahead.
TASA_DESCUENTO = 0.10               # Requisito: 10 %.
CRECIMIENTO_TOPE = 0.10             # Tope de crecimiento FCF anual inicial (conservador).
CRECIMIENTO_HAIRCUT = 0.5           # Se proyecta solo la mitad del CAGR historico observado.
CRECIMIENTO_TERMINAL = 0.025        # < crecimiento nominal del PIB de largo plazo.
HORIZONTE_DCF = 5                   # Requisito: 5 anios.
MARGENES_SEGURIDAD = (0.20, 0.30)   # Requisito: 20 % y 30 %.
BANDAS_SD = (1.5, 2.0)              # Requisito: -1.5 SD y -2.0 SD.


# --- Utilidades XBRL ----------------------------------------------------------------------------
def _facts(ticker_symbol: str) -> pd.DataFrame:
    from src.data.sec_edgar import _asegurar_identidad
    from edgar import Company
    _asegurar_identidad()
    df = Company(ticker_symbol).get_facts().to_pandas()
    for c in ("period_start", "period_end"):
        df[c] = pd.to_datetime(df[c])
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    df["dur"] = (df["period_end"] - df["period_start"]).dt.days
    # Referencia de fecha de presentacion: fin del periodo principal del mismo (fiscal_year, fiscal_period).
    df["ref_filing"] = df.groupby(["fiscal_year", "fiscal_period"])["period_end"].transform("max")
    return df


def _factor_split(splits: pd.Series | None, desde: pd.Timestamp) -> float:
    """Producto de factores de split ocurridos despues de ``desde``."""
    if splits is None or len(splits) == 0:
        return 1.0
    s = splits.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return float(s[s.index > desde].prod()) if (s.index > desde).any() else 1.0


def _serie_concepto(facts, concept, tipo, splits=None, ajusta_split=False):
    """Filas deduplicadas (ultima presentacion) del concepto; opcionalmente ajustadas por splits."""
    f = facts[(facts["concept"] == concept) & (facts["period_type"] == tipo)].dropna(subset=["numeric_value"])
    if f.empty:
        return f
    f = f.sort_values("ref_filing").drop_duplicates(["period_start", "period_end"], keep="last").copy()
    if ajusta_split:
        fac = f["ref_filing"].apply(lambda t: _factor_split(splits, t + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)))
        f["numeric_value"] = f["numeric_value"] / fac
    return f.sort_values("period_end")


def _flujo_trimestral(facts, concept, splits=None, ajusta_split=False) -> pd.Series:
    """Flujo trimestral aislado (index=period_end). Los XBRL de 10-Q/10-K reportan YTD acumulado:
    se diferencian periodos con el mismo ``period_start`` (Q1, H1, 9M, FY) para obtener Q1..Q4."""
    f = _serie_concepto(facts, concept, "duration", splits, ajusta_split)
    f = f[f["dur"] <= 380]
    if f.empty:
        return pd.Series(dtype=float)
    q = {}
    for _, g in f.groupby("period_start"):
        g = g.sort_values("period_end")
        prev = 0.0
        for _, r in g.iterrows():
            if r["dur"] <= 110:
                q[r["period_end"]] = r["numeric_value"]
            else:
                q.setdefault(r["period_end"], r["numeric_value"] - prev)
            prev = r["numeric_value"]
    # Los trimestres aislados prevalecen: se re-escriben con el valor standalone si existe.
    for _, r in f[f["dur"] <= 110].iterrows():
        q[r["period_end"]] = r["numeric_value"]
    return pd.Series(q).sort_index()


def _ttm_diario(q: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """TTM = suma 4 trimestres, disponible REZAGO dias despues del cierre del periodo, ffill diario."""
    ttm = q.rolling(4).sum().dropna()
    ttm.index = ttm.index + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)
    return ttm.reindex(ttm.index.union(idx)).ffill().reindex(idx)


def _puntual_diario(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def _percentiles(serie: pd.Series) -> dict:
    return {f"P{p}": float(np.percentile(serie, p)) for p in PERCENTILES} | {"P50": float(np.percentile(serie, 50))}


def _primer_concepto(facts, conceptos, tipo, **kw):
    for c in conceptos:
        s = _serie_concepto(facts, c, tipo, **kw)
        if not s.empty:
            return c, s
    return None, pd.DataFrame()


# --- P/E historico propio ------------------------------------------------------------------------
def pe_historico(ticker_symbol, ticker_obj, df, facts=None) -> dict | None:
    """P/E diario = Close / EPS diluido TTM (XBRL). Percentiles P10/P25/P40 y pisos = EPS_TTM * Pxx."""
    d = df.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    splits = getattr(ticker_obj, "splits", None)
    fuente, eps_q = "SEC EDGAR XBRL (EarningsPerShareDiluted)", pd.Series(dtype=float)
    try:
        facts = facts if facts is not None else _facts(ticker_symbol)
        eps_q = _flujo_trimestral(facts, "us-gaap:EarningsPerShareDiluted", splits, True)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"EDGAR EPS fallo ({e}); fallback yfinance")
    if len(eps_q) < 8:
        try:
            q = ticker_obj.quarterly_income_stmt
            fila = next(r for r in ("Diluted EPS", "Basic EPS") if r in q.index)
            eps_q = pd.to_numeric(q.loc[fila], errors="coerce").dropna()
            eps_q.index = pd.to_datetime(eps_q.index).tz_localize(None)
            eps_q = eps_q.sort_index()
            fuente = "FALLBACK yfinance quarterly_income_stmt"
        except Exception:  # noqa: BLE001
            return None
    ttm_d = _ttm_diario(eps_q, d.index)
    pe = (d["Close"] / ttm_d)
    pe = pe[(ttm_d > 0) & (pe > PE_MIN) & (pe < PE_MAX)].dropna()
    ttm_actual = float(eps_q.tail(4).sum())
    if len(pe) < MIN_OBS_MULTIPLO or ttm_actual <= 0:
        return None
    pcts = _percentiles(pe)
    return {"fuente": fuente, "eps_ttm": ttm_actual, "n_obs": int(len(pe)),
            "desde": str(pe.index[0].date()), "pe_actual": float(d["Close"].iloc[-1] / ttm_actual),
            "percentiles": pcts,
            "pisos": {k: ttm_actual * v for k, v in pcts.items() if k != "P50"}}


# --- EV/FCF historico ----------------------------------------------------------------------------
def ev_fcf_historico(ticker_symbol, ticker_obj, df, facts=None) -> dict | None:
    """EV/FCF diario = (Close*acciones_diluidas + deuda - caja) / FCF_TTM, con FCF = OCF - CapEx (XBRL).

    EV_piso(Pxx) = multiplo_Pxx * FCF_TTM -> precio = (EV_piso - deuda + caja) / acciones_actuales.
    Devuelve None si falta algun insumo (no se inventa)."""
    d = df.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    splits = getattr(ticker_obj, "splits", None)
    try:
        facts = facts if facts is not None else _facts(ticker_symbol)
        ocf = _flujo_trimestral(facts, "us-gaap:NetCashProvidedByUsedInOperatingActivities")
        capex = _flujo_trimestral(facts, "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment")
        fcf_q = (ocf - capex.reindex(ocf.index).fillna(0.0)).dropna()
        _, sh = _primer_concepto(facts, ["us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"],
                                 "duration", splits=splits, ajusta_split=False)
        sh = sh[sh["dur"] <= 110]
        fac = sh["ref_filing"].apply(lambda t: _factor_split(splits, t + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)))
        sh_q = pd.Series((sh["numeric_value"] * fac).values, index=sh["period_end"].values)
        cajas = []
        for c in ("us-gaap:CashAndCashEquivalentsAtCarryingValue", "us-gaap:MarketableSecuritiesCurrent",
                  "us-gaap:ShortTermInvestments"):
            s = _serie_concepto(facts, c, "instant")
            if not s.empty:
                cajas.append(s.set_index("period_end")["numeric_value"])
        _, deuda_df = _primer_concepto(facts, ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent"], "instant")
        if len(fcf_q) < 8 or sh_q.empty or not cajas:
            return None
        caja = pd.concat(cajas, axis=1).ffill().sum(axis=1)
        deuda = deuda_df.set_index("period_end")["numeric_value"] if not deuda_df.empty else pd.Series(0.0, index=caja.index)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"EV/FCF no disponible: {e}")
        return None
    fcf_d = _ttm_diario(fcf_q, d.index)
    sh_d = _puntual_diario(sh_q.sort_index(), d.index)
    caja_d, deuda_d = _puntual_diario(caja, d.index), _puntual_diario(deuda, d.index).fillna(0.0)
    ev = d["Close"] * sh_d + deuda_d - caja_d
    m = (ev / fcf_d)
    m = m[(fcf_d > 0) & (ev > 0) & (m > 0) & (m < 300)].dropna()
    if len(m) < MIN_OBS_MULTIPLO:
        return None
    fcf_act, sh_act = float(fcf_q.tail(4).sum()), float(sh_q.sort_index().iloc[-1])
    caja_act, deuda_act = float(caja.iloc[-1]), float(deuda.iloc[-1])
    pcts = _percentiles(m)
    pisos = {k: (v * fcf_act - deuda_act + caja_act) / sh_act for k, v in pcts.items() if k != "P50"}
    return {"fuente": "SEC EDGAR XBRL (OCF - CapEx)", "fcf_ttm": fcf_act, "acciones": sh_act,
            "caja": caja_act, "deuda": deuda_act, "n_obs": int(len(m)), "desde": str(m.index[0].date()),
            "percentiles": pcts, "pisos": pisos,
            "nota": "caja = efectivo + valores negociables corrientes; deuda = LongTermDebt (sin leases)"}


# --- DCF conservador ------------------------------------------------------------------------------
def dcf_conservador(ticker_symbol, ticker_obj, info=None, facts=None) -> dict | None:
    """DCF de 5 anios (Graham/Damodaran).

    FCF0 = FCF TTM XBRL (OCF - CapEx). g0 = min(CRECIMIENTO_TOPE, CRECIMIENTO_HAIRCUT * CAGR_FCF_anual),
    con piso 0; g decae linealmente hasta CRECIMIENTO_TERMINAL en el anio 5; VT = FCF5*(1+gT)/(r-gT).
    Valor/accion = (VP(FCF) + VP(VT) + caja - deuda) / acciones. Precios de acumulacion = valor*(1-MoS).
    """
    info = info or {}
    fcf0 = cagr = sh = caja = deuda = None
    fuente = "SEC EDGAR XBRL"
    try:
        facts = facts if facts is not None else _facts(ticker_symbol)
        ocf = _flujo_trimestral(facts, "us-gaap:NetCashProvidedByUsedInOperatingActivities")
        capex = _flujo_trimestral(facts, "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment")
        fcf_q = (ocf - capex.reindex(ocf.index).fillna(0.0)).dropna()
        fcf0 = float(fcf_q.tail(4).sum())
        anual = fcf_q.rolling(4).sum().dropna()
        anual = anual[anual.index >= anual.index[-1] - pd.DateOffset(years=4)].iloc[::-4][::-1]  # anios disjuntos
        if len(anual) >= 2 and anual.iloc[0] > 0 and anual.iloc[-1] > 0:
            cagr = (anual.iloc[-1] / anual.iloc[0]) ** (1.0 / (len(anual) - 1)) - 1.0
        for c in ("us-gaap:CashAndCashEquivalentsAtCarryingValue", "us-gaap:MarketableSecuritiesCurrent",
                  "us-gaap:ShortTermInvestments"):
            s = _serie_concepto(facts, c, "instant")
            caja = (caja or 0.0) + (float(s["numeric_value"].iloc[-1]) if not s.empty else 0.0)
        _, dd = _primer_concepto(facts, ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtNoncurrent"], "instant")
        deuda = float(dd["numeric_value"].iloc[-1]) if not dd.empty else 0.0
        sh_s = _serie_concepto(facts, "dei:EntityCommonStockSharesOutstanding", "instant")
        sh = float(sh_s["numeric_value"].iloc[-1]) if not sh_s.empty else None
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"EDGAR DCF fallo ({e}); fallback yfinance")
    if not fcf0 or fcf0 <= 0 or not sh:
        fcf0 = info.get("freeCashflow") or fcf0
        sh = info.get("sharesOutstanding") or sh
        caja = info.get("totalCash") if caja is None else caja
        deuda = info.get("totalDebt") if deuda is None else deuda
        fuente = "FALLBACK yfinance info"
    if not fcf0 or fcf0 <= 0 or not sh:
        return None
    caja, deuda = caja or 0.0, deuda or 0.0
    g0 = 0.0 if cagr is None else min(CRECIMIENTO_TOPE, max(0.0, CRECIMIENTO_HAIRCUT * cagr))
    gs = np.linspace(g0, CRECIMIENTO_TERMINAL, HORIZONTE_DCF)
    flujos, f = [], fcf0
    for g in gs:
        f *= 1.0 + g
        flujos.append(f)
    vp = sum(fl / (1 + TASA_DESCUENTO) ** (t + 1) for t, fl in enumerate(flujos))
    vt = flujos[-1] * (1 + CRECIMIENTO_TERMINAL) / (TASA_DESCUENTO - CRECIMIENTO_TERMINAL)
    vp_vt = vt / (1 + TASA_DESCUENTO) ** HORIZONTE_DCF
    valor = (vp + vp_vt + caja - deuda) / sh
    return {"fuente": fuente, "fcf0": fcf0, "cagr_fcf_hist": cagr, "g_inicial": float(g0),
            "g_terminal": CRECIMIENTO_TERMINAL, "tasa": TASA_DESCUENTO, "caja": caja, "deuda": deuda,
            "acciones": sh, "vp_flujos": float(vp), "vp_terminal": float(vp_vt),
            "valor_intrinseco": float(valor),
            "precios_mos": {f"MoS{int(m * 100)}": float(valor * (1 - m)) for m in MARGENES_SEGURIDAD}}


# --- Regresion logaritmica ------------------------------------------------------------------------
def bandas_log_regresion(df: pd.DataFrame) -> dict:
    """ln(Close) = a + b*t sobre todo el historial recibido; bandas = exp(fit_hoy - k*SD_residuos)."""
    d = df["Close"].dropna()
    t = (d.index.tz_localize(None) if getattr(d.index, "tz", None) is not None else d.index)
    x = (t - t[0]).days.values / 365.25
    y = np.log(d.values)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    sd = float(res.std(ddof=2))
    fit_hoy = a + b * x[-1]
    return {"anios": float(x[-1]), "cagr_tendencia": float(np.exp(b) - 1), "sd_log": sd,
            "fit_hoy": float(np.exp(fit_hoy)),
            "bandas": {f"-{k}SD": float(np.exp(fit_hoy - k * sd)) for k in BANDAS_SD},
            "r2": float(1 - res.var() / y.var())}


# --- Orquestacion del modulo ----------------------------------------------------------------------
def calcular_pisos_valoracion(ticker_symbol, ticker_obj, df, info=None) -> dict:
    """Devuelve dict con cada modelo y la lista plana 'anclas' (clase 'valoracion')."""
    try:
        facts = _facts(ticker_symbol)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"EDGAR facts no disponible: {e}")
        facts = None
    pe = pe_historico(ticker_symbol, ticker_obj, df, facts)
    evf = ev_fcf_historico(ticker_symbol, ticker_obj, df, facts) if facts is not None else None
    dcf = dcf_conservador(ticker_symbol, ticker_obj, info, facts)
    lg = bandas_log_regresion(df)

    anclas = []

    def add(tipo, nombre, precio, **meta):
        if precio and precio > 0 and np.isfinite(precio):
            anclas.append({"tipo": tipo, "clase": "valoracion", "nombre": nombre, "lo": float(precio),
                           "hi": float(precio), "precio": float(precio), "meta": meta})

    if pe:
        for k, v in pe["pisos"].items():
            add(f"PE_{k}", f"P/E propio {k} ({pe['percentiles'][k]:.1f}x * EPS TTM {pe['eps_ttm']:.2f})", v,
                fuente=pe["fuente"])
    if evf:
        for k, v in evf["pisos"].items():
            add(f"EVFCF_{k}", f"EV/FCF propio {k} ({evf['percentiles'][k]:.1f}x)", v, fuente=evf["fuente"])
    if dcf:
        for k, v in dcf["precios_mos"].items():
            add(f"DCF_{k}", f"DCF 5a r=10% con margen de seguridad {k[3:]}%", v, fuente=dcf["fuente"])
    for k, v in lg["bandas"].items():
        add(f"LOG_{k}", f"Regresion logaritmica {k}", v)
    return {"pe": pe, "ev_fcf": evf, "dcf": dcf, "log_regresion": lg, "anclas": anclas}
