"""Ingesta desde SEC EDGAR (fuente primaria): 13F-HR, Form 4 (código P) y XBRL."""
import os
from datetime import datetime, timedelta

import pandas as pd
from edgar import Company, set_identity

_identidad_ok = False

# Gestores 13F de referencia (CIK). Un 13F lo presenta el fondo, no el emisor,
# por lo que la tenencia de un ticker se obtiene cruzando los filings de estos gestores.
GESTORES_13F = {
    "Berkshire Hathaway": 1067983,
    "Vanguard Group": 102909,
    "State Street": 93751,
    "Bridgewater": 1350694,
    "Renaissance Technologies": 1037389,
}


def configurar_identidad(identidad=None):
    """SEC exige User-Agent con contacto. Usa SEC_EDGAR_IDENTITY o el valor dado."""
    global _identidad_ok
    ident = identidad or os.environ.get("SEC_EDGAR_IDENTITY") or "NivelesScreener contacto@example.com"
    set_identity(ident)
    _identidad_ok = True


def _asegurar_identidad():
    if not _identidad_ok:
        configurar_identidad()


def obtener_form4(ticker_symbol, dias=365, max_filings=60):
    """
    Transacciones de insiders (Form 4) de los últimos `dias`.
    La columna 'Code' permite filtrar P (compra en mercado abierto).
    """
    _asegurar_identidad()
    empresa = Company(ticker_symbol)
    desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    filings = empresa.get_filings(form="4", filing_date=f"{desde}:")
    filas = []
    for i, filing in enumerate(filings):
        if i >= max_filings:
            break
        try:
            df = filing.obj().to_dataframe()
        except Exception:
            continue
        if df is not None and not df.empty:
            filas.append(df)
    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def compras_insiders(ticker_symbol, dias=365, max_filings=60):
    """Solo compras reales en mercado abierto (código P). Devuelve (df, resumen)."""
    df = obtener_form4(ticker_symbol, dias, max_filings)
    if df.empty or "Code" not in df.columns:
        return pd.DataFrame(), {"compras": 0, "acciones": 0.0, "valor_usd": 0.0, "transacciones_analizadas": 0}
    compras = df[df["Code"] == "P"].copy()
    resumen = {
        "compras": int(len(compras)),
        "acciones": float(compras["Shares"].sum()) if not compras.empty else 0.0,
        "valor_usd": float(compras["Value"].sum()) if not compras.empty else 0.0,
        "transacciones_analizadas": int(len(df)),
    }
    return compras, resumen


def _posicion(thirteenf, tk):
    if thirteenf is None or thirteenf.infotable is None:
        return 0.0, 0.0
    it = thirteenf.infotable
    if "Ticker" not in it.columns:
        return 0.0, 0.0
    sel = it[it["Ticker"] == tk]
    return float(sel["SharesPrnAmount"].sum()), float(sel["Value"].sum())


def obtener_13f_ticker(ticker_symbol, gestores=None):
    """
    Posición del ticker en el último 13F-HR de cada gestor, con cambio vs. trimestre previo.
    DataFrame: gestor, periodo, acciones, valor_usd, acciones_previas, cambio_pct.
    """
    _asegurar_identidad()
    gestores = gestores or GESTORES_13F
    tk = ticker_symbol.strip().upper()
    filas = []
    for nombre, cik in gestores.items():
        try:
            lista = list(Company(cik).get_filings(form="13F-HR").latest(2))
            actual = lista[0].obj()
            previo = lista[1].obj() if len(lista) > 1 else None
        except Exception:
            continue

        acc, val = _posicion(actual, tk)
        acc_prev, _ = _posicion(previo, tk)
        if acc <= 0 and acc_prev <= 0:
            continue
        filas.append({
            "gestor": nombre,
            "periodo": str(actual.report_period),
            "acciones": acc,
            "valor_usd": val,
            "acciones_previas": acc_prev,
            "cambio_pct": ((acc - acc_prev) / acc_prev * 100.0) if acc_prev > 0 else None,
        })
    return pd.DataFrame(filas)


def obtener_financieros_xbrl(ticker_symbol):
    """Estados financieros XBRL del último 10-K/10-Q vía edgartools."""
    _asegurar_identidad()
    fin = Company(ticker_symbol).get_financials()
    if fin is None:
        return {}
    return {
        "ingresos": fin.get_revenue(),
        "utilidad_neta": fin.get_net_income(),
        "income_statement": fin.income_statement().to_dataframe(),
        "balance_sheet": fin.balance_sheet().to_dataframe(),
        "cash_flow": fin.cash_flow_statement().to_dataframe(),
    }
