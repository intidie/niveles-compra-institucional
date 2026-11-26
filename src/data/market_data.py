"""Datos de mercado (precios, info, calendario de earnings) vía yfinance."""
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


def descargar_historial(ticker_symbol, periodo="3y"):
    """Devuelve (ticker, df) con OHLCV ajustado y limpio; df vacío si no hay datos."""
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=periodo, auto_adjust=True)
    if df.empty:
        return ticker, df
    return ticker, df.dropna(subset=["Close", "High", "Low", "Volume"])


def obtener_info(ticker):
    try:
        return ticker.info or {}
    except Exception:
        return {}


def verificar_proximidad_earnings(ticker, umbral_dias=10):
    hoy = datetime.now().date()
    futuras = []
    try:
        cal = ticker.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            raw = cal["Earnings Date"]
            if not isinstance(raw, (list, tuple, pd.Series, np.ndarray)):
                raw = [raw]
            futuras += [pd.Timestamp(d).date() for d in raw if pd.notna(d) and pd.Timestamp(d).date() >= hoy]
        elif isinstance(cal, pd.DataFrame) and not cal.empty and "Earnings Date" in cal.index:
            futuras += [pd.Timestamp(d).date() for d in cal.loc["Earnings Date"]
                        if pd.notna(d) and pd.Timestamp(d).date() >= hoy]
    except Exception:
        pass

    if not futuras:
        try:
            ed = ticker.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
                futuras += [t.date() for t in idx if t.date() >= hoy]
        except Exception:
            pass

    if not futuras:
        return {"fecha_str": "No disponible", "dias_restantes": None, "alerta_activa": False,
                "mensaje": "Fecha de próximos Earnings no disponible en fuente pública"}

    prox = min(futuras)
    dias = (prox - hoy).days
    alerta = dias <= umbral_dias
    if dias == 0:
        msg = f"[!] ALERTA CRÍTICA DE EARNINGS: La empresa reporta HOY ({prox}). Pausar órdenes límite."
    elif alerta:
        msg = (f"[!] ALERTA DE EARNINGS PRÓXIMO: Faltan {dias} días ({prox}). "
               f"Evitar dejar órdenes límite GTC expuestas a gap de resultados.")
    else:
        msg = f"Próximo Reporte (Earnings): {prox} (en {dias} días — Fuera de ventana de riesgo)"
    return {"fecha_str": str(prox), "dias_restantes": dias, "alerta_activa": alerta, "mensaje": msg}
