import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime
import subprocess
import glob
import json
import os


def verificar_proximidad_earnings(ticker):
    """
    Detecta la fecha del próximo reporte trimestral de resultados (Earnings)
    y genera una alerta si faltan <= 10 días calendario (riesgo de gap nocturno).
    """
    hoy = datetime.now().date()
    fechas_futuras = []

    # 1. Intentar desde ticker.calendar
    try:
        cal = ticker.calendar
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            raw_dates = cal['Earnings Date']
            if not isinstance(raw_dates, (list, tuple, pd.Series, np.ndarray)):
                raw_dates = [raw_dates]
            for d in raw_dates:
                if pd.notna(d):
                    fecha_d = pd.Timestamp(d).date()
                    if fecha_d >= hoy:
                        fechas_futuras.append(fecha_d)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if 'Earnings Date' in cal.index:
                for d in cal.loc['Earnings Date']:
                    if pd.notna(d):
                        fecha_d = pd.Timestamp(d).date()
                        if fecha_d >= hoy:
                            fechas_futuras.append(fecha_d)
    except Exception:
        pass

    # 2. Respaldo desde get_earnings_dates
    if not fechas_futuras:
        try:
            ed = ticker.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                idx_naive = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
                for ts in idx_naive:
                    fecha_d = ts.date()
                    if fecha_d >= hoy:
                        fechas_futuras.append(fecha_d)
        except Exception:
            pass

    if not fechas_futuras:
        return {
            'fecha_str': 'No disponible',
            'dias_restantes': None,
            'alerta_activa': False,
            'mensaje': 'Fecha de próximos Earnings no disponible en fuente pública'
        }

    proxima_fecha = min(fechas_futuras)
    dias_restantes = (proxima_fecha - hoy).days
    alerta_activa = dias_restantes <= 10

    if dias_restantes == 0:
        msg = f"[!] ALERTA CRÍTICA DE EARNINGS: La empresa reporta HOY ({proxima_fecha}). Pausar órdenes límite."
    elif alerta_activa:
        msg = (
            f"[!] ALERTA DE EARNINGS PRÓXIMO: Faltan {dias_restantes} días ({proxima_fecha}). "
            f"Evitar dejar órdenes límite GTC expuestas a gap de resultados."
        )
    else:
        msg = f"Próximo Reporte (Earnings): {proxima_fecha} (en {dias_restantes} días — Fuera de ventana de riesgo)"

    return {
        'fecha_str': str(proxima_fecha),
        'dias_restantes': dias_restantes,
        'alerta_activa': alerta_activa,
        'mensaje': msg
    }


def calcular_pe_historico_propio(ticker, df, info):
    """
    Calcula la distribución histórica real del P/E de la propia empresa (últimos 3 años)
    cruzando su EPS TTM trimestral contra su serie de precios diarios.
    """
    df_naive = df.copy()
    if df_naive.index.tz is not None:
        df_naive.index = df_naive.index.tz_localize(None)

    forward_eps = info.get('forwardEps', np.nan)
    trailing_eps_info = info.get('trailingEps', np.nan)
    forward_pe = info.get('forwardPE', np.nan)
    trailing_pe = info.get('trailingPE', np.nan)

    pe_validos = pd.Series(dtype=float)
    ultimo_ttm_eps = np.nan

    try:
        q_inc = ticker.quarterly_income_stmt
        if q_inc is not None and not q_inc.empty:
            fila_eps = None
            for nombre_fila in ['Diluted EPS', 'Basic EPS']:
                if nombre_fila in q_inc.index:
                    fila_eps = q_inc.loc[nombre_fila]
                    break

            if fila_eps is not None:
                eps_q = pd.to_numeric(fila_eps, errors='coerce').dropna()
                eps_q.index = pd.to_datetime(eps_q.index).tz_localize(None)
                eps_q = eps_q.sort_index()

                # Suma móvil de 4 trimestres = Trailing Twelve Months (TTM) EPS
                ttm_eps = eps_q.rolling(4).sum().dropna()
                if not ttm_eps.empty:
                    ultimo_ttm_eps = float(ttm_eps.iloc[-1])
                    ttm_diario = ttm_eps.reindex(df_naive.index, method='ffill')
                    pe_diario = df_naive['Close'] / ttm_diario
                    pe_validos = pe_diario[(pe_diario > 3.0) & (pe_diario < 150.0)].dropna()
    except Exception:
        pass

    # Determinar el EPS de referencia para proyectar pisos de precio
    if pd.notna(forward_eps) and forward_eps > 0:
        eps_base = float(forward_eps)
        tipo_eps = "Fwd EPS"
    elif pd.notna(ultimo_ttm_eps) and ultimo_ttm_eps > 0:
        eps_base = ultimo_ttm_eps
        tipo_eps = "TTM EPS"
    elif pd.notna(trailing_eps_info) and trailing_eps_info > 0:
        eps_base = float(trailing_eps_info)
        tipo_eps = "TTM EPS"
    else:
        return None

    # Caso A: Tenemos suficiente historial de P/E propio (>= 60 sesiones)
    if len(pe_validos) >= 60:
        pe_p50 = float(np.percentile(pe_validos, 50))
        pe_p40 = float(np.percentile(pe_validos, 40))
        pe_p25 = float(np.percentile(pe_validos, 25))
        pe_p10 = float(np.percentile(pe_validos, 10))

        niveles_val = [
            (eps_base * pe_p40, f"Valuación Propia ({pe_p40:.1f}x {tipo_eps} | P40 Hist.)"),
            (eps_base * pe_p25, f"Valuación Propia ({pe_p25:.1f}x {tipo_eps} | P25 Hist.)"),
            (eps_base * pe_p10, f"Valuación Propia ({pe_p10:.1f}x {tipo_eps} | P10 Hist.)"),
            (eps_base * (pe_p10 * 0.85), f"Valuación Extrema ({pe_p10*0.85:.1f}x {tipo_eps} | -15% P10)")
        ]
        resumen_pe = f"P/E Mediana Histórica (3A): {pe_p50:.1f}x (Rango P10–P50: {pe_p10:.1f}x – {pe_p50:.1f}x)"
        return {'niveles': niveles_val, 'resumen': resumen_pe, 'pe_mediana': pe_p50}

    # Caso B: Respaldo relativo al múltiplo actual de la empresa (si hay < 4 trimestres de historia)
    pe_ref = np.nan
    if pd.notna(forward_pe) and 3.0 < forward_pe < 150.0:
        pe_ref = float(forward_pe)
    elif pd.notna(trailing_pe) and 3.0 < trailing_pe < 150.0:
        pe_ref = float(trailing_pe)

    if pd.notna(pe_ref):
        niveles_val = [
            (eps_base * (pe_ref * 0.90), f"Valuación Relativa ({pe_ref*0.90:.1f}x {tipo_eps} | -10% P/E)"),
            (eps_base * (pe_ref * 0.80), f"Valuación Relativa ({pe_ref*0.80:.1f}x {tipo_eps} | -20% P/E)"),
            (eps_base * (pe_ref * 0.70), f"Valuación Relativa ({pe_ref*0.70:.1f}x {tipo_eps} | -30% P/E)"),
            (eps_base * (pe_ref * 0.60), f"Valuación Relativa ({pe_ref*0.60:.1f}x {tipo_eps} | -40% P/E)")
        ]
        resumen_pe = f"P/E Referencia Actual: {pe_ref:.1f}x (Historial trimestral insuficiente para percentiles)"
        return {'niveles': niveles_val, 'resumen': resumen_pe, 'pe_mediana': pe_ref}

    return None


def consultar_flujo_congreso_stock_act(ticker_symbol, repo_dir="/content/congress-trading-monitor"):
    """
    Clona/actualiza el repositorio kadoa-org/congress-trading-monitor y audita
    las transacciones divulgadas bajo la ley STOCK Act en public/data/.
    """
    repo_url = "https://github.com/kadoa-org/congress-trading-monitor.git"
    ticker_obj = ticker_symbol.strip().upper()

    try:
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(
                ["git", "-C", repo_dir, "pull", "--ff-only"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        return {
            'pts': 7.5,
            'compras': 0,
            'ventas': 0,
            'resumen': f"Flujo Congreso (STOCK Act): No disponible ({e})",
            'ultimas_tx': []
        }

    data_dir = os.path.join(repo_dir, "public", "data")
    if not os.path.exists(data_dir):
        return {
            'pts': 7.5,
            'compras': 0,
            'ventas': 0,
            'resumen': "Flujo Congreso (STOCK Act): Carpeta public/data no encontrada",
            'ultimas_tx': []
        }

    operaciones_ticker = []
    archivos_json = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)

    for ruta_json in archivos_json:
        nombre_archivo = os.path.basename(ruta_json).lower()
        if any(skip in nombre_archivo for skip in ["meta", "manifest", "schema"]):
            continue

        try:
            with open(ruta_json, "r", encoding="utf-8") as f:
                contenido = json.load(f)
        except Exception:
            continue

        registros = []
        if isinstance(contenido, list):
            registros = contenido
        elif isinstance(contenido, dict):
            for clave in ["trades", "transactions", "filings", "data", "items", "records"]:
                if isinstance(contenido.get(clave), list):
                    registros = contenido[clave]
                    break
            if not registros and isinstance(contenido.get(ticker_obj), list):
                registros = contenido[ticker_obj]

        for reg in registros:
            if not isinstance(reg, dict):
                continue

            tk = str(
                reg.get("ticker")
                or reg.get("symbol")
                or reg.get("asset_ticker")
                or reg.get("stock")
                or ""
            ).strip().upper()

            if tk != ticker_obj:
                continue

            tipo_raw = str(
                reg.get("type")
                or reg.get("tx_type")
                or reg.get("transaction")
                or reg.get("transaction_type")
                or reg.get("action")
                or ""
            ).lower()

            if any(w in tipo_raw for w in ["purchase", "buy", "compra"]):
                lado = "COMPRA"
            elif any(w in tipo_raw for w in ["sale", "sell", "venta"]):
                lado = "VENTA"
            else:
                continue

            fecha = str(
                reg.get("transaction_date")
                or reg.get("tx_date")
                or reg.get("date")
                or reg.get("disclosure_date")
                or "S/F"
            )[:10]

            politico = str(
                reg.get("filer")
                or reg.get("politician")
                or reg.get("representative")
                or reg.get("name")
                or "Funcionario"
            )

            camara = str(reg.get("chamber") or reg.get("source") or reg.get("branch") or "")
            monto = str(reg.get("amount") or reg.get("range") or reg.get("notional") or "")

            operaciones_ticker.append({
                'fecha': fecha,
                'politico': politico,
                'camara': camara,
                'lado': lado,
                'monto': monto
            })

    vistos = set()
    ops_unicas = []
    for op in operaciones_ticker:
        llave = (op['fecha'], op['politico'], op['lado'], op['monto'])
        if llave not in vistos:
            vistos.add(llave)
            ops_unicas.append(op)

    ops_unicas.sort(key=lambda x: x['fecha'], reverse=True)

    compras = sum(1 for o in ops_unicas if o['lado'] == "COMPRA")
    ventas = sum(1 for o in ops_unicas if o['lado'] == "VENTA")
    total_ops = compras + ventas

    if total_ops == 0:
        pts_congreso = 7.5
        resumen = "Flujo Congreso (STOCK Act): Sin operaciones registradas para este ticker (Neutral)"
    else:
        recientes = ops_unicas[:15]
        compras_rec = sum(1 for o in recientes if o['lado'] == "COMPRA")
        ratio_compra = compras_rec / len(recientes)
        pts_congreso = round(ratio_compra * 15.0, 1)
        resumen = (
            f"Flujo Congreso (STOCK Act): {compras} Compras vs {ventas} Ventas totales "
            f"({ratio_compra * 100:.0f}% compras en últimas {len(recientes)} ops)"
        )

    return {
        'pts': pts_congreso,
        'compras': compras,
        'ventas': ventas,
        'resumen': resumen,
        'ultimas_tx': ops_unicas[:4]
    }


def calcular_niveles_institucionales(ticker, df, info):
    """
    Calcula niveles de soporte institucional invariantes al movimiento diario:
    1. High Volume Nodes (Volume Profile de 3 años)
    2. Pivotes estructurales macro (Swing Lows de 40 sesiones)
    3. Anchored VWAP desde el mínimo estructural de 2 años
    4. Pisos de valuación por P/E histórico propio de la empresa
    """
    precio_actual = float(df['Close'].iloc[-1])

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_estructural = float(tr.rolling(14).mean().tail(252).median())

    candidatos = []

    # --- A. VOLUME PROFILE (HIGH VOLUME NODES - 3 AÑOS) ---
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    p_min, p_max = float(df['Low'].min()), float(df['High'].max())
    bins = np.linspace(p_min, p_max, 120)
    vol_hist, bin_edges = np.histogram(typical_price, bins=bins, weights=df['Volume'])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    vol_suavizado = pd.Series(vol_hist).rolling(3, center=True, min_periods=1).mean().values
    picos_idx, _ = find_peaks(
        vol_suavizado,
        prominence=np.percentile(vol_suavizado, 65),
        distance=4
    )
    for idx in picos_idx:
        candidatos.append((float(bin_centers[idx]), "Volume Profile (HVN)", 3))

    # --- B. SWING LOWS ESTRUCTURALES (SOPORTES HISTÓRICOS DE DEMANDA) ---
    ventana = 20
    lows = df['Low'].values
    for i in range(ventana, len(lows) - ventana):
        if lows[i] == np.min(lows[i - ventana:i + ventana + 1]):
            candidatos.append((float(lows[i]), "Soporte Estructural (Swing Low)", 2))

    # --- C. ANCHORED VWAP DESDE EL MÍNIMO ESTRUCTURAL DE 2 AÑOS ---
    df_2y = df.tail(504).copy()
    idx_min_2y = df_2y['Low'].idxmin()
    df_avwap = df.loc[idx_min_2y:].copy()
    tp_avwap = (df_avwap['High'] + df_avwap['Low'] + df_avwap['Close']) / 3.0
    avwap_series = (tp_avwap * df_avwap['Volume']).cumsum() / df_avwap['Volume'].cumsum()
    avwap_actual = float(avwap_series.iloc[-1])
    candidatos.append((avwap_actual, "Anchored VWAP (Ciclo 2A)", 2))

    # --- D. PISOS DE VALUACIÓN POR P/E HISTÓRICO PROPIO ---
    info_pe_propio = calcular_pe_historico_propio(ticker, df, info)
    if info_pe_propio is not None:
        for nivel_val, etiqueta_val in info_pe_propio['niveles']:
            candidatos.append((float(nivel_val), etiqueta_val, 2))

    # --- E. CLUSTERING DE NIVELES CERCANOS (CONFLUENCIA INSTITUCIONAL) ---
    candidatos.sort(key=lambda x: x[0])
    clusters = []
    tolerancia_cluster = max(atr_estructural * 1.2, precio_actual * 0.025)

    for precio, fuente, peso in candidatos:
        if not clusters or (precio - clusters[-1]['precio_medio']) > tolerancia_cluster:
            clusters.append({
                'precios': [precio],
                'fuentes': {fuente},
                'peso': peso,
                'precio_medio': precio
            })
        else:
            c = clusters[-1]
            c['precios'].append(precio)
            c['fuentes'].add(fuente)
            c['peso'] += peso
            c['precio_medio'] = float(np.mean(c['precios']))

    zona_activa = None
    niveles_compra = []

    for c in clusters:
        p = c['precio_medio']
        etiqueta = (
            f"CONFLUENCIA ({' + '.join(sorted(c['fuentes']))})"
            if len(c['fuentes']) > 1
            else list(c['fuentes'])[0]
        )

        if abs(precio_actual - p) / precio_actual <= 0.015:
            zona_activa = (p, etiqueta)

        if p < precio_actual * 0.985:
            niveles_compra.append({
                'precio': p,
                'fuente': etiqueta,
                'peso': c['peso']
            })

    niveles_compra.sort(key=lambda x: x['precio'], reverse=True)

    # --- F. RESPALDO FIBONACCI DE IMPULSO MACRO ---
    if len(niveles_compra) < 4:
        min_macro = float(df['Low'].min())
        max_macro = float(df['High'].max())
        rango_macro = max_macro - min_macro
        for fib in [0.236, 0.382, 0.500, 0.618]:
            nivel_fib = max_macro - (rango_macro * fib)
            if nivel_fib < precio_actual * 0.985:
                if all(abs(nivel_fib - n['precio']) > tolerancia_cluster for n in niveles_compra):
                    niveles_compra.append({
                        'precio': nivel_fib,
                        'fuente': f"Retroceso Macro Fib ({fib*100:.1f}%)",
                        'peso': 1
                    })
        niveles_compra.sort(key=lambda x: x['precio'], reverse=True)

    datos_visuales = {
        'bin_centers': bin_centers,
        'bin_width': float(bin_edges[1] - bin_edges[0]),
        'vol_suavizado': vol_suavizado,
        'picos_idx': picos_idx,
        'avwap_series': avwap_series,
        'idx_min_2y': idx_min_2y
    }

    return niveles_compra[:4], atr_estructural, zona_activa, info_pe_propio, datos_visuales


def verificar_respaldo_institucional(ticker, ticker_symbol, df, info, resultados_niveles):
    """
    Audita la calidad del análisis (0% a 100%) combinando 4 pilares:
    1. Tenencia institucional real 13F e insiders Form 4 -> 30 pts
    2. Flujo de compras del Congreso / STOCK Act (kadoa-org) -> 15 pts
    3. Absorción institucional de volumen en las zonas -> 30 pts
    4. Backtesting de Tasa de Rebote (Bounce Rate a 20 sesiones) -> 25 pts
    """
    detalles_auditoria = []

    # --- PILAR 1: TENENCIA INSTITUCIONAL 13F E INSIDERS (MÁX 30 PTS) ---
    pts_13f = 0.0
    inst_pct = info.get('heldPercentInstitutions', np.nan)
    if pd.notna(inst_pct):
        pts_13f += min(18.0, (float(inst_pct) / 0.70) * 18.0)
        detalles_auditoria.append(f"Tenencia Institucional 13F: {inst_pct * 100:.1f}%")
    else:
        pts_13f += 9.0

    try:
        inst_holders = ticker.institutional_holders
        if inst_holders is not None and not inst_holders.empty:
            top_fondos = inst_holders.iloc[:3, 0].astype(str).tolist()
            detalles_auditoria.append(f"Top Fondos 13F: {', '.join(top_fondos)}")
            if 'pctChange' in inst_holders.columns:
                promedio_cambio = pd.to_numeric(inst_holders['pctChange'], errors='coerce').mean()
                if pd.notna(promedio_cambio) and promedio_cambio > 0:
                    pts_13f += 8.0
                    detalles_auditoria.append(f"Flujo Neto Top Fondos: +{promedio_cambio * 100:.2f}% (Acumulando)")
                else:
                    pts_13f += 4.0
            else:
                pts_13f += 6.0
    except Exception:
        pts_13f += 4.0

    try:
        insider_tx = ticker.insider_transactions
        if insider_tx is not None and not insider_tx.empty:
            texto_tx = insider_tx.astype(str).apply(lambda x: ' '.join(x), axis=1).str.lower()
            compras_insider = texto_tx.str.contains('purchase|buy').sum()
            if compras_insider > 0:
                pts_13f += 4.0
                detalles_auditoria.append(f"Compras Insiders (Form 4): {compras_insider} detectadas")
    except Exception:
        pass

    pts_13f = min(30.0, pts_13f)

    # --- PILAR 2: FLUJO POLÍTICO / CONGRESO STOCK ACT (MÁX 15 PTS) ---
    datos_congreso = consultar_flujo_congreso_stock_act(ticker_symbol)
    pts_congreso = datos_congreso['pts']
    detalles_auditoria.append(datos_congreso['resumen'])
    for tx in datos_congreso['ultimas_tx']:
        detalles_auditoria.append(
            f"   * [{tx['fecha']}] {tx['politico']} ({tx['camara']}): {tx['lado']} {tx['monto']}"
        )

    # --- PILAR 3: ABSORCIÓN INSTITUCIONAL EN LAS ZONAS (MÁX 30 PTS) ---
    df_calc = df.copy()
    df_calc['Vol_SMA20'] = df_calc['Volume'].rolling(20).mean()
    rango_vela = (df_calc['High'] - df_calc['Low']).replace(0, np.nan)
    df_calc['Pos_Cierre'] = (df_calc['Close'] - df_calc['Low']) / rango_vela

    niveles_con_absorcion = 0
    for res in resultados_niveles:
        lim_inf = res['r_inf'] * 0.985
        lim_sup = res['r_sup'] * 1.015
        toques = df_calc[(df_calc['Low'] <= lim_sup) & (df_calc['High'] >= lim_inf)]

        absorciones = toques[
            (toques['Volume'] > toques['Vol_SMA20'] * 1.35) &
            (toques['Pos_Cierre'] >= 0.50)
        ]
        if len(absorciones) >= 1:
            niveles_con_absorcion += 1

    ratio_absorcion = niveles_con_absorcion / max(1, len(resultados_niveles))
    pts_absorcion = ratio_absorcion * 30.0
    detalles_auditoria.append(
        f"Absorción de Volumen Comprobada: {niveles_con_absorcion}/{len(resultados_niveles)} niveles con huella institucional"
    )

    # --- PILAR 4: BACKTESTING DE TASA DE REBOTE (BOUNCE RATE A 20 RUEDAS) (MÁX 25 PTS) ---
    total_toques_eval = 0
    rebotes_exitosos = 0
    horizonte = 20

    closes = df['Close'].values
    lows = df['Low'].values

    for res in resultados_niveles:
        lim_inf = res['r_inf'] * 0.99
        lim_sup = res['r_sup'] * 1.01
        ultimo_toque_idx = -horizonte

        for i in range(len(df) - horizonte):
            if (i - ultimo_toque_idx) < 10:
                continue
            if lim_inf <= lows[i] <= lim_sup:
                ultimo_toque_idx = i
                total_toques_eval += 1
                if closes[i + horizonte] > lows[i]:
                    rebotes_exitosos += 1

    if total_toques_eval > 0:
        bounce_rate = (rebotes_exitosos / total_toques_eval) * 100.0
        pts_backtest = (bounce_rate / 100.0) * 25.0
        detalles_auditoria.append(
            f"Bounce Rate Histórico (20D): {bounce_rate:.1f}% ({rebotes_exitosos}/{total_toques_eval} toques rebotaron)"
        )
    else:
        bounce_rate = np.nan
        pts_backtest = 15.0
        detalles_auditoria.append("Bounce Rate Histórico: N/A (Zona inédita proyectada por valuación/Fibonacci)")

    score_verificacion = round(pts_13f + pts_congreso + pts_absorcion + pts_backtest, 1)

    if score_verificacion >= 75:
        grado = "ALTA CONFIABILIDAD INSTITUCIONAL"
    elif score_verificacion >= 55:
        grado = "CONFIABILIDAD MODERADA / VALIDADA"
    else:
        grado = "BAJA CONFIABILIDAD (ESPECULATIVO)"

    return {
        'score_pct': score_verificacion,
        'grado': grado,
        'bounce_rate_pct': round(bounce_rate, 1) if pd.notna(bounce_rate) else None,
        'congreso_compras': datos_congreso['compras'],
        'congreso_ventas': datos_congreso['ventas'],
        'detalles': detalles_auditoria
    }


def graficar_niveles_institucionales(
    df,
    ticker_symbol,
    precio_actual,
    resultados_niveles,
    precio_promedio,
    score_fund,
    verificacion,
    info_earnings,
    datos_visuales,
    ruta_guardado=None
):
    bg_fig = "#0d1117"
    bg_panel = "#131722"
    grid_color = "#232936"
    text_color = "#e6edf3"
    muted_text = "#8b949e"

    fig, (ax_price, ax_vp) = plt.subplots(
        1, 2,
        figsize=(16, 9),
        sharey=True,
        gridspec_kw={'width_ratios': [4.3, 1.0], 'wspace': 0.025}
    )
    fig.patch.set_facecolor(bg_fig)

    for ax in (ax_price, ax_vp):
        ax.set_facecolor(bg_panel)
        ax.tick_params(colors=muted_text, labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    ax_price.plot(df.index, df['Close'], color="#58a6ff", linewidth=1.4, label="Precio Cierre (Ajustado)", zorder=4)
    ax_price.fill_between(df.index, df['Low'], df['High'], color="#58a6ff", alpha=0.10, zorder=2)

    sma_200 = df['Close'].rolling(200).mean()
    ax_price.plot(df.index, sma_200, color="#8b949e", linestyle=":", linewidth=1.1, label="SMA 200D", zorder=3)

    avwap_series = datos_visuales['avwap_series']
    idx_min_2y = datos_visuales['idx_min_2y']
    ax_price.plot(
        avwap_series.index,
        avwap_series.values,
        color="#f59e0b",
        linewidth=1.5,
        label=f"Anchored VWAP Ciclo (${avwap_series.iloc[-1]:.2f})",
        zorder=5
    )
    ax_price.scatter(
        [idx_min_2y],
        [df.loc[idx_min_2y, 'Low']],
        color="#f59e0b",
        s=45,
        zorder=6,
        label="Ancla Mínimo 2A"
    )

    paleta_niveles = ["#2ea043", "#1f883d", "#00b4d8", "#a371f7"]
    x_label_pos = df.index[max(0, int(len(df) * 0.015))]

    for idx, res in enumerate(resultados_niveles):
        color_lvl = paleta_niveles[idx % len(paleta_niveles)]
        p_centro = res['p_centro']
        pct_desc = ((p_centro / precio_actual) - 1.0) * 100.0

        ax_price.axhspan(res['r_inf'], res['r_sup'], color=color_lvl, alpha=0.16, zorder=1)
        ax_price.axhline(p_centro, color=color_lvl, linestyle="--", linewidth=1.2, alpha=0.9, zorder=3)
        ax_vp.axhline(p_centro, color=color_lvl, linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)

        fuente_limpia = res['fuente'] if len(res['fuente']) <= 48 else res['fuente'][:45] + "..."
        etiqueta_txt = f" N{res['nivel_num']}: ${p_centro:.2f} ({pct_desc:+.1f}%) | {fuente_limpia} "

        ax_price.text(
            x_label_pos,
            p_centro,
            etiqueta_txt,
            color="#ffffff",
            fontsize=8.1,
            fontweight="bold",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=bg_fig, edgecolor=color_lvl, alpha=0.88),
            zorder=7
        )

    ax_price.axhline(
        precio_actual,
        color="#f0f6fc",
        linestyle="-",
        linewidth=1.2,
        alpha=0.85,
        label=f"Cotización Actual (${precio_actual:.2f})",
        zorder=5
    )

    if precio_promedio > 0:
        ax_price.axhline(
            precio_promedio,
            color="#ff7b72",
            linestyle="-.",
            linewidth=1.3,
            label=f"Precio Promedio Cartera (${precio_promedio:.2f})",
            zorder=5
        )

    # Banner visual si faltan <= 10 días para Earnings
    if info_earnings['alerta_activa']:
        ax_price.text(
            0.985,
            0.965,
            f" ALERTA EARNINGS EN {info_earnings['dias_restantes']} DÍAS ({info_earnings['fecha_str']}) — RIESGO DE GAP ",
            transform=ax_price.transAxes,
            color="#ffffff",
            fontsize=9.0,
            fontweight="bold",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#b62324", edgecolor="#ff7b72", alpha=0.95),
            zorder=12
        )

    bin_centers = datos_visuales['bin_centers']
    vol_suavizado = datos_visuales['vol_suavizado']
    picos_idx = set(datos_visuales['picos_idx'])
    bar_height = datos_visuales['bin_width'] * 0.88

    colores_vp = ["#f59e0b" if i in picos_idx else "#30363d" for i in range(len(bin_centers))]
    ax_vp.barh(bin_centers, vol_suavizado, height=bar_height, color=colores_vp, edgecolor="none", alpha=0.85, zorder=2)
    ax_vp.set_xlabel("Volumen 3A\n(Naranja = HVN)", color=muted_text, fontsize=8.5)
    ax_vp.xaxis.set_visible(False)

    todos_precios = [float(df['Low'].min()), float(df['High'].max()), precio_actual]
    for res in resultados_niveles:
        todos_precios.extend([res['r_inf'], res['r_sup']])

    ax_price.set_ylim(min(todos_precios) * 0.95, max(todos_precios) * 1.04)
    ax_price.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax_price.set_ylabel("Precio (USD)", color=text_color, fontsize=10, fontweight="bold")

    titulo = (
        f"{ticker_symbol} — Mapa Institucional Blindado (P/E Propio + Volume Profile + STOCK Act)\n"
        f"Score Fundamental: {score_fund}/100  |  Verificación: {verificacion['score_pct']}% ({verificacion['grado']})  |  "
        f"Próximo Earnings: {info_earnings['fecha_str']}"
    )
    ax_price.set_title(titulo, color=text_color, fontsize=10.6, fontweight="bold", pad=14, loc="left")

    leyenda = ax_price.legend(
        loc="upper left",
        facecolor=bg_fig,
        edgecolor=grid_color,
        labelcolor=text_color,
        fontsize=8.5,
        framealpha=0.9
    )
    leyenda.set_zorder(10)

    plt.tight_layout()

    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=180, bbox_inches="tight", facecolor=bg_fig)
        print(f"  [√] Gráfico institucional exportado en: {ruta_guardado}")

    plt.show()


def screener_institucional_csv():
    print("=" * 80)
    print("  SCREENER CUANTITATIVO DE NIVELES INSTITUCIONALES (BLINDADO)")
    print("=" * 80)

    ticker_symbol = input("\nIngresa el Ticker del activo (ej. NVDA, JPM, KO, META): ").strip().upper()
    if not ticker_symbol:
        ticker_symbol = "NVDA"

    capital_input = input("Ingresa el capital total a asignar en USD (ej. 240 o 10000): ").strip()
    try:
        capital_total = float(capital_input) if capital_input else 10000.0
    except ValueError:
        capital_total = 10000.0

    frac_input = input("¿Permitir acciones fraccionadas? (S/N) [Por defecto: S]: ").strip().upper()
    permitir_fraccionadas = False if frac_input == 'N' else True

    print(f"\n[1/5] Descargando historial ajustado (3 años), fundamentales y calendario para {ticker_symbol}...")

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="3y", auto_adjust=True)
    if df.empty or len(df) < 100:
        print(f"Error: No se encontraron datos históricos suficientes para '{ticker_symbol}'.")
        return

    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    precio_actual = float(df['Close'].iloc[-1])
    maximo_52s = float(df['High'].tail(252).max())
    caida_desde_max = ((precio_actual - maximo_52s) / maximo_52s) * 100.0

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # Verificar calendario de Earnings
    info_earnings = verificar_proximidad_earnings(ticker)

    pe_ratio = info.get('trailingPE', info.get('forwardPE', np.nan))
    forward_pe = info.get('forwardPE', np.nan)
    debt_to_equity = info.get('debtToEquity', np.nan)
    free_cashflow = info.get('freeCashflow', np.nan)
    rev_growth = info.get('revenueGrowth', np.nan)
    op_margins = info.get('operatingMargins', np.nan)

    score_fund = 0
    puntos_fuertes = []
    puntos_atencion = []

    if pd.notna(op_margins) and op_margins > 0.15:
        score_fund += 25
        puntos_fuertes.append(f"Margen Operativo Sólido ({op_margins * 100:.1f}%)")
    else:
        puntos_atencion.append("Margen Operativo Bajo/Negativo")

    if pd.notna(rev_growth) and rev_growth > 0:
        score_fund += 25
        puntos_fuertes.append(f"Crecimiento Ventas Positivo ({rev_growth * 100:.1f}%)")
    else:
        puntos_atencion.append("Crecimiento Ventas Debilitado")

    if pd.notna(debt_to_equity) and debt_to_equity < 150:
        score_fund += 25
        puntos_fuertes.append(f"Deuda Saludable (D/E: {debt_to_equity:.1f}%)")
    else:
        puntos_atencion.append("Elevado Apalancamiento (D/E > 150%)")

    if pd.notna(free_cashflow) and free_cashflow > 0:
        score_fund += 25
        puntos_fuertes.append(f"Flujo Caja Libre Positivo (${free_cashflow / 1e9:.2f}B)")
    else:
        puntos_atencion.append("Flujo de Caja Libre Negativo")

    es_activo_calidad = score_fund >= 50

    print("[2/5] Calculando HVN, pivotes macro, AVWAP y percentiles históricos de P/E...")
    niveles_inst, atr_estructural, zona_activa, info_pe_propio, datos_visuales = calcular_niveles_institucionales(
        ticker, df, info
    )

    ancho_banda = 0.35 * atr_estructural
    pesos_capital = [0.20, 0.30, 0.30, 0.20]

    resultados_niveles = []
    total_acciones = 0.0
    total_invertido = 0.0

    for idx, nivel in enumerate(niveles_inst):
        pct_cap = pesos_capital[idx]
        p_centro = nivel['precio']
        rango_inf = max(0.01, p_centro - ancho_banda)
        rango_sup = p_centro + ancho_banda
        cap_nivel = capital_total * pct_cap

        if permitir_fraccionadas:
            acciones = round(cap_nivel / p_centro, 4)
        else:
            acciones = float(int(cap_nivel // p_centro))

        inv_efectiva = acciones * p_centro
        total_acciones += acciones
        total_invertido += inv_efectiva

        resultados_niveles.append({
            'nivel_num': idx + 1,
            'fuente': nivel['fuente'],
            'p_centro': p_centro,
            'r_inf': rango_inf,
            'r_sup': rango_sup,
            'cap_usd': cap_nivel,
            'acciones': acciones,
            'inv_usd': inv_efectiva
        })

    precio_promedio = total_invertido / total_acciones if total_acciones > 0 else 0.0

    print("[3/5] Auditando respaldo 13F, compras del Congreso (STOCK Act) y backtesting...")
    verificacion = verificar_respaldo_institucional(ticker, ticker_symbol, df, info, resultados_niveles)

    print("\n" + "=" * 115)
    print(f" INFORME DE ACUMULACIÓN INSTITUCIONAL ESTRUCTURAL: {ticker_symbol}")
    print("=" * 115)
    print(f"• Cotización Actual:               ${precio_actual:.2f}")
    print(f"• Máximo 52 Semanas:               ${maximo_52s:.2f} ({caida_desde_max:.2f}% desde máximo)")
    print(f"• ATR Estructural (Volatilidad):   ${atr_estructural:.2f} (Banda zona: ±${ancho_banda:.2f})")
    print(f"• Capital Total Disponible:        ${capital_total:,.2f} USD")
    print(f"• Modalidad de Operación:          {'Acciones Fraccionadas' if permitir_fraccionadas else 'Acciones Enteras'}")
    print(f"• Calendario Corporativo:          {info_earnings['mensaje']}")
    if zona_activa:
        print(f"• [!] ALERTA DE ZONA ACTIVA:       El precio actual ya cotiza sobre el soporte ${zona_activa[0]:.2f} [{zona_activa[1]}]")
    print("-" * 115)
    print(" DIAGNÓSTICO FUNDAMENTAL Y VALUACIÓN PROPIA:")
    print(f"  • Puntuación de Calidad:         {score_fund}/100")
    if pd.notna(pe_ratio):
        f_pe_str = f" | Forward P/E: {forward_pe:.2f}x" if pd.notna(forward_pe) else ""
        print(f"  • Múltiplos Actuales:            Trailing P/E: {pe_ratio:.2f}x{f_pe_str}")
    if info_pe_propio is not None:
        print(f"  • Historial P/E de la Empresa:   {info_pe_propio['resumen']}")
    print(f"  • Fortalezas:                    {', '.join(puntos_fuertes) if puntos_fuertes else 'Ninguna'}")
    if puntos_atencion:
        print(f"  • Puntos de Riesgo:              {', '.join(puntos_atencion)}")

    print("-" * 115)
    print(" AUDITORÍA DE VERIFICACIÓN INSTITUCIONAL (13F + STOCK ACT + ABSORCIÓN + BACKTEST):")
    print(f"  • CALIFICACIÓN DEL ANÁLISIS:     {verificacion['score_pct']}%  -->  [{verificacion['grado']}]")
    for item in verificacion['detalles']:
        print(f"    - {item}")

    print("\n VEREDICTO DEL MODELO:")
    if info_earnings['alerta_activa']:
        print(f"  [!] PAUSA TÁCTICA POR EARNINGS: Reporte trimestral en {info_earnings['dias_restantes']} días ({info_earnings['fecha_str']}).")
        print("      Esperar la publicación de resultados antes de activar órdenes límite GTC.")
    elif es_activo_calidad and verificacion['score_pct'] >= 55:
        print("  [√] ACTIVO Y NIVELES VALIDADOS: Calidad fundamental y respaldo institucional aprobados.")
        print("      Colocar órdenes límite GTC (Good-Til-Canceled) en las zonas estructurales:")
    elif es_activo_calidad:
        print("  [!] CALIDAD FUNDAMENTAL OK, PERO BAJA CONFIRMACIÓN EN NIVELES: Reducir tamaño de posición.")
    else:
        print("  [X] ALERTA DE CALIDAD FUNDAMENTAL: Deterioro financiero detectado.")
        print("      No promediar a la baja sin mejora previa en márgenes o flujo de caja.")

    print("-" * 115)
    print(f"{'Nivel':<8} | {'Ancla Institucional':<45} | {'Zona Objetivo (Desc. vs Hoy)':<31} | {'Acciones':<9} | {'Inversión'}")
    print("-" * 115)
    for res in resultados_niveles:
        pct_centro = ((res['p_centro'] / precio_actual) - 1.0) * 100.0
        zona_str = f"${res['r_inf']:.2f} - ${res['r_sup']:.2f} ({pct_centro:+.1f}%)"
        acc_fmt = f"{res['acciones']:.4f}" if permitir_fraccionadas else f"{int(res['acciones'])}"
        fuente_corta = (res['fuente'][:42] + "...") if len(res['fuente']) > 45 else res['fuente']
        print(f"Nivel {res['nivel_num']:<2} | {fuente_corta:<45} | {zona_str:<31} | {acc_fmt:<9} | ${res['inv_usd']:,.2f}")

    print("-" * 115)
    print(" PROYECCIÓN DE CARTERA (Si se ejecutan los 4 niveles institucionales):")
    acc_tot_str = f"{total_acciones:.4f}" if permitir_fraccionadas else f"{int(total_acciones)}"
    print(f"  • Acciones Totales Acumuladas:   {acc_tot_str} títulos")
    print(f"  • Capital Efectivo Invertido:    ${total_invertido:,.2f} USD (Remanente: ${capital_total - total_invertido:,.2f} USD)")
    if total_acciones > 0:
        print(f"  • PRECIO PROMEDIO PONDERADO:     ${precio_promedio:.2f} USD")
        print(f"  • Margen de Seguridad Medio:     {((precio_promedio / precio_actual) - 1) * 100:.2f}% bajo cotización actual")
    else:
        print("  [!] ADVERTENCIA: Capital insuficiente para acciones enteras. Activa acciones fraccionadas.")
    print("=" * 115)

    timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fecha_str = datetime.now().strftime("%Y-%m-%d")

    datos_escaneo = {
        'Fecha_Hora': timestamp_actual,
        'Ticker': ticker_symbol,
        'Precio_Actual_USD': round(precio_actual, 2),
        'Maximo_52s_USD': round(maximo_52s, 2),
        'Proximo_Earnings': info_earnings['fecha_str'],
        'Dias_Para_Earnings': info_earnings['dias_restantes'],
        'Alerta_Earnings_10D': info_earnings['alerta_activa'],
        'Score_Fundamental': score_fund,
        'PE_Mediana_3A': round(info_pe_propio['pe_mediana'], 2) if info_pe_propio else None,
        'Verificacion_Institucional_Pct': verificacion['score_pct'],
        'Grado_Verificacion': verificacion['grado'],
        'Congreso_Compras': verificacion['congreso_compras'],
        'Congreso_Ventas': verificacion['congreso_ventas'],
        'Bounce_Rate_Historico_Pct': verificacion['bounce_rate_pct'],
        'Apto_Acumulacion': es_activo_calidad,
        'Capital_Total_USD': round(capital_total, 2),
        'Acciones_Totales': round(total_acciones, 4),
        'Capital_Invertido_USD': round(total_invertido, 2),
        'Precio_Promedio_USD': round(precio_promedio, 2),
    }
    for i in range(4):
        if i < len(resultados_niveles):
            datos_escaneo[f'Nivel_{i+1}_Precio_USD'] = round(resultados_niveles[i]['p_centro'], 2)
            datos_escaneo[f'Nivel_{i+1}_Fuente'] = resultados_niveles[i]['fuente']
            datos_escaneo[f'Nivel_{i+1}_Acciones'] = resultados_niveles[i]['acciones']

    df_escaneo = pd.DataFrame([datos_escaneo])

    print("\n[4/5] Guardando registro estructurado...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)

        directorio_raiz = "/content/drive/My Drive/Colab_Screener_History"
        directorio_ticker = os.path.join(directorio_raiz, ticker_symbol)
        os.makedirs(directorio_ticker, exist_ok=True)

        archivo_csv = os.path.join(directorio_ticker, f"institucional_{ticker_symbol}_{fecha_str}.csv")
        ruta_grafico = os.path.join(directorio_ticker, f"mapa_institucional_{ticker_symbol}_{fecha_str}.png")

        df_escaneo.to_csv(
            archivo_csv,
            mode='a' if os.path.exists(archivo_csv) else 'w',
            header=not os.path.exists(archivo_csv),
            index=False
        )
        print(f"  [√] CSV guardado permanentemente en Google Drive: {archivo_csv}")
    except Exception as e:
        archivo_local = f"institucional_{ticker_symbol}.csv"
        ruta_grafico = os.path.abspath(f"mapa_institucional_{ticker_symbol}_{fecha_str}.png")
        df_escaneo.to_csv(
            archivo_local,
            mode='a' if os.path.exists(archivo_local) else 'w',
            header=not os.path.exists(archivo_local),
            index=False
        )
        print(f"  [!] CSV guardado localmente ({e}): {os.path.abspath(archivo_local)}")

    print("\n[5/5] Generando mapa visual de Volume Profile y zonas institucionales...\n")
    graficar_niveles_institucionales(
        df=df,
        ticker_symbol=ticker_symbol,
        precio_actual=precio_actual,
        resultados_niveles=resultados_niveles,
        precio_promedio=precio_promedio,
        score_fund=score_fund,
        verificacion=verificacion,
        info_earnings=info_earnings,
        datos_visuales=datos_visuales,
        ruta_guardado=ruta_grafico
    )


if __name__ == "__main__":
    screener_institucional_csv()