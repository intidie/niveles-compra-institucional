"""Ingesta desde SEC EDGAR (fuente primaria): 13F-HR, Form 4 (código P) y XBRL."""
import logging
import os
import warnings
from datetime import datetime, timedelta

import pandas as pd
from edgar import Company, set_identity

logger = logging.getLogger(__name__)

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

# Gestores pasivos / indexados: sus flujos replican indices y NO cuentan como conviccion. Se muestran en el
# DataFrame de ``obtener_13f_ticker`` (columna ``pasivo``) pero se excluyen del flag de acumulacion.
# MISMA definicion que ``PASIVOS`` en src/validation/case_studies.py (esa lista se mantiene alli por dominio de
# validacion; case_studies importa GESTORES_13F desde este modulo, por lo que NO se importa en sentido inverso
# para evitar un ciclo). tests/test_data_fixes.py verifica que ambas coincidan: si se edita una, editar la otra.
GESTORES_PASIVOS = frozenset({"Vanguard Group", "State Street"})

COLUMNAS_13F = ["gestor", "periodo", "acciones", "valor_usd", "acciones_previas", "cambio_pct", "pasivo"]

# Form 4: si falla el parseo de MAS de esta fraccion de los filings, el resultado no es fiable y se lanza error.
UMBRAL_FALLO_FORM4 = 0.5


def configurar_identidad(identidad=None):
    """SEC exige User-Agent con contacto. Usa el valor dado, EDGAR_IDENTITY o SEC_EDGAR_IDENTITY."""
    global _identidad_ok
    ident = (identidad or os.environ.get("EDGAR_IDENTITY") or os.environ.get("SEC_EDGAR_IDENTITY")
             or "NivelesScreener contacto@example.com")
    set_identity(ident)
    _identidad_ok = True


def _asegurar_identidad():
    if not _identidad_ok:
        configurar_identidad()


def _resumir_error(exc, maxlen=120):
    """'TipoError: mensaje' en una sola linea y acotado a `maxlen` caracteres."""
    msg = " ".join(str(exc).split())
    texto = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return texto[:maxlen]


def obtener_form4(ticker_symbol, dias=365, max_filings=60):
    """
    Transacciones de insiders (Form 4) de los últimos `dias`.
    La columna 'Code' permite filtrar P (compra en mercado abierto).

    Errores (nunca se devuelve un DataFrame vacío para ocultar un fallo):
    * Falla de red / ticker inexistente / error de edgartools al listar filings: la excepción se propaga.
    * Fallo de parseo de filings individuales: se registra (logging + ``warnings.warn``). Si fallan MÁS de
      ``UMBRAL_FALLO_FORM4`` (50 %) de los filings intentados se lanza ``RuntimeError`` (resultado no fiable).
    * "Consulta exitosa sin filings en el periodo" devuelve un DataFrame vacío (cero real).
    ``df.attrs`` incluye ``filings_total`` (intentados) y ``filings_fallidos``.
    """
    _asegurar_identidad()
    empresa = Company(ticker_symbol)
    desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    filings = empresa.get_filings(form="4", filing_date=f"{desde}:")
    filas, total, fallidos, ultimo_error = [], 0, 0, None
    for i, filing in enumerate(filings if filings is not None else []):
        if i >= max_filings:
            break
        total += 1
        try:
            df = filing.obj().to_dataframe()
        except Exception as exc:  # noqa: BLE001
            fallidos += 1
            ultimo_error = exc
            logger.warning("Form 4 %s: filing %d ilegible (%s)", ticker_symbol, i, _resumir_error(exc))
            continue
        if df is not None and not df.empty:
            filas.append(df)
    if total and fallidos / total > UMBRAL_FALLO_FORM4:
        raise RuntimeError(
            f"Form 4 {ticker_symbol}: {fallidos} de {total} filings no pudieron parsearse "
            f"(último error: {_resumir_error(ultimo_error)})") from ultimo_error
    if fallidos:
        warnings.warn(f"Form 4 {ticker_symbol}: {fallidos} de {total} filings no pudieron parsearse; "
                      f"el conteo de compras puede estar subestimado", RuntimeWarning, stacklevel=2)
    salida = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    salida.attrs["filings_total"] = total
    salida.attrs["filings_fallidos"] = fallidos
    return salida


def compras_insiders(ticker_symbol, dias=365, max_filings=60):
    """
    Solo compras reales en mercado abierto (Transaction Code == 'P'). Devuelve (df, resumen).

    Comportamiento ante fallos (contrato para el consumidor):
    * Consulta fallida (red, ticker desconocido, mayoría de filings ilegibles): LANZA excepción
      (ver ``obtener_form4``). Nunca devuelve ceros para un fallo: el consumidor debe capturarla y tratar
      el dato como no disponible.
    * Consulta exitosa con 0 compras P (incluye "sin Form 4 en el periodo"): devuelve resumen con
      ``compras=0`` y ``disponible=True``.
    ``resumen``: compras, acciones, valor_usd, transacciones_analizadas, disponible (siempre True si
    retorna), filings_total, filings_fallidos (filings individuales ilegibles, minoría tolerada).
    """
    df = obtener_form4(ticker_symbol, dias, max_filings)
    meta = {"disponible": True, "filings_total": int(df.attrs.get("filings_total", 0)),
            "filings_fallidos": int(df.attrs.get("filings_fallidos", 0))}
    if df.empty or "Code" not in df.columns:
        # edgartools omite 'Code' cuando ningún Form 4 trae transacciones: es un cero real, no un error.
        return pd.DataFrame(), {"compras": 0, "acciones": 0.0, "valor_usd": 0.0,
                                "transacciones_analizadas": 0, **meta}
    compras = df[df["Code"].astype(str).str.strip() == "P"].copy()
    acciones = pd.to_numeric(compras["Shares"], errors="coerce").sum() if "Shares" in compras else 0.0
    valor = pd.to_numeric(compras["Value"], errors="coerce").sum() if "Value" in compras else 0.0
    resumen = {
        "compras": int(len(compras)),
        "acciones": float(acciones) if not compras.empty else 0.0,
        "valor_usd": float(valor) if not compras.empty else 0.0,
        "transacciones_analizadas": int(len(df)),
        **meta,
    }
    return compras, resumen


def _posicion(thirteenf, tk):
    """(acciones, valor_usd) del ticker en un 13F-HR. Lanza ValueError si la tabla no es legible."""
    it = getattr(thirteenf, "infotable", None)
    if it is None or "Ticker" not in it.columns:
        # No se puede confundir "tabla ilegible" con "sin posición": se trata como fallo del gestor.
        raise ValueError("infotable del 13F-HR ausente o sin columna Ticker")
    sel = it[it["Ticker"] == tk]
    return float(sel["SharesPrnAmount"].sum()), float(sel["Value"].sum())


def _consultar_gestor(nombre, cik, tk):
    """Fila 13F del gestor para `tk`, o None si respondió pero no tiene posición. Lanza si la consulta falla."""
    lista = list(Company(cik).get_filings(form="13F-HR").latest(2))
    if not lista:
        raise LookupError("sin 13F-HR disponibles en EDGAR")
    actual = lista[0].obj()
    previo = lista[1].obj() if len(lista) > 1 else None
    acc, val = _posicion(actual, tk)
    acc_prev = _posicion(previo, tk)[0] if previo is not None else 0.0
    if acc <= 0 and acc_prev <= 0:
        return None
    return {
        "gestor": nombre,
        "periodo": str(actual.report_period),
        "acciones": acc,
        "valor_usd": val,
        "acciones_previas": acc_prev,
        "cambio_pct": ((acc - acc_prev) / acc_prev * 100.0) if acc_prev > 0 else None,
        "pasivo": nombre in GESTORES_PASIVOS,
    }


def obtener_13f_ticker(ticker_symbol, gestores=None):
    """
    Posición del ticker en el último 13F-HR de cada gestor, con cambio vs. trimestre previo.

    DataFrame (una fila por gestor CON posición, activos y pasivos): gestor, periodo, acciones, valor_usd,
    acciones_previas, cambio_pct, pasivo (bool; True para ``GESTORES_PASIVOS``: Vanguard/State Street, que se
    muestran pero NO deben contarse como acumulación de convicción, igual que case_studies.PASIVOS).

    ``df.attrs`` (siempre presentes, también si todos fallan y con ``gestores=`` personalizado; la clasificación
    activo/pasivo se hace por nombre contra ``GESTORES_PASIVOS``):
    * ``consultados``: nº de gestores intentados.
    * ``fallidos``: nombres de los gestores cuya consulta lanzó excepción o no respondió (sin 13F-HR,
      tabla ilegible, etc.). Estos NO se tratan como "sin posición".
    * ``errores``: {nombre: 'TipoError: mensaje'} para los fallidos.
    * ``sin_posicion``: gestores (activos y pasivos) que respondieron y no tienen el ticker (ni ahora ni el
      trimestre previo); ``sin_posicion_activos`` / ``sin_posicion_pasivos`` son su partición por tipo.
    * ``respondieron``: consultados - len(fallidos).
    * ``consultados_activos`` / ``respondieron_activos``: gestores activos intentados / que respondieron sin
      fallar (con o sin posición). ``consultados_pasivos`` / ``respondieron_pasivos``: ídem para pasivos.
    Cada fallo se registra con ``warnings.warn`` (RuntimeWarning) y logging; la función no lanza por gestor.
    Si TODOS fallan devuelve un DataFrame vacío (con las columnas) y ``fallidos`` lleno: el consumidor debe
    tratarlo como dato no disponible, no como cero. Si solo responden pasivos (``respondieron_activos == 0``)
    tampoco hay dato de convicción activa. Ver ``resumir_13f`` para el conteo de acumulación.
    """
    _asegurar_identidad()
    gestores = gestores or GESTORES_13F
    tk = ticker_symbol.strip().upper()
    filas, fallidos, errores, sin_posicion = [], [], {}, []
    for nombre, cik in gestores.items():
        try:
            fila = _consultar_gestor(nombre, cik, tk)
        except Exception as exc:  # noqa: BLE001
            resumen_err = _resumir_error(exc)
            fallidos.append(nombre)
            errores[nombre] = resumen_err
            logger.warning("13F %s (CIK %s) falló para %s: %s", nombre, cik, tk, resumen_err)
            warnings.warn(f"13F {nombre}: consulta fallida para {tk} ({resumen_err})", RuntimeWarning, stacklevel=2)
            continue
        if fila is None:
            sin_posicion.append(nombre)
        else:
            filas.append(fila)
    df = pd.DataFrame(filas, columns=COLUMNAS_13F)
    pasivos = [n for n in gestores if n in GESTORES_PASIVOS]
    activos = [n for n in gestores if n not in GESTORES_PASIVOS]
    fallidos_set = set(fallidos)
    df.attrs.update({
        "consultados": len(gestores), "fallidos": fallidos, "errores": errores,
        "sin_posicion": sin_posicion, "respondieron": len(gestores) - len(fallidos),
        "sin_posicion_activos": [n for n in sin_posicion if n not in GESTORES_PASIVOS],
        "sin_posicion_pasivos": [n for n in sin_posicion if n in GESTORES_PASIVOS],
        "consultados_activos": len(activos),
        "respondieron_activos": sum(1 for n in activos if n not in fallidos_set),
        "consultados_pasivos": len(pasivos),
        "respondieron_pasivos": sum(1 for n in pasivos if n not in fallidos_set),
    })
    if gestores and len(fallidos) == len(gestores):
        warnings.warn(f"13F: los {len(gestores)} gestores fallaron para {tk}; dato no disponible",
                      RuntimeWarning, stacklevel=2)
    return df


def resumir_13f(df):
    """
    Conteo de acumulación 13F con la MISMA definición que case_studies (pasivos excluidos del flag).

    Entrada: DataFrame de ``obtener_13f_ticker``. Salida (dict):
    * ``disponible``: ``respondieron_activos > 0``. Si solo responden pasivos (o ningún gestor), el dato de
      convicción activa NO está disponible. Con al menos un activo respondiendo, "0 gestores con posición" es un
      cero real.
    * ``consultados``, ``respondieron``, ``fallidos`` (lista de nombres).
    * ``consultados_activos``, ``respondieron_activos``, ``consultados_pasivos``, ``respondieron_pasivos``.
    * ``n_con_posicion`` y ``n_acumulando``: SOLO gestores activos (drop-in para ``scoring.s_13f``); ``None``
      (no 0) cuando ``respondieron_activos == 0``. Acumulando = cambio_pct > 0 o posición nueva
      (acciones_previas <= 0 con acciones > 0).
    * ``n_con_posicion_total``, ``n_pasivos_con_posicion``, ``n_pasivos_acumulando``: informativos, siempre int.
    Si el DataFrame no trae los attrs nuevos (``*_activos``/``*_pasivos``) se infieren de la columna ``pasivo`` y
    de ``fallidos``/``sin_posicion``; sin ningún attr, solo cuentan los gestores con fila (un DataFrame vacío sin
    attrs resulta no disponible).
    """
    attrs = getattr(df, "attrs", None) or {}
    if len(df):
        pasivo = df["pasivo"].astype(bool) if "pasivo" in df else df["gestor"].isin(GESTORES_PASIVOS)
        cambio = pd.to_numeric(df["cambio_pct"], errors="coerce")
        nuevas = pd.to_numeric(df["acciones_previas"], errors="coerce").fillna(0) <= 0
        acum = (cambio > 0) | nuevas
    else:
        pasivo = acum = pd.Series(dtype=bool)
    filas_act, filas_pas = int((~pasivo).sum()), int(pasivo.sum())

    fallidos = list(attrs.get("fallidos", []))
    sin_pos = list(attrs.get("sin_posicion", []))
    inf_resp_act = filas_act + sum(1 for n in sin_pos if n not in GESTORES_PASIVOS)
    inf_resp_pas = filas_pas + sum(1 for n in sin_pos if n in GESTORES_PASIVOS)
    inf_cons_act = inf_resp_act + sum(1 for n in fallidos if n not in GESTORES_PASIVOS)
    inf_cons_pas = inf_resp_pas + sum(1 for n in fallidos if n in GESTORES_PASIVOS)

    def _entero(clave, inferido):
        valor = attrs.get(clave)
        return int(valor) if valor is not None else int(inferido)

    # max(...): una fila en el DataFrame prueba que ese gestor respondió, aunque un attr diga lo contrario.
    resp_act = max(_entero("respondieron_activos", inf_resp_act), filas_act)
    resp_pas = max(_entero("respondieron_pasivos", inf_resp_pas), filas_pas)
    cons_act = max(_entero("consultados_activos", inf_cons_act), resp_act)
    cons_pas = max(_entero("consultados_pasivos", inf_cons_pas), resp_pas)
    disponible = resp_act > 0
    return {
        "disponible": disponible,
        "consultados": _entero("consultados", cons_act + cons_pas),
        "respondieron": _entero("respondieron", resp_act + resp_pas),
        "fallidos": fallidos,
        "consultados_activos": cons_act,
        "respondieron_activos": resp_act,
        "consultados_pasivos": cons_pas,
        "respondieron_pasivos": resp_pas,
        "n_con_posicion": filas_act if disponible else None,
        "n_acumulando": int((acum & ~pasivo).sum()) if disponible else None,
        "n_con_posicion_total": int(len(df)),
        "n_pasivos_con_posicion": filas_pas,
        "n_pasivos_acumulando": int((acum & pasivo).sum()),
    }


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
