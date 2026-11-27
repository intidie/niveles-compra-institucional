"""Exportacion de resultados a CSV / JSON estructurado (local y Google Colab)."""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

FORMATOS_VALIDOS = ("csv", "json")

# Alcance de las cifras exportadas (se repite en cada fila del CSV y en ``notas`` del JSON para que nadie lo pierda).
CAPITAL_ALCANCE = "por ticker (en modo watchlist no se reparte entre tickers)"
NOTAS = {
    "capital": "capital por ticker: en modo --watchlist cada ticker recibe el capital completo; no se reparte entre "
               "tickers (si se llenaran las órdenes de N tickers haría falta N veces el capital).",
    "backtest": "bt_*: métricas del rango primario (N1) del ticker con un backtest walk-forward de bt_anios años "
                "(def. 5); no son comparables con el pool de docs/METODOLOGIA.md. Con pocos bt_n_fills (ver avisos) "
                "son ruido.",
    "dist_pct": "dist_pct / z1_dist_pct: distancia % de r_sup (precio del límite GTC) al precio actual; negativo = por "
                "debajo. Es la misma referencia que usan la consola, el gráfico y la UI.",
    "confiabilidad": "confiabilidad_cobertura: fracción del peso con dato. confiabilidad_sin_dato lista los componentes "
                     "excluidos: un dato no disponible no cuenta como 0 ni como neutro.",
    "respaldo_13f": "gestores_13f_*: solo gestores ACTIVOS (Vanguard y State Street, pasivos, se ignoran); vacío = dato "
                    "no disponible (p. ej. todos los gestores fallaron), distinto de 0.",
    "ranking": "score_ranking = promedio de confluencia y confiabilidad: ayuda de organización, no una señal "
               "predictiva (la confluencia domina el score).",
}


def es_colab() -> bool:
    """True si el proceso corre dentro de Google Colab."""
    return "google.colab" in sys.modules


def directorio_salida(directorio: str | None = None) -> str:
    """Directorio de salida (creado). Colab: /content/resultados; local: ./resultados."""
    base = directorio or ("/content/resultados" if es_colab() else "resultados")
    os.makedirs(base, exist_ok=True)
    return base


def a_serializable(obj):
    """Convierte recursivamente numpy/pandas/fechas a tipos JSON; NaN/Inf/NaT -> None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return a_serializable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return a_serializable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return a_serializable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): a_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [a_serializable(v) for v in obj]
    return str(obj)


def _lista_txt(v) -> str | None:
    """Lista de textos -> 'a; b' (None si esta vacia o no hay lista)."""
    return "; ".join(str(x) for x in v) if v else None


def aplanar_resultado(res: dict) -> dict:
    """Fila plana (una por ticker) con las metricas clave para CSV/ranking."""
    zonas = res.get("zonas") or []
    z = zonas[0] if zonas else {}
    conf = res.get("confiabilidad") or {}
    ins = res.get("insiders") or {}
    r13 = res.get("respaldo13f") or {}
    bt = res.get("backtest_resumen") or {}
    sa = res.get("stock_act") or {}
    return {
        "ticker": res.get("ticker"),
        "fecha": res.get("fecha"),
        "precio_actual": res.get("precio_actual"),
        "n_zonas": len(zonas),
        "z1_r_inf": z.get("r_inf"), "z1_r_sup": z.get("r_sup"), "z1_dist_pct": z.get("dist_pct"),
        "z1_n_familias": z.get("n_familias"), "z1_familias": "+".join(z.get("familias", [])) or None,
        "confluencia_score": res.get("confluencia_score"),
        "confiabilidad_indice": conf.get("indice"),
        "confiabilidad_cobertura": conf.get("cobertura"),
        "confiabilidad_sin_dato": _lista_txt(res.get("componentes_sin_dato")),
        "score_ranking": res.get("score_ranking"),
        "insiders_compras": ins.get("compras"), "insiders_valor_usd": ins.get("valor_usd"),
        "gestores_13f_con_posicion": r13.get("n_con_posicion"),
        "gestores_13f_acumulando": r13.get("n_acumulando"),
        "gestores_13f_fallidos": _lista_txt(res.get("r13f_fallidos")),
        "stock_act_pts": sa.get("pts"), "stock_act_disponible": sa.get("disponible"),
        "bt_alcance": res.get("backtest_alcance"),
        "bt_anios": res.get("anios_backtest"), "bt_n_ordenes": res.get("n_ordenes"), "bt_n_fills": res.get("n_fills"),
        "bt_fill_rate": bt.get("fill_rate"), "bt_bounce": bt.get("bounce"),
        "bt_ret60_mediana": bt.get("ret60_mediana"), "bt_ret120_mediana": bt.get("ret120_mediana"),
        "bt_ret252_mediana": bt.get("ret252_mediana"), "bt_mae60_mediana": bt.get("mae60_mediana"),
        "capital": res.get("capital"), "capital_alcance": CAPITAL_ALCANCE,
        "avisos": _lista_txt(res.get("avisos")),
        "errores": "; ".join(res.get("errores", [])) or None,
    }


def _formatos(formatos) -> tuple:
    if isinstance(formatos, str):
        formatos = ("csv", "json") if formatos == "both" else (formatos,)
    fm = tuple(f.lower() for f in formatos)
    malos = [f for f in fm if f not in FORMATOS_VALIDOS]
    if malos:
        raise ValueError(f"Formato no soportado: {malos}. Validos: {FORMATOS_VALIDOS}")
    return fm


def exportar_resultados(resultados: list[dict], nombre: str, directorio: str | None = None,
                        formatos=("csv", "json"), descargar_colab: bool = True, notas: dict | None = None) -> dict:
    """Escribe ``nombre``.csv (tabla plana + ``_zonas.csv``) y ``nombre``.json (estructura completa).

    El JSON lleva ``notas`` (por defecto ``NOTAS``: capital por ticker, alcance del backtest, referencia de la
    distancia, cobertura) y el CSV las columnas ``capital_alcance`` / ``bt_alcance``.
    Devuelve {formato: ruta}. En Colab intenta ``files.download`` (silencioso si falla).
    """
    fm = _formatos(formatos)
    out = directorio_salida(directorio)
    rutas = {}
    if "csv" in fm:
        ruta = os.path.join(out, f"{nombre}.csv")
        pd.DataFrame([aplanar_resultado(r) for r in resultados]).to_csv(ruta, index=False, encoding="utf-8")
        rutas["csv"] = ruta
        zonas = [{"ticker": r.get("ticker"), **z, "dist_ref": "r_sup"} for r in resultados for z in (r.get("zonas") or [])]
        if zonas:
            rz = os.path.join(out, f"{nombre}_zonas.csv")
            pd.DataFrame(a_serializable(zonas)).to_csv(rz, index=False, encoding="utf-8")
            rutas["csv_zonas"] = rz
    if "json" in fm:
        ruta = os.path.join(out, f"{nombre}.json")
        payload = {"generado": datetime.now().isoformat(timespec="seconds"), "n": len(resultados),
                   "notas": a_serializable(NOTAS if notas is None else notas),
                   "resultados": a_serializable(resultados)}
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        rutas["json"] = ruta
    if descargar_colab and es_colab():  # pragma: no cover - solo Colab
        try:
            from google.colab import files
            for r in rutas.values():
                files.download(r)
        except Exception:  # noqa: BLE001
            pass
    return rutas
