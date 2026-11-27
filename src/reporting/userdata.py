"""Rutas de recursos/datos de usuario y watchlist personalizada (funciones puras, sin red ni NiceGUI).

- ``recurso(rel)``: ruta de un archivo EMPAQUETADO (solo lectura): raiz del repo o ``sys._MEIPASS`` si esta congelado.
- ``directorio_datos()``: donde se escribe lo del usuario (watchlist personalizada). Nunca dentro del paquete:
  ``NIVELES_DATA_DIR`` si existe; en la nube (variable PORT) un directorio temporal; en Windows
  ``%APPDATA%\\NivelesInstitucionales``; en el resto ``~/.local/share/NivelesInstitucionales``.
- Watchlist personalizada: CSV ``ticker,nota`` (mismo formato que ``watchlist.csv``), escritura atomica y tolerante a
  archivo corrupto. Todas las funciones reciben el directorio como parametro para poder testearlas con ``tmp_path``.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import tempfile

NOMBRE_APP = "NivelesInstitucionales"
ARCHIVO_WATCHLIST = "mi_watchlist.csv"
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
MAX_NOTA = 80


def en_nube() -> bool:
    """True si corre en un PaaS (Render y similares inyectan PORT)."""
    return "PORT" in os.environ


def congelado() -> bool:
    return bool(getattr(sys, "frozen", False))


def raiz_recursos() -> str:
    if congelado():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def recurso(*partes: str) -> str:
    return os.path.join(raiz_recursos(), *partes)


def directorio_datos() -> str:
    """Directorio de escritura del usuario (no se crea aqui; ``guardar`` lo crea al escribir)."""
    forzado = os.environ.get("NIVELES_DATA_DIR")
    if forzado:
        return forzado
    if en_nube():
        return os.path.join(tempfile.gettempdir(), NOMBRE_APP)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, NOMBRE_APP)


def normalizar_ticker(texto: str | None) -> str | None:
    t = (texto or "").strip().upper()
    return t if TICKER_RE.match(t) else None


def _limpiar_nota(nota: str | None) -> str:
    return " ".join(str(nota or "").replace(",", ";").split())[:MAX_NOTA]


def normalizar(items) -> list[dict]:
    """Lista de ``{"ticker", "nota"}`` validos, en mayusculas y sin duplicados (conserva el orden)."""
    out, vistos = [], set()
    for it in items or []:
        if isinstance(it, str):
            it = {"ticker": it}
        t = normalizar_ticker((it or {}).get("ticker"))
        if t and t not in vistos:
            vistos.add(t)
            out.append({"ticker": t, "nota": _limpiar_nota((it or {}).get("nota"))})
    return out


def ruta_watchlist(directorio: str | None = None) -> str:
    return os.path.join(directorio or directorio_datos(), ARCHIVO_WATCHLIST)


def leer_csv(ruta: str) -> list[dict]:
    """Lee un CSV ``ticker,nota`` (acepta '#' de comentario y CSV sin cabecera). [] si falta o esta corrupto."""
    try:
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            filas = [r for r in csv.reader(f) if r and r[0].strip() and not r[0].strip().startswith("#")]
    except (OSError, UnicodeDecodeError, csv.Error, ValueError):
        return []
    if filas and filas[0][0].strip().lower() == "ticker":
        filas = filas[1:]
    return normalizar([{"ticker": r[0], "nota": r[1] if len(r) > 1 else ""} for r in filas])


def existe(directorio: str | None = None) -> bool:
    return os.path.isfile(ruta_watchlist(directorio))


def leer(directorio: str | None = None) -> list[dict]:
    """Watchlist personalizada ([] si no existe o esta corrupta)."""
    return leer_csv(ruta_watchlist(directorio))


def guardar(items, directorio: str | None = None) -> str:
    """Escritura atomica (archivo temporal + ``os.replace``). Devuelve la ruta escrita."""
    d = directorio or directorio_datos()
    os.makedirs(d, exist_ok=True)
    ruta = os.path.join(d, ARCHIVO_WATCHLIST)
    fd, tmp = tempfile.mkstemp(prefix=".wl_", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "nota"])
            for it in normalizar(items):
                w.writerow([it["ticker"], it["nota"]])
        os.replace(tmp, ruta)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return ruta


def anadir(items, ticker: str | None, nota: str | None = None) -> tuple[list[dict], bool]:
    """(lista nueva, anadido). No anade invalidos ni duplicados."""
    lista = normalizar(items)
    t = normalizar_ticker(ticker)
    if t is None or any(i["ticker"] == t for i in lista):
        return lista, False
    return lista + [{"ticker": t, "nota": _limpiar_nota(nota)}], True


def quitar(items, ticker: str | None) -> tuple[list[dict], bool]:
    lista = normalizar(items)
    t = normalizar_ticker(ticker)
    nueva = [i for i in lista if i["ticker"] != t]
    return nueva, len(nueva) != len(lista)


def fusionar(items, otros) -> tuple[list[dict], int]:
    """Anade ``otros`` a ``items`` sin duplicar. Devuelve (lista, cuantos nuevos)."""
    lista = normalizar(items)
    nueva = normalizar(lista + normalizar(otros))
    return nueva, len(nueva) - len(lista)


def tickers(items) -> list[str]:
    return [i["ticker"] for i in normalizar(items)]
