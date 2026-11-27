"""Flujo de operaciones del Congreso (STOCK Act) vía kadoa-org/congress-trading-monitor."""
import glob
import io
import json
import logging
import os
import shutil
import time
import urllib.request
import zipfile

REPO_URL = "https://github.com/kadoa-org/congress-trading-monitor"
ZIP_URL = REPO_URL + "/archive/refs/heads/main.zip"
REPO_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "congress-trading-monitor")

logger = logging.getLogger(__name__)


def _no_disponible(motivo):
    """Resultado cuando NO se pudo consultar el flujo del Congreso (fallo o datos ilegibles).

    ``pts=None`` y ``disponible=False``: el consumidor debe excluir el componente (no es 0 ni un valor neutro).
    """
    return {"pts": None, "disponible": False, "motivo": motivo, "compras": 0, "ventas": 0,
            "resumen": f"Flujo Congreso (STOCK Act): No disponible ({motivo})", "ultimas_tx": []}


def _resumir_error(exc, maxlen=100):
    msg = " ".join(str(exc).split())
    return (f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__)[:maxlen]


# Nombres de dispositivo reservados en Windows (p. ej. ticker/CON.json no se puede crear).
_RESERVADOS_WIN = ["CON", "PRN", "AUX", "NUL"] + [f"COM{i}" for i in range(1, 10)] + [f"LPT{i}" for i in range(1, 10)]


def sincronizar_repo(repo_dir=REPO_DIR_DEFAULT, max_edad_horas=24):
    """Descarga el ZIP del repo (solo public/data) si el caché no existe o es viejo."""
    marca = os.path.join(repo_dir, ".sync")
    if os.path.exists(marca) and (time.time() - os.path.getmtime(marca)) < max_edad_horas * 3600:
        return
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "niveles-screener"})
    with urllib.request.urlopen(req, timeout=120) as r:
        contenido = r.read()
    destino_tmp = repo_dir + ".tmp"
    shutil.rmtree(destino_tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        for nombre in z.namelist():
            partes = nombre.split("/", 1)
            if len(partes) < 2 or not partes[1].startswith("public/data/") or nombre.endswith("/"):
                continue
            rel = partes[1]
            if os.path.splitext(os.path.basename(rel))[0].upper() in _RESERVADOS_WIN:
                continue  # inválidos en Windows (p. ej. ticker/CON.json)
            ruta = os.path.join(destino_tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, "wb") as f:
                f.write(z.read(nombre))
    shutil.rmtree(repo_dir, ignore_errors=True)
    os.replace(destino_tmp, repo_dir)
    with open(marca, "w") as f:
        f.write(str(time.time()))


def consultar_flujo_congreso_stock_act(ticker_symbol, repo_dir=REPO_DIR_DEFAULT):
    """
    Flujo de compras/ventas de miembros del Congreso (STOCK Act) para `ticker_symbol`.

    Devuelve SIEMPRE un dict con las claves: ``pts`` (0-15 o None), ``disponible`` (bool), ``motivo``
    (str si ``disponible=False``, None si True), ``compras``, ``ventas``, ``resumen`` (str) y
    ``ultimas_tx`` (lista de hasta 4 dicts). Nunca lanza.

    * Con datos y operaciones: ``disponible=True``, ``pts = 15 * fraccion de compras`` entre las 15 ultimas
      (siempre en [0, 15]). ``politico`` sale de ``filer_name`` y ``monto`` de ``amount_range_label``. El
      de-duplicado solo colapsa la misma operacion (mismo ``id``) presente en varios JSON; antes se fusionaban
      operaciones distintas del mismo dia (AAPL: 345 compras/336 ventas -> 392/422 reales), por lo que ``compras``,
      ``ventas`` y la fraccion de las 15 ultimas pueden cambiar respecto a versiones previas.
    * Consultado correctamente y SIN operaciones del ticker: ``disponible=True``, ``pts=0.0`` (cero real).
    * No se pudo consultar (fallo de descarga, carpeta ausente, JSON ilegibles o esquema irreconocible):
      ``disponible=False``, ``pts=None`` y ``motivo``. No es 0 ni un valor neutro: el consumidor debe
      excluir el componente (no contarlo como cobertura).
    """
    ticker_obj = str(ticker_symbol or "").strip().upper()
    if not ticker_obj:
        return _no_disponible("ticker vacio")
    try:
        sincronizar_repo(repo_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("STOCK Act: no se pudo sincronizar el repositorio (%s)", _resumir_error(e))
        return _no_disponible(f"no se pudo sincronizar el repositorio ({_resumir_error(e)})")

    data_dir = os.path.join(repo_dir, "public", "data")
    if not os.path.exists(data_dir):
        return _no_disponible("carpeta public/data no encontrada")

    try:
        return _evaluar_flujo(ticker_obj, data_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("STOCK Act: error inesperado procesando %s (%s)", ticker_obj, _resumir_error(e))
        return _no_disponible(f"error procesando datos ({_resumir_error(e)})")


def _evaluar_flujo(ticker_obj, data_dir):
    """Lee los JSON de `data_dir` y arma el resultado (ver ``consultar_flujo_congreso_stock_act``)."""
    operaciones = []
    archivos_ok, ilegibles, registros_vistos = 0, 0, 0
    for ruta in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        nombre = os.path.basename(ruta).lower()
        if any(s in nombre for s in ("meta", "manifest", "schema")):
            continue
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = json.load(f)
        except Exception as e:  # noqa: BLE001
            ilegibles += 1
            logger.warning("STOCK Act: JSON ilegible %s (%s)", os.path.basename(ruta), _resumir_error(e))
            continue
        archivos_ok += 1

        registros = []
        if isinstance(contenido, list):
            registros = contenido
        elif isinstance(contenido, dict):
            for clave in ("trades", "transactions", "filings", "data", "items", "records"):
                if isinstance(contenido.get(clave), list):
                    registros = contenido[clave]
                    break
            if not registros and isinstance(contenido.get(ticker_obj), list):
                registros = contenido[ticker_obj]

        for reg in registros:
            if not isinstance(reg, dict):
                continue
            tk = str(reg.get("ticker") or reg.get("symbol") or reg.get("asset_ticker")
                     or reg.get("stock") or "").strip().upper()
            if tk:
                registros_vistos += 1
            if tk != ticker_obj:
                continue
            tipo = str(reg.get("type") or reg.get("tx_type") or reg.get("transaction")
                       or reg.get("transaction_type") or reg.get("action") or "").lower()
            if any(w in tipo for w in ("purchase", "buy", "compra")):
                lado = "COMPRA"
            elif any(w in tipo for w in ("sale", "sell", "venta")):
                lado = "VENTA"
            else:
                continue
            operaciones.append({
                "fecha": str(reg.get("transaction_date") or reg.get("tx_date") or reg.get("date")
                             or reg.get("disclosure_date") or "S/F")[:10],
                "politico": str(reg.get("filer_name") or reg.get("filer") or reg.get("politician")
                                or reg.get("representative") or reg.get("name") or "Funcionario"),
                "camara": str(reg.get("chamber") or reg.get("source") or reg.get("branch") or ""),
                "lado": lado,
                "monto": str(reg.get("amount_range_label") or reg.get("amount") or reg.get("range")
                             or reg.get("notional") or ""),
                "_id": str(reg.get("id") or ""),
                "_filer_id": str(reg.get("filer_id") or ""),
                "_owner": str(reg.get("owner") or ""),
            })

    vistos, unicas = set(), []
    for op in operaciones:
        # Duplicado verdadero = misma operacion repetida en varios JSON (ticker/, filer/, trades.json): mismo `id`.
        # Sin `id`: filer (id o nombre), fecha, lado, monto y titular (owner). Operaciones distintas del mismo
        # dia (otro id/titular/monto) se conservan.
        k = op["_id"] or (op["_filer_id"] or op["politico"], op["fecha"], op["lado"], op["monto"], op["_owner"])
        if k not in vistos:
            vistos.add(k)
            unicas.append({c: v for c, v in op.items() if not c.startswith("_")})
    unicas.sort(key=lambda x: x["fecha"], reverse=True)

    compras = sum(1 for o in unicas if o["lado"] == "COMPRA")
    ventas = sum(1 for o in unicas if o["lado"] == "VENTA")
    if archivos_ok == 0:
        return _no_disponible("ningun archivo JSON legible en public/data")
    if registros_vistos == 0:
        return _no_disponible("ningun registro con ticker reconocible (esquema del repositorio distinto al esperado)")
    if compras + ventas == 0:
        if ilegibles:
            # Sin evidencia de operaciones, pero parte de la fuente no se pudo leer: no se afirma "cero".
            return _no_disponible(f"{ilegibles} archivo(s) JSON ilegible(s) y sin operaciones en el resto")
        return {"pts": 0.0, "disponible": True, "motivo": None, "compras": 0, "ventas": 0,
                "resumen": "Flujo Congreso (STOCK Act): Sin operaciones registradas para este ticker (consultado; 0 pts)",
                "ultimas_tx": []}

    recientes = unicas[:15]
    ratio = sum(1 for o in recientes if o["lado"] == "COMPRA") / len(recientes)
    return {
        "pts": round(ratio * 15.0, 1),
        "disponible": True,
        "motivo": None,
        "compras": compras,
        "ventas": ventas,
        "resumen": (f"Flujo Congreso (STOCK Act): {compras} Compras vs {ventas} Ventas totales "
                    f"({ratio * 100:.0f}% compras en últimas {len(recientes)} ops)"),
        "ultimas_tx": unicas[:4],
    }
