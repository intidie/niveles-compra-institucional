"""Flujo de operaciones del Congreso (STOCK Act) vía kadoa-org/congress-trading-monitor."""
import glob
import io
import json
import os
import shutil
import time
import urllib.request
import zipfile

REPO_URL = "https://github.com/kadoa-org/congress-trading-monitor"
ZIP_URL = REPO_URL + "/archive/refs/heads/main.zip"
REPO_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "congress-trading-monitor")


def _neutro(resumen):
    return {"pts": 7.5, "compras": 0, "ventas": 0, "resumen": resumen, "ultimas_tx": []}


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
    ticker_obj = ticker_symbol.strip().upper()
    try:
        sincronizar_repo(repo_dir)
    except Exception as e:
        return _neutro(f"Flujo Congreso (STOCK Act): No disponible ({e})")

    data_dir = os.path.join(repo_dir, "public", "data")
    if not os.path.exists(data_dir):
        return _neutro("Flujo Congreso (STOCK Act): Carpeta public/data no encontrada")

    operaciones = []
    for ruta in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        nombre = os.path.basename(ruta).lower()
        if any(s in nombre for s in ("meta", "manifest", "schema")):
            continue
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = json.load(f)
        except Exception:
            continue

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
                "politico": str(reg.get("filer") or reg.get("politician") or reg.get("representative")
                                or reg.get("name") or "Funcionario"),
                "camara": str(reg.get("chamber") or reg.get("source") or reg.get("branch") or ""),
                "lado": lado,
                "monto": str(reg.get("amount") or reg.get("range") or reg.get("notional") or ""),
            })

    vistos, unicas = set(), []
    for op in operaciones:
        k = (op["fecha"], op["politico"], op["lado"], op["monto"])
        if k not in vistos:
            vistos.add(k)
            unicas.append(op)
    unicas.sort(key=lambda x: x["fecha"], reverse=True)

    compras = sum(1 for o in unicas if o["lado"] == "COMPRA")
    ventas = sum(1 for o in unicas if o["lado"] == "VENTA")
    if compras + ventas == 0:
        return _neutro("Flujo Congreso (STOCK Act): Sin operaciones registradas para este ticker (Neutral)")

    recientes = unicas[:15]
    ratio = sum(1 for o in recientes if o["lado"] == "COMPRA") / len(recientes)
    return {
        "pts": round(ratio * 15.0, 1),
        "compras": compras,
        "ventas": ventas,
        "resumen": (f"Flujo Congreso (STOCK Act): {compras} Compras vs {ventas} Ventas totales "
                    f"({ratio * 100:.0f}% compras en últimas {len(recientes)} ops)"),
        "ultimas_tx": unicas[:4],
    }
