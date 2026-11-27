# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: ``pyinstaller build.spec --noconfirm`` -> dist/NivelesInstitucionales/NivelesInstitucionales.exe

Modo ONEDIR (carpeta; se distribuye comprimida en un .zip) y no onefile, por robustez con NiceGUI nativo:
  - NiceGUI (modo nativo) relanza el ejecutable con multiprocessing (spawn) para abrir la ventana pywebview; en onefile
    cada proceso hijo pasa por el bootloader y la extraccion a %TEMP% (arranque lento, mas falsos positivos de antivirus
    y riesgo de carreras al limpiar el directorio _MEI).
  - onedir arranca mas rapido, no re-extrae nada y deja los recursos en una carpeta estable (_internal).
Las escrituras del usuario (watchlist personalizada) van a %APPDATA%\\NivelesInstitucionales, nunca al paquete.
"""
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

RAIZ = os.path.abspath(SPECPATH)  # noqa: F821  (SPECPATH lo inyecta PyInstaller)

datas = [
    (os.path.join(RAIZ, "watchlist.csv"), "."),
    (os.path.join(RAIZ, "docs", "METODOLOGIA.md"), "docs"),
    (os.path.join(RAIZ, "docs", "GUIA_POST_RESULTADOS.md"), "docs"),
    (os.path.join(RAIZ, "src", "validation", "pesos_calibrados.json"), os.path.join("src", "validation")),
]
binaries = []
hiddenimports = ["main", "app_ui", "multiprocessing", "uvicorn.logging", "uvicorn.loops.auto",
                 "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]
# src/ se importa de forma lazy (p. ej. validacion/backtest dentro de main.analizar_ticker): todos sus modulos.
hiddenimports += collect_submodules("src")

# Paquetes con datos/modulos que el analisis estatico no ve (iconos y JS de nicegui, plantillas de plotly, JS de
# pywebview, referencias de edgartools, certificados y libs nativas de curl_cffi).
for paquete in ("nicegui", "edgar", "smartmoneyconcepts", "webview", "yfinance", "curl_cffi", "certifi"):
    try:
        d, b, h = collect_all(paquete)
    except Exception:  # noqa: BLE001  (paquete opcional no instalado, p. ej. webview en Linux)
        continue
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files("plotly")  # plotly.min.js y templates (package_data)
hiddenimports += collect_submodules("plotly.graph_objs") + collect_submodules("plotly.validators")

a = Analysis(
    [os.path.join(RAIZ, "app_ui.py")],
    pathex=[RAIZ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "IPython", "notebook"],
    # scipy.stats falla al importar desde el PYZ ("NameError: name 'obj'..." en _distn_infrastructure: usa dir() sobre
    # nombres de modulo y rompe con el bytecode precompilado). Se recoge como .py para que Python lo compile al importar.
    module_collection_mode={"scipy": "py"},
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="NivelesInstitucionales",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # app de ventana; main.py/app_ui.py redirigen stdout/stderr a devnull si no hay consola
)
coll = COLLECT(  # noqa: F821
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="NivelesInstitucionales",
)
