# Niveles de Compra Institucional

[![Descargar para Windows](https://img.shields.io/badge/Descargar_para-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/intidie/niveles-compra-institucional/releases/latest)

Screener cuantitativo de niveles de compra de largo plazo anclados a estructuras acumuladas (perfil de volumen, VWAP
anclado, order blocks, swing lows, valoración histórica/DCF) y auditados con backtesting de órdenes límite y datos
institucionales de la SEC (13F, Form 4).

## Descarga para Windows (1 clic)

1. Pulsa **Descargar para Windows** (arriba) y baja `NivelesInstitucionales-Windows.zip`.
2. Descomprime el zip en una carpeta cualquiera.
3. Ejecuta `NivelesInstitucionales.exe`. Se abre una ventana propia con la aplicación.

Aviso honesto: el ejecutable **no está firmado digitalmente**, así que Windows SmartScreen puede mostrar "Windows protegió
su PC". Pulsa "Más información" y luego "Ejecutar de todos modos" si confías en el origen (este repositorio). Tu
watchlist personalizada se guarda en `%APPDATA%\NivelesInstitucionales`.

> No es asesoría financiera. El ranking organiza candidatos; no predice retornos (ver la pestaña **Modelo**).

## Uso local

Abre la aplicación y, en la pestaña **Análisis**, elige un ticker (por ejemplo `NVDA`), el capital por ticker y pulsa
**Recalcular**. El botón de estrella guarda el ticker en **tu watchlist**; en la pestaña **Watchlist** el botón Escanear
calcula solo las acciones de tu lista y las ordena por puntaje. La pestaña **Modelo** explica la metodología.

## Ejecución desde código

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python app_ui.py                  # interfaz (ventana nativa si hay pywebview; si no, navegador)
python main.py --ticker NVDA --capital 1000
python main.py --watchlist watchlist.csv --top 10
```

Variables útiles: `APP_UI_PORT` (puerto local, 8080 por defecto), `APP_UI_NATIVE=0` (forzar navegador),
`APP_UI_SHOW=0` (no abrir el navegador), `NIVELES_DATA_DIR` (carpeta de datos de usuario) y `EDGAR_IDENTITY` (también se acepta `SEC_EDGAR_IDENTITY`)
(identidad que la SEC exige en el User-Agent, formato `"Nombre Apellido correo@dominio.com"`; si no la defines se usa un
valor genérico de ejemplo).

`watchlist.csv` del repositorio es solo una lista de ejemplo (de solo lectura): la UI usa tu lista personalizada y puede
importar esa de ejemplo con "Cargar lista de ejemplo".

## Despliegue en Render (nube)

1. Crea un **Web Service** desde este repositorio.
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `python app_ui.py`
4. Variable de entorno `EDGAR_IDENTITY` (también se acepta `SEC_EDGAR_IDENTITY`) con tu identidad para la SEC (nombre y correo de contacto; no la subas al
   repositorio). Es la variable que el código lee hoy en `src/data/sec_edgar.py`.
5. **Health Check Path**: `/health` (responde `{"status": "ok"}`). Puedes añadir un monitor de UptimeRobot a esa URL cada
   5 minutos para evitar que el servicio se duerma.

Render define `PORT` automáticamente: con ella la app escucha en `0.0.0.0`, no abre ventana nativa, analiza de forma
secuencial en un solo hilo, conserva solo 3 tickers en memoria y viene con el backtest del escaneo desactivado.

Advertencias honestas:
- El plan gratuito tiene **512 MB de RAM**: escaneos grandes o con backtest pueden agotar la memoria. Escanea pocos
  tickers a la vez.
- El servicio gratuito **se duerme tras un rato de inactividad** (la primera visita tarda en despertar).
- El disco es **efímero**: la watchlist personalizada **no persiste** en la nube (se pierde al reiniciar o redesplegar).

## Tests

```bash
python -m pytest -q -W default
```

Los tests no usan red ni abren servidor y aíslan el directorio de datos de usuario en una carpeta temporal.

## Crear el ejecutable de Windows

```bash
pip install -r requirements-build.txt
pyinstaller build.spec --noconfirm   # genera dist/NivelesInstitucionales/NivelesInstitucionales.exe (modo onedir)
```

Publicar un release: crea un tag `vX.Y.Z` y súbelo; el workflow `.github/workflows/release-windows.yml` compila, ejecuta
una prueba de humo (`/health`) y adjunta `NivelesInstitucionales-Windows.zip` al release. También puede lanzarse a mano
(`workflow_dispatch`), en cuyo caso el zip queda como artefacto del workflow.
