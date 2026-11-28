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

La pestaña **Valor Justo** calcula 15 modelos de valoración (DCF a 5 y 10 años sobre FCF, EBITDA e ingresos; P/E,
EV/EBITDA, EV/Ingresos, P/B, P/FCF; DDM; EPV; Graham; Lynch) con los estados financieros de SEC EDGAR, en **dos
escenarios: Medio y Pesimista** (no hay optimista). Muestra 4 tarjetas (precio, valor justo Medio, piso fundamental
Pesimista y diagnóstico de valoración con su incertidumbre), un gráfico histórico con bandas de ±10/20/30 % y, en
acordeones, las barras por modelo, la activación de modelos y los sliders de WACC, g y margen de seguridad (se recalculan
al instante, sin consultar nada). El interruptor **Mostrar Valor Justo** (activado por defecto) dibuja en Análisis solo la línea del Escenario
Medio. Si el ticker aún no tiene datos, el botón **Calcular valor justo de <TICKER>** de esa pestaña lo obtiene en segundos
(sin backtest ni consultas Form 4 / 13F); **Recalcular** hace el análisis completo.

Lectura honesta: ambos escenarios son **supuestos fijados de antemano, no predicciones**. Con el tope de crecimiento
de 42 % durante 3 años, márgenes de hasta 55 % y g de 3,5 %, el Medio puede dar valores muy por encima del precio en
negocios de hipercrecimiento (NVDA: Medio ~220 USD frente a ~149 USD de precio). El Pesimista es un estrés duro
(WACC mínimo 12,5 %), no un «peor caso realista» (en KO queda ~61 % bajo el precio). La reconstrucción histórica es
causal salvo el punto final (ancla); Rf, ERP, beta, sector y sliders no son point-in-time y la proyección a 12 meses es
ilustrativa. Detalle y cifras en la pestaña **Modelo** (sección 10). Es una implementación propia «estilo InvestingPro»
(su metodología es propietaria y no se replica): no es un precio objetivo y es independiente de los niveles técnicos.

### Local vs presentación en Render

| | Local (`python app_ui.py` o el .exe) | Presentación en Render (variable `PORT`) |
|---|---|---|
| Historial | 10 años (`periodo=10y`), serie de valor justo de 10 años | 3 años (`periodo=3y`), serie de 3 años |
| Caché | SQLite en disco (entradas, `info`, ^TNX, serie) | Sin caché en disco; `gc.collect()` tras cada consulta |
| Watchlist | Escaneo completo, secuencial | Un escaneo a la vez y máximo 5 tickers por escaneo |
| Ventana | Nativa (pywebview) o navegador | Navegador |

La nube es un visor ligero para presentar (512 MB); el uso serio es local. Los gráficos de Valor Justo usan Apache ECharts
(incluido en NiceGUI) y el de Análisis TradingView Lightweight Charts: ambos se sirven sin CDN, funcionan sin conexión.

### Acceso directo en el Escritorio (Windows)

Para abrir la app con doble clic y sin ventana de consola:

```powershell
powershell -ExecutionPolicy Bypass -File .\crear_acceso_directo.ps1
```

Crea `Niveles Institucionales.lnk` en tu Escritorio (la ruta real de `[Environment]::GetFolderPath('Desktop')`, que
respeta OneDrive y otras redirecciones). Ejecuta `.venv\Scripts\pythonw.exe app_ui.py` desde la carpeta del repositorio
(la deduce de la ubicación del script), con ventana normal (`WindowStyle = 1`; `pythonw` no abre consola) y usando el
icono de `pythonw.exe`. Abre una ventana de escritorio nativa si `pywebview` está instalado en `.venv` (viene en
`requirements.txt`); si no, abre la app en el navegador en http://127.0.0.1:8080. Requiere haber
creado el entorno virtual (ver abajo); si falta `pythonw.exe` o `app_ui.py` el script se detiene con un mensaje claro.
Como `pythonw` no tiene consola, `app_ui.py` redirige la salida estándar a un dispositivo nulo al arrancar.

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
