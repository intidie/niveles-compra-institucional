"""Valor justo por bateria de 15 modelos (DCF, multiplos, dividendos, valor) en dos ESCENARIOS: Medio y Pesimista.

AVISO HONESTO: la metodologia de InvestingPro es propietaria. Esto es una implementacion PROPIA "estilo" InvestingPro
(misma familia de modelos y misma idea de consolidar en un unico valor justo), NO replica sus cifras. Es un valor
estimado por modelos con supuestos explicitos; no es un precio objetivo, ni una prediccion, ni asesoria financiera.
Los parametros de los dos escenarios (WACC, g∞, topes, compresion de multiplos...) son SUPUESTOS DE ESCENARIO fijados
de antemano, no predicciones. Solo existen "medio" y "pesimista". Es ADITIVO: no modifica ni sustituye los niveles
tecnicos de compra ni los pisos de ``valuation_floors``.

Flujo
-----
1. ``extraer_entradas(ticker, tk_obj, df, info, facts=None, modo=None, anios=None, rf=None)`` arma UNA vez un
   ``EntradasValoracion`` (liviano, serializable con ``to_dict``) reutilizando los helpers XBRL de ``valuation_floors``.
   ``modo="local"`` (por defecto sin variable PORT): hasta ``ANIOS_LOCAL`` = 10 anios fiscales + TTM y cache SQLite en disco
   (``src/reporting/cache_local.py``). ``modo="nube"`` (variable PORT): ``ANIOS_NUBE`` = 3 anios, sin cache en disco y
   ``gc.collect()`` al terminar. Consulta ^TNX para la Rf (cache 24 h, respaldo 4.25 %). Dato faltante = None, nunca 0.
2. ``calcular_escenarios(entradas, params, modelos_activos, margen_seguridad)`` (funcion pura, sin red: mover sliders
   es instantaneo) devuelve ``{"medio", "pesimista", "modelos", "supuestos", ...}``. ``calcular_modelos(e, p, escenario)``
   y ``agregar_valor_justo`` siguen disponibles (escenario Medio por defecto).
3. ``serie_valor_justo_historica(entradas, df, ...)``: curva historica del Valor Justo Medio con 6 bandas (+-10/20/30 %)
   y proyeccion ilustrativa de 12 meses (ver su docstring: PIT, suavizado causal, ancla y limites).

Parametros por escenario (constantes con nombre al inicio del modulo)
---------------------------------------------------------------------
* Rf = ^TNX (/100, valido 0.5-12 %; respaldo 4.25 %); ERP 4.5 %; beta de Blume = 2/3*beta_raw + 1/3;
  ke = Rf + beta*ERP; kd despues de impuestos = kd*(1-t); pesos E/(D+E) y D/(D+E) reales.
* WACC Medio = clip(WACC, 8.0 %, 11.5 %). WACC Pesimista = max(WACC Medio vigente + 200 pb, 12.5 %).
* g∞: 3.5 % Medio / 2.5 % Pesimista (el Pesimista nunca supera al Medio vigente). El DCF exige WACC - g∞ >= 2.5 pp.
* DCF explicito de 10 anios (``dcf_10y_*``; los ids ``dcf_5y_*`` son la variante corta del MISMO motor con el valor
  terminal en el anio 5) en 2 etapas con convencion de MITAD DE ANIO ``DF_t = (1+WACC)^-(t-0.5)``:
  etapa 1 (anios 1-3) crece ``FRACCION_CRECIMIENTO`` = 65 % del crecimiento fundamental, topado a 42 % (Medio) / 15 %
  (Pesimista); etapa 2 (anios 4..N) decae linealmente hasta g∞ y los margenes convergen linealmente a la mediana
  recortada historica de 3-5 anios (FCF/Ingresos en el DCF por ingresos; FCFF/EBITDA en el de EBITDA, con tope de
  margen 55 % / 30 % de ingresos). El DCF de FCF crece el FCF con la misma trayectoria (sin tope de margen). Flujo
  ciclico => base = media de 5 anios y crecimiento = g∞. FCFF = FCF + intereses*(1-t).
  VALOR TERMINAL: ``TV_N = FCF_N*(1+g∞)/(WACC-g∞)`` se descuenta con el MISMO factor del ultimo flujo explicito,
  ``DF_{N-0.5}`` (9.5 con N = 10): como los flujos de la perpetuidad llegan a mitad de cada anio, su valor en t = N es
  ``TV_N*(1+WACC)^0.5``; descontar a t = N subvaluaria el terminal en un factor (1+WACC)^0.5 (eso solo es correcto con
  un multiplo de salida). Valor por accion = (VP + caja - deuda) / acciones diluidas.
* Crecimiento fundamental = mezcla renormalizada (sobre lo disponible) de ``PESO_CAGR`` 50 % (CAGR winsorizado a 3Y y 5Y:
  variaciones anuales recortadas a [-50 %, +100 %] y, con >= 4, winsorizadas a los percentiles 10/90 y COMPUESTAS
  geometricamente), ``PESO_MOMENTUM`` 20 % (variacion YoY del TTM) y ``PESO_FORWARD`` 30 % (yfinance ``revenueGrowth`` para
  ingresos; ``forwardEps/trailingEps - 1`` o ``earningsGrowth`` para beneficios). Sin datos no se inventa 0: g∞.
* Multiplos (P/E, EV/EBITDA, EV/Ingresos, P/B, P/FCF): MEDIANA RECORTADA de los multiplos propios (precio a +45 dias del
  cierre, PIT) acotada a ``RANGOS_MULTIPLOS``, mezclada 70 % propio + 30 % benchmark sectorial (``BENCHMARK_SECTORIAL``,
  APROXIMACION fijada a mano, fecha ``BENCHMARK_FECHA``; sin sector en la tabla solo el propio). Pesimista: -15 %.
* DDM (Gordon con ke del escenario), EPV (Greenwald: se EXCLUYE en el Medio si los ingresos crecen > 20 %; en el
  Pesimista se incluye como piso de crecimiento cero), Graham, Lynch (tope de crecimiento 25 % Medio / 15 % Pesimista).

Consolidacion (``agregar_valor_justo``): filtra por modelos activos -> descarta no aplicables o <= 0 -> IQR (>= 4 validos;
ancho minimo ``IQR_PISO_REL``*mediana) -> promedio IGUALANDO FAMILIAS (1/3 DCF + 1/3 Multiplos + 1/3 Valor/Dividendos;
se renormaliza si una familia no tiene modelos). CAMBIO respecto de la version anterior: antes era el promedio simple.
Promedio simple, mediana, CV e incertidumbre (Baja/Media/Alta, con degradacion por pocas familias) se conservan como
informacion complementaria. ``complementarias`` y ``dcf_inverso`` (escenario Medio) siguen disponibles.

Limitaciones (visibles en la UI): muy sensible a WACC y g∞; el valor terminal domina los DCF; datos TTM de SEC EDGAR con
rezago; no aplica bien a bancos/aseguradoras (EBITDA/FCF), ciclicas ni empresas sin beneficios; el techo de crecimiento
de 42 % y de margen de 55 % del Medio pueden dar valores varias veces el precio en negocios de hipercrecimiento.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import numpy as np
import pandas as pd

# --- Constantes documentadas (la UI las lee de aqui; no se hardcodean en otro sitio) ------------------------
CRECIMIENTO_TERMINAL_DEFECTO = 0.035   # g∞ del escenario Medio (supuesto de escenario fijado por el usuario; antes 2.5 %)
MARGEN_SEGURIDAD_DEFECTO = 0.25
RF_DEFECTO = 0.0425                    # Rf de respaldo si ^TNX no esta disponible o sale del rango valido (supuesto fijado: 4.25 %)
ERP_DEFECTO = 0.045                    # prima de riesgo de mercado (supuesto fijado: 4.5 %)
BETA_DEFECTO, BETA_MIN, BETA_MAX = 1.0, 0.7, 2.0   # betas bajas (defensivas) se acotan: la sensibilidad del DCF al WACC es enorme
WACC_MIN, WACC_MAX = 0.08, 0.115       # limites del WACC automatico del escenario Medio: [8.0 %, 11.5 %]
SPREAD_DEUDA_DEFECTO = 0.015           # kd = rf + spread si no hay gasto por intereses fiable
KD_MIN, KD_MAX = 0.02, 0.12            # rango razonable del costo de deuda implicito (intereses/deuda)
TASA_IMPOSITIVA_DEFECTO = 0.21         # tasa federal de EE. UU. si no hay tasa efectiva
TASA_IMPOSITIVA_MAX = 0.40
TASA_MIN = 0.15                        # piso de la tasa efectiva LTM usada en DCF EBITDA, EPV y WACC (PTON: 0.48 % no es perpetua)
CRECIMIENTO_TOPE = 0.15                # tope del crecimiento inicial proyectado
CRECIMIENTO_MIN = -0.05                # piso del crecimiento inicial proyectado
HAIRCUT = 0.5                          # se proyecta la mitad del CAGR historico (igual que valuation_floors)
MARGEN_FCF_TOPE = 0.30                 # tope del margen FCF objetivo en el DCF por ingresos
NWC_RATIO_MIN, NWC_RATIO_MAX = -0.3, 0.5
LYNCH_CRECIMIENTO_TOPE_PCT = 25.0      # tope del crecimiento (%) en Lynch
GRAHAM_FACTOR = 22.5
MIN_PUNTOS_MULTIPLO = 3                # instantaneas anuales minimas para fijar un multiplo objetivo historico
ANIOS_HISTORIA = 5                     # anios de historia (instantaneas k=1..5; 6 puntos anuales incluyendo LTM)
RANGOS_MULTIPLOS = {"pe": (6.0, 35.0), "ev_ebitda": (4.0, 25.0), "ev_revenue": (0.4, 12.0),
                    "pb": (0.8, 12.0), "p_fcf": (6.0, 40.0)}
DDM_RENDIMIENTO_MIN = 0.01             # dividendo/precio por debajo: el DDM no es informativo (dividendo simbolico)
PB_NO_INFORMATIVO = 25.0               # P/B historico mediano por encima: el valor contable no informa (recompras)
HORIZONTES = {"5y": 5, "10y": 10}
# Agregacion
MIN_MODELOS_IQR = 4                    # con menos validos no se filtran outliers
IQR_K = 1.5
MIN_MODELOS_CONFIABLE = 3              # menos modelos usados => incertidumbre Alta
CV_BAJO, CV_ALTO = 0.15, 0.35
NIVELES_INCERTIDUMBRE = ("Baja", "Media", "Alta")
REZAGO_PUBLICACION_DIAS = 45           # igual que valuation_floors (evita look-ahead en multiplos historicos)
VOL_FCF_ALTA = 0.30                    # CV (ddof=1) de la serie anual por encima del cual (y con ciclo) se normaliza el flujo
MIN_INVERSIONES_CICLO = 2              # cambios de direccion minimos para llamar "ciclico" a una serie (distingue de tendencia)
N_NORMALIZACION = 5                    # anios de la media que normaliza un flujo ciclico
MIN_ANIOS_FCF = 3                      # anios de FCF positivo exigidos (con historia) para el DCF de FCF
GRAHAM_PB_MAX = 15.0                   # Graham no aplica si precio/BVPS lo supera (valor contable no informativo)
IQR_PISO_REL = 0.05                    # ancho minimo de la valla IQR como fraccion de la mediana de los modelos
MARGEN_SEGURIDAD_MIN, MARGEN_SEGURIDAD_MAX = 0.0, 0.95
MIN_FAMILIAS_BAJA = 3                  # incertidumbre "Baja" exige modelos de >= 3 grupos de familia
GRUPOS_FAMILIA = {"DCF": "DCF", "Múltiplos": "Múltiplos", "Dividendos": "Valor / Dividendos", "Valor": "Valor / Dividendos"}
GRUPOS = ("DCF", "Múltiplos", "Valor / Dividendos")
WE_MIN_AJUSTE_KE = 0.25                # piso del peso del patrimonio al trasladar un WACC del usuario a ke
UMBRAL_AVISO_CAJA = 0.10               # aviso si la caja XBRL difiere > 10 % de la de yfinance (sin sobrescribir)
# Procesadores de pago con industria "Credit Services" donde EBITDA/FCF si son informativos (no son prestamistas)
PROCESADORES_PAGO = frozenset({"V", "MA", "PYPL", "FI", "FISV", "GPN", "FIS", "XYZ", "SQ", "TOST", "JKHY", "WU", "CPAY"})
DIF_MIN_WACC_G = 0.025                 # DCF/EPV/DDM exigen (tasa de descuento - g∞) >= 2.5 pp (el valor terminal explota si no)
PB_IMPLICITO_RANGO = (0.05, 100.0)     # precio / BVPS fuera de este rango => dato por accion de otra clase (BRK-B): se descarta
PE_IMPLICITO_RANGO = (0.05, 2000.0)    # precio / EPS (EPS > 0) fuera de este rango => EPS inconsistente con el precio
SUBCONTEO_INFO = 0.5                   # caja/deuda XBRL < 50 % del ``info`` de yfinance: se usa yfinance
MAX_ANTIGUEDAD_LTM_DIAS = 420          # un LTM mas viejo que esto se considera no disponible

# --- Escenarios (SUPUESTOS DE ESCENARIO fijados por el usuario; NO son predicciones) ---------------------------------
# Solo existen dos escenarios: "medio" y "pesimista". No hay escenario optimista y no debe aparecer en ninguna salida.
ESCENARIOS = ("medio", "pesimista")
ETIQUETAS_ESCENARIO = {"medio": "Medio", "pesimista": "Pesimista"}
ANIOS_LOCAL, ANIOS_NUBE = 10, 3        # anios fiscales de historia: local (sin recortes) vs nube (visor ligero)
RF_RANGO_PCT = (0.5, 12.0)             # rango valido (en %) de ^TNX; fuera de el se usa RF_DEFECTO
BLUME_PESO_RAW = 2.0 / 3.0             # beta de Blume: 2/3 * beta_raw + 1/3 (la beta tiende a 1)
BETA_RAW_MIN, BETA_RAW_MAX = 0.0, 4.0  # solo para descartar datos basura; la suavizacion de Blume ya modera extremos
WACC_PESIMISTA_SPREAD = 0.02           # WACC Pesimista = WACC Medio vigente + 200 pb ...
WACC_PESIMISTA_MIN = 0.125             # ... con minimo 12.5 % (estres)
N_ANIOS_DCF = 10                       # horizonte explicito del motor DCF principal
N_ANIOS_DCF_CORTO = 5                  # variante corta (ids dcf_5y_*): mismo motor, valor terminal en el anio 5
ANIOS_ETAPA1 = 3                       # etapa 1 = anios 1..3 (crecimiento alto); etapa 2 = anios 4..N (decaimiento + margenes)
FRACCION_CRECIMIENTO = 0.65            # crecimiento de la etapa 1 = 65 % del crecimiento fundamental compuesto
TOPE_CRECIMIENTO_ETAPA1 = {"medio": 0.42, "pesimista": 0.15}
G_TERMINAL = {"medio": 0.035, "pesimista": 0.025}          # g∞ por escenario
TOPE_MARGEN = {"medio": 0.55, "pesimista": 0.30}           # tope del margen objetivo (FCF/Ingresos y FCF/EBITDA)
DIF_MIN_WACC_G_DCF = DIF_MIN_WACC_G    # DCF exige WACC - g∞ >= 2.5 pp (si no, el DCF no aplica con motivo)
COMPRESION_MULTIPLOS = {"medio": 0.0, "pesimista": -0.15}  # castigo de compresion de multiplos objetivo
EPV_CRECIMIENTO_MAX = 0.20             # Medio: EPV (crecimiento cero) se excluye si los ingresos crecen mas de 20 % anual
LYNCH_TOPE_PCT_ESC = {"medio": LYNCH_CRECIMIENTO_TOPE_PCT, "pesimista": 15.0}   # tope del crecimiento (%) de Lynch
# Crecimiento fundamental = mezcla (renormalizada sobre lo disponible) de CAGR winsorizado, momentum YoY TTM y forward
PESO_CAGR, PESO_MOMENTUM, PESO_FORWARD = 0.50, 0.20, 0.30
CAGR_VENTANAS = (3, 5)                 # CAGR winsorizado a 3Y y 5Y (se promedian las ventanas disponibles)
WINSOR_PCT = (10.0, 90.0)              # percentiles de winsorizacion de las variaciones anuales (si hay >= 4 variaciones)
VARIACION_MIN, VARIACION_MAX = -0.50, 1.00   # recorte duro de cada variacion anual antes de componer
FORWARD_MIN, FORWARD_MAX = -0.50, 1.00       # recorte del crecimiento forward de yfinance
MARGEN_ANIOS_MEDIANA = (3, 5)          # mediana recortada de los ultimos 3 a 5 anios (margen / conversion historica)
RECORTE_MEDIANA = 0.20                 # fraccion recortada en cada extremo (con >= 5 puntos) en medianas recortadas
# Multiplos: mezcla del multiplo propio (mediana recortada) con un benchmark sectorial fijado a mano
PESO_MULTIPLO_PROPIO = 0.70            # el resto (0.30) es el benchmark sectorial si el sector esta en la tabla
BENCHMARK_FECHA = "2026-11"
# APROXIMACION fijada A MANO (no es un dato de mercado en vivo) de medianas sectoriales de largo plazo, fecha BENCHMARK_FECHA.
BENCHMARK_SECTORIAL: dict[str, dict[str, float]] = {
    "Technology": {"pe": 26.0, "ev_ebitda": 18.0, "ev_revenue": 6.0, "pb": 7.0, "p_fcf": 28.0},
    "Healthcare": {"pe": 21.0, "ev_ebitda": 14.0, "ev_revenue": 3.5, "pb": 4.0, "p_fcf": 22.0},
    "Financial Services": {"pe": 13.0, "ev_ebitda": 10.0, "ev_revenue": 2.5, "pb": 1.4, "p_fcf": 12.0},
    "Consumer Cyclical": {"pe": 21.0, "ev_ebitda": 13.0, "ev_revenue": 1.6, "pb": 4.5, "p_fcf": 22.0},
    "Consumer Defensive": {"pe": 21.0, "ev_ebitda": 14.0, "ev_revenue": 1.8, "pb": 4.5, "p_fcf": 21.0},
    "Energy": {"pe": 12.0, "ev_ebitda": 6.0, "ev_revenue": 1.2, "pb": 1.8, "p_fcf": 11.0},
    "Industrials": {"pe": 21.0, "ev_ebitda": 14.0, "ev_revenue": 2.2, "pb": 4.0, "p_fcf": 22.0},
    "Utilities": {"pe": 18.0, "ev_ebitda": 11.0, "ev_revenue": 2.6, "pb": 1.9, "p_fcf": 25.0},
    "Real Estate": {"pe": 30.0, "ev_ebitda": 17.0, "ev_revenue": 7.0, "pb": 1.8, "p_fcf": 20.0},
    "Basic Materials": {"pe": 16.0, "ev_ebitda": 9.0, "ev_revenue": 1.6, "pb": 2.0, "p_fcf": 16.0},
    "Communication Services": {"pe": 19.0, "ev_ebitda": 11.0, "ev_revenue": 2.8, "pb": 3.0, "p_fcf": 17.0},
}
# Cache en disco (solo local): version del codigo/esquema de las entradas; subirla invalida las entradas cacheadas
def _version_codigo(base: str = "fv-esc-1") -> str:
    """Version = base + hash del codigo fuente de este modulo: cualquier cambio del motor/extraccion invalida el cache."""
    try:
        import hashlib
        with open(__file__, "rb") as fh:
            return f"{base}-{hashlib.md5(fh.read()).hexdigest()[:8]}"
    except OSError:     # congelado (PyInstaller) o sin fuente: se usa solo la base
        return base


VERSION_CODIGO = _version_codigo()
# Serie historica
BANDAS_PCT = (30, 20, 10, -10, -20, -30)      # bandas simetricas alrededor del Valor Justo Medio
PROYECCION_DIAS = 365                  # proyeccion ilustrativa hacia adelante: 12 meses
PROYECCION_AMORTIGUACION = 0.5         # g_proy = 0.5*g_etapa1 + 0.5*g∞ (promedio de los DCF Medio), acotado a [0, 15 %]
PROYECCION_TOPE = 0.15
PASO_SERIE_DIAS = 7                    # muestreo de la serie devuelta (semanal)
ANCLA_SEPARACION_DIAS = 7              # un reporte publicado a menos de esto de la ultima vela lo reemplaza el ancla
VJ_ABSURDO_X = 10.0                    # un punto trimestral con valor justo > 10x o < 1/10 del precio de su fecha se descarta
SALTO_ACCIONES_X = 5.0                 # acciones de un trimestre que difieren > 5x de la mediana de sus vecinos: punto descartado
SALTO_ACCIONES = 0.30                  # una serie de acciones que se desvia > 30 % de sus vecinas (que coinciden) es un glitch
TRANSICION_DIAS = 45                   # suavizado CAUSAL: tras cada reporte el valor justo transita en 45 dias (smoothstep)

# (id, nombre, familia)
MODELOS: tuple[tuple[str, str, str], ...] = (
    ("dcf_5y_fcf", "DCF 5 años (FCF)", "DCF"),
    ("dcf_10y_fcf", "DCF 10 años (FCF)", "DCF"),
    ("dcf_5y_ebitda", "DCF 5 años (EBITDA)", "DCF"),
    ("dcf_10y_ebitda", "DCF 10 años (EBITDA)", "DCF"),
    ("dcf_5y_revenue", "DCF 5 años (Ingresos)", "DCF"),
    ("dcf_10y_revenue", "DCF 10 años (Ingresos)", "DCF"),
    ("pe", "P/E", "Múltiplos"),
    ("ev_ebitda", "EV/EBITDA", "Múltiplos"),
    ("ev_revenue", "EV/Ingresos", "Múltiplos"),
    ("pb", "P/B", "Múltiplos"),
    ("p_fcf", "P/FCF", "Múltiplos"),
    ("ddm", "DDM (dividendos)", "Dividendos"),
    ("epv", "EPV (Greenwald)", "Valor"),
    ("graham", "Graham", "Valor"),
    ("lynch", "Lynch (PEG=1)", "Valor"),
)
IDS_MODELOS = tuple(m[0] for m in MODELOS)
FAMILIAS = ("DCF", "Múltiplos", "Dividendos", "Valor")
_META = {m[0]: m for m in MODELOS}
# Modelos basados en EBITDA / FCF / flujos: no aplican a bancos y aseguradoras.
MODELOS_NO_FINANCIERAS = frozenset(i for i in IDS_MODELOS if i.startswith("dcf_") or i in
                                   ("ev_ebitda", "ev_revenue", "p_fcf", "epv"))
_RE_FINANCIERA = re.compile(r"bank|insurance|mortgage|capital markets|financial conglomerates|savings|thrift", re.I)
_RE_CREDITO = re.compile(r"credit services", re.I)   # prestamistas de tarjetas/consumo (COF, AXP, SYF) salvo PROCESADORES_PAGO

AVISO = ("Valor estimado por modelos con supuestos explícitos: no es un precio objetivo, ni una predicción, ni "
         "asesoría financiera. Implementación propia «estilo InvestingPro»: la metodología de InvestingPro es "
         "propietaria y esta herramienta no replica sus cifras.")


# --- Tipos ----------------------------------------------------------------------------------------------------
@dataclass
class EntradasValoracion:
    """Insumos ya extraidos (livianos, sin DataFrames). Todo faltante es None (nunca 0 silencioso).

    Importes en USD; ``hist_*`` son valores anuales LTM (de mas antiguo a mas reciente, el ultimo = LTM actual).
    ``multiplos_hist`` = instantaneas anuales propias por clave de ``RANGOS_MULTIPLOS``.
    """
    ticker: str | None = None
    precio: float | None = None
    acciones_diluidas: float | None = None
    fcf_ltm: float | None = None
    ebitda_ltm: float | None = None
    ebit_ltm: float | None = None
    ingresos_ltm: float | None = None
    eps_ltm: float | None = None
    bvps: float | None = None
    caja: float | None = None
    deuda_total: float | None = None
    tasa_impositiva: float | None = None
    capex_ltm: float | None = None
    da_ltm: float | None = None
    nwc_operativo: float | None = None
    gasto_interes_ltm: float | None = None
    dividendo_anual: float | None = None
    beta: float | None = None
    rf: float | None = None
    hist_fcf: list[float] = field(default_factory=list)
    hist_ingresos: list[float] = field(default_factory=list)
    hist_ebitda: list[float] = field(default_factory=list)
    hist_eps: list[float] = field(default_factory=list)
    multiplos_hist: dict[str, list[float]] = field(default_factory=dict)
    sector: str | None = None
    industria: str | None = None
    fecha_ltm: str | None = None
    moneda_cotizacion: str | None = None   # info["currency"]
    moneda_reporte: str | None = None      # info["financialCurrency"]
    moneda_conflicto: str | None = None    # texto si reporte != cotizacion (no se rellena nada desde yfinance)
    acciones_nota: str | None = None       # origen de las acciones cuando no es el promedio diluido (dei / basico)
    datos_inconsistentes: list[str] = field(default_factory=list)   # p. ej. ["bvps"]: descartados por inconsistentes
    fuentes: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    # --- Extension multi-anio (todo opcional: las entradas antiguas siguen siendo validas) -------------------------------
    modo: str | None = None                # "local" | "nube" con el que se extrajo
    anios_historia: int | None = None      # anios fiscales de historia pedidos (10 local / 3 nube)
    crec_forward_ingresos: float | None = None     # info["revenueGrowth"] (forward/ultimo trimestre; no PIT)
    crec_forward_beneficios: float | None = None   # forwardEps/trailingEps - 1, o info["earningsGrowth"]
    neto_ltm: float | None = None          # beneficio neto LTM
    ocf_ltm: float | None = None           # flujo de caja operativo LTM
    patrimonio: float | None = None        # patrimonio neto (ultimo balance)
    dividendos_pagados_ltm: float | None = None   # dividendos pagados LTM (XBRL PaymentsOfDividends)
    # Series anuales (LTM al cierre de cada ejercicio, de mas antiguo a mas reciente) alineadas con ``hist_fechas``;
    # None = dato no disponible en esa fecha.
    hist_fechas: list[str] = field(default_factory=list)
    hist_ebit: list[float | None] = field(default_factory=list)
    hist_neto: list[float | None] = field(default_factory=list)
    hist_ocf: list[float | None] = field(default_factory=list)
    hist_capex: list[float | None] = field(default_factory=list)
    hist_dividendos: list[float | None] = field(default_factory=list)
    hist_tasa: list[float | None] = field(default_factory=list)
    hist_deuda: list[float | None] = field(default_factory=list)
    hist_caja: list[float | None] = field(default_factory=list)
    hist_patrimonio: list[float | None] = field(default_factory=list)
    hist_acciones: list[float | None] = field(default_factory=list)
    # Serie TTM por cierre trimestral (para la serie historica del Valor Justo): lista de dicts con ``fecha`` y TTM de
    # ing, ebit, ebitda, fcf, eps, neto, da, capex, interes, impuesto, pre + saldos caja, deuda, patrimonio, acciones.
    serie_ttm: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, d: dict | None) -> "EntradasValoracion":
        d = d or {}
        nombres = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in nombres})


@dataclass
class ParametrosValoracion:
    """Supuestos sobreescribibles (sliders de la UI). ``None`` = automatico/historico."""
    wacc: float | None = None            # None -> WACC automatico (CAPM + estructura de capital)
    g: float = CRECIMIENTO_TERMINAL_DEFECTO
    margen_seguridad: float = MARGEN_SEGURIDAD_DEFECTO
    rf: float | None = None              # None -> entradas.rf o RF_DEFECTO
    erp: float = ERP_DEFECTO
    pe_objetivo: float | None = None
    ev_ebitda_objetivo: float | None = None
    ev_revenue_objetivo: float | None = None
    pb_objetivo: float | None = None
    p_fcf_objetivo: float | None = None
    crecimiento_inicial: float | None = None   # solo DCF de FCF (lo usa ``dcf_inverso``); None -> historico

    def con(self, **kw) -> "ParametrosValoracion":
        return replace(self, **kw)


# --- Utilidades numericas -------------------------------------------------------------------------------------
def _f(x: Any) -> float | None:
    """float finito o None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _limpiar(lista: Any) -> list[float]:
    return [v for v in (_f(x) for x in (lista or [])) if v is not None]


def _como_entradas(e: Any) -> EntradasValoracion:
    return e if isinstance(e, EntradasValoracion) else EntradasValoracion.desde_dict(e)


def _como_params(p: Any) -> ParametrosValoracion:
    if p is None:
        return ParametrosValoracion()
    if isinstance(p, ParametrosValoracion):
        return p
    nombres = {f.name for f in fields(ParametrosValoracion)}
    return ParametrosValoracion(**{k: v for k, v in dict(p).items() if k in nombres})


def cagr(hist: list[float] | None, max_puntos: int = ANIOS_HISTORIA + 1) -> float | None:
    """CAGR entre el primero y el ultimo de los ultimos ``max_puntos`` valores anuales (ambos > 0); None si no hay."""
    h = _limpiar(hist)[-max_puntos:]
    if len(h) < 2 or h[0] <= 0 or h[-1] <= 0:
        return None
    return (h[-1] / h[0]) ** (1.0 / (len(h) - 1)) - 1.0


def es_ciclico(hist: list[float] | None) -> bool:
    """Serie anual ciclica: >= 4 puntos, media > 0, CV (ddof=1) > ``VOL_FCF_ALTA`` y al menos ``MIN_INVERSIONES_CICLO``
    cambios de direccion (subir -> bajar o al reves). El CV solo no basta: NVDA tiene CV alto pero es tendencia."""
    h = _limpiar(hist)[-(ANIOS_HISTORIA + 1):]
    if len(h) < 4:
        return False
    mu = float(np.mean(h))
    if mu <= 0:
        return False
    cv = float(np.std(h, ddof=1)) / mu
    signos = [np.sign(b - a) for a, b in zip(h[:-1], h[1:]) if b != a]
    inversiones = sum(1 for a, b in zip(signos[:-1], signos[1:]) if a != b)
    return cv > VOL_FCF_ALTA and inversiones >= MIN_INVERSIONES_CICLO


def es_financiera(sector: str | None, industria: str | None, ticker: str | None = None) -> bool:
    """Banco/aseguradora/broker/prestamista de credito (donde EBITDA/FCF no son significativos), por sector e industria.

    "Credit Services" incluye prestamistas (COF, AXP, SYF: si cuentan) y procesadores de pago (``PROCESADORES_PAGO``: V, MA,
    PYPL...: no cuentan, su FCF si es informativo)."""
    if not sector or "financ" not in sector.lower():
        return False
    ind = industria or ""
    if _RE_FINANCIERA.search(ind):
        return True
    if _RE_CREDITO.search(ind):
        return (ticker or "").upper() not in PROCESADORES_PAGO
    return False


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --- Tasas y supuestos -----------------------------------------------------------------------------------------

G_RANGO_AVISO = (0.015, 0.045)         # g∞ del usuario fuera de este rango => aviso (el motor igualmente lo usa)


def _validar_escenario(escenario: str) -> str:
    if escenario not in ESCENARIOS:
        raise ValueError(f"escenario desconocido: {escenario!r}; solo existen {ESCENARIOS}")
    return escenario


def beta_blume(beta_raw: float) -> float:
    """Beta suavizada de Blume: 2/3 * beta_raw + 1/3 (p. ej. 1.5 -> 1.3333...)."""
    return BLUME_PESO_RAW * beta_raw + (1.0 - BLUME_PESO_RAW)


def calcular_tasas(e: EntradasValoracion, p: ParametrosValoracion, escenario: str = "medio") -> dict:
    """Tasas del ``escenario``: Rf, beta de Blume, ke (CAPM), kd, WACC y g∞.

    Medio: ``wacc = clip(E/(D+E)*ke + D/(D+E)*kd*(1-t), WACC_MIN, WACC_MAX)`` = [8.0 %, 11.5 %]; un WACC del usuario
    (``p.wacc``) lo reemplaza (con aviso si sale del rango) y desplaza ke en paralelo
    (``(WACC_usuario - WACC_auto) / max(peso_patrimonio, WE_MIN_AJUSTE_KE)``). g∞ = ``p.g`` (3.5 % por defecto).
    Pesimista: se DERIVA del Medio vigente: ``wacc = max(wacc_medio + 200 pb, 12.5 %)`` y ``g∞ = min(2.5 %, g_medio)``;
    ke se desplaza igual que el WACC. Rf: ``p.rf`` > ``e.rf`` (^TNX) > ``RF_DEFECTO`` (4.25 %). ERP = ``p.erp`` (4.5 %).
    La tasa impositiva efectiva tiene piso ``TASA_MIN``."""
    _validar_escenario(escenario)
    if p.rf is not None and _f(p.rf) is not None:
        rf, rf_origen = _f(p.rf), "usuario"
    elif _f(e.rf) is not None:
        rf, rf_origen = _f(e.rf), "^TNX (10Y del Tesoro de EE. UU.)"
    else:
        rf, rf_origen = RF_DEFECTO, "respaldo fijo (^TNX no disponible)"
    erp = _f(p.erp) if _f(p.erp) is not None else ERP_DEFECTO
    beta_raw = _f(e.beta)
    if beta_raw is not None:
        beta_raw = _clip(beta_raw, BETA_RAW_MIN, BETA_RAW_MAX)
        beta = beta_blume(beta_raw)
    else:
        beta = BETA_DEFECTO
    t_raw = _f(e.tasa_impositiva)
    t = _clip(t_raw, TASA_MIN, TASA_IMPOSITIVA_MAX) if t_raw is not None else TASA_IMPOSITIVA_DEFECTO
    ke = rf + beta * erp
    deuda = _f(e.deuda_total)
    interes = _f(e.gasto_interes_ltm)
    kd, kd_origen = rf + SPREAD_DEUDA_DEFECTO, "rf + spread por defecto"
    if deuda and deuda > 0 and interes is not None and interes > 0 and KD_MIN <= interes / deuda <= KD_MAX:
        kd, kd_origen = interes / deuda, "intereses LTM / deuda"
    cap = (_f(e.precio) or 0.0) * (_f(e.acciones_diluidas) or 0.0)
    if cap > 0 and deuda is not None and deuda > 0:
        w_d = deuda / (deuda + cap)
        w_e = 1.0 - w_d
    else:
        w_d, w_e = 0.0, 1.0
    wacc_bruto = w_e * ke + w_d * kd * (1.0 - t)
    wacc_auto = _clip(wacc_bruto, WACC_MIN, WACC_MAX)
    wacc_medio = _f(p.wacc) if p.wacc is not None else wacc_auto
    ke_medio = ke + (wacc_medio - wacc_auto) / max(w_e, WE_MIN_AJUSTE_KE)
    g_medio = _f(p.g) if _f(p.g) is not None else G_TERMINAL["medio"]
    avisos = []
    if p.wacc is not None and not WACC_MIN - 1e-12 <= wacc_medio <= WACC_MAX + 1e-12:
        avisos.append(f"WACC del usuario {wacc_medio * 100:.1f} % fuera del rango del escenario Medio "
                      f"[{WACC_MIN * 100:.1f} %, {WACC_MAX * 100:.1f} %]")
    if not G_RANGO_AVISO[0] - 1e-12 <= g_medio <= G_RANGO_AVISO[1] + 1e-12:
        avisos.append(f"g∞ {g_medio * 100:.1f} % fuera del rango habitual [{G_RANGO_AVISO[0] * 100:.1f} %, "
                      f"{G_RANGO_AVISO[1] * 100:.1f} %]")
    if escenario == "medio":
        wacc, wacc_auto_esc, ke_usado, g = wacc_medio, wacc_auto, ke_medio, g_medio
    else:
        wacc = max(wacc_medio + WACC_PESIMISTA_SPREAD, WACC_PESIMISTA_MIN)
        wacc_auto_esc = max(wacc_auto + WACC_PESIMISTA_SPREAD, WACC_PESIMISTA_MIN)
        ke_usado = ke_medio + (wacc - wacc_medio) / max(w_e, WE_MIN_AJUSTE_KE)
        g = min(G_TERMINAL["pesimista"], g_medio)
    topes_ke, topes_wacc = [], []
    if _f(e.beta) is not None and beta_raw != _f(e.beta):
        topes_ke.append(f"beta {_f(e.beta):.2f} acotada a {beta_raw:.2f} antes de Blume")
    if escenario == "medio" and p.wacc is None and wacc_bruto != wacc_auto:
        topes_wacc.append(f"WACC {wacc_bruto * 100:.2f} % acotado a {wacc_auto * 100:.2f} %")
    if escenario == "pesimista" and wacc == WACC_PESIMISTA_MIN and wacc_medio + WACC_PESIMISTA_SPREAD < WACC_PESIMISTA_MIN:
        topes_wacc.append(f"WACC Pesimista: mínimo {WACC_PESIMISTA_MIN * 100:.1f} % activo")
    return {"escenario": escenario, "rf": rf, "rf_origen": rf_origen, "erp": erp, "beta": beta, "beta_raw": beta_raw,
            "beta_origen": "yfinance, suavizada de Blume (2/3 raw + 1/3)" if beta_raw is not None
            else "no disponible: se usa el valor por defecto", "ke_capm": ke, "ke": ke_usado, "kd": kd,
            "kd_origen": kd_origen, "tasa_impositiva": t,
            "tasa_origen": "efectiva LTM (piso %.0f %%)" % (TASA_MIN * 100) if t_raw is not None else "por defecto (sin dato)",
            "peso_deuda": w_d, "peso_capital": w_e, "wacc_bruto": wacc_bruto, "wacc_auto": wacc_auto_esc, "wacc": wacc,
            "wacc_medio": wacc_medio, "g": g, "g_medio": g_medio,
            "wacc_origen": ("usuario" if p.wacc is not None else "automatico (CAPM + estructura de capital)")
            if escenario == "medio" else f"Medio vigente + {WACC_PESIMISTA_SPREAD * 1e4:.0f} pb (mín. {WACC_PESIMISTA_MIN * 100:.1f} %)",
            "avisos": avisos, "topes_ke": topes_ke, "topes_wacc": topes_wacc, "topes": topes_ke + topes_wacc}



def mediana_recortada(valores: list[float] | None, recorte: float = RECORTE_MEDIANA) -> float | None:
    """Mediana recortada: con >= 5 puntos descarta ``floor(n*recorte)`` extremos de cada lado antes de la mediana."""
    v = sorted(_limpiar(valores))
    if not v:
        return None
    k = int(len(v) * recorte) if len(v) >= 5 else 0
    if k >= 1:
        v = v[k:-k]
    return float(np.median(v))


def _variaciones_anuales(hist: list[float] | None, anios: int) -> list[float]:
    h = _limpiar(hist)[-(anios + 1):]
    return [b / a - 1.0 for a, b in zip(h[:-1], h[1:]) if a > 0 and b > 0]


def _winsorizar(variaciones: list[float]) -> np.ndarray:
    """Recorte duro a [VARIACION_MIN, VARIACION_MAX] y, con >= 4 variaciones, winsorizacion a los percentiles WINSOR_PCT."""
    x = np.clip(np.asarray(variaciones, dtype=float), VARIACION_MIN, VARIACION_MAX)
    if len(x) >= 4:
        lo, hi = np.percentile(x, WINSOR_PCT)
        x = np.clip(x, lo, hi)
    return x


def cagr_winsorizado(hist: list[float] | None, anios: int) -> float | None:
    """CAGR robusto a ``anios``: variaciones anuales (ambos extremos > 0) recortadas/winsorizadas y COMPUESTAS
    geometricamente (``prod(1+r)^(1/n) - 1``). None con menos de 2 variaciones validas."""
    rs = _variaciones_anuales(hist, anios)
    if len(rs) < 2:
        return None
    x = _winsorizar(rs)
    return float(np.prod(1.0 + x) ** (1.0 / len(x)) - 1.0)


def momentum_yoy(hist: list[float] | None) -> float | None:
    """Variacion del ultimo valor LTM frente al LTM de un anio antes (recortada); None si alguno es <= 0."""
    h = _limpiar(hist)
    if len(h) < 2 or h[-2] <= 0 or h[-1] <= 0:
        return None
    return _clip(h[-1] / h[-2] - 1.0, VARIACION_MIN, VARIACION_MAX)


def crecimiento_fundamental(hist: list[float] | None, forward: float | None = None) -> dict:
    """Crecimiento fundamental = ``PESO_CAGR*CAGR + PESO_MOMENTUM*momentum + PESO_FORWARD*forward`` renormalizado sobre
    los componentes disponibles (CAGR = promedio de los CAGR winsorizados a 3Y y 5Y disponibles). Nunca devuelve 0 si
    no hay datos: ``valor`` es None."""
    c = [cagr_winsorizado(hist, n) for n in CAGR_VENTANAS]
    cs = [v for v in c if v is not None]
    fw = _f(forward)
    comp = {"cagr": float(np.mean(cs)) if cs else None, "momentum": momentum_yoy(hist),
            "forward": _clip(fw, FORWARD_MIN, FORWARD_MAX) if fw is not None else None}
    pesos = {"cagr": PESO_CAGR, "momentum": PESO_MOMENTUM, "forward": PESO_FORWARD}
    disp = {k: v for k, v in comp.items() if v is not None}
    valor = (sum(pesos[k] * v for k, v in disp.items()) / sum(pesos[k] for k in disp)) if disp else None
    return {"valor": valor, "componentes": {**comp, "cagr3": c[0], "cagr5": c[1]}, "pesos": pesos}


def crecimiento_etapa1(hist: list[float] | None, forward: float | None, g_inf: float, escenario: str,
                       fijo: float | None = None) -> dict:
    """Crecimiento de la etapa 1 (anios 1-3): ``FRACCION_CRECIMIENTO (65 %) * crecimiento_fundamental`` acotado a
    [CRECIMIENTO_MIN, tope del escenario] (42 % Medio / 15 % Pesimista). Flujo ciclico o sin datos => g∞ (con nota)."""
    tope = TOPE_CRECIMIENTO_ETAPA1[escenario]
    if fijo is not None:
        return {"g1": float(fijo), "nota": "crecimiento inicial fijado (DCF inverso)", "tope": None, "fundamental": None}
    if es_ciclico(hist):
        return {"g1": g_inf, "nota": "flujo cíclico normalizado: crecimiento etapa 1 = g∞", "tope": None, "fundamental": None}
    cf = crecimiento_fundamental(hist, forward)
    if cf["valor"] is None:
        return {"g1": g_inf, "nota": "sin datos de crecimiento: etapa 1 = g∞ (no se inventa crecimiento)", "tope": None,
                "fundamental": cf}
    bruto = FRACCION_CRECIMIENTO * cf["valor"]
    g1 = _clip(bruto, CRECIMIENTO_MIN, tope)
    tope_txt = None
    if bruto > tope:
        tope_txt = f"crecimiento etapa 1 {bruto * 100:.1f} % topado a {tope * 100:.0f} %"
    elif bruto < CRECIMIENTO_MIN:
        tope_txt = f"crecimiento etapa 1 {bruto * 100:.1f} % acotado a {CRECIMIENTO_MIN * 100:.0f} %"
    return {"g1": g1, "nota": f"{FRACCION_CRECIMIENTO * 100:.0f} % x crecimiento fundamental {cf['valor'] * 100:.1f} % "
            f"(tope {tope * 100:.0f} %)", "tope": tope_txt, "fundamental": cf}


def _tope_crecimiento(hist: list[float] | None, forward: float | None = None, escenario: str = "medio") -> str | None:
    """Texto si el crecimiento de la etapa 1 queda acotado (None si no hay tope activo)."""
    return crecimiento_etapa1(hist, forward, G_TERMINAL[escenario], escenario)["tope"]



def _crecimiento_inicial(hist: list[float] | None, g: float, forward: float | None = None,
                         escenario: str = "medio") -> tuple[float, str]:
    """(crecimiento de la etapa 1, nota) del ``escenario`` con g∞ = ``g``."""
    r = crecimiento_etapa1(hist, forward, g, escenario)
    return r["g1"], r["nota"]


def _trayectoria(g1: float, g_inf: float, n: int) -> list[float]:
    """Crecimiento anual t=1..n: etapa 1 (t <= 3) = g1; etapa 2 decae LINEALMENTE hasta g∞ en t = n."""
    out = []
    for t in range(1, n + 1):
        out.append(g1 if t <= ANIOS_ETAPA1 else g1 + (g_inf - g1) * (t - ANIOS_ETAPA1) / (n - ANIOS_ETAPA1))
    return out


def _convergencia(x0: float, x_obj: float, t: int, n: int) -> float:
    """Valor en el anio t de una razon que converge linealmente de x0 (t <= 3) a x_obj (t = n)."""
    return x0 if t <= ANIOS_ETAPA1 else x0 + (x_obj - x0) * (t - ANIOS_ETAPA1) / (n - ANIOS_ETAPA1)



def _base_crecimiento(hist: list[float] | None, ltm: float, g: float, forward: float | None = None,
                      escenario: str = "medio") -> tuple[float, float, bool, str | None]:
    """(base, crecimiento etapa 1, ciclico, tope) de un flujo. Ciclico: base = media de los ultimos
    ``N_NORMALIZACION`` anios y crecimiento = g∞."""
    if es_ciclico(hist):
        return float(np.mean(_limpiar(hist)[-N_NORMALIZACION:])), g, True, None
    r = crecimiento_etapa1(hist, forward, g, escenario)
    return ltm, r["g1"], False, r["tope"]


def _ajuste_fcff(e: EntradasValoracion, t: float) -> float:
    """Intereses LTM despues de impuestos que convierten FCF apalancado en FCFF (0 si no hay gasto de intereses)."""
    interes = _f(e.gasto_interes_ltm)
    return interes * (1.0 - t) if interes is not None and interes > 0 else 0.0



def _historia_razon(num: list[float] | None, den: list[float] | None, ajuste: float = 0.0) -> list[float]:
    """Razones ``(num_i + ajuste)/den_i`` de los ultimos MARGEN_ANIOS_MEDIANA[1] anios (alineadas por el final)."""
    hn, hd = _limpiar(num), _limpiar(den)
    n = min(len(hn), len(hd), MARGEN_ANIOS_MEDIANA[1])
    return [(hn[len(hn) - n + i] + ajuste) / hd[len(hd) - n + i] for i in range(n) if hd[len(hd) - n + i] > 0] if n else []


def _margen_detalle(e: EntradasValoracion, escenario: str = "medio") -> dict:
    """Margen FCF/Ingresos actual y objetivo (FCFF: suma los intereses despues de impuestos LTM).

    Objetivo = ``min(TOPE_MARGEN[escenario], mediana recortada de los ultimos 3-5 anios)``; con menos de 3 anios de
    historia, el margen actual acotado por el tope."""
    tope = TOPE_MARGEN[escenario]
    vacio = {"actual": None, "objetivo": None, "origen": "FCF", "mediano": None, "tope_activo": False, "tope": tope}
    rev, fcf = _f(e.ingresos_ltm), _f(e.fcf_ltm)
    origen = "FCF"
    if rev is None or rev <= 0:
        return vacio
    t = _clip(_f(e.tasa_impositiva) if _f(e.tasa_impositiva) is not None else TASA_IMPOSITIVA_DEFECTO,
              TASA_MIN, TASA_IMPOSITIVA_MAX)
    ajuste = 0.0
    if fcf is None:
        ebit = _f(e.ebit_ltm)
        if ebit is None:
            return vacio
        fcf, origen = ebit * (1.0 - t), "margen operativo despues de impuestos (sin FCF)"
    else:
        ajuste = _ajuste_fcff(e, t) / rev   # el mismo ajuste (en puntos de margen) a la mediana historica
    actual = fcf / rev + ajuste
    margenes = _historia_razon(e.hist_fcf, e.hist_ingresos)
    mediano = (mediana_recortada(margenes) + ajuste) if len(margenes) >= MARGEN_ANIOS_MEDIANA[0] and origen == "FCF" else actual
    return {"actual": actual, "objetivo": min(tope, mediano), "origen": origen, "mediano": mediano,
            "tope_activo": mediano > tope, "tope": tope}


def _conversion_ebitda(e: EntradasValoracion, escenario: str, c0: float) -> dict:
    """Conversion FCFF/EBITDA objetivo del DCF por EBITDA: mediana recortada de 3-5 anios acotada por el tope de margen
    del escenario expresado en base EBITDA (``tope / margen EBITDA``, max 100 %). Sin historia: ``c0`` (sin convergencia)."""
    tope = TOPE_MARGEN[escenario]
    eb, rev = _f(e.ebitda_ltm), _f(e.ingresos_ltm)
    t = _clip(_f(e.tasa_impositiva) if _f(e.tasa_impositiva) is not None else TASA_IMPOSITIVA_DEFECTO, TASA_MIN,
              TASA_IMPOSITIVA_MAX)
    razones = _historia_razon(e.hist_fcf, e.hist_ebitda, _ajuste_fcff(e, t))
    mediano = mediana_recortada(razones) if len(razones) >= MARGEN_ANIOS_MEDIANA[0] else None
    cap = min(1.0, tope / (eb / rev)) if eb and rev and rev > 0 and eb > 0 else 1.0
    base = mediano if mediano is not None else c0
    return {"objetivo": min(cap, base), "mediano": mediano, "tope_conv": cap, "tope_activo": base > cap}



def _margen_fcf(e: EntradasValoracion, escenario: str = "medio") -> tuple[float | None, float | None, str]:
    """(margen actual, margen objetivo, origen) para el DCF por ingresos."""
    d = _margen_detalle(e, escenario)
    return d["actual"], d["objetivo"], d["origen"]



def multiplo_objetivo(clave: str, e: EntradasValoracion, p: ParametrosValoracion, escenario: str = "medio") -> dict:
    """Multiplo objetivo del ``escenario``. {valor|None, origen, n, mediana, rango, tope, ...}.

    Propio = MEDIANA RECORTADA de los multiplos historicos propios (>= ``MIN_PUNTOS_MULTIPLO``), acotada a
    ``RANGOS_MULTIPLOS``; se mezcla ``PESO_MULTIPLO_PROPIO`` (70 %) propio + 30 % benchmark sectorial (tabla fijada a
    mano ``BENCHMARK_SECTORIAL``, solo si ``e.sector`` esta en ella) y se acota de nuevo. Pesimista: x (1 - 15 %)
    (compresion de multiplos). Un multiplo fijado por el usuario reemplaza al propio/sectorial (y tambien se comprime)."""
    _validar_escenario(escenario)
    comp = COMPRESION_MULTIPLOS[escenario]
    fijo = _f(getattr(p, f"{clave}_objetivo", None))
    lo, hi = RANGOS_MULTIPLOS[clave]
    if fijo is not None:
        return {"valor": fijo * (1.0 + comp), "valor_sin_compresion": fijo, "origen": "parametro del usuario", "n": 0,
                "mediana": None, "rango": (lo, hi), "tope": None, "benchmark": None, "compresion": comp}
    pts = [v for v in _limpiar((e.multiplos_hist or {}).get(clave)) if v > 0]
    if len(pts) < MIN_PUNTOS_MULTIPLO:
        return {"valor": None, "origen": "historia insuficiente", "n": len(pts), "mediana": None, "rango": (lo, hi),
                "tope": None, "benchmark": None, "compresion": comp}
    med = mediana_recortada(pts)
    propio = _clip(med, lo, hi)
    bm = (BENCHMARK_SECTORIAL.get(e.sector or "") or {}).get(clave)
    mezcla = PESO_MULTIPLO_PROPIO * propio + (1.0 - PESO_MULTIPLO_PROPIO) * bm if bm is not None else propio
    base = _clip(mezcla, lo, hi)
    tope = None if propio == med else f"objetivo: mediana histórica {med:.1f}x topada a {propio:g}x"
    origen = f"mediana recortada propia de {len(pts)} anios, acotada a [{lo:g}, {hi:g}]"
    if bm is not None:
        origen += f"; {PESO_MULTIPLO_PROPIO:.0%} propio + {1 - PESO_MULTIPLO_PROPIO:.0%} sector ({bm:g}x, aprox. a mano {BENCHMARK_FECHA})"
    return {"valor": base * (1.0 + comp), "valor_sin_compresion": base, "origen": origen, "n": len(pts), "mediana": med,
            "rango": (lo, hi), "tope": tope, "benchmark": bm, "compresion": comp}



def supuestos(entradas: Any, params: Any = None, escenario: str = "medio") -> dict:
    """Todos los supuestos del ``escenario`` (la UI los muestra). Dict plano serializable.

    Son SUPUESTOS DE ESCENARIO, no predicciones. ``topes_activos`` lista los topes/acotaciones que estan condicionando
    el resultado ("por construccion")."""
    _validar_escenario(escenario)
    e, p = _como_entradas(entradas), _como_params(params)
    tasas = calcular_tasas(e, p, escenario)
    g = tasas["g"]
    out: dict[str, Any] = {
        "escenario": escenario, "etiqueta": ETIQUETAS_ESCENARIO[escenario],
        "wacc": tasas["wacc"], "wacc_auto": tasas["wacc_auto"], "wacc_origen": tasas["wacc_origen"],
        "wacc_medio": tasas["wacc_medio"], "g": g, "g_medio": tasas["g_medio"], "margen_seguridad": p.margen_seguridad,
        "ke": tasas["ke"], "kd": tasas["kd"], "kd_origen": tasas["kd_origen"], "rf": tasas["rf"],
        "rf_origen": tasas["rf_origen"], "erp": tasas["erp"], "beta": tasas["beta"], "beta_raw": tasas["beta_raw"],
        "beta_origen": tasas["beta_origen"], "tasa_impositiva": tasas["tasa_impositiva"],
        "tasa_origen": tasas["tasa_origen"], "peso_deuda": tasas["peso_deuda"],
        "horizontes": dict(HORIZONTES), "etapa1_anios": ANIOS_ETAPA1, "fraccion_crecimiento": FRACCION_CRECIMIENTO,
        "tope_crecimiento": TOPE_CRECIMIENTO_ETAPA1[escenario], "tope_margen": TOPE_MARGEN[escenario],
        "compresion_multiplos": COMPRESION_MULTIPLOS[escenario],
        "g_valido": (tasas["wacc"] - g) >= DIF_MIN_WACC_G_DCF - 1e-12, "dif_min_wacc_g": DIF_MIN_WACC_G_DCF,
        "avisos": list(tasas["avisos"]), "modo": e.modo, "anios_historia": e.anios_historia}
    topes = list(tasas["topes"])
    ciclicos = []
    for clave, hist, fwd, nombre in (("fcf", e.hist_fcf, e.crec_forward_beneficios, "FCF"),
                                     ("ebitda", e.hist_ebitda, e.crec_forward_beneficios, "EBITDA"),
                                     ("revenue", e.hist_ingresos, e.crec_forward_ingresos, "ingresos")):
        r = crecimiento_etapa1(hist, fwd, g, escenario)
        out[f"crecimiento_inicial_{clave}"] = r["g1"]
        out[f"crecimiento_inicial_{clave}_nota"] = r["nota"]
        out[f"crecimiento_fundamental_{clave}"] = (r["fundamental"] or {}).get("valor")
        if es_ciclico(hist):
            ciclicos.append(nombre)
        elif r["tope"]:
            topes.append(f"{nombre}: {r['tope']}")
    out["flujos_ciclicos_normalizados"] = ciclicos
    md = _margen_detalle(e, escenario)
    out.update(margen_fcf_actual=md["actual"], margen_fcf_objetivo=md["objetivo"], margen_fcf_base=md["origen"],
               margen_fcf_tope=TOPE_MARGEN[escenario], ajuste_fcff=_ajuste_fcff(e, tasas["tasa_impositiva"]))
    if md["tope_activo"]:
        topes.append(f"margen FCF objetivo: mediana histórica {md['mediano'] * 100:.1f} % topada a "
                     f"{TOPE_MARGEN[escenario] * 100:.0f} %")
    out["multiplos"] = {k: multiplo_objetivo(k, e, p, escenario) for k in RANGOS_MULTIPLOS}
    for k, nombre in (("pe", "P/E"), ("ev_ebitda", "EV/EBITDA"), ("ev_revenue", "EV/Ingresos"), ("pb", "P/B"),
                      ("p_fcf", "P/FCF")):
        if out["multiplos"][k]["tope"]:
            topes.append(f"{nombre} " + out["multiplos"][k]["tope"])
    cl = _crecimiento_lynch(e, escenario)
    out["crecimiento_eps_lynch"] = cl[0] / 100.0 if cl else None
    if cl and cl[2]:
        topes.append(f"Lynch: crecimiento de EPS {cl[1]:.0f} % topado a {cl[0]:.0f} %")
    out["topes_activos"] = topes
    out["financiera"] = es_financiera(e.sector, e.industria, e.ticker)
    return out


# --- Modelos --------------------------------------------------------------------------------------------------
def _res(id_: str, precio: float | None = None, motivo: str | None = None, **extra) -> dict:
    _, nombre, familia = _META[id_]
    precio = _f(precio)
    aplicable = precio is not None and motivo is None
    if aplicable and precio <= 0:  # type: ignore[operator]
        aplicable, motivo = False, "valor estimado <= 0 (la deuda supera el valor operativo o el flujo es negativo)"
    if precio is None and motivo is None:
        motivo = "no se pudo calcular"
    topes = list(extra.pop("topes", None) or [])
    return {"id": id_, "nombre": nombre, "familia": familia,
            "precio_estimado": precio if aplicable else None, "aplicable": aplicable,
            "motivo_exclusion": None if aplicable else motivo, "topes": topes if aplicable else [],
            "tope_activo": bool(topes) and aplicable, **extra}


def _neto(e: EntradasValoracion) -> tuple[float | None, str | None]:
    """Caja neta (caja - deuda total) o (None, motivo)."""
    if _f(e.caja) is None:
        return None, "caja no disponible"
    if _f(e.deuda_total) is None:
        return None, "deuda total no disponible"
    return float(e.caja) - float(e.deuda_total), None  # type: ignore[arg-type]


def _acciones(e: EntradasValoracion) -> tuple[float | None, str | None]:
    a = _f(e.acciones_diluidas)
    return (a, None) if a and a > 0 else (None, "acciones diluidas no disponibles")



def _valor_flujos(flujos: list[float], wacc: float, g: float) -> float:
    """Valor presente de los flujos con convencion de MITAD DE ANIO y valor terminal de Gordon.

    ``DF_t = (1+WACC)^-(t-0.5)``. Valor terminal ``TV_N = FCF_N*(1+g)/(WACC-g)``. Como los flujos llegan a mitad de
    cada anio (N+0.5, N+1.5, ...), el valor en t=N de la perpetuidad es ``TV_N*(1+WACC)^0.5``; descontarlo N anios
    equivale a descontar ``TV_N`` con el MISMO factor del ultimo flujo explicito, ``DF_N = (1+WACC)^-(N-0.5)``
    (t = 9.5 con N = 10). Descontarlo a t = N subvaluaria el terminal en un factor (1+WACC)^0.5 (la convencion de t = N
    solo es coherente con un multiplo de salida, no con Gordon a mitad de anio)."""
    n = len(flujos)
    vp = sum(f / (1.0 + wacc) ** (t - 0.5) for t, f in enumerate(flujos, 1))
    vt = flujos[-1] * (1.0 + g) / (wacc - g)
    return vp + vt / (1.0 + wacc) ** (n - 0.5)


def _motivo_dif_dcf(wacc: float, g: float) -> str | None:
    """Motivo si WACC - g∞ < ``DIF_MIN_WACC_G_DCF`` (2.5 pp): el DCF no aplica."""
    return _motivo_dif_tasas(wacc, g, "WACC")


def _motivo_dif_tasas(tasa: float, g: float, nombre: str) -> str | None:
    """Motivo si la tasa de descuento no supera a g por al menos ``DIF_MIN_WACC_G`` (None si es suficiente)."""
    if tasa <= g:
        return f"g ({g * 100:.1f} %) >= {nombre} ({tasa * 100:.1f} %): el valor terminal no converge"
    if round(tasa - g, 10) < DIF_MIN_WACC_G:
        return (f"diferencia {nombre}-g < {DIF_MIN_WACC_G * 100:g} pp ({nombre} {tasa * 100:.2f} %, g {g * 100:.2f} %): "
                f"el valor terminal explota (no aplica)")
    return None


def _modelo_dcf(id_: str, base: str, n: int, e: EntradasValoracion, p: ParametrosValoracion, tasas: dict,
                escenario: str = "medio") -> dict:
    """DCF explicito de ``n`` anios (10; 5 en la variante corta) en 2 etapas con mitad de anio. Ver el docstring del
    modulo. ``base``: "fcf" | "ebitda" | "revenue"."""
    wacc, g = tasas["wacc"], tasas["g"]
    acc, motivo = _acciones(e)
    if motivo:
        return _res(id_, motivo=motivo)
    neto, motivo = _neto(e)
    if motivo:
        return _res(id_, motivo=motivo)
    motivo = _motivo_dif_dcf(wacc, g)
    if motivo:
        return _res(id_, motivo=motivo)
    t = tasas["tasa_impositiva"]
    topes = list(tasas["topes"])
    ciclico = False
    fijo = p.crecimiento_inicial if base == "fcf" else None
    if base == "fcf":
        f_ltm = _f(e.fcf_ltm)
        if f_ltm is None:
            return _res(id_, motivo="FCF no disponible")
        if f_ltm <= 0:
            return _res(id_, motivo="FCF LTM negativo o cero")
        hf = _limpiar(e.hist_fcf)
        if hf and sum(1 for v in hf if v > 0) < MIN_ANIOS_FCF:
            return _res(id_, motivo=f"historia insuficiente de FCF positivo (< {MIN_ANIOS_FCF} años con FCF > 0)")
        ciclico = fijo is None and es_ciclico(e.hist_fcf)
        f0 = float(np.mean(hf[-N_NORMALIZACION:])) if ciclico else f_ltm
        r = crecimiento_etapa1(e.hist_fcf, e.crec_forward_beneficios, g, escenario, fijo)
        if f0 <= 0:
            return _res(id_, motivo="FCF normalizado negativo o cero")
        f0 += _ajuste_fcff(e, t)   # FCFE -> FCFF: se suman los intereses despues de impuestos (la deuda se resta aparte)
        flujos, f = [], f0
        for gt in _trayectoria(r["g1"], g, n):
            f *= 1 + gt
            flujos.append(f)
    elif base == "ebitda":
        eb_ltm, da, capex = _f(e.ebitda_ltm), _f(e.da_ltm), _f(e.capex_ltm)
        if eb_ltm is None:
            return _res(id_, motivo="EBITDA no disponible")
        if eb_ltm <= 0:
            return _res(id_, motivo="EBITDA LTM negativo o cero")
        if da is None or capex is None:
            return _res(id_, motivo="D&A o CapEx no disponibles")
        da_r, cx_r = _clip(da / eb_ltm, 0.0, 1.0), max(capex, 0.0) / eb_ltm
        nwc = _f(e.nwc_operativo)
        nwc_r = _clip(nwc / eb_ltm, NWC_RATIO_MIN, NWC_RATIO_MAX) if nwc is not None else 0.0
        ciclico = es_ciclico(e.hist_ebitda)
        eb0 = float(np.mean(_limpiar(e.hist_ebitda)[-N_NORMALIZACION:])) if ciclico else eb_ltm
        if eb0 <= 0:
            return _res(id_, motivo="EBITDA normalizado negativo o cero")
        r = crecimiento_etapa1(e.hist_ebitda, e.crec_forward_beneficios, g, escenario)
        c_obj = _conversion_ebitda(e, escenario, (1.0 - t) + da_r * t - cx_r)
        if c_obj["tope_activo"]:
            topes.append(f"conversión FCF/EBITDA objetivo topada a {c_obj['objetivo'] * 100:.0f} % "
                         f"(tope de margen {TOPE_MARGEN[escenario] * 100:.0f} % de ingresos)")
        flujos, prev = [], eb0
        for i, gt in enumerate(_trayectoria(r["g1"], g, n), 1):
            eb = prev * (1 + gt)
            estructural = eb * (1 - t) + da_r * eb * t - cx_r * eb - nwc_r * (eb - prev)
            w = 0.0 if i <= ANIOS_ETAPA1 else (i - ANIOS_ETAPA1) / (n - ANIOS_ETAPA1)
            flujos.append((1 - w) * estructural + w * c_obj["objetivo"] * eb)
            prev = eb
        if flujos[-1] <= 0:
            return _res(id_, motivo="flujo terminal negativo (CapEx/impuestos superan el EBITDA)")
    else:  # revenue
        rev_ltm = _f(e.ingresos_ltm)
        if rev_ltm is None or rev_ltm <= 0:
            return _res(id_, motivo="ingresos no disponibles")
        md = _margen_detalle(e, escenario)
        m_act, m_obj = md["actual"], md["objetivo"]
        if m_act is None or m_obj is None:
            return _res(id_, motivo="margen FCF/operativo no disponible")
        if m_obj <= 0:
            return _res(id_, motivo="margen FCF historico no positivo")
        ciclico = es_ciclico(e.hist_ingresos)
        rev0 = float(np.mean(_limpiar(e.hist_ingresos)[-N_NORMALIZACION:])) if ciclico else rev_ltm
        r = crecimiento_etapa1(e.hist_ingresos, e.crec_forward_ingresos, g, escenario)
        if md["tope_activo"]:
            topes.append(f"margen FCF objetivo: mediana histórica {md['mediano'] * 100:.1f} % topada a "
                         f"{TOPE_MARGEN[escenario] * 100:.0f} %")
        flujos, rev = [], rev0
        for i, gt in enumerate(_trayectoria(r["g1"], g, n), 1):
            rev *= 1 + gt
            flujos.append(rev * _convergencia(m_act, m_obj, i, n))
        if flujos[-1] <= 0:
            return _res(id_, motivo="flujo terminal negativo")
    if r["tope"]:
        topes.append(r["tope"])
    valor = (_valor_flujos(flujos, wacc, g) + neto) / acc  # type: ignore[operator]
    return _res(id_, valor, wacc=wacc, g=g, g1=r["g1"], horizonte=n, ciclico=ciclico, topes=topes,
                crecimiento_nota=r["nota"])


def _motivo_inconsistente(e: EntradasValoracion, *campos: str) -> str | None:
    mal = [c for c in campos if c in (e.datos_inconsistentes or [])]
    return f"dato por acción inconsistente ({', '.join(mal)} no cuadra con el precio: ¿otra clase de acciones?)" if mal \
        else None



def _modelo_multiplo(id_: str, clave: str, e: EntradasValoracion, p: ParametrosValoracion,
                     escenario: str = "medio") -> dict:
    inc = _motivo_inconsistente(e, *{"pe": ("eps_ltm",), "pb": ("bvps",)}.get(clave, ()))
    if inc:
        return _res(id_, motivo=inc)
    mo = multiplo_objetivo(clave, e, p, escenario)
    if mo["valor"] is None:
        return _res(id_, motivo=f"historia insuficiente para el multiplo objetivo (< {MIN_PUNTOS_MULTIPLO} anios)")
    m = mo["valor"]
    extra = {"multiplo_objetivo": m, "topes": [f"{mo['tope']}"] if mo["tope"] else []}
    if clave == "pe":
        eps = _f(e.eps_ltm)
        if eps is None:
            return _res(id_, motivo="EPS no disponible")
        if eps <= 0:
            return _res(id_, motivo="EPS LTM negativo o cero")
        return _res(id_, eps * m, **extra)
    if clave == "pb":
        bv = _f(e.bvps)
        if bv is None:
            return _res(id_, motivo="valor contable por accion no disponible")
        if bv <= 0:
            return _res(id_, motivo="valor contable negativo o cero")
        if mo["mediana"] is not None and mo["mediana"] > PB_NO_INFORMATIVO:
            return _res(id_, motivo=f"P/B historico mediano {mo['mediana']:.0f}x: el valor contable no es informativo "
                                    f"(recompras/intangibles)")
        return _res(id_, bv * m, **extra)
    acc, motivo = _acciones(e)
    if motivo:
        return _res(id_, motivo=motivo)
    if clave == "p_fcf":
        fcf = _f(e.fcf_ltm)
        if fcf is None:
            return _res(id_, motivo="FCF no disponible")
        if fcf <= 0:
            return _res(id_, motivo="FCF LTM negativo o cero")
        return _res(id_, fcf / acc * m, **extra)  # type: ignore[operator]
    neto, motivo = _neto(e)
    if motivo:
        return _res(id_, motivo=motivo)
    if clave == "ev_ebitda":
        eb = _f(e.ebitda_ltm)
        if eb is None:
            return _res(id_, motivo="EBITDA no disponible")
        if eb <= 0:
            return _res(id_, motivo="EBITDA LTM negativo o cero")
        return _res(id_, (eb * m + neto) / acc, **extra)  # type: ignore[operator]
    rev = _f(e.ingresos_ltm)  # ev_revenue
    if rev is None or rev <= 0:
        return _res(id_, motivo="ingresos no disponibles")
    return _res(id_, (rev * m + neto) / acc, **extra)  # type: ignore[operator]



def _modelo_ddm(e: EntradasValoracion, p: ParametrosValoracion, tasas: dict) -> dict:
    d0 = _f(e.dividendo_anual)
    if d0 is None:
        return _res("ddm", motivo="dividendo no disponible")
    if d0 <= 0:
        return _res("ddm", motivo="sin dividendos")
    px = _f(e.precio)
    if px and px > 0 and d0 / px < DDM_RENDIMIENTO_MIN:
        return _res("ddm", motivo=f"dividendo simbólico (rendimiento {d0 / px * 100:.1f} % < "
                                  f"{DDM_RENDIMIENTO_MIN * 100:.0f} %): el DDM subestimaría el valor")
    ke, g = tasas["ke"], tasas["g"]
    if ke <= g:
        return _res("ddm", motivo=f"ke ({ke * 100:.1f} %) <= g ({g * 100:.1f} %): el modelo de Gordon no converge")
    motivo = _motivo_dif_tasas(ke, g, "ke")
    if motivo:
        return _res("ddm", motivo=motivo)
    return _res("ddm", d0 * (1 + g) / (ke - g), ke=ke, g=g, topes=tasas["topes_ke"])



def _crecimiento_ingresos_anual(e: EntradasValoracion) -> float | None:
    """Crecimiento anual de ingresos: YoY del LTM (historia XBRL); sin historia, el forward de yfinance."""
    m = momentum_yoy(e.hist_ingresos)
    if m is not None:
        h = _limpiar(e.hist_ingresos)
        return h[-1] / h[-2] - 1.0     # sin recortar: el umbral del 20 % se compara con el crecimiento real
    return _f(e.crec_forward_ingresos)


def _modelo_epv(e: EntradasValoracion, p: ParametrosValoracion, tasas: dict, escenario: str = "medio") -> dict:
    if escenario == "medio":
        cr = _crecimiento_ingresos_anual(e)
        if cr is not None and cr > EPV_CRECIMIENTO_MAX:
            return _res("epv", motivo=f"crecimiento de ingresos {cr * 100:.0f} % > {EPV_CRECIMIENTO_MAX * 100:.0f} %: el EPV "
                                      f"(crecimiento cero) no es informativo en el escenario Medio")
    ebit = _f(e.ebit_ltm)
    if ebit is None:
        return _res("epv", motivo="resultado operativo (EBIT) no disponible")
    if ebit <= 0:
        return _res("epv", motivo="EBIT LTM negativo o cero")
    acc, motivo = _acciones(e)
    if motivo:
        return _res("epv", motivo=motivo)
    neto, motivo = _neto(e)
    if motivo:
        return _res("epv", motivo=motivo)
    wacc = tasas["wacc"]
    if wacc <= 0:
        return _res("epv", motivo="WACC no positivo")
    motivo = _motivo_dif_tasas(wacc, tasas["g"], "WACC")
    if motivo:
        return _res("epv", motivo=motivo)
    nopat = ebit * (1 - tasas["tasa_impositiva"])
    return _res("epv", (nopat / wacc + neto) / acc, wacc=wacc, topes=tasas["topes"])  # type: ignore[operator]


def _modelo_graham(e: EntradasValoracion) -> dict:
    inc = _motivo_inconsistente(e, "eps_ltm", "bvps")
    if inc:
        return _res("graham", motivo=inc)
    eps, bv, px = _f(e.eps_ltm), _f(e.bvps), _f(e.precio)
    if eps is None or bv is None:
        return _res("graham", motivo="EPS o valor contable por accion no disponibles")
    if eps <= 0:
        return _res("graham", motivo="EPS LTM negativo o cero")
    if bv <= 0:
        return _res("graham", motivo="valor contable negativo o cero")
    if px and px / bv > GRAHAM_PB_MAX:
        return _res("graham", motivo=f"P/B actual {px / bv:.0f}x > {GRAHAM_PB_MAX:g}x: el valor contable no es informativo "
                                     f"(recompras/intangibles)")
    pb = [v for v in _limpiar((e.multiplos_hist or {}).get("pb")) if v > 0]
    if len(pb) >= MIN_PUNTOS_MULTIPLO and float(np.median(pb)) > PB_NO_INFORMATIVO:
        return _res("graham", motivo=f"P/B historico mediano {np.median(pb):.0f}x: el valor contable no es informativo "
                                     f"(recompras/intangibles)")
    return _res("graham", math.sqrt(GRAHAM_FACTOR * eps * bv))



def _crecimiento_lynch(e: EntradasValoracion, escenario: str = "medio") -> tuple[float, float, bool] | None:
    """(crecimiento % usado, crecimiento % bruto, topado) de EPS para Lynch, o None. Bruto = crecimiento fundamental
    (CAGR winsorizado + momentum + forward) de EPS; el tope es 25 % (Medio) / 15 % (Pesimista)."""
    c = crecimiento_fundamental(e.hist_eps, e.crec_forward_beneficios)["valor"]
    if c is None:
        return None
    bruto = c * 100.0
    usado = min(LYNCH_TOPE_PCT_ESC[escenario], bruto)
    return usado, bruto, bruto > usado


def _modelo_lynch(e: EntradasValoracion, escenario: str = "medio") -> dict:
    inc = _motivo_inconsistente(e, "eps_ltm")
    if inc:
        return _res("lynch", motivo=inc)
    eps = _f(e.eps_ltm)
    if eps is None:
        return _res("lynch", motivo="EPS no disponible")
    if eps <= 0:
        return _res("lynch", motivo="EPS LTM negativo o cero")
    cl = _crecimiento_lynch(e, escenario)
    if cl is None:
        return _res("lynch", motivo="crecimiento historico de EPS no disponible")
    crec, bruto, topado = cl
    if bruto <= 0:
        return _res("lynch", motivo="crecimiento de EPS <= 0")
    topes = [f"crecimiento de EPS {bruto:.0f} % topado a {crec:.0f} %"] if topado else []
    return _res("lynch", eps * crec, crecimiento_pct=crec, topes=topes)



def calcular_modelos(entradas: Any, params: Any = None, escenario: str = "medio") -> list[dict]:
    """Los 15 modelos (en el orden de ``MODELOS``) del ``escenario`` ("medio" por defecto o "pesimista"). Funcion pura;
    nunca lanza por datos faltantes.

    Cada dict: id, nombre, familia, precio_estimado (float|None), aplicable, motivo_exclusion (+ datos del modelo).
    Los ids ``dcf_5y_*`` son la variante corta (mismo motor, valor terminal en el anio 5); ``dcf_10y_*`` el motor completo."""
    _validar_escenario(escenario)
    e, p = _como_entradas(entradas), _como_params(params)
    tasas = calcular_tasas(e, p, escenario)
    financiera = es_financiera(e.sector, e.industria, e.ticker)
    out: list[dict] = []
    for id_, _, _ in MODELOS:
        if financiera and id_ in MODELOS_NO_FINANCIERAS:
            out.append(_res(id_, motivo="no aplica a bancos/aseguradoras (EBITDA y FCF no son significativos)"))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if id_.startswith("dcf_"):
                    _, anios, base = id_.split("_")
                    r = _modelo_dcf(id_, base, HORIZONTES[anios], e, p, tasas, escenario)
                elif id_ in ("pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf"):
                    r = _modelo_multiplo(id_, id_, e, p, escenario)
                elif id_ == "ddm":
                    r = _modelo_ddm(e, p, tasas)
                elif id_ == "epv":
                    r = _modelo_epv(e, p, tasas, escenario)
                elif id_ == "graham":
                    r = _modelo_graham(e)
                else:
                    r = _modelo_lynch(e, escenario)
        except Exception as ex:  # noqa: BLE001  (un modelo roto no tumba al resto)
            r = _res(id_, motivo=f"error de calculo: {type(ex).__name__}")
        if e.moneda_conflicto and not r["aplicable"] and any(
                k in (r["motivo_exclusion"] or "") for k in ("no disponible", "historia insuficiente")):
            r["motivo_exclusion"] = e.moneda_conflicto  # la causa real: faltan fundamentales en la moneda del precio
        out.append(r)
    return out


# --- Agregacion ------------------------------------------------------------------------------------------------
def clasificar_incertidumbre(cv: float | None, n_usados: int, n_grupos: int | None = None) -> str:
    """Baja (CV < CV_BAJO), Media (CV_BAJO..CV_ALTO), Alta (CV > CV_ALTO o menos de MIN_MODELOS_CONFIABLE modelos).

    Si se informa ``n_grupos`` y es < ``MIN_FAMILIAS_BAJA``, "Baja" se degrada a "Media": los modelos de pocas familias
    comparten cifras de entrada y coinciden por construccion (falsa tranquilidad)."""
    if cv is None or n_usados < MIN_MODELOS_CONFIABLE or cv > CV_ALTO:
        return "Alta"
    if cv < CV_BAJO:
        return "Media" if n_grupos is not None and n_grupos < MIN_FAMILIAS_BAJA else "Baja"
    return "Media"



def agregar_valor_justo(resultados_modelos: list[dict], modelos_activos: list[str] | None = None,
                        margen_seguridad: float = MARGEN_SEGURIDAD_DEFECTO,
                        precio_actual: float | None = None) -> dict:
    """Consolida los modelos en un unico valor justo (funcion pura, sin red).

    Pasos: filtra por ``modelos_activos`` (None = todos; lista vacia = ninguno) -> descarta no aplicables o precio <= 0
    -> con >= ``MIN_MODELOS_IQR`` validos excluye outliers (IQR) -> ``fair_value`` = PROMEDIO IGUALANDO FAMILIAS:
    media de cada grupo (DCF / Multiplos / Valor-Dividendos) y media de esas medias, es decir 1/3 cada una; si un
    grupo no tiene modelos usados se renormaliza entre los restantes (``pesos_familias``). Si algun modelo no trae
    ``familia`` no se puede igualar y se usa el promedio simple (``metodo`` lo dice).
    Informacion complementaria (mismos modelos usados): ``promedio_simple``, ``mediana``, ``submedias``, ``cv`` y
    ``incertidumbre`` (CV poblacional de los modelos usados + regla de degradacion por familias).
    ``n_validos`` cuenta los validos ANTES del IQR; ``modelos_no_aplicables`` = ids sin precio valido;
    ``modelos_desactivados`` = ids aplicables que el usuario no incluyo (un id nunca esta en ambas listas).
    El ``margen_seguridad`` se acota a [MARGEN_SEGURIDAD_MIN, MARGEN_SEGURIDAD_MAX]."""
    margen_seguridad = _clip(_f(margen_seguridad) if _f(margen_seguridad) is not None else MARGEN_SEGURIDAD_DEFECTO,
                             MARGEN_SEGURIDAD_MIN, MARGEN_SEGURIDAD_MAX)
    resultados = list(resultados_modelos or [])
    activos = None if modelos_activos is None else set(modelos_activos)
    no_aplicables, validos, desactivados = [], {}, []
    for r in resultados:
        px = _f(r.get("precio_estimado"))
        ok = bool(r.get("aplicable")) and px is not None and px > 0
        if not ok:
            no_aplicables.append(r["id"])
        elif activos is None or r["id"] in activos:
            validos[r["id"]] = px
        else:
            desactivados.append(r["id"])
    outliers: list[str] = []
    usados = dict(validos)
    if len(validos) >= MIN_MODELOS_IQR:
        vals = np.array(list(validos.values()), dtype=float)
        q1, q3 = np.percentile(vals, [25, 75])
        ancho = max(IQR_K * (q3 - q1), IQR_PISO_REL * float(np.median(vals)))
        lo, hi = q1 - ancho, q3 + ancho
        outliers = [k for k, v in validos.items() if v < lo or v > hi]
        usados = {k: v for k, v in validos.items() if k not in outliers}
    fam_de = {r["id"]: r.get("familia") for r in resultados}
    familias = [] if any(fam_de.get(k) is None for k in usados) else sorted({GRUPOS_FAMILIA.get(fam_de[k], fam_de[k]) for k in usados},
                                                                           key=lambda x: (GRUPOS + (x,)).index(x))
    cv_pre = None
    if validos:
        vv = np.array(list(validos.values()), dtype=float)
        cv_pre = float(vv.std(ddof=0) / vv.mean()) if vv.mean() > 0 else None
    base = {"fair_value": None, "upside_pct": None, "margin_of_safety_price": None, "incertidumbre": "Alta",
            "cv": None, "cv_prefiltro": cv_pre, "n_validos": len(validos), "n_usados": len(usados),
            "modelos_usados": list(usados), "familias_usadas": familias,
            "modelos_outlier": outliers, "modelos_desactivados": desactivados,
            "modelos_no_aplicables": no_aplicables, "margen_seguridad": margen_seguridad,
            "precio_actual": _f(precio_actual), "motivo": None, "rango": None, "metodo": None,
            "promedio_simple": None, "mediana": None, "submedias": {}, "pesos_familias": {}}
    if not usados:
        sin_activos = activos is not None and not any(r["id"] in activos for r in resultados)
        base["motivo"] = ("ningún modelo activo" if sin_activos
                          else "ningún modelo válido: todos desactivados o no aplicables")
        return base
    v = np.array(list(usados.values()), dtype=float)
    mu = float(v.mean())
    sigma = float(v.std(ddof=0))
    cv = sigma / mu if mu > 0 else None
    px = _f(precio_actual)
    n_grupos = len(familias) if familias else None
    if familias:
        por_grupo: dict[str, list[float]] = {}
        for k, val in usados.items():
            por_grupo.setdefault(GRUPOS_FAMILIA.get(fam_de[k], fam_de[k]), []).append(val)
        sub = {g: float(np.mean(por_grupo[g])) for g in GRUPOS if g in por_grupo}
        fv_ = float(np.mean(list(sub.values())))
        base.update(metodo="igualacion_familias", submedias=sub, pesos_familias={g: 1.0 / len(sub) for g in sub})
    else:
        fv_ = mu
        base.update(metodo="promedio_simple")
    base.update(fair_value=fv_, cv=cv, incertidumbre=clasificar_incertidumbre(cv, len(usados), n_grupos),
                promedio_simple=mu, mediana=float(np.median(v)),
                margin_of_safety_price=fv_ * (1.0 - margen_seguridad), rango=(float(v.min()), float(v.max())),
                upside_pct=(fv_ - px) / px * 100.0 if px is not None and px > 0 else None)
    return base



def complementarias(agregado: dict, modelos: list[dict] | None = None) -> dict | None:
    """Estadisticas INFORMATIVAS: promedio simple, mediana, submedias por grupo de familia y promedio igualando grupos
    (que desde el motor de escenarios es el propio ``fair_value`` cuando hay familias). Los modelos no son independientes:
    6 DCF comparten WACC/g/crecimiento y los de Valor comparten EPS y valor contable, asi que N modelos no son N
    confirmaciones. None sin modelos usados."""
    modelos = modelos if modelos is not None else (agregado or {}).get("modelos") or []
    usados = set((agregado or {}).get("modelos_usados") or [])
    por_grupo: dict[str, list[float]] = {}
    todos: list[float] = []
    for m in modelos:
        if m["id"] in usados and _f(m.get("precio_estimado")) is not None:
            por_grupo.setdefault(GRUPOS_FAMILIA.get(m.get("familia"), m.get("familia")), []).append(float(m["precio_estimado"]))
            todos.append(float(m["precio_estimado"]))
    if not todos:
        return None
    sub = {g: float(np.mean(por_grupo[g])) for g in GRUPOS if g in por_grupo}
    prom_fam = float(np.mean(list(sub.values())))
    px = _f((agregado or {}).get("precio_actual"))
    up = (lambda x: (x - px) / px * 100.0 if px and px > 0 else None)  # noqa: E731
    simple = float(np.mean(todos))
    return {"mediana": float(np.median(todos)), "promedio_simple": simple, "submedias": sub,
            "n_por_grupo": {g: len(por_grupo[g]) for g in sub}, "promedio_familias": prom_fam,
            "upside_mediana": up(float(np.median(todos))), "upside_promedio_simple": up(simple),
            "upside_promedio_familias": up(prom_fam)}



def _valor_dcf(e: EntradasValoracion, p: ParametrosValoracion, id_: str) -> float | None:
    _, anios, base = id_.split("_")
    r = _modelo_dcf(id_, base, HORIZONTES[anios], e, p, calcular_tasas(e, p, "medio"), "medio")
    return r["precio_estimado"] if r["aplicable"] else None


def _biseccion(f, lo: float, hi: float, objetivo: float, creciente: bool, it: int = 60) -> float | None:
    """Raiz de f(x) = objetivo en [lo, hi] por biseccion (f monotona); None si el objetivo no esta en el rango."""
    flo, fhi = f(lo), f(hi)
    if flo is None or fhi is None:
        return None
    a, b = (flo, fhi) if creciente else (fhi, flo)
    if not a <= objetivo <= b:
        return None
    for _ in range(it):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm is None:
            return None
        if (fm < objetivo) == creciente:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2



def dcf_inverso(entradas: Any, params: Any = None, precio: float | None = None, modelo: str = "dcf_10y_fcf") -> dict:
    """DCF inverso (funcion pura, escenario Medio): WACC y crecimiento de la etapa 1 que igualan el ``modelo`` DCF al
    precio.

    ``wacc_implicito``: el WACC (con g∞ y crecimiento usados) que da exactamente el precio; ``g0_implicito``: el
    crecimiento de la etapa 1 (sin tope; con el WACC usado) que lo da. Cada uno es None si no existe en su rango (WACC en
    [g∞ + ``DIF_MIN_WACC_G_DCF``, 40 %]; crecimiento en [-20 %, 60 %]) o si el modelo no aplica. Lectura: si el WACC
    implicito es menor o el crecimiento implicito mayor que los usados, el mercado descuenta menos riesgo o mas
    crecimiento que estos supuestos."""
    e, p = _como_entradas(entradas), _como_params(params)
    px = _f(precio) if precio is not None else _f(e.precio)
    out = {"modelo": modelo, "precio": px, "wacc_usado": None, "wacc_implicito": None, "g0_usado": None,
           "g0_implicito": None, "valor_con_supuestos": None}
    if not px or px <= 0 or modelo not in IDS_MODELOS or not modelo.startswith("dcf_") or modelo.endswith("revenue") \
            or modelo.endswith("ebitda"):
        return out
    tasas = calcular_tasas(e, p, "medio")
    out["wacc_usado"] = tasas["wacc"]
    out["valor_con_supuestos"] = _valor_dcf(e, p, modelo)
    g0u = crecimiento_etapa1(e.hist_fcf, e.crec_forward_beneficios, tasas["g"], "medio")["g1"]
    out["g0_usado"] = p.crecimiento_inicial if p.crecimiento_inicial is not None else g0u
    out["wacc_implicito"] = _biseccion(lambda w: _valor_dcf(e, p.con(wacc=w), modelo), tasas["g"] + DIF_MIN_WACC_G_DCF, 0.40,
                                       px, creciente=False)
    out["g0_implicito"] = _biseccion(lambda g0: _valor_dcf(e, p.con(crecimiento_inicial=g0), modelo), -0.20, 0.60, px,
                                     creciente=True)
    return out



def valor_justo(entradas: Any, params: Any = None, modelos_activos: list[str] | None = None,
                precio_actual: float | None = None, escenario: str = "medio") -> dict:
    """Atajo: ``calcular_modelos`` + ``agregar_valor_justo`` (escenario Medio por defecto) con el margen de ``params``."""
    e, p = _como_entradas(entradas), _como_params(params)
    px = precio_actual if precio_actual is not None else e.precio
    modelos = calcular_modelos(e, p, escenario)
    agg = agregar_valor_justo(modelos, modelos_activos, p.margen_seguridad, px)
    agg["modelos"] = modelos
    return agg


AVISO_ESCENARIOS = ("Los escenarios Medio y Pesimista son SUPUESTOS DE ESCENARIO (tasas, crecimientos, topes y "
                    "compresión de múltiplos fijados de antemano), no predicciones. Solo existen estos dos.")


def calcular_escenarios(entradas: Any, params: Any = None, modelos_activos: list[str] | None = None,
                        margen_seguridad: float | None = None, precio_actual: float | None = None) -> dict:
    """Valor justo en los dos escenarios (funcion pura y sin red: recalcular con otros WACC/g/sliders es instantaneo).

    Devuelve ``{"medio": agg, "pesimista": agg, "modelos": [...], "supuestos": {...}, "escenarios": ("medio",
    "pesimista"), "aviso": str}``. Cada ``agg`` es el de ``agregar_valor_justo`` (+ ``modelos`` y ``complementarias``).
    ``modelos`` trae por modelo: id, nombre, familia, ``aplicable`` (en algun escenario), ``motivo``, ``precio_medio``,
    ``precio_pesimista``, ``aplicable_medio/_pesimista``, ``motivo_medio/_pesimista``, ``topes_medio/_pesimista``,
    ``activo``. ``supuestos`` = {"medio": ..., "pesimista": ..., "avisos": [...], "nota": ...}. Los sliders del usuario
    (``params``: wacc, g, rf, erp, multiplos objetivo) sobreescriben el Medio; el Pesimista se deriva del Medio vigente.
    ``margen_seguridad`` None => ``params.margen_seguridad`` (0.25 por defecto)."""
    e, p = _como_entradas(entradas), _como_params(params)
    mos = margen_seguridad if margen_seguridad is not None else p.margen_seguridad
    px = precio_actual if precio_actual is not None else e.precio
    ms = {esc: calcular_modelos(e, p, esc) for esc in ESCENARIOS}
    out: dict[str, Any] = {}
    for esc in ESCENARIOS:
        agg = agregar_valor_justo(ms[esc], modelos_activos, mos, px)
        agg["modelos"] = ms[esc]
        agg["complementarias"] = complementarias(agg, ms[esc])
        agg["escenario"] = esc
        out[esc] = agg
    activos = None if modelos_activos is None else set(modelos_activos)
    modelos = []
    for m, q in zip(ms["medio"], ms["pesimista"]):
        modelos.append({"id": m["id"], "nombre": m["nombre"], "familia": m["familia"],
                        "aplicable": bool(m["aplicable"] or q["aplicable"]),
                        "motivo": m["motivo_exclusion"] or q["motivo_exclusion"],
                        "aplicable_medio": m["aplicable"], "aplicable_pesimista": q["aplicable"],
                        "motivo_medio": m["motivo_exclusion"], "motivo_pesimista": q["motivo_exclusion"],
                        "precio_medio": m["precio_estimado"], "precio_pesimista": q["precio_estimado"],
                        "topes_medio": m["topes"], "topes_pesimista": q["topes"],
                        "activo": activos is None or m["id"] in activos})
    sup = {esc: supuestos(e, p, esc) for esc in ESCENARIOS}
    out["modelos"] = modelos
    out["supuestos"] = {**sup, "avisos": list(sup["medio"]["avisos"]), "nota": AVISO_ESCENARIOS,
                        "modo": e.modo, "anios_historia": e.anios_historia}
    out["escenarios"] = ESCENARIOS
    out["aviso"] = AVISO_ESCENARIOS
    return out


# --- Extraccion de entradas (reutiliza los helpers XBRL de valuation_floors) ----------------------------------
_REV = ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet"]
_OCF = ["us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
_CAPEX = ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets",
          "us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment"]
_EBIT = ["us-gaap:OperatingIncomeLoss"]
_DA = ["us-gaap:DepreciationDepletionAndAmortization", "us-gaap:DepreciationAndAmortization",
       "us-gaap:DepreciationAmortizationAndAccretionNet", "us-gaap:Depreciation"]
_EPS = ["us-gaap:EarningsPerShareDiluted"]
_INTERES = ["us-gaap:InterestExpense", "us-gaap:InterestExpenseNonoperating", "us-gaap:InterestExpenseDebt",
            "us-gaap:InterestAndDebtExpense"]
_IMPUESTO = ["us-gaap:IncomeTaxExpenseBenefit"]
_ANTES_IMP = ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
              "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]
_EQUITY = ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
_ACTIVO_CORR, _PASIVO_CORR = ["us-gaap:AssetsCurrent"], ["us-gaap:LiabilitiesCurrent"]
_CAJA_TOTAL = ["us-gaap:CashCashEquivalentsAndShortTermInvestments"]
_CAJA_EFECTIVO = ["us-gaap:CashAndCashEquivalentsAtCarryingValue"]
# Inversiones corrientes: el primer concepto presente en cada fecha (cada emisor usa uno distinto).
_CAJA_INVERSIONES = ["us-gaap:MarketableSecuritiesCurrent", "us-gaap:ShortTermInvestments",
                     "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                     "us-gaap:AvailableForSaleSecuritiesDebtSecurities", "us-gaap:MarketableSecurities"]
_DEUDA_CP = ["us-gaap:CommercialPaper", "us-gaap:ShortTermBorrowings"]
_ACCIONES_DIL = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
_ACCIONES_BAS = "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"
_ACCIONES_DEI = "dei:EntityCommonStockSharesOutstanding"


def _vf():
    from src.levels import valuation_floors as vf
    return vf


def _flujo(facts, conceptos, splits=None, ajusta=False) -> pd.Series:
    """Flujo trimestral aislado; los conceptos de menor prioridad solo rellenan fechas que faltan."""
    vf = _vf()
    out: dict = {}
    for c in reversed(conceptos):
        s = vf._flujo_trimestral(facts, c, splits, ajusta)
        out.update(s.to_dict())
    return _corregir_q4_anual(pd.Series(out, dtype=float).sort_index()) if out else pd.Series(dtype=float)


def _corregir_q4_anual(q: pd.Series) -> pd.Series:
    """Corrige un "trimestre" que en realidad es el ANUAL sin diferenciar (el helper de valuation_floors deja Q4 = FY cuando
    falta el acumulado de 9 meses; p. ej. MSFT 2016-06: OCF 33.3 B en un trimestre). Si un valor excede 2.5x la mediana de
    sus vecinos y ``valor - suma de los 3 trimestres previos`` cae entre 0.3x y 3x esa mediana, se reemplaza por esa
    diferencia (la resta que faltaba). Sin ese ajuste, un TTM con un anio incluido de mas distorsiona FCF/EBITDA historicos."""
    if len(q) < 6:
        return q
    v = q.to_numpy(dtype=float).copy()
    for i in range(3, len(v)):
        vec = [v[j] for j in range(max(0, i - 4), min(len(v), i + 5)) if j != i]
        pos = [x for x in vec if x > 0]
        if len(pos) < 4 or v[i] <= 0:
            continue
        med = float(np.median(pos))
        if v[i] > 2.5 * med:
            d = v[i] - v[i - 3:i].sum()
            if 0.3 * med <= d <= 3.0 * med:
                v[i] = d
    return pd.Series(v, index=q.index)


def _instante(facts, conceptos) -> pd.Series:
    vf = _vf()
    out: dict = {}
    for c in reversed(conceptos):
        s = vf._serie_concepto(facts, c, "instant")
        if not s.empty:
            out.update(s.set_index("period_end")["numeric_value"].to_dict())
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def _sin_huecos(q: pd.Series, n: int) -> bool:
    """Los ultimos ``n`` trimestres son consecutivos (60-130 dias entre cierres)."""
    if len(q) < n:
        return False
    d = pd.Series(q.index[-n:]).diff().dropna().dt.days
    return bool(((d >= 60) & (d <= 130)).all())


def _ltm(q: pd.Series) -> float | None:
    return float(q.tail(4).sum()) if _sin_huecos(q, 4) else None


def _anual(q: pd.Series, k: int) -> tuple[float, pd.Timestamp] | None:
    """Suma de 4 trimestres consecutivos que terminan ``k`` anios antes del ultimo (None si no hay datos)."""
    fin = len(q) - 4 * k
    if fin < 4:
        return None
    ventana = q.iloc[fin - 4:fin]
    d = pd.Series(ventana.index).diff().dropna().dt.days
    if not ((d >= 60) & (d <= 130)).all():
        return None
    return float(ventana.sum()), ventana.index[-1]


def _historia(q: pd.Series, n: int = ANIOS_HISTORIA + 1) -> list[float]:
    """Valores anuales LTM de mas antiguo a mas reciente (se detiene en el primer hueco)."""
    out = []
    for k in range(n):
        a = _anual(q, k)
        if a is None:
            break
        out.append(a[0])
    return out[::-1]


def _asof(s: pd.Series, fecha: pd.Timestamp, max_dias: int = 140) -> float | None:
    """Ultimo valor de ``s`` en o antes de ``fecha`` (max ``max_dias`` de antiguedad)."""
    if s is None or s.empty:
        return None
    s2 = s[s.index <= fecha]
    if s2.empty or (fecha - s2.index[-1]).days > max_dias:
        return None
    return float(s2.iloc[-1])


def _caja_serie(facts) -> pd.Series:
    """Caja + inversiones corrientes por fecha: el total tageado si existe; si no, la suma de componentes."""
    total = _instante(facts, _CAJA_TOTAL)
    comps = [_instante(facts, lista) for lista in (_CAJA_EFECTIVO, _CAJA_INVERSIONES)]
    comps = [c for c in comps if not c.empty]
    fechas = sorted(set(total.index).union(*[set(c.index) for c in comps])) if (comps or not total.empty) else []
    vals = {}
    for d in fechas:
        if d in total.index:
            vals[d] = float(total[d])
        else:
            parte = [float(c[d]) for c in comps if d in c.index]
            if parte:
                vals[d] = sum(parte)
    return pd.Series(vals, dtype=float).sort_index()


def _deuda_serie(facts) -> pd.Series:
    """Deuda financiera: largo plazo total (+ papel comercial / prestamos corto plazo). Sin arrendamientos."""
    lt_tot = _instante(facts, ["us-gaap:LongTermDebt", "us-gaap:LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"])
    lt_nc = _instante(facts, ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebtAndCapitalLeaseObligations"])
    lt_c = _instante(facts, ["us-gaap:LongTermDebtCurrent", "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent",
                             "us-gaap:DebtCurrent"])
    cp = _instante(facts, _DEUDA_CP)
    fechas = sorted(set(lt_tot.index) | set(lt_nc.index) | set(cp.index))
    vals = {}
    for d in fechas:
        if d in lt_tot.index:
            base = float(lt_tot[d])
        elif d in lt_nc.index:
            base = float(lt_nc[d]) + (float(lt_c[d]) if d in lt_c.index else 0.0)
        elif d in cp.index:
            base = 0.0
        else:
            continue
        vals[d] = base + (float(cp[d]) if d in cp.index else 0.0)
    return pd.Series(vals, dtype=float).sort_index()


def _acciones_serie(facts, splits, conceptos=(_ACCIONES_DIL,)) -> pd.Series:
    """Acciones promedio ponderado por trimestre (y anual como respaldo), ajustadas por splits posteriores.

    ``conceptos`` por prioridad: con (diluido, basico) el basico solo rellena las fechas en que el diluido falta (p. ej.
    XOM dejo de etiquetar el diluido en 2013)."""
    vf = _vf()
    out: dict = {}
    for c in reversed(conceptos):
        s = vf._serie_concepto(facts, c, "duration")
        if s.empty:
            continue
        fac = s["ref_filing"].apply(lambda t: vf._factor_split(splits, t + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)))
        s = s.assign(v=s["numeric_value"] * fac)
        q = s[s["dur"] <= 110].set_index("period_end")["v"].to_dict()
        for _, r in s[(s["dur"] >= 350) & (s["dur"] <= 380)].iterrows():
            q.setdefault(r["period_end"], r["v"])
        out.update(q)
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


def _acciones_portada(facts, fecha: pd.Timestamp) -> float | None:
    """Acciones en circulacion de la portada del 10-Q/10-K (``dei:EntityCommonStockSharesOutstanding``) del ultimo periodo.

    None si no hay, si esta lejos de ``fecha`` o si hay varios valores distintos en la misma fecha (varias clases de
    acciones: las filas no traen la dimension de clase, asi que no se puede sumar con seguridad)."""
    f = facts[facts["concept"] == _ACCIONES_DEI].dropna(subset=["numeric_value"])
    if f.empty:
        return None
    ult = f["period_end"].max()
    if abs((ult - fecha).days) > 200:
        return None
    vals = f[f["period_end"] == ult]["numeric_value"].unique()
    return float(vals[0]) if len(vals) == 1 and vals[0] > 0 else None


def _precio_sin_ajuste_dividendos(df: pd.DataFrame) -> pd.Series:
    """Precio dividido por el factor acumulado de dividendos (yfinance ``auto_adjust`` ajusta el pasado).

    adj_t = raw_t * F_t con F_{d-1} = F_d / (1 + F_d*D/adj_{d-1}) hacia atras desde cada ex-dividendo d. Conserva el ajuste
    por splits (los EPS XBRL tambien se ajustan por splits). Sin columna ``Dividends`` devuelve el Close tal cual."""
    close = df["Close"].astype(float)
    if "Dividends" not in df.columns:
        return close
    adj, div = close.to_numpy(), df["Dividends"].fillna(0.0).to_numpy(dtype=float)
    F = np.ones(len(adj))
    f = 1.0
    for i in np.where(div > 0)[0][::-1]:
        if i == 0 or adj[i - 1] <= 0:
            continue
        f = f / (1.0 + f * div[i] / adj[i - 1])
        F[:i] = f
    return pd.Series(adj / F, index=close.index)



def _instantaneas_multiplos(precio_raw: pd.Series, eps_q, fcf_q, ebitda_q, rev_q, sh_q, caja, deuda, equity,
                            k_max: int = ANIOS_HISTORIA) -> dict:
    """Multiplos propios en k = 1..``k_max`` anios atras (precio 45 dias despues del cierre del periodo)."""
    out: dict[str, list[float]] = {k: [] for k in RANGOS_MULTIPLOS}
    for k in range(k_max, 0, -1):
        def px_en(fecha):
            f = fecha + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)
            s = precio_raw[precio_raw.index <= f]
            return None if s.empty or f > precio_raw.index[-1] or (f - s.index[-1]).days > 10 else float(s.iloc[-1])
        a_eps = _anual(eps_q, k)
        if a_eps and a_eps[0] > 0:
            px = px_en(a_eps[1])
            if px:
                out["pe"].append(px / a_eps[0])
        for clave, serie in (("p_fcf", fcf_q), ("ev_ebitda", ebitda_q), ("ev_revenue", rev_q)):
            a = _anual(serie, k)
            if not a or a[0] <= 0:
                continue
            px, sh = px_en(a[1]), _asof(sh_q, a[1])
            if not px or not sh:
                continue
            cap = px * sh
            if clave == "p_fcf":
                out[clave].append(cap / a[0])
                continue
            c, d = _asof(caja, a[1]), _asof(deuda, a[1])
            if c is None or d is None:
                continue
            out[clave].append((cap + d - c) / a[0])
        a_eq = _asof(equity, a_eps[1]) if a_eps else None
        if a_eq and a_eq > 0 and a_eps:
            px, sh = px_en(a_eps[1]), _asof(sh_q, a_eps[1])
            if px and sh:
                out["pb"].append(px * sh / a_eq)
    return out


# --- Series multi-anio -----------------------------------------------------------------------------------------
_NETO = ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss", "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"]
_DIVIDENDOS = ["us-gaap:PaymentsOfDividends", "us-gaap:PaymentsOfDividendsCommonStock"]


def _ttm_rolling(q: pd.Series) -> pd.Series:
    """TTM (suma de 4 trimestres CONSECUTIVOS, 60-130 dias entre cierres) indexado por el cierre del ultimo trimestre."""
    if q is None or len(q) < 4:
        return pd.Series(dtype=float)
    idx, vals = q.index, q.to_numpy(dtype=float)
    out = {}
    for i in range(3, len(q)):
        d = np.diff(idx[i - 3:i + 1].values).astype("timedelta64[D]").astype(int)
        if ((d >= 60) & (d <= 130)).all():
            out[idx[i]] = float(vals[i - 3:i + 1].sum())
    return pd.Series(out, dtype=float).sort_index()


def _valor_en(ttm: pd.Series, fecha: pd.Timestamp, tol: int = 10) -> float | None:
    """Valor del TTM cuyo cierre coincide con ``fecha`` (+-``tol`` dias); None si no hay."""
    if ttm is None or ttm.empty:
        return None
    i = ttm.index.searchsorted(fecha)
    for j in (i - 1, i):
        if 0 <= j < len(ttm) and abs((ttm.index[j] - fecha).days) <= tol:
            return float(ttm.iloc[j])
    return None


def _historia_fechas(q: pd.Series, n: int) -> tuple[list[float], list[pd.Timestamp]]:
    """Como ``_historia`` pero devolviendo tambien las fechas de cierre (de mas antiguo a mas reciente)."""
    vals, fechas = [], []
    for k in range(n):
        a = _anual(q, k)
        if a is None:
            break
        vals.append(a[0])
        fechas.append(a[1])
    return vals[::-1], fechas[::-1]


def _depurar_acciones(s: pd.Series) -> pd.Series:
    """Corrige glitches puntuales de la serie de acciones (p. ej. un trimestre sin ajustar por un split): un valor que se
    desvia > ``SALTO_ACCIONES`` de sus dos vecinas, que a su vez coinciden entre si (+-15 %), se sustituye por su media."""
    if s is None or len(s) < 3:
        return s
    v = s.to_numpy(dtype=float).copy()
    # 1) ESCALA: emisores que en algunos periodos reportan en miles/millones (KO 2017-18: 4334 en vez de 4.334e9). Un valor
    #    a >50x de la mediana de la serie se reescala por 10^(+-3k) si asi cae a menos de 5x de ella.
    med = float(np.median(v[v > 0])) if (v > 0).any() else 0.0
    if med > 0:
        for i, x in enumerate(v):
            if x > 0 and (x < med / 50.0 or x > med * 50.0):
                for k in (3, -3, 6, -6, 9, -9):
                    y = x * 10.0 ** k
                    if med / 5.0 <= y <= med * 5.0:
                        v[i] = y
                        break
    for i in range(1, len(v) - 1):
        a, b, c = v[i - 1], v[i], v[i + 1]
        if a > 0 and c > 0 and abs(a / c - 1.0) < 0.15 and abs(b / ((a + c) / 2.0) - 1.0) > SALTO_ACCIONES:
            v[i] = (a + c) / 2.0
    return pd.Series(v, index=s.index)


def _construir_serie_ttm(series: dict[str, pd.Series], saldos: dict[str, pd.Series]) -> list[dict]:
    """Una fila por cierre trimestral con TTM de cada flujo y saldos puntuales (para la serie historica del valor justo)."""
    ttms = {k: _ttm_rolling(q) for k, q in series.items()}
    fechas = sorted(set().union(*[set(t.index) for k, t in ttms.items() if k in ("ing", "ocf", "eps", "ebit")]))
    filas = []
    for d in fechas:
        fila: dict[str, Any] = {"fecha": str(d.date())}
        for k, t in ttms.items():
            v = _valor_en(t, d, 0)
            fila[k] = v
        for k, s in saldos.items():
            fila[k] = _asof(s, d, 150)
        filas.append(fila)
    return filas


# --- Modo local / nube, Rf (^TNX) y cache ---------------------------------------------------------------------------
def modo_por_defecto() -> str:
    """"nube" si hay variable PORT (Render y similares); si no, "local"."""
    from src.reporting import userdata
    return "nube" if userdata.en_nube() else "local"


def anios_para_modo(modo: str) -> int:
    """Anios fiscales de historia: ``ANIOS_LOCAL`` (10) en local sin recortes; ``ANIOS_NUBE`` (3) en el visor ligero."""
    return ANIOS_NUBE if modo == "nube" else ANIOS_LOCAL


def _resolver_modo(modo: str | None, anios: int | None) -> tuple[str, int]:
    m = modo if modo in ("local", "nube") else modo_por_defecto()
    a = int(anios) if anios else anios_para_modo(m)
    return m, max(1, min(a, 15))


def normalizar_tnx(valor: Any) -> float | None:
    """Rendimiento de ^TNX como FRACCION. yfinance lo da en puntos porcentuales (4.25 = 4.25 %): se divide /100. Valida
    el rango ``RF_RANGO_PCT`` (0.5 %-12 %); fuera de el devuelve None. Un valor < 0.2 se interpreta como fraccion."""
    v = _f(valor)
    if v is None or v <= 0:
        return None
    pct = v * 100.0 if v < 0.2 else v
    if not RF_RANGO_PCT[0] <= pct <= RF_RANGO_PCT[1]:
        return None
    return pct / 100.0


_RF_MEM: dict[str, Any] = {"t": 0.0, "v": None, "ttl": 0.0}
_RF_TTL_FALLO = 300.0        # tras un fallo de red no se reintenta durante 5 min (evita colgar cada consulta offline)


def _descargar_tnx() -> float | None:
    import yfinance as yf
    h = yf.Ticker("^TNX").history(period="5d")
    cierre = h["Close"].dropna()
    return float(cierre.iloc[-1]) if len(cierre) else None


def obtener_rf(fetch: Any = None, usar_cache: bool = True) -> float | None:
    """Tasa libre de riesgo (fraccion) del ^TNX con cache de 24 h (memoria del proceso siempre; disco solo en local).

    None si no se pudo obtener o esta fuera del rango valido (el motor usa entonces ``RF_DEFECTO`` = 4.25 %).
    ``fetch`` (inyectable en tests) devuelve el valor crudo del ^TNX."""
    import time
    from src.reporting import cache_local
    ahora = time.time()
    if usar_cache:
        if ahora - _RF_MEM["t"] < _RF_MEM["ttl"]:
            return _RF_MEM["v"]
        previo = cache_local.obtener("rf:^TNX", VERSION_CODIGO)
        if _f(previo) is not None:
            _RF_MEM.update(t=ahora, v=float(previo), ttl=cache_local.TTL_POR_TIPO["rf"])
            return float(previo)
    try:
        crudo = (fetch or _descargar_tnx)()
    except Exception:  # noqa: BLE001  (sin red: se usa el respaldo)
        crudo = None
    v = normalizar_tnx(crudo)
    if usar_cache:
        _RF_MEM.update(t=ahora, v=v, ttl=cache_local.TTL_POR_TIPO["rf"] if v is not None else _RF_TTL_FALLO)
        if v is not None:
            cache_local.guardar("rf:^TNX", v, "rf", version=VERSION_CODIGO)
    return v


def obtener_info_mercado(ticker: str, tk_obj: Any = None, fetch: Any = None) -> dict:
    """``info`` de yfinance con cache de 6 h en disco (solo local). ``fetch(ticker)`` o ``tk_obj.info``; {} si falla."""
    from src.reporting import cache_local
    clave = f"info:{ticker.upper()}"
    previo = cache_local.obtener(clave, VERSION_CODIGO)
    if isinstance(previo, dict) and previo:
        return previo
    try:
        info = dict(fetch(ticker) if fetch else (tk_obj.info if tk_obj is not None else {}))
    except Exception:  # noqa: BLE001
        info = {}
    if info:
        cache_local.guardar(clave, info, "info", ticker, version=VERSION_CODIGO)
    return info


def _crecimientos_forward(e: EntradasValoracion, info: dict) -> None:
    """Crecimiento forward de yfinance (no PIT): ingresos = ``revenueGrowth``; beneficios = ``forwardEps/trailingEps - 1``
    o, sin eso, ``earningsGrowth``. Con conflicto de moneda no se usan. Lo que falte queda en None."""
    if e.moneda_conflicto:
        return
    rg = _f(info.get("revenueGrowth"))
    if rg is not None:
        e.crec_forward_ingresos = rg
        e.fuentes["crec_forward_ingresos"] = "yfinance info (revenueGrowth)"
    fe, te = _f(info.get("forwardEps")), _f(info.get("trailingEps"))
    if fe is not None and te is not None and fe > 0 and te > 0:
        e.crec_forward_beneficios = fe / te - 1.0
        e.fuentes["crec_forward_beneficios"] = "yfinance info (forwardEps / trailingEps - 1)"
    elif _f(info.get("earningsGrowth")) is not None:
        e.crec_forward_beneficios = float(info["earningsGrowth"])
        e.fuentes["crec_forward_beneficios"] = "yfinance info (earningsGrowth)"


def _ultima_fecha(df) -> str | None:
    try:
        return str(pd.Timestamp(df.index[-1]).date())
    except Exception:  # noqa: BLE001
        return None


def _clave_entradas(ticker: str, anios: int, df) -> str:
    return f"entradas:{ticker.upper()}:{anios}:{_ultima_fecha(df)}"


def extraer_entradas(ticker: str, tk_obj: Any, df: pd.DataFrame, info: dict | None = None,
                     facts: pd.DataFrame | None = None, modo: str | None = None, anios: int | None = None,
                     rf: float | None = None, usar_cache: bool = True) -> EntradasValoracion:
    """Arma las entradas UNA vez (sin peticiones a yfinance salvo ^TNX; EDGAR solo si no se pasa ``facts``).

    ``modo``: "local" (hasta ``ANIOS_LOCAL`` = 10 anios fiscales + TTM, cache SQLite en disco) o "nube" (``ANIOS_NUBE`` = 3
    anios, SIN cache en disco, ``gc.collect()`` al terminar); por defecto se decide con ``en_nube`` (variable PORT).
    ``anios`` fuerza los anios de historia. ``rf``: Rf ya conocida (si no, ``obtener_rf()`` consulta ^TNX con cache de 24 h).
    Cache de disco (solo local, solo cuando los facts se piden a EDGAR aqui, no cuando se inyectan): clave
    ticker + anios + fecha de la ultima vela; TTL 24 h; al recuperar se refresca el precio con el ``df`` actual.
    XBRL (10-K/10-Q) es la fuente primaria; el ``info`` de yfinance solo rellena lo que falte (marcado en ``fuentes``).
    ``tk_obj`` se acepta por simetria con ``valuation_floors`` pero no se usa. Todo dato faltante queda en None y se
    anota en ``avisos``."""
    info = info or {}
    modo, anios = _resolver_modo(modo, anios)
    from src.reporting import cache_local
    cachear = bool(usar_cache and facts is None and modo == "local" and cache_local.habilitado())
    if cachear:
        previo = cache_local.obtener(_clave_entradas(ticker, anios, df), VERSION_CODIGO)
        if isinstance(previo, dict):
            e = EntradasValoracion.desde_dict(previo)
            if df is not None and not df.empty and "Close" in df.columns:
                e.precio = _f(df["Close"].iloc[-1])
            if _f(info.get("beta")) is not None:
                e.beta = _f(info.get("beta"))
            e.rf = rf if rf is not None else obtener_rf()
            _crecimientos_forward(e, info)
            e.fuentes["cache"] = "cache local en disco (entradas)"
            return e
    e = EntradasValoracion(ticker=ticker, sector=info.get("sector"), industria=info.get("industry"), modo=modo,
                           anios_historia=anios)
    fu, av = e.fuentes, e.avisos
    cot, rep_ = info.get("currency"), info.get("financialCurrency")
    e.moneda_cotizacion, e.moneda_reporte = (str(cot) if cot else None), (str(rep_) if rep_ else None)
    if cot and rep_ and str(cot) != str(rep_):
        e.moneda_conflicto = f"moneda de reporte ({rep_}) ≠ cotización ({cot})"
        av.append(f"{e.moneda_conflicto}: los fundamentales de yfinance NO se usan (no hay conversión FX); solo se "
                  f"usan los de SEC EDGAR si están en la moneda del precio")
    if df is not None and not df.empty and "Close" in df.columns:
        e.precio = _f(df["Close"].iloc[-1])
    e.beta = _f(info.get("beta"))
    if e.beta is not None:
        fu["beta"] = "yfinance info"
    splits = None
    if df is not None and "Stock Splits" in df.columns:
        sp = df["Stock Splits"]
        splits = sp[sp > 0]
        if getattr(splits.index, "tz", None) is not None:
            splits = splits.copy()
            splits.index = splits.index.tz_localize(None)
    edgar_ok = True
    if facts is None:
        try:
            facts = _vf()._facts(ticker)
        except Exception as ex:  # noqa: BLE001
            av.append(f"EDGAR no disponible ({type(ex).__name__}): se usa yfinance donde se pueda")
            facts, edgar_ok = None, False
    if facts is not None:
        try:
            _extraer_xbrl(e, facts, splits, df, anios,
                          info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        except Exception as ex:  # noqa: BLE001
            av.append(f"XBRL incompleto ({type(ex).__name__}: {ex}): se usa yfinance donde se pueda")
            edgar_ok = False
    _completar_con_info(e, info, df)
    _validar_por_accion(e)
    _crecimientos_forward(e, info)
    e.rf = rf if rf is not None else obtener_rf()
    if e.rf is not None:
        fu["rf"] = "^TNX (yfinance), cache 24 h"
    if cachear and edgar_ok and e.fecha_ltm:
        cache_local.guardar(_clave_entradas(ticker, anios, df), e.to_dict(), "entradas", ticker, version=VERSION_CODIGO)
    if modo == "nube":
        import gc
        del facts
        gc.collect()
    return e


def _validar_por_accion(e: EntradasValoracion) -> None:
    """Descarta BVPS/EPS cuyo cociente con el precio es absurdo (datos de otra clase de acciones, p. ej. BRK-A vs BRK-B)."""
    px = _f(e.precio)
    if not px or px <= 0:
        return
    bv, eps = _f(e.bvps), _f(e.eps_ltm)
    if bv is not None and bv > 0 and not PB_IMPLICITO_RANGO[0] <= px / bv <= PB_IMPLICITO_RANGO[1]:
        e.avisos.append(f"BVPS {bv:,.2f} inconsistente con el precio {px:,.2f} (P/B implícito {px / bv:.3g}): descartado")
        e.bvps = None
        e.datos_inconsistentes.append("bvps")
    if eps is not None and eps > 0 and not PE_IMPLICITO_RANGO[0] <= px / eps <= PE_IMPLICITO_RANGO[1]:
        e.avisos.append(f"EPS {eps:,.2f} inconsistente con el precio {px:,.2f} (P/E implícito {px / eps:.3g}): descartado")
        e.eps_ltm = None
        e.datos_inconsistentes.append("eps_ltm")



def _extraer_xbrl(e: EntradasValoracion, facts: pd.DataFrame, splits, df, anios: int = ANIOS_HISTORIA,
                  acciones_info: float | None = None) -> None:
    """Rellena ``e`` desde XBRL: LTM, hasta ``anios`` anios fiscales de historia (+ el LTM actual), series anuales
    alineadas y serie TTM trimestral. La ventana de facts es ``anios + 3`` anios (margen para CAGR a 5Y y multiplos)."""
    fu, av = e.fuentes, e.avisos
    ult = facts["period_end"].max()
    facts = facts[facts["period_end"] >= ult - pd.DateOffset(years=anios + 3)]
    n_hist = anios + 1
    rev_q, ocf_q, cx_q = _flujo(facts, _REV), _flujo(facts, _OCF), _flujo(facts, _CAPEX)
    ebit_q, da_q = _flujo(facts, _EBIT), _flujo(facts, _DA)
    eps_q = _flujo(facts, _EPS, splits, True)
    fecha = max((q.index[-1] for q in (rev_q, ocf_q, eps_q, ebit_q) if not q.empty), default=None)
    if fecha is None or (pd.Timestamp.now() - fecha).days > MAX_ANTIGUEDAD_LTM_DIAS:
        av.append("XBRL sin trimestres recientes (¿emisor extranjero o sin 10-Q?)")
        return
    e.fecha_ltm = str(fecha.date())

    def vigente(q):  # la serie debe terminar en el mismo trimestre que el resto
        return (not q.empty) and abs((q.index[-1] - fecha).days) <= 100

    imp_q, pre_q, int_q = _flujo(facts, _IMPUESTO), _flujo(facts, _ANTES_IMP), _flujo(facts, _INTERES)
    nombre_ebit = "OperatingIncomeLoss"
    if not vigente(ebit_q) and vigente(pre_q) and vigente(int_q):   # sin OperatingIncomeLoss (p. ej. XOM)
        idx = pre_q.index.intersection(int_q.index)
        ebit_q = (pre_q.reindex(idx) + int_q.reindex(idx).abs()).dropna()
        nombre_ebit = "utilidad antes de impuestos + gasto por intereses (sin OperatingIncomeLoss)"
    fcf_q = pd.Series(dtype=float)
    if vigente(ocf_q) and vigente(cx_q):
        idx = ocf_q.index.intersection(cx_q.index)
        fcf_q = (ocf_q.reindex(idx) - cx_q.reindex(idx)).dropna()
    ebitda_q = pd.Series(dtype=float)
    if vigente(ebit_q) and vigente(da_q):
        idx = ebit_q.index.intersection(da_q.index)
        ebitda_q = (ebit_q.reindex(idx) + da_q.reindex(idx)).dropna()
    neto_q, div_q = _flujo(facts, _NETO), _flujo(facts, _DIVIDENDOS)
    for campo, q, nombre in (("fcf_ltm", fcf_q, "OCF - CapEx"), ("ebitda_ltm", ebitda_q, "EBIT + D&A"),
                             ("ebit_ltm", ebit_q, nombre_ebit), ("ingresos_ltm", rev_q, "Revenues"),
                             ("eps_ltm", eps_q, "EPS diluido"), ("da_ltm", da_q, "D&A"), ("capex_ltm", cx_q, "CapEx"),
                             ("neto_ltm", neto_q, "NetIncomeLoss"), ("ocf_ltm", ocf_q, "OCF"),
                             ("dividendos_pagados_ltm", div_q, "PaymentsOfDividends")):
        v = _ltm(q) if vigente(q) else None
        if v is not None:
            setattr(e, campo, v)
            fu[campo] = f"SEC EDGAR XBRL ({nombre})"
    if e.capex_ltm is not None:
        e.capex_ltm = abs(e.capex_ltm)
    if e.dividendos_pagados_ltm is not None:
        e.dividendos_pagados_ltm = abs(e.dividendos_pagados_ltm)
    fechas_por: dict[str, list] = {}
    for campo, q in (("hist_fcf", fcf_q), ("hist_ebitda", ebitda_q), ("hist_ingresos", rev_q), ("hist_eps", eps_q)):
        if vigente(q):
            vals, fs = _historia_fechas(q, n_hist)
            setattr(e, campo, vals)
            fechas_por[campo] = fs
    fechas_hist = next((fechas_por[c] for c in ("hist_ingresos", "hist_ebitda", "hist_fcf", "hist_eps")
                        if fechas_por.get(c)), [])
    # Impuestos y intereses
    imp, pre = (_ltm(imp_q) if vigente(imp_q) else None), (_ltm(pre_q) if vigente(pre_q) else None)
    if imp is not None and pre is not None and pre > 0:
        e.tasa_impositiva = _clip(imp / pre, 0.0, TASA_IMPOSITIVA_MAX)
        fu["tasa_impositiva"] = "SEC EDGAR XBRL (impuesto / utilidad antes de impuestos, LTM)"
    gi = _ltm(int_q) if vigente(int_q) else None
    if gi is not None:
        e.gasto_interes_ltm = abs(gi)
    # Balance
    caja_s, deuda_s = _caja_serie(facts), _deuda_serie(facts)
    sh_dil = _acciones_serie(facts, splits)
    sh_q = _depurar_acciones(_acciones_serie(facts, splits, (_ACCIONES_DIL, _ACCIONES_BAS)))   # historia: diluido y, si falta, basico
    eq_s = _instante(facts, _EQUITY)
    c, d = _asof(caja_s, fecha, 150), _asof(deuda_s, fecha, 150)
    if c is not None:
        e.caja = c
        fu["caja"] = "SEC EDGAR XBRL (efectivo + inversiones corrientes; sin inversiones no corrientes)"
    if d is not None:
        e.deuda_total = d
        fu["deuda_total"] = "SEC EDGAR XBRL (deuda financiera; sin arrendamientos)"
    # Acciones actuales: promedio diluido reciente; si falta o esta desactualizado, portada (dei); si no, promedio basico.
    # Con la portada o el basico NO se incluyen las acciones potenciales (opciones, convertibles): subestima la dilucion;
    # con el diluido de empresas con convertibles "if-converted" (PTON) y la deuda restada aparte hay doble dilucion.
    sh = _asof(sh_dil, fecha, 150)
    origen = "SEC EDGAR XBRL (promedio ponderado diluido, ajustado por splits)"
    if not sh or sh <= 0:
        sh, origen = _acciones_portada(facts, fecha), "SEC EDGAR XBRL (portada dei:EntityCommonStockSharesOutstanding; sin dilucion)"
    if not sh or sh <= 0:
        sh, origen = _asof(sh_q, fecha, 150), "SEC EDGAR XBRL (promedio ponderado basico; sin dilucion)"
    if sh and sh > 0:
        e.acciones_diluidas = sh
        fu["acciones_diluidas"] = origen
        if "diluido" not in origen:
            e.acciones_nota = origen
    sh_hist = sh if sh and sh > 0 else (_f(acciones_info) if acciones_info and not e.moneda_conflicto else None)
    if sh_hist and sh_hist > 0 and not rev_q.empty and _asof(sh_q, fecha, 150) is None:   # p. ej. V: sin promedio reciente
        sh_q = pd.Series({d: (_asof(sh_q, d, 150) or float(sh_hist)) for d in rev_q.index}, dtype=float).sort_index()
        av.append("acciones históricas no disponibles en XBRL: donde faltan se asume el número actual de acciones "
                  "(aproximación para múltiplos y serie historica)")
    eq = _asof(eq_s, fecha, 150)
    if eq is not None:
        e.patrimonio = eq
    if eq is not None and sh and sh > 0:
        e.bvps = eq / sh
        fu["bvps"] = "SEC EDGAR XBRL (patrimonio / acciones)"
    ac, pc = _asof(_instante(facts, _ACTIVO_CORR), fecha, 150), _asof(_instante(facts, _PASIVO_CORR), fecha, 150)
    if ac is not None and pc is not None and c is not None:
        deuda_cp = _asof(_instante(facts, _DEUDA_CP + ["us-gaap:LongTermDebtCurrent"]), fecha, 150) or 0.0
        e.nwc_operativo = (ac - c) - (pc - deuda_cp)
        fu["nwc_operativo"] = "SEC EDGAR XBRL ((activo corriente - caja) - (pasivo corriente - deuda corto plazo))"
    # Series anuales alineadas (cada ejercicio = LTM al cierre del trimestre correspondiente)
    if fechas_hist:
        e.hist_fechas = [str(f.date()) for f in fechas_hist]
        ttm = {k: _ttm_rolling(q) for k, q in (("ebit", ebit_q), ("neto", neto_q), ("ocf", ocf_q), ("capex", cx_q),
                                               ("div", div_q), ("imp", imp_q), ("pre", pre_q))}
        e.hist_ebit = [_valor_en(ttm["ebit"], f) for f in fechas_hist]
        e.hist_neto = [_valor_en(ttm["neto"], f) for f in fechas_hist]
        e.hist_ocf = [_valor_en(ttm["ocf"], f) for f in fechas_hist]
        e.hist_capex = [abs(v) if v is not None else None for v in (_valor_en(ttm["capex"], f) for f in fechas_hist)]
        e.hist_dividendos = [abs(v) if v is not None else None for v in (_valor_en(ttm["div"], f) for f in fechas_hist)]
        tasas = []
        for f in fechas_hist:
            i_, p_ = _valor_en(ttm["imp"], f), _valor_en(ttm["pre"], f)
            tasas.append(_clip(i_ / p_, 0.0, TASA_IMPOSITIVA_MAX) if i_ is not None and p_ and p_ > 0 else None)
        e.hist_tasa = tasas
        e.hist_deuda = [_asof(deuda_s, f, 150) for f in fechas_hist]
        e.hist_caja = [_asof(caja_s, f, 150) for f in fechas_hist]
        e.hist_patrimonio = [_asof(eq_s, f, 150) for f in fechas_hist]
        e.hist_acciones = [_asof(sh_dil, f, 150) or _asof(sh_q, f, 150) for f in fechas_hist]
    # Serie TTM trimestral (insumo de la serie historica del valor justo)
    e.serie_ttm = _construir_serie_ttm(
        {"ing": rev_q, "ebit": ebit_q, "ebitda": ebitda_q, "fcf": fcf_q, "eps": eps_q, "neto": neto_q, "da": da_q,
         "capex": cx_q, "interes": int_q, "impuesto": imp_q, "pre": pre_q, "ocf": ocf_q},
        {"caja": caja_s, "deuda": deuda_s, "patrimonio": eq_s, "acciones": sh_q, "ac": _instante(facts, _ACTIVO_CORR),
         "pc": _instante(facts, _PASIVO_CORR), "dcp": _instante(facts, _DEUDA_CP + ["us-gaap:LongTermDebtCurrent"])})
    # Multiplos historicos propios
    if df is not None and not df.empty:
        idx = pd.DatetimeIndex(df.index)
        precio = _precio_sin_ajuste_dividendos(df)
        precio.index = idx.tz_localize(None) if idx.tz is not None else idx
        e.multiplos_hist = _instantaneas_multiplos(
            precio, eps_q if vigente(eps_q) else pd.Series(dtype=float), fcf_q, ebitda_q,
            rev_q if vigente(rev_q) else pd.Series(dtype=float), sh_q, caja_s, deuda_s, eq_s, k_max=anios)


def _completar_con_info(e: EntradasValoracion, info: dict, df) -> None:
    """Rellena SOLO lo que falta con yfinance ``info`` y el ``df`` (dividendos); lo marca en ``fuentes``."""
    fu = e.fuentes
    mismatch = bool(e.moneda_conflicto)   # reporte != cotizacion: los importes de ``info`` no son comparables con el precio

    def poner(campo, valor, nota="yfinance info"):
        v = _f(valor)
        if not mismatch and getattr(e, campo) is None and v is not None:
            setattr(e, campo, v)
            fu[campo] = nota
    for campo, clave in (("caja", "totalCash"), ("deuda_total", "totalDebt")):
        if mismatch:
            break
        x, i = _f(getattr(e, campo)), _f(info.get(clave))
        if x is not None and i is not None and i > 0 and x < SUBCONTEO_INFO * i:  # concepto XBRL incompleto/desfasado
            e.avisos.append(f"{campo}: XBRL ({x:,.0f}) < {SUBCONTEO_INFO:.0%} del valor de yfinance ({i:,.0f}); "
                            f"se usa yfinance")
            setattr(e, campo, None)
        elif campo == "caja" and x is not None and i is not None and i > 0 and abs(x - i) / i > UMBRAL_AVISO_CAJA:
            # diferencia moderada: no se sobrescribe (yfinance puede incluir inversiones a largo plazo; el XBRL de
            # algunos emisores etiqueta solo el total de valores negociables, corrientes y no corrientes)
            e.avisos.append(f"caja: XBRL ({x:,.0f}) difiere {abs(x - i) / i:.0%} de yfinance ({i:,.0f}); se usa XBRL "
                            f"(solo inversiones corrientes cuando el emisor las distingue)")
    poner("fcf_ltm", info.get("freeCashflow"))
    poner("ebitda_ltm", info.get("ebitda"))
    poner("ingresos_ltm", info.get("totalRevenue"))
    poner("eps_ltm", info.get("trailingEps"))
    poner("bvps", info.get("bookValue"))
    poner("caja", info.get("totalCash"))
    poner("deuda_total", info.get("totalDebt"))
    poner("acciones_diluidas", info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
    if e.ebit_ltm is None and _f(info.get("operatingMargins")) is not None and e.ingresos_ltm:
        e.ebit_ltm = float(info["operatingMargins"]) * e.ingresos_ltm
        fu["ebit_ltm"] = "yfinance info (margen operativo x ingresos)"
    # Dividendo anual: suma de los ultimos 12 meses del df (dato propio); respaldo en info.
    div = None
    if df is not None and "Dividends" in getattr(df, "columns", []) and len(df):
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        s = df["Dividends"].fillna(0.0).astype(float).copy()
        s.index = idx
        div = float(s[s.index > idx[-1] - pd.Timedelta(days=365)].sum())
        if div > 0:
            fu["dividendo_anual"] = "yfinance history (dividendos de los ultimos 12 meses)"
    if div is None or div <= 0:
        alt = None if mismatch else (_f(info.get("trailingAnnualDividendRate")) or _f(info.get("dividendRate")))
        if alt:
            div, fu["dividendo_anual"] = alt, "yfinance info"
        elif div is not None:
            fu["dividendo_anual"] = "yfinance history (sin dividendos en 12 meses)"
    e.dividendo_anual = div
    if e.fcf_ltm is None:
        e.avisos.append("FCF no disponible")


# --- Serie historica del Valor Justo (estilo GuruFocus GF Value) -----------------------------------------------------
def _hash_estable(*objs: Any) -> str:
    import hashlib
    return hashlib.md5(repr(objs).encode("utf-8")).hexdigest()[:12]


def _cercano(fechas: np.ndarray, objetivo: pd.Timestamp, tol_dias: int) -> int | None:
    """Indice de la fecha mas cercana a ``objetivo`` (dentro de ``tol_dias``) en el arreglo ordenado ``fechas``."""
    if len(fechas) == 0:
        return None
    i = int(np.searchsorted(fechas, np.datetime64(objetivo)))
    mejor, dmin = None, None
    for j in (i - 1, i):
        if 0 <= j < len(fechas):
            d = abs((pd.Timestamp(fechas[j]) - objetivo).days)
            if d <= tol_dias and (dmin is None or d < dmin):
                mejor, dmin = j, d
    return mejor


def _entradas_en(e: EntradasValoracion, rows: list[dict], fechas_rows: np.ndarray, i: int, t: pd.Timestamp,
                 px_raw: pd.Series, div: pd.Series | None, k_max: int) -> EntradasValoracion | None:
    """``EntradasValoracion`` de la fecha de PUBLICACION ``t`` del cierre ``rows[i]``: solo usa filas ``<= i`` y precios
    ``<= t`` (sin look-ahead). Rf, ERP, beta, sector y moneda son los VIGENTES (no son PIT). Sin forward de yfinance."""
    r = rows[i]
    px_t = _px_asof(px_raw, t)
    sh = _f(r.get("acciones"))
    if px_t is None or not sh or sh <= 0:
        return None
    d_i = pd.Timestamp(r["fecha"])
    pre, imp = _f(r.get("pre")), _f(r.get("impuesto"))
    eq = _f(r.get("patrimonio"))
    ac, pc, caja, dcp = _f(r.get("ac")), _f(r.get("pc")), _f(r.get("caja")), _f(r.get("dcp")) or 0.0
    ent = EntradasValoracion(
        ticker=e.ticker, precio=px_t, acciones_diluidas=sh, fcf_ltm=_f(r.get("fcf")), ebitda_ltm=_f(r.get("ebitda")),
        ebit_ltm=_f(r.get("ebit")), ingresos_ltm=_f(r.get("ing")), eps_ltm=_f(r.get("eps")),
        bvps=eq / sh if eq is not None else None, caja=caja, deuda_total=_f(r.get("deuda")),
        tasa_impositiva=_clip(imp / pre, 0.0, TASA_IMPOSITIVA_MAX) if imp is not None and pre and pre > 0 else None,
        capex_ltm=abs(_f(r.get("capex"))) if _f(r.get("capex")) is not None else None, da_ltm=_f(r.get("da")),
        nwc_operativo=(ac - caja) - (pc - dcp) if ac is not None and pc is not None and caja is not None else None,
        gasto_interes_ltm=abs(_f(r.get("interes"))) if _f(r.get("interes")) is not None else None,
        beta=e.beta, rf=e.rf, sector=e.sector, industria=e.industria, moneda_conflicto=e.moneda_conflicto)
    if div is not None and len(div):
        ent.dividendo_anual = float(div[(div.index > t - pd.Timedelta(days=365)) & (div.index <= t)].sum())
    # Series anuales hasta d_i: un punto por ejercicio (filas ~1 anio antes cada una)
    for campo, clave in (("hist_fcf", "fcf"), ("hist_ingresos", "ing"), ("hist_ebitda", "ebitda"), ("hist_eps", "eps")):
        vals = []
        for k in range(ANIOS_HISTORIA, -1, -1):
            j = _cercano(fechas_rows[:i + 1], d_i - pd.DateOffset(years=k), 50)
            v = _f(rows[j].get(clave)) if j is not None else None
            if v is not None:
                vals.append(v)
        setattr(ent, campo, vals)
    # Multiplos propios en k = 1..k_max anios antes (snapshot publicado hace >= 1 anio: nunca posterior a t)
    mh: dict[str, list[float]] = {k: [] for k in RANGOS_MULTIPLOS}
    for k in range(k_max, 0, -1):
        j = _cercano(fechas_rows[:i + 1], d_i - pd.DateOffset(years=k), 50)
        if j is None:
            continue
        rj = rows[j]
        pj = _px_asof(px_raw, pd.Timestamp(rj["fecha"]) + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS))
        shj = _f(rj.get("acciones"))
        if pj is None or not shj:
            continue
        cap = pj * shj
        eps_j, fcf_j, eb_j, rev_j, eq_j = (_f(rj.get(x)) for x in ("eps", "fcf", "ebitda", "ing", "patrimonio"))
        if eps_j and eps_j > 0:
            mh["pe"].append(pj / eps_j)
        if fcf_j and fcf_j > 0:
            mh["p_fcf"].append(cap / fcf_j)
        c_j, d_j = _f(rj.get("caja")), _f(rj.get("deuda"))
        if c_j is not None and d_j is not None:
            if eb_j and eb_j > 0:
                mh["ev_ebitda"].append((cap + d_j - c_j) / eb_j)
            if rev_j and rev_j > 0:
                mh["ev_revenue"].append((cap + d_j - c_j) / rev_j)
        if eq_j and eq_j > 0:
            mh["pb"].append(cap / eq_j)
    ent.multiplos_hist = mh
    _validar_por_accion(ent)
    return ent


def _px_asof(s: pd.Series, t: pd.Timestamp, max_dias: int = 10) -> float | None:
    """Ultimo cierre en o antes de ``t`` (maximo ``max_dias`` de antiguedad); None si no hay."""
    i = int(s.index.searchsorted(t, side="right")) - 1
    if i < 0 or (t - s.index[i]).days > max_dias:
        return None
    return float(s.iloc[i])


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def suavizar_causal(eventos: list[tuple[pd.Timestamp, pd.Timestamp, float]], grilla: pd.DatetimeIndex) -> np.ndarray:
    """Evalua en ``grilla`` la curva que, en cada evento ``(inicio, fin, valor)``, transita de forma monotona (smoothstep,
    C1) del valor anterior al nuevo entre ``inicio`` y ``fin`` y se mantiene hasta el siguiente evento. Es CAUSAL: el valor
    en ``x`` solo depende de eventos con ``inicio <= x`` (un spline interpolante entre reportes usaria el reporte
    siguiente = look-ahead). Antes del primer evento devuelve NaN."""
    ini = np.array([pd.Timestamp(a).value for a, _, _ in eventos], dtype=float)
    fin = np.array([pd.Timestamp(b).value for _, b, _ in eventos], dtype=float)
    val = np.array([v for _, _, v in eventos], dtype=float)
    x = grilla.values.astype("datetime64[ns]").astype(np.int64).astype(float)
    k = np.searchsorted(ini, x, side="right") - 1
    out = np.full(len(x), np.nan)
    ok = k >= 0
    kk = np.where(ok, k, 0)
    prev = np.where(kk > 0, val[np.maximum(kk - 1, 0)], val[kk])
    dur = np.maximum(fin[kk] - ini[kk], 1.0)
    u = (x - ini[kk]) / dur
    out[ok] = (prev + (val[kk] - prev) * _smoothstep(u))[ok]
    return out


def _fmt_banda(p: int) -> str:
    return f"{'+' if p > 0 else '-'}{abs(p)}%"


def _g_proyeccion(modelos_medio: list[dict]) -> tuple[float, str]:
    """Tasa de proyeccion ILUSTRATIVA: ``PROYECCION_AMORTIGUACION*g1_medio + (1-PROYECCION_AMORTIGUACION)*g∞``, con g1_medio =
    promedio de la etapa 1 de los DCF aplicables del Medio; acotada a [0, PROYECCION_TOPE]."""
    g1s = [m["g1"] for m in modelos_medio if m.get("aplicable") and m.get("g1") is not None]
    ginf = G_TERMINAL["medio"]
    if not g1s:
        return ginf, "sin DCF aplicables: g∞ del Medio"
    g = PROYECCION_AMORTIGUACION * float(np.mean(g1s)) + (1.0 - PROYECCION_AMORTIGUACION) * ginf
    return _clip(g, 0.0, PROYECCION_TOPE), (f"{PROYECCION_AMORTIGUACION:.0%} x g etapa 1 medio de los DCF "
                                            f"({np.mean(g1s) * 100:.1f} %) + {1 - PROYECCION_AMORTIGUACION:.0%} x g∞")


def serie_valor_justo_historica(entradas: Any, df: pd.DataFrame, params: Any = None,
                                modelos_activos: list[str] | None = None, modo: str | None = None,
                                anios: int | None = None, usar_cache: bool = True,
                                paso_dias: int = PASO_SERIE_DIAS) -> dict:
    """Curva historica del Valor Justo (escenario MEDIO) con 6 bandas simetricas y proyeccion ilustrativa de 12 meses.

    Metodo
    ------
    1. Para cada cierre trimestral de ``entradas.serie_ttm`` (TTM de EDGAR) se arma un ``EntradasValoracion`` valido en su
       FECHA DE PUBLICACION (cierre + ``REZAGO_PUBLICACION_DIAS`` = 45 d) y se corre el motor Medio completo (mismos 15
       modelos y misma consolidacion por familias que el valor actual). PIT estricto: cada punto usa solo filas y precios
       publicados hasta su fecha; la mediana de multiplos en t usa solo ejercicios >= 1 anio anteriores; el crecimiento
       forward de yfinance NO se usa (no es PIT).
    2. Suavizado CAUSAL entre reportes (``suavizar_causal``): tras cada reporte la curva transita en ``TRANSICION_DIAS`` (45 d)
       con smoothstep monotono (sin escalones, sin overshoot) y se mantiene hasta el siguiente. Se prefiere a un spline
       interpolante porque este usaria el reporte siguiente (look-ahead).
    3. ANCLA: el ultimo punto (ultima vela) es el valor justo Medio ACTUAL (``entradas`` completas, con forward y el ultimo
       balance); el tramo final se ajusta para aterrizar en el (es lo unico no-PIT, y se informa en ``notas``/``ancla``).
    4. Bandas: valor justo x (1 + p), p en {+30, +20, +10, -10, -20, -30} %.
    5. Proyeccion (ILUSTRATIVA, no una prediccion): el valor justo crece ``g_proyeccion`` anual durante 12 meses desde el
       ancla (``_g_proyeccion``: 50 % del crecimiento de etapa 1 de los DCF Medio + 50 % de g∞, tope 15 %).
    No son PIT y se declaran: Rf (^TNX), ERP, beta, sector y los sliders del usuario (WACC/g) son los VIGENTES para todas
    las fechas. Con menos de MIN_PUNTOS_MULTIPLO ejercicios previos los multiplos no aplican y la curva se apoya en DCF y
    Valor (en nube, 3 anios de historia, hay menos puntos).
    Devuelve estructuras ligeras (fechas ISO y floats; None donde no hay dato): {ticker, modo, anios, fechas, valor_justo,
    precio, bandas{"+30%"...}, proyeccion{fechas, valor_justo, bandas, g_anual, metodo}, ancla{fecha, valor_justo},
    puntos_trimestrales[...], notas[...], ilustrativa_proyeccion, desde_cache, tiempo_s}. Con cache de disco en local."""
    import time
    t0 = time.time()
    e, p = _como_entradas(entradas), _como_params(params)
    modo, anios = _resolver_modo(modo or e.modo, anios or (None if modo else e.anios_historia))
    ticker = e.ticker or ""
    base = {"ticker": ticker, "modo": modo, "anios": anios, "fechas": [], "valor_justo": [], "precio": [],
            "bandas": {_fmt_banda(b): [] for b in BANDAS_PCT},
            "proyeccion": {"fechas": [], "valor_justo": [], "bandas": {_fmt_banda(b): [] for b in BANDAS_PCT},
                           "g_anual": None, "metodo": None},
            "ancla": None, "puntos_trimestrales": [], "notas": [], "ilustrativa_proyeccion": True,
            "escenario": "medio", "desde_cache": False, "tiempo_s": 0.0}
    notas = base["notas"]
    if df is None or len(df) == 0 or "Close" not in df.columns:
        notas.append("sin precios: no se puede construir la serie")
        return base
    from src.reporting import cache_local
    cachear = bool(usar_cache and modo == "local" and cache_local.habilitado())
    clave = f"serie_vj:{ticker.upper()}:{anios}:{_ultima_fecha(df)}:{_hash_estable(asdict(p), sorted(modelos_activos) if modelos_activos is not None else None, paso_dias, e.fecha_ltm, e.precio)}"
    if cachear:
        previo = cache_local.obtener(clave, VERSION_CODIGO)
        if isinstance(previo, dict) and previo.get("fechas"):
            previo["desde_cache"], previo["tiempo_s"] = True, round(time.time() - t0, 4)
            return previo
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    px_raw = _precio_sin_ajuste_dividendos(df)
    px_raw.index = idx
    div = None
    if "Dividends" in df.columns:
        div = df["Dividends"].fillna(0.0).astype(float).copy()
        div.index = idx
        div = div[div > 0]
    ultima = idx[-1]
    rows = [r for r in (e.serie_ttm or []) if r.get("fecha")]
    fechas_rows = np.array([np.datetime64(pd.Timestamp(r["fecha"])) for r in rows])
    inicio = ultima - pd.DateOffset(years=anios)
    notas.append("Parámetros de mercado NO point-in-time: Rf (^TNX), ERP, beta, sector y los sliders de WACC/g son los "
                 "vigentes en todas las fechas.")
    notas.append("Suavizado causal: tras cada reporte el valor justo transita en "
                 f"{TRANSICION_DIAS} días; no se usa el reporte siguiente (sin look-ahead).")
    notas.append("El crecimiento forward de yfinance no se usa en los puntos históricos (no es PIT); el último punto (ancla) "
                 "sí lo incluye.")
    k_max = min(anios, 10)
    puntos: list[dict] = []
    descartados: list[str] = []
    for i, r in enumerate(rows):
        t = pd.Timestamp(r["fecha"]) + pd.Timedelta(days=REZAGO_PUBLICACION_DIAS)
        if t < inicio - pd.Timedelta(days=100) or t > ultima - pd.Timedelta(days=ANCLA_SEPARACION_DIAS):
            continue
        ent = _entradas_en(e, rows, fechas_rows, i, t, px_raw, div, k_max)
        if ent is None:
            continue
        veci = [_f(rows[j].get("acciones")) for j in range(max(0, i - 4), min(len(rows), i + 5)) if j != i]
        veci = [x for x in veci if x and x > 0]
        if veci and ent.acciones_diluidas:
            m_v = float(np.median(veci))
            if not m_v / SALTO_ACCIONES_X <= ent.acciones_diluidas <= m_v * SALTO_ACCIONES_X:
                descartados.append(f"{r['fecha']}: acciones {ent.acciones_diluidas:,.0f} difieren > {SALTO_ACCIONES_X:g}x de sus vecinas")
                continue
        ms = calcular_modelos(ent, p, "medio")
        agg = agregar_valor_justo(ms, modelos_activos, p.margen_seguridad, ent.precio)
        if agg["fair_value"] is None:
            continue
        if ent.precio and not ent.precio / VJ_ABSURDO_X <= agg["fair_value"] <= ent.precio * VJ_ABSURDO_X:
            descartados.append(f"{r['fecha']}: valor justo {agg['fair_value']:,.2f} absurdo frente al precio {ent.precio:,.2f} "
                               f"(fuera de [1/{VJ_ABSURDO_X:g}x, {VJ_ABSURDO_X:g}x])")
            continue
        puntos.append({"fecha_cierre": r["fecha"], "fecha_publicacion": str(t.date()),
                       "valor_justo": float(agg["fair_value"]), "n_modelos": agg["n_usados"]})
    if descartados:
        notas.append(f"{len(descartados)} punto(s) trimestral(es) descartado(s) por dato absurdo (la curva no los usa): "
                     + "; ".join(descartados))
    base["puntos_descartados"] = len(descartados)
    # Ancla: valor justo Medio ACTUAL
    ms_act = calcular_modelos(e, p, "medio")
    agg_act = agregar_valor_justo(ms_act, modelos_activos, p.margen_seguridad, e.precio)
    v_act = agg_act["fair_value"]
    if v_act is None and not puntos:
        notas.append("sin valor justo calculable (datos insuficientes)")
        base["tiempo_s"] = round(time.time() - t0, 4)
        return base
    if v_act is None:
        notas.append("sin valor justo actual: no hay ancla; la serie termina en el ultimo reporte")
    base["puntos_trimestrales"] = puntos
    ev: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    for k, pt in enumerate(puntos):
        ini = pd.Timestamp(pt["fecha_publicacion"])
        sig = pd.Timestamp(puntos[k + 1]["fecha_publicacion"]) if k + 1 < len(puntos) else None
        fin = ini + pd.Timedelta(days=TRANSICION_DIAS)
        if sig is not None:
            fin = min(fin, sig)
        ev.append((ini, fin, pt["valor_justo"]))
    ancla_info = None
    if v_act is not None:
        ini_a = ultima - pd.Timedelta(days=TRANSICION_DIAS)
        if ev and ev[-1][0] > ini_a:              # ultimo reporte reciente: el tramo hasta el ancla se reparte a mitades
            ini_a = ev[-1][0] + (ultima - ev[-1][0]) / 2
        if ev:
            ev[-1] = (ev[-1][0], min(ev[-1][1], ini_a), ev[-1][2])
        ev.append((ini_a, ultima, float(v_act)))
        prev = ev[-2][2] if len(ev) > 1 else None
        ancla_info = {"fecha": str(ultima.date()), "valor_justo": float(v_act), "precio": _f(e.precio),
                      "salto_vs_reporte_previo_pct": ((v_act / prev - 1.0) * 100.0) if prev else None}
        base["ancla"] = ancla_info
    ini_s = max(ev[0][0], inicio) if ev else inicio
    grilla = pd.date_range(ini_s, ultima, freq=f"{max(1, int(paso_dias))}D")
    if len(grilla) == 0 or grilla[-1] != ultima:
        grilla = grilla.append(pd.DatetimeIndex([ultima]))
    vj = suavizar_causal(ev, grilla)
    ok = np.isfinite(vj)
    grilla, vj = grilla[ok], vj[ok]
    base["fechas"] = [str(d.date()) for d in grilla]
    base["valor_justo"] = [round(float(v), 4) for v in vj]
    base["precio"] = [(_px_asof(px_raw, d) if d <= ultima else None) for d in grilla]
    base["precio"] = [round(v, 4) if v is not None else None for v in base["precio"]]
    base["bandas"] = {_fmt_banda(b): [round(float(v) * (1 + b / 100.0), 4) for v in vj] for b in BANDAS_PCT}
    if v_act is not None:
        g, metodo = _g_proyeccion(ms_act)
        dias = np.arange(0, PROYECCION_DIAS + 1, max(1, int(paso_dias)))
        if dias[-1] != PROYECCION_DIAS:
            dias = np.append(dias, PROYECCION_DIAS)
        vp = float(v_act) * (1.0 + g) ** (dias / 365.0)
        base["proyeccion"] = {
            "fechas": [str((ultima + pd.Timedelta(days=int(d))).date()) for d in dias],
            "valor_justo": [round(float(v), 4) for v in vp],
            "bandas": {_fmt_banda(b): [round(float(v) * (1 + b / 100.0), 4) for v in vp] for b in BANDAS_PCT},
            "g_anual": g, "metodo": metodo + "; ILUSTRATIVA, no una predicción"}
    if not puntos:
        notas.append("sin reportes trimestrales utilizables en la ventana: solo el valor justo actual")
    base["tiempo_s"] = round(time.time() - t0, 4)
    if cachear:
        cache_local.guardar(clave, base, "serie_vj", ticker, version=VERSION_CODIGO)
    if modo == "nube":
        import gc
        gc.collect()
    return base
