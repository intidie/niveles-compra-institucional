"""Valor justo por bateria de 15 modelos (DCF, multiplos, dividendos, valor), estilo InvestingPro.

AVISO HONESTO: la metodologia de InvestingPro es propietaria. Esto es una implementacion PROPIA "estilo" InvestingPro
(misma familia de modelos y misma idea de consolidar en un unico valor justo), NO replica sus cifras. Es un valor
estimado por modelos con supuestos explicitos (WACC, g, multiplos objetivo); no es un precio objetivo, ni una
prediccion, ni asesoria financiera. Es ADITIVO: no modifica ni sustituye los niveles tecnicos de compra ni los
pisos de ``valuation_floors``.

Flujo (sin red salvo ``extraer_entradas``)
------------------------------------------
1. ``extraer_entradas(ticker, tk_obj, df, info)`` arma UNA vez un ``EntradasValoracion`` (liviano, sin DataFrames,
   serializable con ``to_dict``) reutilizando los helpers XBRL de ``valuation_floors`` (``_facts``, ``_flujo_trimestral``,
   ``_serie_concepto``, ``_factor_split``). No hace ninguna peticion a yfinance (usa el ``df`` y el ``info`` ya
   descargados); la unica consulta externa es ``Company.get_facts()`` de EDGAR (que edgartools reutiliza de su cache al
   ser la misma que hace ``calcular_pisos_valoracion``). Todo dato que no se pueda obtener es ``None``, nunca 0.
2. ``calcular_modelos(entradas, params)`` devuelve la lista de 15 modelos (funcion pura).
3. ``agregar_valor_justo(modelos, modelos_activos, margen_seguridad, precio_actual)`` consolida (funcion pura).
   La UI recalcula 2 y 3 en memoria al mover un slider o una casilla.

Modelos (ids estables)
----------------------
Tasas: WACC = E/(D+E)*ke + D/(D+E)*kd*(1-t) con ke = rf + beta*ERP (CAPM), acotado a [WACC_MIN, WACC_MAX]. Un WACC
del usuario (``ParametrosValoracion.wacc``) reemplaza al automatico y desplaza ke en paralelo (el delta del WACC
dividido por el peso del patrimonio, con piso ``WE_MIN_AJUSTE_KE``, que es lo que mueve ke si kd no cambia). La tasa
impositiva efectiva LTM tiene piso ``TASA_MIN`` (una tasa de 0,5 % de un ano no se proyecta a perpetuidad).
Todas las tasas, rf, ERP y multiplos por defecto son PARAMETROS (no datos de mercado en vivo).

* dcf_5y_fcf / dcf_10y_fcf: FCF (OCF - CapEx, XBRL) llevado a FCFF sumando los intereses despues de impuestos
  (``FCF + gasto_interes_ltm*(1-t)``; sin gasto de intereses se suma 0): el OCF ya esta neto de intereses y se descuenta
  al WACC restando ademas la deuda, asi que sin este ajuste los intereses se contarian dos veces. P/FCF NO se ajusta (es
  un multiplo de patrimonio). Se proyecta N anios; el crecimiento parte de ``min(CRECIMIENTO_TOPE, HAIRCUT*CAGR
  historico)`` y converge linealmente hacia g en el anio N; valor terminal de Gordon ``FCF_N*(1+g)/(WACC-g)``
  descontado; + caja - deuda total; / acciones diluidas. Si el flujo es CICLICO (``es_ciclico``: CV > ``VOL_FCF_ALTA`` y
  >= ``MIN_INVERSIONES_CICLO`` cambios de direccion) la base es la media de los ultimos ``N_NORMALIZACION`` anios y el
  crecimiento inicial es g ("flujo ciclico normalizado"): extrapolar valle-pico de un ciclo de commodities infla el DCF.
  Con historia pero menos de ``MIN_ANIOS_FCF`` anios de FCF positivo el DCF de FCF no aplica (no se extrapola un solo
  ano bueno).
* dcf_5y_ebitda / dcf_10y_ebitda: EBITDA = EBIT + D&A (LTM) proyectado igual; flujo anual
  ``EBITDA*(1-t) + D&A*t - CapEx - dNWC`` con D&A y CapEx como proporcion constante del EBITDA LTM, dNWC =
  ``nwc/EBITDA`` (acotado) por la variacion del EBITDA (0 si no hay balance), t = tasa efectiva (21 % si falta).
* dcf_5y_revenue / dcf_10y_revenue: ingresos proyectados con el mismo crecimiento (CAGR de ingresos) y margen FCF que
  converge linealmente desde el margen LTM hasta ``min(MARGEN_FCF_TOPE, mediana historica del margen FCF)``.
* pe: EPS LTM * P/E objetivo; ev_ebitda: (EBITDA*m + caja - deuda)/acciones; ev_revenue analogo; pb: BVPS * P/B
  objetivo; p_fcf: FCF/accion * P/FCF objetivo. Multiplo objetivo = MEDIANA de los multiplos propios de los ultimos
  anios (instantaneas anuales con el precio sin ajuste por dividendos y 45 dias de rezago de publicacion; minimo
  ``MIN_PUNTOS_MULTIPLO`` puntos) acotada a ``RANGOS_MULTIPLOS``; o el valor fijado por el usuario.
* ddm: Gordon ``D0*(1+g)/(ke-g)``; no aplica sin dividendo, con rendimiento < ``DDM_RENDIMIENTO_MIN`` (dividendo
  simbolico) o con ke <= g.
* epv (Greenwald): ``EBIT*(1-t)/WACC`` (sin crecimiento; el ajuste de capex de mantenimiento supone capex de
  mantenimiento = D&A, asi que NOPAT ajustado = NOPAT) + caja - deuda, / acciones.
* graham: ``sqrt(22.5*EPS*BVPS)``; lynch (PEG = 1): ``EPS * min(25, CAGR esperado de EPS en %)``.

Agregacion (``agregar_valor_justo``): filtra por modelos activos -> descarta no aplicables o precio <= 0 ->
con >= ``MIN_MODELOS_IQR`` validos excluye outliers fuera de [Q1-a, Q3+a] con a = max(1.5*IQR, ``IQR_PISO_REL``*mediana)
(cuartiles con interpolacion lineal, numpy; el piso evita que con modelos casi identicos, IQR ~ 0, una diferencia del 1 %
sea "outlier") -> fair_value = promedio simple -> upside = (fv-px)/px*100 -> precio con margen de seguridad
= fv*(1-MoS) -> incertidumbre por CV = sigma/mu (sigma POBLACIONAL, ddof=0: se mide la dispersion de los modelos
usados, no se estima una poblacion mayor): Baja si CV < CV_BAJO, Media entre CV_BAJO y CV_ALTO, Alta si CV > CV_ALTO
o si quedan menos de ``MIN_MODELOS_CONFIABLE`` modelos usados. "Baja" se degrada a "Media" si los modelos usados
pertenecen a menos de ``MIN_FAMILIAS_BAJA`` grupos de familia (``GRUPOS_FAMILIA``): modelos derivados de las mismas
cifras (p. ej. P/E, P/B, DDM, Graham y Lynch de un banco salen de EPS y valor contable) coinciden por construccion, no por
confirmacion independiente. El resultado incluye ``cv_prefiltro`` (CV antes del IQR) y ``familias_usadas``.
``complementarias`` (solo informativa) da mediana, submedias por grupo y promedio igualando grupos; ``dcf_inverso`` da el
WACC y el crecimiento implicitos en el precio. El margen de seguridad se acota a [0, ``MARGEN_SEGURIDAD_MAX``].

Limitaciones (visibles en la UI): muy sensible a WACC y g; el valor terminal domina los DCF; datos LTM (trailing)
de SEC EDGAR con rezago; no aplica bien a bancos/aseguradoras (EBITDA/FCF), ciclicas ni empresas sin beneficios.
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
CRECIMIENTO_TERMINAL_DEFECTO = 0.025   # g por defecto (mismo valor que valuation_floors.CRECIMIENTO_TERMINAL)
MARGEN_SEGURIDAD_DEFECTO = 0.25
RF_DEFECTO = 0.045                     # tasa libre de riesgo por defecto: PARAMETRO (no se consulta ^TNX; el 10Y ronda 4.5-4.9 %)
ERP_DEFECTO = 0.05                     # prima de riesgo de mercado por defecto: PARAMETRO
BETA_DEFECTO, BETA_MIN, BETA_MAX = 1.0, 0.7, 2.0   # betas bajas (defensivas) se acotan: la sensibilidad del DCF al WACC es enorme
WACC_MIN, WACC_MAX = 0.075, 0.14       # limites del WACC automatico
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
DIF_MIN_WACC_G = 0.01                  # DCF/EPV/DDM exigen (tasa de descuento - g) >= 1 pp (el valor terminal explota si no)
PB_IMPLICITO_RANGO = (0.05, 100.0)     # precio / BVPS fuera de este rango => dato por accion de otra clase (BRK-B): se descarta
PE_IMPLICITO_RANGO = (0.05, 2000.0)    # precio / EPS (EPS > 0) fuera de este rango => EPS inconsistente con el precio
SUBCONTEO_INFO = 0.5                   # caja/deuda XBRL < 50 % del ``info`` de yfinance: se usa yfinance
MAX_ANTIGUEDAD_LTM_DIAS = 420          # un LTM mas viejo que esto se considera no disponible

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
def calcular_tasas(e: EntradasValoracion, p: ParametrosValoracion) -> dict:
    """ke (CAPM), kd, WACC automatico y WACC usado. Si ``p.wacc`` esta fijado, ke se desplaza en paralelo.

    El desplazamiento de ke es ``(WACC_usuario - WACC_auto) / max(peso_patrimonio, WE_MIN_AJUSTE_KE)``: con deuda, mover el
    WACC un punto mueve ke mas de un punto (WACC = w_e*ke + w_d*kd*(1-t) con kd fijo)."""
    rf = _f(p.rf) if p.rf is not None else (_f(e.rf) if _f(e.rf) is not None else RF_DEFECTO)
    beta_raw = _f(e.beta)
    beta = _clip(beta_raw, BETA_MIN, BETA_MAX) if beta_raw is not None else BETA_DEFECTO
    t_raw = _f(e.tasa_impositiva)
    t = _clip(t_raw, TASA_MIN, TASA_IMPOSITIVA_MAX) if t_raw is not None else TASA_IMPOSITIVA_DEFECTO
    ke = rf + beta * p.erp
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
    wacc = _f(p.wacc) if p.wacc is not None else wacc_auto
    ke_usado = ke + (wacc - wacc_auto) / max(w_e, WE_MIN_AJUSTE_KE)
    topes_ke, topes_wacc = [], []
    if beta_raw is not None and beta != beta_raw:
        topes_ke.append(f"beta {beta_raw:.2f} acotada a {beta:.2f}")
    if p.wacc is None and wacc_bruto != wacc_auto:
        topes_wacc.append(f"WACC {wacc_bruto * 100:.2f} % acotado a {wacc_auto * 100:.2f} %")
    return {"rf": rf, "erp": p.erp, "beta": beta, "beta_origen": "yfinance (acotada)" if beta_raw is not None
            else "no disponible: se usa el valor por defecto", "ke_capm": ke, "ke": ke_usado, "kd": kd,
            "kd_origen": kd_origen, "tasa_impositiva": t,
            "tasa_origen": "efectiva LTM (piso %.0f %%)" % (TASA_MIN * 100) if t_raw is not None else "por defecto (sin dato)",
            "peso_deuda": w_d, "peso_capital": w_e, "wacc_auto": wacc_auto, "wacc": wacc,
            "wacc_origen": "usuario" if p.wacc is not None else "automatico (CAPM + estructura de capital)",
            "topes_ke": topes_ke, "topes_wacc": topes_wacc, "topes": topes_ke + topes_wacc}


def _tope_crecimiento(hist: list[float] | None) -> str | None:
    """Texto si el crecimiento inicial historico (``HAIRCUT*CAGR``) queda acotado a [CRECIMIENTO_MIN, CRECIMIENTO_TOPE]."""
    c = cagr(hist)
    if c is None:
        return None
    bruto = HAIRCUT * c
    if bruto > CRECIMIENTO_TOPE:
        return f"crecimiento inicial {bruto * 100:.1f} % (CAGR {c * 100:.1f} %) topado a {CRECIMIENTO_TOPE * 100:.0f} %"
    if bruto < CRECIMIENTO_MIN:
        return f"crecimiento inicial {bruto * 100:.1f} % (CAGR {c * 100:.1f} %) acotado a {CRECIMIENTO_MIN * 100:.0f} %"
    return None


def _crecimiento_inicial(hist: list[float] | None, g: float) -> tuple[float, str]:
    if es_ciclico(hist):
        return g, "flujo cíclico normalizado: crecimiento inicial = g"
    c = cagr(hist)
    if c is None:
        return g, "sin historia suficiente: crecimiento inicial = g"
    g0 = _clip(HAIRCUT * c, CRECIMIENTO_MIN, CRECIMIENTO_TOPE)
    return g0, f"CAGR historico {c * 100:.1f} % x {HAIRCUT} (acotado a [{CRECIMIENTO_MIN * 100:.0f} %, " \
               f"{CRECIMIENTO_TOPE * 100:.0f} %])"


def _base_crecimiento(hist: list[float] | None, ltm: float, g: float) -> tuple[float, float, bool, str | None]:
    """(base, crecimiento inicial, ciclico, tope) de un flujo.

    Ciclico: base = media de los ultimos ``N_NORMALIZACION`` anios y crecimiento inicial = g. Si no: base = LTM y
    crecimiento = ``min(CRECIMIENTO_TOPE, HAIRCUT*CAGR)`` (``tope`` describe el tope si esta activo)."""
    if es_ciclico(hist):
        return float(np.mean(_limpiar(hist)[-N_NORMALIZACION:])), g, True, None
    g0, _ = _crecimiento_inicial(hist, g)
    return ltm, g0, False, _tope_crecimiento(hist)


def _ajuste_fcff(e: EntradasValoracion, t: float) -> float:
    """Intereses LTM despues de impuestos que convierten FCF apalancado en FCFF (0 si no hay gasto de intereses)."""
    interes = _f(e.gasto_interes_ltm)
    return interes * (1.0 - t) if interes is not None and interes > 0 else 0.0


def _margen_detalle(e: EntradasValoracion) -> dict:
    """Margen FCF actual / objetivo para el DCF por ingresos (FCFF: suma los intereses despues de impuestos LTM)."""
    vacio = {"actual": None, "objetivo": None, "origen": "FCF", "mediano": None, "tope_activo": False}
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
    hf, hr = _limpiar(e.hist_fcf), _limpiar(e.hist_ingresos)
    n = min(len(hf), len(hr))
    margenes = [hf[-n + i] / hr[-n + i] for i in range(n) if hr[-n + i] > 0] if n else []
    mediano = float(np.median(margenes)) + ajuste if len(margenes) >= MIN_PUNTOS_MULTIPLO and origen == "FCF" else actual
    return {"actual": actual, "objetivo": min(MARGEN_FCF_TOPE, mediano), "origen": origen, "mediano": mediano,
            "tope_activo": mediano > MARGEN_FCF_TOPE}


def _margen_fcf(e: EntradasValoracion) -> tuple[float | None, float | None, str]:
    """(margen actual, margen objetivo, origen) para el DCF por ingresos."""
    d = _margen_detalle(e)
    return d["actual"], d["objetivo"], d["origen"]


def multiplo_objetivo(clave: str, e: EntradasValoracion, p: ParametrosValoracion) -> dict:
    """Multiplo objetivo: el del usuario o la mediana historica propia acotada. {valor|None, origen, n, mediana, tope}."""
    fijo = _f(getattr(p, f"{clave}_objetivo", None))
    lo, hi = RANGOS_MULTIPLOS[clave]
    if fijo is not None:
        return {"valor": fijo, "origen": "parametro del usuario", "n": 0, "mediana": None, "rango": (lo, hi), "tope": None}
    pts = [v for v in _limpiar((e.multiplos_hist or {}).get(clave)) if v > 0]
    if len(pts) < MIN_PUNTOS_MULTIPLO:
        return {"valor": None, "origen": "historia insuficiente", "n": len(pts), "mediana": None, "rango": (lo, hi),
                "tope": None}
    med = float(np.median(pts))
    valor = _clip(med, lo, hi)
    tope = None if valor == med else f"objetivo: mediana histórica {med:.1f}x topada a {valor:g}x"
    return {"valor": valor, "origen": f"mediana historica propia de {len(pts)} anios, acotada a "
            f"[{lo:g}, {hi:g}]", "n": len(pts), "mediana": med, "rango": (lo, hi), "tope": tope}


def supuestos(entradas: Any, params: Any = None) -> dict:
    """Todos los supuestos usados (la UI los muestra). Dict plano serializable.

    ``topes_activos`` lista los topes/acotaciones que estan condicionando el resultado ("por construccion")."""
    e, p = _como_entradas(entradas), _como_params(params)
    tasas = calcular_tasas(e, p)
    g = p.g
    out: dict[str, Any] = {
        "wacc": tasas["wacc"], "wacc_auto": tasas["wacc_auto"], "wacc_origen": tasas["wacc_origen"],
        "g": g, "margen_seguridad": p.margen_seguridad, "ke": tasas["ke"], "kd": tasas["kd"],
        "kd_origen": tasas["kd_origen"], "rf": tasas["rf"], "erp": tasas["erp"], "beta": tasas["beta"],
        "beta_origen": tasas["beta_origen"], "tasa_impositiva": tasas["tasa_impositiva"],
        "tasa_origen": tasas["tasa_origen"], "peso_deuda": tasas["peso_deuda"],
        "horizontes": dict(HORIZONTES), "g_valido": (tasas["wacc"] - g) >= DIF_MIN_WACC_G - 1e-12}
    topes = list(tasas["topes"])
    ciclicos = []
    for clave, hist, nombre in (("fcf", e.hist_fcf, "FCF"), ("ebitda", e.hist_ebitda, "EBITDA"),
                                ("revenue", e.hist_ingresos, "ingresos")):
        g0, nota = _crecimiento_inicial(hist, g)
        out[f"crecimiento_inicial_{clave}"] = g0
        out[f"crecimiento_inicial_{clave}_nota"] = nota
        if es_ciclico(hist):
            ciclicos.append(nombre)
        elif _tope_crecimiento(hist):
            topes.append(f"{nombre}: {_tope_crecimiento(hist)}")
    out["flujos_ciclicos_normalizados"] = ciclicos
    md = _margen_detalle(e)
    out.update(margen_fcf_actual=md["actual"], margen_fcf_objetivo=md["objetivo"], margen_fcf_base=md["origen"],
               margen_fcf_tope=MARGEN_FCF_TOPE, ajuste_fcff=_ajuste_fcff(e, tasas["tasa_impositiva"]))
    if md["tope_activo"]:
        topes.append(f"margen FCF objetivo: mediana histórica {md['mediano'] * 100:.1f} % topada a "
                     f"{MARGEN_FCF_TOPE * 100:.0f} %")
    out["multiplos"] = {k: multiplo_objetivo(k, e, p) for k in RANGOS_MULTIPLOS}
    for k, nombre in (("pe", "P/E"), ("ev_ebitda", "EV/EBITDA"), ("ev_revenue", "EV/Ingresos"), ("pb", "P/B"),
                      ("p_fcf", "P/FCF")):
        if out["multiplos"][k]["tope"]:
            topes.append(f"{nombre} " + out["multiplos"][k]["tope"])
    out["crecimiento_eps_lynch"] = cagr(e.hist_eps)
    if out["crecimiento_eps_lynch"] is not None and out["crecimiento_eps_lynch"] * 100 > LYNCH_CRECIMIENTO_TOPE_PCT:
        topes.append(f"Lynch: crecimiento de EPS {out['crecimiento_eps_lynch'] * 100:.0f} % topado a "
                     f"{LYNCH_CRECIMIENTO_TOPE_PCT:.0f} %")
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


def _vp_gordon(flujos: list[float], wacc: float, g: float) -> float:
    n = len(flujos)
    vp = sum(f / (1 + wacc) ** (t + 1) for t, f in enumerate(flujos))
    vt = flujos[-1] * (1 + g) / (wacc - g)
    return vp + vt / (1 + wacc) ** n


def _motivo_dif_tasas(tasa: float, g: float, nombre: str) -> str | None:
    """Motivo si la tasa de descuento no supera a g por al menos ``DIF_MIN_WACC_G`` (None si es suficiente)."""
    if tasa <= g:
        return f"g ({g * 100:.1f} %) >= {nombre} ({tasa * 100:.1f} %): el valor terminal no converge"
    if round(tasa - g, 10) < DIF_MIN_WACC_G:
        return (f"diferencia {nombre}-g < {DIF_MIN_WACC_G * 100:.0f} pp ({nombre} {tasa * 100:.2f} %, g {g * 100:.2f} %): "
                f"el valor terminal explota")
    return None


def _modelo_dcf(id_: str, base: str, n: int, e: EntradasValoracion, p: ParametrosValoracion, tasas: dict) -> dict:
    wacc, g = tasas["wacc"], p.g
    acc, motivo = _acciones(e)
    if motivo:
        return _res(id_, motivo=motivo)
    neto, motivo = _neto(e)
    if motivo:
        return _res(id_, motivo=motivo)
    motivo = _motivo_dif_tasas(wacc, g, "WACC")
    if motivo:
        return _res(id_, motivo=motivo)
    t = tasas["tasa_impositiva"]
    topes = list(tasas["topes"])
    ciclico = False
    if base == "fcf":
        f_ltm = _f(e.fcf_ltm)
        if f_ltm is None:
            return _res(id_, motivo="FCF no disponible")
        if f_ltm <= 0:
            return _res(id_, motivo="FCF LTM negativo o cero")
        hf = _limpiar(e.hist_fcf)
        if hf and sum(1 for v in hf if v > 0) < MIN_ANIOS_FCF:
            return _res(id_, motivo=f"historia insuficiente de FCF positivo (< {MIN_ANIOS_FCF} años con FCF > 0)")
        f0, g0, ciclico, tope = _base_crecimiento(e.hist_fcf, f_ltm, g)
        if p.crecimiento_inicial is not None:   # DCF inverso: crecimiento fijado desde fuera
            g0, ciclico, tope = float(p.crecimiento_inicial), False, None
            f0 = f_ltm
        if f0 <= 0:
            return _res(id_, motivo="FCF normalizado negativo o cero")
        f0 += _ajuste_fcff(e, t)   # FCFE -> FCFF: se suman los intereses despues de impuestos (la deuda se resta aparte)
        if tope:
            topes.append(tope)
        flujos, f = [], f0
        for gt in np.linspace(g0, g, n):
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
        eb0, g0, ciclico, tope = _base_crecimiento(e.hist_ebitda, eb_ltm, g)
        if eb0 <= 0:
            return _res(id_, motivo="EBITDA normalizado negativo o cero")
        if tope:
            topes.append(tope)
        flujos, prev = [], eb0
        for gt in np.linspace(g0, g, n):
            eb = prev * (1 + gt)
            flujos.append(eb * (1 - t) + da_r * eb * t - cx_r * eb - nwc_r * (eb - prev))
            prev = eb
        if flujos[-1] <= 0:
            return _res(id_, motivo="flujo terminal negativo (CapEx/impuestos superan el EBITDA)")
    else:  # revenue
        rev_ltm = _f(e.ingresos_ltm)
        if rev_ltm is None or rev_ltm <= 0:
            return _res(id_, motivo="ingresos no disponibles")
        md = _margen_detalle(e)
        m_act, m_obj = md["actual"], md["objetivo"]
        if m_act is None or m_obj is None:
            return _res(id_, motivo="margen FCF/operativo no disponible")
        if m_obj <= 0:
            return _res(id_, motivo="margen FCF historico no positivo")
        rev0, g0, ciclico, tope = _base_crecimiento(e.hist_ingresos, rev_ltm, g)
        if tope:
            topes.append(tope)
        if md["tope_activo"]:
            topes.append(f"margen FCF objetivo: mediana histórica {md['mediano'] * 100:.1f} % topada a "
                         f"{MARGEN_FCF_TOPE * 100:.0f} %")
        flujos, rev = [], rev0
        for i, gt in enumerate(np.linspace(g0, g, n), 1):
            rev *= 1 + gt
            flujos.append(rev * (m_act + (m_obj - m_act) * i / n))
        if flujos[-1] <= 0:
            return _res(id_, motivo="flujo terminal negativo")
    valor = (_vp_gordon(flujos, wacc, g) + neto) / acc  # type: ignore[operator]
    return _res(id_, valor, wacc=wacc, g=g, ciclico=ciclico, topes=topes)


def _motivo_inconsistente(e: EntradasValoracion, *campos: str) -> str | None:
    mal = [c for c in campos if c in (e.datos_inconsistentes or [])]
    return f"dato por acción inconsistente ({', '.join(mal)} no cuadra con el precio: ¿otra clase de acciones?)" if mal \
        else None


def _modelo_multiplo(id_: str, clave: str, e: EntradasValoracion, p: ParametrosValoracion) -> dict:
    inc = _motivo_inconsistente(e, *{"pe": ("eps_ltm",), "pb": ("bvps",)}.get(clave, ()))
    if inc:
        return _res(id_, motivo=inc)
    mo = multiplo_objetivo(clave, e, p)
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
    ke, g = tasas["ke"], p.g
    if ke <= g:
        return _res("ddm", motivo=f"ke ({ke * 100:.1f} %) <= g ({g * 100:.1f} %): el modelo de Gordon no converge")
    motivo = _motivo_dif_tasas(ke, g, "ke")
    if motivo:
        return _res("ddm", motivo=motivo)
    return _res("ddm", d0 * (1 + g) / (ke - g), ke=ke, g=g, topes=tasas["topes_ke"])


def _modelo_epv(e: EntradasValoracion, p: ParametrosValoracion, tasas: dict) -> dict:
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
    motivo = _motivo_dif_tasas(wacc, p.g, "WACC")
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


def _modelo_lynch(e: EntradasValoracion) -> dict:
    inc = _motivo_inconsistente(e, "eps_ltm")
    if inc:
        return _res("lynch", motivo=inc)
    eps = _f(e.eps_ltm)
    if eps is None:
        return _res("lynch", motivo="EPS no disponible")
    if eps <= 0:
        return _res("lynch", motivo="EPS LTM negativo o cero")
    c = cagr(e.hist_eps)
    if c is None:
        return _res("lynch", motivo="crecimiento historico de EPS no disponible")
    if c <= 0:
        return _res("lynch", motivo="crecimiento de EPS <= 0")
    crec = min(LYNCH_CRECIMIENTO_TOPE_PCT, c * 100.0)
    topes = [f"crecimiento de EPS {c * 100:.0f} % topado a {LYNCH_CRECIMIENTO_TOPE_PCT:.0f} %"] if c * 100.0 > crec else []
    return _res("lynch", eps * crec, crecimiento_pct=crec, topes=topes)


def calcular_modelos(entradas: Any, params: Any = None) -> list[dict]:
    """Los 15 modelos (en el orden de ``MODELOS``). Funcion pura; nunca lanza por datos faltantes.

    Cada dict: id, nombre, familia, precio_estimado (float|None), aplicable, motivo_exclusion (+ datos del modelo).
    """
    e, p = _como_entradas(entradas), _como_params(params)
    tasas = calcular_tasas(e, p)
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
                    r = _modelo_dcf(id_, base, HORIZONTES[anios], e, p, tasas)
                elif id_ in ("pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf"):
                    r = _modelo_multiplo(id_, id_, e, p)
                elif id_ == "ddm":
                    r = _modelo_ddm(e, p, tasas)
                elif id_ == "epv":
                    r = _modelo_epv(e, p, tasas)
                elif id_ == "graham":
                    r = _modelo_graham(e)
                else:
                    r = _modelo_lynch(e)
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

    ``modelos_activos`` None = todos; lista vacia = ninguno. Ver el docstring del modulo para la logica exacta.
    ``n_validos`` cuenta los validos (activos, aplicables y > 0) ANTES del filtro IQR; ``modelos_usados`` son los que
    quedan tras excluir outliers y ``incertidumbre`` usa ``len(modelos_usados)`` (CV poblacional de los usados).
    ``modelos_no_aplicables`` = ids sin precio valido; ``modelos_desactivados`` = ids aplicables que el usuario no
    incluyo (un id nunca esta en ambas listas: "no aplica" prevalece). ``cv_prefiltro`` = CV de los validos antes del
    IQR; ``familias_usadas`` = grupos de familia (``GRUPOS_FAMILIA``) de los modelos usados.
    El ``margen_seguridad`` se acota a [MARGEN_SEGURIDAD_MIN, MARGEN_SEGURIDAD_MAX].
    """
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
            "precio_actual": _f(precio_actual), "motivo": None, "rango": None}
    if not usados:
        sin_activos = activos is not None and not any(r["id"] in activos for r in resultados)
        base["motivo"] = ("ningún modelo activo" if sin_activos
                          else "ningún modelo válido: todos desactivados o no aplicables")
        return base
    v = np.array(list(usados.values()), dtype=float)
    mu = float(v.mean())
    fv, sigma = mu, float(v.std(ddof=0))
    cv = sigma / mu if mu > 0 else None
    px = _f(precio_actual)
    n_grupos = len(familias) if familias else None
    base.update(fair_value=fv, cv=cv, incertidumbre=clasificar_incertidumbre(cv, len(usados), n_grupos),
                margin_of_safety_price=fv * (1.0 - margen_seguridad), rango=(float(v.min()), float(v.max())),
                upside_pct=(fv - px) / px * 100.0 if px is not None and px > 0 else None)
    return base


def complementarias(agregado: dict, modelos: list[dict] | None = None) -> dict | None:
    """Estadisticas INFORMATIVAS (no sustituyen al consolidado): mediana, submedias por grupo de familia y promedio
    igualando grupos (cada grupo pesa lo mismo). Los modelos no son independientes: 6 DCF comparten WACC/g/crecimiento y los
    de Valor comparten EPS y valor contable, asi que N modelos no son N confirmaciones. None sin modelos usados."""
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
    return {"mediana": float(np.median(todos)), "submedias": sub, "n_por_grupo": {g: len(por_grupo[g]) for g in sub},
            "promedio_familias": prom_fam, "upside_mediana": up(float(np.median(todos))),
            "upside_promedio_familias": up(prom_fam)}


def _valor_dcf(e: EntradasValoracion, p: ParametrosValoracion, id_: str) -> float | None:
    _, anios, base = id_.split("_")
    r = _modelo_dcf(id_, base, HORIZONTES[anios], e, p, calcular_tasas(e, p))
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
    """DCF inverso (funcion pura): WACC y crecimiento inicial que igualan el ``modelo`` DCF al precio.

    ``wacc_implicito``: el WACC (con g y crecimiento usados) que da exactamente el precio; ``g0_implicito``: el
    crecimiento inicial (con el WACC usado) que lo da. Cada uno es None si no existe en su rango (WACC en
    [g + ``DIF_MIN_WACC_G``, 40 %]; crecimiento en [-20 %, 60 %]) o si el modelo no aplica. Lectura: si el WACC
    implicito es menor o el crecimiento implicito mayor que los usados, el mercado descuenta menos riesgo o mas
    crecimiento que estos supuestos."""
    e, p = _como_entradas(entradas), _como_params(params)
    px = _f(precio) if precio is not None else _f(e.precio)
    out = {"modelo": modelo, "precio": px, "wacc_usado": None, "wacc_implicito": None, "g0_usado": None,
           "g0_implicito": None, "valor_con_supuestos": None}
    if not px or px <= 0 or modelo not in IDS_MODELOS or not modelo.startswith("dcf_") or modelo.endswith("revenue") \
            or modelo.endswith("ebitda"):
        return out
    tasas = calcular_tasas(e, p)
    out["wacc_usado"] = tasas["wacc"]
    out["valor_con_supuestos"] = _valor_dcf(e, p, modelo)
    g0u = _base_crecimiento(e.hist_fcf, _f(e.fcf_ltm) or 0.0, p.g)[1]
    out["g0_usado"] = p.crecimiento_inicial if p.crecimiento_inicial is not None else g0u
    out["wacc_implicito"] = _biseccion(lambda w: _valor_dcf(e, p.con(wacc=w), modelo), p.g + DIF_MIN_WACC_G, 0.40, px,
                                       creciente=False)
    out["g0_implicito"] = _biseccion(lambda g0: _valor_dcf(e, p.con(crecimiento_inicial=g0), modelo), -0.20, 0.60, px,
                                     creciente=True)
    return out


def valor_justo(entradas: Any, params: Any = None, modelos_activos: list[str] | None = None,
                precio_actual: float | None = None) -> dict:
    """Atajo: ``calcular_modelos`` + ``agregar_valor_justo`` con el margen de seguridad de ``params``."""
    e, p = _como_entradas(entradas), _como_params(params)
    px = precio_actual if precio_actual is not None else e.precio
    modelos = calcular_modelos(e, p)
    agg = agregar_valor_justo(modelos, modelos_activos, p.margen_seguridad, px)
    agg["modelos"] = modelos
    return agg


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
    return pd.Series(out, dtype=float).sort_index() if out else pd.Series(dtype=float)


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


def _instantaneas_multiplos(precio_raw: pd.Series, eps_q, fcf_q, ebitda_q, rev_q, sh_q, caja, deuda, equity) -> dict:
    """Multiplos propios en k = 1..ANIOS_HISTORIA anios atras (precio 45 dias despues del cierre del periodo)."""
    out: dict[str, list[float]] = {k: [] for k in RANGOS_MULTIPLOS}
    for k in range(ANIOS_HISTORIA, 0, -1):
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


def extraer_entradas(ticker: str, tk_obj: Any, df: pd.DataFrame, info: dict | None = None,
                     facts: pd.DataFrame | None = None) -> EntradasValoracion:
    """Arma las entradas UNA vez (sin peticiones a yfinance; EDGAR solo si no se pasa ``facts``).

    XBRL (10-K/10-Q) es la fuente primaria; el ``info`` de yfinance solo rellena lo que falte (marcado en ``fuentes``).
    ``tk_obj`` se acepta por simetria con ``valuation_floors`` pero no se usa (los splits y dividendos salen del ``df``).
    Todo dato faltante queda en None y se anota en ``avisos``.
    """
    info = info or {}
    e = EntradasValoracion(ticker=ticker, sector=info.get("sector"), industria=info.get("industry"))
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
    if facts is None:
        try:
            facts = _vf()._facts(ticker)
        except Exception as ex:  # noqa: BLE001
            av.append(f"EDGAR no disponible ({type(ex).__name__}): se usa yfinance donde se pueda")
            facts = None
    if facts is not None:
        try:
            _extraer_xbrl(e, facts, splits, df)
        except Exception as ex:  # noqa: BLE001
            av.append(f"XBRL incompleto ({type(ex).__name__}: {ex}): se usa yfinance donde se pueda")
    _completar_con_info(e, info, df)
    _validar_por_accion(e)
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


def _extraer_xbrl(e: EntradasValoracion, facts: pd.DataFrame, splits, df) -> None:
    fu, av = e.fuentes, e.avisos
    ult = facts["period_end"].max()
    facts = facts[facts["period_end"] >= ult - pd.DateOffset(years=ANIOS_HISTORIA + 3)]
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
    for campo, q, nombre in (("fcf_ltm", fcf_q, "OCF - CapEx"), ("ebitda_ltm", ebitda_q, "EBIT + D&A"),
                             ("ebit_ltm", ebit_q, nombre_ebit), ("ingresos_ltm", rev_q, "Revenues"),
                             ("eps_ltm", eps_q, "EPS diluido"), ("da_ltm", da_q, "D&A"), ("capex_ltm", cx_q, "CapEx")):
        v = _ltm(q) if vigente(q) else None
        if v is not None:
            setattr(e, campo, v)
            fu[campo] = f"SEC EDGAR XBRL ({nombre})"
    if e.capex_ltm is not None:
        e.capex_ltm = abs(e.capex_ltm)
    for campo, q in (("hist_fcf", fcf_q), ("hist_ebitda", ebitda_q), ("hist_ingresos", rev_q), ("hist_eps", eps_q)):
        if vigente(q):
            setattr(e, campo, _historia(q))
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
    sh_q = _acciones_serie(facts, splits, (_ACCIONES_DIL, _ACCIONES_BAS))   # historia: diluido y, si falta, basico
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
    eq = _asof(eq_s, fecha, 150)
    if eq is not None and sh and sh > 0:
        e.bvps = eq / sh
        fu["bvps"] = "SEC EDGAR XBRL (patrimonio / acciones)"
    ac, pc = _asof(_instante(facts, _ACTIVO_CORR), fecha, 150), _asof(_instante(facts, _PASIVO_CORR), fecha, 150)
    if ac is not None and pc is not None and c is not None:
        deuda_cp = _asof(_instante(facts, _DEUDA_CP + ["us-gaap:LongTermDebtCurrent"]), fecha, 150) or 0.0
        e.nwc_operativo = (ac - c) - (pc - deuda_cp)
        fu["nwc_operativo"] = "SEC EDGAR XBRL ((activo corriente - caja) - (pasivo corriente - deuda corto plazo))"
    # Multiplos historicos propios
    if df is not None and not df.empty:
        idx = pd.DatetimeIndex(df.index)
        precio = _precio_sin_ajuste_dividendos(df)
        precio.index = idx.tz_localize(None) if idx.tz is not None else idx
        e.multiplos_hist = _instantaneas_multiplos(
            precio, eps_q if vigente(eps_q) else pd.Series(dtype=float), fcf_q, ebitda_q,
            rev_q if vigente(rev_q) else pd.Series(dtype=float), sh_q, caja_s, deuda_s, eq_s)


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
