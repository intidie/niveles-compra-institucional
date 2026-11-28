"""Correcciones de la revision de codigo del valor justo (ronda 1) y de la revision financiera (ronda 2). Sin red."""
import asyncio
import math

import numpy as np
import pandas as pd
import pytest

import app_ui as ui_mod
from src.levels import fair_value as fv
from src.levels.fair_value import ParametrosValoracion as Par
from test_fair_value import (_anot_fair, _df_precios, _entradas_ricas, _facts_falsos, _sin_red_ni_disco, cliente,  # noqa: F401
                             ent, modelo, vista)


# ============================================================================== UI
def _dos(v):
    """Texto de la fila de un modelo igual en ambos escenarios (Graham no depende del escenario)."""
    return f"Medio ${v:,.2f} · Pesimista ${v:,.2f}"


def test_cambio_de_ticker_con_interruptor_activo_repinta_la_pestana(vista, df_sintetico, monkeypatch):
    vista.sw_vj.value = True
    otra = _entradas_ricas()
    otra.ticker, otra.eps_ltm = "OTRA", 9.0
    monkeypatch.setitem(ui_mod.CACHE, "OTRA", {"ticker": "OTRA", "precio_actual": 80.0, "df": df_sintetico, "zonas": [],
                                               "errores": [], "entradas_valoracion": otra.to_dict()})
    vista.ticker = "OTRA"
    vista.refrescar()
    assert vista.vj_entradas.ticker == "OTRA" and vista.vj_kpi["precio"].text == "$80.00"
    assert vista.vj_kpi["fair"].text == f"${vista.vj_agg['fair_value']:,.2f}"
    assert vista.vj_rows["graham"]["precio"].text == _dos(math.sqrt(22.5 * 9.0 * 25.0))
    assert _anot_fair(vista) and f"${vista.vj_agg['fair_value']:,.2f}" in _anot_fair(vista)[0]


def test_recalcular_con_interruptor_activo_repinta_la_pestana(vista):
    vista.sw_vj.value = True
    nuevo = ui_mod.CACHE["RICA"]
    ent_nueva = _entradas_ricas()
    ent_nueva.eps_ltm = 12.0
    ui_mod.CACHE["RICA"] = {**nuevo, "entradas_valoracion": ent_nueva.to_dict()}   # como un nuevo Recalcular
    vista.refrescar()
    assert vista.vj_rows["graham"]["precio"].text == _dos(math.sqrt(22.5 * 12.0 * 25.0))


def test_init_sin_ticker_los_sliders_coinciden_con_los_parametros(cliente):
    v = ui_mod.Vista()
    v.construir()
    assert v.ticker is None
    assert v.vj_sl["g"].value == round(fv.CRECIMIENTO_TERMINAL_DEFECTO * 100, 1)
    assert v.vj_sl["mos"].value == round(fv.MARGEN_SEGURIDAD_DEFECTO * 100)
    assert 4.0 < v.vj_sl["wacc"].value < 16.0
    assert "3,5 %" in v.vj_sl_lbl["g"].text and "25 %" in v.vj_sl_lbl["mos"].text
    assert v.vj_params == fv.ParametrosValoracion()


def test_escaneo_no_borra_las_entradas_del_ticker_activo(cliente, df_sintetico, monkeypatch):
    from src.reporting import userdata
    userdata.guardar(["RICA"])
    previo = {"ticker": "RICA", "precio_actual": 100.0, "df": df_sintetico, "zonas": [], "errores": [],
              "entradas_valoracion": _entradas_ricas().to_dict()}
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(None))
    ui_mod.CACHE["RICA"] = previo
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "RICA"
    v.refrescar()
    antes = v.vj_agg["fair_value"]

    async def falso(t, cap, bt, sec, vj=True):   # el escaneo en la nube no pide valor justo: no hay entradas
        return {"ticker": t, "precio_actual": 101.0, "zonas": [], "errores": [], "confluencia_score": 50.0,
                "confiabilidad": {"indice": 50.0, "cobertura": 1.0}}
    monkeypatch.setattr(ui_mod, "calcular_ticker", falso)
    v.sw_vj_wl.value = False

    async def correr():
        with cliente:
            await v.escanear()
    asyncio.run(correr())
    assert ui_mod.CACHE["RICA"]["precio_actual"] == 101.0                     # el resultado nuevo SI se guardo
    assert ui_mod.CACHE["RICA"]["entradas_valoracion"] is previo["entradas_valoracion"]
    v.refrescar()
    assert v.vj_kpi["fair"].text != "n/d" and v.vj_agg["fair_value"] == pytest.approx(antes)


def test_expulsion_del_lru_no_pierde_las_entradas_del_ticker_activo(vista):
    assert vista.vj_agg is not None
    del ui_mod.CACHE["RICA"]       # lo que hace el LRU de la nube al expulsar el ticker activo
    vista.refrescar()
    assert vista.vj_kpi["fair"].text != "n/d" and vista.vj_entradas.ticker == "RICA"


def test_mover_controles_con_el_interruptor_apagado_no_reconstruye_el_grafico_de_analisis(vista, monkeypatch):
    llamadas = []
    orig = ui_mod.overlays_grafico
    monkeypatch.setattr(ui_mod, "overlays_grafico", lambda *a, **k: llamadas.append(1) or orig(*a, **k))
    vista.sw_vj.value = False                      # activado por defecto: se apaga para la prueba
    llamadas.clear()
    vista.vj_sl["wacc"].value = 11.0
    vista.vj_rows["pe"]["chk"].value = False
    vista._vj_seleccionar(ui_mod.ids_por_familia("DCF"))
    vista._vj_restablecer()
    assert llamadas == []
    assert vista.vj_agg is not None
    vista.sw_vj.value = True                       # al encenderlo si se dibuja
    assert len(llamadas) == 1
    vista.vj_sl["wacc"].value = 10.0
    assert len(llamadas) == 2


def test_texto_de_outliers_describe_el_comportamiento_real():
    sec = next(s for s in ui_mod.contenido_modelo() if s["id"] == "valor_justo")
    txt = ui_mod.texto_seccion(sec)
    assert "volver a incluirlos" not in txt
    assert "Un outlier queda excluido mientras haya 4 o más modelos válidos" in txt


def test_valor_justo_absurdo_no_deforma_el_eje_del_grafico(df_sintetico):
    px_ = float(df_sintetico["Close"].iloc[-1])
    datos = ui_mod.datos_grafico(df_sintetico, "T")
    ref = ui_mod.overlays_grafico(datos, "T", [], px_)
    arriba = ui_mod.overlays_grafico(datos, "T", [], px_, valor_justo={"fair_value": px_ * 35.0, "upside_pct": 3400.0})
    assert arriba["extras"] == ref["extras"]                                  # la escala no incluye el valor absurdo
    vj = arriba["vj"]
    assert vj["fuera_de_rango"] == "arriba" and "fuera de rango" in vj["etiqueta"] and "▲" in vj["etiqueta"]
    bajo = ui_mod.overlays_grafico(datos, "T", [], px_, valor_justo={"fair_value": px_ * 0.1, "upside_pct": -90.0})
    assert bajo["vj"]["fuera_de_rango"] == "abajo" and "▼" in bajo["vj"]["etiqueta"]
    lo, hi = ui_mod.VJ_RANGO_EJE
    dentro = ui_mod.overlays_grafico(datos, "T", [], px_, valor_justo={"fair_value": px_ * hi, "upside_pct": 200.0})
    assert dentro["vj"]["fuera_de_rango"] is None                            # en el borde: se dibuja
    assert max(dentro["extras"]) >= px_ * hi - 1e-3


# ============================================================================== monedas y datos por accion
def _info_fx(**kw):
    base = {"currency": "USD", "financialCurrency": "TWD", "freeCashflow": 1.0e12, "totalRevenue": 4.9e12,
            "totalCash": 4.0e12, "totalDebt": 1.0e12, "trailingEps": 15.0, "bookValue": 5.2, "ebitda": 2.0e12,
            "sharesOutstanding": 5.2e9, "trailingAnnualDividendRate": 3.0, "beta": 1.2}
    base.update(kw)
    return base


def _sin_edgar(monkeypatch):
    from src.levels import valuation_floors as vf

    def boom(t):
        raise ConnectionError("sin xbrl")
    monkeypatch.setattr(vf, "_facts", boom)


def test_moneda_de_reporte_distinta_no_rellena_fundamentales_desde_yfinance(monkeypatch):
    _sin_edgar(monkeypatch)
    e = fv.extraer_entradas("TSM", None, _df_precios(), _info_fx())
    for campo in ("fcf_ltm", "ingresos_ltm", "caja", "deuda_total", "eps_ltm", "bvps", "ebitda_ltm", "acciones_diluidas"):
        assert getattr(e, campo) is None, campo
    assert e.dividendo_anual == 0.0 and e.beta == 1.2           # beta y dividendos del df no dependen de la moneda
    assert e.moneda_conflicto == "moneda de reporte (TWD) ≠ cotización (USD)"
    assert any("moneda de reporte (TWD)" in a for a in e.avisos)
    ms = fv.calcular_modelos(e)
    assert not any(m["aplicable"] for m in ms)
    assert all(m["motivo_exclusion"] == "moneda de reporte (TWD) ≠ cotización (USD)" for m in ms if m["id"] != "ddm")
    assert next(m for m in ms if m["id"] == "ddm")["motivo_exclusion"] == "sin dividendos"
    agg = fv.agregar_valor_justo(ms, None, 0.25, e.precio)
    assert agg["fair_value"] is None and agg["motivo"]


def test_misma_moneda_si_rellena_desde_yfinance(monkeypatch):
    _sin_edgar(monkeypatch)
    e = fv.extraer_entradas("X", None, _df_precios(), _info_fx(financialCurrency="USD", bookValue=20.0, trailingEps=5.0))
    assert e.moneda_conflicto is None and e.fcf_ltm == 1.0e12 and e.caja == 4.0e12 and e.bvps == 20.0
    sin_dato = fv.extraer_entradas("X", None, _df_precios(), {"freeCashflow": 5.0})   # sin monedas: comportamiento anterior
    assert sin_dato.fcf_ltm == 5.0 and sin_dato.moneda_conflicto is None


def test_moneda_distinta_pero_xbrl_en_usd_se_usa_y_no_se_mezcla_con_info():
    e = fv.extraer_entradas("X", None, _df_precios(), _info_fx(totalCash=9.9e12, freeCashflow=7.0e12),
                            facts=_facts_falsos())
    assert e.fcf_ltm == 100.0 and e.caja == 500.0          # XBRL (USD) manda; el info en TWD ni rellena ni corrige
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(pe_objetivo=10.0))}
    assert ms["dcf_5y_fcf"]["aplicable"] and ms["pe"]["aplicable"]


def test_bvps_de_otra_clase_se_descarta_y_los_modelos_lo_explican(monkeypatch):
    _sin_edgar(monkeypatch)
    info = {"currency": "USD", "financialCurrency": "USD", "bookValue": 536413.0, "trailingEps": 35.7,
            "sharesOutstanding": 1.4e9, "totalCash": 3.7e11, "totalDebt": 1.3e11}
    e = fv.extraer_entradas("BRK-B", None, _df_precios(precio=580.0), info)
    assert e.bvps is None and e.datos_inconsistentes == ["bvps"] and e.eps_ltm == 35.7
    assert any("BVPS" in a and "descartado" in a for a in e.avisos)
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(pe_objetivo=15.0, pb_objetivo=1.5))}
    for i in ("pb", "graham"):
        assert not ms[i]["aplicable"] and ms[i]["motivo_exclusion"].startswith("dato por acción inconsistente"), i
    assert ms["pe"]["aplicable"]


def test_eps_inconsistente_con_el_precio_se_descarta():
    e = fv.extraer_entradas("X", None, _df_precios(precio=100.0), {"trailingEps": 0.001, "bookValue": 50.0})
    assert e.eps_ltm is None and "eps_ltm" in e.datos_inconsistentes and e.bvps == 50.0
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(pe_objetivo=15.0))}
    assert "inconsistente" in ms["pe"]["motivo_exclusion"] and "inconsistente" in ms["lynch"]["motivo_exclusion"]
    normal = fv.extraer_entradas("X", None, _df_precios(precio=100.0), {"trailingEps": 6.0, "bookValue": 40.0})
    assert normal.datos_inconsistentes == [] and normal.bvps == 40.0
    lo, hi = fv.PB_IMPLICITO_RANGO
    limite = fv.extraer_entradas("X", None, _df_precios(precio=100.0), {"bookValue": 100.0 / hi})
    assert limite.bvps is not None   # justo en el borde: se conserva


# ============================================================================== diferencia minima WACC - g
@pytest.mark.parametrize("wacc, g", [(0.05, 0.0499), (0.10, 0.095), (0.0599, 0.05)])
def test_dcf_epv_exigen_dos_puntos_y_medio_de_diferencia(wacc, g):
    e = ent(fcf_ltm=100.0, ebitda_ltm=200.0, da_ltm=40.0, capex_ltm=40.0, ingresos_ltm=1000.0, ebit_ltm=100.0)
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(wacc=wacc, g=g))}
    for i in [k for k in ms if k.startswith("dcf_")] + ["epv"]:
        assert not ms[i]["aplicable"] and "diferencia WACC-g < 2.5 pp" in ms[i]["motivo_exclusion"], i
    assert fv.DIF_MIN_WACC_G == 0.025


def test_ddm_exige_dos_puntos_y_medio_entre_ke_y_g():
    e = ent(dividendo_anual=3.0, beta=1.0)   # ke = 0.04 + 0.05 = 0.09
    assert "diferencia ke-g < 2.5 pp" in modelo(e, "ddm", Par(rf=0.04, erp=0.05, g=0.07))["motivo_exclusion"]
    assert modelo(e, "ddm", Par(rf=0.04, erp=0.05, g=0.06))["aplicable"]


def test_exactamente_dos_puntos_y_medio_si_aplica():
    e = ent(fcf_ltm=100.0, ebit_ltm=100.0)
    p = Par(wacc=0.10, g=0.075)
    assert modelo(e, "dcf_5y_fcf", p)["aplicable"] and modelo(e, "epv", p)["aplicable"]


def test_dcf_con_wacc_cercano_a_g_ya_no_explota():
    e = _entradas_ricas()
    agg = fv.valor_justo(e, Par(wacc=0.05, g=0.0499))
    assert not any(m["aplicable"] for m in agg["modelos"] if m["id"].startswith("dcf_"))
    assert agg["fair_value"] is None or agg["fair_value"] < 10 * e.precio


# ============================================================================== series crecientes (desfase / look-ahead)
def _facts_crecientes(n_trim=28):
    """Ingresos = 100*(i+1), EPS = 0.1*(i+1), OCF = 30*(i+1) y capex 5 por trimestre i (0 = el mas antiguo)."""
    fin = (pd.Timestamp.now().normalize() - pd.Timedelta(days=50))
    cierres = [fin - pd.Timedelta(days=91 * i) for i in range(n_trim)][::-1]
    filas = []

    def add(concepto, tipo, f):
        for i, d in enumerate(cierres):
            filas.append(dict(concept=concepto, period_type=tipo,
                              period_start=(d - pd.Timedelta(days=91)) if tipo == "duration" else pd.NaT,
                              period_end=d, numeric_value=f(i), fiscal_year=d.year,
                              fiscal_period=f"Q{(d.month - 1) // 3 + 1}"))
    add("us-gaap:Revenues", "duration", lambda i: 100.0 * (i + 1))
    add("us-gaap:EarningsPerShareDiluted", "duration", lambda i: 0.1 * (i + 1))
    add("us-gaap:NetCashProvidedByUsedInOperatingActivities", "duration", lambda i: 30.0 * (i + 1))
    add("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "duration", lambda i: 5.0)
    add("us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding", "duration", lambda i: 10.0)
    add("us-gaap:CashAndCashEquivalentsAtCarryingValue", "instant", lambda i: 50.0)
    add("us-gaap:LongTermDebt", "instant", lambda i: 20.0)
    add("us-gaap:StockholdersEquity", "instant", lambda i: 100.0 * (i + 1))
    f = pd.DataFrame(filas)
    f["dur"] = (f["period_end"] - f["period_start"]).dt.days
    f["ref_filing"] = f.groupby(["fiscal_year", "fiscal_period"])["period_end"].transform("max")
    return f, cierres


def _df_rampa(dias=2300, base=50.0, pendiente=0.5):
    """Precio que SUBE cada rueda: cualquier desfase en la fecha del lookup cambia el multiplo."""
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=dias)
    px_ = base + pendiente * np.arange(dias)
    return pd.DataFrame({"Open": px_, "High": px_, "Low": px_, "Close": px_, "Volume": 1e6, "Dividends": 0.0,
                         "Stock Splits": 0.0}, index=idx)


def _suma(fn, n, k):
    """Suma de los 4 trimestres que terminan k anios antes del ultimo (indices n-4-4k ... n-1-4k)."""
    return sum(fn(i) for i in range(n - 4 - 4 * k, n - 4 * k))


def test_historias_anuales_y_ltm_son_exactos_con_series_crecientes():
    f, cierres = _facts_crecientes()
    e = fv.extraer_entradas("T", None, _df_rampa(), {}, facts=f)
    n = len(cierres)
    # 28 trimestres = 7 ejercicios LTM (k = 6 ... 0) en modo local (hasta 10 anios + TTM)
    esperado_rev = [_suma(lambda i: 100.0 * (i + 1), n, k) for k in range(6, -1, -1)]
    esperado_fcf = [_suma(lambda i: 30.0 * (i + 1) - 5.0, n, k) for k in range(6, -1, -1)]
    esperado_eps = [_suma(lambda i: 0.1 * (i + 1), n, k) for k in range(6, -1, -1)]
    assert e.hist_ingresos == pytest.approx(esperado_rev, rel=1e-12)
    assert e.hist_fcf == pytest.approx(esperado_fcf, rel=1e-12)
    assert e.hist_eps == pytest.approx(esperado_eps, rel=1e-12)
    assert e.ingresos_ltm == 100.0 * (25 + 26 + 27 + 28)
    assert e.fcf_ltm == esperado_fcf[-1] and e.eps_ltm == pytest.approx(esperado_eps[-1])


def test_multiplos_historicos_exactos_y_sin_lookahead():
    f, cierres = _facts_crecientes()
    df = _df_rampa()
    e = fv.extraer_entradas("T", None, df, {}, facts=f)
    n = len(cierres)
    px_ = df["Close"]
    pe, pfcf, pb = [], [], []
    for k in range(6, 0, -1):     # hasta 6 anios atras con 28 trimestres de datos
        fecha = cierres[n - 1 - 4 * k] + pd.Timedelta(days=fv.REZAGO_PUBLICACION_DIAS)
        precio = float(px_[px_.index <= fecha].iloc[-1])                 # ultimo precio CONOCIDO (<= cierre + rezago)
        pe.append(precio / _suma(lambda i: 0.1 * (i + 1), n, k))
        pfcf.append(precio * 10.0 / _suma(lambda i: 30.0 * (i + 1) - 5.0, n, k))
        pb.append(precio * 10.0 / (100.0 * (n - 4 * k)))                 # patrimonio del cierre de la ventana
        assert float(px_[px_.index <= fecha + pd.Timedelta(days=30)].iloc[-1]) > precio   # la rampa delata un look-ahead
    assert e.multiplos_hist["pe"] == pytest.approx(pe, rel=1e-12)
    assert e.multiplos_hist["p_fcf"] == pytest.approx(pfcf, rel=1e-12)
    assert e.multiplos_hist["pb"] == pytest.approx(pb, rel=1e-12)


def test_un_precio_posterior_al_cierre_mas_rezago_no_cambia_el_multiplo():
    f, cierres = _facts_crecientes()
    base = fv.extraer_entradas("T", None, _df_rampa(), {}, facts=f)
    df2 = _df_rampa()
    n = len(cierres)
    k = fv.ANIOS_HISTORIA
    corte = cierres[n - 1 - 4 * k] + pd.Timedelta(days=fv.REZAGO_PUBLICACION_DIAS)
    df2.loc[(df2.index > corte) & (df2.index <= corte + pd.Timedelta(days=20)), "Close"] *= 10.0   # solo POSTERIOR al corte
    alt = fv.extraer_entradas("T", None, df2, {}, facts=f)
    assert alt.multiplos_hist["pe"][0] == pytest.approx(base.multiplos_hist["pe"][0], rel=1e-12)
    df3 = _df_rampa()
    df3.loc[df3.index <= corte, "Close"] *= 10.0                                                   # ANTERIOR: si cambia
    assert fv.extraer_entradas("T", None, df3, {}, facts=f).multiplos_hist["pe"][0] != pytest.approx(
        base.multiplos_hist["pe"][0], rel=1e-3)


def test_una_ventana_anual_desfasada_se_detecta():
    f, cierres = _facts_crecientes()
    e = fv.extraer_entradas("T", None, _df_rampa(), {}, facts=f)
    n = len(cierres)
    desfasada = [_suma(lambda i: 100.0 * (i + 1), n - 1, k) for k in range(5, -1, -1)]   # un trimestre antes
    assert e.hist_ingresos != pytest.approx(desfasada, rel=1e-6)
