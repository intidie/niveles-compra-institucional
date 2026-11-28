"""Tests del valor justo (src/levels/fair_value.py) y de su integracion en la UI. Todo sin red.

Los casos numericos se calculan a mano con aritmetica simple (ver comentarios); la tolerancia es explicita en cada
``approx``. Las entradas son ``EntradasValoracion`` sinteticas; ``extraer_entradas`` se prueba con una tabla XBRL falsa.
"""
import asyncio
import json
import math
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import app_ui as ui_mod
from src.levels import fair_value as fv
from src.levels.fair_value import EntradasValoracion as Ent
from src.levels.fair_value import ParametrosValoracion as Par

TOL = 1e-9
MA = 1.1 ** 0.5     # factor de mitad de anio con WACC = 10 %: (1+0.10)^0.5


@pytest.fixture(autouse=True)
def _sin_red_ni_disco(monkeypatch, tmp_path):
    """Ningun test toca la red (^TNX) ni el directorio de datos real del usuario (cache SQLite)."""
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "datos"))
    monkeypatch.setattr(fv, "obtener_rf", lambda *a, **k: None)


def ent(**kw):
    """Entradas base: precio 100, 10 acciones, sin deuda ni caja (caja neta = 0), sin historia."""
    base = dict(ticker="T", precio=100.0, acciones_diluidas=10.0, caja=0.0, deuda_total=0.0, tasa_impositiva=0.25)
    base.update(kw)
    return Ent(**base)


def modelo(entradas, id_, params=None):
    return next(m for m in fv.calcular_modelos(entradas, params) if m["id"] == id_)


def px(entradas, id_, params=None):
    return modelo(entradas, id_, params)["precio_estimado"]


# ============================================================================== estructura
def test_quince_modelos_con_ids_estables_y_campos():
    ms = fv.calcular_modelos(ent())
    assert [m["id"] for m in ms] == ["dcf_5y_fcf", "dcf_10y_fcf", "dcf_5y_ebitda", "dcf_10y_ebitda", "dcf_5y_revenue",
                                     "dcf_10y_revenue", "pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf", "ddm", "epv",
                                     "graham", "lynch"]
    for m in ms:
        assert {"id", "nombre", "familia", "precio_estimado", "aplicable", "motivo_exclusion"} <= set(m)
        assert m["familia"] in fv.FAMILIAS and m["nombre"]
    assert {m["familia"] for m in ms} == {"DCF", "Múltiplos", "Dividendos", "Valor"}


def test_sin_datos_nada_aplica_con_motivo_y_sin_lanzar():
    for e in (Ent(), {}, None, {"ticker": "X", "campo_desconocido": 1}):
        ms = fv.calcular_modelos(e)
        assert len(ms) == 15 and not any(m["aplicable"] for m in ms)
        assert all(m["precio_estimado"] is None and m["motivo_exclusion"] for m in ms)


def test_entradas_dict_equivale_a_objeto_y_es_serializable():
    e = ent(fcf_ltm=100.0, hist_fcf=[90.0, 100.0])
    d = e.to_dict()
    json.dumps(d)  # liviano y serializable
    assert Ent.desde_dict(d) == e
    assert fv.calcular_modelos(d, {"g": 0.02}) == fv.calcular_modelos(e, Par(g=0.02))


def test_sin_nan_ni_inf_en_ningun_modelo():
    e = ent(fcf_ltm=1e12, ebitda_ltm=2e12, ebit_ltm=1.5e12, ingresos_ltm=5e12, eps_ltm=3.0, bvps=9.0, da_ltm=1e11,
            capex_ltm=1e11, dividendo_anual=2.0, hist_fcf=[1.0, 1e12], hist_eps=[1e-9, 3.0],
            multiplos_hist={k: [10.0] * 5 for k in fv.RANGOS_MULTIPLOS})
    for m in fv.calcular_modelos(e):
        assert m["precio_estimado"] is None or math.isfinite(m["precio_estimado"])


# ============================================================================== formulas DCF
def test_dcf_fcf_crecimiento_constante_es_gordon_exacto():
    # Sin historia el crecimiento = g en todos los anios: FCF_t = 100*(1.02)^t. Con flujos a MITAD DE ANIO (DF_t =
    # 1.1^-(t-0.5)) y el valor terminal descontado con el DF del ultimo flujo (t = N-0.5), el total es la perpetuidad
    # de Gordon desplazada medio anio: 100*1.02/0.08 * 1.1^0.5 = 1275 * 1.0488088 = 1337.23 => 133.723 por accion
    # (caja neta 0, 10 acciones), igual a 5 y a 10 anios.
    e, p = ent(fcf_ltm=100.0), Par(wacc=0.10, g=0.02)
    assert px(e, "dcf_5y_fcf", p) == pytest.approx(127.5 * MA, rel=TOL)
    assert px(e, "dcf_10y_fcf", p) == pytest.approx(127.5 * MA, rel=TOL)


def test_dcf_fcf_suma_caja_neta_y_divide_entre_acciones_diluidas():
    # (1275*1.1^0.5 + caja 50 - deuda 20) / 10 ; con 20 acciones: (1275*1.1^0.5 + 30) / 20
    e, p = ent(fcf_ltm=100.0, caja=50.0, deuda_total=20.0), Par(wacc=0.10, g=0.02)
    assert px(e, "dcf_5y_fcf", p) == pytest.approx((1275 * MA + 30) / 10, rel=TOL)
    assert px(ent(fcf_ltm=100.0, caja=50.0, deuda_total=20.0, acciones_diluidas=20.0), "dcf_5y_fcf", p) == \
        pytest.approx((1275 * MA + 30) / 20, rel=TOL)


def _dcf_manual(f0, g1, g, w, n, neto=0.0, acc=10.0):
    """DCF de 2 etapas a mano: anios 1-3 crecen g1; anios 4..n decaen linealmente hasta g en el anio n; mitad de anio."""
    flujos, f = [], f0
    for t in range(1, n + 1):
        gt = g1 if t <= 3 else g1 + (g - g1) * (t - 3) / (n - 3)
        f *= 1 + gt
        flujos.append(f)
    vp = sum(fl / (1 + w) ** (t - 0.5) for t, fl in enumerate(flujos, 1))
    vt = flujos[-1] * (1 + g) / (w - g) / (1 + w) ** (n - 0.5)     # TV con el DF del ultimo flujo (t = n - 0.5)
    return (vp + vt + neto) / acc


def test_dcf_fcf_crecimiento_converge_linealmente_hacia_g():
    # hist [100,110,121]: CAGR winsorizado 10 % (3Y y 5Y), momentum 10 %, sin forward -> crecimiento fundamental 10 %;
    # etapa 1 = 65 % * 10 % = 6.5 % (anios 1-3); la etapa 2 decae linealmente hasta g = 2 % en el anio n.
    e, p = ent(fcf_ltm=121.0, hist_fcf=[100.0, 110.0, 121.0]), Par(wacc=0.10, g=0.02)
    assert fv.FRACCION_CRECIMIENTO == 0.65 and fv.ANIOS_ETAPA1 == 3
    assert px(e, "dcf_5y_fcf", p) == pytest.approx(_dcf_manual(121.0, 0.065, 0.02, 0.10, 5), rel=1e-9)
    assert px(e, "dcf_10y_fcf", p) == pytest.approx(_dcf_manual(121.0, 0.065, 0.02, 0.10, 10), rel=1e-9)
    assert px(e, "dcf_5y_fcf", p) > 121.0 * 1.02 / 0.08 / 10.0 * MA   # mas que con crecimiento constante g


def test_dcf_crecimiento_inicial_se_acota_al_tope():
    e = ent(fcf_ltm=100.0, hist_fcf=[1.0, 1000.0])  # variacion +99900 % -> recorte duro a +100 % -> 65 % -> tope
    r = fv.crecimiento_etapa1(e.hist_fcf, None, 0.02, "medio")
    assert r["g1"] == fv.TOPE_CRECIMIENTO_ETAPA1["medio"] == 0.42 and "topado a 42 %" in r["tope"]
    assert fv.crecimiento_etapa1(e.hist_fcf, None, 0.02, "pesimista")["g1"] == 0.15      # tope del Pesimista
    g0, _ = fv._crecimiento_inicial([100.0, 1.0], 0.02)  # caida fuerte (-50 % recortado) * 65 % = -32.5 % -> piso
    assert g0 == fv.CRECIMIENTO_MIN


def test_dcf_ebitda_formula_documentada():
    # flujo0 = EBITDA*(1-t) + D&A*t - capex = 200*0.75 + 40*0.25 - 40 = 120 ; sin NWC, sin historia (la conversion
    # objetivo = la actual 0.6) y crecimiento g constante: valor = 120*1.02/0.08 * 1.1^0.5 = 1530*MA -> 153.0*MA
    e = ent(ebitda_ltm=200.0, da_ltm=40.0, capex_ltm=40.0)
    p = Par(wacc=0.10, g=0.02)
    assert px(e, "dcf_5y_ebitda", p) == pytest.approx(153.0 * MA, rel=TOL)
    assert px(e, "dcf_10y_ebitda", p) == pytest.approx(153.0 * MA, rel=TOL)


def test_dcf_ebitda_con_capital_de_trabajo():
    # nwc 100 / EBITDA 200 = 0.5 ; k = 0.6. Anios 1-3 (etapa 1): flujo_t = eb_t*0.6 - 0.5*(eb_t - eb_{t-1}) con
    # eb_t = 200*1.02^t. Desde el anio 4 el flujo converge linealmente a la conversion objetivo (0.6, sin historia):
    # flujo_t = (1-w)*estructural + w*0.6*eb_t con w = (t-3)/(n-3). Calculo explicito (n = 5, WACC 10 %, g 2 %):
    n = 5
    prev, flujos = 200.0, []
    for t in range(1, n + 1):
        eb = prev * 1.02
        estr = eb * 0.6 - 0.5 * (eb - prev)
        wt = 0.0 if t <= 3 else (t - 3) / (n - 3)
        flujos.append((1 - wt) * estr + wt * 0.6 * eb)
        prev = eb
    esperado = (sum(f / 1.1 ** (t - 0.5) for t, f in enumerate(flujos, 1)) + flujos[-1] * 1.02 / 0.08 / 1.1 ** 4.5) / 10
    e = ent(ebitda_ltm=200.0, da_ltm=40.0, capex_ltm=40.0, nwc_operativo=100.0)
    assert px(e, "dcf_5y_ebitda", Par(wacc=0.10, g=0.02)) == pytest.approx(esperado, rel=1e-12)
    assert esperado < 153.0 * MA      # el capital de trabajo resta en la etapa 1


def test_dcf_revenue_margen_constante():
    # ingresos 1000, FCF 100 -> margen 10 % constante: igual que el DCF de FCF = 127.5 * 1.1^0.5
    e = ent(ingresos_ltm=1000.0, fcf_ltm=100.0)
    p = Par(wacc=0.10, g=0.02)
    assert px(e, "dcf_5y_revenue", p) == pytest.approx(127.5 * MA, rel=TOL)
    assert px(e, "dcf_10y_revenue", p) == pytest.approx(127.5 * MA, rel=TOL)


def test_dcf_revenue_margen_converge_al_objetivo_historico():
    # margen actual 5 %, mediana recortada historica 10 % -> objetivo 10 %; ingresos constantes (crecimiento 0): el margen
    # se mantiene 5 % en los anios 1-3 y converge linealmente en los anios 4-5 (n = 5): m_4 = 7.5 %, m_5 = 10 % ->
    # flujos 50, 50, 50, 75, 100 ; VT = 100/0.10 = 1000 descontado con DF(4.5)
    e = ent(ingresos_ltm=1000.0, fcf_ltm=50.0, hist_fcf=[100.0, 100.0, 100.0], hist_ingresos=[1000.0] * 3)
    esperado = (sum(f / 1.1 ** (t - 0.5) for t, f in enumerate([50, 50, 50, 75, 100], 1)) + 1000 / 1.1 ** 4.5) / 10
    assert px(e, "dcf_5y_revenue", Par(wacc=0.10, g=0.0)) == pytest.approx(esperado, rel=1e-9)


def test_dcf_revenue_margen_objetivo_tiene_tope():
    e = ent(ingresos_ltm=1000.0, fcf_ltm=600.0, hist_fcf=[600.0] * 3, hist_ingresos=[1000.0] * 3)  # margen 60 %
    act, obj, _ = fv._margen_fcf(e)
    assert act == 0.6 and obj == fv.TOPE_MARGEN["medio"] == 0.55         # tope Medio 55 %
    assert fv._margen_fcf(e, "pesimista")[1] == fv.TOPE_MARGEN["pesimista"] == 0.30   # tope Pesimista 30 %
    assert px(e, "dcf_5y_revenue", Par(wacc=0.10, g=0.0)) < px(e, "dcf_5y_fcf", Par(wacc=0.10, g=0.0))
    assert modelo(e, "dcf_5y_revenue", Par(wacc=0.10, g=0.0))["tope_activo"]


def test_dcf_revenue_sin_fcf_usa_margen_operativo_despues_de_impuestos():
    e = ent(ingresos_ltm=1000.0, ebit_ltm=200.0)  # 200*(1-0.25)=150 -> margen 15 %
    assert fv._margen_fcf(e)[2].startswith("margen operativo")
    assert px(e, "dcf_5y_revenue", Par(wacc=0.10, g=0.02)) == pytest.approx(150 * 1.02 / 0.08 / 10 * MA, rel=TOL)


# ============================================================================== multiplos
def test_pe_con_multiplo_fijado_y_con_mediana_historica():
    assert px(ent(eps_ltm=4.0), "pe", Par(pe_objetivo=15.0)) == pytest.approx(60.0, rel=TOL)
    e = ent(eps_ltm=4.0, multiplos_hist={"pe": [10.0, 12.0, 14.0, 16.0, 18.0]})  # mediana 14 -> 56
    assert px(e, "pe") == pytest.approx(56.0, rel=TOL)


def test_multiplo_historico_se_acota_al_rango_y_exige_historia():
    lo, hi = fv.RANGOS_MULTIPLOS["pe"]
    assert px(ent(eps_ltm=2.0, multiplos_hist={"pe": [80.0] * 4}), "pe") == pytest.approx(2.0 * hi, rel=TOL)
    assert px(ent(eps_ltm=2.0, multiplos_hist={"pe": [1.0] * 4}), "pe") == pytest.approx(2.0 * lo, rel=TOL)
    m = modelo(ent(eps_ltm=2.0, multiplos_hist={"pe": [10.0, 12.0]}), "pe")
    assert not m["aplicable"] and "historia insuficiente" in m["motivo_exclusion"]


def test_ev_ebitda_y_ev_revenue():
    # (EBITDA*m + caja - deuda)/acciones = (200*10 + 50 - 20)/10 = 203 ; EV/Ingresos: (1000*2 + 30)/10 = 203
    e = ent(ebitda_ltm=200.0, ingresos_ltm=1000.0, caja=50.0, deuda_total=20.0)
    assert px(e, "ev_ebitda", Par(ev_ebitda_objetivo=10.0)) == pytest.approx(203.0, rel=TOL)
    assert px(e, "ev_revenue", Par(ev_revenue_objetivo=2.0)) == pytest.approx(203.0, rel=TOL)


def test_pb_y_p_fcf():
    assert px(ent(bvps=20.0), "pb", Par(pb_objetivo=3.0)) == pytest.approx(60.0, rel=TOL)   # BVPS * P/B
    assert px(ent(fcf_ltm=100.0), "p_fcf", Par(p_fcf_objetivo=12.0)) == pytest.approx(120.0, rel=TOL)  # (100/10)*12


def test_pb_no_informativo_con_multiplo_historico_enorme():
    m = modelo(ent(bvps=5.0, multiplos_hist={"pb": [fv.PB_NO_INFORMATIVO + 5] * 4}), "pb")
    assert not m["aplicable"] and "no es informativo" in m["motivo_exclusion"]


# ============================================================================== dividendos y valor
def test_ddm_gordon_con_ke_capm():
    # beta de Blume(1.0) = 1.0 ; ke = rf + beta*ERP = 0.04 + 1*0.05 = 0.09 ; g∞ Medio = 3.5 % ; D1 = 2*1.035 = 2.07 ;
    # valor = 2.07/(0.09-0.035) = 37.636363
    e = ent(dividendo_anual=2.0, beta=1.0)
    p = Par(rf=0.04, erp=0.05)
    assert px(e, "ddm", p) == pytest.approx(2.07 / 0.055, rel=TOL)


def test_ddm_el_wacc_del_usuario_desplaza_ke_en_paralelo():
    # sin deuda: WACC automatico = ke = 0.09 (dentro de [8 %, 11.5 %]) ; con WACC = 0.10 el ke pasa a 0.10 -> 2.07/(0.10-0.035)
    e = ent(dividendo_anual=2.0, beta=1.0)
    assert px(e, "ddm", Par(rf=0.04, erp=0.05, wacc=0.10)) == pytest.approx(2.07 / 0.065, rel=TOL)


@pytest.mark.parametrize("kw, params, texto", [
    (dict(dividendo_anual=0.0), None, "sin dividendos"),
    (dict(dividendo_anual=None), None, "dividendo no disponible"),
    (dict(dividendo_anual=0.5), None, "simbólico"),                       # 0.5 % de rendimiento
    (dict(dividendo_anual=2.0, beta=1.0), Par(rf=0.04, erp=0.05, g=0.09), "no converge"),   # ke 9 % <= g 9 %
])
def test_ddm_no_aplicable(kw, params, texto):
    m = modelo(ent(**kw), "ddm", params)
    assert not m["aplicable"] and texto in m["motivo_exclusion"]


def test_epv_greenwald():
    # NOPAT = 100*(1-0.2) = 80 ; EPV = 80/0.10 = 800 ; + caja neta 30 -> 830 / 10 = 83
    e = ent(ebit_ltm=100.0, tasa_impositiva=0.2, caja=50.0, deuda_total=20.0)
    assert px(e, "epv", Par(wacc=0.10)) == pytest.approx(83.0, rel=TOL)


def test_graham():
    assert px(ent(eps_ltm=4.0, bvps=10.0), "graham") == pytest.approx(30.0, rel=TOL)  # sqrt(22.5*4*10) = sqrt(900)
    for kw in (dict(eps_ltm=-1.0, bvps=10.0), dict(eps_ltm=4.0, bvps=-2.0), dict(eps_ltm=0.0, bvps=10.0)):
        assert not modelo(ent(**kw), "graham")["aplicable"]


def test_lynch_peg_uno_con_tope():
    hist = [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]  # CAGR 10 % a 5 anios
    assert px(ent(eps_ltm=2.0, hist_eps=hist), "lynch") == pytest.approx(20.0, rel=1e-9)   # 2 * 10
    m = modelo(ent(eps_ltm=2.0, hist_eps=[1.0, 10.0]), "lynch")                              # +900 % -> tope 25
    assert m["precio_estimado"] == pytest.approx(2.0 * fv.LYNCH_CRECIMIENTO_TOPE_PCT, rel=TOL)
    assert not modelo(ent(eps_ltm=-1.0, hist_eps=hist), "lynch")["aplicable"]
    assert "<= 0" in modelo(ent(eps_ltm=2.0, hist_eps=[10.0, 8.0]), "lynch")["motivo_exclusion"]
    assert not modelo(ent(eps_ltm=2.0, hist_eps=[]), "lynch")["aplicable"]


# ============================================================================== WACC y supuestos
def test_wacc_ponderado_por_estructura_de_capital():
    # capitalizacion 100*10 = 1000, deuda 1000 -> 50/50 ; ke = 0.04+1*0.08 = 0.12 ; kd = 60/1000 = 6 % ;
    # WACC = 0.5*0.12 + 0.5*0.06*(1-0.25) = 0.06 + 0.0225 = 0.0825 (por encima del piso WACC_MIN = 7.5 %)
    e = ent(deuda_total=1000.0, gasto_interes_ltm=60.0, beta=1.0)
    t = fv.calcular_tasas(e, Par(rf=0.04, erp=0.08))
    assert t["peso_deuda"] == pytest.approx(0.5) and t["kd"] == pytest.approx(0.06)
    assert t["ke_capm"] == pytest.approx(0.12) and t["wacc"] == pytest.approx(0.0825, abs=1e-12)


def test_wacc_automatico_acotado_y_beta_acotada():
    # beta_raw 5.0 se descarta a BETA_RAW_MAX = 4.0 y Blume la lleva a 2/3*4+1/3 = 3.0 ; ke = 0.04+3*0.10 = 0.34 -> WACC 11.5 %
    alto = fv.calcular_tasas(ent(beta=5.0), Par(rf=0.04, erp=0.10))
    assert alto["beta"] == pytest.approx(3.0) and alto["wacc"] == fv.WACC_MAX == 0.115
    # beta_raw 0.1 -> Blume 0.4 ; ke = 0.01+0.4*0.05 = 0.03 -> WACC 8.0 % (piso)
    bajo = fv.calcular_tasas(ent(beta=0.1), Par(rf=0.01, erp=0.05))
    assert bajo["beta"] == pytest.approx(0.4) and bajo["wacc"] == fv.WACC_MIN == 0.08
    sin_beta = fv.calcular_tasas(ent(), Par())
    assert sin_beta["beta"] == fv.BETA_DEFECTO and "no disponible" in sin_beta["beta_origen"]


def test_wacc_del_usuario_sustituye_al_automatico():
    t = fv.calcular_tasas(ent(beta=1.0), Par(wacc=0.123))
    assert t["wacc"] == 0.123 and t["wacc_origen"] == "usuario" and t["wacc_auto"] != 0.123


def test_g_mayor_o_igual_que_wacc_desactiva_dcf_con_motivo_pero_no_los_multiplos():
    e = ent(fcf_ltm=100.0, ebitda_ltm=200.0, da_ltm=40.0, capex_ltm=40.0, ingresos_ltm=1000.0, eps_ltm=4.0,
            ebit_ltm=100.0)
    p = Par(wacc=0.10, g=0.10, pe_objetivo=15.0)
    ms = {m["id"]: m for m in fv.calcular_modelos(e, p)}
    for i in [k for k in ms if k.startswith("dcf_")]:
        assert not ms[i]["aplicable"] and "WACC" in ms[i]["motivo_exclusion"], i
    assert ms["pe"]["aplicable"] and not ms["epv"]["aplicable"]   # el EPV tambien exige WACC - g >= 1 pp
    assert not fv.supuestos(e, p)["g_valido"]
    assert fv.supuestos(e, Par(wacc=0.10, g=0.02))["g_valido"]


def test_supuestos_expone_todo_lo_usado():
    e = ent(fcf_ltm=100.0, hist_fcf=[100.0, 121.0], beta=1.0, multiplos_hist={"pe": [10.0, 12.0, 14.0]})
    s = fv.supuestos(e, Par(wacc=0.11, g=0.03, margen_seguridad=0.2))
    assert s["wacc"] == 0.11 and s["g"] == 0.03 and s["margen_seguridad"] == 0.2 and s["g_valido"]
    assert s["crecimiento_inicial_fcf"] == pytest.approx(0.65 * 0.21)   # momentum 21 % (1 sola variacion: sin CAGR)
    assert set(s["multiplos"]) == set(fv.RANGOS_MULTIPLOS) and s["multiplos"]["pe"]["valor"] == 12.0
    assert s["multiplos"]["ev_ebitda"]["valor"] is None
    assert {"ke", "kd", "rf", "erp", "beta", "tasa_impositiva", "margen_fcf_objetivo"} <= set(s)
    json.dumps({k: v for k, v in s.items()}, default=str)


# ============================================================================== aplicabilidad
def test_flujos_negativos_o_cero_excluyen_con_motivo():
    e = ent(fcf_ltm=-5.0, ebitda_ltm=-1.0, da_ltm=1.0, capex_ltm=1.0, ingresos_ltm=100.0, eps_ltm=-0.5, bvps=3.0,
            ebit_ltm=-2.0, dividendo_anual=0.0)
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(pe_objetivo=15.0, p_fcf_objetivo=10.0, ev_ebitda_objetivo=8.0))}
    assert "FCF LTM negativo" in ms["dcf_5y_fcf"]["motivo_exclusion"] and not ms["p_fcf"]["aplicable"]
    assert "EBITDA LTM negativo" in ms["dcf_10y_ebitda"]["motivo_exclusion"] and not ms["ev_ebitda"]["aplicable"]
    assert "EPS LTM negativo" in ms["pe"]["motivo_exclusion"]
    assert not ms["graham"]["aplicable"] and not ms["lynch"]["aplicable"] and not ms["epv"]["aplicable"]
    assert ms["pb"]["motivo_exclusion"] is not None  # sin multiplo historico


def test_acciones_o_caja_faltantes_excluyen_los_modelos_que_las_usan():
    e = ent(acciones_diluidas=None, fcf_ltm=100.0, eps_ltm=4.0, ebitda_ltm=50.0, ebit_ltm=40.0, ingresos_ltm=500.0)
    p = Par(ev_ebitda_objetivo=10.0, p_fcf_objetivo=10.0)
    for i in ("dcf_5y_fcf", "dcf_10y_revenue", "ev_ebitda", "p_fcf", "epv"):
        assert "acciones diluidas" in (modelo(e, i, p)["motivo_exclusion"] or ""), i
    assert modelo(ent(caja=None, fcf_ltm=100.0), "dcf_5y_fcf")["motivo_exclusion"] == "caja no disponible"
    assert modelo(ent(deuda_total=None, fcf_ltm=100.0), "dcf_5y_fcf")["motivo_exclusion"] == "deuda total no disponible"
    assert px(ent(eps_ltm=4.0), "pe", Par(pe_objetivo=15.0)) == 60.0  # el P/E no necesita acciones ni caja


def test_valor_estimado_negativo_se_marca_no_aplicable():
    e = ent(fcf_ltm=10.0, deuda_total=1e6)  # la deuda supera con creces el valor operativo
    m = modelo(e, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))
    assert not m["aplicable"] and m["precio_estimado"] is None and "<= 0" in m["motivo_exclusion"]


def test_bancos_y_aseguradoras_solo_usan_modelos_de_resultados():
    banco = ent(sector="Financial Services", industria="Banks - Diversified", fcf_ltm=1.0, ebitda_ltm=1.0, da_ltm=1.0,
                capex_ltm=1.0, ingresos_ltm=10.0, ebit_ltm=2.0, eps_ltm=4.0, bvps=10.0, dividendo_anual=3.0,
                hist_eps=[1.0, 2.0, 3.0])
    ms = {m["id"]: m for m in fv.calcular_modelos(banco, Par(pe_objetivo=10.0, pb_objetivo=1.0))}
    for i in fv.MODELOS_NO_FINANCIERAS:
        assert not ms[i]["aplicable"] and "bancos" in ms[i]["motivo_exclusion"], i
    for i in ("pe", "pb", "ddm", "graham", "lynch"):
        assert ms[i]["aplicable"], i
    assert fv.es_financiera("Financial Services", "Insurance - Life")
    assert not fv.es_financiera("Financial Services", "Credit Services", "V")   # Visa/Mastercard: FCF si es informativo
    assert not fv.es_financiera("Technology", "Software - Infrastructure")
    assert not fv.es_financiera(None, None)


# ============================================================================== agregacion
_FAMS = ["DCF", "Múltiplos", "Valor", "Dividendos"]


def _res(id_, precio, aplicable=True):
    fam = _FAMS[int(id_[1:]) % 4] if id_[1:].isdigit() else "DCF"   # m0, m1, m2... reparten familias
    return {"id": id_, "nombre": id_, "familia": fam, "precio_estimado": precio, "aplicable": aplicable,
            "motivo_exclusion": None if aplicable else "x"}


def _lista(valores):
    return [_res(f"m{i}", v) for i, v in enumerate(valores)]


def test_agregar_promedio_upside_y_precio_con_margen():
    # promedio de 100, 120, 80 = 100 ; precio 80 -> upside +25 % ; MoS 25 % -> 75
    a = fv.agregar_valor_justo(_lista([100.0, 120.0, 80.0]), None, 0.25, 80.0)
    assert a["fair_value"] == pytest.approx(100.0, rel=TOL) and a["upside_pct"] == pytest.approx(25.0, rel=TOL)
    assert a["margin_of_safety_price"] == pytest.approx(75.0, rel=TOL)
    assert a["n_validos"] == 3 and a["modelos_usados"] == ["m0", "m1", "m2"] and a["motivo"] is None
    b = fv.agregar_valor_justo(_lista([100.0, 120.0, 80.0]), None, 0.25, 125.0)
    assert b["upside_pct"] == pytest.approx(-20.0, rel=TOL)


@pytest.mark.parametrize("precio", [None, 0, 0.0, -5.0, float("nan")])
def test_agregar_sin_precio_valido_no_da_upside(precio):
    a = fv.agregar_valor_justo(_lista([100.0, 100.0, 100.0]), None, 0.25, precio)
    assert a["upside_pct"] is None and a["fair_value"] == pytest.approx(100.0)


def test_agregar_descarta_no_aplicables_y_precios_no_positivos():
    lista = _lista([100.0, 100.0, 100.0]) + [_res("na", None, False), _res("cero", 0.0), _res("neg", -4.0),
                                             _res("nan", float("nan"))]
    a = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    assert a["modelos_usados"] == ["m0", "m1", "m2"] and a["n_validos"] == 3
    assert set(a["modelos_no_aplicables"]) == {"na", "cero", "neg", "nan"}


def test_filtro_iqr_con_cuatro_o_mas_validos():
    # [98,100,101,102,1000]: Q1 = 100, Q3 = 102, IQR = 2 -> valla [97, 105] -> 1000 es outlier. Familias de _res:
    # m0 DCF 100 ; m1 Multiplos 102 ; m2 Valor 98 ; m3 Dividendos 101 (grupo Valor/Dividendos = (98+101)/2 = 99.5)
    # fair_value = (100 + 102 + 99.5)/3 = 100.5 ; el promedio simple (informativo) = 401/4 = 100.25
    a = fv.agregar_valor_justo(_lista([100.0, 102.0, 98.0, 101.0, 1000.0]), None, 0.25, 100.0)
    assert a["modelos_outlier"] == ["m4"] and a["n_validos"] == 5 and a["n_usados"] == 4
    assert a["fair_value"] == pytest.approx(301.5 / 3, rel=TOL) and a["promedio_simple"] == pytest.approx(100.25, rel=TOL)
    assert a["metodo"] == "igualacion_familias" and a["mediana"] == pytest.approx(100.5, rel=TOL)
    assert a["incertidumbre"] == "Baja"


def test_sin_filtro_iqr_con_menos_de_cuatro_validos():
    a = fv.agregar_valor_justo(_lista([100.0, 102.0, 1000.0]), None, 0.25, 100.0)
    assert fv.MIN_MODELOS_IQR == 4 and a["modelos_outlier"] == []
    assert a["fair_value"] == pytest.approx((100 + 102 + 1000) / 3, rel=TOL)
    assert a["incertidumbre"] == "Alta"


@pytest.mark.parametrize("valores, nivel, cv", [
    ([90.0, 100.0, 110.0], "Baja", math.sqrt(200 / 3) / 100),     # sigma poblacional = sqrt(200/3) = 8.165 -> CV 0.0816
    ([70.0, 100.0, 130.0], "Media", math.sqrt(600) / 100),         # sigma = 24.49 -> CV 0.2449
    ([40.0, 100.0, 160.0], "Alta", math.sqrt(2400) / 100),         # sigma = 48.99 -> CV 0.49
    ([100.0, 100.0, 100.0], "Baja", 0.0),
])
def test_cv_poblacional_y_los_tres_niveles(valores, nivel, cv):
    a = fv.agregar_valor_justo(_lista(valores), None, 0.25, 100.0)
    assert a["cv"] == pytest.approx(cv, rel=1e-9) and a["incertidumbre"] == nivel


def test_fronteras_de_incertidumbre():
    assert fv.CV_BAJO == 0.15 and fv.CV_ALTO == 0.35 and fv.MIN_MODELOS_CONFIABLE == 3
    c = fv.clasificar_incertidumbre
    assert c(0.1499, 5) == "Baja" and c(0.15, 5) == "Media"
    assert c(0.35, 5) == "Media" and c(0.3501, 5) == "Alta"
    assert c(0.01, 2) == "Alta" and c(0.01, 3) == "Baja" and c(None, 9) == "Alta"


@pytest.mark.parametrize("n", [0, 1, 2])
def test_menos_de_tres_modelos_validos_es_incertidumbre_alta(n):
    a = fv.agregar_valor_justo(_lista([100.0] * n), None, 0.25, 100.0)
    assert a["incertidumbre"] == "Alta"
    if n:
        assert a["fair_value"] == 100.0
    else:
        assert a["fair_value"] is None and a["motivo"] and a["upside_pct"] is None and a["margin_of_safety_price"] is None


def test_modelos_activos_none_lista_vacia_y_subconjuntos():
    lista = _lista([100.0, 110.0, 120.0, 130.0])
    todos = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    # DCF 100 ; Multiplos 110 ; Valor/Dividendos (120+130)/2 = 125 -> (100+110+125)/3 ; promedio simple 115
    assert todos["fair_value"] == pytest.approx(335.0 / 3) and todos["promedio_simple"] == pytest.approx(115.0)
    assert todos["modelos_desactivados"] == []
    sub = fv.agregar_valor_justo(lista, ["m0", "m1"], 0.25, 100.0)
    assert sub["fair_value"] == pytest.approx(105.0) and sub["modelos_desactivados"] == ["m2", "m3"]
    assert sub["modelos_usados"] == ["m0", "m1"] and sub["incertidumbre"] == "Alta"   # < 3 modelos
    vacia = fv.agregar_valor_justo(lista, [], 0.25, 100.0)
    assert vacia["fair_value"] is None and "activo" in vacia["motivo"]
    assert vacia["modelos_desactivados"] == ["m0", "m1", "m2", "m3"]
    desconocido = fv.agregar_valor_justo(lista, ["zzz"], 0.25, 100.0)
    assert desconocido["fair_value"] is None


def test_agregar_es_funcion_pura():
    lista = _lista([100.0, 102.0, 98.0, 101.0, 1000.0])
    copia = json.loads(json.dumps(lista))
    a1 = fv.agregar_valor_justo(lista, ["m0", "m1", "m2"], 0.3, 90.0)
    a2 = fv.agregar_valor_justo(lista, ["m0", "m1", "m2"], 0.3, 90.0)
    assert a1 == a2 and lista == copia


def _entradas_ricas():
    """Entradas con casi todos los modelos aplicables y precios dispares (para probar recalculos)."""
    return Ent(ticker="RICA", precio=100.0, acciones_diluidas=10.0, fcf_ltm=100.0, ebitda_ltm=160.0, ebit_ltm=120.0,
               ingresos_ltm=900.0, eps_ltm=6.0, bvps=25.0, caja=100.0, deuda_total=200.0, tasa_impositiva=0.2,
               capex_ltm=30.0, da_ltm=40.0, nwc_operativo=-20.0, gasto_interes_ltm=10.0, dividendo_anual=3.0, beta=1.0,
               hist_fcf=[70.0, 78.0, 86.0, 93.0, 100.0], hist_ingresos=[600.0, 680.0, 760.0, 830.0, 900.0],
               hist_ebitda=[110.0, 120.0, 132.0, 146.0, 160.0], hist_eps=[3.5, 4.0, 4.7, 5.3, 6.0],
               multiplos_hist={"pe": [14.0, 15.0, 16.0, 17.0, 18.0], "ev_ebitda": [8.0, 9.0, 10.0, 11.0, 12.0],
                               "ev_revenue": [1.5, 1.7, 1.9, 2.0, 2.2], "pb": [3.0, 3.2, 3.5, 3.8, 4.0],
                               "p_fcf": [11.0, 12.0, 13.0, 14.0, 15.0]}, sector="Industrials", industria="Machinery")


def test_desactivar_ddm_y_otros_subconjuntos_cambia_el_valor_justo():
    e = _entradas_ricas()
    p = Par()
    ms = fv.calcular_modelos(e, p)
    assert all(m["aplicable"] for m in ms), [m["motivo_exclusion"] for m in ms if not m["aplicable"]]
    todos = [m["id"] for m in ms]
    base = fv.agregar_valor_justo(ms, todos, p.margen_seguridad, e.precio)["fair_value"]
    sin_ddm = fv.agregar_valor_justo(ms, [i for i in todos if i != "ddm"], p.margen_seguridad, e.precio)["fair_value"]
    solo_dcf = fv.agregar_valor_justo(ms, [i for i in todos if i.startswith("dcf_")], 0.25, e.precio)["fair_value"]
    solo_mult = fv.agregar_valor_justo(ms, ["pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf"], 0.25, e.precio)["fair_value"]
    assert len({round(v, 6) for v in (base, sin_ddm, solo_dcf, solo_mult)}) == 4
    # sin DDM: igualacion de familias a mano sobre los modelos usados (DCF / Multiplos / Valor+Dividendos)
    ag = fv.agregar_valor_justo(ms, [i for i in todos if i != "ddm"], p.margen_seguridad, e.precio)
    px_ = {m["id"]: m["precio_estimado"] for m in ms}
    grupos = {"DCF": [], "M": [], "V": []}
    for i in ag["modelos_usados"]:
        grupos["DCF" if i.startswith("dcf_") else "M" if i in ("pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf") else "V"].append(px_[i])
    manual = sum(sum(v) / len(v) for v in grupos.values() if v) / sum(1 for v in grupos.values() if v)
    assert sin_ddm == pytest.approx(manual, rel=1e-12) and "ddm" not in ag["modelos_usados"]


def test_wacc_mayor_baja_los_dcf_y_g_mayor_los_sube():
    e = _entradas_ricas()
    base = {m["id"]: m["precio_estimado"] for m in fv.calcular_modelos(e, Par(wacc=0.09, g=0.025))}
    wacc_alto = {m["id"]: m["precio_estimado"] for m in fv.calcular_modelos(e, Par(wacc=0.12, g=0.025))}
    g_alto = {m["id"]: m["precio_estimado"] for m in fv.calcular_modelos(e, Par(wacc=0.09, g=0.04))}
    for i in base:
        if i.startswith("dcf_") or i in ("epv", "ddm"):
            assert wacc_alto[i] < base[i], i
        if i.startswith("dcf_") or i == "ddm":
            assert g_alto[i] > base[i], i
    assert wacc_alto["pe"] == base["pe"] and wacc_alto["graham"] == base["graham"]


def test_valor_justo_atajo_usa_el_margen_de_los_parametros():
    e = _entradas_ricas()
    a = fv.valor_justo(e, Par(margen_seguridad=0.4))
    assert a["margin_of_safety_price"] == pytest.approx(a["fair_value"] * 0.6) and len(a["modelos"]) == 15
    assert a["precio_actual"] == 100.0


# ============================================================================== extraccion
def _facts_falsos(n_trim=28, **extra):
    """Tabla XBRL falsa con el esquema de ``valuation_floors._facts``: trimestres aislados, fechas relativas a hoy."""
    fin = (pd.Timestamp.now().normalize() - pd.Timedelta(days=50))
    cierres = [fin - pd.Timedelta(days=91 * i) for i in range(n_trim)][::-1]
    por_trim = {"us-gaap:Revenues": 250.0, "us-gaap:NetCashProvidedByUsedInOperatingActivities": 30.0,
                "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": 5.0, "us-gaap:OperatingIncomeLoss": 40.0,
                "us-gaap:DepreciationDepletionAndAmortization": 10.0, "us-gaap:EarningsPerShareDiluted": 1.0,
                "us-gaap:IncomeTaxExpenseBenefit": 5.0,
                "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": 20.0,
                "us-gaap:InterestExpense": 2.0, "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 10.0}
    por_trim.update(extra.pop("flujos", {}))
    instantes = {"us-gaap:CashAndCashEquivalentsAtCarryingValue": 300.0, "us-gaap:MarketableSecuritiesCurrent": 200.0,
                 "us-gaap:LongTermDebt": 400.0, "us-gaap:CommercialPaper": 100.0, "us-gaap:StockholdersEquity": 250.0,
                 "us-gaap:AssetsCurrent": 700.0, "us-gaap:LiabilitiesCurrent": 300.0}
    instantes.update(extra.pop("instantes", {}))
    filas = []
    for c, v in por_trim.items():
        if v is None:
            continue
        for d in cierres:
            filas.append(dict(concept=c, period_type="duration", period_start=d - pd.Timedelta(days=91), period_end=d,
                              numeric_value=v, fiscal_year=d.year, fiscal_period=f"Q{(d.month - 1) // 3 + 1}"))
    for c, v in instantes.items():
        if v is None:
            continue
        for d in cierres:
            filas.append(dict(concept=c, period_type="instant", period_start=pd.NaT, period_end=d, numeric_value=v,
                              fiscal_year=d.year, fiscal_period=f"Q{(d.month - 1) // 3 + 1}"))
    f = pd.DataFrame(filas)
    f["dur"] = (f["period_end"] - f["period_start"]).dt.days
    f["ref_filing"] = f.groupby(["fiscal_year", "fiscal_period"])["period_end"].transform("max")
    return f


def _df_precios(precio=100.0, dividendo=0.0):
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=2300)
    return pd.DataFrame({"Open": precio, "High": precio, "Low": precio, "Close": precio, "Volume": 1e6,
                         "Dividends": 0.0, "Stock Splits": 0.0}, index=idx).assign(
        Dividends=lambda d: np.where(np.arange(len(d)) % 63 == 0, dividendo, 0.0))


def test_extraer_entradas_desde_xbrl_con_cifras_conocidas():
    e = fv.extraer_entradas("T", None, _df_precios(), {"beta": 1.2, "sector": "Technology", "industry": "Software"},
                            facts=_facts_falsos())
    # LTM = 4 trimestres: ingresos 1000, OCF 120 - capex 20 = FCF 100, EBIT 160, D&A 40, EBITDA 200, EPS 4
    assert e.ingresos_ltm == 1000.0 and e.fcf_ltm == 100.0 and e.ebit_ltm == 160.0 and e.da_ltm == 40.0
    assert e.ebitda_ltm == 200.0 and e.eps_ltm == 4.0 and e.capex_ltm == 20.0 and e.gasto_interes_ltm == 8.0
    assert e.tasa_impositiva == pytest.approx(0.25)              # 20 / 80
    assert e.acciones_diluidas == 10.0 and e.bvps == pytest.approx(25.0)   # 250 / 10
    assert e.caja == 500.0 and e.deuda_total == 500.0            # 300 + 200 ; 400 + 100 (papel comercial)
    assert e.nwc_operativo == pytest.approx((700 - 500) - (300 - 100))   # (activo corr. - caja) - (pasivo corr. - deuda cp)
    assert e.beta == 1.2 and e.sector == "Technology" and e.precio == 100.0 and e.dividendo_anual == 0.0
    # 28 trimestres = 7 ejercicios LTM (modo local: hasta 10 anios + TTM sin recortes)
    assert len(e.hist_fcf) == 7 and set(e.hist_fcf) == {100.0} and set(e.hist_ingresos) == {1000.0}
    assert e.modo == "local" and e.anios_historia == 10 and len(e.hist_fechas) == 7 and len(e.serie_ttm) == 25
    assert e.fecha_ltm is not None and e.fuentes["fcf_ltm"].startswith("SEC EDGAR")
    json.dumps(e.to_dict())


def test_extraer_entradas_multiplos_historicos_propios():
    # precio constante 100 sin dividendos: P/E = 100/4 = 25 ; P/FCF = 100*10/100 = 10 ; EV/EBITDA = (1000 + 500 - 500)/200 = 5
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos())
    mh = e.multiplos_hist
    assert len(mh["pe"]) == 6 and all(v == pytest.approx(25.0) for v in mh["pe"])   # k = 1..6 anios atras
    assert all(v == pytest.approx(10.0) for v in mh["p_fcf"])
    assert all(v == pytest.approx(5.0) for v in mh["ev_ebitda"])
    assert all(v == pytest.approx(1.0) for v in mh["ev_revenue"])   # (1000 + 0) / 1000
    assert all(v == pytest.approx(1000.0 / 250.0) for v in mh["pb"])  # 100*10 / 250 = 4
    assert fv.multiplo_objetivo("pe", e, Par())["valor"] == pytest.approx(25.0)


def test_extraer_entradas_dato_faltante_es_none_y_se_completa_con_info():
    facts = _facts_falsos(flujos={"us-gaap:NetCashProvidedByUsedInOperatingActivities": None})
    e = fv.extraer_entradas("T", None, _df_precios(), {"freeCashflow": 77.0}, facts=facts)
    assert e.fcf_ltm == 77.0 and e.fuentes["fcf_ltm"] == "yfinance info" and e.hist_fcf == []
    sin = fv.extraer_entradas("T", None, _df_precios(), {}, facts=facts)
    assert sin.fcf_ltm is None and "FCF no disponible" in sin.avisos      # nunca 0 silencioso
    assert fv.calcular_modelos(sin)[0]["motivo_exclusion"] in ("FCF no disponible", "caja no disponible",
                                                                "deuda total no disponible", "acciones diluidas no disponibles")


def test_extraer_entradas_sin_edgar_cae_a_yfinance_sin_lanzar(monkeypatch):
    from src.levels import valuation_floors as vf

    def boom(t):
        raise ConnectionError("sin red")
    monkeypatch.setattr(vf, "_facts", boom)
    info = {"freeCashflow": 50.0, "ebitda": 90.0, "totalRevenue": 400.0, "trailingEps": 2.0, "bookValue": 8.0,
            "totalCash": 30.0, "totalDebt": 10.0, "sharesOutstanding": 5.0, "beta": 0.9, "trailingAnnualDividendRate": 1.0}
    e = fv.extraer_entradas("T", None, _df_precios(), info)
    assert any("EDGAR no disponible" in a for a in e.avisos)
    assert (e.fcf_ltm, e.ebitda_ltm, e.ingresos_ltm, e.eps_ltm, e.bvps, e.caja, e.deuda_total, e.acciones_diluidas) == \
        (50.0, 90.0, 400.0, 2.0, 8.0, 30.0, 10.0, 5.0)
    assert e.dividendo_anual == 1.0 and e.hist_fcf == [] and e.multiplos_hist == {}
    assert e.ebit_ltm is None and e.tasa_impositiva is None


def test_extraer_entradas_usa_facts_inyectados_y_no_pide_nada_a_yfinance(monkeypatch):
    from src.levels import valuation_floors as vf
    monkeypatch.setattr(vf, "_facts", lambda t: pytest.fail("no debe pedir EDGAR si se pasan los facts"))

    class Boom:
        def __getattr__(self, k):
            pytest.fail(f"no debe tocar el objeto de yfinance ({k})")
    e = fv.extraer_entradas("T", Boom(), _df_precios(), {}, facts=_facts_falsos())
    assert e.fcf_ltm == 100.0


def test_caja_xbrl_subcontada_se_reemplaza_por_info():
    facts = _facts_falsos(instantes={"us-gaap:MarketableSecuritiesCurrent": None})   # solo efectivo: 300
    e = fv.extraer_entradas("T", None, _df_precios(), {"totalCash": 900.0}, facts=facts)
    assert e.caja == 900.0 and any("caja" in a for a in e.avisos)
    ok = fv.extraer_entradas("T", None, _df_precios(), {"totalCash": 320.0}, facts=facts)
    assert ok.caja == 300.0 and not any("caja" in a for a in ok.avisos)
    medio = fv.extraer_entradas("T", None, _df_precios(), {"totalCash": 400.0}, facts=facts)   # 25 % de diferencia
    assert medio.caja == 300.0 and any("difiere 25%" in a for a in medio.avisos)       # avisa, no sobrescribe


def test_dividendo_anual_suma_los_ultimos_12_meses():
    df = _df_precios(dividendo=0.5)   # un pago de 0.5 cada 63 ruedas -> ~4 pagos por anio
    e = fv.extraer_entradas("T", None, df, {}, facts=_facts_falsos())
    pagos = df["Dividends"][df.index > df.index[-1] - pd.Timedelta(days=365)].sum()
    assert e.dividendo_anual == pytest.approx(pagos) and pagos >= 1.5


def test_precio_sin_ajuste_por_dividendos():
    # precios reales 100,100,100 y dividendo 1 en el ex-dividendo (dia 2): yfinance ajusta el pasado por 0.99
    idx = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame({"Close": [99.0, 99.0, 100.0], "Dividends": [0.0, 0.0, 1.0]}, index=idx)
    assert list(fv._precio_sin_ajuste_dividendos(df)) == pytest.approx([100.0, 100.0, 100.0])
    assert list(fv._precio_sin_ajuste_dividendos(df.drop(columns="Dividends"))) == [99.0, 99.0, 100.0]


def test_cagr_y_helpers():
    assert fv.cagr([100.0, 110.0, 121.0]) == pytest.approx(0.10)
    assert fv.cagr([100.0]) is None and fv.cagr([-1.0, 5.0]) is None and fv.cagr(None) is None
    assert fv.cagr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0]) == fv.cagr([2.0, 3.0, 4.0, 5.0, 6.0, 1000.0])  # ultimos 6 puntos


# ============================================================================== main.analizar_ticker
def _simular_pipeline(monkeypatch, df):
    import main as m
    import src.data.market_data as md
    import src.levels.valuation_floors as vf
    import src.validation.backtest_ranges as br
    monkeypatch.setattr(md, "descargar_historial", lambda t, p: (object(), df))
    monkeypatch.setattr(md, "obtener_info", lambda tk: {"sector": "Technology"})
    monkeypatch.setattr(md, "verificar_proximidad_earnings", lambda tk: {"alerta_activa": False, "mensaje": "ok"})
    monkeypatch.setattr(br, "calcular_zonas_pit", lambda d: {"atr_semanal": 1.0, "clusters": [], "primario": None})
    monkeypatch.setattr(vf, "calcular_pisos_valoracion", lambda *a, **k: {"anclas": [], "pe": None})
    return m


def test_analizar_ticker_guarda_entradas_livianas_para_la_ui(monkeypatch, df_sintetico):
    m = _simular_pipeline(monkeypatch, df_sintetico)
    llamadas = []

    def falso(t, tk, df, info, facts=None):
        llamadas.append((t, info))
        return fv.EntradasValoracion(ticker=t, precio=float(df["Close"].iloc[-1]), fcf_ltm=1.0)
    monkeypatch.setattr(fv, "extraer_entradas", falso)
    res = m.analizar_ticker("AAA", 1000, backtest=False, sec=False, verbose=False)
    ev = res["entradas_valoracion"]
    assert isinstance(ev, dict) and ev["ticker"] == "AAA" and ev["fcf_ltm"] == 1.0
    assert llamadas == [("AAA", {"sector": "Technology"})]          # se extrae UNA sola vez, con el info ya descargado
    json.dumps(ev)                                                  # liviano: sin DataFrames
    assert not any(isinstance(v, pd.DataFrame) for v in ev.values())
    assert res["errores"] == [] and res["zonas"] == []              # aditivo: no toca zonas ni scoring


def test_analizar_ticker_sin_valor_justo_o_con_fallo_no_rompe_el_resto(monkeypatch, df_sintetico):
    m = _simular_pipeline(monkeypatch, df_sintetico)
    monkeypatch.setattr(fv, "extraer_entradas", lambda *a, **k: pytest.fail("valor_justo=False no debe extraer"))
    res = m.analizar_ticker("AAA", 1000, backtest=False, sec=False, verbose=False, valor_justo=False)
    assert "entradas_valoracion" not in res

    def boom(*a, **k):
        raise RuntimeError("edgar")
    monkeypatch.setattr(fv, "extraer_entradas", boom)
    res = m.analizar_ticker("AAA", 1000, backtest=False, sec=False, verbose=False)
    assert "entradas_valoracion" not in res and any(e.startswith("valor_justo:") for e in res["errores"])
    assert res["precio_actual"] is not None


def test_exporter_serializa_las_entradas(tmp_path, monkeypatch, df_sintetico):
    from src.reporting.exporter import exportar_resultados
    m = _simular_pipeline(monkeypatch, df_sintetico)
    monkeypatch.setattr(fv, "extraer_entradas", lambda t, *a, **k: fv.EntradasValoracion(ticker=t, fcf_ltm=1.0))
    res = m.analizar_ticker("AAA", 1000, backtest=False, sec=False, verbose=False)
    rutas = exportar_resultados([m._liviano(res)], "vj", str(tmp_path), "json")
    with open(rutas["json"], encoding="utf-8") as f:
        datos = json.load(f)
    assert datos["resultados"][0]["entradas_valoracion"]["fcf_ltm"] == 1.0


def test_import_de_app_ui_y_fair_value_no_usa_red_ni_servidor():
    codigo = ("import socket\n"
              "def _no(*a, **k): raise AssertionError('red prohibida al importar')\n"
              "socket.socket.connect = _no; socket.create_connection = _no\n"
              "import app_ui, src.levels.fair_value\n"
              "from nicegui import core\n"
              "assert not getattr(core.app, 'is_started', False)\n"
              "print('ok')")
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120,
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


# ============================================================================== UI: funciones puras
def test_etiqueta_fair_value():
    assert ui_mod.etiqueta_fair_value(123.456, 12.3) == "Fair Value: $123.46 (+12% Upside)"
    assert ui_mod.etiqueta_fair_value(50.0, -20.4) == "Fair Value: $50.00 (-20% Upside)"
    assert ui_mod.etiqueta_fair_value(50.0, None) == "Fair Value: $50.00"


def test_estado_modelo_prioridad_y_textos():
    agg = {"modelos_outlier": ["pe"]}
    base = {"id": "pe", "aplicable": True, "motivo_exclusion": None}
    assert ui_mod.estado_modelo(base, agg, True) == ("outlier", "Outlier: excluido del promedio (filtro IQR)")
    assert ui_mod.estado_modelo(base, agg, False) == ("desactivado", "Desactivado por usuario")
    assert ui_mod.estado_modelo({**base, "id": "x"}, agg, True) == ("activo", "Activo")
    na = {"id": "pe", "aplicable": False, "motivo_exclusion": "EPS LTM negativo o cero"}
    assert ui_mod.estado_modelo(na, agg, False) == ("no_aplica", "No aplica: EPS LTM negativo o cero")


def _esc_rico(activos=None, e=None, params=None, px_=100.0):
    e = e or _entradas_ricas()
    ids = None if activos is None else [i for i, v in activos.items() if v]
    return fv.calcular_escenarios(e, params, ids, None, px_)


def test_opciones_barras_dos_series_lineas_y_estados():
    e = _entradas_ricas()
    e.eps_ltm = -1.0   # P/E, Graham y Lynch no aplican
    activos = {i: i != "ddm" for i in fv.IDS_MODELOS}
    esc = _esc_rico(activos, e)
    opc = ui_mod.opciones_barras(esc, 100.0, activos)
    medio, pes = opc["series"]
    assert [s_["name"] for s_ in opc["series"]] == ["Escenario Medio", "Escenario Pesimista"]
    assert all(s_["type"] == "bar" and len(s_["data"]) == 15 for s_ in opc["series"]) and opc["yAxis"]["inverse"] is True
    nombres = opc["yAxis"]["data"]
    assert len(nombres) == 15 and "P/E · no aplica" in nombres and "DDM (dividendos) · desactivado" in nombres
    i_pe, i_ddm, i_dcf = nombres.index("P/E · no aplica"), nombres.index("DDM (dividendos) · desactivado"), 0
    assert medio["data"][i_pe]["value"] is None and pes["data"][i_pe]["value"] is None            # no aplica: sin barra
    d = medio["data"][i_ddm]
    assert d["itemStyle"]["color"] == ui_mod.VJ_INACTIVO and d["itemStyle"]["opacity"] < 1           # desactivado: gris y atenuado
    assert medio["data"][i_dcf]["itemStyle"]["color"] in (ui_mod.VJ_ACTIVO, ui_mod.AVISO_TXT)
    assert pes["data"][i_dcf]["itemStyle"]["color"] in (ui_mod.VJ_PESIMISTA, ui_mod.AVISO_TXT)
    lineas = medio["markLine"]["data"]
    assert [round(l["xAxis"], 2) for l in lineas] == [round(esc["medio"]["fair_value"], 2),
                                                      round(esc["pesimista"]["fair_value"], 2), 100.0]
    assert [l["lineStyle"]["color"] for l in lineas] == [ui_mod.COLOR_FAIR_VALUE, ui_mod.VJ_PESIMISTA, ui_mod.COLOR_PRECIO]
    assert "Valor justo Medio $" in lineas[0]["label"]["formatter"] and "Piso Pesimista $" in lineas[1]["label"]["formatter"]
    vacia = ui_mod.opciones_barras(None, None, None, "Sin análisis")
    assert vacia["title"]["text"] == "Sin análisis" and all(s_["data"] == [] for s_ in vacia["series"])
    assert ui_mod.alto_barras(15) >= 420


def test_opciones_barras_marca_outliers_topes_y_fuera_de_escala():
    base = _esc_rico()
    esc = {**base, "modelos": [dict(m) for m in base["modelos"]]}
    esc["medio"] = {**base["medio"], "modelos_outlier": ["pb"]}
    esc["modelos"][9]["precio_medio"] = 100000.0          # pb: fuera de escala
    esc["modelos"][6]["topes_medio"] = ["P/E topado a 35x"]
    opc = ui_mod.opciones_barras(esc, 100.0, None)
    medio = opc["series"][0]["data"]
    pb, pe = medio[9], medio[6]
    assert pb["itemStyle"]["color"] == ui_mod.AVISO_TXT and "outlier" in pb["label"]["formatter"]
    assert "▶" in pb["label"]["formatter"] and pb["value"] <= opc["xAxis"]["max"]    # no deforma la escala: se recorta con flecha
    assert opc["xAxis"]["max"] < 10000
    assert pe["label"]["formatter"].endswith("· tope")


def _overlays(df, zonas=None, px=None, **kw):
    datos = ui_mod.datos_grafico(df, "T")
    px = px if px is not None else float(df["Close"].iloc[-1])
    return ui_mod.overlays_grafico(datos, "T", zonas or [], px, **kw)


def test_figura_analisis_linea_de_valor_justo(df_sintetico):
    px_ = float(df_sintetico["Close"].iloc[-1])
    assert _overlays(df_sintetico)["vj"] is None                      # sin valor justo: sin linea
    vj = {"fair_value": px_ * 1.3, "upside_pct": 30.0}
    linea = _overlays(df_sintetico, valor_justo=vj)["vj"]
    assert re.fullmatch(r"Fair Value Medio: \$[\d,]+\.\d\d \(\+30% Upside\)", linea["etiqueta"])
    assert linea["precio"] == pytest.approx(px_ * 1.3, abs=1e-3) and linea["fuera_de_rango"] is None
    bajista = _overlays(df_sintetico, valor_justo={"fair_value": px_ * 0.7, "upside_pct": -30.0})["vj"]
    assert "(-30% Upside)" in bajista["etiqueta"]


def test_badges_de_incertidumbre_cumplen_contraste():
    css = ui_mod.css_tema()
    from test_app_ui import contraste
    for clase in ("vj-baja", "vj-media", "vj-alta"):
        m = re.search(r"\." + clase + r"\{background:(#[0-9a-fA-F]{6});color:(#[0-9a-fA-F]{6})\}", css)
        assert m, clase
        assert contraste(m.group(2), m.group(1)) >= 4.5, clase


def test_colores_nuevos_del_valor_justo_son_legibles():
    from test_app_ui import contraste
    m = ui_mod
    assert contraste(m.COLOR_FAIR_VALUE, m.BG_ETIQUETA, base=m.BG_PANEL) >= 4.5
    for barra in (m.VJ_ACTIVO, m.VJ_INACTIVO, m.AVISO_TXT):   # barras: no texto, minimo 3:1 sobre el panel
        assert contraste(barra, m.BG_PANEL) >= 3.0, barra
    assert contraste(m.COLOR_FAIR_VALUE, m.BG_PANEL) >= 4.5
    for otro in (m.COLOR_VALORACION, m.COLOR_PRECIO, *m.COLORES_NIVEL.values()):
        assert m.COLOR_FAIR_VALUE != otro


def test_resumen_y_ranking_con_upside():
    res = {"ticker": "A", "precio_actual": 100.0, "zonas": [], "entradas_valoracion": _entradas_ricas().to_dict()}
    r = ui_mod.resumen_valor_justo(res)
    assert set(r) == {"fair_value", "upside_pct", "incertidumbre", "n_usados", "cv"} and r["n_usados"] >= 3
    assert ui_mod.resumen_valor_justo({"precio_actual": 1.0}) is None
    filas, _ = ui_mod.filas_ranking([{**res, "valor_justo": r}, {"ticker": "B", "precio_actual": 5.0, "zonas": []}])
    assert filas[0]["upside"] == pytest.approx(r["upside_pct"]) and filas[1]["upside"] is None


def test_calcular_ticker_solo_pasa_valor_justo_si_se_desactiva(monkeypatch):
    vistos = []
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setattr(ui_mod.nucleo, "analizar_ticker", lambda t, c, **kw: vistos.append(kw) or {"ticker": t})
    asyncio.run(ui_mod.calcular_ticker("A", 1.0))
    asyncio.run(ui_mod.calcular_ticker("A", 1.0, valor_justo=False))
    assert "valor_justo" not in vistos[0] and vistos[1]["valor_justo"] is False


# ============================================================================== UI: Vista (sin red)
@pytest.fixture
def cliente():
    from nicegui.client import Client
    from nicegui.page import page
    with Client(page("/")) as c:
        yield c


@pytest.fixture
def vista(cliente, df_sintetico, monkeypatch):
    """Vista construida con un resultado en CACHE que trae entradas de valoracion. EDGAR y yfinance estan prohibidos."""
    from src.levels import valuation_floors as vf
    monkeypatch.setattr(vf, "_facts", lambda t: pytest.fail("la UI no debe consultar EDGAR"))
    monkeypatch.setattr("yfinance.Ticker", lambda *a, **k: pytest.fail("la UI no debe consultar yfinance"))
    res = {"ticker": "RICA", "precio_actual": 100.0, "df": df_sintetico, "zonas": [], "errores": [], "form4": None,
           "r13f": None, "confluencia_score": 60.0, "confiabilidad": {"indice": 40.0, "cobertura": 0.8},
           "backtest_resumen": {}, "entradas_valoracion": _entradas_ricas().to_dict()}
    monkeypatch.setitem(ui_mod.CACHE, "RICA", res)
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "RICA"
    v.refrescar()
    return v


def _anot_fair(v):
    vj = (v.grafico_overlays or {}).get("vj")      # etiqueta de la linea del Escenario Medio en el grafico de Analisis
    return [vj["etiqueta"]] if vj else []


def _clic(cliente, accion):
    """Dispara el handler de un boton por su data-accion (sin servidor)."""
    el = next(e for e in cliente.elements.values() if e.props.get("data-accion") == accion)
    from nicegui import events
    for ls in el._event_listeners.values():
        events.handle_event(ls.handler, events.ClickEventArguments(sender=el, client=cliente))


def test_vista_tab_valor_justo_y_controles(cliente, vista):
    from nicegui import ui
    els = list(cliente.elements.values())
    assert [e.props["label"] for e in els if isinstance(e, ui.tab)] == ["Análisis", "Valor Justo", "Watchlist", "Modelo"]
    assert len(vista.vj_rows) == 15 and set(vista.vj_rows) == set(fv.IDS_MODELOS)
    assert len([e for e in els if isinstance(e, ui.checkbox) and e.text in
                {m[1] for m in fv.MODELOS}]) == 15
    assert {e.props.get("data-accion") for e in els} >= {"vj-todos", "vj-dcf", "vj-multiplos", "vj-reset"}
    assert [e.props["data-slider"] for e in els if isinstance(e, ui.slider)] == ["wacc", "g", "mos"]
    sw = next(e for e in els if isinstance(e, ui.switch) and e.text == "Mostrar Valor Justo")
    assert sw.value is True          # activado por defecto: la linea del Escenario Medio se ve sin tocar nada
    assert "propietaria" in " ".join(str(getattr(e, "text", "")) for e in els if isinstance(e, ui.label))


def test_vista_kpis_coinciden_con_el_calculo(vista):
    """Cuatro tarjetas: precio, valor justo Medio, piso Pesimista y diagnostico (con incertidumbre)."""
    esc, agg = vista.vj_esc, vista.vj_agg
    assert agg is esc["medio"] and agg["fair_value"] is not None and esc["pesimista"]["fair_value"] is not None
    assert set(vista.vj_kpi) == {"precio", "fair", "piso", "diag", "incert"}
    assert vista.vj_kpi["precio"].text == "$100.00"
    assert vista.vj_kpi["fair"].text == f"${agg['fair_value']:,.2f}"
    assert vista.vj_kpi["piso"].text == f"${esc['pesimista']['fair_value']:,.2f}"
    assert esc["pesimista"]["fair_value"] < agg["fair_value"]                       # el Pesimista es un piso, no un techo
    assert agg["margin_of_safety_price"] == pytest.approx(agg["fair_value"] * 0.75)
    assert f"${agg['margin_of_safety_price']:,.2f}" in vista.vj_kpi_sub["fair"].text   # precio con margen: subtexto
    assert "estrés duro" in vista.vj_kpi_sub["piso"].text
    dx = ui_mod.diagnostico_valoracion(100.0, agg["fair_value"], agg["incertidumbre"], agg["n_usados"])
    assert vista.vj_kpi["diag"].text.startswith(dx["texto"]) and dx["clave"] in ("sobre", "infra", "linea")
    clase = "vj-dx-debil" if dx["debil"] else f"vj-dx-{dx['clave']}"
    assert clase in vista.vj_kpi["diag"]._classes and "vj-badge" in vista.vj_kpi["diag"]._classes
    assert vista.vj_kpi["incert"].text == f"Incertidumbre {agg['incertidumbre']}"
    clase = {"Baja": "vj-baja", "Media": "vj-media", "Alta": "vj-alta"}[agg["incertidumbre"]]
    assert clase in vista.vj_kpi["incert"]._classes and "vj-badge" in vista.vj_kpi["incert"]._classes


def test_diagnostico_de_valoracion_umbrales_con_nombre_y_soporte_debil():
    f = ui_mod.diagnostico_valoracion
    assert ui_mod.UMBRAL_DIAGNOSTICO == 0.10
    assert f(111.0, 100.0)["clave"] == "sobre" and f(111.0, 100.0)["texto"] == "Sobrevalorada"
    assert f(110.0, 100.0)["clave"] == "sobre"                      # el borde (+10 %) ya es sobrevaloracion
    assert f(109.0, 100.0)["clave"] == "linea" and f(109.0, 100.0)["texto"] == "En línea"
    assert f(91.0, 100.0)["clave"] == "linea" and f(90.0, 100.0)["clave"] == "infra"
    assert f(70.0, 100.0)["texto"] == "Infravalorada"
    assert f(111.0, 100.0)["premium_pct"] == pytest.approx(11.0)
    assert f(None, 100.0)["clave"] is None and f(100.0, None)["texto"] == "n/d" and f(100.0, -5.0)["clave"] is None
    assert f(150.0, 100.0, "Alta", 9)["debil"] == "incert. Alta" and f(150.0, 100.0, "Media", 2)["debil"] == "n=2"
    assert f(150.0, 100.0, "Baja", 9)["debil"] is None


def test_diagnostico_debil_atenua_el_badge(vista):
    vista.vj_activos = {i: i in ("pe", "pb") for i in fv.IDS_MODELOS}       # solo 2 modelos: n < 3 => soporte debil
    vista._vj_aplicar()
    assert vista.vj_agg["n_usados"] == 2
    assert "vj-dx-debil" in vista.vj_kpi["diag"]._classes and "n=2" in vista.vj_kpi["diag"].text


def test_vista_casillas_estado_y_no_aplicables_deshabilitadas(cliente, df_sintetico, monkeypatch):
    e = _entradas_ricas()
    e.eps_ltm = -1.0     # P/E, Graham, Lynch no aplican
    e.dividendo_anual = 0.0
    monkeypatch.setitem(ui_mod.CACHE, "NA", {"ticker": "NA", "precio_actual": 100.0, "df": df_sintetico, "zonas": [],
                                             "errores": [], "entradas_valoracion": e.to_dict()})
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "NA"
    v.refrescar()
    for i in ("pe", "graham", "lynch", "ddm"):
        fila = v.vj_rows[i]
        assert fila["chk"].enabled is False and fila["chk"].value is False
        assert fila["estado"].text.startswith("No aplica: ") and fila["precio"].text == "n/d"
    for i in ("dcf_5y_fcf", "epv", "pb"):
        assert v.vj_rows[i]["chk"].enabled is True and v.vj_rows[i]["chk"].value is True
        assert v.vj_rows[i]["estado"].text in ("Activo", "Outlier: excluido del promedio (filtro IQR)")
    assert set(v.vj_agg["modelos_no_aplicables"]) == {"pe", "graham", "lynch", "ddm"}


def test_vista_casilla_recalcula_kpis_grafico_y_linea(vista):
    vista.sw_vj.value = True    # dispara _toggle_vj
    antes = vista.vj_agg["fair_value"]
    assert _anot_fair(vista) and f"${antes:,.2f}" in _anot_fair(vista)[0]
    vista.vj_rows["dcf_5y_fcf"]["chk"].value = False
    despues = vista.vj_agg["fair_value"]
    assert despues != antes and vista.vj_activos["dcf_5y_fcf"] is False
    assert vista.vj_rows["dcf_5y_fcf"]["estado"].text == "Desactivado por usuario"
    assert "dcf_5y_fcf" in vista.vj_agg["modelos_desactivados"]
    assert vista.vj_kpi["fair"].text == f"${despues:,.2f}"
    assert f"${despues:,.2f}" in _anot_fair(vista)[0]                       # la linea de Analisis sigue a los modelos activos
    assert vista.vj_barras.options["yAxis"]["data"][0].endswith("desactivado")      # dcf_5y_fcf es el primer modelo
    vista.vj_rows["dcf_5y_fcf"]["chk"].value = True
    assert vista.vj_agg["fair_value"] == pytest.approx(antes)


def test_vista_desactivar_ddm_cambia_el_valor_justo(vista):
    antes = vista.vj_agg["fair_value"]
    vista.vj_rows["ddm"]["chk"].value = False
    assert vista.vj_agg["fair_value"] != antes
    assert "ddm" not in vista.vj_agg["modelos_usados"]


def test_vista_botones_todos_solo_dcf_solo_multiplos(cliente, vista):
    _clic(cliente, "vj-dcf")
    assert [i for i, v in vista.vj_activos.items() if v] == ui_mod.ids_por_familia("DCF") and len(ui_mod.ids_por_familia("DCF")) == 6
    assert all(vista.vj_rows[i]["chk"].value is (i.startswith("dcf_")) for i in fv.IDS_MODELOS)
    assert set(vista.vj_agg["modelos_usados"]) <= set(ui_mod.ids_por_familia("DCF"))
    fv_dcf = vista.vj_agg["fair_value"]
    _clic(cliente, "vj-multiplos")
    assert [i for i, v in vista.vj_activos.items() if v] == ["pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf"]
    assert vista.vj_agg["fair_value"] != fv_dcf
    assert all(vista.vj_rows[i]["estado"].text == "Desactivado por usuario" for i in ui_mod.ids_por_familia("DCF"))
    _clic(cliente, "vj-todos")
    assert all(vista.vj_activos.values()) and all(vista.vj_rows[i]["chk"].value for i in fv.IDS_MODELOS)


def test_vista_sliders_recalculan_y_restablecer_vuelve_al_automatico(cliente, vista):
    auto_w = vista.vj_sl["wacc"].value
    assert vista.vj_params.wacc is None and "automático" in vista.vj_sl_lbl["wacc"].text
    base = {m["id"]: m["precio_estimado"] for m in vista.vj_agg["modelos"]}
    fair0 = vista.vj_agg["fair_value"]
    vista.vj_sl["wacc"].value = 12.0
    assert vista.vj_params.wacc == pytest.approx(0.12) and "automático" not in vista.vj_sl_lbl["wacc"].text
    nuevo = {m["id"]: m["precio_estimado"] for m in vista.vj_agg["modelos"]}
    assert nuevo["dcf_5y_fcf"] < base["dcf_5y_fcf"] and nuevo["epv"] < base["epv"] and nuevo["pe"] == base["pe"]
    assert vista.vj_agg["fair_value"] != fair0
    vista.vj_sl["g"].value = 4.0
    assert vista.vj_params.g == pytest.approx(0.04) and \
        {m["id"]: m["precio_estimado"] for m in vista.vj_agg["modelos"]}["dcf_5y_fcf"] > nuevo["dcf_5y_fcf"]
    vista.vj_sl["mos"].value = 40
    assert vista.vj_params.margen_seguridad == pytest.approx(0.4)
    assert vista.vj_agg["margin_of_safety_price"] == pytest.approx(vista.vj_agg["fair_value"] * 0.6)
    assert f"${vista.vj_agg['margin_of_safety_price']:,.2f}" in vista.vj_kpi_sub["fair"].text
    _clic(cliente, "vj-reset")
    assert vista.vj_params == fv.ParametrosValoracion() and vista.vj_sl["wacc"].value == auto_w
    assert vista.vj_sl["g"].value == round(fv.CRECIMIENTO_TERMINAL_DEFECTO * 100, 1)
    assert vista.vj_sl["mos"].value == round(fv.MARGEN_SEGURIDAD_DEFECTO * 100)
    assert vista.vj_agg["fair_value"] == pytest.approx(fair0)


def test_vista_el_slider_de_g_se_limita_a_wacc_menos_dos_puntos_y_medio(vista):
    vista.vj_sl["wacc"].value = 5.0
    vista.vj_sl["g"].value = 5.0
    assert vista.vj_sl["g"].value == pytest.approx(2.5) and vista.vj_params.g == pytest.approx(0.025)
    assert float(vista.vj_sl["g"].props["max"]) == pytest.approx(2.5)
    assert next(m for m in vista.vj_agg["modelos"] if m["id"] == "dcf_5y_fcf")["aplicable"]   # 5 % - 2.5 % = 2.5 pp
    vista.vj_sl["wacc"].value = 12.0
    assert float(vista.vj_sl["g"].props["max"]) == pytest.approx(5.0)


def test_vista_aviso_de_diferencia_wacc_g_si_los_parametros_lo_violan(vista):
    vista.vj_params = vista.vj_params.con(wacc=0.05, g=0.0499)    # no alcanzable con los sliders, si por otras vias
    vista.vj_modelos_cache = None
    vista._vj_actualizar()
    assert not next(m for m in vista.vj_agg["modelos"] if m["id"] == "dcf_5y_fcf")["aplicable"]
    assert "diferencia WACC-g" in vista.vj_rows["dcf_5y_fcf"]["estado"].text


def test_vista_slider_con_debounce_recalcula_una_sola_vez(vista, monkeypatch):
    monkeypatch.setattr(ui_mod, "VJ_DEBOUNCE_S", 0.02)
    llamadas = []
    orig = vista._vj_actualizar
    monkeypatch.setattr(vista, "_vj_actualizar", lambda: llamadas.append(1) or orig())

    async def rafaga():
        for w in (10.0, 10.5, 11.0, 11.5, 12.0):   # rafaga de cambios: solo el ultimo debe recalcular
            vista.vj_sl["wacc"].value = w
        assert llamadas == []                     # nada recalculado todavia (debounce)
        assert vista.vj_params.wacc == pytest.approx(0.12)   # pero los parametros ya estan al dia
        await asyncio.sleep(0.15)
    asyncio.run(rafaga())
    assert llamadas == [1]


def test_vista_toggle_mostrar_valor_justo_en_analisis(vista):
    assert _anot_fair(vista)                              # activado por defecto: la linea ya esta
    vista.sw_vj.value = False
    assert not _anot_fair(vista)
    vista.sw_vj.value = True
    txt = _anot_fair(vista)
    assert len(txt) == 1 and re.fullmatch(r"Fair Value Medio: \$[\d,]+\.\d\d \([+-]\d+% Upside\)( · .+)?", txt[0])
    assert f"{vista.vj_agg['upside_pct']:+.0f}% Upside" in txt[0]
    vista.vj_sl["wacc"].value = 13.0                      # la linea sigue a los sliders
    assert f"${vista.vj_agg['fair_value']:,.2f}" in _anot_fair(vista)[0]
    vista.sw_vj.value = False
    assert not _anot_fair(vista)
    assert vista.vj_kpi["fair"].text != "n/d"             # la pestana sigue calculando


def test_vista_sin_analisis_pide_recalcular(cliente):
    v = ui_mod.Vista()
    v.construir()
    assert "Pulsa Recalcular" in v.vj_msg.text and v.vj_kpi["fair"].text == "n/d" and v.vj_agg is None
    assert all(f["chk"].enabled is False for f in v.vj_rows.values())
    v.sw_vj.value = True
    v.refrescar(solo_figura=True)                          # sin resultado: no lanza ni dibuja la linea


def test_vista_resultado_sin_entradas_mensaje_claro(cliente, df_sintetico, monkeypatch):
    monkeypatch.setitem(ui_mod.CACHE, "SIN", {"ticker": "SIN", "precio_actual": 10.0, "df": df_sintetico, "zonas": [],
                                              "errores": ["valor_justo: RuntimeError: x"]})
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "SIN"
    v.refrescar()
    assert "no trae entradas" in v.vj_msg.text and v.vj_kpi["fair"].text == "n/d"


def test_vista_cambiar_de_ticker_recarga_entradas_y_reinicia_wacc(vista, df_sintetico, monkeypatch):
    vista.vj_sl["wacc"].value = 12.0
    otra = _entradas_ricas()
    otra.beta, otra.ticker = 0.5, "OTRA"
    monkeypatch.setitem(ui_mod.CACHE, "OTRA", {"ticker": "OTRA", "precio_actual": 80.0, "df": df_sintetico, "zonas": [],
                                               "errores": [], "entradas_valoracion": otra.to_dict()})
    vista.ticker = "OTRA"
    vista.refrescar()
    assert vista.vj_params.wacc is None and vista.vj_entradas.ticker == "OTRA"
    assert vista.vj_kpi["precio"].text == "$80.00"
    assert vista.vj_agg["precio_actual"] == 80.0


def test_vista_no_modifica_zonas_ni_tabla(vista, df_sintetico, monkeypatch):
    zonas = [{"nivel": 1, "r_inf": 90.0, "r_sup": 95.0, "p_centro": 92.5, "dist_pct": -5.0, "n_familias": 2, "n_anclas": 2,
              "familias": ["volumen", "ob"], "tipos": ["POC", "OB"], "fuente": "POC+OB"}]
    ui_mod.CACHE["RICA"]["zonas"] = zonas
    vista.refrescar()
    filas = [dict(f) for f in vista.tabla.rows]
    vista.sw_vj.value = True
    vista.vj_sl["wacc"].value = 9.0
    vista.vj_rows["pe"]["chk"].value = False
    vista.refrescar()
    assert vista.tabla.rows == filas and ui_mod.CACHE["RICA"]["zonas"] == zonas


def test_vista_watchlist_tiene_columna_upside_opcional(cliente):
    v = ui_mod.Vista()
    v.construir()
    assert "upside" in [c["name"] for c in v.tabla_wl.columns]
    assert v.sw_vj_wl.value is True          # local: activo; en la nube arranca apagado (ver test de la nube)


def test_vista_en_la_nube_el_escaneo_no_calcula_valor_justo(cliente, monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    v = ui_mod.Vista()
    v.construir()
    assert v.sw_vj_wl.value is False


def test_escaneo_local_agrega_upside_desde_las_entradas(cliente, monkeypatch):
    from src.reporting import userdata
    userdata.guardar(["AAA"])
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(None))
    pedidos = []

    async def falso(t, cap, bt, sec, vj=True):
        pedidos.append(vj)
        return {"ticker": t, "precio_actual": 100.0, "zonas": [], "errores": [], "confluencia_score": 50.0,
                "confiabilidad": {"indice": 50.0, "cobertura": 1.0}, "entradas_valoracion": _entradas_ricas().to_dict()}
    monkeypatch.setattr(ui_mod, "calcular_ticker", falso)
    v = ui_mod.Vista()
    v.construir()

    async def correr():
        with cliente:
            await v.escanear()
    asyncio.run(correr())
    assert pedidos == [True]
    assert v.tabla_wl.rows[0]["upside"] is not None and v.tabla_wl.rows[0]["upside"] == pytest.approx(
        ui_mod.resumen_valor_justo({"precio_actual": 100.0, "entradas_valoracion": _entradas_ricas().to_dict()})["upside_pct"])


# ============================================================================== pestana Modelo
def test_modelo_seccion_valor_justo_existe_y_es_honesta():
    secs = ui_mod.contenido_modelo()
    sec = next(s for s in secs if s["id"] == "valor_justo")
    assert secs[-1] is sec or secs[-1]["id"] == "valor_justo"
    txt = ui_mod.texto_seccion(sec)
    for clave in ("InvestingPro", "propietaria", "no replica", "DCF", "Múltiplos", "Dividendos", "margen de seguridad",
                  "incertidumbre", "coeficiente de variación", "Outlier", "niveles técnicos", "Sensibilidad",
                  "valor terminal", "LTM", "bancos", "no es una predicción", "WACC", "no mueve las zonas"):
        assert clave.lower() in txt.lower(), clave


def test_modelo_seccion_valor_justo_lee_las_constantes_del_codigo(monkeypatch):
    def texto():
        return ui_mod.texto_seccion(next(s for s in ui_mod.contenido_modelo() if s["id"] == "valor_justo"))
    base = texto()
    for cifra in ("0,15", "0,35", "1,5", "42 %", "15 %", "55 %", "30 %", "3,5 %", "2,5 %", "12,5 %", "2026-11"):
        assert cifra in base, cifra
    monkeypatch.setattr(fv, "CV_BAJO", 0.22)
    monkeypatch.setattr(fv, "CV_ALTO", 0.41)
    monkeypatch.setattr(fv, "IQR_K", 2.5)
    monkeypatch.setattr(fv, "MIN_MODELOS_CONFIABLE", 5)
    monkeypatch.setattr(fv, "MIN_MODELOS_IQR", 7)
    monkeypatch.setattr(fv, "TOPE_CRECIMIENTO_ETAPA1", {"medio": 0.37, "pesimista": 0.11})
    monkeypatch.setattr(fv, "TOPE_MARGEN", {"medio": 0.61, "pesimista": 0.27})
    monkeypatch.setattr(fv, "G_TERMINAL", {"medio": 0.033, "pesimista": 0.021})
    monkeypatch.setattr(fv, "WACC_PESIMISTA_MIN", 0.133)
    monkeypatch.setattr(fv, "BENCHMARK_FECHA", "2030-01")
    nuevo = texto()
    assert "0,22" in nuevo and "0,41" in nuevo and "2,5" in nuevo and "menos de 5 modelos" in nuevo
    assert "con 7 o más" in nuevo and "37 %" in nuevo and "11 %" in nuevo and "61 %" in nuevo and "27 %" in nuevo
    assert "13,3 %" in nuevo and "2030-01" in nuevo and "3,3 %" in nuevo and "2,1 %" in nuevo
    assert nuevo != base


def test_modelo_seccion_valor_justo_sin_textos_obsoletos_y_con_el_motor_nuevo():
    txt = ui_mod.texto_seccion(next(s for s in ui_mod.contenido_modelo() if s["id"] == "valor_justo")).lower()
    for obsoleto in ("haircut", "dcf a 5 años", "tope 15 %", "descuentan a fin de año", "optimista, "):
        assert obsoleto not in txt, obsoleto
    for clave in ("dos escenarios", "medio", "pesimista", "no hay optimista", "^tnx", "blume", "mitad de año", "10 años",
                  "2 etapas", "mediana recortada", "aproximación fijada a mano", "iguala el peso de cada familia", "iqr",
                  "causal", "ancla", "no son point-in-time", "ilustrativa", "hipercrecimiento", "nvda", "estrés duro",
                  "supuestos, no predicciones", "no un «peor caso realista»", "ejemplo"):
        assert clave in txt, clave


def test_ejemplo_numerico_del_modelo_sale_del_codigo_real():
    ej = ui_mod._ejemplo_valor_justo()
    esc = fv.calcular_escenarios(ej["entradas"], ej["params"], ["dcf_10y_fcf", "pe", "graham"])
    assert ej["agg"]["fair_value"] == pytest.approx(esc["medio"]["fair_value"])
    assert ej["agg_pes"]["fair_value"] == pytest.approx(esc["pesimista"]["fair_value"])
    assert ej["agg_pes"]["fair_value"] < ej["agg"]["fair_value"]
    media = (ej["dcf"] + ej["pe"] + ej["graham"]) / 3
    assert ej["agg"]["fair_value"] == pytest.approx(media)          # una por familia: el promedio iguala familias
    txt = ui_mod.texto_seccion(next(s for s in ui_mod.contenido_modelo() if s["id"] == "valor_justo"))
    assert ui_mod._es(ej["agg"]["fair_value"], 1) in txt and ui_mod._es(ej["agg_pes"]["fair_value"], 1) in txt


def test_modelo_seccion_valor_justo_se_renderiza_en_su_expansion(cliente):
    from nicegui import ui
    ui_mod.renderizar_modelo(ui_mod.contenido_modelo())
    exp = [e for e in cliente.elements.values() if isinstance(e, ui.expansion)]
    assert exp[-1].props["data-seccion"] == "valor_justo" and exp[-1].props["model-value"] is False
    assert exp[-1].props["label"].startswith("10. Valor justo")


def test_nota_valor_justo_en_la_guia_post_resultados():
    from pathlib import Path
    guia = (Path(__file__).resolve().parent.parent / "docs" / "GUIA_POST_RESULTADOS.md").read_text(encoding="utf-8")
    assert "Valor Justo" in guia and "no es un precio objetivo" in guia
