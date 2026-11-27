"""Escenarios Medio / Pesimista del valor justo, motor DCF de 10 anios, consolidacion por familias, modo local/nube y
serie historica con bandas. Casos numericos a mano (se indica la derivacion); sin red ni escritura fuera de ``tmp_path``.

Recordatorio: los parametros de los escenarios son SUPUESTOS fijados por el usuario, no predicciones.
"""
import gc
import json
import math
import os
import time

import numpy as np
import pandas as pd
import pytest

from src.levels import fair_value as fv
from src.levels.fair_value import EntradasValoracion as Ent
from src.levels.fair_value import ParametrosValoracion as Par
from src.reporting import cache_local
from test_fair_value import _df_precios, _facts_falsos

MA = 1.1 ** 0.5


@pytest.fixture(autouse=True)
def _aislado(monkeypatch, tmp_path):
    """Sin red (^TNX) y con el directorio de datos en tmp_path (cache SQLite aislada)."""
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "datos"))
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(fv, "_descargar_tnx", lambda: None)
    monkeypatch.setitem(fv._RF_MEM, "t", 0.0)
    monkeypatch.setitem(fv._RF_MEM, "ttl", 0.0)


def _df_largo():
    """Precio constante 100 con ~13 anios de velas (para que los multiplos de 10 anios atras tengan precio)."""
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=3400)
    return pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1e6, "Dividends": 0.0,
                         "Stock Splits": 0.0}, index=idx)


def ent(**kw):
    base = dict(ticker="T", precio=100.0, acciones_diluidas=10.0, caja=0.0, deuda_total=0.0, tasa_impositiva=0.25)
    base.update(kw)
    return Ent(**base)


def px(e, id_, params=None, esc="medio"):
    return next(m for m in fv.calcular_modelos(e, params, esc) if m["id"] == id_)["precio_estimado"]


def modelo(e, id_, params=None, esc="medio"):
    return next(m for m in fv.calcular_modelos(e, params, esc) if m["id"] == id_)


# ===================================================================================== tasas
def test_beta_de_blume():
    assert fv.beta_blume(1.5) == pytest.approx(4.0 / 3.0)          # 2/3*1.5 + 1/3 = 1.3333...
    assert fv.beta_blume(1.0) == pytest.approx(1.0) and fv.beta_blume(0.4) == pytest.approx(0.6)
    t = fv.calcular_tasas(ent(beta=1.5), Par(rf=0.04, erp=0.045))
    assert t["beta"] == pytest.approx(4.0 / 3.0) and t["beta_raw"] == 1.5
    assert t["ke_capm"] == pytest.approx(0.04 + 4.0 / 3.0 * 0.045)  # 0.04 + 0.06 = 0.10


def test_wacc_medio_acotado_entre_8_y_11_5_por_ciento():
    assert (fv.WACC_MIN, fv.WACC_MAX) == (0.08, 0.115)
    bajo = fv.calcular_tasas(ent(beta=0.5), Par(rf=0.02, erp=0.04))      # ke = 0.02 + 0.6667*0.04 = 4.7 % -> 8 %
    alto = fv.calcular_tasas(ent(beta=3.0), Par(rf=0.05, erp=0.05))      # ke = 0.05 + 2.3333*0.05 = 16.7 % -> 11.5 %
    medio = fv.calcular_tasas(ent(beta=1.0), Par(rf=0.05, erp=0.05))     # ke = 10 % (sin deuda) -> sin tope
    assert bajo["wacc"] == 0.08 and alto["wacc"] == 0.115 and medio["wacc"] == pytest.approx(0.10)
    assert any("acotado" in x for x in bajo["topes_wacc"]) and medio["topes_wacc"] == []


def test_wacc_ponderado_por_estructura_de_capital_real_y_kd_despues_de_impuestos():
    # capitalizacion 100*10 = 1000 y deuda 1000: 50/50. beta 1.0 -> ke = 0.05+0.05 = 0.10 ; kd = 80/1000 = 8 % ;
    # t = 25 % -> kd despues de impuestos 6 % ; WACC = 0.5*0.10 + 0.5*0.06 = 0.08 (justo en el piso)
    e = ent(deuda_total=1000.0, gasto_interes_ltm=80.0, beta=1.0)
    t = fv.calcular_tasas(e, Par(rf=0.05, erp=0.05))
    assert t["peso_deuda"] == pytest.approx(0.5) and t["kd"] == pytest.approx(0.08)
    assert t["wacc_bruto"] == pytest.approx(0.5 * 0.10 + 0.5 * 0.08 * 0.75)


def test_wacc_pesimista_es_medio_mas_200pb_con_minimo_12_5():
    assert fv.WACC_PESIMISTA_SPREAD == 0.02 and fv.WACC_PESIMISTA_MIN == 0.125
    e = ent(beta=1.0)
    # Medio 9 % -> 11 % < 12.5 % -> minimo 12.5 % ; Medio 11.5 % -> 13.5 %
    for wacc_medio, esperado in ((0.09, 0.125), (0.105, 0.125), (0.115, 0.135), (0.11, 0.13)):
        t = fv.calcular_tasas(e, Par(wacc=wacc_medio), "pesimista")
        assert t["wacc"] == pytest.approx(esperado) and t["wacc_medio"] == pytest.approx(wacc_medio)
    # sin override: se deriva del WACC automatico del Medio
    auto = fv.calcular_tasas(e, Par(rf=0.05, erp=0.05))["wacc"]
    assert fv.calcular_tasas(e, Par(rf=0.05, erp=0.05), "pesimista")["wacc"] == pytest.approx(max(auto + 0.02, 0.125))
    assert any("mínimo" in x for x in fv.calcular_tasas(e, Par(wacc=0.09), "pesimista")["topes_wacc"])


def test_g_terminal_por_escenario():
    assert fv.G_TERMINAL == {"medio": 0.035, "pesimista": 0.025} and fv.CRECIMIENTO_TERMINAL_DEFECTO == 0.035
    e = ent()
    assert fv.calcular_tasas(e, Par())["g"] == 0.035 and fv.calcular_tasas(e, Par(), "pesimista")["g"] == 0.025
    # el Pesimista nunca supera al Medio vigente: con g = 2 % ambos usan 2 %
    assert fv.calcular_tasas(e, Par(g=0.02), "pesimista")["g"] == 0.02
    assert fv.calcular_tasas(e, Par(g=0.04), "pesimista")["g"] == 0.025


def test_rf_prioridad_usuario_tnx_respaldo():
    assert fv.calcular_tasas(ent(rf=0.05), Par(rf=0.03))["rf"] == 0.03
    t = fv.calcular_tasas(ent(rf=0.05), Par())
    assert t["rf"] == 0.05 and "TNX" in t["rf_origen"]
    t = fv.calcular_tasas(ent(), Par())
    assert t["rf"] == fv.RF_DEFECTO == 0.0425 and "respaldo" in t["rf_origen"]
    assert fv.ERP_DEFECTO == 0.045


def test_normalizar_tnx_en_puntos_porcentuales_y_rango():
    assert fv.normalizar_tnx(4.25) == pytest.approx(0.0425)       # yfinance: puntos porcentuales -> /100
    assert fv.normalizar_tnx(0.0425) == pytest.approx(0.0425)      # ya es fraccion
    assert fv.normalizar_tnx(0.5) == pytest.approx(0.005) and fv.normalizar_tnx(12.0) == pytest.approx(0.12)
    for malo in (0.3, 0.45, 12.5, 40.0, 0.0, -1.0, None, float("nan"), "x"):
        assert fv.normalizar_tnx(malo) is None


def test_obtener_rf_cachea_24h_y_cae_a_none_sin_red(monkeypatch):
    llamadas = []

    def fetch():
        llamadas.append(1)
        return 4.4
    assert fv.obtener_rf(fetch=fetch) == pytest.approx(0.044) and llamadas == [1]
    assert fv.obtener_rf(fetch=fetch) == pytest.approx(0.044) and llamadas == [1]      # memoria (24 h)
    # tras reiniciar la memoria: lo sirve el cache de disco (local) sin llamar a la red
    fv._RF_MEM.update(t=0.0, ttl=0.0)
    assert fv.obtener_rf(fetch=lambda: pytest.fail("no debe ir a la red")) == pytest.approx(0.044)
    # sin red ni cache: None (el motor usa RF_DEFECTO)
    cache_local.purgar(todo=True)
    fv._RF_MEM.update(t=0.0, ttl=0.0)

    def sin_red():
        raise ConnectionError("sin red")
    assert fv.obtener_rf(fetch=sin_red) is None
    fv._RF_MEM.update(t=0.0, ttl=0.0)
    assert fv.obtener_rf(fetch=lambda: 99.0) is None            # fuera de rango


# ===================================================================================== DCF de 10 anios
def test_valor_terminal_a_mitad_de_anio_se_descuenta_con_el_df_del_ultimo_flujo():
    # flujos constantes 100 (g = 0) con WACC 10 %: Gordon con flujos a mitad de anio = 100/0.10 * 1.1^0.5 = 1048.81
    # = anualidad de 10 flujos + TV 1000 descontado a t = 9.5. Descontarlo a t = 10 daria 1029.99 (subvalua 1.8 %).
    e = ent(fcf_ltm=100.0)
    p = Par(wacc=0.10, g=0.0)
    assert px(e, "dcf_10y_fcf", p) == pytest.approx(1000.0 * MA / 10, rel=1e-12)
    flujos = [100.0] * 10
    assert fv._valor_flujos(flujos, 0.10, 0.0) == pytest.approx(1000.0 * MA, rel=1e-12)
    erroneo = sum(100.0 / 1.1 ** (t - 0.5) for t in range(1, 11)) + 1000.0 / 1.1 ** 10
    assert erroneo == pytest.approx(1029.9909, abs=1e-3) and erroneo < fv._valor_flujos(flujos, 0.10, 0.0)


def test_dcf_10_anios_dos_etapas_a_mano():
    # FCF0 = 100 ; etapa 1 (anios 1-3) g1 = 10 % ; etapa 2 (4-10) decae linealmente 1 pp por anio hasta g∞ = 3 % :
    # 9, 8, 7, 6, 5, 4, 3 % ; WACC 10 % ; mitad de anio. Flujos: 110, 121, 133.1, 145.079, 156.68532, 167.653292,
    # 177.712490, 186.598114, 194.062039, 199.883900. VP(flujos) = 973.950355 ; TV = 199.8839*1.03/0.07 = 2941.1
    # descontado con DF(9.5) = 1.1^-9.5 -> 1189.286504 ; total 2163.236860 -> 216.3236859 por accion (10 acciones).
    e = ent(fcf_ltm=100.0)
    p = Par(wacc=0.10, g=0.03, crecimiento_inicial=0.10)
    assert px(e, "dcf_10y_fcf", p) == pytest.approx(216.3236859453451, rel=1e-12)
    m = modelo(e, "dcf_10y_fcf", p)
    assert m["horizonte"] == 10 and m["g1"] == 0.10 and m["g"] == 0.03
    # con el valor terminal descontado en t = 10 (convencion descartada) darian 210.789: el motor NO lo usa
    assert abs(px(e, "dcf_10y_fcf", p) - 210.78905449034733) > 5


def test_trayectoria_de_crecimiento_y_convergencia_lineales():
    assert fv._trayectoria(0.10, 0.03, 10) == pytest.approx([0.10] * 3 + [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
    assert fv._trayectoria(0.10, 0.04, 5) == pytest.approx([0.10, 0.10, 0.10, 0.07, 0.04])   # variante corta
    assert [fv._convergencia(0.2, 0.3, t, 10) for t in (1, 3, 4, 10)] == pytest.approx([0.2, 0.2, 0.2 + 0.1 / 7, 0.3])


@pytest.mark.parametrize("wacc, g, aplica", [(0.05, 0.03, False), (0.0599, 0.035, False), (0.055, 0.03, True),
                                              (0.10, 0.075, True), (0.10, 0.0751, False)])
def test_guarda_wacc_menos_g_infinito_de_2_5_pp(wacc, g, aplica):
    # ¡Gordon exige WACC - g∞ >= 0.025! (si no, el DCF no aplica con motivo; el resto de modelos no cambia)
    e = ent(fcf_ltm=100.0, ebitda_ltm=200.0, da_ltm=40.0, capex_ltm=40.0, ingresos_ltm=1000.0, eps_ltm=4.0)
    ms = {m["id"]: m for m in fv.calcular_modelos(e, Par(wacc=wacc, g=g, pe_objetivo=15.0))}
    for i in ("dcf_5y_fcf", "dcf_10y_fcf", "dcf_10y_ebitda", "dcf_10y_revenue"):
        assert ms[i]["aplicable"] is aplica, i
        if not aplica:
            assert "2.5 pp" in ms[i]["motivo_exclusion"]
    assert ms["pe"]["aplicable"]
    assert fv.DIF_MIN_WACC_G_DCF == 0.025


def test_topes_de_crecimiento_etapa1_42_y_15_por_ciento():
    # variaciones muy grandes -> 65 % del crecimiento fundamental (>= 65 %) topado por escenario
    hist = [10.0, 40.0, 160.0, 640.0, 2560.0]
    assert fv.crecimiento_etapa1(hist, None, 0.035, "medio")["g1"] == 0.42
    assert fv.crecimiento_etapa1(hist, None, 0.025, "pesimista")["g1"] == 0.15
    assert fv.TOPE_CRECIMIENTO_ETAPA1 == {"medio": 0.42, "pesimista": 0.15}
    # y el Medio supera al Pesimista en el DCF que depende del crecimiento
    e = ent(fcf_ltm=2560.0, hist_fcf=hist)
    assert px(e, "dcf_10y_fcf", Par(wacc=0.10)) > px(e, "dcf_10y_fcf", Par(wacc=0.10), "pesimista")
    assert modelo(e, "dcf_10y_fcf", Par(wacc=0.10))["g1"] == 0.42
    assert modelo(e, "dcf_10y_fcf", Par(wacc=0.10), "pesimista")["g1"] == 0.15


def test_topes_de_margen_55_y_30_por_ciento():
    assert fv.TOPE_MARGEN == {"medio": 0.55, "pesimista": 0.30}
    e = ent(ingresos_ltm=1000.0, fcf_ltm=700.0, hist_fcf=[700.0] * 4, hist_ingresos=[1000.0] * 4)   # margen 70 %
    assert fv._margen_fcf(e)[1] == 0.55 and fv._margen_fcf(e, "pesimista")[1] == 0.30
    # margen por debajo del tope: se usa la mediana recortada (20 %) en ambos
    e2 = ent(ingresos_ltm=1000.0, fcf_ltm=200.0, hist_fcf=[200.0] * 4, hist_ingresos=[1000.0] * 4)
    assert fv._margen_fcf(e2)[1] == fv._margen_fcf(e2, "pesimista")[1] == pytest.approx(0.20)


def test_dcf_ebitda_converge_a_la_conversion_historica_con_tope():
    # hist FCF/EBITDA = 0.5 (mediana recortada) ; margen EBITDA = 200/1000 = 20 % -> tope Medio en base EBITDA =
    # min(1, 0.55/0.20) = 1 (no activo) y Pesimista = 0.30/0.20 = 1.5 -> 1 : objetivo 0.5 en ambos
    e = ent(ebitda_ltm=200.0, ingresos_ltm=1000.0, da_ltm=20.0, capex_ltm=20.0, fcf_ltm=100.0,
            hist_fcf=[100.0] * 4, hist_ebitda=[200.0] * 4, hist_ingresos=[1000.0] * 4)
    assert fv._conversion_ebitda(e, "medio", 0.9)["objetivo"] == pytest.approx(0.5)
    # margen EBITDA 80 %: tope Pesimista = 0.30/0.80 = 0.375 < 0.5 -> activo ; Medio 0.55/0.8 = 0.6875 -> 0.5
    e2 = ent(ebitda_ltm=800.0, ingresos_ltm=1000.0, hist_fcf=[400.0] * 4, hist_ebitda=[800.0] * 4)
    assert fv._conversion_ebitda(e2, "pesimista", 0.9)["objetivo"] == pytest.approx(0.375)
    assert fv._conversion_ebitda(e2, "pesimista", 0.9)["tope_activo"] is True
    assert fv._conversion_ebitda(e2, "medio", 0.9)["objetivo"] == pytest.approx(0.5)


# ===================================================================================== crecimiento fundamental
def test_cagr_winsorizado_compone_variaciones_recortadas():
    assert fv.cagr_winsorizado([100.0, 110.0, 121.0], 3) == pytest.approx(0.10)
    # 3 variaciones: 10 %, 10 % y +726 % (recorte duro a +100 %): (1.1*1.1*2)^(1/3) - 1
    assert fv.cagr_winsorizado([100.0, 110.0, 121.0, 1000.0], 3) == pytest.approx((1.1 * 1.1 * 2.0) ** (1 / 3) - 1)
    # 5 variaciones [.1,.1,.1,.1,+100 %]: percentil 90 (interpolado) = 0.1 + 0.6*0.9 = 0.64 recorta la ultima
    h = [100.0, 110.0, 121.0, 133.1, 146.41, 400.0]
    assert fv.cagr_winsorizado(h, 5) == pytest.approx((1.1 ** 4 * 1.64) ** (1 / 5) - 1, rel=1e-9)
    assert fv.cagr_winsorizado([100.0], 3) is None and fv.cagr_winsorizado([100.0, 90.0], 3) is None
    assert fv.cagr_winsorizado([-1.0, 5.0, 6.0], 3) is None        # una sola variacion valida (>0 en ambos extremos)


def test_crecimiento_fundamental_mezcla_pesos_y_renormaliza():
    assert (fv.PESO_CAGR, fv.PESO_MOMENTUM, fv.PESO_FORWARD) == (0.5, 0.2, 0.3)
    h = [100.0, 110.0, 121.0]
    assert fv.crecimiento_fundamental(h)["valor"] == pytest.approx(0.10)                         # sin forward: renormaliza
    assert fv.crecimiento_fundamental(h, 0.20)["valor"] == pytest.approx(0.5 * 0.10 + 0.2 * 0.10 + 0.3 * 0.20)
    assert fv.crecimiento_fundamental([], 0.20)["valor"] == pytest.approx(0.20)                  # solo forward
    assert fv.crecimiento_fundamental(h, 5.0)["componentes"]["forward"] == 1.0                    # forward recortado
    assert fv.crecimiento_fundamental([], None)["valor"] is None


def test_sin_datos_de_crecimiento_nunca_es_cero_silencioso():
    r = fv.crecimiento_etapa1([], None, 0.035, "medio")
    assert r["g1"] == 0.035 and "sin datos" in r["nota"]          # g∞, no 0 %
    e = ent(fcf_ltm=100.0)
    assert modelo(e, "dcf_10y_fcf", Par(wacc=0.10))["crecimiento_nota"].startswith("sin datos")


def test_mediana_recortada():
    assert fv.mediana_recortada([10.0, 12.0, 14.0]) == 12.0                 # < 5 puntos: mediana simple
    assert fv.mediana_recortada([1.0, 14.0, 15.0, 16.0, 100.0]) == 15.0       # 5 puntos: descarta 1 por lado
    assert fv.mediana_recortada([1.0, 12.0, 14.0, 16.0, 18.0, 100.0]) == 15.0  # [12,14,16,18] -> 15
    assert fv.mediana_recortada([]) is None


# ===================================================================================== multiplos
def test_multiplo_objetivo_mediana_recortada_y_benchmark_sectorial():
    hist = {"pe": [1.0, 12.0, 14.0, 16.0, 18.0, 100.0]}          # recortada = 15 ; sin sector: 15
    e = ent(eps_ltm=4.0, multiplos_hist=hist)
    assert px(e, "pe") == pytest.approx(60.0)
    # Technology: benchmark P/E 26 -> 0.7*15 + 0.3*26 = 18.3 -> 73.2
    et = ent(eps_ltm=4.0, multiplos_hist=hist, sector="Technology")
    assert (fv.PESO_MULTIPLO_PROPIO, fv.BENCHMARK_SECTORIAL["Technology"]["pe"]) == (0.7, 26.0)
    assert px(et, "pe") == pytest.approx(4.0 * (0.7 * 15.0 + 0.3 * 26.0))
    mo = fv.multiplo_objetivo("pe", et, Par())
    assert mo["benchmark"] == 26.0 and "sector" in mo["origen"] and fv.BENCHMARK_FECHA in mo["origen"]
    # sector fuera de la tabla: solo el propio
    assert px(ent(eps_ltm=4.0, multiplos_hist=hist, sector="Marte"), "pe") == pytest.approx(60.0)


def test_compresion_de_multiplos_15_por_ciento_en_el_pesimista():
    assert fv.COMPRESION_MULTIPLOS == {"medio": 0.0, "pesimista": -0.15}
    e = ent(eps_ltm=4.0, bvps=10.0, fcf_ltm=100.0, ebitda_ltm=200.0, ingresos_ltm=1000.0, caja=50.0, deuda_total=20.0)
    p = Par(pe_objetivo=15.0, pb_objetivo=3.0, p_fcf_objetivo=12.0, ev_ebitda_objetivo=10.0, ev_revenue_objetivo=2.0)
    for id_, medio in (("pe", 60.0), ("pb", 30.0), ("p_fcf", 120.0)):
        assert px(e, id_, p) == pytest.approx(medio)
        assert px(e, id_, p, "pesimista") == pytest.approx(medio * 0.85)
    # EV/EBITDA: (200*10*0.85 + 30)/10 = 173.0 ; el castigo se aplica al multiplo, no a la caja neta
    assert px(e, "ev_ebitda", p, "pesimista") == pytest.approx((200 * 8.5 + 30) / 10)
    assert px(e, "ev_revenue", p, "pesimista") == pytest.approx((1000 * 1.7 + 30) / 10)
    # tambien al propio historico (constante 15x): 60 -> 51
    assert px(ent(eps_ltm=4.0, multiplos_hist={"pe": [15.0] * 4}), "pe", None, "pesimista") == pytest.approx(51.0)


def test_rangos_de_cordura_y_marca_de_tope_se_mantienen():
    e = ent(eps_ltm=2.0, multiplos_hist={"pe": [80.0] * 4})
    m = modelo(e, "pe")
    assert m["precio_estimado"] == pytest.approx(2.0 * fv.RANGOS_MULTIPLOS["pe"][1]) and m["tope_activo"]
    assert "topada" in m["topes"][0]


# ===================================================================================== valor / dividendos
def test_epv_se_excluye_en_medio_si_los_ingresos_crecen_mas_de_20_por_ciento():
    base = dict(ebit_ltm=100.0, tasa_impositiva=0.2)
    rapido = ent(hist_ingresos=[100.0, 125.0], **base)            # +25 % anual
    lento = ent(hist_ingresos=[100.0, 115.0], **base)             # +15 %
    p = Par(wacc=0.10)
    m = modelo(rapido, "epv", p)
    assert not m["aplicable"] and "25 %" in m["motivo_exclusion"] and "20 %" in m["motivo_exclusion"]
    # Pesimista: se incluye como piso de crecimiento cero: NOPAT 80 / WACC pesimista 12.5 % = 640 -> 64
    pes = modelo(rapido, "epv", p, "pesimista")
    assert pes["aplicable"] and pes["precio_estimado"] == pytest.approx(80.0 / 0.125 / 10)
    assert modelo(lento, "epv", p)["aplicable"] and px(lento, "epv", p) == pytest.approx(80.0 / 0.10 / 10)
    # exactamente 20 % no se excluye (umbral estricto)
    assert modelo(ent(hist_ingresos=[100.0, 120.0], **base), "epv", p)["aplicable"]
    assert fv.EPV_CRECIMIENTO_MAX == 0.20


def test_ddm_y_lynch_por_escenario():
    e = ent(dividendo_anual=2.0, beta=1.0, eps_ltm=2.0, hist_eps=[100.0, 110.0, 121.0])
    p = Par(rf=0.04, erp=0.05)
    # DDM Medio: ke 9 %, g 3.5 % -> 2.07/0.055 ; Pesimista: ke 9 %+(12.5 %-9 %... ) se desplaza con el WACC (+0.035)
    assert px(e, "ddm", p) == pytest.approx(2.0 * 1.035 / (0.09 - 0.035))
    tp = fv.calcular_tasas(e, p, "pesimista")
    assert tp["wacc"] == pytest.approx(0.125) and tp["ke"] == pytest.approx(0.125)       # sin deuda: ke = WACC
    assert px(e, "ddm", p, "pesimista") == pytest.approx(2.0 * 1.025 / (0.125 - 0.025))
    # Lynch: crecimiento fundamental 10 % -> EPS*10 = 20 en ambos (tope 25 % / 15 % no activo)
    assert px(e, "lynch", p) == pytest.approx(20.0) and px(e, "lynch", p, "pesimista") == pytest.approx(20.0)
    # crecimiento 40 %: Medio topa a 25, Pesimista a 15
    e2 = ent(eps_ltm=2.0, hist_eps=[100.0, 140.0, 196.0])
    assert px(e2, "lynch") == pytest.approx(50.0) and px(e2, "lynch", None, "pesimista") == pytest.approx(30.0)


# ===================================================================================== consolidacion
def _m(id_, precio, familia):
    return {"id": id_, "nombre": id_, "familia": familia, "precio_estimado": precio, "aplicable": precio is not None,
            "motivo_exclusion": None if precio is not None else "x"}


def test_consolidado_iguala_las_tres_familias():
    lista = [_m("d1", 100.0, "DCF"), _m("d2", 110.0, "DCF"), _m("d3", 90.0, "DCF"), _m("d4", 100.0, "DCF"),
             _m("m1", 80.0, "Múltiplos"), _m("m2", 90.0, "Múltiplos"),
             _m("dd", 120.0, "Dividendos"), _m("v1", 100.0, "Valor")]
    a = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    # DCF = 100 ; Multiplos = 85 ; Valor/Dividendos = 110 -> (100+85+110)/3 = 98.3333
    assert a["fair_value"] == pytest.approx(295.0 / 3)
    assert a["pesos_familias"] == {g: pytest.approx(1 / 3) for g in fv.GRUPOS}
    assert a["submedias"] == {"DCF": 100.0, "Múltiplos": 85.0, "Valor / Dividendos": 110.0}
    assert a["promedio_simple"] == pytest.approx(790.0 / 8) and a["mediana"] == pytest.approx(100.0)
    assert a["metodo"] == "igualacion_familias" and a["incertidumbre"] in fv.NIVELES_INCERTIDUMBRE


def test_consolidado_renormaliza_si_falta_una_familia():
    lista = [_m("d1", 100.0, "DCF"), _m("d2", 140.0, "DCF"), _m("v1", 60.0, "Valor"), _m("dd", 80.0, "Dividendos")]
    a = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    # sin Multiplos: DCF = 120 ; Valor/Dividendos = 70 -> pesos 1/2 y 1/2 -> 95
    assert a["fair_value"] == pytest.approx(95.0) and a["pesos_familias"] == {"DCF": 0.5, "Valor / Dividendos": 0.5}
    solo = fv.agregar_valor_justo(lista, ["v1", "dd"], 0.25, 100.0)
    assert solo["fair_value"] == pytest.approx(70.0) and solo["pesos_familias"] == {"Valor / Dividendos": 1.0}


def test_iqr_descarta_outliers_antes_de_igualar_familias():
    # 5 validos: Q1 = 100, Q3 = 102 -> valla [97, 105]: 1000 queda fuera ; las familias se igualan SIN el outlier
    lista = [_m("d1", 100.0, "DCF"), _m("m1", 102.0, "Múltiplos"), _m("v1", 98.0, "Valor"), _m("dd", 101.0, "Dividendos"),
             _m("d2", 1000.0, "DCF")]
    a = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    assert a["modelos_outlier"] == ["d2"] and a["fair_value"] == pytest.approx((100 + 102 + 99.5) / 3)
    # con menos de 4 validos no hay IQR
    a3 = fv.agregar_valor_justo(lista[:2] + [lista[4]], None, 0.25, 100.0)
    assert a3["modelos_outlier"] == [] and a3["fair_value"] == pytest.approx((550.0 + 102.0) / 2)


def test_sin_familia_en_los_modelos_cae_al_promedio_simple():
    lista = [{"id": f"x{i}", "precio_estimado": v, "aplicable": True} for i, v in enumerate((100.0, 200.0, 300.0))]
    a = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    assert a["fair_value"] == pytest.approx(200.0) and a["metodo"] == "promedio_simple"


def _entradas_ricas():
    return Ent(ticker="RICA", precio=100.0, acciones_diluidas=10.0, fcf_ltm=100.0, ebitda_ltm=160.0, ebit_ltm=120.0,
               ingresos_ltm=900.0, eps_ltm=6.0, bvps=25.0, caja=100.0, deuda_total=200.0, tasa_impositiva=0.2,
               capex_ltm=30.0, da_ltm=40.0, nwc_operativo=-20.0, gasto_interes_ltm=10.0, dividendo_anual=3.0, beta=1.0,
               hist_fcf=[70.0, 78.0, 86.0, 93.0, 100.0], hist_ingresos=[600.0, 680.0, 760.0, 830.0, 900.0],
               hist_ebitda=[110.0, 120.0, 132.0, 146.0, 160.0], hist_eps=[3.5, 4.0, 4.7, 5.3, 6.0],
               multiplos_hist={"pe": [14.0, 15.0, 16.0, 17.0, 18.0], "ev_ebitda": [8.0, 9.0, 10.0, 11.0, 12.0],
                               "ev_revenue": [1.5, 1.7, 1.9, 2.0, 2.2], "pb": [3.0, 3.2, 3.5, 3.8, 4.0],
                               "p_fcf": [11.0, 12.0, 13.0, 14.0, 15.0]}, sector="Industrials", industria="Machinery")


def test_calcular_escenarios_estructura_y_sin_escenario_optimista():
    r = fv.calcular_escenarios(_entradas_ricas())
    assert set(r) == {"medio", "pesimista", "modelos", "supuestos", "escenarios", "aviso"}
    assert r["escenarios"] == ("medio", "pesimista") == fv.ESCENARIOS
    assert len(r["modelos"]) == 15 and [m["id"] for m in r["modelos"]] == list(fv.IDS_MODELOS)
    for m in r["modelos"]:
        assert {"id", "nombre", "familia", "aplicable", "motivo", "precio_medio", "precio_pesimista", "activo"} <= set(m)
    assert "optimista" not in json.dumps(r, default=str).lower() and "optimist" not in json.dumps(r, default=str).lower()
    with pytest.raises(ValueError):
        fv.calcular_modelos(_entradas_ricas(), None, "optimista")
    with pytest.raises(ValueError):
        fv.calcular_tasas(_entradas_ricas(), Par(), "optimista")
    assert set(r["supuestos"]) == {"medio", "pesimista", "avisos", "nota", "modo", "anios_historia"}
    assert r["medio"]["fair_value"] > r["pesimista"]["fair_value"]
    assert "supuestos de escenario" in r["aviso"].lower() and "no predicciones" in r["aviso"].lower()


def test_calcular_modelos_y_valor_justo_por_defecto_son_el_escenario_medio():
    e = _entradas_ricas()
    r = fv.calcular_escenarios(e)
    assert fv.calcular_modelos(e) == fv.calcular_modelos(e, None, "medio")
    assert [m["precio_estimado"] for m in fv.calcular_modelos(e)] == [m["precio_medio"] for m in r["modelos"]]
    assert fv.valor_justo(e)["fair_value"] == pytest.approx(r["medio"]["fair_value"])
    assert fv.valor_justo(e, escenario="pesimista")["fair_value"] == pytest.approx(r["pesimista"]["fair_value"])


def test_el_consolidado_cambia_al_desactivar_modelos_en_ambos_escenarios():
    e = _entradas_ricas()
    todos = fv.calcular_escenarios(e)
    sin_dcf = fv.calcular_escenarios(e, modelos_activos=[i for i in fv.IDS_MODELOS if not i.startswith("dcf_")])
    solo_mult = fv.calcular_escenarios(e, modelos_activos=["pe", "ev_ebitda", "ev_revenue", "pb", "p_fcf"])
    for esc in fv.ESCENARIOS:
        v = {round(x[esc]["fair_value"], 6) for x in (todos, sin_dcf, solo_mult)}
        assert len(v) == 3, esc
    assert sin_dcf["medio"]["pesos_familias"].keys() == {"Múltiplos", "Valor / Dividendos"}
    assert not any(m["activo"] for m in sin_dcf["modelos"] if m["id"].startswith("dcf_"))
    # solo multiplos: el consolidado es el promedio de sus modelos (una sola familia)
    ms = {m["id"]: m for m in solo_mult["modelos"]}
    usados = solo_mult["medio"]["modelos_usados"]
    assert solo_mult["medio"]["fair_value"] == pytest.approx(np.mean([ms[i]["precio_medio"] for i in usados]))


def test_sliders_sobreescriben_el_medio_y_el_pesimista_se_deriva_con_aviso():
    e = _entradas_ricas()
    r = fv.calcular_escenarios(e, Par(wacc=0.10, g=0.03))
    assert r["supuestos"]["medio"]["wacc"] == 0.10 and r["supuestos"]["medio"]["g"] == 0.03
    assert r["supuestos"]["pesimista"]["wacc"] == pytest.approx(0.125)          # max(10 % + 2 pp, 12.5 %)
    assert r["supuestos"]["pesimista"]["g"] == 0.025 and r["supuestos"]["avisos"] == []
    fuera = fv.calcular_escenarios(e, Par(wacc=0.13, g=0.05))
    assert fuera["supuestos"]["pesimista"]["wacc"] == pytest.approx(0.15)       # deriva del Medio vigente
    assert any("fuera del rango" in a for a in fuera["supuestos"]["avisos"])
    assert any("g∞" in a for a in fuera["supuestos"]["avisos"])


def test_recalcular_con_otros_parametros_es_instantaneo_y_sin_red(monkeypatch):
    e = _entradas_ricas()
    monkeypatch.setattr(fv, "_descargar_tnx", lambda: pytest.fail("el recalculo no debe usar la red"))
    monkeypatch.setattr(fv, "_vf", lambda: pytest.fail("el recalculo no debe tocar EDGAR"))
    t0 = time.perf_counter()
    for w in np.linspace(0.08, 0.115, 20):
        fv.calcular_escenarios(e, Par(wacc=float(w), g=0.03))
    assert (time.perf_counter() - t0) / 20 < 0.25


# ===================================================================================== modo local / nube
def test_modo_por_defecto_segun_la_variable_port(monkeypatch):
    assert fv.modo_por_defecto() == "local"
    monkeypatch.setenv("PORT", "10000")
    assert fv.modo_por_defecto() == "nube"
    assert (fv.ANIOS_LOCAL, fv.ANIOS_NUBE) == (10, 3) and fv.anios_para_modo("nube") == 3 and fv.anios_para_modo("local") == 10
    assert fv._resolver_modo(None, None) == ("nube", 3) and fv._resolver_modo("local", None) == ("local", 10)
    assert fv._resolver_modo("local", 5) == ("local", 5)


def test_extraccion_local_10_anios_vs_nube_3_anios():
    f = _facts_falsos(n_trim=44)               # 11 anios de trimestres
    df = _df_largo()
    loc = fv.extraer_entradas("T", None, df, {}, facts=f, modo="local", rf=0.04)
    nub = fv.extraer_entradas("T", None, df, {}, facts=f, modo="nube", rf=0.04)
    assert (loc.modo, loc.anios_historia) == ("local", 10) and (nub.modo, nub.anios_historia) == ("nube", 3)
    assert len(loc.hist_ingresos) == 11 and len(nub.hist_ingresos) == 4        # anios + 1 (incluye el LTM actual)
    assert len(loc.multiplos_hist["pe"]) == 10 and len(nub.multiplos_hist["pe"]) == 3
    assert len(loc.serie_ttm) > len(nub.serie_ttm) and len(loc.hist_fechas) == 11 and len(nub.hist_fechas) == 4
    assert loc.ingresos_ltm == nub.ingresos_ltm == 1000.0 and loc.rf == nub.rf == 0.04


def test_extraccion_multi_anio_campos_normalizados():
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(n_trim=44), rf=0.04)
    # 4 trimestres: neto no esta en la tabla falsa -> None (nunca 0) ; OCF 120 ; capex 20 ; tasa 25 %
    assert e.neto_ltm is None and e.ocf_ltm == 120.0 and e.capex_ltm == 20.0 and e.patrimonio == 250.0
    assert e.hist_ocf[-1] == 120.0 and e.hist_capex[-1] == 20.0 and e.hist_ebit[-1] == 160.0
    assert e.hist_tasa[-1] == pytest.approx(0.25) and e.hist_deuda[-1] == 500.0 and e.hist_caja[-1] == 500.0
    assert e.hist_patrimonio[-1] == 250.0 and e.hist_acciones[-1] == 10.0
    assert all(n == len(e.hist_fechas) for n in map(len, (e.hist_ebit, e.hist_neto, e.hist_ocf, e.hist_capex, e.hist_tasa,
                                                          e.hist_deuda, e.hist_caja, e.hist_patrimonio, e.hist_acciones)))
    assert e.hist_neto[-1] is None                       # dato ausente: None, no 0
    json.dumps(e.to_dict())                              # serializable (cache y exportacion)
    assert Ent.desde_dict(e.to_dict()).serie_ttm == e.serie_ttm


def test_forward_de_yfinance_se_usa_si_existe_y_no_se_inventa():
    info = {"revenueGrowth": 0.12, "forwardEps": 5.5, "trailingEps": 5.0}
    e = fv.extraer_entradas("T", None, _df_precios(), info, facts=_facts_falsos(), rf=0.04)
    assert e.crec_forward_ingresos == 0.12 and e.crec_forward_beneficios == pytest.approx(0.10)
    e2 = fv.extraer_entradas("T", None, _df_precios(), {"earningsGrowth": 0.3}, facts=_facts_falsos(), rf=0.04)
    assert e2.crec_forward_ingresos is None and e2.crec_forward_beneficios == 0.3
    e3 = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(), rf=0.04)
    assert e3.crec_forward_ingresos is None and e3.crec_forward_beneficios is None


def test_cache_en_disco_solo_en_local(monkeypatch, tmp_path):
    from src.levels import valuation_floors as vf
    llamadas = []

    def facts(t):
        llamadas.append(t)
        return _facts_falsos()
    monkeypatch.setattr(vf, "_facts", facts)
    df = _df_precios()
    ruta = tmp_path / "datos" / cache_local.NOMBRE_ARCHIVO
    a = fv.extraer_entradas("T", None, df, {}, modo="local", rf=0.04)
    assert llamadas == ["T"] and ruta.exists()
    b = fv.extraer_entradas("T", None, df, {}, modo="local", rf=0.04)           # caliente: cero red
    assert llamadas == ["T"] and b.fuentes.get("cache") and b.ingresos_ltm == a.ingresos_ltm and b.hist_fcf == a.hist_fcf
    # el precio se refresca con el df actual al leer del cache
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Close")] = 123.0
    assert fv.extraer_entradas("T", None, df2, {}, modo="local", rf=0.04).precio == 123.0
    assert llamadas == ["T"]                                                         # sigue sin red (nueva ultima vela = misma fecha)
    # nube: ni se lee ni se escribe cache de disco
    os.remove(ruta)
    fv.extraer_entradas("T", None, df, {}, modo="nube", rf=0.04)
    fv.extraer_entradas("T", None, df, {}, modo="nube", rf=0.04)
    assert llamadas.count("T") == 3 and not ruta.exists()
    # facts inyectados: nunca cachea
    fv.extraer_entradas("T", None, df, {}, facts=_facts_falsos(), modo="local", rf=0.04)
    assert not ruta.exists()
    # usar_cache=False tampoco
    fv.extraer_entradas("T", None, df, {}, modo="local", rf=0.04, usar_cache=False)
    assert not ruta.exists()


def test_nube_ejecuta_gc_collect_tras_la_consulta(monkeypatch):
    n = []
    monkeypatch.setattr(gc, "collect", lambda *a: n.append(1) or 0)
    fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(), modo="nube", rf=0.04)
    assert n
    n.clear()
    fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(), modo="local", rf=0.04)
    assert not n


def test_cache_local_deshabilitado_en_la_nube(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    assert not cache_local.habilitado() and cache_local.cache_por_defecto() is None
    assert cache_local.guardar("k", {"a": 1}) is False and cache_local.obtener("k") is None


# ===================================================================================== serie historica
def _serie_sintetica(n=40, salto_en=30, fin="2026-01-30"):
    """40 cierres trimestrales (2016-03-31 ... 2025-12-31) con crecimiento del 2 % por trimestre y un salto del 50 % en el
    ingreso en la fila ``salto_en``; precios diarios en rampa. Devuelve (entradas actuales, df)."""
    cierres = pd.date_range("2016-03-31", periods=n, freq="QE")
    rows = []
    for i, d in enumerate(cierres):
        ing = 4000.0 * 1.02 ** i * (1.5 if i >= salto_en else 1.0)
        rows.append({"fecha": str(d.date()), "ing": ing, "ebit": 0.20 * ing, "da": 0.03 * ing, "ebitda": 0.23 * ing,
                     "capex": 0.04 * ing, "ocf": 0.18 * ing, "fcf": 0.14 * ing, "neto": 0.12 * ing, "eps": 0.12 * ing / 100.0,
                     "interes": 5.0, "impuesto": 0.032 * ing, "pre": 0.16 * ing, "caja": 500.0, "deuda": 300.0,
                     "patrimonio": 2000.0 + 40.0 * i, "acciones": 100.0, "ac": 1500.0, "pc": 900.0, "dcp": 100.0})
    idx = pd.bdate_range("2015-06-01", fin)
    px_ = 20.0 + 0.03 * np.arange(len(idx))
    df = pd.DataFrame({"Open": px_, "High": px_, "Low": px_, "Close": px_, "Volume": 1e6, "Dividends": 0.0,
                       "Stock Splits": 0.0}, index=idx)
    base = Ent(ticker="SYN", beta=1.0, rf=0.0425, sector="Industrials", industria="Machinery", serie_ttm=rows,
               modo="local", anios_historia=10)
    fechas = np.array([np.datetime64(pd.Timestamp(r["fecha"])) for r in rows])
    pr = fv._precio_sin_ajuste_dividendos(df)
    actual = fv._entradas_en(base, rows, fechas, n - 1, idx[-1], pr, None, 10)
    actual.serie_ttm, actual.beta, actual.rf, actual.sector = rows, 1.0, 0.0425, "Industrials"
    actual.modo, actual.anios_historia, actual.ticker = "local", 10, "SYN"
    return actual, df


def test_serie_estructura_serializable_y_bandas_exactas():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    json.dumps(s)
    n = len(s["fechas"])
    assert n > 300 and n == len(s["valor_justo"]) == len(s["precio"]) and s["fechas"] == sorted(s["fechas"])
    assert list(s["bandas"]) == ["+30%", "+20%", "+10%", "-10%", "-20%", "-30%"] and s["escenario"] == "medio"
    vj = np.array(s["valor_justo"])
    for pct, serie in s["bandas"].items():
        k = 1 + int(pct.replace("%", "")) / 100.0
        assert np.allclose(serie, vj * k, rtol=2e-5), pct                  # bandas exactas +-10/20/30 %
    for pct, serie in s["proyeccion"]["bandas"].items():
        k = 1 + int(pct.replace("%", "")) / 100.0
        assert np.allclose(serie, np.array(s["proyeccion"]["valor_justo"]) * k, rtol=2e-5)
    assert all(isinstance(f, str) and len(f) == 10 for f in s["fechas"])


def test_serie_anclada_al_valor_justo_medio_actual():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    actual = fv.valor_justo(e)["fair_value"]
    assert s["fechas"][-1] == str(df.index[-1].date()) and s["valor_justo"][-1] == pytest.approx(actual, rel=1e-6)
    assert s["ancla"]["valor_justo"] == pytest.approx(actual) and s["ancla"]["fecha"] == s["fechas"][-1]
    # la proyeccion arranca exactamente en el ancla
    assert s["proyeccion"]["fechas"][0] == s["fechas"][-1] and s["proyeccion"]["valor_justo"][0] == pytest.approx(actual, rel=1e-6)
    # con otros WACC/g (slider) el ancla sigue al Medio vigente
    s2 = fv.serie_valor_justo_historica(e, df, Par(wacc=0.11, g=0.03), usar_cache=False)
    assert s2["valor_justo"][-1] == pytest.approx(fv.valor_justo(e, Par(wacc=0.11, g=0.03))["fair_value"], rel=1e-6)
    assert s2["valor_justo"][-1] != s["valor_justo"][-1]


def test_proyeccion_12_meses_ilustrativa_crece_a_la_tasa_declarada():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    pr = s["proyeccion"]
    assert s["ilustrativa_proyeccion"] is True and "ILUSTRATIVA" in pr["metodo"]
    g = pr["g_anual"]
    assert 0.0 <= g <= fv.PROYECCION_TOPE
    ultima = pd.Timestamp(s["fechas"][-1])
    assert pd.Timestamp(pr["fechas"][-1]) - ultima == pd.Timedelta(days=365)
    assert pr["valor_justo"][-1] == pytest.approx(s["valor_justo"][-1] * (1 + g), rel=2e-5)
    d = [(pd.Timestamp(f) - ultima).days for f in pr["fechas"]]
    assert pr["valor_justo"][3] == pytest.approx(s["valor_justo"][-1] * (1 + g) ** (d[3] / 365.0), rel=2e-5)
    # g_proy = 0.5*g1 medio + 0.5*g∞ (acotado): coincide con la funcion
    ms = fv.calcular_modelos(e, None, "medio")
    assert fv._g_proyeccion(ms)[0] == pytest.approx(g)


def test_serie_sin_escalones_entre_reportes():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    vj = np.array(s["valor_justo"])
    saltos_pts = np.abs(np.diff([p["valor_justo"] for p in s["puntos_trimestrales"]]))
    assert saltos_pts.max() > 0
    paso = np.abs(np.diff(vj)).max()
    # smoothstep de 45 d muestreado cada 7 d: el paso maximo es <= 1.5*7/45 = 23 % del salto entre reportes
    # (el ultimo tramo hacia el ancla puede ser el mayor: se incluye en la comparacion)
    saltos = list(saltos_pts) + [abs(s["ancla"]["valor_justo"] - s["puntos_trimestrales"][-1]["valor_justo"])]
    assert paso <= 0.25 * max(saltos) + 1e-9
    # y hay un salto real (fila 30: ingresos +50 %) que se reparte en varios pasos, no en uno
    assert (np.abs(np.diff(vj)) > 0.5 * paso).sum() >= 3


def test_serie_pit_sin_lookahead_alterando_el_futuro():
    e, df = _serie_sintetica()
    base = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    # se altera TODO lo posterior a la fila 20: fundamentales x5 y precios x0.4 despues de su fecha de publicacion
    j = 20
    t_j = pd.Timestamp(e.serie_ttm[j]["fecha"]) + pd.Timedelta(days=fv.REZAGO_PUBLICACION_DIAS)
    e2 = Ent.desde_dict(e.to_dict())
    for r in e2.serie_ttm[j + 1:]:
        for k in ("ing", "ebit", "ebitda", "fcf", "eps", "neto", "ocf"):
            r[k] = r[k] * 2.0
        r["patrimonio"] = r["patrimonio"] * 2.0
    df2 = df.copy()
    df2.loc[df2.index > t_j, ["Open", "High", "Low", "Close"]] *= 0.7
    alt = fv.serie_valor_justo_historica(e2, df2, usar_cache=False)
    # el ancla (valor actual) no se altera: es el mismo e con las mismas columnas actuales
    t_sig = pd.Timestamp(e.serie_ttm[j + 1]["fecha"]) + pd.Timedelta(days=fv.REZAGO_PUBLICACION_DIAS)
    comunes = [(a, b, f) for a, b, f in zip(base["valor_justo"], alt["valor_justo"], base["fechas"])
               if pd.Timestamp(f) < t_sig and f in set(alt["fechas"])]
    assert len(comunes) > 20
    for a, b, f in comunes:
        assert a == pytest.approx(b, rel=1e-9), f
    # los puntos trimestrales hasta la fila j son identicos y los posteriores si cambian
    pa = {p["fecha_cierre"]: p["valor_justo"] for p in base["puntos_trimestrales"]}
    pb = {p["fecha_cierre"]: p["valor_justo"] for p in alt["puntos_trimestrales"]}
    ant = [k for k in pa if k <= e.serie_ttm[j]["fecha"]]
    assert ant and all(pa[k] == pytest.approx(pb[k], rel=1e-12) for k in ant)
    despues = [k for k in pa if k > e.serie_ttm[j + 1]["fecha"] and k in pb]
    assert despues and any(abs(pa[k] - pb[k]) > 1e-6 for k in despues)


def test_serie_usa_rezago_de_45_dias_y_la_mediana_solo_con_historia_previa():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    for p in s["puntos_trimestrales"]:
        assert (pd.Timestamp(p["fecha_publicacion"]) - pd.Timestamp(p["fecha_cierre"])).days == fv.REZAGO_PUBLICACION_DIAS
    # la primera curva no puede usar multiplos: en las primeras filas hay < 3 ejercicios previos (se nota en n_modelos)
    n = [p["n_modelos"] for p in s["puntos_trimestrales"]]
    assert n[0] < n[-1]
    primera = pd.Timestamp(s["fechas"][0])
    assert primera >= pd.Timestamp(e.serie_ttm[0]["fecha"]) + pd.Timedelta(days=45)
    # entradas PIT de una fila temprana: multiplos solo de ejercicios >= 1 anio antes
    rows = e.serie_ttm
    fechas = np.array([np.datetime64(pd.Timestamp(r["fecha"])) for r in rows])
    pr = fv._precio_sin_ajuste_dividendos(df)
    ent_i = fv._entradas_en(e, rows, fechas, 12, pd.Timestamp(rows[12]["fecha"]) + pd.Timedelta(days=45), pr, None, 10)
    assert len(ent_i.multiplos_hist["pe"]) == 3 and len(ent_i.hist_ingresos) == 4     # filas 12, 8, 4, 0


def test_serie_notas_declaran_lo_que_no_es_pit():
    e, df = _serie_sintetica()
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    texto = " ".join(s["notas"])
    assert "NO point-in-time" in texto and "Rf" in texto and "ERP" in texto and "beta" in texto
    assert "forward" in texto and "causal" in texto


def test_serie_nube_limita_a_3_anios_y_no_usa_cache_de_disco(tmp_path):
    e, df = _serie_sintetica()
    ruta = tmp_path / "datos" / cache_local.NOMBRE_ARCHIVO
    s = fv.serie_valor_justo_historica(e, df, modo="nube")
    assert s["modo"] == "nube" and s["anios"] == 3 and not ruta.exists()
    assert pd.Timestamp(s["fechas"][0]) >= pd.Timestamp(s["fechas"][-1]) - pd.DateOffset(years=3) - pd.Timedelta(days=1)
    loc = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    assert len(loc["fechas"]) > len(s["fechas"]) * 2 and not ruta.exists()


def test_serie_cache_caliente_en_local_menos_de_100ms():
    e, df = _serie_sintetica()
    ruta = os.path.join(os.environ["NIVELES_DATA_DIR"], cache_local.NOMBRE_ARCHIVO)
    fria = fv.serie_valor_justo_historica(e, df)
    assert os.path.exists(ruta) and fria["desde_cache"] is False
    t0 = time.perf_counter()
    caliente = fv.serie_valor_justo_historica(e, df)
    dt = time.perf_counter() - t0
    assert caliente["desde_cache"] is True and caliente["valor_justo"] == fria["valor_justo"] and dt < 0.25
    # otros parametros -> otra clave (no sirve la serie vieja)
    otra = fv.serie_valor_justo_historica(e, df, Par(wacc=0.11))
    assert otra["desde_cache"] is False and otra["valor_justo"][-1] != fria["valor_justo"][-1]
    # purgar por ticker invalida
    assert cache_local.purgar(ticker="SYN") >= 1
    assert fv.serie_valor_justo_historica(e, df)["desde_cache"] is False


def test_serie_sin_datos_devuelve_estructura_vacia_con_nota():
    assert fv.serie_valor_justo_historica(Ent(ticker="X"), pd.DataFrame())["notas"]
    e, df = _serie_sintetica()
    e.serie_ttm = []
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    assert s["puntos_trimestrales"] == [] and s["notas"] and all(len(v) == len(s["fechas"]) for v in s["bandas"].values())
    assert not math.isnan(sum(v for v in s["valor_justo"] if v is not None) if s["valor_justo"] else 0.0)


def test_suavizar_causal_es_monotono_continuo_y_causal():
    ev = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-15"), 100.0),
          (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-16"), 160.0),
          (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-16"), 120.0)]
    g = pd.date_range("2023-12-01", "2024-09-01", freq="D")
    v = fv.suavizar_causal(ev, g)
    assert np.isnan(v[0]) and v[g.get_loc(pd.Timestamp("2024-01-01"))] == pytest.approx(100.0)
    assert v[g.get_loc(pd.Timestamp("2024-03-15"))] == pytest.approx(100.0)               # meseta
    assert v[g.get_loc(pd.Timestamp("2024-05-16"))] == pytest.approx(160.0)               # llega al nuevo valor
    tramo = v[g.get_loc(pd.Timestamp("2024-04-01")):g.get_loc(pd.Timestamp("2024-05-16")) + 1]
    assert (np.diff(tramo) >= -1e-12).all() and tramo.max() <= 160.0 + 1e-9               # monotono, sin overshoot
    # causal: cambiar el ultimo evento no altera nada anterior a su inicio
    ev2 = ev[:2] + [(ev[2][0], ev[2][1], 999.0)]
    v2 = fv.suavizar_causal(ev2, g)
    antes = g < pd.Timestamp("2024-06-01")
    assert np.allclose(v[antes & ~np.isnan(v)], v2[antes & ~np.isnan(v2)])


def test_depurar_acciones_corrige_un_glitch_de_split_sin_tocar_cambios_graduales():
    idx = pd.date_range("2019-01-31", periods=6, freq="QE")
    s = pd.Series([18.8e9, 18.6e9, 4.6e9, 18.5e9, 18.0e9, 17.6e9], index=idx)      # un trimestre sin ajustar (x1/4)
    d = fv._depurar_acciones(s)
    assert d.iloc[2] == pytest.approx((18.6e9 + 18.5e9) / 2) and list(d.iloc[[0, 1, 3, 4, 5]]) == list(s.iloc[[0, 1, 3, 4, 5]])
    recompras = pd.Series([100.0, 97.0, 94.0, 91.0], index=idx[:4])
    assert list(fv._depurar_acciones(recompras)) == list(recompras)
    split_real = pd.Series([100.0, 100.0, 400.0, 400.0], index=idx[:4])             # un split legitimo (vecinas distintas)
    assert list(fv._depurar_acciones(split_real)) == list(split_real)


def test_serie_ultimo_reporte_reciente_reparte_el_tramo_con_el_ancla():
    e, df = _serie_sintetica(fin="2026-01-30")
    e.serie_ttm = e.serie_ttm[:-1]                  # el ultimo cierre disponible: 2025-09-30 (publicado 2025-11-14)
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    assert s["valor_justo"][-1] == pytest.approx(fv.valor_justo(e)["fair_value"], rel=1e-6)
    ultimo = s["puntos_trimestrales"][-1]
    assert ultimo["fecha_cierre"] == "2025-09-30" and np.isfinite(s["valor_justo"]).all()


# ===================================================================================== datos absurdos (caso KO 2017-18)
def test_depurar_acciones_reescala_un_tramo_largo_en_millones():
    idx = pd.date_range("2016-03-31", periods=10, freq="QE")
    s = pd.Series([4.36e9, 4.36e9, 4.34e9] + [4334.0, 4327.0, 4320.0, 4313.0] + [4.30e9, 4.29e9, 4.28e9], index=idx)
    d = fv._depurar_acciones(s)
    assert d.iloc[3] == pytest.approx(4.334e9) and d.iloc[6] == pytest.approx(4.313e9)
    assert list(d.iloc[[0, 1, 2, 7, 8, 9]]) == list(s.iloc[[0, 1, 2, 7, 8, 9]])
    # un valor fuera de escala que no es potencia de 1000 no se reescala por esta via
    raro = pd.Series([4.0e9, 4.0e9, 4.0e9, 3.0e5, 4.0e9, 4.0e9], index=idx[:6])
    assert fv._depurar_acciones(raro).iloc[3] == pytest.approx(4.0e9)       # lo corrige el filtro de vecinas


def test_corregir_q4_anual_resta_los_tres_trimestres_previos():
    idx = pd.date_range("2015-03-31", periods=8, freq="QE")
    # trimestres 10, 11, 9, 10 ... y el ultimo "trimestre" es el anual sin diferenciar: 11+9+10 (previos) + 10 (Q4 real) = 40
    q = pd.Series([10.0, 11.0, 9.0, 10.0, 10.0, 11.0, 9.0, 40.0], index=idx)
    fix = fv._corregir_q4_anual(q)
    assert fix.iloc[7] == pytest.approx(40.0 - (10.0 + 11.0 + 9.0)) and list(fix.iloc[:7]) == list(q.iloc[:7])    # 10
    sano = pd.Series([10.0, 11.0, 9.0, 10.0, 10.0, 11.0, 9.0, 10.0], index=idx)
    assert list(fv._corregir_q4_anual(sano)) == list(sano)


def test_serie_descarta_trimestres_con_acciones_absurdas_y_deja_nota():
    e, df = _serie_sintetica()
    for i in (20, 21):
        e.serie_ttm[i]["acciones"] = 1e-3 * 100.0          # acciones ~ 0 en dos trimestres
    e.serie_ttm[25]["acciones"] = 1e8                        # escala erronea x1e6
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    cierres = {p["fecha_cierre"] for p in s["puntos_trimestrales"]}
    for i in (20, 21, 25):
        assert e.serie_ttm[i]["fecha"] not in cierres
    assert s["puntos_descartados"] == 3 and any("descartado" in n for n in s["notas"])
    vj = np.array(s["valor_justo"]); pr = np.array([p for p in s["precio"]], dtype=float)
    assert np.all(vj / pr < fv.VJ_ABSURDO_X) and np.all(vj / pr > 1 / fv.VJ_ABSURDO_X)


def test_serie_descarta_valor_justo_absurdo_frente_al_precio():
    e, df = _serie_sintetica()
    r = e.serie_ttm[15]
    for k in ("fcf", "ebitda", "ebit", "ing", "eps", "neto", "ocf"):
        r[k] = r[k] * 1e4                                   # fundamentales x10000 en un solo trimestre
    s = fv.serie_valor_justo_historica(e, df, usar_cache=False)
    assert r["fecha"] not in {p["fecha_cierre"] for p in s["puntos_trimestrales"]} and s["puntos_descartados"] >= 1
    assert max(s["valor_justo"]) < 10 * max(p for p in s["precio"] if p)
