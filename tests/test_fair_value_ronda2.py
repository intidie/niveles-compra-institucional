"""Ronda 2 de correcciones (revision financiera de fair_value.py). Casos numericos a mano; sin red."""
import numpy as np
import pandas as pd
import pytest

import app_ui as ui_mod
from src.levels import fair_value as fv
from src.levels.fair_value import EntradasValoracion as Ent
from src.levels.fair_value import ParametrosValoracion as Par
from test_fair_value import (_df_precios, _entradas_ricas, _facts_falsos, _sin_red_ni_disco, cliente, ent,  # noqa: F401
                             modelo, px, vista)


def _textos(cliente_):
    from nicegui import ui
    return [e.text for e in cliente_.elements.values() if isinstance(e, ui.label)]


# ================================================================ A1: FCFF (sin doble conteo de intereses)
def test_dcf_fcf_suma_los_intereses_despues_de_impuestos():
    # FCFE 105 + intereses 60*(1-0.25) = 150 (FCFF) ; valor = 150*1.02/(0.09-0.02) * 1.09^0.5 (mitad de anio) ;
    # - deuda 1000 ; / 10 (sin el factor de mitad de anio seria 118.5714)
    e = ent(fcf_ltm=105.0, deuda_total=1000.0, gasto_interes_ltm=60.0)
    esperado = (150.0 * 1.02 / 0.07 * 1.09 ** 0.5 - 1000.0) / 10.0
    assert esperado == pytest.approx(128.19527, abs=1e-4)
    assert px(e, "dcf_5y_fcf", Par(wacc=0.09, g=0.02)) == pytest.approx(esperado, rel=1e-9)
    assert px(e, "dcf_10y_fcf", Par(wacc=0.09, g=0.02)) == pytest.approx(esperado, rel=1e-9)


def test_dcf_fcf_sin_gasto_de_intereses_no_ajusta_y_p_fcf_no_cambia():
    e = ent(fcf_ltm=105.0, deuda_total=1000.0)
    assert px(e, "dcf_5y_fcf", Par(wacc=0.09, g=0.02)) == pytest.approx((105 * 1.02 / 0.07 * 1.09 ** 0.5 - 1000) / 10,
                                                                        rel=1e-9)
    con = ent(fcf_ltm=105.0, deuda_total=1000.0, gasto_interes_ltm=60.0)
    assert px(con, "p_fcf", Par(p_fcf_objetivo=12.0)) == pytest.approx(105 / 10 * 12.0)   # multiplo de patrimonio: sin ajuste


def test_dcf_revenue_tambien_usa_margen_fcff():
    e = ent(ingresos_ltm=1000.0, fcf_ltm=105.0, deuda_total=1000.0, gasto_interes_ltm=60.0)
    assert px(e, "dcf_5y_revenue", Par(wacc=0.09, g=0.02)) == pytest.approx((150 * 1.02 / 0.07 * 1.09 ** 0.5 - 1000) / 10,
                                                                            rel=1e-9)


def test_dcf_fcf_ya_no_coincide_con_ebitda_solo_por_los_intereses():
    # caso del auditor: EBIT 200, t 25 %, deuda 1000 al 6 %: FCFF = 200*0.75 = 150 = el FCF apalancado 105 + 60*0.75
    fcff = ent(fcf_ltm=105.0, deuda_total=1000.0, gasto_interes_ltm=60.0, ebitda_ltm=200.0, da_ltm=0.0, capex_ltm=0.0)
    p = Par(wacc=0.09, g=0.02)
    assert px(fcff, "dcf_5y_fcf", p) == pytest.approx(px(fcff, "dcf_5y_ebitda", p), rel=1e-9)


# ================================================================ A2: es_financiera
@pytest.mark.parametrize("ticker, industria, esperado", [
    ("COF", "Credit Services", True), ("AXP", "Credit Services", True), ("SYF", "Credit Services", True),
    ("V", "Credit Services", False), ("MA", "Credit Services", False), ("PYPL", "Credit Services", False),
    ("FI", "Credit Services", False), ("JPM", "Banks - Diversified", True), ("GS", "Capital Markets", True),
    ("BRK-B", "Insurance - Diversified", True), ("AAPL", "Consumer Electronics", False)])
def test_es_financiera_prestamistas_si_procesadores_de_pago_no(ticker, industria, esperado):
    sector = "Technology" if ticker == "AAPL" else "Financial Services"
    assert fv.es_financiera(sector, industria, ticker) is esperado


def test_prestamista_de_credito_pierde_los_dcf_y_el_procesador_los_conserva():
    base = dict(precio=100.0, acciones_diluidas=10.0, caja=0.0, deuda_total=0.0, fcf_ltm=100.0, ebitda_ltm=200.0,
                da_ltm=10.0, capex_ltm=10.0, ingresos_ltm=1000.0, ebit_ltm=150.0, sector="Financial Services",
                industria="Credit Services", tasa_impositiva=0.25)
    cof = Ent(ticker="COF", **base)
    v = Ent(ticker="V", **base)
    p = Par(wacc=0.10, g=0.02)
    assert not modelo(cof, "dcf_5y_fcf", p)["aplicable"] and "bancos" in modelo(cof, "dcf_5y_fcf", p)["motivo_exclusion"]
    assert modelo(v, "dcf_5y_fcf", p)["aplicable"]


# ================================================================ A3: Graham y P/B no informativo
def test_graham_no_aplica_con_pb_actual_alto_o_pb_historico_no_informativo():
    ok = ent(eps_ltm=4.0, bvps=10.0)             # P/B actual 10x
    assert px(ok, "graham") == pytest.approx(30.0)
    alto = modelo(ent(eps_ltm=4.0, bvps=100.0 / (fv.GRAHAM_PB_MAX + 1)), "graham")
    assert not alto["aplicable"] and "P/B actual" in alto["motivo_exclusion"]
    borde = ent(eps_ltm=4.0, bvps=100.0 / fv.GRAHAM_PB_MAX)   # justo en el limite: aplica
    assert modelo(borde, "graham")["aplicable"]
    hist = modelo(ent(eps_ltm=4.0, bvps=10.0, multiplos_hist={"pb": [fv.PB_NO_INFORMATIVO + 5] * 4}), "graham")
    assert not hist["aplicable"] and "P/B historico mediano" in hist["motivo_exclusion"]


# ================================================================ A4: acciones y EBIT de respaldo
def _con_dei(facts, valores):
    fecha = facts["period_end"].max()
    extra = pd.DataFrame([dict(concept="dei:EntityCommonStockSharesOutstanding", period_type="instant",
                               period_start=pd.NaT, period_end=fecha, numeric_value=v, fiscal_year=fecha.year,
                               fiscal_period="Q9", dur=np.nan, ref_filing=fecha) for v in valores])
    return pd.concat([facts, extra], ignore_index=True)


_DIL = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
_BAS = "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"


def test_acciones_diluidas_manda_si_existe():
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_con_dei(_facts_falsos(), [8.0]))
    assert e.acciones_diluidas == 10.0 and e.acciones_nota is None and "diluido" in e.fuentes["acciones_diluidas"]


def test_acciones_desde_la_portada_si_falta_el_promedio_diluido():
    facts = _con_dei(_facts_falsos(flujos={_DIL: None, _BAS: 9.0}), [8.0])
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=facts)
    assert e.acciones_diluidas == 8.0 and "portada" in e.fuentes["acciones_diluidas"] and e.acciones_nota
    assert e.bvps == pytest.approx(250.0 / 8.0)


def test_acciones_basicas_si_no_hay_diluido_ni_portada_y_portada_ambigua_se_ignora():
    sin_dei = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(flujos={_DIL: None, _BAS: 9.0}))
    assert sin_dei.acciones_diluidas == 9.0 and "basico" in sin_dei.fuentes["acciones_diluidas"]
    dos_clases = _con_dei(_facts_falsos(flujos={_DIL: None, _BAS: 9.0}), [8.0, 3.0])   # varias clases: ambiguo
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=dos_clases)
    assert e.acciones_diluidas == 9.0


def test_historia_de_acciones_rellena_con_el_basico_y_los_multiplos_siguen_calculandose():
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(flujos={_DIL: None, _BAS: 10.0}))
    assert len(e.multiplos_hist["p_fcf"]) == 6 and all(v == pytest.approx(10.0) for v in e.multiplos_hist["p_fcf"])


def test_ebit_de_respaldo_desde_utilidad_antes_de_impuestos_mas_intereses():
    # sin OperatingIncomeLoss: pretax 20/trim * 4 = 80 ; intereses 2/trim * 4 = 8 -> EBIT 88
    e = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos(
        flujos={"us-gaap:OperatingIncomeLoss": None}))
    assert e.ebit_ltm == pytest.approx(88.0) and "utilidad antes de impuestos" in e.fuentes["ebit_ltm"]
    assert e.ebitda_ltm == pytest.approx(88.0 + 40.0)
    normal = fv.extraer_entradas("T", None, _df_precios(), {}, facts=_facts_falsos())
    assert normal.ebit_ltm == 160.0 and "OperatingIncomeLoss" in normal.fuentes["ebit_ltm"]


# ================================================================ A5: agregacion
def _lista(valores, familias=("DCF", "Múltiplos", "Valor", "Dividendos")):
    return [{"id": f"m{i}", "nombre": f"m{i}", "familia": familias[i % len(familias)], "precio_estimado": v,
             "aplicable": True, "motivo_exclusion": None} for i, v in enumerate(valores)]


def test_iqr_con_modelos_casi_identicos_no_marca_outliers_por_un_1_pct():
    a = fv.agregar_valor_justo(_lista([100.0, 100.0, 100.0, 100.0, 101.0, 99.0]), None, 0.25, 100.0)
    assert a["modelos_outlier"] == [] and a["n_usados"] == 6
    b = fv.agregar_valor_justo(_lista([100.0, 100.0, 100.0, 100.0, 150.0, 99.0]), None, 0.25, 100.0)
    assert b["modelos_outlier"] == ["m4"]     # un 50 % mas si es outlier (ancho minimo 5 % de la mediana = 5)
    assert fv.IQR_PISO_REL == 0.05


def test_margen_de_seguridad_acotado():
    lista = _lista([100.0, 100.0, 100.0])
    assert fv.agregar_valor_justo(lista, None, 1.5, 100.0)["margen_seguridad"] == fv.MARGEN_SEGURIDAD_MAX
    a = fv.agregar_valor_justo(lista, None, 1.5, 100.0)
    assert a["margin_of_safety_price"] == pytest.approx(100.0 * (1 - fv.MARGEN_SEGURIDAD_MAX))
    b = fv.agregar_valor_justo(lista, None, -0.3, 100.0)
    assert b["margen_seguridad"] == 0.0 and b["margin_of_safety_price"] == pytest.approx(100.0)


def test_un_id_no_esta_a_la_vez_desactivado_y_no_aplicable():
    lista = _lista([100.0, 100.0, 100.0]) + [{"id": "na", "nombre": "na", "familia": "DCF", "precio_estimado": None,
                                               "aplicable": False, "motivo_exclusion": "x"}]
    a = fv.agregar_valor_justo(lista, ["m0", "m1"], 0.25, 100.0)     # "na" y "m2" fuera de la seleccion
    assert a["modelos_no_aplicables"] == ["na"] and a["modelos_desactivados"] == ["m2"]
    assert not set(a["modelos_no_aplicables"]) & set(a["modelos_desactivados"])


def test_familias_usadas_cv_prefiltro_y_degradacion_de_baja():
    tres = fv.agregar_valor_justo(_lista([100.0, 101.0, 99.0]), None, 0.25, 100.0)   # DCF, Multiplos, Valor: 3 grupos
    assert tres["familias_usadas"] == ["DCF", "Múltiplos", "Valor / Dividendos"] and tres["incertidumbre"] == "Baja"
    dos = fv.agregar_valor_justo(_lista([100.0, 101.0, 99.0, 100.0], familias=("Múltiplos", "Valor", "Dividendos", "Valor")),
                                 None, 0.25, 100.0)        # Multiplos + Valor/Dividendos: 2 grupos
    assert dos["familias_usadas"] == ["Múltiplos", "Valor / Dividendos"] and dos["cv"] < fv.CV_BAJO
    assert dos["incertidumbre"] == "Media"                  # CV bajo pero de pocas familias: falsa tranquilidad
    assert fv.MIN_FAMILIAS_BAJA == 3
    assert dos["cv_prefiltro"] == pytest.approx(dos["cv"])    # sin outliers coinciden
    con_out = fv.agregar_valor_justo(_lista([100.0, 100.0, 100.0, 100.0, 400.0]), None, 0.25, 100.0)
    assert con_out["cv"] == 0.0 and con_out["cv_prefiltro"] > fv.CV_ALTO      # el outlier se ve en el CV previo
    assert fv.clasificar_incertidumbre(0.05, 4, 2) == "Media" and fv.clasificar_incertidumbre(0.05, 4, 3) == "Baja"
    assert fv.clasificar_incertidumbre(0.05, 2, 3) == "Alta" and fv.clasificar_incertidumbre(0.4, 4, 1) == "Alta"


def test_sin_familia_en_los_resultados_no_se_degrada():
    sin = [{"id": f"x{i}", "nombre": "x", "precio_estimado": 100.0, "aplicable": True, "motivo_exclusion": None}
           for i in range(3)]
    a = fv.agregar_valor_justo(sin, None, 0.25, 100.0)
    assert a["familias_usadas"] == [] and a["incertidumbre"] == "Baja"


# ================================================================ A6 / A7: tasas
def test_tasa_efectiva_tiene_piso():
    e = ent(tasa_impositiva=0.0048, ebit_ltm=100.0)
    assert fv.calcular_tasas(e, Par())["tasa_impositiva"] == fv.TASA_MIN == 0.15
    assert px(e, "epv", Par(wacc=0.10)) == pytest.approx(100 * 0.85 / 0.10 / 10)     # NOPAT con 15 %, no con 0,48 %
    alta = ent(tasa_impositiva=0.9)
    assert fv.calcular_tasas(alta, Par())["tasa_impositiva"] == fv.TASA_IMPOSITIVA_MAX
    assert fv.calcular_tasas(ent(tasa_impositiva=None), Par())["tasa_impositiva"] == fv.TASA_IMPOSITIVA_DEFECTO


def test_pisos_de_beta_y_wacc_y_rf_por_defecto():
    assert (fv.WACC_MIN, fv.WACC_MAX, fv.RF_DEFECTO, fv.ERP_DEFECTO) == (0.08, 0.115, 0.0425, 0.045)
    # beta 0.3 -> Blume 2/3*0.3 + 1/3 = 0.5333 ; ke = 0.04 + 0.5333*0.05 = 0.066667 < piso: WACC = 0.08 (piso Medio)
    t = fv.calcular_tasas(ent(beta=0.3), Par(rf=0.04, erp=0.05))
    assert t["beta"] == pytest.approx(0.5333333) and t["ke_capm"] == pytest.approx(0.04 + 0.5333333 * 0.05)
    assert t["wacc"] == 0.08 and any("WACC" in x for x in t["topes_wacc"])
    t2 = fv.calcular_tasas(ent(beta=0.7), Par(rf=0.03, erp=0.04))    # ke bajo el piso: WACC acotado
    assert t2["wacc"] == fv.WACC_MIN and any("WACC" in x for x in t2["topes_wacc"])
    assert fv.calcular_tasas(ent(beta=1.0), Par(rf=0.04, erp=0.05))["topes"] == []


# ================================================================ A8: flujo ciclico
def test_es_ciclico():
    assert fv.es_ciclico([23.4, 59.6, 37.8, 32.8, 23.8, 43.0])         # XOM: sube, baja, sube; CV 0.37
    assert not fv.es_ciclico([8.1, 9.5, 46.8, 72.0, 127.0])             # NVDA: tendencia (CV alto, sin inversiones)
    assert not fv.es_ciclico([93.0, 111.4, 99.6, 108.8, 98.8, 136.1])    # AAPL: CV 0.14
    assert not fv.es_ciclico([10.0, 20.0, 10.0])                          # < 4 puntos
    assert not fv.es_ciclico([-5.0, 3.0, -4.0, 6.0])                      # media <= 0
    assert fv.VOL_FCF_ALTA == 0.30 and fv.N_NORMALIZACION == 5


def test_dcf_fcf_ciclico_se_normaliza_a_la_media_y_crece_a_g():
    hist = [23.4, 59.6, 37.8, 32.8, 23.8, 43.0]
    e = ent(fcf_ltm=43.0, hist_fcf=hist)
    # base = media de los ultimos 5 = (59.6+37.8+32.8+23.8+43.0)/5 = 39.4 ; crecimiento = g -> Gordon exacto
    m = modelo(e, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))
    assert m["precio_estimado"] == pytest.approx(39.4 * 1.02 / 0.08 / 10 * 1.1 ** 0.5, rel=1e-9) and m["ciclico"] is True
    assert px(e, "dcf_10y_fcf", Par(wacc=0.10, g=0.02)) == pytest.approx(39.4 * 1.02 / 0.08 / 10 * 1.1 ** 0.5, rel=1e-9)
    sin_ciclo = ent(fcf_ltm=43.0, hist_fcf=[30.0, 33.0, 36.0, 39.0, 41.0, 43.0])
    assert modelo(sin_ciclo, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))["ciclico"] is False
    s = fv.supuestos(e, Par(wacc=0.10, g=0.02))
    assert s["flujos_ciclicos_normalizados"] == ["FCF"] and s["crecimiento_inicial_fcf"] == 0.02


# ================================================================ A9: pocos anios de FCF positivo
def test_dcf_fcf_exige_anios_de_fcf_positivo_si_hay_historia():
    corto = ent(fcf_ltm=0.5, hist_fcf=[-1.4, -1.9, -1.0, -0.5, -0.3, 0.5])      # un solo anio positivo
    m = modelo(corto, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))
    assert not m["aplicable"] and "historia insuficiente de FCF positivo" in m["motivo_exclusion"]
    tres = ent(fcf_ltm=0.5, hist_fcf=[-1.4, -1.9, -0.3, 0.1, 0.4, 0.5])         # tres positivos: aplica
    assert modelo(tres, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))["aplicable"] and fv.MIN_ANIOS_FCF == 3
    sin_hist = ent(fcf_ltm=100.0)                                                # sin historia: g (comportamiento previo)
    assert modelo(sin_hist, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))["aplicable"]
    assert not modelo(ent(fcf_ltm=100.0, hist_fcf=[100.0, 110.0]), "dcf_10y_fcf")["aplicable"]   # solo 2 anios de historia


# ================================================================ B1: complementarias
def test_complementarias_mediana_submedias_y_promedio_igualando_familias():
    lista = [{"id": "d1", "familia": "DCF", "precio_estimado": 100.0, "aplicable": True},
             {"id": "d2", "familia": "DCF", "precio_estimado": 120.0, "aplicable": True},
             {"id": "m1", "familia": "Múltiplos", "precio_estimado": 200.0, "aplicable": True},
             {"id": "v1", "familia": "Valor", "precio_estimado": 60.0, "aplicable": True},
             {"id": "dd", "familia": "Dividendos", "precio_estimado": 40.0, "aplicable": True}]
    agg = fv.agregar_valor_justo(lista, None, 0.25, 100.0)
    c = fv.complementarias(agg, lista)
    assert c["mediana"] == 100.0 and c["submedias"] == {"DCF": 110.0, "Múltiplos": 200.0, "Valor / Dividendos": 50.0}
    assert c["promedio_familias"] == pytest.approx((110 + 200 + 50) / 3) and c["n_por_grupo"]["DCF"] == 2
    assert c["upside_mediana"] == pytest.approx(0.0) and c["upside_promedio_familias"] == pytest.approx(20.0)
    # el consolidado ES el promedio igualando familias (110+200+50)/3 = 120; el promedio simple (520/5 = 104) es informativo
    assert agg["fair_value"] == pytest.approx(120.0) and agg["promedio_simple"] == pytest.approx(104.0)
    assert c["promedio_simple"] == pytest.approx(104.0)
    assert fv.complementarias({"modelos_usados": []}, lista) is None


# ================================================================ B2: DCF inverso
def test_dcf_inverso_encuentra_el_wacc_que_iguala_el_precio():
    e = ent(fcf_ltm=100.0)       # sin historia: crecimiento = g ; valor = 100*(1+g)/(w-g)*(1+w)^0.5/10 (mitad de anio)
    p = Par(wacc=0.09, g=0.02)
    di = fv.dcf_inverso(e, p, precio=102 * 1.10 ** 0.5 / 0.08 / 10)     # el precio que da exactamente w = 0.10
    assert di["wacc_implicito"] == pytest.approx(0.10, abs=1e-6) and di["wacc_usado"] == 0.09
    assert di["valor_con_supuestos"] == pytest.approx(102 * 1.09 ** 0.5 / 0.07 / 10)
    assert fv.dcf_inverso(e, p, precio=102 * 1.08 ** 0.5 / 0.06 / 10)["wacc_implicito"] == pytest.approx(0.08, abs=1e-6)
    assert fv.dcf_inverso(e, p, precio=1e9)["wacc_implicito"] is None and fv.dcf_inverso(e, p, precio=1e-9)["wacc_implicito"] is None


def test_dcf_inverso_encuentra_el_crecimiento_inicial_que_iguala_el_precio():
    e = ent(fcf_ltm=100.0)
    p = Par(wacc=0.10, g=0.02)
    objetivo = px(e, "dcf_10y_fcf", p.con(crecimiento_inicial=0.07))
    di = fv.dcf_inverso(e, p, precio=objetivo)
    assert di["g0_implicito"] == pytest.approx(0.07, abs=1e-6) and di["g0_usado"] == 0.02
    assert fv.dcf_inverso(e, p, precio=1e12)["g0_implicito"] is None


def test_dcf_inverso_solo_para_dcf_de_fcf_y_sin_datos():
    e = ent(fcf_ltm=100.0, ebitda_ltm=100.0)
    assert fv.dcf_inverso(e, None, precio=100.0, modelo="dcf_10y_ebitda")["wacc_implicito"] is None
    assert fv.dcf_inverso(Ent(), None, precio=100.0)["wacc_implicito"] is None
    assert fv.dcf_inverso(e, None, precio=None)["precio"] == 100.0       # usa el precio de las entradas
    assert fv.dcf_inverso(e, None, precio=0)["wacc_implicito"] is None


# ================================================================ B3: topes activos
def test_topes_activos_en_supuestos_y_en_los_modelos():
    e = ent(eps_ltm=2.0, fcf_ltm=1000.0, ingresos_ltm=500.0, beta=5.0,
            multiplos_hist={"pe": [50.0, 52.0, 48.0, 51.0]},
            hist_fcf=[10.0, 40.0, 160.0, 640.0, 1000.0], hist_ingresos=[20.0, 40.0, 60.0, 100.0, 150.0])
    s = fv.supuestos(e, Par(wacc=0.10, g=0.02))
    t = " | ".join(s["topes_activos"])
    assert "P/E objetivo: mediana histórica 50.5x topada a 35x" in t
    assert "FCF: crecimiento etapa 1" in t and "topado a 42 %" in t      # variaciones +100 % -> 65 % -> tope Medio 42 %
    assert "margen FCF objetivo" in t and "beta 5.00 acotada a 4.00" in t
    pe = modelo(e, "pe")
    assert pe["tope_activo"] is True and "topada a 35x" in pe["topes"][0]
    assert modelo(e, "dcf_5y_fcf", Par(wacc=0.10, g=0.02))["tope_activo"] is True
    assert modelo(ent(eps_ltm=2.0), "graham")["tope_activo"] is False
    lim = fv.RANGOS_MULTIPLOS["pe"]
    en_rango = ent(eps_ltm=2.0, multiplos_hist={"pe": [(lim[0] + lim[1]) / 2] * 4})
    assert modelo(en_rango, "pe")["tope_activo"] is False and fv.supuestos(en_rango)["topes_activos"] == []


def test_tope_de_lynch_y_ddm_con_beta_acotada():
    e = ent(eps_ltm=2.0, hist_eps=[1.0, 10.0], dividendo_anual=3.0, beta=5.0)
    assert modelo(e, "lynch")["tope_activo"] is True
    assert modelo(e, "ddm")["tope_activo"] is True and "beta" in modelo(e, "ddm")["topes"][0]


# ================================================================ B4: soporte debil
def test_nota_de_soporte_y_etiqueta():
    assert ui_mod.nota_soporte("Alta", 10) == "incert. Alta" and ui_mod.nota_soporte("Media", 2) == "n=2"
    assert ui_mod.nota_soporte("Alta", 2) == "n=2" and ui_mod.nota_soporte("Baja", 5) is None
    assert ui_mod.nota_soporte(None, None) is None
    assert ui_mod.etiqueta_fair_value(100.0, 10.0, "incert. Alta") == "Fair Value: $100.00 (+10% Upside) · incert. Alta"
    assert ui_mod.etiqueta_fair_value(100.0, 10.0) == "Fair Value: $100.00 (+10% Upside)"


def test_linea_del_grafico_indica_soporte_debil(df_sintetico):
    px_ = float(df_sintetico["Close"].iloc[-1])
    datos = ui_mod.datos_grafico(df_sintetico, "T")
    debil = ui_mod.overlays_grafico(datos, "T", [], px_, valor_justo={
        "fair_value": px_ * 1.2, "upside_pct": 20.0, "incertidumbre": "Alta", "n_usados": 8})
    assert debil["vj"]["etiqueta"].endswith(" · incert. Alta")
    fuerte = ui_mod.overlays_grafico(datos, "T", [], px_, valor_justo={
        "fair_value": px_ * 1.2, "upside_pct": 20.0, "incertidumbre": "Media", "n_usados": 8})
    assert "·" not in fuerte["vj"]["etiqueta"]


def test_ranking_marca_el_upside_con_poco_soporte():
    base = {"ticker": "A", "precio_actual": 10.0, "zonas": []}
    filas, _ = ui_mod.filas_ranking([
        {**base, "valor_justo": {"upside_pct": 30.0, "incertidumbre": "Alta", "n_usados": 9}},
        {**base, "ticker": "B", "valor_justo": {"upside_pct": 10.0, "incertidumbre": "Baja", "n_usados": 9}},
        {**base, "ticker": "C", "valor_justo": {"upside_pct": 5.0, "incertidumbre": "Media", "n_usados": 2}},
        {**base, "ticker": "D"}])
    por = {f["ticker"]: f for f in filas}
    assert por["A"]["upside_nota"] == "incert. Alta" and por["B"]["upside_nota"] is None
    assert por["C"]["upside_nota"] == "n=2" and por["D"]["upside_nota"] is None and por["D"]["upside"] is None


# ================================================================ B6: ke con WACC del usuario y deuda
def test_wacc_del_usuario_mueve_ke_dividido_por_el_peso_del_patrimonio():
    e = ent(deuda_total=1000.0, gasto_interes_ltm=60.0, beta=1.0)      # 50 % deuda, 50 % patrimonio
    auto = fv.calcular_tasas(e, Par(rf=0.04, erp=0.08))                # WACC 0.0825 ; ke 0.12
    t = fv.calcular_tasas(e, Par(rf=0.04, erp=0.08, wacc=auto["wacc"] + 0.01))
    assert t["ke"] - t["ke_capm"] == pytest.approx(0.01 / 0.5)         # 1 pp de WACC = 2 pp de ke
    sin_deuda = fv.calcular_tasas(ent(beta=1.0), Par(rf=0.04, erp=0.08, wacc=0.10))
    assert sin_deuda["ke"] - sin_deuda["ke_capm"] == pytest.approx(0.10 - sin_deuda["wacc_auto"])


# ================================================================ UI: bloques nuevos
def test_vista_muestra_complementarias_dcf_inverso_topes_y_notas(cliente, vista):
    textos = _textos(cliente)
    comp = next(t for t in textos if t.startswith("Complementarias del Escenario Medio"))
    assert "Mediana $" in comp and "Promedio simple $" in comp and "DCF $" in comp and "Múltiplos $" in comp
    assert "iguala el peso de las familias" in comp
    assert any("no son independientes" in t for t in textos)
    assert any(t.startswith("DCF inverso") for t in textos)
    assert any("se mantienen al cambiar de ticker" in t and "«Todos»" in t for t in textos)
    assert any("Solo DCF" in t and "visiones distintas" in t for t in textos)
    assert all(f["tope"].text == "" or f["tope"].text.startswith("tope activo: ") for f in vista.vj_rows.values())
    vista.vj_params = vista.vj_params.con(wacc=0.12)
    vista._vj_actualizar()
    assert any("WACC implícito" in t for t in _textos(cliente))


def test_vista_marca_tope_activo_en_la_fila(vista):
    e = _entradas_ricas()
    e.multiplos_hist = {**e.multiplos_hist, "pe": [90.0] * 5}      # P/E histórico 90x: tope a 35x
    ui_mod.CACHE["RICA"] = {**ui_mod.CACHE["RICA"], "entradas_valoracion": e.to_dict()}
    vista.refrescar()
    assert vista.vj_rows["pe"]["tope"].text.startswith("tope activo: ") and "topada a 35x" in vista.vj_rows["pe"]["tope"].text
    assert vista.vj_rows["graham"]["tope"].text == ""
    assert any("Topes activos" in t for t in [l.text for l in vista.vj_supuestos.default_slot.children])


def test_vista_dcf_inverso_texto_segun_el_signo(cliente, vista):
    di = fv.dcf_inverso(vista.vj_entradas, vista.vj_params, 100.0)
    lineas = vista._vj_lineas_dcf_inverso()
    assert len(lineas) == 1 and lineas[0].startswith("DCF inverso")
    if di["wacc_implicito"] is not None and di["wacc_implicito"] < di["wacc_usado"]:
        assert "menos riesgo o más crecimiento" in lineas[0]
    else:
        assert "más riesgo o menos crecimiento" in lineas[0] or "menos riesgo" in lineas[0]


def test_seccion_modelo_textos_corregidos():
    txt = ui_mod.texto_seccion(next(s for s in ui_mod.contenido_modelo() if s["id"] == "valor_justo"))
    assert "NO son «conservadoras»" in txt.replace("**", "") or "NO son «conservadoras»" in txt
    for clave in ("no son independientes", "Ciclos y commodities", "REITs y utilities", "SBC", "arrendamientos",
                  "tope activo", "mitad de año", "Poco soporte", "Otros DCF del proyecto", "visiones distintas",
                  "promedio que iguala el peso de cada familia"):
        assert clave.lower() in txt.lower(), clave
    assert str(int(fv.GRAHAM_PB_MAX)) in txt and str(fv.MIN_ANIOS_FCF) in txt
