"""Pestana Valor Justo (Apache ECharts): opciones de los graficos, KPIs, controles con recalculo en memoria, serie historica
fuera del hilo de eventos, resize al cambiar de pestana y modo local vs nube. Sin red ni servidor."""
import asyncio
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import app_ui as ui_mod
from src.levels import fair_value as fv
from test_app_ui import contraste
from test_fair_value import (_entradas_ricas, _sin_red_ni_disco, cliente, vista)  # noqa: F401

RAIZ = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------------------------- datos sinteticos
def _serie(n=60, v0=100.0, paso=0.5, px0=95.0):
    """Serie historica sintetica con el mismo formato que ``fv.serie_valor_justo_historica``."""
    import pandas as pd
    fechas = [str(d.date()) for d in pd.date_range("2024-01-05", periods=n, freq="7D")]
    vj = [round(v0 + paso * i, 4) for i in range(n)]
    pf = [str((pd.Timestamp(fechas[-1]) + pd.Timedelta(days=int(d))).date()) for d in range(0, 366, 7)]
    vp = [round(vj[-1] * (1.05) ** (d / 365.0), 4) for d in range(0, 366, 7)]
    pcts = (30, 20, 10, -10, -20, -30)
    fb = lambda p: f"{'+' if p > 0 else '-'}{abs(p)}%"  # noqa: E731
    return {"fechas": fechas, "valor_justo": vj, "precio": [round(px0 + i * 0.7, 4) for i in range(n)],
            "bandas": {fb(p): [round(v * (1 + p / 100), 4) for v in vj] for p in pcts},
            "proyeccion": {"fechas": pf, "valor_justo": vp, "g_anual": 0.05, "metodo": "x",
                           "bandas": {fb(p): [round(v * (1 + p / 100), 4) for v in vp] for p in pcts}},
            "ancla": {"fecha": fechas[-1], "valor_justo": vj[-1]}, "notas": ["nota de prueba"]}


def _esc(valor_medio=130.0, valor_pes=80.0):
    return {"medio": {"fair_value": valor_medio}, "pesimista": {"fair_value": valor_pes}, "modelos": []}


def _por_nombre(opc):
    return {s["name"]: s for s in opc["series"]}


# ============================================================================== opciones del grafico historico
def test_historico_siempre_las_mismas_10_series_con_o_sin_datos():
    lleno = ui_mod.opciones_historico(_serie(), _esc())
    vacio = ui_mod.opciones_historico(None, None, "Sin datos")
    nombres = ["-30%", "-20%", "-10%", "0%", "+10%", "+20%", "+30%", ui_mod.SERIE_VJ, ui_mod.SERIE_PROY, ui_mod.SERIE_PRECIO]
    assert [s["name"] for s in lleno["series"]] == nombres == [s["name"] for s in vacio["series"]]
    assert len(lleno["series"]) == len(vacio["series"]) == 10
    assert vacio["title"]["text"] == "Sin datos" and all(s["data"] == [] for s in vacio["series"])
    assert lleno["title"]["text"] == ""


def test_historico_bandas_exactas_mas_menos_10_20_30_por_ciento():
    serie = _serie()
    opc = ui_mod.opciones_historico(serie, _esc())
    s = _por_nombre(opc)
    n_h = len(serie["fechas"])
    # las areas apiladas: acumulado = banda absoluta (el primero es la banda -30 % y el resto son diferencias)
    acum = {}
    corriente = [0.0] * len(s["-30%"]["data"])
    for k in ("-30%", "-20%", "-10%", "0%", "+10%", "+20%", "+30%"):
        corriente = [c + (v[1] or 0.0) for c, v in zip(corriente, s[k]["data"])]
        acum[k] = corriente
    for k in ("-30%", "-20%", "-10%", "+10%", "+20%", "+30%"):
        esperado = serie["bandas"][k]
        assert acum[k][:n_h] == pytest.approx(esperado, abs=0.02), k
        proy = serie["proyeccion"]["bandas"][k]
        assert acum[k][n_h - 1:] == pytest.approx(proy, abs=0.02), k            # la proyeccion arranca en el ancla
    assert acum["0%"][:n_h] == pytest.approx(serie["valor_justo"], abs=0.02)
    # cada banda es +-10/20/30 % del valor justo EN ESA fecha
    vj = acum["0%"]
    for k, p in (("+10%", 1.10), ("+20%", 1.20), ("+30%", 1.30), ("-10%", 0.90), ("-20%", 0.80), ("-30%", 0.70)):
        assert acum[k] == pytest.approx([v * p for v in vj], rel=2e-3), k
    assert all(sr["stack"] == "bandas" for k, sr in s.items() if k in acum)
    # eje x comun: las 7 series apiladas comparten las mismas fechas (historia + proyeccion de 12 meses)
    fechas0 = [d for d, _ in s["+30%"]["data"]]
    assert all([d for d, _ in s[k]["data"]] == fechas0 for k in acum)
    assert fechas0[:n_h] == serie["fechas"] and fechas0[-1] == serie["proyeccion"]["fechas"][-1] and len(set(fechas0)) == len(fechas0)


def test_historico_degradados_cian_arriba_y_magenta_abajo_con_etiquetas_en_el_borde_derecho():
    s = _por_nombre(ui_mod.opciones_historico(_serie(), _esc()))
    for k in ("+10%", "+20%", "+30%"):
        stops = s[k]["areaStyle"]["color"]["colorStops"]
        assert s[k]["areaStyle"]["color"]["type"] == "linear" and len(stops) == 2
        assert all(ui_mod.con_alfa(ui_mod.VJ_CIAN, 0.0)[:14] == c["color"][:14] for c in stops)   # cian (r, g)
        assert s[k]["endLabel"]["formatter"] == k and s[k]["endLabel"]["color"] == ui_mod.VJ_CIAN
    for k in ("-10%", "-20%", "-30%"):
        assert s[k]["endLabel"]["formatter"] == k and s[k]["endLabel"]["color"] == ui_mod.VJ_MAGENTA
        if k != "-30%":
            assert s[k]["areaStyle"]["color"]["colorStops"][0]["color"].startswith("rgba(232,121,249")
    assert "areaStyle" not in s["-30%"] and "endLabel" not in s["0%"] and s["0%"]["lineStyle"]["width"] == 0
    # la intensidad crece alejandose del valor justo (efecto GuruFocus)
    alfa = lambda c: float(c["areaStyle"]["color"]["colorStops"][0]["color"].rstrip(")").split(",")[-1])  # noqa: E731
    assert alfa(s["+30%"]) > alfa(s["+10%"]) and alfa(s["-20%"]) > alfa(s["-10%"])


def test_historico_curva_continua_proyeccion_punteada_precio_y_piso():
    serie = _serie()
    esc = _esc(valor_pes=77.77)
    opc = ui_mod.opciones_historico(serie, esc)
    s = _por_nombre(opc)
    n_h = len(serie["fechas"])
    vj, pr, px = s[ui_mod.SERIE_VJ], s[ui_mod.SERIE_PROY], s[ui_mod.SERIE_PRECIO]
    assert vj["lineStyle"].get("type") in (None, "solid") and pr["lineStyle"]["type"] == "dotted"       # continua vs punteada
    assert [v for _, v in vj["data"][:n_h]] == pytest.approx(serie["valor_justo"], abs=0.01)
    assert all(v is None for _, v in vj["data"][n_h:])                                              # la curva no se extiende
    assert all(v is None for _, v in pr["data"][:n_h - 1]) and pr["data"][n_h - 1][1] == pytest.approx(serie["valor_justo"][-1], abs=0.01)
    assert pr["data"][-1][1] == pytest.approx(serie["proyeccion"]["valor_justo"][-1], abs=0.01)
    assert [v for _, v in px["data"][:n_h]] == pytest.approx(serie["precio"], abs=0.01) and all(v is None for _, v in px["data"][n_h:])
    assert px["lineStyle"]["color"] == ui_mod.COLOR_PRECIO and vj["lineStyle"]["color"] == ui_mod.COLOR_FAIR_VALUE
    ml = vj["markLine"]                                                                                # piso del Pesimista actual
    assert ml["data"] == [{"yAxis": 77.77}] and ml["lineStyle"]["color"] == ui_mod.VJ_PESIMISTA
    assert "Piso fundamental (Pesimista) $77.77" == ml["label"]["formatter"]
    sin_piso = ui_mod.opciones_historico(serie, {"medio": {}, "pesimista": {"fair_value": None}})
    assert _por_nombre(sin_piso)[ui_mod.SERIE_VJ]["markLine"]["data"] == []


def test_historico_dataZoom_slider_y_rueda_y_ejes():
    opc = ui_mod.opciones_historico(_serie(), _esc())
    tipos = [z["type"] for z in opc["dataZoom"]]
    assert tipos == ["inside", "slider"] and all(z["xAxisIndex"] == 0 for z in opc["dataZoom"])
    assert opc["xAxis"]["type"] == "time" and opc["yAxis"]["scale"] is True
    assert opc["grid"]["right"] >= 60                                         # sitio para las etiquetas +30 % ... -30 %
    assert opc["tooltip"]["trigger"] == "axis" and opc["legend"]["data"] == [ui_mod.SERIE_VJ, ui_mod.SERIE_PROY, ui_mod.SERIE_PRECIO]


def test_historico_tooltip_declara_fecha_precio_valor_justo_y_porcentaje():
    js = ui_mod.opciones_historico(_serie(), _esc())["tooltip"][":formatter"]
    for campo in ("Fecha:", "Precio:", "Valor Justo Medio", "% Sobre/Infravalorado", "Sobrevalorado", "Infravalorado"):
        assert campo in js, campo
    assert json.dumps(ui_mod.SERIE_PRECIO, ensure_ascii=False) in js and ui_mod.SERIE_VJ in js and ui_mod.SERIE_PROY in js
    assert "ensure" not in js and "__C__" not in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node no disponible")
def test_historico_tooltip_se_ejecuta_y_calcula_el_porcentaje_en_esa_fecha():
    js = ui_mod.opciones_historico(_serie(), _esc())["tooltip"][":formatter"]
    prog = ("const f = (%s);\n"
            "const ps = [{seriesName:'-30%%', value:['2025-03-07', 70]},"
            " {seriesName:%s, value:['2025-03-07', 110]}, {seriesName:%s, value:['2025-03-07', 100]}];\n"
            "console.log(f(ps));\n"
            "const ps2 = [{seriesName:%s, value:['2026-01-01', 120]}, {seriesName:%s, value:['2026-01-01', null]}];\n"
            "console.log(f(ps2));" % (js, json.dumps(ui_mod.SERIE_PRECIO), json.dumps(ui_mod.SERIE_VJ),
                                      json.dumps(ui_mod.SERIE_PROY), json.dumps(ui_mod.SERIE_PRECIO)))
    r = subprocess.run(["node", "-e", prog], capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    hist, proy = r.stdout.strip().split("\n", 1)[0], r.stdout.strip().split("\n", 1)[1]
    assert "Fecha:</b> 2025-03-07" in hist and "$110.00" in hist and "$100.00" in hist
    assert "+10.0 % (Sobrevalorado)" in hist                                  # 110 / 100 - 1
    assert "% Sobre/Infravalorado" not in proy and "proyección ilustrativa" in proy and "Precio:</b> n/d" in proy


def test_historico_sin_colores_fuera_de_la_paleta_y_alfa_derivado():
    assert ui_mod.con_alfa("#22d3ee", 0.5) == "rgba(34,211,238,0.50)"
    raw = json.dumps(ui_mod.opciones_historico(_serie(), _esc()))
    colores = set(re.findall(r"#[0-9a-fA-F]{6}\b", raw))
    assert colores <= {v.lower() for v in ui_mod.PALETA.values() if v.startswith("#")} | {v for v in ui_mod.PALETA.values() if v.startswith("#")}


# ============================================================================== textos de ayuda
def test_ayuda_del_historico_explica_bandas_y_limites():
    txt = " ".join(ui_mod.texto_ayuda_historico())
    for clave in ("+10 a +30 %", "−10 a −30 %", "margen de seguridad", "sobrevaloración", "ILUSTRATIVA", "Piso fundamental"):
        assert clave in txt, clave
    for clave in ("causal", "ancla final", "point-in-time", "reconstrucción", "no son point-in-time"):
        assert clave in txt.lower(), clave
    assert str(fv.TRANSICION_DIAS) in txt and str(fv.REZAGO_PUBLICACION_DIAS) in txt


def test_texto_de_sliders_usa_las_constantes_y_avisa_del_limite():
    assert "8,0 %" in ui_mod.texto_slider("wacc") and "12,5 %" in ui_mod.texto_slider("wacc") and "200 pb" in ui_mod.texto_slider("wacc")
    assert "2,5 pp" in ui_mod.texto_slider("g") and "10" in ui_mod.texto_slider("g")
    assert "no mueve el valor justo" in ui_mod.texto_slider("mos")


def test_explicacion_de_modelos_usa_los_motivos_reales_del_motor():
    e = _entradas_ricas()
    e.eps_ltm = -1.0
    esc = fv.calcular_escenarios(e, None, None, None, 100.0)
    ex = ui_mod.explicacion_modelos(esc["modelos"])
    assert [f[0] for f in ex["familias"]] == ["DCF (flujos descontados)", "Múltiplos", "Valor / Dividendos"]
    assert len(ex["modelos"]) == 15 and all(m["linea"] for m in ex["modelos"])
    por = {n["id"]: n["motivo"] for n in ex["no_aplican"]}
    assert set(por) == {"pe", "graham", "lynch"} and all(por.values())
    motivo_motor = next(m for m in esc["modelos"] if m["id"] == "pe")["motivo"]
    assert por["pe"] == motivo_motor
    assert ui_mod.explicacion_modelos(None)["no_aplican"] == []


# ============================================================================== Vista: estructura y sincronizacion
def test_vista_estructura_de_bloques_y_acordeones_cerrados(cliente, vista):
    from nicegui import ui
    els = list(cliente.elements.values())
    bloques = [e.props["data-bloque"] for e in els if "data-bloque" in e.props]
    assert bloques == ["kpis", "historico", "desglose"]
    exp = {e.props["data-exp"]: e for e in els if isinstance(e, ui.expansion) and "data-exp" in e.props}
    assert set(exp) == {"vj-ayuda", "vj-barras", "vj-explicacion", "vj-supuestos", "vj-modelos", "vj-parametros"}
    assert all(e.props["model-value"] is False or not e.value for e in exp.values())          # colapsados por defecto
    assert exp["vj-ayuda"].props["label"].startswith("? Cómo interpretar")
    assert exp["vj-modelos"].props["label"] == "Activar / Desactivar modelos"
    assert exp["vj-parametros"].props["label"] == "Parámetros de sensibilidad"
    assert len([e for e in els if isinstance(e, ui.card) and e.props.get("data-kpi")]) == 4
    assert [e.props["data-kpi"] for e in els if isinstance(e, ui.card) and e.props.get("data-kpi")] == ["precio", "fair", "piso", "diag"]
    assert not [e for e in els if isinstance(e, ui.plotly)]


def test_vista_graficos_se_actualizan_sin_recrearse(cliente, vista):
    from nicegui import ui
    antes = [e for e in cliente.elements.values() if isinstance(e, ui.echart)]
    n_series = len(vista.vj_hist.options["series"])
    vista.vj_rows["dcf_5y_fcf"]["chk"].value = False
    vista.vj_sl["wacc"].value = 11.0
    despues = [e for e in cliente.elements.values() if isinstance(e, ui.echart)]
    assert antes == despues == [vista.vj_hist, vista.vj_barras]                              # mismos elementos
    assert len(vista.vj_hist.options["series"]) == n_series == 10                             # misma estructura: ECharts fusiona
    assert len(vista.vj_barras.options["series"]) == 2


def test_vista_los_cuatro_lugares_dicen_lo_mismo_tras_cada_cambio(vista):
    """KPI Medio, marca vertical de las barras, ultimo punto (ancla) del historico y linea de Analisis: una sola verdad."""
    vista.sw_vj.value = True

    def coherente():
        med = vista.vj_agg["fair_value"]
        assert vista.vj_kpi["fair"].text == f"${med:,.2f}"
        assert vista.vj_barras.options["series"][0]["markLine"]["data"][0]["xAxis"] == pytest.approx(med, abs=0.01)
        assert vista.vj_serie["ancla"]["valor_justo"] == pytest.approx(med, rel=1e-6)
        centro = [v for _, v in _por_nombre(vista.vj_hist.options)[ui_mod.SERIE_VJ]["data"] if v is not None]
        assert centro[-1] == pytest.approx(med, abs=0.01)
        assert vista.grafico_overlays["vj"]["precio"] == pytest.approx(med, abs=0.01)
        piso = vista.vj_esc["pesimista"]["fair_value"]
        assert vista.vj_kpi["piso"].text == f"${piso:,.2f}"
        assert _por_nombre(vista.vj_hist.options)[ui_mod.SERIE_VJ]["markLine"]["data"][0]["yAxis"] == pytest.approx(piso, abs=0.01)
    coherente()
    vista.vj_rows["dcf_5y_fcf"]["chk"].value = False
    coherente()
    vista.vj_sl["wacc"].value = 10.0
    coherente()
    vista.vj_sl["g"].value = 3.0
    coherente()
    vista._vj_seleccionar(ui_mod.ids_por_familia("DCF"))
    coherente()
    vista._vj_restablecer()
    coherente()


def test_vista_analisis_solo_recibe_la_linea_del_escenario_medio(vista):
    vista.sw_vj.value = True
    ov = vista.grafico_overlays["vj"]
    assert ov["etiqueta"].startswith("Fair Value Medio: ") and ov["precio"] == pytest.approx(vista.vj_agg["fair_value"], abs=0.01)
    assert f"{vista.vj_esc['pesimista']['fair_value']:.2f}" not in json.dumps(vista.grafico_overlays)
    assert ui_mod.valor_justo_medio(vista.vj_esc)["fair_value"] == vista.vj_agg["fair_value"]


def test_vista_sliders_wacc_fuera_de_rango_muestran_avisos_del_motor(vista):
    vista.vj_sl["wacc"].value = 6.0
    avisos = [e.text for e in vista.vj_avisos.default_slot.children]
    assert any("fuera del rango del escenario Medio" in a for a in avisos)
    vista._vj_restablecer()
    assert not [e for e in vista.vj_avisos.default_slot.children]
    assert vista.vj_params == fv.ParametrosValoracion()


def test_vista_el_wacc_del_usuario_sustituye_al_medio_y_deriva_el_pesimista(vista):
    vista.vj_sl["wacc"].value = 12.0
    sp = vista.vj_esc["supuestos"]
    assert sp["medio"]["wacc"] == pytest.approx(0.12) and sp["pesimista"]["wacc"] == pytest.approx(0.14)
    assert "Pesimista" in "".join(e.text for e in vista.vj_supuestos.default_slot.children if hasattr(e, "text"))


def test_vista_explicacion_muestra_no_aplica_con_motivo_real(cliente, df_sintetico, monkeypatch):
    e = _entradas_ricas()
    e.eps_ltm = -1.0
    monkeypatch.setitem(ui_mod.CACHE, "NA", {"ticker": "NA", "precio_actual": 100.0, "df": df_sintetico, "zonas": [],
                                             "errores": [], "entradas_valoracion": e.to_dict()})
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "NA"
    v.refrescar()
    textos = [x.text for x in v.vj_expl.default_slot.children if hasattr(x, "text")]
    motivo = next(m for m in v.vj_esc["modelos"] if m["id"] == "pe")["motivo"]
    assert any(t == f"• P/E: {motivo}" for t in textos) and any("No aplica" in t for t in textos)


# ============================================================================== serie historica: cache, hilo y huella
def _falsa_serie(llamadas):
    def falsa(entradas, df, params, activos, anios):
        llamadas.append({"hilo": threading.get_ident(), "activos": list(activos), "anios": anios, "wacc": params.wacc})
        return _serie()
    return falsa


def test_serie_cacheada_por_huella_ticker_parametros_y_modelos(vista, monkeypatch):
    ui_mod.CACHE_SERIES.clear()
    llamadas = []
    monkeypatch.setattr(ui_mod, "calcular_serie", _falsa_serie(llamadas))
    vista._vj_aplicar()
    n0 = len(llamadas)
    vista._vj_aplicar()                                     # mismos parametros y modelos: cache
    assert len(llamadas) == n0
    vista.vj_rows["pe"]["chk"].value = False                # otro conjunto de modelos: recalcula
    assert len(llamadas) == n0 + 1 and "pe" not in llamadas[-1]["activos"]
    vista.vj_rows["pe"]["chk"].value = True                 # vuelve a la huella anterior: cache (sin llamar)
    assert len(llamadas) == n0 + 1
    vista.vj_sl["wacc"].value = 12.0                        # otro WACC: recalcula
    assert len(llamadas) == n0 + 2 and llamadas[-1]["wacc"] == pytest.approx(0.12)
    vista.vj_sl["mos"].value = 40                           # el margen de seguridad no cambia el valor justo: cache
    assert len(llamadas) == n0 + 2


def test_serie_se_calcula_fuera_del_hilo_de_eventos(cliente, vista, monkeypatch):
    ui_mod.CACHE_SERIES.clear()
    llamadas = []
    monkeypatch.setattr(ui_mod, "calcular_serie", _falsa_serie(llamadas))
    visto = {}

    async def correr():
        with cliente:
            principal = threading.get_ident()
            vista.vj_activos["ddm"] = False
            vista._vj_aplicar()
            visto["cargando"] = vista.vj_cargando.visible          # indicador de carga mientras se calcula
            for _ in range(100):
                await asyncio.sleep(0.02)
                if llamadas:
                    break
            await asyncio.sleep(0.05)
            visto["principal"] = principal
    asyncio.run(correr())
    assert visto["cargando"] is True and llamadas and llamadas[-1]["hilo"] != visto["principal"]
    assert vista.vj_cargando.visible is False and vista.vj_serie is not None
    assert len(_por_nombre(vista.vj_hist.options)[ui_mod.SERIE_VJ]["data"]) > 60


def test_serie_atrasada_se_descarta_si_hay_una_peticion_mas_nueva(cliente, vista, monkeypatch):
    ui_mod.CACHE_SERIES.clear()
    pintadas = []
    monkeypatch.setattr(vista, "_vj_pintar_serie", lambda s, m=None: pintadas.append(s))
    h1, h2 = ("a",), ("b",)
    vista._vj_seq_serie = 5
    vista._vj_guardar_serie(h1, {"fechas": ["2024-01-01"], "id": 1}, 4)       # llega tarde (peticion 4 < 5)
    assert pintadas == [] and h1 in ui_mod.CACHE_SERIES                       # no se pinta, pero se guarda para otra vez
    vista._vj_guardar_serie(h2, {"fechas": ["2024-01-01"], "id": 2}, 5)
    assert [p["id"] for p in pintadas] == [2]


def test_serie_con_error_no_se_cachea_y_calcular_serie_no_lanza(df_sintetico):
    ui_mod.CACHE_SERIES.clear()
    s = ui_mod.calcular_serie({"ticker": "X"}, df_sintetico, "no-son-parametros", [], 3)
    assert s.get("error") and s["fechas"] == [] and "No se pudo" in s["notas"][0]


def test_serie_real_con_entradas_y_precios_sinteticos(vista, df_sintetico):
    """Sin red: la serie real (motor de A) de las entradas ricas termina en el valor justo Medio actual (ancla)."""
    s = ui_mod.calcular_serie(vista.vj_entradas, df_sintetico, vista.vj_params, list(fv.IDS_MODELOS), 10)
    assert s["fechas"] and s["ancla"]["valor_justo"] == pytest.approx(vista.vj_agg["fair_value"], rel=1e-6)
    opc = ui_mod.opciones_historico(s, vista.vj_esc)
    assert len(opc["series"]) == 10 and opc["series"][-3]["markLine"]["data"]


# ============================================================================== resize de ECharts
def test_resize_se_emite_al_cambiar_de_pestana_y_al_abrir_acordeones(cliente, vista, monkeypatch):
    from nicegui import ui
    llamadas = []
    monkeypatch.setattr(ui.echart, "run_chart_method", lambda self, nombre, *a, **k: llamadas.append((self.props.get("data-grafico"), nombre)))
    n0 = vista.n_resize_vj
    vista.tabs.set_value("valor_justo")
    assert vista.n_resize_vj == n0 + 1
    assert ("vj-historico", "resize") in llamadas and ("vj-barras", "resize") in llamadas
    vista.tabs.set_value("analisis")                          # al salir de la pestana no hace falta
    assert vista.n_resize_vj == n0 + 1
    vista.tabs.set_value("valor_justo")
    assert vista.n_resize_vj == n0 + 2
    exp = next(e for e in cliente.elements.values() if isinstance(e, ui.expansion) and e.props.get("data-exp") == "vj-barras")
    exp.set_value(True)                                       # abrir el acordeon del grafico de barras
    assert vista.n_resize_vj == n0 + 3


def test_resize_sin_cliente_conectado_no_lanza(cliente, vista, monkeypatch):
    from nicegui import ui

    def boom(*a, **k):
        raise RuntimeError("sin cliente")
    monkeypatch.setattr(ui.echart, "run_chart_method", boom)
    vista._vj_resize()


# ============================================================================== local vs nube
def test_periodo_y_anios_segun_el_entorno(monkeypatch, vista):
    monkeypatch.delenv("PORT", raising=False)
    assert ui_mod.periodo_consulta() == "10y" and vista._vj_anios() == 10 and ui_mod.anios_velas_defecto() == 10
    monkeypatch.setenv("PORT", "10000")
    assert ui_mod.periodo_consulta() == "3y" and vista._vj_anios() == 3 and ui_mod.anios_velas_defecto() == 3
    vista.anios_velas = 2                                     # parametro unico de historial
    assert vista._vj_anios() == 2


def test_calcular_ticker_pasa_el_periodo_segun_local_o_nube(monkeypatch):
    vistos = []
    monkeypatch.setattr(ui_mod.nucleo, "analizar_ticker", lambda t, c, **kw: vistos.append(kw) or {"ticker": t})
    monkeypatch.delenv("PORT", raising=False)
    from nicegui import run

    async def en_hilo(fn, *a, **k):
        return fn(*a, **k)
    monkeypatch.setattr(run, "cpu_bound", en_hilo)
    asyncio.run(ui_mod.calcular_ticker("A", 1.0))
    monkeypatch.setenv("PORT", "10000")
    asyncio.run(ui_mod.calcular_ticker("A", 1.0))
    assert vistos[0]["periodo"] == "10y" and vistos[1]["periodo"] == "3y"


def test_serie_usa_3_anios_en_la_nube_y_10_en_local(vista, monkeypatch):
    for entorno, esperado in ((None, 10), ("10000", 3)):
        ui_mod.CACHE_SERIES.clear()
        monkeypatch.delenv("PORT", raising=False)
        if entorno:
            monkeypatch.setenv("PORT", entorno)
        llamadas = []
        monkeypatch.setattr(ui_mod, "calcular_serie", _falsa_serie(llamadas))
        vista._vj_aplicar()
        assert llamadas and llamadas[-1]["anios"] == esperado


def test_cache_de_series_retiene_una_en_la_nube_y_varias_en_local(vista, monkeypatch):
    monkeypatch.setattr(ui_mod, "calcular_serie", _falsa_serie([]))
    monkeypatch.setenv("PORT", "10000")
    ui_mod.CACHE_SERIES.clear()
    for w in (9.0, 10.0, 11.0):
        vista.vj_sl["wacc"].value = w
    assert len(ui_mod.CACHE_SERIES) == ui_mod.MAX_SERIES_NUBE == 1
    monkeypatch.delenv("PORT")
    ui_mod.CACHE_SERIES.clear()
    for w in (9.0, 10.0, 11.0):
        vista.vj_sl["wacc"].value = w
    assert 3 <= len(ui_mod.CACHE_SERIES) <= ui_mod.MAX_SERIES_LOCAL


def test_gc_tras_cada_recalculo_solo_en_la_nube(vista, monkeypatch):
    llamadas = []
    monkeypatch.setattr(ui_mod.gc, "collect", lambda *a: llamadas.append(1) or 0)
    monkeypatch.delenv("PORT", raising=False)
    vista.vj_sl["g"].value = 3.2
    assert llamadas == []                                      # local: no se fuerza el recolector
    monkeypatch.setenv("PORT", "10000")
    vista.vj_sl["g"].value = 3.0
    assert llamadas                                            # nube: gc tras el recalculo


def test_limite_de_tickers_por_escaneo_solo_en_la_nube():
    muchos = [f"T{i}" for i in range(12)]
    assert ui_mod.MAX_TICKERS_ESCANEO_NUBE == 5
    assert ui_mod.limitar_tickers_escaneo(muchos, nube=False) == (muchos, 0)           # local: sin limite
    assert ui_mod.limitar_tickers_escaneo(muchos, nube=True) == (muchos[:5], 7)
    assert ui_mod.limitar_tickers_escaneo(muchos[:5], nube=True) == (muchos[:5], 0)


def _escanear(cliente, v):
    async def interno():
        with cliente:
            await v.escanear()
    asyncio.run(interno())


def _falso_calcular(vistos):
    async def falso(t, cap, bt, sec, vj=True):
        vistos.append(t)
        return {"ticker": t, "precio_actual": 10.0, "zonas": [], "errores": []}
    return falso


def test_escaneo_en_la_nube_limita_los_tickers_con_mensaje_claro(cliente, monkeypatch):
    from src.reporting import userdata
    userdata.guardar([f"T{i}" for i in range(8)])
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(3))
    vistos = []
    monkeypatch.setattr(ui_mod, "calcular_ticker", _falso_calcular(vistos))
    avisos = []
    from nicegui import ui
    monkeypatch.setattr(ui, "notify", lambda msg, **k: avisos.append(msg))
    v = ui_mod.Vista()
    v.construir()
    _escanear(cliente, v)
    assert vistos == [f"T{i}" for i in range(5)]                                         # secuencial y solo 5
    assert any("solo se escanean 5 tickers" in a and "se omiten 3" in a for a in avisos)
    assert "máximo 5 por escaneo" in v.lbl_scan.text
    assert ui_mod.ESCANEO_NUBE["activo"] is False                                        # se libera al terminar


def test_escaneo_en_la_nube_rechaza_un_segundo_escaneo_simultaneo(cliente, monkeypatch):
    from src.reporting import userdata
    userdata.guardar(["AAA", "BBB"])
    monkeypatch.setenv("PORT", "10000")
    vistos, avisos = [], []
    monkeypatch.setattr(ui_mod, "calcular_ticker", _falso_calcular(vistos))
    from nicegui import ui
    monkeypatch.setattr(ui, "notify", lambda msg, **k: avisos.append(msg))
    v = ui_mod.Vista()
    v.construir()
    monkeypatch.setitem(ui_mod.ESCANEO_NUBE, "activo", True)                            # otro cliente ya escanea
    _escanear(cliente, v)
    assert vistos == [] and any("Ya hay un escaneo en curso" in a for a in avisos)


def test_escaneo_local_no_se_limita_ni_usa_el_bloqueo_de_la_nube(cliente, monkeypatch):
    from src.reporting import userdata
    userdata.guardar([f"T{i}" for i in range(9)])
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setitem(ui_mod.ESCANEO_NUBE, "activo", True)                            # irrelevante en local
    vistos = []
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(None))
    monkeypatch.setattr(ui_mod, "calcular_ticker", _falso_calcular(vistos))
    v = ui_mod.Vista()
    v.construir()
    _escanear(cliente, v)
    assert vistos == [f"T{i}" for i in range(9)]


# ============================================================================== main.analizar_ticker: periodo / modo / info
def _pipeline(monkeypatch, df, capturas):
    import main as m
    import src.data.market_data as md
    import src.levels.valuation_floors as vf
    import src.validation.backtest_ranges as br
    monkeypatch.setattr(md, "descargar_historial", lambda t, p: capturas.setdefault("periodo", p) and (object(), df))
    monkeypatch.setattr(md, "obtener_info", lambda tk: capturas.__setitem__("info_directo", capturas.get("info_directo", 0) + 1) or {"sector": "Technology"})
    monkeypatch.setattr(md, "verificar_proximidad_earnings", lambda tk: {"alerta_activa": False, "mensaje": "ok"})
    monkeypatch.setattr(br, "calcular_zonas_pit", lambda d: {"atr_semanal": 1.0, "clusters": [], "primario": None})
    monkeypatch.setattr(vf, "calcular_pisos_valoracion", lambda *a, **k: {"anclas": [], "pe": None})

    def extraer(t, tk, d, info, facts=None, **kw):
        capturas["kw_vj"] = kw
        return fv.EntradasValoracion(ticker=t, precio=float(d["Close"].iloc[-1]))
    monkeypatch.setattr(fv, "extraer_entradas", extraer)
    return m


def test_analizar_ticker_acepta_periodo_y_modo_desde_la_ui(monkeypatch, df_sintetico):
    cap = {}
    m = _pipeline(monkeypatch, df_sintetico, cap)
    m.analizar_ticker("AAA", 1000, periodo="3y", backtest=False, sec=False, verbose=False, modo="nube")
    assert cap["periodo"] == "3y" and cap["kw_vj"] == {"modo": "nube"}
    cap.clear()
    m.analizar_ticker("AAA", 1000, backtest=False, sec=False, verbose=False)
    assert cap["periodo"] == "10y" and cap["kw_vj"] == {}                     # sin modo: el motor decide por PORT


def test_info_de_mercado_se_cachea_solo_en_local(monkeypatch, df_sintetico):
    import src.reporting.cache_local as cl
    cap = {}
    m = _pipeline(monkeypatch, df_sintetico, cap)
    monkeypatch.delenv("PORT", raising=False)
    m.analizar_ticker("CCC", 1000, backtest=False, sec=False, verbose=False, valor_justo=False)
    m.analizar_ticker("CCC", 1000, backtest=False, sec=False, verbose=False, valor_justo=False)
    assert cap["info_directo"] == 1 and cl.obtener("info:CCC", fv.VERSION_CODIGO) == {"sector": "Technology"}   # 2a vez: cache
    monkeypatch.setenv("PORT", "10000")
    cap["info_directo"] = 0
    m.analizar_ticker("CCC", 1000, backtest=False, sec=False, verbose=False, valor_justo=False)
    m.analizar_ticker("CCC", 1000, backtest=False, sec=False, verbose=False, valor_justo=False)
    assert cap["info_directo"] == 2                                            # nube: sin cache en disco


# ============================================================================== sin Plotly, offline y empaquetado
def test_plotly_ya_no_existe_en_la_app_ni_en_el_empaquetado():
    for nombre in ("app_ui.py", "main.py", "requirements.txt", "build.spec", "requirements-build.txt"):
        f = RAIZ / nombre
        if f.exists():
            assert "plotly" not in f.read_text(encoding="utf-8").lower(), nombre
    for f in (RAIZ / "src").rglob("*.py"):
        assert "plotly" not in f.read_text(encoding="utf-8").lower(), f


def test_echarts_viene_dentro_de_nicegui_y_el_exe_lo_empaqueta():
    import nicegui
    dist = Path(nicegui.__file__).parent / "elements" / "echart" / "dist"
    assert dist.is_dir() and any(dist.glob("*.js"))                           # sin CDN: el motor de ECharts va con el paquete
    assert 'collect_all' in (RAIZ / "build.spec").read_text(encoding="utf-8") and '"nicegui"' in (RAIZ / "build.spec").read_text(encoding="utf-8")
    src = Path(ui_mod.__file__).read_text(encoding="utf-8")
    assert not re.search(r"https?://[^\s\"')]*(cdn|jsdelivr|unpkg|cloudflare)", src.lower())


def test_import_app_ui_es_inocuo_y_no_carga_plotly():
    codigo = ("import socket, sys\n"
              "def _no(*a, **k): raise AssertionError('red prohibida al importar')\n"
              "socket.socket.connect = _no; socket.create_connection = _no\n"
              "import app_ui\n"
              "from nicegui import core\n"
              "assert not getattr(core.app, 'is_started', False)\n"
              "assert 'plotly' not in sys.modules, 'plotly sigue importandose'\n"
              "print('ok')")
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=120, cwd=str(RAIZ))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_badges_del_diagnostico_cumplen_contraste():
    css = ui_mod.css_tema()
    for clase in ("vj-dx-infra", "vj-dx-linea", "vj-dx-sobre", "vj-dx-debil"):
        m = re.search(r"\." + clase + r"\{background:(#[0-9a-fA-F]{6});color:(#[0-9a-fA-F]{6})", css)
        assert m, clase
        assert contraste(m.group(2), m.group(1)) >= 4.5, clase


def test_serie_con_puntos_absurdos_se_sanea_sin_tocar_el_original():
    serie = _serie()
    serie["valor_justo"][5] = 24_000_000.0                      # dato por accion corrupto (caso real KO 2017)
    serie["bandas"]["+30%"][5] = 31_200_000.0
    limpia, n = ui_mod.sanear_serie(serie)
    assert n == 1 and limpia["valor_justo"][5] is None and limpia["bandas"]["+30%"][5] is None
    assert serie["valor_justo"][5] == 24_000_000.0 and limpia["valor_justo"][4] == serie["valor_justo"][4]
    assert "se omiten del gráfico" in limpia["notas"][-1]
    opc = ui_mod.opciones_historico(limpia, _esc())
    todos = [v for s_ in opc["series"] for _, v in s_["data"] if v is not None]
    assert max(todos) < 1000                                      # la escala ya no se deforma
    sana, cero = ui_mod.sanear_serie(_serie())
    assert cero == 0 and sana is not None and ui_mod.sanear_serie(None) == (None, 0)


def test_vista_pinta_la_serie_saneada(vista, monkeypatch):
    mala = _serie()
    mala["valor_justo"][3] = 9e9
    vista._vj_pintar_serie(mala)
    centro = [v for _, v in _por_nombre(vista.vj_hist.options)[ui_mod.SERIE_VJ]["data"] if v is not None]
    assert max(centro) < 1000 and any("se omiten" in e.text for e in vista.vj_hist_notas.default_slot.children)
