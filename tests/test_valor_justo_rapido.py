"""Valor justo sin friccion: interruptor activado por defecto, boton de calculo rapido en el estado vacio de la pestana
(``analizar_ticker`` con backtest=False y sec=False), sin pisar un resultado completo, y etiquetas escalonadas del
grafico de barras. Sin red: ``main.analizar_ticker`` esta simulado."""
import asyncio
import functools

import pytest

import app_ui as ui_mod
from src.reporting import userdata
from test_fair_value import _anot_fair, _entradas_ricas, _sin_red_ni_disco, cliente  # noqa: F401


# ============================================================================== utilidades
def _res_rapido(t="AAA", df=None, precio=100.0, entradas=True):
    """Lo que devuelve ``analizar_ticker(backtest=False, sec=False)``: sin backtest ni SEC, con entradas."""
    r = {"ticker": t, "precio_actual": precio, "df": df, "zonas": [], "errores": [], "avisos": [], "form4": None,
         "r13f": None, "confluencia_score": 50.0, "confiabilidad": {"indice": 30.0, "cobertura": 0.5}}
    if entradas:
        e = _entradas_ricas()
        e.ticker = t
        r["entradas_valoracion"] = e.to_dict()
    return r


def _res_completo(t="AAA", df=None):
    """Resultado de un Recalcular completo (backtest + SEC) pero SIN entradas de valoracion."""
    return {"ticker": t, "precio_actual": 105.0, "df": df, "errores": [], "avisos": ["aviso del backtest"],
            "zonas": [{"nivel": 1, "r_inf": 90.0, "r_sup": 95.0, "p_centro": 92.5, "dist_pct": -9.5, "n_familias": 3,
                       "n_anclas": 3, "familias": ["volumen", "ob", "avwap"], "tipos": ["POC", "OB", "AVWAP"],
                       "fuente": "POC+OB+AVWAP"}],
            "backtest_resumen": {"n_ordenes": 40, "n_fills": 25, "fill_rate": 0.6, "bounce": 0.7},
            "n_ordenes": 40, "n_fills": 25, "form4": [], "r13f": {}, "confluencia_score": 80.0,
            "confiabilidad": {"indice": 70.0, "cobertura": 1.0}}


@pytest.fixture
def espia(monkeypatch):
    """Sustituye ``main.analizar_ticker`` y registra los kwargs de cada llamada (sin red)."""
    llamadas, respuesta = [], {}

    @functools.wraps(ui_mod.nucleo.analizar_ticker)   # misma firma: la pestana Modelo lee sus valores por defecto
    def falso(ticker, capital, **kw):
        llamadas.append({"ticker": ticker, "capital": capital, **kw})
        return respuesta["fn"](ticker)
    monkeypatch.setattr(ui_mod.nucleo, "analizar_ticker", falso)
    respuesta["fn"] = lambda t: _res_rapido(t)
    return llamadas, respuesta


@pytest.fixture
def cache_vacia(monkeypatch):
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(None))
    monkeypatch.setattr(ui_mod, "CACHE_SERIES", ui_mod.CacheLRU(None))


def _vista(tickers=("AAA",)):
    userdata.guardar(list(tickers))
    v = ui_mod.Vista()
    v.construir()
    return v


def _correr(cliente, coro_fn):
    async def _c():
        with cliente:
            await coro_fn()
            await asyncio.sleep(0.05)   # deja terminar la serie historica (hilo) si se pidio
    asyncio.run(_c())


# ============================================================================== 1. interruptor activado por defecto
@pytest.mark.parametrize("nube", [False, True])
def test_mostrar_valor_justo_activado_por_defecto_local_y_nube(cliente, cache_vacia, monkeypatch, nube):
    if nube:
        monkeypatch.setenv("PORT", "10000")
    else:
        monkeypatch.delenv("PORT", raising=False)
    v = _vista()
    assert ui_mod.VJ_LINEA_POR_DEFECTO is True and v.sw_vj.value is True


def test_con_entradas_la_linea_del_escenario_medio_aparece_sin_tocar_el_interruptor(cliente, cache_vacia, df_sintetico):
    ui_mod.CACHE["AAA"] = _res_rapido("AAA", df_sintetico)
    v = _vista()
    v.refrescar()
    assert _anot_fair(v) and f"${v.vj_agg['fair_value']:,.2f}" in _anot_fair(v)[0]


# ============================================================================== 2. boton del estado vacio
def test_estado_vacio_muestra_boton_con_el_ticker_y_no_calcula_solo(cliente, cache_vacia, espia):
    llamadas, _ = espia
    v = _vista()
    assert v.ticker == "AAA" and v.vj_fila_calc.visible is True
    assert v.btn_vj_calc.text == "Calcular valor justo de AAA" and v.btn_vj_calc.props.get("data-accion") == "vj-calcular"
    assert "Calcular valor justo de AAA" in v.vj_msg.text and v.vj_kpi["fair"].text == "n/d"
    assert llamadas == []                                   # abrir la app no lanza ningun calculo


def test_sin_ticker_no_hay_boton(cliente, cache_vacia):
    v = _vista(())
    assert v.ticker is None and v.vj_fila_calc.visible is False and "Pulsa Recalcular" in v.vj_msg.text


def test_boton_llama_a_analizar_ticker_sin_backtest_ni_sec(cliente, cache_vacia, espia, df_sintetico):
    llamadas, resp = espia
    resp["fn"] = lambda t: _res_rapido(t, df_sintetico)
    v = _vista()
    v.sw_bt.value = True                                     # aunque Backtest (lento) y SEC esten activos
    v.sw_sec.value = True
    _correr(cliente, v.calcular_valor_justo)
    assert len(llamadas) == 1
    kw = llamadas[0]
    assert kw["ticker"] == "AAA" and kw["backtest"] is False and kw["sec"] is False
    assert kw.get("valor_justo", True) is True and kw["verbose"] is False
    assert v.ocupado is False and v.btn.enabled and "loading" not in v.btn_vj_calc.props


def test_calcular_ticker_rapido_pasa_backtest_y_sec_falsos(monkeypatch):
    vistos = []

    async def falso(t, cap, backtest=True, sec=True, valor_justo=True):
        vistos.append((t, cap, backtest, sec, valor_justo))
        return {}
    monkeypatch.setattr(ui_mod, "calcular_ticker", falso)
    asyncio.run(ui_mod.calcular_valor_justo_rapido("NVDA", 500.0))
    assert vistos == [("NVDA", 500.0, False, False, True)]


# ============================================================================== 3. no pisar un resultado completo
def test_no_pisa_un_resultado_completo_solo_anade_las_entradas(cliente, cache_vacia, espia, df_sintetico):
    _, resp = espia
    resp["fn"] = lambda t: _res_rapido(t, df_sintetico, precio=99.0)
    completo = _res_completo("AAA", df_sintetico)
    ui_mod.CACHE["AAA"] = completo
    v = _vista()
    v.refrescar()
    assert v.vj_fila_calc.visible is True                    # el completo no trae entradas: estado vacio
    _correr(cliente, v.calcular_valor_justo)
    r = ui_mod.CACHE["AAA"]
    assert r["entradas_valoracion"]["ticker"] == "AAA"
    for k in ("zonas", "backtest_resumen", "form4", "r13f", "confiabilidad", "avisos"):
        assert r[k] is completo[k], k                        # zonas/backtest/SEC del completo intactos
    assert r["precio_actual"] == 105.0 and not r.get("calculo_rapido")
    assert ui_mod.AVISO_CALCULO_RAPIDO not in r["avisos"]
    assert v.vj_kpi["precio"].text == "$105.00" and v.vj_kpi["fair"].text != "n/d"
    assert v.tabla.rows and v.tabla.rows[0]["fill_rate"] is not None   # la tabla conserva el backtest


def test_fusion_pura():
    f = ui_mod.fusionar_valor_justo_rapido
    rapido = _res_rapido()
    completo = _res_completo()
    m = f(completo, rapido)
    assert m is not completo and m["entradas_valoracion"] is rapido["entradas_valoracion"]
    assert {k: v for k, v in m.items() if k != "entradas_valoracion"} == completo
    assert "entradas_valoracion" not in completo                        # el previo no se muta
    assert f(completo, _res_rapido(entradas=False)) is completo          # sin entradas nuevas: nada que anadir
    solo = f(None, rapido)
    assert solo["calculo_rapido"] is True and solo["avisos"][-1] == ui_mod.AVISO_CALCULO_RAPIDO
    assert rapido["avisos"] == []                                       # el rapido no se muta
    assert f({"ticker": "AAA", "precio_actual": None}, rapido)["calculo_rapido"] is True   # previo sin datos: no cuenta
    assert f(None, solo)["avisos"].count(ui_mod.AVISO_CALCULO_RAPIDO) == 1
    assert "Recalcular" in ui_mod.AVISO_CALCULO_RAPIDO and "backtest" in ui_mod.AVISO_CALCULO_RAPIDO


# ============================================================================== 4. KPIs, barras, serie y linea tras el calculo rapido
def test_tras_el_calculo_rapido_se_actualizan_kpis_barras_serie_y_linea(cliente, cache_vacia, espia, df_sintetico):
    _, resp = espia
    resp["fn"] = lambda t: _res_rapido(t, df_sintetico)
    v = _vista()
    assert not _anot_fair(v) and v.vj_kpi["fair"].text == "n/d"
    _correr(cliente, v.calcular_valor_justo)
    r = ui_mod.CACHE["AAA"]
    assert r["calculo_rapido"] is True and ui_mod.AVISO_CALCULO_RAPIDO in r["avisos"]
    assert v.vj_agg is not None
    fair = v.vj_agg["fair_value"]
    assert v.vj_kpi["fair"].text == f"${fair:,.2f}" and v.vj_kpi["precio"].text == "$100.00"
    assert v.vj_kpi["piso"].text != "n/d" and v.vj_msg.text == ""
    assert v.vj_fila_calc.visible is False                    # ya hay datos: el boton desaparece
    assert v.vj_barras.options["series"][0]["markLine"]["data"]
    assert _anot_fair(v) and f"${fair:,.2f}" in _anot_fair(v)[0]      # linea en Analisis sin tocar el interruptor
    textos = [getattr(e, "text", "") for e in v.avisos.default_slot.children]
    assert ui_mod.AVISO_CALCULO_RAPIDO in textos              # Avisos de Analisis: sin backtest, Recalcular = completo
    assert v.tarj["precio"].text == "$100.00"


def test_serie_historica_se_pinta_tras_el_calculo_rapido(cliente, cache_vacia, espia, df_sintetico, monkeypatch):
    _, resp = espia
    resp["fn"] = lambda t: _res_rapido(t, df_sintetico)
    serie = {"fechas": ["2024-01-02", "2024-01-03"], "valor_justo": [100.0, 101.0], "precio": [90.0, 91.0],
             "bandas": {}, "proyeccion": {}, "ancla": None, "notas": []}
    monkeypatch.setattr(ui_mod, "calcular_serie", lambda *a, **k: serie)
    v = _vista()

    async def calcular_y_esperar():
        await v.calcular_valor_justo()
        for _ in range(50):
            if v.vj_serie and v.vj_serie.get("fechas"):
                break
            await asyncio.sleep(0.02)
    _correr(cliente, calcular_y_esperar)
    assert v.vj_serie and v.vj_serie["fechas"] == serie["fechas"]


def test_sin_precio_no_guarda_nada_y_el_boton_sigue(cliente, cache_vacia, espia):
    _, resp = espia
    resp["fn"] = lambda t: {"ticker": t, "precio_actual": None, "zonas": [], "errores": ["mercado: sin datos"]}
    v = _vista()
    _correr(cliente, v.calcular_valor_justo)
    assert "AAA" not in ui_mod.CACHE and v.vj_fila_calc.visible is True and "Calcular valor justo de AAA" in v.vj_msg.text


def test_sin_entradas_guarda_el_resultado_y_ofrece_reintentar(cliente, cache_vacia, espia, df_sintetico):
    _, resp = espia
    resp["fn"] = lambda t: {**_res_rapido(t, df_sintetico, entradas=False), "errores": ["valor_justo: RuntimeError: x"]}
    v = _vista()
    _correr(cliente, v.calcular_valor_justo)
    assert ui_mod.CACHE["AAA"]["calculo_rapido"] is True
    assert "no trae entradas" in v.vj_msg.text and v.vj_fila_calc.visible is True


def test_ocupado_no_lanza_un_segundo_calculo(cliente, cache_vacia, espia):
    llamadas, _ = espia
    v = _vista()
    v.ocupado = True
    _correr(cliente, v.calcular_valor_justo)
    assert llamadas == []


# ============================================================================== 5. cambio de ticker
def test_cambiar_de_ticker_sin_entradas_muestra_el_boton_sin_calcular(cliente, cache_vacia, espia, df_sintetico):
    llamadas, _ = espia
    ui_mod.CACHE["AAA"] = _res_rapido("AAA", df_sintetico)
    v = _vista(("AAA", "BBB"))
    v.refrescar()
    v.tabs.set_value("valor_justo")
    assert v.vj_fila_calc.visible is False and v.vj_agg is not None
    v.sel_ticker.value = "BBB"                                 # dispara _ticker_cambiado
    assert v.ticker == "BBB" and v.vj_fila_calc.visible is True
    assert v.btn_vj_calc.text == "Calcular valor justo de BBB" and v.vj_kpi["fair"].text == "n/d"
    assert llamadas == []                                      # nunca recalcula sin accion del usuario
    v.sel_ticker.value = "AAA"                                 # vuelve a uno con entradas: el boton se oculta
    assert v.vj_fila_calc.visible is False and v.vj_kpi["fair"].text != "n/d"


def test_cambio_de_ticker_durante_el_calculo_guarda_sin_repintar_otro(cliente, cache_vacia, espia, df_sintetico):
    _, resp = espia
    v = _vista(("AAA", "BBB"))

    def rapido(t):
        v.ticker = "BBB"                                       # el usuario cambia de ticker mientras se calcula
        return _res_rapido(t, df_sintetico)
    resp["fn"] = rapido
    _correr(cliente, v.calcular_valor_justo)
    assert ui_mod.CACHE["AAA"]["entradas_valoracion"]["ticker"] == "AAA"
    assert v.vj_agg is None and v.btn_vj_calc.text == "Calcular valor justo de BBB"


# ============================================================================== 6. etiquetas del grafico de barras
def test_filas_de_etiquetas_una_por_linea_en_orden_de_valor():
    f = ui_mod.filas_etiquetas_barras
    assert f([101.5, 98.7, 100.0]) == {1: 0, 2: 1, 0: 2}
    assert f([None, 50.0, 50.0]) == {1: 0, 2: 1}               # valores iguales: filas distintas igualmente
    assert f([None, None, None]) == {}


def test_etiquetas_de_lineas_cercanas_no_comparten_fila_ni_tocan_la_leyenda():
    from test_fair_value import _esc_rico
    base = _esc_rico()
    esc = {**base, "medio": {**base["medio"], "fair_value": 101.5}, "pesimista": {**base["pesimista"], "fair_value": 98.7}}
    opc = ui_mod.opciones_barras(esc, 100.0, None)
    lineas = opc["series"][0]["markLine"]["data"]
    assert [l["xAxis"] for l in lineas] == [101.5, 98.7, 100.0]          # orden Medio, Pesimista, Precio (sin cambios)
    offs = [l["label"]["offset"][1] for l in lineas]
    assert len(set(offs)) == 3 and all(o <= 0 for o in offs)             # una fila distinta por linea, hacia arriba
    alto = ui_mod.BARRAS_ALTO_FILA_ETIQUETA
    assert sorted(-o for o in offs) == [0, alto, 2 * alto]
    assert all(l["label"]["position"] == "start" for l in lineas)       # franja sobre el area de trazado (eje Y invertido)
    assert opc["grid"]["top"] >= ui_mod.BARRAS_TOP_LEYENDA + 3 * alto   # cabe la franja bajo la leyenda
    assert all(l["label"]["backgroundColor"] == ui_mod.BG_ETIQUETA for l in lineas)


def test_etiqueta_cerca_del_borde_derecho_se_alinea_a_la_izquierda_de_la_linea():
    from test_fair_value import _esc_rico
    base = _esc_rico()
    opc = ui_mod.opciones_barras(base, 100.0, None)
    xmax = opc["xAxis"]["max"]
    for l in opc["series"][0]["markLine"]["data"]:
        assert l["label"]["align"] == ("right" if l["xAxis"] > 0.6 * xmax else "left")
    sin_lineas = ui_mod.opciones_barras(None, None, None, "Sin análisis")
    assert sin_lineas["series"][0]["markLine"]["data"] == [] and sin_lineas["grid"]["top"] >= ui_mod.BARRAS_TOP_LEYENDA


# ============================================================================== 7. grafico de Analisis calculado desde otra pestana
def test_con_analisis_oculta_el_grafico_se_envia_al_volver(cliente, cache_vacia, espia, df_sintetico):
    """Con la pestana Valor Justo activa el contenedor del grafico de Analisis esta fuera del DOM: el calculo no le envia
    nada (el JS agotaria sus reintentos y quedaria en «Sin datos»); al volver a Analisis se envia UNA vez el estado
    completo (velas + overlays con la linea del Escenario Medio)."""
    _, resp = espia
    resp["fn"] = lambda t: _res_rapido(t, df_sintetico)
    v = _vista()
    enviados = []
    v._grafico_js = enviados.append
    v.tabs.set_value("valor_justo")
    assert v._analisis_visible() is False
    _correr(cliente, v.calcular_valor_justo)
    assert enviados == [] and v._grafico_pendiente is True       # nada enviado con la pestana oculta
    assert v.grafico_datos and v.grafico_datos.get("t")           # pero el payload ya esta preparado
    v.tabs.set_value("analisis")
    assert len(enviados) == 1 and '"datos"' in enviados[0] and "Fair Value Medio" in enviados[0]
    assert v._grafico_pendiente is False
    v.tabs.set_value("watchlist")
    v.tabs.set_value("analisis")
    assert len(enviados) == 1                                      # sin cambios mientras tanto: no se reenvia


def test_analisis_visible_al_inicio_envia_directo(cliente, cache_vacia, df_sintetico):
    ui_mod.CACHE["AAA"] = _res_rapido("AAA", df_sintetico)
    v = _vista()
    assert v._analisis_visible() is True                          # al inicio tabs.value es el objeto Tab de Analisis
    enviados = []
    v._grafico_js = enviados.append
    v.refrescar()
    assert len(enviados) == 1 and v._grafico_pendiente is False
