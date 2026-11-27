"""Tests de las funciones puras de app_ui.py (datos sinteticos, sin red ni servidor)."""
import numpy as np
import pandas as pd
import pytest

import app_ui as ui_mod


def _zonas_ricas():
    return [
        {"nivel": 1, "r_inf": 90.0, "r_sup": 95.0, "p_centro": 92.5, "dist_pct": -5.0, "n_familias": 3, "n_anclas": 5,
         "familias": ["avwap", "ob", "volumen"], "tipos": ["AVWAP", "HVN", "OB", "POC"], "fuente": "AVWAP+HVN+OB+POC"},
        {"nivel": 2, "r_inf": 80.0, "r_sup": 84.0, "p_centro": 82.0, "dist_pct": -16.0, "n_familias": 2, "n_anclas": 2,
         "familias": ["logreg", "swing"], "tipos": ["LOG", "SWING"], "fuente": "LOG+SWING"},
        {"nivel": 3, "r_inf": 70.0, "r_sup": 72.0, "p_centro": 71.0, "dist_pct": -28.0, "n_familias": 1, "n_anclas": 1,
         "familias": ["ob"], "tipos": ["OB"], "fuente": "OB"}]


def _res_sintetico(df, zonas):
    return {"ticker": "TEST", "precio_actual": 100.0, "df": df, "zonas": zonas, "errores": [],
            "form4": None, "r13f": None, "confluencia_score": 60.0,
            "confiabilidad": {"indice": 40.0, "cobertura": 0.8},
            "valoracion": {"pe": {"pisos": {"P10": 91.0, "P25": 85.0}}, "ev_fcf": None,
                           "dcf": {"precios_mos": {"MOS30": 150.0}}},
            "backtest_resumen": {"bounce": 0.67, "fill_rate": 0.34, "mae60_mediana": -0.084}}


def _todas():
    return ui_mod.familias_activas(ui_mod.TOGGLES)


def test_import_expone_funciones_puras():
    assert callable(ui_mod.datos_grafico) and callable(ui_mod.overlays_grafico) and callable(ui_mod.filtrar_zonas)
    assert not hasattr(ui_mod, "construir_figura")  # Analisis ya no usa Plotly


def test_mapeo_familias_alineado_con_backtest():
    br = pytest.importorskip("src.validation.backtest_ranges")
    assert ui_mod.TIPO_A_FAMILIA == br.FAMILIAS
    declaradas = {f for t in ui_mod.TOGGLES.values() for f in t["familias"]} - {"valoracion"}
    assert declaradas == set(br.FAMILIAS.values())


def test_familias_activas():
    assert ui_mod.familias_activas(["HVN", "OB"]) == {"volumen", "ob"}
    assert ui_mod.familias_activas([]) == set()
    assert ui_mod.familias_activas(["NOEXISTE"]) == set()


def test_filtrar_todo_activo_equivale_al_original():
    out = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], capital=1000.0)
    assert [x["nivel"] for x in out] == [1, 2, 3]
    assert [x["r_sup"] for x in out] == [95.0, 84.0, 72.0]
    assert sum(x["capital_asignado"] for x in out) == pytest.approx(1000.0)


def test_filtrar_no_muta_entrada_y_renumera():
    z = _zonas_ricas()
    original = [dict(x) for x in z]
    out = ui_mod.filtrar_zonas(z, {"ob"}, [], capital=900.0)
    assert z == original and "capital_asignado" not in z[0]
    assert [x["nivel"] for x in out] == [1, 2]  # N2 (logreg+swing) desaparece
    assert all(x["familias"] == ["ob"] and x["n_familias"] == 1 for x in out)
    assert out[0]["tipos"] == ["OB"] and out[0]["fuente"] == "OB"
    assert sum(x["capital_asignado"] for x in out) == pytest.approx(900.0)


def test_filtrar_ninguna_familia():
    assert ui_mod.filtrar_zonas(_zonas_ricas(), set(), [], capital=100.0) == []


def test_filtrar_valoracion_en_zona():
    z = _zonas_ricas()
    av = ui_mod.anclas_valoracion(_res_sintetico(None, z))
    assert {a["tipo"] for a in av} == {"PE_P10", "PE_P25", "DCF_MOS30"}
    sin = ui_mod.filtrar_zonas(z, {"volumen"}, av)  # toggle apagado: no cuenta
    assert all("valoracion" not in x["familias"] for x in sin)
    con = ui_mod.filtrar_zonas(z, {"volumen", "valoracion"}, av)
    n1 = next(x for x in con if x["r_sup"] == 95.0)
    assert "valoracion" in n1["familias"] and "PE_P10" in n1["tipos"] and n1["n_familias"] == 2
    n2 = next(x for x in con if x["r_sup"] == 84.0)
    assert n2["familias"] == ["valoracion"]  # PE_P25 = 85 cae en 80-84 con +2 %


def test_anclas_valoracion_vacia():
    assert ui_mod.anclas_valoracion({}) == []
    assert ui_mod.anclas_valoracion({"valoracion": {"pe": None, "dcf": {"precios_mos": {"x": float("nan")}}}}) == []


def test_precio_promedio_ponderado():
    z = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], capital=1000.0)
    p = ui_mod.precio_promedio_ponderado(z)
    assert p == pytest.approx(sum(x["capital_asignado"] for x in z) / sum(x["acciones"] for x in z))
    assert min(x["r_sup"] for x in z) <= p <= max(x["r_sup"] for x in z)
    assert ui_mod.precio_promedio_ponderado([]) is None


def test_filas_tabla_con_y_sin_backtest():
    z = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], capital=1000.0)
    filas = ui_mod.filas_tabla(z, {"bounce": 0.67, "fill_rate": 0.34, "mae60_mediana": -0.084}, 100.0)
    assert filas[0]["nivel"] == "N1" and filas[0]["bounce"] == pytest.approx(67.0)
    assert filas[0]["mae"] == pytest.approx(-8.4) and filas[0]["entrada"] == 95.0
    assert filas[0]["dist_pct"] == pytest.approx(-5.0)  # (r_sup 95 / precio 100 - 1) * 100: siempre contra r_sup
    vacias = ui_mod.filas_tabla(z, {}, 100.0)
    assert vacias[0]["bounce"] is None and vacias[0]["fill_rate"] is None and vacias[0]["mae"] is None
    assert ui_mod.filas_tabla(z, {"bounce": float("nan")}, None)[0]["bounce"] is None
    sin_capital = ui_mod.filas_tabla(_zonas_ricas(), None, 100.0)  # sin asignar: entrada n/d
    assert sin_capital[0]["entrada"] is None


def test_resumen_tarjetas_y_avisos():
    res = _res_sintetico(None, _zonas_ricas())
    res["earnings"] = {"alerta_activa": True, "mensaje": "[!] ALERTA DE EARNINGS"}
    res["errores"] = ["backtest: boom"]
    z = ui_mod.filtrar_zonas(res["zonas"], _todas(), [], 1000.0)
    d = ui_mod.resumen_tarjetas(res, z)
    assert d["precio"] == 100.0 and d["confiabilidad"] == 40.0 and d["earnings_alerta"]
    assert any("ALERTA" in a for a in d["avisos"]) and any("SEC" in a for a in d["avisos"])
    assert d["errores"] == ["backtest: boom"] and d["precio_promedio"] is not None


def test_filas_ranking_ordena_y_separa_fallidos():
    def r(t, score):
        return {"ticker": t, "precio_actual": 10.0, "score_ranking": score, "confluencia_score": score,
                "confiabilidad": {"indice": score}, "zonas": [{"r_inf": 1.0, "r_sup": 2.0, "dist_pct": -3.0}]}
    caido = {"ticker": "X", "precio_actual": None, "zonas": [], "errores": ["x"]}
    filas, fallidos = ui_mod.filas_ranking([r("A", 10.0), r("B", 80.0), r("C", None), caido])
    assert [f["ticker"] for f in filas] == ["B", "A", "C"] and [f["pos"] for f in filas] == [1, 2, 3]
    assert fallidos == ["X"] and filas[0]["mejor_zona"] == "$1.00 - $2.00" and filas[2]["score"] is None


def test_validadores():
    assert ui_mod.validar_ticker(" brk-b ") == "BRK-B" and ui_mod.validar_ticker("NVDA") == "NVDA"
    assert ui_mod.validar_ticker("") is None and ui_mod.validar_ticker("no valido!") is None
    assert ui_mod.validar_ticker(None) is None
    assert ui_mod.validar_capital(1) and not ui_mod.validar_capital(0) and not ui_mod.validar_capital(None)
    assert not ui_mod.validar_capital(-5) and not ui_mod.validar_capital(float("nan"))


def test_cargar_watchlist_real_y_faltante(tmp_path):
    assert ui_mod.cargar_watchlist(str(tmp_path / "no_existe.csv")) == []
    p = tmp_path / "w.csv"
    p.write_text("ticker,nota\nnvda,x\nAAPL,y\nNVDA,z\n", encoding="utf-8")
    assert ui_mod.cargar_watchlist(str(p)) == ["NVDA", "AAPL"]


def _form4(df):
    return pd.DataFrame({"Code": ["P", "S", "P"], "Date": [df.index[100], df.index[120], df.index[300]],
                         "Price": [np.nan, 50.0, 110.0], "Shares": [10, 5, 7], "Value": [1000, 250, 770]})


def _r13f(df):
    return pd.DataFrame({"gestor": ["A", "B"], "periodo": [df.index[200].strftime("%Y-%m-%d")] * 2,
                         "acciones": [10.0, 5.0], "acciones_previas": [5.0, 8.0], "cambio_pct": [100.0, -37.5]})


# =====================================================================================================
# Modo oscuro: paleta, contraste WCAG, Plotly (hoverlabel/modebar/leyenda/anotaciones), CSS y colores Quasar
# =====================================================================================================
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

FONDOS = ("BG_PAGINA", "BG_PAPEL", "BG_PANEL", "BG_DRAWER", "BG_TARJETA", "BG_POPUP")


def _rgba(c):
    c = c.strip()
    if c.startswith("#"):
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 1.0)
    p = [float(x) for x in re.match(r"rgba?\(([^)]*)\)", c).group(1).split(",")]
    return (p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0)


def _opaco(c, base):
    """Color ``c`` (posiblemente con alfa) compuesto sobre ``base`` -> (r, g, b)."""
    r, g, b, a = _rgba(c)
    rb, gb, bb, _ = _rgba(base)
    return (r * a + rb * (1 - a), g * a + gb * (1 - a), b * a + bb * (1 - a))


def _lum(rgb):
    def f(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contraste(fg, bg, base=None):
    """Razon de contraste WCAG entre ``fg`` y ``bg`` (ambos compuestos sobre ``base`` si tienen alfa)."""
    base = base or ui_mod.BG_PAGINA
    b = _opaco(bg, base)
    f = _opaco(fg, "rgb(%d,%d,%d)" % tuple(round(x) for x in b))
    l1, l2 = sorted((_lum(f), _lum(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def test_contraste_helper_conocido():
    assert contraste("#ffffff", "#000000") == pytest.approx(21.0, rel=1e-3)
    assert contraste("#777777", "#ffffff") == pytest.approx(4.48, abs=0.05)


@pytest.mark.parametrize("texto", ["TXT", "MUTED", "ACENTO_TXT", "OK_TXT", "AVISO_TXT", "ERROR_TXT"])
def test_paleta_contraste_texto_sobre_todos_los_fondos(texto):
    for fondo in FONDOS:
        r = contraste(getattr(ui_mod, texto), getattr(ui_mod, fondo))
        assert r >= 4.5, f"{texto} sobre {fondo}: {r:.2f}:1"


def test_paleta_contraste_estados_y_superposiciones():
    m = ui_mod
    assert contraste(m.BLANCO, m.ACENTO) >= 4.5            # boton primario
    for bg in (m.OK_BG, m.ERROR_BG, m.INFO_BG):            # notify positive / negative / info: texto blanco
        assert contraste(m.BLANCO, bg) >= 4.5
    assert contraste(m.TXT_SOBRE_AVISO, m.AVISO_BG) >= 4.5  # notify warning: texto oscuro
    for sup in (m.HOVER_FILA, m.SEL_FILA, m.ACTIVO_ITEM):   # filas hover/seleccionada e item activo
        assert contraste(m.TXT, sup, base=m.BG_TARJETA) >= 4.5
        assert contraste(m.ACENTO_TXT, sup, base=m.BG_POPUP) >= 4.5
    assert contraste(m.BLANCO, m.BG_ETIQUETA, base=m.BG_PANEL) >= 4.5       # etiquetas de bandas
    assert contraste(m.COLOR_VALORACION, m.BG_ETIQUETA, base=m.BG_PANEL) >= 4.5
    for banda in (m.VJ_CIAN, m.VJ_MAGENTA, m.VJ_PESIMISTA):                  # bandas / piso del grafico historico
        assert contraste(banda, m.BG_PANEL) >= 3.0, banda


def test_colores_quasar_contraste():
    q = ui_mod.COLORES_QUASAR
    assert q["dark_page"] == ui_mod.BG_PAGINA and q["dark"] == ui_mod.BG_TARJETA
    assert contraste(ui_mod.BLANCO, q["primary"]) >= 4.5
    for k in ("positive", "negative", "info"):
        assert contraste(ui_mod.BLANCO, q[k]) >= 4.5, k
    assert contraste(ui_mod.TXT_SOBRE_AVISO, q["warning"]) >= 4.5


def test_paleta_publica_sin_restos_de_plotly():
    assert {"BG_PAPEL", "BG_PANEL", "TXT", "MUTED", "GRILLA", "VJ_CIAN", "VJ_MAGENTA", "VJ_PESIMISTA"} <= set(ui_mod.PALETA)
    assert _rgba(ui_mod.BG_PAPEL)[3] == 1.0
    assert not {"BG_MODEBAR", "TRANSPARENTE"} & set(ui_mod.PALETA)
    for nombre in ("_hoverlabel", "_layout_tema", "_figura_vacia", "construir_figura_valor_justo", "go"):
        assert not hasattr(ui_mod, nombre), nombre


def test_css_tema_usa_la_paleta():
    css = ui_mod.css_tema()
    assert "$" not in css  # todas las variables de la plantilla resueltas
    for token in (ui_mod.BG_PAGINA, ui_mod.BG_DRAWER, ui_mod.BG_POPUP, ui_mod.TXT, ui_mod.MUTED, ui_mod.ERROR_TXT):
        assert token in css
    for sel in ("body", ".q-menu", ".q-tooltip", ".q-dialog", ".q-notification", ".q-table", ".q-tab",
                ".q-field--error", ".q-item--active", ".nicegui-markdown"):
        assert sel in css, sel
    assert "color-scheme:dark" in css and "color-scheme" in ui_mod.HEAD_TEMA


def _fuente_fuera_de_paleta():
    src = inspect.getsource(ui_mod)
    ini = src.index("# ===================================== PALETA")
    fin = src.index("# =================================== FIN PALETA")
    return src[:ini] + src[fin:]


def test_sin_literales_de_color_fuera_de_la_paleta():
    resto = _fuente_fuera_de_paleta()
    assert not re.findall(r"#[0-9a-fA-F]{6}\b", resto), "hex fuera del bloque PALETA"
    assert not re.findall(r"rgba?\(\s*\d", resto), "rgb()/rgba() fuera del bloque PALETA"
    assert not re.findall(r"""["'](white|black|grey|gray)["']""", resto)
    assert not re.search(r"text-grey(?![-\w])", resto), "text-grey de bajo contraste: usar txt-muted"


def test_modo_oscuro_forzado_en_ui_run():
    src = inspect.getsource(ui_mod)
    assert "dark=True" in src and "ui.dark_mode(True)" in src and "ui.colors(**COLORES_QUASAR)" in src


def test_graficos_echarts_usan_tema_oscuro_explicito():
    """La pestana Valor Justo usa Apache ECharts: fondo, tooltip, ejes y leyenda salen de la paleta oscura."""
    for opc in (ui_mod.opciones_historico(None, None, "x"), ui_mod.opciones_barras(None, None, None, "x")):
        assert opc["backgroundColor"] == ui_mod.BG_PANEL and opc["animation"] is False
        tt = opc["tooltip"]
        assert tt["backgroundColor"] == ui_mod.BG_POPUP and tt["borderColor"] == ui_mod.BORDE
        assert contraste(tt["textStyle"]["color"], tt["backgroundColor"]) >= 4.5
        assert contraste(opc["title"]["textStyle"]["color"], ui_mod.BG_PANEL) >= 4.5
        assert contraste(opc["legend"]["textStyle"]["color"], ui_mod.BG_PANEL) >= 4.5
        assert contraste(opc["xAxis"]["axisLabel"]["color"], ui_mod.BG_PANEL) >= 4.5


def test_barras_del_perfil_visibles_sobre_el_panel():
    assert contraste(ui_mod.COLOR_BARRA_VP, ui_mod.BG_PANEL) >= 1.5  # barras no-HVN visibles sobre el panel (grafico)


# =====================================================================================================
# Ayuda contextual del panel lateral y construccion completa de la Vista (sin servidor ni red)
# =====================================================================================================
def test_ayudas_cubren_los_controles_del_panel():
    claves = {"ticker", "capital", "backtest", "sec", "recalcular", "valor_justo"} | set(ui_mod.TOGGLES)
    assert claves == set(ui_mod.AYUDAS)
    for k, texto in ui_mod.AYUDAS.items():
        assert 20 <= len(texto) <= 260, k
    for k in ui_mod.TOGGLES:
        assert "no se recalculan" in ui_mod.AYUDAS[k]
    assert "Recalcular" in ui_mod.AYUDAS["backtest"] and "Recalcular" in ui_mod.AYUDAS["sec"]


@pytest.fixture
def cliente():
    from nicegui.client import Client
    from nicegui.page import page
    with Client(page("/")) as c:
        yield c


def _guardar_usuario(tickers):
    from src.reporting import userdata
    userdata.guardar(tickers)


def test_vista_construye_tabs_ayudas_y_tema_oscuro(cliente):
    from nicegui import ui
    v = ui_mod.Vista()
    v.construir()
    els = list(cliente.elements.values())
    assert [e.props["name"] for e in els if isinstance(e, ui.tab)] == ["analisis", "valor_justo", "watchlist", "modelo"]
    assert [e.props["label"] for e in els if isinstance(e, ui.tab)] == ["Análisis", "Valor Justo", "Watchlist", "Modelo"]
    iconos = [e.props["data-ayuda"] for e in els if isinstance(e, ui.icon) and "data-ayuda" in e.props]
    assert sorted(iconos) == sorted(ui_mod.AYUDAS)
    assert len([e for e in els if isinstance(e, ui.tooltip)]) >= len(ui_mod.AYUDAS)
    assert any(isinstance(e, ui.dark_mode) and e.value is True for e in els)
    col = next(e for e in els if isinstance(e, ui.colors))
    assert col.props["dark-page"] == ui_mod.BG_PAGINA and col.props["primary"] == ui_mod.COLORES_QUASAR["primary"]
    assert "color-scheme" in cliente._head_html and ui_mod.BG_POPUP in cliente._head_html
    assert not [e for e in els if isinstance(e, ui.table) and "dark" not in e.props]
    assert len([e for e in els if isinstance(e, ui.expansion)]) == 10 + 6  # pestaña Modelo + 6 acordeones de Valor Justo


def test_vista_refresca_con_datos_sinteticos(cliente, df_sintetico, monkeypatch):
    res = _res_sintetico(df_sintetico, _zonas_ricas())
    monkeypatch.setitem(ui_mod.CACHE, "TEST", res)
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "TEST"
    v.refrescar()
    assert v.tarj["precio"].text == "$100.00" and v.tarj["confiabilidad"].text == "40.0/100"
    assert v.tarj_cob.text == "cobertura 80 %"
    assert len(v.tabla.rows) == 3
    assert v.grafico_datos["c"][-1] > 0 and len(v.grafico_overlays["zonas"]) == 3
    assert v.grafico_overlays["titulo"].startswith("TEST - Mapa de niveles")
    v.ticker = None
    v.refrescar()
    assert v.tarj_cob.text == ""


# =====================================================================================================
# Pestaña "Modelo": contenido como datos (sin red), fidelidad con el codigo y con docs/METODOLOGIA.md
# =====================================================================================================
SECCIONES = ["que_es", "logica", "familias", "grafico", "panel", "metricas", "confiabilidad", "limitaciones", "watchlist",
             "valor_justo"]


def _sec(secciones, id_):
    return next(s for s in secciones if s["id"] == id_)


def _txt(id_):
    return ui_mod.texto_seccion(_sec(ui_mod.contenido_modelo(), id_))


def _tabla_pesos(secciones):
    for b in _sec(secciones, "confiabilidad")["parrafos"]:
        if isinstance(b, dict) and "tabla" in b and b["tabla"]["columnas"][0] == "Componente" \
                and "Peso a priori" in b["tabla"]["columnas"][1]:
            return b["tabla"]
    raise AssertionError("tabla de pesos no encontrada")


def _num_pct(celda):
    return float(celda.replace(" %", "").replace(",", "."))


def _comp_por_etiqueta():
    return {v: k for k, v in ui_mod.ETIQUETAS_COMPONENTES.items()}


def test_modelo_secciones_obligatorias_y_estructura():
    secs = ui_mod.contenido_modelo()
    assert [s["id"] for s in secs] == SECCIONES
    for s in secs:
        assert s["titulo"].startswith(f"{SECCIONES.index(s['id']) + 1}. ") and s["parrafos"]
        assert all(isinstance(b, str) or set(b) <= {"lista", "tabla", "ejemplo", "aviso"} for b in s["parrafos"])
    assert _sec(secs, "que_es")["abierta"] and _sec(secs, "limitaciones")["abierta"]
    assert ui_mod.texto_modelo(secs) == ui_mod.texto_modelo()


@pytest.mark.parametrize("id_, claves", [
    ("que_es", ["En una frase", "nivel de compra", "no es un precio mágico", "invarianza", "cierre de un día",
                "95 a 98 USD"]),
    ("familias", ["HVN / Volume Profile", "Order Blocks", "AVWAP", "Swing Lows macro", "LogReg", "DCF / P/E",
                  "Analogía", "N1 a N4", "Confluencia", "95 a 98 USD", "3 familias", "1 familia",
                  "Score de confluencia"]),
    ("grafico", ["Velas", "Bandas de colores", "Form 4", "▲", "13F", "◆", "Volume Profile", "naranjas", "doradas",
                 "línea blanca"]),
    ("panel", ["Ticker", "Capital USD", "1000 USD", "3/6", "500,00", "5,2083", "Backtest (lento)", "SEC: Form 4 / 13F",
               "no se recalculan", "Recalcular", "Necesario", "No necesario", "por ticker",
               "no se reparte entre tickers"]),
    ("metricas", ["Fill Rate", "10 órdenes", "Bounce Rate", "MAE", "compré a 100", "-10 %", "60 / 120 / 252",
                  "Dist. a r_sup", "Acciones", "Precio prom. pond.", "rango primario", "no son comparables",
                  "muestra pequeña: N fills", "backtest de 5 años"]),
    ("confiabilidad", ["Cobertura", "se excluye", "reescalan", "Score de ranking", "promedio simple", "Score de confluencia",
                       "NO", "probabilidad", "cinco componentes", "no es un cero ni un valor neutro", "sin dato",
                       "solo cuentan los gestores activos", "no respondieron", "cobertura distinta"]),
    ("limitaciones", ["No es asesoría financiera", "NO es una probabilidad", "poder predictivo", "Spearman",
                      "casi idéntica a la línea base", "MAE es sistemáticamente peor", "se repiten en todas las filas",
                      "rango primario", "Sesgo de la muestra", "supervivencia", "no coinciden mejor que el azar",
                      "point-in-time", "costos de transacción", "ayuda de organización, no una señal", "domina",
                      "no respalda que más familias", "muestra pequeña: N fills"]),
])
def test_modelo_contenido_obligatorio_por_seccion(id_, claves):
    txt = _txt(id_)
    faltan = [c for c in claves if c.lower() not in txt.lower()]
    assert not faltan, faltan


def test_modelo_aviso_no_es_asesoria_visible_en_la_capa_de_render(cliente):
    from nicegui import ui
    ui_mod.renderizar_modelo(ui_mod.contenido_modelo())
    els = list(cliente.elements.values())
    exp = [e for e in els if isinstance(e, ui.expansion)]
    assert [e.props["data-seccion"] for e in exp] == SECCIONES
    assert [e.props["model-value"] for e in exp] == [True, True, False, False, False, False, False, True, False, False]
    md = " ".join(str(getattr(e, "content", "")) for e in els if isinstance(e, ui.markdown))
    assert "No es asesoría financiera" in md and md.count("No es asesoría financiera") >= 2  # arriba y en Limitaciones
    assert any(isinstance(e, ui.card) and "aviso-card" in e._classes for e in els)  # cuadro de aviso fuera de las expansiones


def test_modelo_pesos_coinciden_con_cargar_pesos():
    from src.validation import scoring
    t = _tabla_pesos(ui_mod.contenido_modelo())
    reales, prior = scoring.cargar_pesos(), scoring.PESOS_PRIOR
    por_etiqueta = _comp_por_etiqueta()
    assert len(t["filas"]) == len(scoring.COMPONENTES) == 5
    assert sum(_num_pct(f[2]) for f in t["filas"]) == pytest.approx(100.0, abs=0.3)
    for fila in t["filas"]:
        k = por_etiqueta[fila[0]]
        assert _num_pct(fila[2]) == pytest.approx(reales[k] * 100, abs=0.05), k
        assert _num_pct(fila[1]) == pytest.approx(prior[k] * 100, abs=0.05), k
        assert fila[3]  # explica como se normaliza


def test_modelo_sin_pesos_hardcodeados(monkeypatch):
    """Si cambian los pesos (otro pesos_calibrados.json), el texto cambia: no hay cifras copiadas a mano."""
    from src.validation import scoring
    reales = dict(scoring.cargar_pesos())
    falsos = {"respaldo_13f_form4": 0.41, "stock_act": 0.07, "absorcion": 0.13, "backtest": 0.22, "confluencia": 0.17}
    priors = {"respaldo_13f_form4": 0.05, "stock_act": 0.35, "absorcion": 0.15, "backtest": 0.28, "confluencia": 0.17}
    monkeypatch.setattr(scoring, "cargar_pesos", lambda: falsos)
    monkeypatch.setattr(scoring, "PESOS_PRIOR", priors)
    secs = ui_mod.contenido_modelo()
    t = _tabla_pesos(secs)
    por_etiqueta = _comp_por_etiqueta()
    for fila in t["filas"]:
        k = por_etiqueta[fila[0]]
        assert _num_pct(fila[2]) == pytest.approx(falsos[k] * 100) and _num_pct(fila[1]) == pytest.approx(priors[k] * 100)
    texto = ui_mod.texto_seccion(_sec(secs, "confiabilidad"))
    for k, w in reales.items():
        assert f"{w * 100:.1f}".replace(".", ",") + " %" not in texto, f"peso real de {k} sigue en el texto"
    # el ejemplo numerico se recalcula con los pesos vigentes (sin STOCK Act: cobertura = 1 - peso de STOCK Act)
    assert ui_mod._pct(1 - falsos["stock_act"]) in texto and ui_mod._pct(falsos["stock_act"]) in texto


def test_modelo_ejemplo_de_confiabilidad_coincide_con_scoring():
    from src.validation import scoring
    secs = ui_mod.contenido_modelo()
    comp = {"respaldo_13f_form4": 50.0, "stock_act": None, "absorcion": 26.0, "backtest": 60.0, "confluencia": 40.0}
    r = scoring.calcular_indice_confiabilidad(comp)
    txt = ui_mod.texto_seccion(_sec(secs, "confiabilidad"))
    assert ui_mod._es(r["indice"], 1) in txt and ui_mod._pct(r["cobertura"]) in txt
    tabla = next(b["tabla"] for b in _sec(secs, "confiabilidad")["parrafos"] if isinstance(b, dict) and "tabla" in b
                 and b["tabla"]["columnas"][1] == "Valor (0-100)")
    aportes = [float(f[3].replace(",", ".")) for f in tabla["filas"] if f[3] != "-"]
    assert sum(aportes) == pytest.approx(r["indice"], abs=0.3)  # redondeos de 0,1 por fila


def test_modelo_normalizacion_consistente_con_scoring():
    """Las formulas descritas en palabras (50/30/20, 60/40, 15 pts, 1 M USD, +15 %, -25 %) son las del codigo."""
    from src.validation import scoring as sc
    assert sc.s_13f(10, 10, 0) == pytest.approx(50) and sc.s_13f(0, 10, 1_000_000) == pytest.approx(50)
    assert sc.s_13f(0, 10, 5_000_000) == pytest.approx(50)  # compras de insiders saturan en 1 millon de USD
    assert sc.s_stock_act(15) == pytest.approx(100) and sc.s_stock_act(7.5) == pytest.approx(50)
    assert sc.s_absorcion(0.26) == pytest.approx(26)
    assert sc.s_backtest(1.0, 0.15, 0.0) == pytest.approx(100) and sc.s_backtest(1.0, 0.45, 0.0) == pytest.approx(100)
    assert sc.s_backtest(1.0, 0.0, 0.0) == pytest.approx(70)     # 50 % bounce + 20 % MAE sin caida
    assert sc.s_backtest(0.0, 0.15, 0.0) == pytest.approx(50)    # 30 % retorno + 20 % MAE
    assert sc.s_backtest(1.0, 0.0, -0.25) == pytest.approx(50)   # MAE de -25 % aporta cero
    assert sc.s_confluencia(1, 0) == 0 and sc.s_confluencia(4, 0) == pytest.approx(60)
    assert sc.s_confluencia(1, 2) == pytest.approx(40) and sc.s_confluencia(4, 2) == pytest.approx(100)
    norm = ui_mod.normalizacion_componentes(ui_mod._parametros())
    assert set(norm) == set(sc.COMPONENTES)  # un componente nuevo en scoring obliga a documentarlo aqui
    for fragmento in ("1 millón", "15 puntos", "50 %", "30 %", "20 %", "+15 %", "-25 %"):
        assert fragmento in " ".join(norm.values()), fragmento
    assert "60 %" in norm["confluencia"] and "40 %" in norm["confluencia"]


def test_modelo_cifras_de_limitaciones_estan_en_metodologia():
    if not os.path.exists(ui_mod.RUTA_METODOLOGIA):
        pytest.skip("docs/METODOLOGIA.md no disponible")
    with open(ui_mod.RUTA_METODOLOGIA, encoding="utf-8") as f:
        doc = f.read()
    assert ui_mod.FECHA_AUDITORIA in doc
    for k, v in ui_mod.CIFRAS_AUDITORIA.items():
        assert v in doc, f"{k}={v} ya no aparece en docs/METODOLOGIA.md: actualiza CIFRAS_AUDITORIA"
    lim = _txt("limitaciones")
    for k in ("spearman_bounce", "spearman_ret60", "mediana252_fill", "mediana252_base", "mae60_mediana", "mae60_base",
              "mae252_mediana", "mae252_base", "solape_rodante", "solape_control", "solape_estricto"):
        assert ui_mod._c(k) in lim, k


def test_modelo_limitaciones_incluyen_evidencia_del_json():
    from src.validation import scoring
    if not os.path.exists(scoring.RUTA_PESOS):
        pytest.skip("pesos_calibrados.json no disponible")
    with open(scoring.RUTA_PESOS, encoding="utf-8") as f:
        ev = json.load(f)["evidencia"]
    lim = _txt("limitaciones")
    for k in scoring.COMPONENTES:
        assert ev[k]["detalle"] in lim, k


def test_modelo_lee_parametros_del_codigo(monkeypatch):
    from src.levels import volume_liquidity as vl
    from src.validation import backtest_ranges as br
    monkeypatch.setattr(br, "BOUNCE_PCT", 0.07)
    monkeypatch.setattr(br, "BOUNCE_VENTANA", 33)
    monkeypatch.setattr(vl, "VP_ANIOS", 4)
    monkeypatch.setattr(ui_mod.nucleo, "MAX_TRAMOS_CAPITAL", 2)
    m = ui_mod.texto_seccion(_sec(ui_mod.contenido_modelo(), "metricas"))
    assert "+7 %" in m and "33 ruedas" in m and "+5 %" not in m
    assert "4 años" in _txt("grafico") and "durante los últimos 4 años" in _txt("familias")
    assert "Solo reciben capital las 2 mejores zonas" in _txt("panel")


def test_modelo_ejemplo_capital_usa_asignar_capital():
    secs = ui_mod.contenido_modelo()
    tabla = next(b["tabla"] for b in _sec(secs, "panel")["parrafos"] if isinstance(b, dict) and "tabla" in b
                 and b["tabla"]["columnas"][0] == "Zona")
    esperado = ui_mod._ejemplo_capital(1000.0)
    zonas = esperado["zonas"]
    assert [f[1] for f in tabla["filas"][:3]] == ["3", "2", "1"]
    assert sum(z["capital_asignado"] for z in zonas) == pytest.approx(1000.0)
    for fila, z in zip(tabla["filas"], zonas):
        assert fila[2] == ui_mod._es(z["capital_asignado"], 2) and fila[4] == ui_mod._es(z["acciones"], 4)
    # el reparto es proporcional al numero de familias y coincide con la funcion que usa la app
    assert [round(z["capital_asignado"], 2) for z in zonas] == [500.0, 333.33, 166.67]
    reales = [{"nivel": 1, "r_inf": 94.0, "r_sup": 96.0, "n_familias": 3, "familias": ["volumen", "avwap", "ob"],
               "tipos": ["POC", "AVWAP", "OB"], "dist_pct": -4.0},
              {"nivel": 2, "r_inf": 85.0, "r_sup": 88.0, "n_familias": 2, "familias": ["swing", "logreg"],
               "tipos": ["SWING", "LOG"], "dist_pct": -12.0},
              {"nivel": 3, "r_inf": 78.0, "r_sup": 80.0, "n_familias": 1, "familias": ["ob"], "tipos": ["OB"],
               "dist_pct": -20.0}]
    via_ui = ui_mod.filtrar_zonas(reales, _todas(), [], 1000.0)  # la misma ruta que usa la UI al mostrar la tabla
    assert [round(z["capital_asignado"], 2) for z in via_ui] == [500.0, 333.33, 166.67]
    assert [ui_mod._es(z["acciones"], 4) for z in via_ui] == [f[4] for f in tabla["filas"][:3]]  # acciones mostradas
    assert ui_mod._es(ui_mod.precio_promedio_ponderado(via_ui), 2) in _txt("panel")


def _ohlc(n, cambios=None):
    idx = pd.bdate_range("2024-01-02", periods=n)
    h = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 100.5, "Close": 100.0, "Volume": 1e6}, index=idx)
    h.loc[h.index[1], "Low"] = 99.5  # la orden limite en 100 se ejecuta en la rueda 1 (Low <= r_sup)
    for (i, col), v in (cambios or {}).items():
        h.loc[h.index[i], col] = v
    return h


def _orden(h, r_inf=96.0, r_sup=100.0):
    from src.validation import backtest_ranges as br
    return br.evaluar_orden(h, 0, r_inf, r_sup, np.full(len(h), np.nan))


def test_modelo_definiciones_coinciden_con_backtest_ranges():
    """Fill rate, Bounce (+5 % antes de r_inf - 3 % en 20 ruedas), MAE y retorno: los ejemplos del texto son los del codigo."""
    # Fill: se ejecuta si el minimo del dia toca el techo; sin toque no hay ejecucion
    sin_toque = _ohlc(30, {(1, "Low"): 100.5})
    assert _orden(sin_toque)["llenada"] is False and _orden(_ohlc(30))["llenada"] is True
    # Bounce: meta = entrada * 1.05 = 105; invalidacion = r_inf * 0.97 = 93,12
    assert _orden(_ohlc(40, {(5, "High"): 105.0}))["bounce"] is True
    assert _orden(_ohlc(40, {(5, "High"): 104.9}))["bounce"] is False                 # no llega a +5 %
    assert _orden(_ohlc(40, {(3, "Low"): 93.0, (5, "High"): 105.0}))["bounce"] is False   # cae antes (< 93,12)
    assert _orden(_ohlc(40, {(3, "Low"): 93.2, (5, "High"): 105.0}))["bounce"] is True    # 93,2 > 93,12: no invalida
    assert _orden(_ohlc(40, {(3, "Low"): 93.0, (3, "High"): 106.0}))["bounce"] is False   # mismo dia: cuenta como fallo
    assert _orden(_ohlc(40, {(22, "High"): 105.0}))["bounce"] is False                # fuera de las 20 ruedas
    # MAE y retorno a 60 ruedas: "compre a 100, minimo posterior 90 = -10 %", "cierre 106 = +6 %"
    o = _orden(_ohlc(100, {(10, "Low"): 90.0, (61, "Close"): 106.0}))
    assert o["entrada"] == 100.0 and o["mae60"] == pytest.approx(-0.10) and o["ret60"] == pytest.approx(0.06)
    txt = _txt("metricas")
    assert "93,12" in txt and "compré a 100" in txt and "-10 %" in txt and "+6 %" in txt


def test_modelo_texto_markdown_sin_enfasis_accidentales():
    import markdown2
    assert ui_mod._esc_md("r_sup y `a_b` y PE_P25") == "r\\_sup y `a_b` y PE\\_P25"
    html = markdown2.markdown(ui_mod._esc_md("el techo r_sup y el piso r_inf (PE_P25, DCF_MoS30)"))
    assert "<em>" not in html and "r_sup y el piso r_inf" in html and "DCF_MoS30" in html
    for s in ui_mod.contenido_modelo():
        for b in s["parrafos"]:
            if isinstance(b, dict) and "tabla" in b:
                n = len(b["tabla"]["columnas"])
                assert all(len(f) == n for f in b["tabla"]["filas"]), s["id"]
                md = ui_mod._md_tabla(b["tabla"])
                assert md.count("\n") == len(b["tabla"]["filas"]) + 1
                assert "<em>" not in markdown2.markdown(md, extras=["tables"]), s["id"]


# =====================================================================================================
# Correcciones del auditor: cobertura / datos no disponibles, avisos, distancia a r_sup, capital por ticker,
# ticker inicial coherente y pestaña Modelo actualizada (todo sin red)
# =====================================================================================================
def _res_con_avisos(df=None, zonas=None):
    res = _res_sintetico(df, zonas if zonas is not None else _zonas_ricas())
    res.update(
        confiabilidad={"indice": 41.2, "cobertura": 0.65,
                       "detalle": {"absorcion": {"valor": 26.0, "peso_efectivo": 0.5},
                                   "backtest": {"valor": 60.0, "peso_efectivo": 0.3},
                                   "confluencia": {"valor": 40.0, "peso_efectivo": 0.2}}},
        componentes_sin_dato=["respaldo_13f_form4", "stock_act"],
        avisos=["muestra pequeña: 4 fills (< 10); Bounce, Fill rate y MAE del backtest no son concluyentes",
                "13f: no respondieron: Bridgewater, State Street",
                "STOCK Act no disponible: sin red (excluido de la confiabilidad; no cuenta como neutro)"],
        n_fills=4, n_ordenes=11, anios_backtest=5)
    return res


def _textos_avisos(v):
    return [e.text for e in v.avisos.default_slot.children if hasattr(e, "text")]


def test_resumen_tarjetas_cobertura_sin_dato_y_avisos_nuevos():
    res = _res_con_avisos()
    z = ui_mod.filtrar_zonas(res["zonas"], _todas(), [], 1000.0)
    d = ui_mod.resumen_tarjetas(res, z)
    assert d["cobertura"] == 0.65
    assert d["sin_dato"] == ["Respaldo 13F / Form 4", "STOCK Act (Congreso)"]  # excluidos, con su etiqueta legible
    assert any("muestra pequeña: 4 fills" in a for a in d["avisos"])
    assert any("13f: no respondieron: Bridgewater" in a for a in d["avisos"])
    assert any("STOCK Act no disponible" in a for a in d["avisos"])
    assert d["avisos_datos"] == res["avisos"]
    assert d["errores"] == []  # no son errores de etapa: se muestran como avisos


def test_sin_dato_se_deriva_de_detalle_si_falta_la_lista():
    res = _res_con_avisos()
    del res["componentes_sin_dato"]
    assert set(ui_mod.nombres_sin_dato(res)) == {"Respaldo 13F / Form 4", "STOCK Act (Congreso)"}
    assert ui_mod.nombres_sin_dato(_res_sintetico(None, [])) == []  # sin 'detalle' no se afirma nada


def test_texto_alcance_backtest_rango_primario_anios_y_fills():
    t = ui_mod.texto_alcance_backtest(_res_con_avisos())
    assert "rango primario (N1)" in t and "backtest de 5 años" in t and "4 fills de 11 órdenes" in t
    assert "no comparable con el pool" in t and "r_sup" in t
    assert "Sin backtest" in ui_mod.texto_alcance_backtest({})


def test_distancia_unica_a_r_sup_en_tabla_ranking_y_figura(df_sintetico):
    """tabla, ranking y etiqueta del grafico miden la distancia al MISMO punto: r_sup (nunca al centro)."""
    zonas = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], 1000.0)
    px = 100.0
    esperado = [(z["r_sup"] / px - 1) * 100 for z in zonas]
    assert [f["dist_pct"] for f in ui_mod.filas_tabla(zonas, None, px)] == pytest.approx(esperado)
    rk = ui_mod.filas_ranking([{"ticker": "A", "precio_actual": px, "zonas": zonas, "score_ranking": 1.0}])[0][0]
    assert rk["dist_pct"] == pytest.approx(esperado[0])
    etiquetas = [b["etiqueta"] for b in ui_mod.bandas_zonas(zonas, px)]
    assert len(etiquetas) == 3
    for z, e in zip(zonas, etiquetas):
        assert ui_mod.ETIQUETA_DIST_RSUP in e and f"{(z['r_sup'] / px - 1) * 100:+.1f}%" in e
        centro = (z["r_inf"] + z["r_sup"]) / 2
        assert f"{(centro / px - 1) * 100:+.1f}%" not in e  # N1: centro -7.5 %, r_sup -5.0 %


def test_filas_ranking_cobertura_y_sin_dato():
    def r(t, score, cob, sin):
        return {"ticker": t, "precio_actual": 10.0, "score_ranking": score, "confluencia_score": score,
                "confiabilidad": {"indice": score, "cobertura": cob}, "componentes_sin_dato": sin,
                "zonas": [{"r_inf": 1.0, "r_sup": 2.0}]}
    filas, _ = ui_mod.filas_ranking([r("A", 80.0, 1.0, []), r("B", 60.0, 0.65, ["stock_act", "backtest"]),
                                     r("C", 50.0, None, [])])
    por = {f["ticker"]: f for f in filas}
    assert por["A"]["cobertura"] == pytest.approx(100.0) and por["A"]["sin_dato"] == ""
    assert por["B"]["cobertura"] == pytest.approx(65.0) and por["B"]["sin_dato"] == "STOCK Act (Congreso), Backtest"
    assert por["C"]["cobertura"] is None
    assert ui_mod.coberturas_distintas(filas) is True
    assert ui_mod.coberturas_distintas(filas[:1]) is False
    assert ui_mod.coberturas_distintas([{"cobertura": 65.0}, {"cobertura": 65.2}]) is False


def test_nota_ranking_calcula_el_peso_de_la_confluencia_al_vuelo(monkeypatch):
    from src.validation import scoring
    assert f"≈{ui_mod._es(ui_mod.nucleo.peso_confluencia_ranking() * 100, 1)} %" in ui_mod.nota_ranking()
    monkeypatch.setattr(scoring, "cargar_pesos", lambda: {"confluencia": 0.30})
    assert "≈65,0 %" in ui_mod.nota_ranking()  # 0.5 + 0.5 * 0.30
    assert "no una señal" in ui_mod.nota_ranking() and "no son comparables" in ui_mod.nota_ranking()


def test_vista_ticker_inicial_coincide_con_el_select(cliente):
    v = ui_mod.Vista()
    v.construir()
    esperado = ui_mod.validar_ticker(v.watchlist[0]) if v.watchlist else None
    assert v.ticker == esperado and v.sel_ticker.value == esperado  # lo que se ve es lo que se recalcula


def test_vista_muestra_el_resultado_cacheado_del_ticker_inicial(cliente, df_sintetico, monkeypatch):
    _guardar_usuario(["TEST", "AAA"])
    monkeypatch.setitem(ui_mod.CACHE, "TEST", _res_con_avisos(df_sintetico))
    v = ui_mod.Vista()
    v.construir()  # sin tocar el select: antes self.ticker era None y la pagina mostraba "n/d" con TEST seleccionado
    assert v.ticker == "TEST" and v.tarj["precio"].text == "$100.00"


def test_vista_sin_cache_invita_a_recalcular_el_ticker_mostrado(cliente, monkeypatch):
    _guardar_usuario(["ZZZ"])
    monkeypatch.delitem(ui_mod.CACHE, "ZZZ", raising=False)
    v = ui_mod.Vista()
    v.construir()
    assert any("ZZZ" in t and "Recalcular" in t for t in _textos_avisos(v))
    _guardar_usuario([])
    v2 = ui_mod.Vista()
    v2.construir()
    assert v2.ticker is None and v2.sel_ticker.value is None


def test_vista_tarjeta_confiabilidad_muestra_cobertura_y_excluidos(cliente, df_sintetico, monkeypatch):
    monkeypatch.setitem(ui_mod.CACHE, "TEST", _res_con_avisos(df_sintetico))
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "TEST"
    v.refrescar()
    assert v.tarj_cob.text == "cobertura 65 %"
    assert v.tarj_excl.text == "sin dato: Respaldo 13F / Form 4, STOCK Act (Congreso)"
    textos = _textos_avisos(v)
    assert any("muestra pequeña: 4 fills" in t for t in textos)
    assert any("13f: no respondieron" in t for t in textos) and any("STOCK Act no disponible" in t for t in textos)
    assert "rango primario (N1)" in v.lbl_bt.text and "5 años" in v.lbl_bt.text
    v.ticker = None
    v.refrescar()
    assert v.tarj_excl.text == ""


def test_vista_watchlist_nota_de_capital_cobertura_y_distancia(cliente):
    from nicegui import ui
    v = ui_mod.Vista()
    v.construir()
    textos = [e.text for e in cliente.elements.values() if isinstance(e, ui.label)]
    assert ui_mod.nucleo.NOTA_CAPITAL_BATCH in textos and ui_mod.nota_ranking() in textos
    assert "no se reparte entre tickers" in v.nota_capital.text
    cols = {c["name"]: c["label"] for c in v.tabla_wl.columns}
    assert "cobertura" in cols and cols["dist_pct"] == "Dist. a r_sup %"
    assert {c["name"]: c["label"] for c in v.tabla.columns}["dist_pct"] == "Dist. a r_sup %"
    assert "q-tooltip" in v.tabla_wl.slots["body-cell-cobertura"].template  # tooltip con los componentes sin dato


def test_ayuda_capital_dice_por_ticker():
    assert "POR TICKER" in ui_mod.AYUDAS["capital"] and "no se reparte entre tickers" in ui_mod.AYUDAS["capital"]


def test_modelo_gestores_13f_se_leen_del_codigo(monkeypatch):
    from src.data import sec_edgar
    monkeypatch.setattr(sec_edgar, "GESTORES_13F", {"Fondo Alfa": 1, "Fondo Beta": 2, "Indexa": 3})
    monkeypatch.setattr(sec_edgar, "GESTORES_PASIVOS", frozenset({"Indexa"}))
    assert ui_mod.gestores_13f() == {"activos": ["Fondo Alfa", "Fondo Beta"], "pasivos": ["Indexa"]}
    t = _txt("confiabilidad")
    assert "Fondo Alfa, Fondo Beta" in t and "Indexa" in t and "consulta 3 gestores" in t
    assert "Bridgewater" not in t  # nada hardcodeado


def test_modelo_gestores_reales_activos_y_pasivos():
    from src.data import sec_edgar
    g = ui_mod.gestores_13f()
    assert set(g["activos"]) | set(g["pasivos"]) == set(sec_edgar.GESTORES_13F)
    assert set(g["pasivos"]) == set(sec_edgar.GESTORES_PASIVOS) & set(sec_edgar.GESTORES_13F)
    t = _txt("confiabilidad")
    assert all(n in t for n in sec_edgar.GESTORES_13F)


def test_modelo_ranking_dominado_por_confluencia_se_calcula_al_renderizar(monkeypatch):
    from src.validation import scoring
    real = scoring.cargar_pesos()
    esperado = f"≈{ui_mod._es((0.5 + 0.5 * real['confluencia']) * 100, 1)} %"
    lim = _txt("limitaciones")
    assert esperado in lim and "ayuda de organización, no una señal" in lim
    assert "no respalda que más familias coincidiendo mejoren el retorno" in lim
    monkeypatch.setattr(scoring, "cargar_pesos", lambda: {**real, "confluencia": 0.30})
    assert "≈65,0 %" in _txt("limitaciones")


def test_modelo_muestra_pequena_usa_la_constante_del_codigo(monkeypatch):
    monkeypatch.setattr(ui_mod.nucleo, "MIN_FILLS_AVISO", 25)
    assert "menos de 25 ejecuciones simuladas" in _txt("limitaciones")
    assert "menos de 25 ejecuciones simuladas" in _txt("metricas")


def test_modelo_normalizacion_13f_dice_activos_y_exclusion():
    norm = ui_mod.normalizacion_componentes(ui_mod._parametros())
    assert "ACTIVOS" in norm["respaldo_13f_form4"] and "se excluye" in norm["respaldo_13f_form4"]
    assert "no vale 0" in norm["stock_act"]


def test_modelo_ventana_de_absorcion_son_velas_contando_la_del_toque():
    """El backtest mira range(k, k + VENTANA_ABS + 1): la vela del toque y las VENTANA_ABS siguientes."""
    from src.validation import backtest_ranges as br
    n = 30

    def hay_absorcion(j):  # vela de rechazo (tercio superior, volumen 5x la media) en la rueda j; el toque es la rueda 1
        h = _ohlc(n, {(j, "Low"): 99.0, (j, "High"): 101.0, (j, "Close"): 100.8, (j, "Volume"): 5e6})
        return br.evaluar_orden(h, 0, 96.0, 100.0, np.full(n, 1e6))["absorcion"]
    k = 1
    assert hay_absorcion(k) and hay_absorcion(k + br.VENTANA_ABS) and not hay_absorcion(k + br.VENTANA_ABS + 1)
    velas = f"{br.VENTANA_ABS + 1} velas"
    norm = ui_mod.normalizacion_componentes(ui_mod._parametros())["absorcion"]
    assert velas in norm and "la del toque" in norm and f"{br.VENTANA_ABS} ruedas desde el toque" not in norm
    lim = _txt("limitaciones")
    assert velas in lim and "la del toque" in lim and f"({br.VENTANA_ABS} ruedas)" not in lim


def test_modelo_valoracion_no_consultada_se_excluye_y_consultada_sin_anclas_vale_cero():
    conf = _txt("familias")
    assert "se consultó y ninguna ancla cae en la zona" in conf and "vale 0" in conf
    assert "no se pudo consultar" in conf and "se excluye" in conf and "reescala" in conf
    norm = ui_mod.normalizacion_componentes(ui_mod._parametros())["confluencia"]
    assert "no se pudo consultar" in norm and "se excluye" in norm and "vale 0" in norm
    assert "valoración sin consultar" in _txt("confiabilidad") and "ninguna ancla de valoración en la zona" in _txt("confiabilidad")


def test_modelo_13f_sin_gestor_activo_cuenta_como_no_disponible():
    t = _txt("confiabilidad")
    assert "no respondió ningún gestor activo" in t and "aunque sí respondan los pasivos" in t


# ---- Seccion "La logica detras: como y por que (en simple)" ---------------------------------------------------
def _logica():
    return _sec(ui_mod.contenido_modelo(), "logica")


def test_logica_existe_visible_y_justo_despues_de_la_seccion_1():
    secs = ui_mod.contenido_modelo()
    assert [s["id"] for s in secs][:2] == ["que_es", "logica"]
    s = _logica()
    assert s["abierta"] and s["titulo"].startswith("2. La lógica detrás") and "en simple" in s["titulo"]
    txt = ui_mod.texto_seccion(s)
    for c in ["La idea central", "Analogía", "paso a paso", "De dónde salen los datos", "Por qué debería funcionar",
              "Lo demostrado", "Limitaciones, en corto", "Material educativo; no es asesoría financiera"]:
        assert c.lower() in txt.lower(), c
    assert s["parrafos"][-1] == {"aviso": "Material educativo; no es asesoría financiera."}


def test_logica_menciona_cada_api_real_y_libreria():
    txt = ui_mod.texto_seccion(_logica())
    for c in ["yfinance", "SEC EDGAR", "edgartools", "13F-HR", "Form 4", "código P", "10-K", "10-Q", "STOCK Act",
              "kadoa-org/congress-trading-monitor", "smartmoneyconcepts", "45 días", "2 días hábiles", "24 horas",
              "dato no disponible"]:
        assert c in txt, c


def test_logica_funciones_citadas_existen_y_estan_en_el_texto():
    import importlib
    txt = ui_mod.texto_seccion(_logica())
    assert ui_mod.FUNCIONES_CITADAS
    for mod, fn in ui_mod.FUNCIONES_CITADAS:
        assert callable(getattr(importlib.import_module(mod), fn)), (mod, fn)
        assert fn in txt, fn
    for fn in ("obtener_13f_ticker", "compras_insiders"):
        assert fn in txt
    from src.data import stock_act
    assert stock_act.REPO_URL.endswith("kadoa-org/congress-trading-monitor")


def test_logica_apis_reales_declaradas_en_requirements():
    import pathlib
    req = (pathlib.Path(ui_mod.__file__).parent / "requirements.txt").read_text(encoding="utf-8-sig").lower()
    for lib in ("yfinance", "edgartools", "smartmoneyconcepts"):
        assert lib in req


def test_logica_sin_promesas_de_rentabilidad():
    txt = ui_mod.texto_seccion(_logica()).lower()
    for prohibido in ["rentabilidad garantizada", "ganancia garantizada", "garantiza", "sí funciona", "si funciona",
                      "siempre rebota", "seguro que sube", "retorno asegurado"]:
        assert prohibido not in txt, prohibido
    assert "no se puede afirmar" in txt and "no discrimina" in txt


def test_logica_incluye_evidencia_negativa_con_cifras_de_metodologia():
    txt = ui_mod.texto_seccion(_logica())
    for k in ("spearman_bounce", "spearman_ret60", "mediana252_fill", "mediana252_base", "mae60_mediana", "mae60_base",
              "solape_rodante", "solape_control", "n_trimestres_13f"):
        assert ui_mod._c(k) in txt, k
    for c in ["no discrimina resultados", "MAE es peor", "no coinciden mejor que el azar", "sesgo de supervivencia",
              "no se probaron", "muestra es muy pequeña"]:
        assert c.lower() in txt.lower(), c
    for c in ["disciplina", "escalonadas", "trazabilidad", "cierre intradía"]:
        assert c in txt


def test_logica_ejemplo_semiconductores_marcado_como_ilustrativo():
    s = _logica()
    txt = ui_mod.texto_seccion(s)
    assert "ILUSTRATIVO, NO REAL" in txt and "inventad" in txt and "NVDA" in txt
    assert "cíclico" in txt and "volátil" in txt and "MAE" in txt and "escalonamiento" in txt
    assert "ciclo del sector" in txt
    ej = ui_mod.EJ_SEMI
    assert f"{ui_mod._es(ej['precio'], 0)} USD" in txt and "(inventado)" in txt
    # el bloque que introduce las cifras precede a las cifras
    i_marca = txt.index("ILUSTRATIVO, NO REAL")
    assert i_marca < txt.index(f"{ui_mod._es(ej['precio'], 0)} USD")


def test_logica_gestores_y_parametros_se_leen_del_codigo(monkeypatch):
    from src.data import sec_edgar
    monkeypatch.setattr(sec_edgar, "GESTORES_13F", {"Fondo Ficticio A": 1, "Vanguard Group": 2})
    txt = ui_mod.texto_seccion(_logica_nueva())
    assert "Fondo Ficticio A" in txt and "Berkshire" not in txt
    from src.validation import backtest_ranges as br
    monkeypatch.setattr(br, "VIDA_ORDEN", 77)
    assert "77 ruedas" in ui_mod.texto_seccion(_logica_nueva())


def _logica_nueva():
    return _sec(ui_mod.contenido_modelo(), "logica")


def test_logica_se_renderiza_en_su_expansion(cliente):
    from nicegui import ui
    ui_mod.renderizar_modelo(ui_mod.contenido_modelo())
    els = list(cliente.elements.values())
    exp = next(e for e in els if isinstance(e, ui.expansion) and e.props.get("data-seccion") == "logica")
    assert exp.props["model-value"] is True
    md = " ".join(str(getattr(e, "content", "")) for e in els if isinstance(e, ui.markdown))
    assert "Material educativo; no es asesoría financiera" in md
