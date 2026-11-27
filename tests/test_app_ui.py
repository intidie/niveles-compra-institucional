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
    assert callable(ui_mod.construir_figura) and callable(ui_mod.filtrar_zonas)


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
    assert filas[0]["dist_pct"] == -5.0
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


def _nombres(fig):
    return [t.name for t in fig.data]


def test_figura_completa(df_sintetico):
    z = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], 1000.0)
    av = [{"tipo": "DCF_MOS30", "familia": "valoracion", "precio": float(df_sintetico["Close"].iloc[-1])}]
    fig = ui_mod.construir_figura(df_sintetico, "TEST", z, None, _form4(df_sintetico), _r13f(df_sintetico), av, 1)
    nombres = _nombres(fig)
    assert any(n and n.startswith("Compra insider Form 4") for n in nombres)
    assert any(n and n.startswith("Acumulacion 13F") for n in nombres)
    assert fig.data[0].type == "candlestick"
    assert any(t.type == "bar" and t.orientation == "h" for t in fig.data)
    assert fig.layout.xaxis.rangeslider.visible is False
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.yaxis2.matches == "y"  # eje Y compartido
    assert fig.layout.xaxis.domain[1] > 0.7 and fig.layout.xaxis2.domain[0] > 0.7  # ~75/25
    # 3 bandas en el panel de precio; la seleccionada (N1) es la mas opaca
    ops = sorted(s.opacity for s in fig.layout.shapes if s.type == "rect" and s.yref == "y" and s.xref == "x domain")
    assert len(ops) == 3 and ops[-1] == pytest.approx(0.42)
    fig.to_json()


def test_figura_sin_marcadores_ni_zonas(df_sintetico):
    fig = ui_mod.construir_figura(df_sintetico, "TEST", [], None, None, None, None, None)
    assert fig.data[0].type == "candlestick"
    assert not any(n and ("Form 4" in n or "13F" in n) for n in _nombres(fig))
    fig2 = ui_mod.construir_figura(df_sintetico, "TEST", None, None, pd.DataFrame(), pd.DataFrame())
    assert fig2.data[0].type == "candlestick"


def test_figura_sin_datos():
    for d in (None, pd.DataFrame(), pd.DataFrame({"Close": [1.0]})):
        fig = ui_mod.construir_figura(d, "X", [], None)
        assert len(fig.data) == 0 and fig.layout.annotations
        fig.to_json()


def test_figura_indice_con_zona_horaria(df_sintetico):
    d = df_sintetico.copy()
    d.index = d.index.tz_localize("America/New_York")
    fig = ui_mod.construir_figura(d, "TZ", [], None, _form4(df_sintetico), None)
    assert fig.data[0].type == "candlestick" and len(fig.data[0].x) == len(d)


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
    assert contraste(m.MUTED, m.BG_MODEBAR, base=m.BG_PAPEL) >= 4.5         # iconos de la modebar


def test_colores_quasar_contraste():
    q = ui_mod.COLORES_QUASAR
    assert q["dark_page"] == ui_mod.BG_PAGINA and q["dark"] == ui_mod.BG_TARJETA
    assert contraste(ui_mod.BLANCO, q["primary"]) >= 4.5
    for k in ("positive", "negative", "info"):
        assert contraste(ui_mod.BLANCO, q[k]) >= 4.5, k
    assert contraste(ui_mod.TXT_SOBRE_AVISO, q["warning"]) >= 4.5


def test_paleta_publica_y_papel_plotly_opaco():
    assert {"BG_PAPEL", "BG_PANEL", "TXT", "MUTED", "GRILLA"} <= set(ui_mod.PALETA)
    assert _rgba(ui_mod.BG_PAPEL)[3] == 1.0, "papel transparente => hover/modebar de Plotly calculados sobre blanco"


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


def _hojas(obj, ruta=()):
    """Hojas (ruta, valor) de dicts anidados y listas de dicts; ignora arrays de datos."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _hojas(v, ruta + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, tuple)):
                yield from _hojas(v, ruta + (i,))
    else:
        yield ruta, obj


def _figura_rica(df):
    z = ui_mod.filtrar_zonas(_zonas_ricas(), _todas(), [], 1000.0)
    av = [{"tipo": "DCF_MoS30", "familia": "valoracion", "precio": float(df["Close"].iloc[-1])}]
    return ui_mod.construir_figura(df, "TEST", z, None, _form4(df), _r13f(df), av, 1, "sub")


def test_figura_hoverlabel_oscuro_explicito(df_sintetico):
    for fig in (_figura_rica(df_sintetico), ui_mod.construir_figura(None, "X")):
        hl = fig.layout.hoverlabel
        assert hl.bgcolor == ui_mod.BG_POPUP and hl.font.color == ui_mod.TXT and hl.bordercolor == ui_mod.BORDE
        assert contraste(hl.font.color, hl.bgcolor) >= 4.5
        assert hl.namelength == -1  # nombres largos de traza (Form 4 / 13F) sin truncar
        assert fig.layout.paper_bgcolor == ui_mod.BG_PAPEL and fig.layout.plot_bgcolor == ui_mod.BG_PANEL


def test_figura_cada_traza_con_hoverlabel_oscuro(df_sintetico):
    fig = _figura_rica(df_sintetico)
    tipos = {t.type for t in fig.data}
    assert {"candlestick", "scatter", "bar"} <= tipos
    assert len(fig.data) >= 6  # velas, SMA, 13F, Form 4, valoracion, Volume Profile
    for t in fig.data:
        assert t.hoverlabel.bgcolor == ui_mod.BG_POPUP, t.name
        assert t.hoverlabel.font.color == ui_mod.TXT, t.name
        assert contraste(t.hoverlabel.font.color, t.hoverlabel.bgcolor) >= 4.5
    assert fig.layout.hovermode == "x unified"
    assert fig.data[0].yhoverformat == ",.2f"  # el '$' ya lo pone tickprefix (evita '$$214.75' en el popup)
    assert fig.layout.yaxis.tickprefix == "$"


def test_figura_json_sin_fondos_claros_ni_fuentes_ilegibles(df_sintetico):
    """Recorre TODO el JSON de la figura (sin la plantilla): ningun bgcolor claro, ninguna fuente de bajo contraste."""
    for fig in (_figura_rica(df_sintetico), ui_mod.construir_figura(None, "X", [], None)):
        j = json.loads(fig.to_json())
        j["layout"].pop("template", None)
        n_bg = n_font = 0
        for ruta, v in _hojas(j):
            if not isinstance(v, str):
                continue
            if ruta[-1] == "bgcolor":
                n_bg += 1
                assert contraste(ui_mod.TXT, v, base=ui_mod.BG_PANEL) >= 4.5, (ruta, v)
            if ruta[-1] == "color" and any("font" in str(r).lower() for r in ruta[-3:-1]):
                n_font += 1
                assert contraste(v, ui_mod.BG_PANEL) >= 4.5 and contraste(v, ui_mod.BG_PAPEL) >= 4.5, (ruta, v)
        assert n_bg >= 3 and n_font >= 4


def test_figura_titulo_sin_ticker_no_muestra_guion_suelto():
    assert ui_mod.construir_figura(None, "-").layout.title.text == "Mapa de niveles institucionales"
    assert ui_mod.construir_figura(None, "AAPL").layout.title.text == "AAPL - Mapa de niveles institucionales"


def test_figura_anotaciones_legibles(df_sintetico):
    fig = _figura_rica(df_sintetico)
    anots = list(fig.layout.annotations)
    assert len(anots) >= 4  # 3 bandas + linea de valoracion
    for a in anots:
        fondo = a.bgcolor or ui_mod.BG_PANEL
        assert contraste(a.font.color, fondo, base=ui_mod.BG_PANEL) >= 4.5, a.text
    vacia = ui_mod.construir_figura(None, "X")
    assert contraste(vacia.layout.annotations[0].font.color, ui_mod.BG_PANEL) >= 4.5


def test_figura_modebar_leyenda_titulo_y_ejes(df_sintetico):
    for fig in (_figura_rica(df_sintetico), ui_mod.construir_figura(None, "X")):
        mb = fig.layout.modebar
        assert contraste(mb.color, ui_mod.BG_PAPEL) >= 4.5 and contraste(mb.activecolor, ui_mod.BG_PAPEL) >= 4.5
        assert contraste(ui_mod.TXT, mb.bgcolor, base=ui_mod.BG_PAPEL) >= 4.5
        assert fig.layout.legend.font.color == ui_mod.TXT
        assert fig.layout.font.color == ui_mod.TXT
        assert fig.layout.title.font.color == ui_mod.TXT
    fig = _figura_rica(df_sintetico)
    assert fig.layout.xaxis.tickfont.color == ui_mod.TXT and fig.layout.yaxis.tickfont.color == ui_mod.TXT
    assert fig.layout.xaxis.showspikes is True


def test_figura_colores_de_series_vienen_de_la_paleta(df_sintetico):
    fig = _figura_rica(df_sintetico)
    f4 = next(t for t in fig.data if t.name and t.name.startswith("Compra insider"))
    q13 = next(t for t in fig.data if t.name and t.name.startswith("Acumulacion 13F"))
    assert f4.marker.color == ui_mod.COLOR_FORM4 and q13.marker.color == ui_mod.COLOR_13F
    assert fig.data[0].increasing.line.color == ui_mod.VELA_ALZA
    barras = next(t for t in fig.data if t.type == "bar")
    assert set(barras.marker.color) <= {ui_mod.COLOR_HVN, ui_mod.COLOR_BARRA_VP}
    assert contraste(ui_mod.COLOR_BARRA_VP, ui_mod.BG_PANEL) >= 1.5  # barras no-HVN visibles sobre el panel (grafico)


# =====================================================================================================
# Ayuda contextual del panel lateral y construccion completa de la Vista (sin servidor ni red)
# =====================================================================================================
def test_ayudas_cubren_los_controles_del_panel():
    claves = {"ticker", "capital", "backtest", "sec", "recalcular"} | set(ui_mod.TOGGLES)
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


def test_vista_construye_tabs_ayudas_y_tema_oscuro(cliente):
    from nicegui import ui
    v = ui_mod.Vista()
    v.construir()
    els = list(cliente.elements.values())
    assert [e.props["name"] for e in els if isinstance(e, ui.tab)] == ["analisis", "watchlist", "modelo"]
    assert [e.props["label"] for e in els if isinstance(e, ui.tab)] == ["Análisis", "Watchlist", "Modelo"]
    iconos = [e.props["data-ayuda"] for e in els if isinstance(e, ui.icon) and "data-ayuda" in e.props]
    assert sorted(iconos) == sorted(ui_mod.AYUDAS)
    assert len([e for e in els if isinstance(e, ui.tooltip)]) >= len(ui_mod.AYUDAS)
    assert any(isinstance(e, ui.dark_mode) and e.value is True for e in els)
    col = next(e for e in els if isinstance(e, ui.colors))
    assert col.props["dark-page"] == ui_mod.BG_PAGINA and col.props["primary"] == ui_mod.COLORES_QUASAR["primary"]
    assert "color-scheme" in cliente._head_html and ui_mod.BG_POPUP in cliente._head_html
    assert not [e for e in els if isinstance(e, ui.table) and "dark" not in e.props]
    assert len([e for e in els if isinstance(e, ui.expansion)]) == 7  # pestaña Modelo


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
    assert v.grafico.figure.layout.hoverlabel.bgcolor == ui_mod.BG_POPUP
    v.ticker = None
    v.refrescar()
    assert v.tarj_cob.text == ""


# =====================================================================================================
# Pestaña "Modelo": contenido como datos (sin red), fidelidad con el codigo y con docs/METODOLOGIA.md
# =====================================================================================================
SECCIONES = ["que_es", "familias", "grafico", "panel", "metricas", "confiabilidad", "limitaciones"]


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
               "no se recalculan", "Recalcular", "Necesario", "No necesario"]),
    ("metricas", ["Fill Rate", "10 órdenes", "Bounce Rate", "MAE", "compré a 100", "-10 %", "60 / 120 / 252",
                  "Dist. %", "Acciones", "Precio prom. pond.", "rango primario"]),
    ("confiabilidad", ["Cobertura", "se excluye", "reescalan", "Score de ranking", "promedio simple", "Score de confluencia",
                       "NO", "probabilidad", "cinco componentes"]),
    ("limitaciones", ["No es asesoría financiera", "NO es una probabilidad", "poder predictivo", "Spearman",
                      "casi idéntica a la línea base", "MAE es sistemáticamente peor", "se repiten en todas las filas",
                      "rango primario", "Sesgo de la muestra", "supervivencia", "no coinciden mejor que el azar",
                      "point-in-time", "costos de transacción"]),
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
    assert [e.props["model-value"] for e in exp] == [True, False, False, False, False, False, True]
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
