"""Grafico de Analisis (TradingView Lightweight Charts): funciones puras que preparan el JSON, recursos estaticos y la
Vista. Sin red ni navegador (la verificacion visual con Playwright se hace aparte)."""
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pytest

import app_ui as ui_mod
from test_app_ui import _form4, _r13f, _zonas_ricas, contraste

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _leer(*partes):
    with open(os.path.join(RAIZ, *partes), encoding="utf-8") as f:
        return f.read()


def _df_largo(anios=12, fin="2026-06-30", semilla=1):
    """OHLCV diario de dias habiles (varios anios) para probar los limites de historial."""
    idx = pd.bdate_range(end=fin, periods=int(anios * 261))
    rng = np.random.default_rng(semilla)
    c = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, len(idx))))
    return pd.DataFrame({"Open": c * 0.998, "High": c * 1.012, "Low": c * 0.988, "Close": c,
                         "Volume": rng.integers(1_000_000, 9_000_000, len(idx)).astype(float)}, index=idx)


@pytest.fixture
def cliente():
    from nicegui.client import Client
    from nicegui.page import page
    with Client(page("/")) as c:
        yield c


# ====================================================================================== velas: OHLC -> JSON
def test_velas_json_columnas_tiempos_ascendentes_en_segundos(df_sintetico):
    d = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    n = len(df_sintetico)
    assert {len(d[k]) for k in ("t", "o", "h", "l", "c")} == {n}
    assert d["t"][0] == int(pd.Timestamp(df_sintetico.index[0]).timestamp())       # segundos UNIX (UTC)
    assert all(b > a for a, b in zip(d["t"], d["t"][1:]))                        # estrictamente ascendente
    assert d["c"][-1] == pytest.approx(float(df_sintetico["Close"].iloc[-1]), abs=1e-4)
    assert all(h >= max(o, c) and l <= min(o, c) for o, h, l, c in zip(d["o"], d["h"], d["l"], d["c"]))
    assert d["rango"] == [pytest.approx(df_sintetico["Low"].min(), abs=1e-3), pytest.approx(df_sintetico["High"].max(), abs=1e-3)]
    json.dumps(d, allow_nan=False)                                                 # JSON estricto (sin NaN)


def test_velas_json_es_compacto(df_sintetico):
    d = ui_mod.datos_grafico(_df_largo(10), "T", nube=False)
    kb = len(json.dumps({k: d[k] for k in ("t", "o", "h", "l", "c")}, separators=(",", ":"))) / 1024
    assert len(d["t"]) > 2500 and kb < 180      # 10 anios diarios: ~150 kB como columnas


def test_indice_con_zona_horaria_se_normaliza(df_sintetico):
    d = df_sintetico.copy()
    d.index = d.index.tz_localize("America/New_York")
    out = ui_mod.datos_grafico(d, "TZ", _form4(df_sintetico), None, nube=False)
    assert len(out["t"]) == len(d) and all(b > a for a, b in zip(out["t"], out["t"][1:]))


def test_filas_invalidas_y_duplicados_se_descartan(df_sintetico):
    d = df_sintetico.copy()
    d.iloc[5, d.columns.get_loc("Close")] = np.nan
    d.iloc[7, d.columns.get_loc("Open")] = 0.0
    d = pd.concat([d, d.iloc[[10]]])                    # indice duplicado
    d = d.sort_index(ascending=False)                   # desordenado
    out = ui_mod.datos_grafico(d, "X", nube=False)
    assert len(out["t"]) == len(df_sintetico) - 2 and all(b > a for a, b in zip(out["t"], out["t"][1:]))
    assert not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in out["c"])


def test_sma200_alineada_y_ausente_con_pocas_velas(df_sintetico):
    d = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    assert d["tiene_sma"] and len(d["sma"]) == len(d["t"])
    assert d["sma"][:199] == [None] * 199 and d["sma"][199] == pytest.approx(df_sintetico["Close"].iloc[:200].mean(), abs=0.01)
    corto = ui_mod.datos_grafico(df_sintetico.iloc[:150], "T", nube=False)
    assert corto["tiene_sma"] is False and corto["sma"] is None


@pytest.mark.parametrize("malo", [None, pd.DataFrame(), pd.DataFrame({"Close": [1.0]}), "texto"])
def test_datos_faltantes_devuelven_vacio_y_overlays_solo_titulo(malo):
    d = ui_mod.datos_grafico(malo, "X")
    assert list(d) == ["vacio"] and d["vacio"]
    ov = ui_mod.overlays_grafico(d, "X", _zonas_ricas(), 100.0, valor_justo={"fair_value": 120.0})
    assert ov["zonas"] == [] and ov["vj"] is None and ov["precio"] is None and ov["extras"] == []
    assert ov["titulo"] == "X - Mapa de niveles institucionales"
    assert ui_mod.overlays_grafico(None, "-", None, None)["titulo"] == "Mapa de niveles institucionales"


# ====================================================================================== limites de anios (nube/local)
def _anios_cubiertos(d):
    return (d["t"][-1] - d["t"][0]) / (365.25 * 86400)


def test_limite_3_anios_en_nube_y_10_en_local():
    df = _df_largo(12)
    nube = ui_mod.datos_grafico(df, "T", nube=True)
    local = ui_mod.datos_grafico(df, "T", nube=False)
    assert nube["anios"] == 3 and _anios_cubiertos(nube) <= 3.01 and len(nube["t"]) < 800
    assert local["anios"] == 10 and 9.9 <= _anios_cubiertos(local) <= 10.01
    # 10 anios locales sin recortar mas: una sola constante gobierna ambos modos
    assert (ui_mod.ANIOS_VELAS_NUBE, ui_mod.ANIOS_VELAS_LOCAL) == (3, 10)
    assert ui_mod.anios_velas_defecto(True) == 3 and ui_mod.anios_velas_defecto(False) == 10


def test_parametro_unico_anios_velas_sobreescribe_el_modo():
    df = _df_largo(12)
    assert _anios_cubiertos(ui_mod.datos_grafico(df, "T", anios_velas=5, nube=True)) <= 5.01
    assert _anios_cubiertos(ui_mod.datos_grafico(df, "T", anios_velas=1, nube=False)) <= 1.01
    assert ui_mod.datos_grafico(df, "T", anios_velas=1, nube=False)["anios"] == 1


def test_nube_usa_menos_bins_de_perfil_y_menos_bytes():
    df = _df_largo(12)
    nube = ui_mod.datos_grafico(df, "T", nube=True)
    local = ui_mod.datos_grafico(df, "T", nube=False)
    assert len(nube["vp"]["p"]) == ui_mod.BINS_PERFIL_NUBE < len(local["vp"]["p"]) == ui_mod.BINS_PERFIL_LOCAL
    assert len(json.dumps(nube)) < len(json.dumps(local)) / 2


def test_modo_nube_por_variable_de_entorno(monkeypatch):
    df = _df_largo(12)
    monkeypatch.setenv("PORT", "10000")
    assert ui_mod.datos_grafico(df, "T")["anios"] == 3
    monkeypatch.delenv("PORT")
    assert ui_mod.datos_grafico(df, "T")["anios"] == 10


# ====================================================================================== Volume Profile
def test_perfil_alineado_a_bins_de_precio(df_sintetico):
    vp = ui_mod.datos_grafico(df_sintetico, "T", nube=False)["vp"]
    p, v, paso = vp["p"], vp["v"], vp["paso"]
    assert len(p) == len(v) == ui_mod.BINS_PERFIL_LOCAL
    assert all(x == pytest.approx(paso, rel=1e-3) for x in np.diff(p))              # centros equiespaciados un bin
    lo = float(df_sintetico["Low"].min())
    assert p[0] - paso / 2 == pytest.approx(lo, abs=0.02)                            # el primer bin arranca en el minimo
    assert p[-1] + paso / 2 == pytest.approx(float(df_sintetico["High"].max()), abs=0.05)
    vmax = max(v)
    assert vp["hvn"] == [i for i, x in enumerate(v) if x >= ui_mod.HVN_FRACCION * vmax] and vp["hvn"]
    assert all(x >= 0 for x in v)


def test_perfil_cubre_historial_completo_aunque_se_recorten_las_velas():
    df = _df_largo(12)
    corto = ui_mod.datos_grafico(df, "T", anios_velas=1, nube=False)
    assert len(corto["vp"]["p"]) == ui_mod.BINS_PERFIL_LOCAL and corto["vp"]["paso"] > 0


# ====================================================================================== marcadores Form 4 / 13F
def test_form4_ordenado_sin_duplicados_de_tiempo_y_sobre_barras(df_sintetico):
    d = df_sintetico
    f4 = pd.DataFrame({"Code": ["P", "P", "P", "S", "P"],
                       "Date": [d.index[300], d.index[100], d.index[100], d.index[120], d.index[100] + pd.Timedelta(days=1)],
                       "Price": [110.0, 100.0, 104.0, 50.0, np.nan]})
    out = ui_mod.datos_grafico(d, "T", f4, None, nube=False)["form4"]
    barras = set(ui_mod.datos_grafico(d, "T", nube=False)["t"])
    ts = [m["t"] for m in out]
    assert ts == sorted(ts) and len(ts) == len(set(ts))                                # orden + sin tiempos repetidos
    assert all(t in barras for t in ts)                                                # cada marca cae en una barra
    por_t = {m["t"]: m for m in out}
    t100 = int(pd.Timestamp(d.index[100]).timestamp())
    assert por_t[t100]["n"] == 2 and por_t[t100]["p"] == pytest.approx(102.0)          # dos compras el mismo dia
    assert sum(m["n"] for m in out) == 4                                                # la venta (S) no cuenta
    assert por_t[int(pd.Timestamp(d.index[300]).timestamp())]["f"] == d.index[300].strftime("%Y-%m-%d")


def test_form4_en_fin_de_semana_se_ajusta_a_la_barra_anterior(df_sintetico):
    d = df_sintetico
    sabado = pd.Timestamp("2023-06-10")                 # sabado; el viernes 9 es barra
    assert sabado.weekday() == 5
    out = ui_mod.datos_grafico(d, "T", pd.DataFrame({"Code": ["P"], "Date": [sabado], "Price": [100.0]}), None, nube=False)["form4"]
    assert [m["t"] for m in out] == [int(pd.Timestamp("2023-06-09").timestamp())]


def test_form4_fuera_de_rango_y_recorte_de_anios():
    df = _df_largo(12)
    viejo = pd.DataFrame({"Code": ["P"], "Date": [df.index[10]], "Price": [40.0]})
    assert ui_mod.datos_grafico(df, "T", viejo, None, anios_velas=3, nube=False)["form4"] == []   # fuera de las velas dibujadas
    reciente = pd.DataFrame({"Code": ["P"], "Date": [df.index[-10]], "Price": [40.0]})
    assert len(ui_mod.datos_grafico(df, "T", reciente, None, anios_velas=3, nube=False)["form4"]) == 1


def test_marcadores_vacios_o_invalidos(df_sintetico):
    for f4 in (None, pd.DataFrame(), pd.DataFrame({"Code": ["S"], "Date": [df_sintetico.index[50]], "Price": [1.0]})):
        assert ui_mod.datos_grafico(df_sintetico, "T", f4, None, nube=False)["form4"] == []
    for q in (None, pd.DataFrame(), pd.DataFrame({"x": [1]})):
        assert ui_mod.datos_grafico(df_sintetico, "T", None, q, nube=False)["q13"] == []


def test_13f_trimestres_con_banda_y_sin_duplicados(df_sintetico):
    d = df_sintetico
    r = pd.DataFrame({"gestor": ["A", "B", "C"], "periodo": ["2023-12-31", "2023-12-31", "2024-06-30"],
                      "acciones": [10.0, 5.0, 7.0], "acciones_previas": [5.0, 1.0, 0.0], "cambio_pct": [100.0, 400.0, 0.0]})
    out = ui_mod.datos_grafico(d, "T", None, r, nube=False)["q13"]
    barras = set(ui_mod.datos_grafico(d, "T", nube=False)["t"])
    assert [q["f"] for q in out] == ["2023-12-31", "2024-06-30"]                       # un trimestre aunque haya 2 gestores
    assert all(q["t"] in barras and q["t0"] in barras and q["t0"] < q["t"] for q in out)
    assert out[0]["t"] == int(pd.Timestamp("2023-12-29").timestamp())                  # 31-dic-2023 era domingo
    assert [q["t"] for q in out] == sorted(q["t"] for q in out)
    assert all(q["p"] > 0 for q in out)
    # con datos de ejemplo del resto de la suite
    assert len(ui_mod.datos_grafico(d, "T", None, _r13f(d), nube=False)["q13"]) == 1


# ====================================================================================== zonas -> bandas
def test_bandas_con_etiqueta_distancia_a_r_sup_y_resaltado():
    b = ui_mod.bandas_zonas(_zonas_ricas(), 100.0, nivel_seleccionado=2)
    assert [x["nivel"] for x in b] == [1, 2, 3] and [x["sel"] for x in b] == [False, True, False]
    assert b[0]["lo"] == 90.0 and b[0]["hi"] == 95.0
    assert b[0]["etiqueta"] == f"N1: $90.00-$95.00 | {ui_mod.ETIQUETA_DIST_RSUP} -5.0% | AVWAP+HVN+OB+POC"
    assert "-16.0%" in b[1]["etiqueta"] and "-28.0%" in b[2]["etiqueta"]               # contra r_sup, no el centro
    assert all(len(x["etiqueta"]) <= ui_mod.MAX_ETIQUETA_ZONA for x in b)
    assert ui_mod.bandas_zonas(_zonas_ricas(), 100.0)[0]["sel"] is False


@pytest.mark.parametrize("n", [0, 1, 2])
def test_cero_uno_o_dos_zonas(df_sintetico, n):
    zonas = _zonas_ricas()[:n]
    datos = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    ov = ui_mod.overlays_grafico(datos, "T", zonas, 100.0)
    assert len(ov["zonas"]) == n
    assert len(ov["extras"]) == 2 * n + 1                                              # lo/hi de cada zona + precio
    json.dumps(ov, allow_nan=False)


def test_etiqueta_larga_se_trunca_y_zona_invalida_se_descarta():
    z = {"nivel": 3, "r_inf": 1.0, "r_sup": 2.0, "fuente": "X" * 200}
    e = ui_mod.etiqueta_zona(z, 2.0)
    assert len(e) == ui_mod.MAX_ETIQUETA_ZONA and e.endswith("...")
    malas = [{"nivel": 1, "r_inf": float("nan"), "r_sup": 2.0}, {"nivel": 2, "r_inf": 5.0, "r_sup": 4.0},
             {"nivel": 3, "r_inf": None, "r_sup": 1.0}]
    assert ui_mod.bandas_zonas(malas + [z], 2.0) and len(ui_mod.bandas_zonas(malas + [z], 2.0)) == 1
    assert "n/d" in ui_mod.etiqueta_zona({"nivel": 1, "r_inf": 1.0, "r_sup": 2.0}, 0.0)   # precio invalido: sin distancia


def test_zonas_que_se_solapan_se_conservan_todas(df_sintetico):
    """N2 y N3 solapadas (caso real): ambas bandas viajan; la colision de etiquetas la resuelve el JS."""
    z = [{"nivel": 2, "r_inf": 109.7, "r_sup": 117.1, "fuente": "LOG"}, {"nivel": 3, "r_inf": 107.4, "r_sup": 114.8, "fuente": "OB"}]
    b = ui_mod.bandas_zonas(z, 148.0)
    assert [x["nivel"] for x in b] == [2, 3] and b[0]["lo"] < b[1]["hi"] and b[1]["lo"] < b[0]["hi"]


# ====================================================================================== anclas de valoracion
def test_anclas_visibles_filtra_por_rango_del_historial():
    a = [{"tipo": "DCF_MoS30", "precio": 150.0}, {"tipo": "PE_P10", "precio": 5000.0}, {"tipo": "X", "precio": float("nan")},
         {"tipo": "Y", "precio": 20.0}]
    out = ui_mod.anclas_visibles(a, (50.0, 160.0))
    assert [x["tipo"] for x in out] == ["DCF_MoS30"]
    assert ui_mod.anclas_visibles(None, (1.0, 2.0)) == [] and ui_mod.anclas_visibles(a, None) == []


def test_overlays_incluye_anclas_en_extras_solo_si_se_pasan(df_sintetico):
    datos = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    px = float(df_sintetico["Close"].iloc[-1])
    sin = ui_mod.overlays_grafico(datos, "T", [], px, anclas_val=None)
    con = ui_mod.overlays_grafico(datos, "T", [], px, anclas_val=[{"tipo": "DCF_MoS30", "precio": px * 1.1}])
    assert sin["anclas"] == [] and [a["tipo"] for a in con["anclas"]] == ["DCF_MoS30"]
    assert px * 1.1 == pytest.approx(con["extras"][-1], abs=1e-3)


# ====================================================================================== linea del valor justo medio
def test_linea_valor_justo_ausente_o_presente(df_sintetico):
    datos = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    px = 100.0
    assert ui_mod.overlays_grafico(datos, "T", [], px, valor_justo=None)["vj"] is None
    assert ui_mod.overlays_grafico(datos, "T", [], px, valor_justo={"fair_value": None})["vj"] is None
    assert ui_mod.overlays_grafico(datos, "T", [], px, valor_justo={"fair_value": -5.0})["vj"] is None
    vj = ui_mod.overlays_grafico(datos, "T", [], px, valor_justo={"fair_value": 130.0, "upside_pct": 30.0})["vj"]
    assert vj == {"precio": 130.0, "etiqueta": "Fair Value Medio: $130.00 (+30% Upside)", "fuera_de_rango": None}


def test_linea_valor_justo_conserva_nota_de_baja_confianza():
    a = ui_mod.linea_valor_justo({"fair_value": 130.0, "upside_pct": 30.0, "incertidumbre": "Alta", "n_usados": 8}, 100.0)
    assert a["etiqueta"] == "Fair Value Medio: $130.00 (+30% Upside) · incert. Alta"
    b = ui_mod.linea_valor_justo({"fair_value": 130.0, "upside_pct": 30.0, "incertidumbre": "Baja", "n_usados": 2}, 100.0)
    assert b["etiqueta"].endswith("· n=2")
    c = ui_mod.linea_valor_justo({"fair_value": 90.0, "upside_pct": -10.0, "incertidumbre": "Media", "n_usados": 9}, 100.0)
    assert c["etiqueta"] == "Fair Value Medio: $90.00 (-10% Upside)"


def test_valor_justo_unicamente_la_linea_media_sin_bandas_ni_pesimista(df_sintetico):
    datos = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    agg = {"medio": {"fair_value": 140.0, "upside_pct": 40.0}, "pesimista": {"fair_value": 70.0, "upside_pct": -30.0},
           "banda": {"bajo": 60.0, "alto": 160.0}, "incertidumbre": "Alta", "n_usados": 7}
    ov = ui_mod.overlays_grafico(datos, "T", [], 100.0, valor_justo=agg)
    assert ov["vj"]["precio"] == 140.0 and ov["vj"]["etiqueta"].startswith("Fair Value Medio: $140.00 (+40% Upside)")
    assert ov["vj"]["etiqueta"].endswith("incert. Alta")                     # hereda la incertidumbre del padre
    txt = json.dumps(ov)
    assert "pesimista" not in txt.lower() and "70.0" not in txt and "banda" not in txt


def test_acceso_unico_al_escenario_medio():
    assert ui_mod.valor_justo_medio(None) is None and ui_mod.valor_justo_medio("x") is None
    plano = {"fair_value": 10.0, "upside_pct": 5.0}
    assert ui_mod.valor_justo_medio(plano) is plano
    assert ui_mod.valor_justo_medio({"medio": {"fair_value": 11.0}, "incertidumbre": "Alta", "n_usados": 5}) == {
        "incertidumbre": "Alta", "n_usados": 5, "fair_value": 11.0}
    assert ui_mod.valor_justo_medio({"medio": {"fair_value": 11.0, "incertidumbre": "Baja"}, "incertidumbre": "Alta"})[
        "incertidumbre"] == "Baja"


def test_valor_justo_fuera_de_rango_no_se_dibuja_ni_estira_la_escala(df_sintetico):
    datos = ui_mod.datos_grafico(df_sintetico, "T", nube=False)
    ref = ui_mod.overlays_grafico(datos, "T", [], 100.0)
    alto = ui_mod.overlays_grafico(datos, "T", [], 100.0, valor_justo={"fair_value": 3500.0, "upside_pct": 3400.0})
    assert alto["extras"] == ref["extras"] and alto["vj"]["fuera_de_rango"] == "arriba" and "▲" in alto["vj"]["etiqueta"]
    bajo = ui_mod.overlays_grafico(datos, "T", [], 100.0, valor_justo={"fair_value": 10.0, "upside_pct": -90.0})
    assert bajo["vj"]["fuera_de_rango"] == "abajo" and "▼" in bajo["vj"]["etiqueta"]


# ====================================================================================== payload JS y tema
def test_js_actualizar_json_estricto_y_sin_nan(df_sintetico):
    datos = ui_mod.datos_grafico(df_sintetico, "T", _form4(df_sintetico), _r13f(df_sintetico), nube=False)
    js = ui_mod.js_actualizar("ng-7", ui_mod.tema_grafico(), datos, ui_mod.overlays_grafico(datos, "T", [], 100.0))
    m = re.fullmatch(r'NivelesGrafico\.actualizar\("ng-7",(.*)\);', js, re.S)
    assert m
    msg = json.loads(m.group(1))
    assert set(msg) == {"tema", "datos", "overlays"} and len(msg["datos"]["t"]) == len(df_sintetico)
    assert "NaN" not in js and "Infinity" not in js
    solo = ui_mod.js_actualizar("ng-7", None, None, {"titulo": "x"})
    assert json.loads(re.search(r",(\{.*\})\);", solo).group(1)) == {"overlays": {"titulo": "x"}}
    with pytest.raises(ValueError):
        ui_mod.js_actualizar("ng-7", None, {"c": [float("nan")]}, None)


def test_tema_viene_de_la_paleta_y_es_legible():
    t = ui_mod.tema_grafico()
    paleta = set(ui_mod.PALETA.values())
    assert all(v in paleta for v in t.values() if isinstance(v, str))
    assert t["niveles"] == {str(k): v for k, v in ui_mod.COLORES_NIVEL.items()}
    assert set(t["niveles"]) == {"1", "2", "3", "4"} and t["vela_alza"] == ui_mod.VELA_ALZA
    assert t["color_fair_value"] == ui_mod.COLOR_FAIR_VALUE and t["color_hvn"] == ui_mod.COLOR_HVN
    # textos del grafico sobre sus fondos reales: >= 4.5:1
    assert contraste(t["blanco"], t["bg_etiqueta"], base=t["bg_panel"]) >= 4.5            # etiquetas de zonas
    assert contraste(t["color_fair_value"], t["bg_etiqueta"], base=t["bg_panel"]) >= 4.5   # rotulo del valor justo
    assert contraste(t["color_valoracion"], t["bg_etiqueta"], base=t["bg_panel"]) >= 4.5   # rotulos de anclas
    assert contraste(t["txt"], t["bg_popup"]) >= 4.5 and contraste(t["muted"], t["bg_popup"]) >= 4.5   # tooltip
    assert contraste(t["txt"], t["bg_panel"]) >= 4.5 and contraste(t["muted"], t["bg_panel"]) >= 4.5   # cabecera/leyenda
    for c in t["niveles"].values():
        assert contraste(c, t["bg_panel"]) >= 3.0                                           # bordes de bandas (no texto)


def test_js_sin_literales_de_color_ni_cdn():
    js = _leer("static", "charts", "grafico_analisis.js")
    assert not re.findall(r"#[0-9a-fA-F]{6}\b", js) and not re.findall(r"rgba?\(\s*\d", js)   # colores: solo del tema
    assert "http://" not in js and "https://" not in js and "cdn" not in js.lower()
    assert "console.error" in js and "__ngErrores" in js                                       # errores visibles en consola
    assert "ResizeObserver" in js and "autoSize: true" in js                                   # pestanas/redimensionado


# ====================================================================================== recursos estaticos / empaquetado
def test_lightweight_charts_empaquetado_con_licencia_y_atribucion():
    js = _leer("static", "vendor", "lightweight-charts.standalone.production.js")
    assert ui_mod.VERSION_LW in js[:300] and "Apache License 2.0" in js[:300] and "TradingView" in js[:300]
    assert "Apache License" in _leer("static", "vendor", "LICENSE-lightweight-charts.txt")
    aviso = _leer("static", "vendor", "NOTICE-lightweight-charts.txt")
    assert "https://www.tradingview.com/" in aviso and ui_mod.VERSION_LW in aviso
    assert "tradingview" in ui_mod.ATRIBUCION_LW.lower() and ui_mod.URL_TRADINGVIEW == "https://www.tradingview.com/"


def test_scripts_apuntan_a_archivos_locales_servidos_por_nicegui():
    from nicegui import app
    assert all(u.startswith(ui_mod.URL_STATIC + "/") for u in ui_mod.SCRIPTS_GRAFICO)
    for u in ui_mod.SCRIPTS_GRAFICO:
        ruta = u.split("?")[0][len(ui_mod.URL_STATIC) + 1:]
        assert os.path.isfile(os.path.join(ui_mod.DIR_STATIC, *ruta.split("/"))), ruta
    assert any(getattr(r, "path", "").startswith(ui_mod.URL_STATIC + "/") for r in app.routes)
    assert not any(u.startswith("http") for u in ui_mod.SCRIPTS_GRAFICO)                    # sin CDN


def test_build_spec_es_valido_y_empaqueta_static():
    src = _leer("build.spec")
    compile(src, "build.spec", "exec")
    assert '"static"' in src and os.path.isdir(os.path.join(RAIZ, "static", "vendor")) and \
        os.path.isdir(os.path.join(RAIZ, "static", "charts"))
    assert os.path.normpath(ui_mod.DIR_STATIC) == os.path.normpath(os.path.join(RAIZ, "static"))


# ====================================================================================== Vista (sin servidor ni navegador)
@pytest.fixture
def vista(cliente, df_sintetico, monkeypatch):
    res = {"ticker": "TEST", "precio_actual": 100.0, "df": df_sintetico, "zonas": _zonas_ricas(), "errores": [],
           "form4": _form4(df_sintetico), "r13f": _r13f(df_sintetico), "confluencia_score": 60.0,
           "confiabilidad": {"indice": 40.0, "cobertura": 0.8},
           "valoracion": {"pe": {"pisos": {"P10": 91.0}}, "ev_fcf": None, "dcf": None}, "backtest_resumen": {}}
    monkeypatch.setitem(ui_mod.CACHE, "TEST", res)
    v = ui_mod.Vista()
    v.construir()
    v.ticker = "TEST"
    v.refrescar()
    return v


def test_vista_el_grafico_es_un_contenedor_y_valor_justo_usa_echarts(cliente):
    from nicegui import ui
    v = ui_mod.Vista()
    v.construir()
    assert v.grafico.tag == "div" and v.grafico_id == v.grafico.html_id
    assert v.grafico.props["data-grafico"] == "analisis"
    assert not [e for e in cliente.elements.values() if isinstance(e, ui.plotly)]       # Plotly ya no existe en la app
    ech = [e for e in cliente.elements.values() if isinstance(e, ui.echart)]
    assert ech == [v.vj_hist, v.vj_barras]                                      # Valor Justo usa Apache ECharts (2 graficos)
    assert all(u in cliente._head_html for u in ui_mod.SCRIPTS_GRAFICO)
    assert any(isinstance(e, ui.link) and ui_mod.URL_TRADINGVIEW in str(e.props.get("href", "")) for e in cliente.elements.values())


def test_vista_envia_datos_pesados_solo_cuando_cambian(vista, df_sintetico, monkeypatch):
    enviados = []
    monkeypatch.setattr(vista, "_grafico_js", lambda codigo: enviados.append(codigo))
    vista.refrescar(solo_figura=True)                          # sin cambios de datos (p. ej. seleccion de fila)
    vista.nivel_sel = 1
    vista.refrescar(solo_figura=True)
    vista.activos["OB"] = False
    vista.refrescar()                                          # toggle de familia
    assert len(enviados) == 3 and all('"datos"' not in js for js in enviados) and all('"overlays"' in js for js in enviados)
    assert vista.grafico_overlays["zonas"][0]["sel"] is False or vista.nivel_sel == 1
    # ticker nuevo -> si se reenvian las velas
    otro = {"ticker": "OTRO", "precio_actual": 50.0, "df": df_sintetico * 0.5, "zonas": [], "errores": [], "form4": None,
            "r13f": None, "confiabilidad": {}, "backtest_resumen": {}}
    monkeypatch.setitem(ui_mod.CACHE, "OTRO", otro)
    vista.ticker = "OTRO"
    vista.refrescar()
    assert '"datos"' in enviados[-1] and vista.grafico_datos["ticker"] == "OTRO"


def test_vista_seleccion_de_fila_resalta_la_banda(vista):
    vista.nivel_sel = 2
    vista.refrescar(solo_figura=True)
    assert [b["sel"] for b in vista.grafico_overlays["zonas"]] == [False, True, False]
    vista.nivel_sel = None
    vista.refrescar(solo_figura=True)
    assert not any(b["sel"] for b in vista.grafico_overlays["zonas"])


def test_vista_toggle_de_familias_cambia_las_bandas(vista):
    assert len(vista.grafico_overlays["zonas"]) == 3
    for k in ("HVN", "AVWAP", "LOGREG", "SWING", "VALORACION"):
        vista.activos[k] = False
    vista.refrescar()
    assert [b["nivel"] for b in vista.grafico_overlays["zonas"]] == [1, 2]                # quedan las 2 zonas con OB (renumeradas)


def test_vista_anclas_de_valoracion_siguen_al_toggle_dcf_pe(vista):
    assert vista.grafico_overlays["anclas"]
    vista.activos["VALORACION"] = False
    vista.refrescar(solo_figura=True)
    assert vista.grafico_overlays["anclas"] == []


def test_vista_vacia_sin_resultado(cliente):
    v = ui_mod.Vista()
    v.construir()
    assert v.grafico_datos == {"vacio": ui_mod.MSG_SIN_DATOS} and v.grafico_overlays["zonas"] == []
    assert "NivelesGrafico.actualizar" in v.ultimo_js_grafico and '"vacio"' in v.ultimo_js_grafico


def test_vista_anios_velas_unico_parametro(vista, monkeypatch):
    df = _df_largo(12)
    monkeypatch.setitem(ui_mod.CACHE, "TEST", {**ui_mod.CACHE["TEST"], "df": df})
    vista.refrescar()
    assert vista.grafico_datos["anios"] == 10
    vista.anios_velas = 2
    vista.refrescar()
    assert vista.grafico_datos["anios"] == 2 and _anios_cubiertos(vista.grafico_datos) <= 2.01
    monkeypatch.setenv("PORT", "10000")
    vista.anios_velas = None
    vista.refrescar()
    assert vista.grafico_datos["anios"] == 3


def test_vista_toggle_valor_justo_dibuja_solo_la_linea_media(vista, monkeypatch):
    vista.sw_vj.value = False
    vista.refrescar(solo_figura=True)
    assert vista.grafico_overlays["vj"] is None
    monkeypatch.setattr(vista, "_vj_para_grafico", lambda: {"medio": {"fair_value": 130.0, "upside_pct": 30.0},
                                                            "pesimista": {"fair_value": 61.7, "upside_pct": -38.3}})
    vista.refrescar(solo_figura=True)
    assert vista.grafico_overlays["vj"]["etiqueta"].startswith("Fair Value Medio: $130.00 (+30% Upside)")
    assert "61.7" not in json.dumps(vista.grafico_overlays)
