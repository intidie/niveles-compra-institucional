"""Correcciones del auditor en main.py: datos no disponibles != 0, 13F solo activos, STOCK Act excluido, muestra
pequeña del backtest, capital por ticker y distancia unica a r_sup. Todo con mocks: sin red ni SEC ni yfinance."""
import pandas as pd
import pytest

import main as m
from src.reporting import charts


# --------------------------------------------------------------------------------------------------------
# Helpers: 13F con atributos del contrato (fallidos / consultados / columna pasivo) y pipeline simulado
# --------------------------------------------------------------------------------------------------------
def _r13(filas, fallidos=(), consultados=5):
    cols = ["gestor", "periodo", "acciones", "valor_usd", "acciones_previas", "cambio_pct", "pasivo"]
    df = pd.DataFrame(filas, columns=cols)
    df.attrs["fallidos"] = list(fallidos)
    df.attrs["consultados"] = consultados
    return df


def _fila(gestor, acciones, previas, pasivo=False):
    cambio = (acciones - previas) / previas * 100.0 if previas > 0 else None
    return [gestor, "2026-09-30", acciones, acciones * 10.0, previas, cambio, pasivo]


def _simular(monkeypatch, df, *, form4="ok", r13=None, r13_error=False, stock=None, stock_error=False,
             n_fills=30, n_ordenes=60):
    """Parchea todas las fuentes externas que usa ``analizar_ticker`` (importadas dentro de la funcion)."""
    import src.data.market_data as md
    import src.data.sec_edgar as se
    import src.data.stock_act as sa
    import src.levels.fair_value as fvm
    import src.levels.valuation_floors as vf
    import src.validation.backtest_ranges as br

    px = float(df["Close"].iloc[-1])
    # el valor justo consultaria EDGAR: los tests no usan red
    monkeypatch.setattr(fvm, "extraer_entradas", lambda t, *a, **k: fvm.EntradasValoracion(ticker=t, precio=px))
    monkeypatch.setattr(md, "descargar_historial", lambda t, p: (object(), df))
    monkeypatch.setattr(md, "obtener_info", lambda tk: {})
    monkeypatch.setattr(md, "verificar_proximidad_earnings", lambda tk: {"alerta_activa": False, "mensaje": "sin alerta"})
    cluster = {"lo": px * 0.85, "hi": px * 0.90, "n_familias": 3, "n_anclas": 4, "dist_pct": 0.10,
               "familias": ["avwap", "ob", "volumen"], "tipos": ["AVWAP", "HVN", "OB", "POC"]}
    cluster2 = {"lo": px * 0.70, "hi": px * 0.75, "n_familias": 2, "n_anclas": 2, "dist_pct": 0.25,
                "familias": ["logreg", "swing"], "tipos": ["LOG", "SWING"]}
    monkeypatch.setattr(br, "calcular_zonas_pit", lambda d: {"px": px, "atr_semanal": 1.0, "clusters": [cluster, cluster2],
                                                             "primario": cluster})
    monkeypatch.setattr(vf, "calcular_pisos_valoracion", lambda *a, **k: {"anclas": [], "pe": None})

    def f4(ticker):
        if form4 == "error":
            raise ConnectionError("sec caida")
        return pd.DataFrame(), {"compras": 2, "acciones": 100.0, "valor_usd": 250_000.0 if form4 == "ok" else 0.0}
    monkeypatch.setattr(se, "compras_insiders", f4)

    def o13(ticker):
        if r13_error:
            raise ConnectionError("13f caido")
        return r13 if r13 is not None else _r13([_fila("Berkshire Hathaway", 10, 5)])
    monkeypatch.setattr(se, "obtener_13f_ticker", o13)

    def so(ticker):
        if stock_error:
            raise TimeoutError("congreso")
        return stock if stock is not None else {"pts": 7.5, "disponible": True, "motivo": None, "compras": 3, "ventas": 3,
                                                "resumen": "ok", "ultimas_tx": []}
    monkeypatch.setattr(sa, "consultar_flujo_congreso_stock_act", so)

    resumen = {"n_fechas": n_ordenes, "n_ordenes": n_ordenes, "n_fills": n_fills, "fill_rate": n_fills / max(n_ordenes, 1),
               "absorcion": 0.25, "bounce": 0.65, "ret60_mediana": 0.05, "mae60_mediana": -0.08}
    monkeypatch.setattr(br, "backtest_ticker", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(br, "resumen", lambda o: dict(resumen))
    return px


# --------------------------------------------------------------------------------------------------------
# 1. Form 4: fallo -> None (no 0); exito -> valor (0 es valido)
# --------------------------------------------------------------------------------------------------------
def test_respaldo_form4_fallido_es_none_y_con_exito_cero_es_cero():
    r13 = _r13([_fila("Berkshire Hathaway", 10, 5)])
    assert m.construir_respaldo(None, False, r13)["compras_form4_usd"] is None
    assert m.construir_respaldo({"valor_usd": 0.0}, True, r13)["compras_form4_usd"] == 0.0
    assert m.construir_respaldo({"valor_usd": 5e5}, True, r13)["compras_form4_usd"] == 5e5


def test_pipeline_form4_con_excepcion_queda_en_none_y_en_errores(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, form4="error")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["respaldo13f"]["compras_form4_usd"] is None
    assert any(e.startswith("form4:") for e in res["errores"])
    assert res["respaldo13f"]["n_con_posicion"] == 1  # el 13F si respondio: no se pierde por fallar Form 4
    # el componente sigue con la mitad 13F (no se hunde a 0 por un Form 4 que no respondio)
    det = res["confiabilidad"]["detalle"]["respaldo_13f_form4"]
    assert det["valor"] == pytest.approx(100.0)  # 1 de 1 gestor activo acumula; mitad Form 4 excluida


def test_pipeline_form4_con_exito_y_cero_compras_puntua_cero(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, form4="cero", r13=_r13([_fila("Berkshire Hathaway", 10, 5)]))
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["respaldo13f"]["compras_form4_usd"] == 0.0
    assert res["confiabilidad"]["detalle"]["respaldo_13f_form4"]["valor"] == pytest.approx(50.0)  # (100 + 0) / 2


# --------------------------------------------------------------------------------------------------------
# 2. 13F: todos fallaron -> None; algunos fallaron -> avisos; consultados sin posicion -> 0
# --------------------------------------------------------------------------------------------------------
def test_13f_todos_fallaron_es_none_no_cero():
    todos = _r13([], fallidos=["A", "B", "C"], consultados=3)
    assert m.resumen_13f(todos) == (None, None)
    assert m.construir_respaldo({"valor_usd": 0.0}, True, todos)["n_con_posicion"] is None
    assert m.resumen_13f(None) == (None, None)


def test_13f_consultados_sin_posicion_es_cero_valido():
    assert m.resumen_13f(_r13([], fallidos=[], consultados=5)) == (0, 0)
    parcial = _r13([], fallidos=["A", "B"], consultados=5)  # 3 respondieron y no tienen posicion
    assert m.resumen_13f(parcial) == (0, 0)


def test_13f_sin_saber_cuantos_se_consultaron_no_inventa_un_cero():
    df = _r13([], fallidos=["A"], consultados=None)
    assert m.resumen_13f(df) == (None, None)
    df2 = _r13([_fila("Bridgewater", 10, 5)], fallidos=["A"], consultados=None)
    assert m.resumen_13f(df2) == (1, 1)


def test_pipeline_13f_todos_fallidos_registra_gestores_y_excluye_la_mitad(monkeypatch, df_sintetico):
    todos = _r13([], fallidos=["Berkshire Hathaway", "Bridgewater"], consultados=2)
    _simular(monkeypatch, df_sintetico, r13=todos, form4="cero")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["respaldo13f"]["n_con_posicion"] is None and res["respaldo13f"]["n_acumulando"] is None
    assert res["r13f_fallidos"] == ["Berkshire Hathaway", "Bridgewater"] and res["r13f_consultados"] == 2
    aviso = next(a for a in res["avisos"] if a.startswith("13f: no respondieron"))
    assert "Berkshire Hathaway, Bridgewater" in aviso and "no disponible" in aviso
    # solo queda la mitad Form 4 (0 USD, dato valido): el componente puntua 0 con ese dato, no con un 0 de 13F inventado
    assert res["confiabilidad"]["detalle"]["respaldo_13f_form4"]["valor"] == pytest.approx(0.0)


def test_pipeline_13f_y_form4_sin_dato_excluyen_el_componente(monkeypatch, df_sintetico):
    todos = _r13([], fallidos=["A", "B"], consultados=2)
    _simular(monkeypatch, df_sintetico, r13=todos, form4="error")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert "respaldo_13f_form4" in res["componentes_sin_dato"]
    assert "respaldo_13f_form4" not in res["confiabilidad"]["detalle"]
    assert res["confiabilidad"]["cobertura"] < 1.0


def test_pipeline_13f_con_excepcion_conserva_form4(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, r13_error=True)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["respaldo13f"]["n_con_posicion"] is None and res["respaldo13f"]["compras_form4_usd"] == 250_000.0
    assert any(e.startswith("13f:") for e in res["errores"])


# --------------------------------------------------------------------------------------------------------
# 3. 13F: solo cuentan los gestores activos
# --------------------------------------------------------------------------------------------------------
def test_13f_pasivos_no_cuentan_ni_en_posicion_ni_en_acumulacion():
    df = _r13([_fila("Vanguard Group", 100, 90, pasivo=True),          # pasivo que compra: NO cuenta
               _fila("State Street", 50, 40, pasivo=True),
               _fila("Berkshire Hathaway", 10, 20),                    # activo que vende
               _fila("Bridgewater", 30, 10),                            # activo que aumenta
               _fila("Renaissance Technologies", 5, 0)])                # activo con posicion nueva
    assert m.resumen_13f(df) == (3, 2)
    con_pasivos = m.construir_respaldo({"valor_usd": 0.0}, True, df)
    assert (con_pasivos["n_con_posicion"], con_pasivos["n_acumulando"]) == (3, 2)


def test_13f_solo_pasivos_respondieron_es_cero_activos():
    df = _r13([_fila("Vanguard Group", 100, 90, pasivo=True), _fila("State Street", 50, 40, pasivo=True)])
    assert m.resumen_13f(df) == (0, 0)


def test_13f_sin_columna_pasivo_cuenta_todas_las_filas():
    df = _r13([_fila("Berkshire Hathaway", 10, 20), _fila("Bridgewater", 30, 10)]).drop(columns=["pasivo"])
    assert m.resumen_13f(df) == (2, 1)


def test_pipeline_13f_con_pasivos_alimenta_la_confiabilidad_solo_con_activos(monkeypatch, df_sintetico):
    df = _r13([_fila("Vanguard Group", 100, 50, pasivo=True), _fila("State Street", 100, 50, pasivo=True),
               _fila("Berkshire Hathaway", 10, 20)])  # el unico activo VENDE: con pasivos contando daria 2 de 3
    _simular(monkeypatch, df_sintetico, r13=df, form4="cero")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert (res["respaldo13f"]["n_con_posicion"], res["respaldo13f"]["n_acumulando"]) == (1, 0)
    assert res["confiabilidad"]["detalle"]["respaldo_13f_form4"]["valor"] == pytest.approx(0.0)
    assert not [a for a in res["avisos"] if a.startswith("13f:")]


# --------------------------------------------------------------------------------------------------------
# 4. STOCK Act no disponible: excluido, cobertura menor, aviso
# --------------------------------------------------------------------------------------------------------
def test_stock_act_no_disponible_se_excluye_y_baja_la_cobertura(monkeypatch, df_sintetico):
    from src.validation import scoring
    _simular(monkeypatch, df_sintetico)
    con = m.analizar_ticker("AAA", 1000, verbose=False)
    _simular(monkeypatch, df_sintetico, stock={"pts": None, "disponible": False, "motivo": "sin red (timeout)",
                                               "compras": 0, "ventas": 0, "resumen": "no disponible", "ultimas_tx": []})
    sin = m.analizar_ticker("AAA", 1000, verbose=False)
    assert con["confiabilidad"]["cobertura"] == pytest.approx(1.0) and "stock_act" not in con["componentes_sin_dato"]
    assert "stock_act" in sin["componentes_sin_dato"] and "stock_act" not in sin["confiabilidad"]["detalle"]
    peso = scoring.cargar_pesos()
    assert sin["confiabilidad"]["cobertura"] == pytest.approx(1.0 - peso["stock_act"] / sum(peso.values()))
    assert sin["stock_act"]["disponible"] is False and sin["stock_act"]["pts"] is None
    assert "ultimas_tx" not in sin["stock_act"]
    assert any("STOCK Act no disponible" in a and "sin red (timeout)" in a for a in sin["avisos"])
    assert not any("STOCK Act" in a for a in con["avisos"])


def test_stock_act_con_excepcion_queda_excluido(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, stock_error=True)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["stock_act"] == {"pts": None, "disponible": False, "motivo": "TimeoutError: congreso"}
    assert "stock_act" in res["componentes_sin_dato"] and any(e.startswith("stock_act:") for e in res["errores"])


def test_stock_act_valido_con_cero_puntos_si_cuenta(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, stock={"pts": 0.0, "disponible": True, "motivo": None, "compras": 0, "ventas": 9,
                                               "resumen": "solo ventas", "ultimas_tx": []})
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["confiabilidad"]["detalle"]["stock_act"]["valor"] == 0.0 and "stock_act" not in res["componentes_sin_dato"]
    assert m.aviso_stock_act({"pts": 0.0, "disponible": True}) is None
    assert m.aviso_stock_act({"pts": 7.5}) is None  # forma antigua sin 'disponible'


def test_sin_sec_excluye_los_componentes_sec_sin_avisos_de_fallo(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico)
    res = m.analizar_ticker("AAA", 1000, sec=False, verbose=False)
    assert {"respaldo_13f_form4", "stock_act"} <= set(res["componentes_sin_dato"])
    assert not [a for a in res["avisos"] if "STOCK" in a or a.startswith("13f")]
    assert res["confiabilidad"]["cobertura"] < 1.0


# --------------------------------------------------------------------------------------------------------
# 5. Muestra pequeña del backtest y alcance de las metricas
# --------------------------------------------------------------------------------------------------------
def test_aviso_muestra_pequena_y_umbral():
    assert m.MIN_FILLS_AVISO == 10
    assert m.aviso_muestra_pequena(4).startswith("muestra pequeña: 4 fills")
    assert m.aviso_muestra_pequena(0).startswith("muestra pequeña: 0 fills")
    assert m.aviso_muestra_pequena(m.MIN_FILLS_AVISO) is None and m.aviso_muestra_pequena(120) is None
    assert m.aviso_muestra_pequena(None) is None


def test_pipeline_expone_n_ordenes_n_fills_y_avisa(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, n_fills=4, n_ordenes=12)
    res = m.analizar_ticker("AAA", 1000, anios_backtest=3, verbose=False)
    assert (res["n_ordenes"], res["n_fills"], res["anios_backtest"]) == (12, 4, 3)
    assert any(a.startswith("muestra pequeña: 4 fills") for a in res["avisos"])
    assert "rango primario (N1)" in res["backtest_alcance"] and "3 años" in res["backtest_alcance"]
    assert "no comparable con el pool" in res["backtest_alcance"]


def test_pipeline_sin_aviso_con_muestra_suficiente_y_default_5_anios(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico, n_fills=m.MIN_FILLS_AVISO, n_ordenes=40)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["n_fills"] == m.MIN_FILLS_AVISO and res["anios_backtest"] == 5
    assert not [a for a in res["avisos"] if a.startswith("muestra pequeña")]


def test_pipeline_sin_backtest_no_inventa_muestra(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico)
    res = m.analizar_ticker("AAA", 1000, backtest=False, verbose=False)
    assert res["n_fills"] is None and res["n_ordenes"] is None and res["anios_backtest"] is None
    assert not [a for a in res["avisos"] if a.startswith("muestra")]
    assert "backtest" in res["componentes_sin_dato"] and "absorcion" in res["componentes_sin_dato"]


def test_pipeline_backtest_sin_ordenes_cuenta_como_muestra_vacia(monkeypatch, df_sintetico):
    import src.validation.backtest_ranges as br
    _simular(monkeypatch, df_sintetico)
    monkeypatch.setattr(br, "backtest_ticker", lambda *a, **k: pd.DataFrame())
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["n_fills"] == 0 and res["n_ordenes"] == 0
    assert any(a.startswith("muestra pequeña: 0 fills") for a in res["avisos"])


# --------------------------------------------------------------------------------------------------------
# 6. Distancia unificada: SIEMPRE de r_sup al precio actual (consola, zonas, grafico)
# --------------------------------------------------------------------------------------------------------
def test_formula_unica_de_distancia():
    assert charts.dist_a_r_sup_pct(95.0, 100.0) == pytest.approx(-5.0)
    assert charts.dist_a_r_sup_pct(110.0, 100.0) == pytest.approx(10.0)
    assert charts.dist_a_r_sup_pct(95.0, 0.0) is None and charts.dist_a_r_sup_pct(95.0, None) is None
    assert charts.dist_a_r_sup_pct(float("nan"), 100.0) is None
    assert charts.ETIQUETA_DIST_RSUP == "dist. a r_sup"


def test_zonas_de_analizar_ticker_miden_la_distancia_a_r_sup(monkeypatch, df_sintetico):
    px = _simular(monkeypatch, df_sintetico)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["precio_actual"] == pytest.approx(px) and len(res["zonas"]) == 2
    for z in res["zonas"]:
        assert z["dist_pct"] == pytest.approx((z["r_sup"] / px - 1) * 100)
        centro = (z["r_inf"] + z["r_sup"]) / 2
        assert z["dist_pct"] != pytest.approx((centro / px - 1) * 100)  # no es la distancia al centro
    assert res["zonas"][0]["dist_pct"] == pytest.approx(-10.0)


def test_consola_y_grafico_usan_la_misma_distancia(monkeypatch, df_sintetico, capsys, tmp_path):
    px = _simular(monkeypatch, df_sintetico)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    m.imprimir_individual(res)
    salida = capsys.readouterr().out
    z1 = res["zonas"][0]
    assert f"dist. a r_sup {z1['dist_pct']:+.1f}%" in salida
    import matplotlib.pyplot as plt
    fig = charts.graficar_niveles(df_sintetico, "AAA", res["zonas"], px)
    textos = [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)
    etiqueta = next(t for t in textos if t.strip().startswith("N1:"))
    assert "dist. a r_sup" in etiqueta and f"{z1['dist_pct']:+.1f}%" in etiqueta
    centro = (z1["r_inf"] + z1["r_sup"]) / 2
    assert f"{(centro / px - 1) * 100:+.1f}%" not in etiqueta


def test_grafico_dibuja_la_linea_en_r_sup_no_en_el_centro(df_sintetico):
    import matplotlib.pyplot as plt
    zonas = [{"nivel": 1, "r_inf": 80.0, "r_sup": 90.0, "fuente": "POC"}]
    fig = charts.graficar_niveles(df_sintetico, "X", zonas, 100.0)
    ys = [float(l.get_ydata()[0]) for l in fig.axes[0].lines if len(set(l.get_ydata())) == 1]
    plt.close(fig)
    assert 90.0 in ys and 85.0 not in ys  # 85 seria el centro


# --------------------------------------------------------------------------------------------------------
# 7. Capital por ticker y encabezados de la consola
# --------------------------------------------------------------------------------------------------------
def test_ayuda_de_capital_dice_por_ticker_y_no_se_reparte():
    ayuda = " ".join(m.build_parser().format_help().split())
    assert "capital por ticker; en modo --watchlist no se reparte entre tickers" in ayuda


def test_consola_individual_muestra_capital_por_ticker_cobertura_y_avisos(monkeypatch, df_sintetico, capsys):
    _simular(monkeypatch, df_sintetico, n_fills=4, n_ordenes=12,
             stock={"pts": None, "disponible": False, "motivo": "sin red", "ultimas_tx": []})
    res = m.analizar_ticker("AAA", 1000, anios_backtest=5, verbose=False)
    m.imprimir_individual(res)
    out = capsys.readouterr().out
    assert "Capital por ticker $1,000" in out
    assert "sin dato: STOCK Act" in out and "cobertura" in out
    assert "Backtest (rango primario (N1), backtest de 5 años" in out and "4 fills de 12 ordenes" in out
    assert "[aviso] muestra pequeña: 4 fills" in out and "[aviso] STOCK Act no disponible" in out


def test_ranking_batch_explicita_capital_por_ticker_y_cobertura(capsys):
    rs = [{"ticker": "AAA", "precio_actual": 10.0, "score_ranking": 55.0, "confluencia_score": 60.0,
           "confiabilidad": {"indice": 50.0, "cobertura": 0.65},
           "zonas": [{"r_inf": 8.0, "r_sup": 9.0, "dist_pct": -10.0, "n_familias": 3}]}]
    m.imprimir_ranking(rs, 2500.0)
    out = capsys.readouterr().out
    assert m.NOTA_CAPITAL_BATCH in out and "$2,500" in out and "no se reparte entre tickers" in out
    assert "Cob." in out and "65%" in out and "-10.0%" in out and "Dist r_sup" in out
    assert "no comparar scores con cobertura distinta" in out


def test_main_batch_pasa_el_capital_completo_a_cada_ticker(monkeypatch, tmp_path):
    vistos = []

    def falso(ticker, capital, **kw):
        vistos.append((ticker, capital))
        return {"ticker": ticker, "fecha": "2026-01-01", "capital": capital, "df": None, "form4": None, "r13f": None,
                "precio_actual": 10.0, "errores": [], "avisos": ["muestra pequeña: 3 fills (< 10); x"], "zonas": [],
                "confluencia_score": 50.0, "confiabilidad": {"indice": 50.0, "cobertura": 1.0}, "score_ranking": 50.0}
    monkeypatch.setattr(m, "analizar_ticker", falso)
    w = tmp_path / "w.csv"
    w.write_text("ticker\nAAA\nBBB\nCCC\n", encoding="utf-8")
    assert m.main(["--watchlist", str(w), "--capital", "700", "--salida", str(tmp_path / "o")]) == 0
    assert vistos == [("AAA", 700.0), ("BBB", 700.0), ("CCC", 700.0)]  # sin repartir entre tickers
    import json
    d = json.loads((tmp_path / "o" / "ranking_watchlist.json").read_text(encoding="utf-8"))
    assert "no se reparte entre tickers" in d["notas"]["capital"]
    filas = pd.read_csv(tmp_path / "o" / "ranking_watchlist.csv")
    assert set(filas["capital"]) == {700.0} and filas["capital_alcance"].str.contains("por ticker").all()
    assert filas["avisos"].str.contains("muestra pequeña").all()


# --------------------------------------------------------------------------------------------------------
# 8. Helpers de cobertura y ranking
# --------------------------------------------------------------------------------------------------------
def test_componentes_sin_dato_desde_detalle_o_excluidos():
    assert m.componentes_sin_dato(None) == [] and m.componentes_sin_dato({"indice": 1.0}) == []  # sin detalle: no se afirma
    det = {"detalle": {"backtest": {}, "confluencia": {}}}
    assert m.componentes_sin_dato(det) == ["respaldo_13f_form4", "stock_act", "absorcion"]
    assert m.componentes_sin_dato({"detalle": {}, "excluidos": ["stock_act"]}) == ["stock_act"]  # lo explicito manda
    assert m.nombres_componentes(["stock_act", "backtest"]) == "STOCK Act, backtest"


def test_peso_confluencia_en_el_ranking_sale_de_los_pesos(monkeypatch):
    from src.validation import scoring
    w = scoring.cargar_pesos()["confluencia"]
    assert m.peso_confluencia_ranking() == pytest.approx(0.5 + 0.5 * w)
    assert 0.5 < m.peso_confluencia_ranking() < 0.6  # ~55 % con los pesos calibrados vigentes
    assert m.peso_confluencia_ranking({"confluencia": 0.2}) == pytest.approx(0.6)
    monkeypatch.setattr(scoring, "cargar_pesos", lambda: {"confluencia": 0.4})
    assert m.peso_confluencia_ranking() == pytest.approx(0.7)


def test_res_inicial_trae_los_campos_nuevos_aun_si_falla_el_mercado(monkeypatch):
    import src.data.market_data as md
    monkeypatch.setattr(md, "descargar_historial", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
    r = m.analizar_ticker("AAA", 10, verbose=False)
    assert r["avisos"] == [] and r["componentes_sin_dato"] == [] and r["n_fills"] is None and r["anios_backtest"] is None


# --------------------------------------------------------------------------------------------------------
# 9. Valoracion no disponible (None) != valoracion consultada sin anclas en la zona (0)
# --------------------------------------------------------------------------------------------------------
def _valoracion_falla(monkeypatch):
    import src.levels.valuation_floors as vf

    def boom(*a, **k):
        raise ConnectionError("yfinance caido")
    monkeypatch.setattr(vf, "calcular_pisos_valoracion", boom)


def test_valoracion_consultada_sin_anclas_es_cero_valido_y_sin_aviso(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico)  # calcular_pisos_valoracion devuelve anclas=[]
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["n_valoracion_en_zona"] == 0
    assert not [a for a in res["avisos"] if a.startswith("valoracion")]
    assert res["confluencia_score"] == pytest.approx(40.0)  # 3 familias: 0.6 * 2/3 = 40; valoracion = 0 es dato valido


def test_valoracion_fallida_es_none_y_avisa_sin_romper_el_scoring(monkeypatch, df_sintetico):
    _simular(monkeypatch, df_sintetico)
    _valoracion_falla(monkeypatch)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["n_valoracion_en_zona"] is None  # no disponible, no 0
    assert any(e.startswith("valoracion:") for e in res["errores"])
    assert any(a.startswith("valoracion") and "no disponible" in a for a in res["avisos"])
    assert res["confluencia_score"] is not None and res["confiabilidad"]["indice"] is not None  # sin TypeError
    assert not [e for e in res["errores"] if e.startswith("scoring:")]


def test_n_valoracion_para_scoring_segun_soporte_de_none(monkeypatch):
    from src.validation import scoring
    assert m.n_valoracion_para_scoring(0) == (0, None) and m.n_valoracion_para_scoring(2) == (2, None)
    # scoring actual: s_confluencia(n, None) lanza TypeError -> se mantiene 0 pero con aviso explicito
    n, aviso = m.n_valoracion_para_scoring(None)
    try:
        scoring.s_confluencia(1, None)
        soporta_none = True
    except TypeError:
        soporta_none = False
    assert (n is None) == soporta_none and "no disponible" in aviso
    if not soporta_none:
        assert n == 0 and "puntua 0" in aviso
    # si scoring pasa a excluir la mitad de valoracion (None), main le pasa None y lo dice
    orig = scoring.s_confluencia
    monkeypatch.setattr(scoring, "s_confluencia",
                        lambda nf, nv=0: orig(nf, 0) / 0.6 if nv is None and nf is not None else orig(nf, nv))
    n2, aviso2 = m.n_valoracion_para_scoring(None)
    assert n2 is None and "se excluye" in aviso2


def test_valoracion_fallida_con_scoring_que_admite_none(monkeypatch, df_sintetico):
    from src.validation import scoring
    orig_conf, orig_idx = scoring.s_confluencia, scoring.indice_desde_modulos
    llamadas = []

    def conf(nf, nv=0):
        llamadas.append(nv)
        return 100 * min(1.0, max(0.0, (nf - 1) / 3)) if nv is None else orig_conf(nf, nv)
    monkeypatch.setattr(scoring, "s_confluencia", conf)
    monkeypatch.setattr(scoring, "indice_desde_modulos",
                        lambda bt, r13, st, ab, nf, nv, **k: orig_idx(bt, r13, st, ab, nf, 0 if nv is None else nv, **k))
    _simular(monkeypatch, df_sintetico)
    _valoracion_falla(monkeypatch)
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert None in llamadas  # main le paso 'sin dato' a scoring en vez de 0
    assert res["confluencia_score"] == pytest.approx(100 * 2 / 3)  # solo la mitad de familias, reescalada
    assert any("se excluye de la confluencia" in a for a in res["avisos"])


def _scoring_admite_none():
    from src.validation import scoring
    try:
        scoring.s_confluencia(3, None)
        return True
    except TypeError:
        return False


def test_valoracion_no_consultada_vs_consultada_sin_anclas_con_el_scoring_vigente(monkeypatch, df_sintetico):
    """Contrato final: no consultada -> se excluye y se reescala; consultada sin anclas -> 0. Si el scoring instalado aun
    no admite None, main mantiene 0 PERO lo avisa (el test comprueba la rama que corresponda)."""
    _simular(monkeypatch, df_sintetico)
    consultada = m.analizar_ticker("AAA", 1000, verbose=False)
    assert consultada["n_valoracion_en_zona"] == 0 and consultada["confluencia_score"] == pytest.approx(40.0)
    _valoracion_falla(monkeypatch)
    no_consultada = m.analizar_ticker("AAA", 1000, verbose=False)
    assert no_consultada["n_valoracion_en_zona"] is None
    aviso = next(a for a in no_consultada["avisos"] if a.startswith("valoracion"))
    if _scoring_admite_none():
        assert no_consultada["confluencia_score"] == pytest.approx(100 * 2 / 3)  # (n_fam - 1) / 3 reescalado a 0-100
        assert "se excluye de la confluencia" in aviso and "puntua 0" not in aviso
        assert no_consultada["confiabilidad"]["detalle"]["confluencia"]["valor"] == pytest.approx(100 * 2 / 3)
    else:
        assert no_consultada["confluencia_score"] == pytest.approx(40.0) and "puntua 0" in aviso


# --------------------------------------------------------------------------------------------------------
# 10. 13F: ningun gestor ACTIVO respondio (aunque respondan los pasivos) = dato no disponible
# --------------------------------------------------------------------------------------------------------
def _r13_activos(filas, fallidos, consultados, respondieron_activos, consultados_activos=3):
    df = _r13(filas, fallidos=fallidos, consultados=consultados)
    df.attrs.update(consultados_activos=consultados_activos, respondieron_activos=respondieron_activos,
                    consultados_pasivos=consultados - consultados_activos,
                    respondieron_pasivos=consultados - consultados_activos - len([f for f in fallidos if f in
                                                                                  ("Vanguard Group", "State Street")]))
    return df


ACTIVOS_CAIDOS = ["Berkshire Hathaway", "Bridgewater", "Renaissance Technologies"]


def test_13f_solo_respondieron_pasivos_es_none_no_cero():
    df = _r13_activos([_fila("Vanguard Group", 100, 90, pasivo=True), _fila("State Street", 50, 40, pasivo=True)],
                      fallidos=ACTIVOS_CAIDOS, consultados=5, respondieron_activos=0)
    assert m.resumen_13f(df) == (None, None)  # sin el attr nuevo daria (0, 0): "0 gestores activos" inventado
    sin_attr = df.copy()
    sin_attr.attrs.pop("respondieron_activos")
    sin_attr.attrs.pop("consultados_activos")
    assert m.resumen_13f(sin_attr) == (0, 0)  # contrato antiguo: comportamiento anterior intacto
    assert m.construir_respaldo({"valor_usd": 0.0}, True, df)["n_con_posicion"] is None


def test_13f_respondieron_activos_positivo_cuenta_normal():
    df = _r13_activos([_fila("Vanguard Group", 100, 90, pasivo=True), _fila("Bridgewater", 30, 10)],
                      fallidos=["Berkshire Hathaway", "Renaissance Technologies"], consultados=5, respondieron_activos=1)
    assert m.resumen_13f(df) == (1, 1)
    # respondio un activo sin posicion (no aparece en el df): 0 es dato valido
    vacio = _r13_activos([], fallidos=["Berkshire Hathaway"], consultados=5, respondieron_activos=2)
    assert m.resumen_13f(vacio) == (0, 0)


def test_respondieron_activos_cero_sin_consultados_activos_es_none():
    df = _r13_activos([], fallidos=[], consultados=2, respondieron_activos=0, consultados_activos=0)
    assert m.resumen_13f(df) == (None, None)
    assert m.avisos_13f(df)[-1].startswith("13f: no respondió ningún gestor activo (ninguno consultado)")


def test_avisos_13f_nombra_solo_los_activos_caidos():
    df = _r13_activos([_fila("State Street", 50, 40, pasivo=True)],
                      fallidos=ACTIVOS_CAIDOS + ["Vanguard Group"], consultados=5, respondieron_activos=0)
    av = m.avisos_13f(df)
    assert av[0].startswith("13f: no respondieron: Berkshire Hathaway, Bridgewater, Renaissance Technologies, Vanguard")
    assert av[1] == ("13f: no respondió ningún gestor activo (Berkshire Hathaway, Bridgewater, Renaissance Technologies); "
                     "dato 13F no disponible, no cuenta como 0")
    assert m.avisos_13f(_r13([_fila("Bridgewater", 30, 10)])) == []  # sin fallos: sin avisos
    solo_uno = _r13_activos([_fila("Bridgewater", 30, 10)], fallidos=["Berkshire Hathaway"], consultados=5,
                            respondieron_activos=2)
    assert m.avisos_13f(solo_uno) == ["13f: no respondieron: Berkshire Hathaway"]


def test_pipeline_13f_solo_pasivos_excluye_la_mitad_y_avisa(monkeypatch, df_sintetico):
    df = _r13_activos([_fila("Vanguard Group", 100, 90, pasivo=True), _fila("State Street", 50, 40, pasivo=True)],
                      fallidos=ACTIVOS_CAIDOS, consultados=5, respondieron_activos=0)
    _simular(monkeypatch, df_sintetico, r13=df, form4="cero")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert res["respaldo13f"]["n_con_posicion"] is None and res["respaldo13f"]["n_acumulando"] is None
    assert any(a.startswith("13f: no respondió ningún gestor activo (Berkshire Hathaway, Bridgewater") for a in res["avisos"])
    # queda solo la mitad Form 4 (0 USD, dato valido): no se mezcla con un "0 gestores activos" inventado
    assert res["confiabilidad"]["detalle"]["respaldo_13f_form4"]["valor"] == pytest.approx(0.0)
    assert res["r13f_fallidos"] == ACTIVOS_CAIDOS


def test_pipeline_13f_solo_pasivos_y_form4_fallido_excluye_el_componente(monkeypatch, df_sintetico):
    df = _r13_activos([_fila("Vanguard Group", 100, 90, pasivo=True)], fallidos=ACTIVOS_CAIDOS + ["State Street"],
                      consultados=5, respondieron_activos=0)
    _simular(monkeypatch, df_sintetico, r13=df, form4="error")
    res = m.analizar_ticker("AAA", 1000, verbose=False)
    assert "respaldo_13f_form4" in res["componentes_sin_dato"]
