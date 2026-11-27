import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.reporting import exporter as ex


def _leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _res():
    return [{"ticker": "AAA", "fecha": "2026-01-01", "precio_actual": np.float64(10.5), "capital": 1000,
             "zonas": [{"nivel": 1, "r_inf": 8.0, "r_sup": 9.0, "n_familias": 3, "familias": ["volumen", "ob", "swing"],
                        "fecha": pd.Timestamp("2025-05-05")}],
             "confluencia_score": np.float64(60.0),
             "confiabilidad": {"indice": float("nan"), "cobertura": 0.4},
             "backtest_resumen": {"fill_rate": np.float32(0.5), "bounce": np.nan},
             "insiders": {"compras": np.int64(2), "valor_usd": 5000.0},
             "form4_compras": pd.DataFrame({"Code": ["P"], "Date": [pd.Timestamp("2025-01-01")]}),
             "errores": ["x: fallo"]},
            {"ticker": "BBB", "errores": []}]


def test_a_serializable_tipos():
    out = ex.a_serializable({"a": np.int64(3), "b": np.float64("nan"), "c": pd.Timestamp("2025-01-02"),
                             "d": np.array([1, 2]), "e": pd.NaT, "f": {1, }, "g": np.bool_(True), "h": float("inf")})
    assert out["a"] == 3 and out["b"] is None and out["c"].startswith("2025-01-02")
    assert out["d"] == [1, 2] and out["e"] is None and out["f"] == [1] and out["g"] is True and out["h"] is None
    json.dumps(out, allow_nan=False)


def test_exportar_csv_json(tmp_path):
    rutas = ex.exportar_resultados(_res(), "t", str(tmp_path))
    assert set(rutas) == {"csv", "csv_zonas", "json"}
    d = _leer_json(rutas["json"])
    assert d["n"] == 2 and d["resultados"][0]["backtest_resumen"]["bounce"] is None
    assert d["resultados"][0]["form4_compras"][0]["Code"] == "P"
    df = pd.read_csv(rutas["csv"])
    assert list(df["ticker"]) == ["AAA", "BBB"] and df.loc[0, "z1_n_familias"] == 3
    assert df.loc[0, "z1_familias"] == "volumen+ob+swing"
    z = pd.read_csv(rutas["csv_zonas"])
    assert len(z) == 1 and z.loc[0, "ticker"] == "AAA"


def test_solo_un_formato_y_invalido(tmp_path):
    assert list(ex.exportar_resultados(_res(), "t", str(tmp_path), "json")) == ["json"]
    assert "json" not in ex.exportar_resultados(_res(), "t2", str(tmp_path), "csv")
    with pytest.raises(ValueError):
        ex.exportar_resultados(_res(), "t", str(tmp_path), ["xlsx"])


def test_lista_vacia(tmp_path):
    rutas = ex.exportar_resultados([], "vacio", str(tmp_path))
    assert _leer_json(rutas["json"])["n"] == 0


def test_colab_directorio_y_descarga(monkeypatch, tmp_path):
    assert ex.es_colab() is False
    descargados = []
    gc = types.ModuleType("google.colab")
    gc.files = types.SimpleNamespace(download=descargados.append)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", gc)
    assert ex.es_colab() is True
    rutas = ex.exportar_resultados(_res(), "c", str(tmp_path), "json")
    assert descargados == [rutas["json"]]


def test_directorio_local_por_defecto(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert ex.directorio_salida() == "resultados" and (tmp_path / "resultados").is_dir()


def _res_completo():
    r = _res()[0]
    r.update(avisos=["muestra pequeña: 4 fills (< 10); x", "13f: no respondieron: A, B"],
             componentes_sin_dato=["stock_act", "respaldo_13f_form4"], r13f_fallidos=["A", "B"],
             stock_act={"pts": None, "disponible": False, "motivo": "sin red"},
             respaldo13f={"n_con_posicion": None, "n_acumulando": None, "compras_form4_usd": 0.0},
             n_ordenes=12, n_fills=4, anios_backtest=5,
             backtest_alcance="rango primario (N1), backtest de 5 años; no comparable con el pool de docs/METODOLOGIA.md")
    return r


def test_aplanar_incluye_alcance_cobertura_avisos_y_muestra():
    f = ex.aplanar_resultado(_res_completo())
    assert f["capital_alcance"] == ex.CAPITAL_ALCANCE and "por ticker" in f["capital_alcance"]
    assert f["bt_n_ordenes"] == 12 and f["bt_n_fills"] == 4 and f["bt_anios"] == 5
    assert "rango primario (N1)" in f["bt_alcance"]
    assert f["confiabilidad_sin_dato"] == "stock_act; respaldo_13f_form4"
    assert f["gestores_13f_fallidos"] == "A; B" and f["gestores_13f_con_posicion"] is None  # None, no 0
    assert f["stock_act_pts"] is None and f["stock_act_disponible"] is False
    assert "muestra pequeña: 4 fills" in f["avisos"] and "13f: no respondieron" in f["avisos"]
    vacio = ex.aplanar_resultado({"ticker": "Z"})
    assert vacio["avisos"] is None and vacio["confiabilidad_sin_dato"] is None and vacio["bt_n_fills"] is None


def test_exportacion_lleva_notas_y_columnas_de_alcance(tmp_path):
    rutas = ex.exportar_resultados([_res_completo()], "n", str(tmp_path))
    d = _leer_json(rutas["json"])
    assert set(ex.NOTAS) <= set(d["notas"])
    assert "no se reparte entre tickers" in d["notas"]["capital"] and "r_sup" in d["notas"]["dist_pct"]
    assert "rango primario (N1)" in d["notas"]["backtest"] and "no son comparables" in d["notas"]["backtest"]
    assert d["resultados"][0]["avisos"][0].startswith("muestra pequeña")
    df = pd.read_csv(rutas["csv"])
    assert df.loc[0, "capital_alcance"] == ex.CAPITAL_ALCANCE and df.loc[0, "bt_n_fills"] == 4
    assert pd.isna(df.loc[0, "gestores_13f_con_posicion"])  # dato no disponible: celda vacia, no 0
    z = pd.read_csv(rutas["csv_zonas"])
    assert (z["dist_ref"] == "r_sup").all()


def test_notas_personalizadas(tmp_path):
    rutas = ex.exportar_resultados(_res(), "p", str(tmp_path), "json", notas={"x": "y"})
    assert _leer_json(rutas["json"])["notas"] == {"x": "y"}

