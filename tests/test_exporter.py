import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.reporting import exporter as ex


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
    d = json.loads(open(rutas["json"], encoding="utf-8").read())
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
    assert json.loads(open(rutas["json"], encoding="utf-8").read())["n"] == 0


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
