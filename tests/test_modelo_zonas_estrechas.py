"""Alineacion de la UI/Modelo con las zonas estrechas y la recalibracion (sin red)."""
import pytest

import app_ui as ui_mod
from src.validation import backtest_ranges as br
from src.validation import scoring


def _txt(id_):
    return " ".join(ui_mod.texto_seccion(s) for s in ui_mod.contenido_modelo() if s["id"] == id_)


@pytest.fixture
def cliente():
    from nicegui.client import Client
    from nicegui.page import page
    with Client(page("/")) as c:
        yield c


def test_ancho_zona_pct():
    z = {"r_inf": 95.0, "r_sup": 100.0}
    assert ui_mod.ancho_zona_pct(z, 100.0) == pytest.approx(5.0)
    assert ui_mod.ancho_zona_pct(z, None) == pytest.approx(5.0 / 97.5 * 100)
    assert ui_mod.ancho_zona_pct({}, 100.0) is None
    fila = ui_mod.filas_tabla([z], None, 200.0)[0]
    assert fila["ancho_pct"] == pytest.approx(2.5)


def test_familias_explica_zonas_estrechas_con_valores_del_codigo():
    t = _txt("familias")
    assert f"{br.ANCHO_MAX_CLUSTER_PCT * 100:.0f} % del precio" in t
    assert "2 ATR" in t and "107,35" in t and "2026-11-25" in t
    assert "no se usa como rango operativo" in t  # NUCLEO_OPERATIVO=False
    assert "raros" in t and "-3,7 pp" in t and "mejora demostrada" in t


def test_nucleo_operativo_se_lee_del_codigo(monkeypatch):
    monkeypatch.setattr(br, "NUCLEO_OPERATIVO", True)
    assert "activo como rango operativo" in _txt("familias")


def test_limitaciones_honestas_sobre_score_y_absorcion():
    t = _txt("limitaciones")
    assert "≈ 0" not in t and "cero" not in t.lower().replace("ceros", "")
    assert "no discrimina resultados en conjunto" in t
    assert "+0,154" in t and "-0,084" in t
    assert "look-ahead" in t and "no concluyente" in t.lower() or "no concluyente" in t
    w = scoring.cargar_pesos()["absorcion"]
    assert f"{w * 100:.1f}".replace(".", ",") + " %" in t  # peso leido de scoring, no hardcodeado
    assert "Confluencia rara" in t


def test_tabla_zonas_con_cero_uno_y_dos_zonas(cliente, df_sintetico, monkeypatch):
    base = {"ticker": "TEST", "precio_actual": 100.0, "df": df_sintetico, "errores": [], "form4": None, "r13f": None,
            "confluencia_score": 20.0, "confiabilidad": {"indice": 40.0, "cobertura": 0.8}, "backtest_resumen": None}
    zs = [{"nivel": 1, "r_inf": 90.0, "r_sup": 94.0, "p_centro": 92.0, "dist_pct": -6.0, "n_familias": 1, "n_anclas": 1,
           "familias": ["volumen"], "tipos": ["HVN"], "fuente": "HVN"},
          {"nivel": 2, "r_inf": 80.0, "r_sup": 83.0, "p_centro": 81.5, "dist_pct": -17.0, "n_familias": 2,
           "n_anclas": 2, "familias": ["swing", "ob"], "tipos": ["SWING", "OB"], "fuente": "SWING+OB"}]
    v = ui_mod.Vista()
    v.construir()
    for n in (0, 1, 2):
        monkeypatch.setitem(ui_mod.CACHE, "TEST", {**base, "zonas": [dict(z) for z in zs[:n]]})
        v.ticker = "TEST"
        v.refrescar()
        assert len(v.tabla.rows) == n
        # filtrar_zonas renumera por familias: se compara el conjunto de anchos, no el numero de nivel
        assert sorted(r["ancho_pct"] for r in v.tabla.rows) == pytest.approx(sorted([4.0, 3.0][:n][::-1] if n == 2 else [4.0][:n]))
