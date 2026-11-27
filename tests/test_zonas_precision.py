"""Precision de las zonas (ancho acotado, sin fusion en cadena, nucleo, compatibilidad, determinismo, sin look-ahead).

Series sinteticas deterministas (sin red) + anclas construidas a mano para probar la logica de ``clusters_pit``.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.validation import backtest_ranges as br

CLAVES_HISTORICAS = {"lo", "hi", "anclas", "n_anclas", "n_familias", "familias", "tipos", "px", "dist_pct", "ancho_pct"}


def _serie(n=1500, vol=0.025, seed=7, drift=0.0004):
    """OHLCV sintetico (paseo geometrico) con indice de dias habiles y volumen aleatorio."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(r))
    opn = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    vol_ = rng.integers(8e5, 2e6, n).astype(float)
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": vol_},
                        index=pd.bdate_range("2015-01-02", periods=n))


def _a(tipo, lo, hi, familia=None, centro=None, puntual=False, peso=1.0):
    fam = familia or br.FAMILIAS[tipo]
    return {"tipo": tipo, "familia": fam, "lo": float(lo), "hi": float(hi),
            "centro": float((lo + hi) / 2 if centro is None else centro), "puntual": puntual, "peso": peso}


def _ancho(c):
    return c["hi_amplio"] - c["lo_amplio"]


# --- Ancho acotado por el tope ---------------------------------------------------------------------------
@pytest.mark.parametrize("vol,seed", [(0.025, 7), (0.045, 11)])
def test_ancho_clusters_acotado_por_tope_atr_y_pct(vol, seed):
    z = br.calcular_zonas_pit(_serie(vol=vol, seed=seed))
    assert z["clusters"], "la serie sintetica debe producir zonas"
    px, atr = z["px"], z["atr_semanal"]
    tope = min(br.ANCHO_MAX_CLUSTER_ATR * atr, br.ANCHO_MAX_CLUSTER_PCT * px) * (1 + 1e-9)
    for c in z["clusters"]:
        assert c["hi"] - c["lo"] <= tope
        assert c["ancho_pct"] <= br.ANCHO_MAX_CLUSTER_PCT + 1e-9
        assert c["hi"] <= px * (1 - br.DIST_MIN) + 1e-9 and c["lo"] >= px * (1 - br.DIST_MAX) - 1e-9


def test_historico_es_mas_ancho_que_el_vigente_en_serie_volatil():
    df = _serie(vol=0.045, seed=11)
    nuevo = br.calcular_zonas_pit(df)
    hist = br.calcular_zonas_pit(df, br.params_zonas(**br.PARAMS_HISTORICOS))
    assert max(c["ancho_pct"] for c in hist["clusters"]) > br.ANCHO_MAX_CLUSTER_PCT
    assert max(c["ancho_pct"] for c in nuevo["clusters"]) <= br.ANCHO_MAX_CLUSTER_PCT + 1e-9


def test_constantes_sobreescribibles_por_monkeypatch(monkeypatch):
    monkeypatch.setattr(br, "ANCHO_MAX_CLUSTER_PCT", 0.02)
    z = br.calcular_zonas_pit(_serie())
    assert all(c["ancho_pct"] <= 0.02 + 1e-9 for c in z["clusters"])


def test_ancla_mas_ancha_que_el_tope_se_acota_y_ob_conserva_su_techo():
    px, atr = 100.0, 2.0                      # tope = min(2 ATR = 4, 5 % = 5) = 4
    hvn = _a("HVN", 70, 90, centro=82.0)      # nodo de 20 de ancho (FWHM completo), pico en 82
    ob = _a("OB", 60, 92)                     # OB de 32 de ancho, techo en 92
    cl = br.clusters_pit([hvn, ob], px, atr)
    assert all(_ancho(c) <= 4.0 + 1e-9 for c in cl)
    c_hvn = next(c for c in cl if "HVN" in c["tipos"])
    assert c_hvn["lo"] <= 82.0 <= c_hvn["hi"]                # el HVN queda centrado en su pico
    c_ob = next(c for c in cl if "OB" in c["tipos"])
    assert c_ob["hi"] == pytest.approx(92.0)                  # el OB conserva su borde superior (toca primero el precio)
    assert c_ob["lo"] == pytest.approx(90.0)                  # ancho = OB_ANCHO_MAX_ATR (1 ATR = 2)


# --- Sin fusion en cadena -----------------------------------------------------------------------------------
def _cadena(n=8, paso=0.6, ancho=1.0, base=80.0):
    tipos = ["SWING", "AVWAP", "LOG", "POC"]
    return [_a(tipos[i % 4], base + paso * i, base + paso * i + ancho, puntual=False) for i in range(n)]


@pytest.mark.parametrize("modo", ["cadena", "ventana"])
def test_cadena_de_anclas_solapadas_no_forma_una_zona_enorme(modo):
    anclas = _cadena()                         # 8 anclas encadenadas: union total = 5.2 de ancho
    cl = br.clusters_pit(anclas, 100.0, 2.0, br.params_zonas(modo=modo))   # tope = 4
    assert len(cl) >= 2
    assert all(_ancho(c) <= 4.0 + 1e-9 for c in cl)
    assert sum(c["n_anclas"] for c in cl) == len(anclas)       # cada ancla en exactamente un cluster
    ids = [id(a) for c in cl for a in c["anclas"]]
    assert len(ids) == len(set(ids))


def test_historico_si_fusiona_la_cadena_completa():
    cl = br.clusters_pit(_cadena(), 100.0, 2.0, br.params_zonas(**br.PARAMS_HISTORICOS))   # tope = 3 ATR = 6
    assert len(cl) == 1 and _ancho(cl[0]) == pytest.approx(5.2)


def test_modo_ventana_elige_la_ventana_de_mayor_confluencia():
    # una ancla aislada de 1 familia cerca del precio y 3 familias distintas juntas mas abajo
    anclas = [_a("OB", 95, 96), _a("POC", 82, 82.8), _a("AVWAP", 82.2, 83.0), _a("SWING", 82.5, 83.3)]
    cl = br.clusters_pit(anclas, 100.0, 2.0, br.params_zonas(modo="ventana"))
    mejor = max(cl, key=lambda c: (c["n_familias"], c["n_anclas"]))
    assert mejor["n_familias"] == 3 and mejor["lo"] < 83


# --- Banda de busqueda: sin astillas -----------------------------------------------------------------------
def test_ancla_fuera_de_la_banda_no_deja_astilla():
    px, atr = 100.0, 5.0                       # banda [70, 99.5]
    astilla = _a("VAH", 67.75, 70.25, centro=69.0, puntual=True)    # solo 0.25 dentro de la banda
    cl = br.clusters_pit([astilla], px, atr)
    assert cl == []                            # vigente: se descarta
    cl_h = br.clusters_pit([astilla], px, atr, br.params_zonas(**br.PARAMS_HISTORICOS))
    assert len(cl_h) == 1 and cl_h[0]["ancho_pct"] < 0.01      # historico: astilla de 0.25 de ancho


def test_ancla_sobre_el_precio_no_es_zona_de_compra():
    avwap = _a("AVWAP", 96.0, 104.0, centro=100.0, puntual=True)    # cruza el precio: solo ~44 % bajo el techo
    assert br.clusters_pit([avwap], 100.0, 4.0) == []


# --- Nucleo --------------------------------------------------------------------------------------------------
def _anclas_nucleo():
    return [_a("POC", 90.0, 90.8), _a("AVWAP", 90.1, 90.9), _a("SWING", 90.2, 91.0), _a("LOG", 90.9, 92.0)]


def test_nucleo_dentro_del_rango_amplio_y_con_la_mayoria_ponderada():
    c = br.clusters_pit(_anclas_nucleo(), 100.0, 2.0)[0]
    assert c["n_anclas"] == 4
    assert c["lo_amplio"] <= c["lo_nucleo"] < c["hi_nucleo"] <= c["hi_amplio"]
    assert c["hi_nucleo"] - c["lo_nucleo"] < _ancho(c)
    pesos = {a["tipo"]: a["peso"] for a in c["anclas"]}
    assert sum(pesos[t] for t in c["anclas_nucleo"]) >= 0.6 * c["peso_total"]
    assert "LOG" not in c["anclas_nucleo"]      # el outlier queda fuera del nucleo
    assert (c["lo"], c["hi"]) == (c["lo_amplio"], c["hi_amplio"])   # por defecto lo/hi = rango amplio


def test_nucleo_operativo_reemplaza_lo_hi_y_conserva_el_amplio():
    amplio = br.clusters_pit(_anclas_nucleo(), 100.0, 2.0)[0]
    c = br.clusters_pit(_anclas_nucleo(), 100.0, 2.0, br.params_zonas(nucleo=True))[0]
    assert (c["lo"], c["hi"]) == (c["lo_nucleo"], c["hi_nucleo"])
    assert (c["lo_amplio"], c["hi_amplio"]) == (amplio["lo_amplio"], amplio["hi_amplio"])
    assert c["ancho_pct"] < c["ancho_amplio_pct"]
    assert c["dist_pct"] == pytest.approx((100.0 - c["hi"]) / 100.0)


def test_nucleo_respeta_ancho_minimo_y_nunca_sale_del_amplio():
    c = br.clusters_pit([_a("POC", 90.0, 90.2)], 100.0, 2.0, br.params_zonas(nucleo=True))[0]
    assert c["lo_amplio"] <= c["lo_nucleo"] <= c["hi_nucleo"] <= c["hi_amplio"]


# --- Compatibilidad de claves y trazabilidad ----------------------------------------------------------------
def test_claves_compatibles_y_trazabilidad_de_pesos():
    z = br.calcular_zonas_pit(_serie())
    assert {"px", "atr_semanal", "clusters", "primario"} <= set(z)
    for c in z["clusters"]:
        assert CLAVES_HISTORICAS <= set(c)
        assert {"lo_amplio", "hi_amplio", "lo_nucleo", "hi_nucleo", "peso_total", "anclas_nucleo"} <= set(c)
        assert c["n_anclas"] == len(c["anclas"]) and c["n_familias"] == len(c["familias"])
        assert all({"tipo", "familia", "lo", "hi", "peso"} <= set(a) for a in c["anclas"])
        assert c["peso_total"] == pytest.approx(sum(a["peso"] for a in c["anclas"]))
        assert set(c["anclas_nucleo"]) <= set(c["tipos"])
        assert c["lo"] < c["hi"] and math.isfinite(c["dist_pct"])
    p = z["primario"]
    assert p is not None and p is br.rango_primario(z["clusters"])
    assert p["n_familias"] == max(c["n_familias"] for c in z["clusters"])


def test_rango_primario_sin_clusters_es_none():
    assert br.rango_primario([]) is None
    assert br.clusters_pit([], 100.0, 2.0) == []


def test_params_historicos_reproducen_el_comportamiento_anterior():
    p = br.params_zonas(**br.PARAMS_HISTORICOS)
    assert (p.ancho_max_cluster_atr, p.ancho_max_cluster_pct, p.acotar_anclas, p.hvn_semiancho_atr,
            p.ob_ancho_max_atr, p.frac_min_en_banda) == (3.0, None, False, None, None, None)


# --- Determinismo y ausencia de look-ahead -----------------------------------------------------------------
def _firma(z):
    return [(round(c["lo"], 9), round(c["hi"], 9), c["n_anclas"], c["n_familias"], tuple(c["tipos"])) for c in z["clusters"]], \
        (None if z["primario"] is None else (round(z["primario"]["lo"], 9), round(z["primario"]["hi"], 9)))


def test_determinismo():
    df = _serie()
    assert _firma(br.calcular_zonas_pit(df)) == _firma(br.calcular_zonas_pit(df.copy()))


def test_sin_look_ahead_en_zonas_df_truncado_igual_a_prefijo():
    df = _serie(n=1500)
    k = 1100
    futuro = df.copy()
    futuro.iloc[k:, :4] = futuro.iloc[k:, :4] * 3.0          # el futuro cambia radicalmente
    a = br.calcular_zonas_pit(df.iloc[:k])
    b = br.calcular_zonas_pit(futuro.iloc[:k])
    assert _firma(a) == _firma(b)
    assert _firma(a) != _firma(br.calcular_zonas_pit(futuro))    # y el calculo si reacciona a datos nuevos


def test_sin_look_ahead_en_backtest_ticker_rangos_no_dependen_del_futuro():
    df = _serie(n=1400)
    k = 1000
    futuro = df.copy()
    futuro.iloc[k:, :4] = futuro.iloc[k:, :4] * 0.4
    base = br.backtest_ticker("SIM", anios=3, paso=63, df=df, verbose=False)
    alt = br.backtest_ticker("SIM", anios=3, paso=63, df=futuro, verbose=False)
    m = base.merge(alt, on="fecha", suffixes=("_b", "_a"))
    previas = m[m["fecha"] < df.index[k - 1]]                 # rangos fijados con datos <= k-1
    assert len(previas) >= 2
    for col in ("r_inf", "r_sup", "px"):
        assert np.allclose(previas[f"{col}_b"].astype(float), previas[f"{col}_a"].astype(float), equal_nan=True)


def test_backtest_ticker_expone_ancho_amplio_y_acepta_cache_de_anclas():
    df = _serie(n=1400)
    cache: dict = {}
    o1 = br.backtest_ticker("SIM", anios=1, paso=63, df=df, verbose=False, anclas_cache=cache)
    assert cache and {"ancho_pct", "ancho_amplio_pct", "lo_amplio", "hi_amplio"} <= set(o1.columns)
    o2 = br.backtest_ticker("SIM", anios=1, paso=63, df=df, verbose=False, anclas_cache=cache,
                            params=br.params_zonas(nucleo=True))
    ok = o1["sin_rango"].eq(False) & o2["sin_rango"].eq(False)
    assert (o2.loc[ok, "ancho_pct"] <= o1.loc[ok, "ancho_pct"] + 1e-12).all()
    assert (o2.loc[ok, "hi_amplio"].astype(float).round(9) == o1.loc[ok, "hi_amplio"].astype(float).round(9)).all()
