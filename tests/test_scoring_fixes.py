"""Correcciones de src/validation/scoring.py y coherencia de docs/METODOLOGIA.md con el codigo.

Principio probado: un dato NO DISPONIBLE (None/NaN) se excluye y los pesos se renormalizan (la cobertura baja);
un dato obtenido que vale cero es un dato y puntua cero. Sin red ni caches: todo es sintetico.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from src.validation import scoring as sc

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DOC = os.path.join(RAIZ, "docs", "METODOLOGIA.md")

# Pesos de prueba (a-priori): independientes de pesos_calibrados.json.
P = dict(sc.PESOS_PRIOR)
BT = {"bounce": 0.669, "ret60_mediana": 0.062, "mae60_mediana": -0.084, "absorcion": 0.263}
R13_OK = {"n_acumulando": 2, "n_con_posicion": 3, "compras_form4_usd": 500_000.0}
SA_OK = {"pts": 7.5, "disponible": True}
SA_FALLO = {"pts": None, "disponible": False, "motivo": "sin red"}


def _indice(bt=BT, r13=R13_OK, sa=SA_OK, n_fam=3, n_val=1, absorcion=None):
    return sc.indice_desde_modulos(bt, r13, sa, absorcion, n_fam, n_val, pesos=P)


def _esperado(comp: dict) -> float:
    """Promedio ponderado manual de los componentes con dato (pesos de prueba)."""
    pres = {k: v for k, v in comp.items() if v is not None}
    return sum(P[k] * v for k, v in pres.items()) / sum(P[k] for k in pres)


# --- (a) cargar_pesos cierra el fichero -------------------------------------------------------------------
def test_cargar_pesos_cierra_el_fichero(tmp_path, monkeypatch):
    ruta = tmp_path / "pesos.json"
    pesos = {k: 0.2 for k in sc.COMPONENTES}
    ruta.write_text(json.dumps({"pesos": pesos}), encoding="utf-8")
    abiertos, real_open = [], open

    def espia(*a, **k):
        f = real_open(*a, **k)
        abiertos.append(f)
        return f

    monkeypatch.setattr(sc, "RUTA_PESOS", str(ruta))
    monkeypatch.setattr(sc, "open", espia, raising=False)  # el modulo resuelve `open` en su namespace antes que el builtin
    assert sc.cargar_pesos() == pesos
    assert abiertos, "cargar_pesos no abrio el fichero con el open del modulo"
    assert all(f.closed for f in abiertos), "cargar_pesos dejo el fichero abierto"


def test_cargar_pesos_sin_fichero_o_corrupto_devuelve_apriori_sin_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "RUTA_PESOS", str(tmp_path / "no_existe.json"))
    p = sc.cargar_pesos()
    assert p == sc.PESOS_PRIOR
    p["backtest"] = 99.0
    assert sc.PESOS_PRIOR["backtest"] != 99.0  # copia, no alias
    mal = tmp_path / "mal.json"
    mal.write_text("{no es json", encoding="utf-8")
    monkeypatch.setattr(sc, "RUTA_PESOS", str(mal))
    assert sc.cargar_pesos() == sc.PESOS_PRIOR


# --- (b) s_13f: mitades disponibles ---------------------------------------------------------------------------
@pytest.mark.parametrize("args,esperado", [
    ((None, None, None), None),                  # nada disponible -> componente excluido
    ((None, None), None),
    ((float("nan"), float("nan"), float("nan")), None),
    ((None, 5, None), None),                     # fraccion no calculable sin n_acumulando
    ((3, 6, None), 50.0),                        # solo gestores: vale el 100 % del componente (0.5 de la fraccion)
    ((6, 6, None), 100.0),
    ((0, 6, None), 0.0),
    ((None, None, 500_000), 50.0),               # solo Form 4: vale el 100 % del componente
    ((None, None, 1_000_000), 100.0),
    ((None, None, 5_000_000), 100.0),            # satura en 1 M USD
    ((None, None, 0.0), 0.0),                    # 0 USD consultado correctamente: dato valido
    ((0, 0, None), 0.0),                         # "0 gestores con posicion": dato valido, NO None
    ((None, 0, None), 0.0),                      # n_con_posicion == 0 basta, aunque n_acumulando sea None
    ((0, 0, 1_000_000), 50.0),                   # fraccion 0 (valida) y Form 4 al maximo
    ((3, 6, 500_000), 50.0),
    ((3, 6, 1_000_000), 75.0),
    ((6, 6, 1_000_000), 100.0),
    ((10, 10, 0), 50.0),
    ((0, 10, 1_000_000), 50.0),
    ((None, 5, 1_000_000), 100.0),               # mitad de gestores no calculable: solo Form 4
    ((8, 5, None), 100.0),                       # acotada a 1
    ((3, 6, float("nan")), 50.0),                # NaN en Form 4 = no disponible
])
def test_s_13f_promedia_solo_las_mitades_con_dato(args, esperado):
    r = sc.s_13f(*args)
    if esperado is None:
        assert r is None
    else:
        assert r == pytest.approx(esperado)


def test_s_13f_antes_subestimaba_a_la_mitad():
    """Caso del bug: 13F sin respuesta (None) con Form 4 disponible salia a 50 % de lo debido; ahora vale el 100 %."""
    assert sc.s_13f(None, None, 1_000_000) == pytest.approx(100.0)
    # y Form 4 fallido con gestores disponibles ya no cuenta como 0 USD
    assert sc.s_13f(6, 6, None) == pytest.approx(100.0)
    # pero un 0 real si baja el score
    assert sc.s_13f(6, 6, 0.0) == pytest.approx(50.0)


# --- (c) s_stock_act y exclusion en el indice -------------------------------------------------------------------
@pytest.mark.parametrize("pts,esperado", [(None, None), (float("nan"), None), (0, 0.0), (0.0, 0.0), (7.5, 50.0),
                                          (15, 100.0), (30, 100.0), (-3, 0.0)])
def test_s_stock_act_none_es_no_disponible_y_cero_es_dato(pts, esperado):
    r = sc.s_stock_act(pts)
    if esperado is None:
        assert r is None
    else:
        assert r == pytest.approx(esperado)


def test_stock_act_fallido_se_excluye_del_indice():
    r = _indice(sa=SA_FALLO)
    assert "stock_act" not in r["detalle"]
    assert r["cobertura"] == pytest.approx(1 - P["stock_act"])


@pytest.mark.parametrize("sa", [None, {}, {"pts": None}, {"pts": float("nan")}, {"pts": None, "disponible": False},
                                {"pts": 12.0, "disponible": False}])  # disponible=False prevalece sobre un pts residual
def test_stock_act_no_disponible_en_todas_sus_formas(sa):
    r = _indice(sa=sa)
    assert "stock_act" not in r["detalle"]
    assert r["cobertura"] == pytest.approx(1 - P["stock_act"])


def test_stock_act_cero_consultado_es_dato_y_no_baja_la_cobertura():
    r = _indice(sa={"pts": 0.0, "disponible": True})
    assert r["detalle"]["stock_act"]["valor"] == 0.0
    assert r["cobertura"] == pytest.approx(1.0)


def test_stock_act_sin_clave_disponible_con_pts_numerico_sigue_siendo_dato():
    r = _indice(sa={"pts": 7.5})  # formato anterior (sin 'disponible')
    assert r["detalle"]["stock_act"]["valor"] == pytest.approx(50.0)


def test_cobertura_baja_nunca_se_infla_y_pesos_se_renormalizan():
    completo = _indice()
    assert completo["cobertura"] == pytest.approx(1.0)
    assert set(completo["detalle"]) == set(sc.COMPONENTES)
    assert sum(d["peso_efectivo"] for d in completo["detalle"].values()) == pytest.approx(1.0)

    sin_sa = _indice(sa=SA_FALLO)
    # Form 4 fallido pero gestores 13F disponibles: el componente sigue presente (cobertura igual) con la mitad de gestores
    sin_f4 = _indice(sa=SA_FALLO, r13={"n_acumulando": 2, "n_con_posicion": 3, "compras_form4_usd": None})
    assert sin_f4["cobertura"] == pytest.approx(sin_sa["cobertura"])
    assert sin_f4["detalle"]["respaldo_13f_form4"]["valor"] == pytest.approx(100 * 2 / 3)
    # 13F sin respuesta y Form 4 fallido: el componente desaparece y la cobertura baja
    sin_13f = _indice(sa=SA_FALLO, r13={"n_acumulando": None, "n_con_posicion": None, "compras_form4_usd": None})
    assert "respaldo_13f_form4" not in sin_13f["detalle"]
    assert sin_13f["cobertura"] == pytest.approx(1 - P["stock_act"] - P["respaldo_13f_form4"])
    assert completo["cobertura"] > sin_sa["cobertura"] > sin_13f["cobertura"]

    # pesos efectivos = peso / suma de pesos presentes (suman 1)
    pres = [k for k in sc.COMPONENTES if k in sin_13f["detalle"]]
    wsum = sum(P[k] for k in pres)
    for k in pres:
        assert sin_13f["detalle"][k]["peso_efectivo"] == pytest.approx(P[k] / wsum)
    assert sum(d["peso_efectivo"] for d in sin_13f["detalle"].values()) == pytest.approx(1.0)


def test_indice_es_promedio_de_lo_disponible_no_cero_ni_neutro():
    comp = {"respaldo_13f_form4": sc.s_13f(2, 3, 500_000), "stock_act": None, "absorcion": sc.s_absorcion(0.263),
            "backtest": sc.s_backtest(0.669, 0.062, -0.084), "confluencia": sc.s_confluencia(3, 1)}
    r = _indice(sa=SA_FALLO)
    assert r["indice"] == pytest.approx(_esperado(comp))
    # sustituir el dato faltante por 0 o por neutro (50) daria otro indice: ya no se hace
    con_cero = _esperado({**comp, "stock_act": 0.0})
    con_neutro = _esperado({**comp, "stock_act": 50.0})
    assert r["indice"] != pytest.approx(con_cero) and r["indice"] != pytest.approx(con_neutro)


@pytest.mark.parametrize("r13", [None, {}, {"n_acumulando": None, "n_con_posicion": None, "compras_form4_usd": None}])
def test_respaldo13f_ausente_o_con_claves_none_excluye_el_componente(r13):
    r = _indice(r13=r13)
    assert "respaldo_13f_form4" not in r["detalle"]
    assert r["cobertura"] == pytest.approx(1 - P["respaldo_13f_form4"])


def test_respaldo13f_cero_gestores_es_dato_valido():
    r = _indice(r13={"n_acumulando": 0, "n_con_posicion": 0, "compras_form4_usd": 0.0})
    assert r["detalle"]["respaldo_13f_form4"]["valor"] == 0.0
    assert r["cobertura"] == pytest.approx(1.0)


def test_todo_sin_datos_externos_deja_backtest_absorcion_y_confluencia():
    r = _indice(r13=None, sa=None)
    assert set(r["detalle"]) == {"absorcion", "backtest", "confluencia"}
    assert r["cobertura"] == pytest.approx(P["absorcion"] + P["backtest"] + P["confluencia"])
    assert r["indice"] is not None


@pytest.mark.parametrize("bt", [None, {}])
def test_backtest_ausente_excluye_backtest_y_absorcion(bt):
    r = _indice(bt=bt)
    assert "backtest" not in r["detalle"] and "absorcion" not in r["detalle"]
    assert r["cobertura"] == pytest.approx(1 - P["backtest"] - P["absorcion"])
    # ... salvo que la absorcion se pase aparte
    r2 = _indice(bt=bt, absorcion=0.30)
    assert r2["detalle"]["absorcion"]["valor"] == pytest.approx(30.0)


def test_backtest_sin_fills_nan_excluye_backtest_y_absorcion():
    bt = {"bounce": float("nan"), "ret60_mediana": float("nan"), "mae60_mediana": float("nan"), "absorcion": float("nan")}
    r = _indice(bt=bt)
    assert "backtest" not in r["detalle"] and "absorcion" not in r["detalle"]


def test_nada_disponible_devuelve_indice_none_y_cobertura_cero():
    r = sc.indice_desde_modulos(None, None, None, None, None, 0, pesos=P)
    assert r == {"indice": None, "cobertura": 0.0, "detalle": {}}


# --- s_confluencia: valoracion no consultada (None) frente a consultada sin anclas (0) --------------------------------
def _confluencia_completa_previa(n_fam, n_val):
    """Formula completa vigente antes del cambio (referencia de no regresion para n_valoracion entero)."""
    return 100 * min(1.0, max(0.0, 0.6 * (n_fam - 1) / 3 + 0.4 * min(1, n_val / 2)))


@pytest.mark.parametrize("n_fam", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("n_val", [0, 1, 2, 3])
def test_confluencia_con_valoracion_consultada_no_cambia(n_fam, n_val):
    assert sc.s_confluencia(n_fam, n_val) == pytest.approx(_confluencia_completa_previa(n_fam, n_val))


def test_confluencia_valores_conocidos_y_default_cero():
    assert sc.s_confluencia(1, 0) == 0 and sc.s_confluencia(4, 0) == pytest.approx(60)
    assert sc.s_confluencia(1, 2) == pytest.approx(40) and sc.s_confluencia(4, 2) == pytest.approx(100)
    assert sc.s_confluencia(3) == pytest.approx(sc.s_confluencia(3, 0)) == pytest.approx(40)  # default n_valoracion=0


@pytest.mark.parametrize("n_fam,esperado", [(0, 0.0), (1, 0.0), (2, 100 / 3), (3, 200 / 3), (4, 100.0), (5, 100.0),
                                            (8, 100.0)])
def test_confluencia_valoracion_no_consultada_reescala_las_familias_a_0_100(n_fam, esperado):
    """Formula: 100 * clip((n_familias - 1) / 3, 0, 1); 4 familias -> 100 (no 60)."""
    assert sc.s_confluencia(n_fam, None) == pytest.approx(esperado)
    assert sc.s_confluencia(n_fam, float("nan")) == pytest.approx(esperado)  # NaN = no consultada


def test_confluencia_none_distinto_de_cero():
    assert sc.s_confluencia(3, None) == pytest.approx(200 / 3)
    assert sc.s_confluencia(3, 0) == pytest.approx(40.0)  # consultada y sin anclas: dato valido que puntua 0
    assert sc.s_confluencia(3, None) != sc.s_confluencia(3, 0)
    # el reescalado equivale a dividir la parte de familias de la formula completa (con 0 anclas) por su peso 0.6
    for n in (1, 2, 3, 4):
        assert sc.s_confluencia(n, None) == pytest.approx(sc.s_confluencia(n, 0) / 0.6)


@pytest.mark.parametrize("n_val", [None, 0, 1, 2, float("nan")])
def test_confluencia_sin_familias_es_none_sea_cual_sea_la_valoracion(n_val):
    assert sc.s_confluencia(None, n_val) is None
    assert sc.s_confluencia(float("nan"), n_val) is None


def test_confluencia_none_no_lanza_typeerror():
    """Regresion: min(1, None / 2) lanzaba TypeError (incluso con 1 familia, que es lo que sondea main.py)."""
    assert sc.s_confluencia(1, None) == 0.0


def test_indice_desde_modulos_pasa_n_valoracion_none_tal_cual():
    sin_val = _indice(n_fam=4, n_val=None)
    assert sin_val["detalle"]["confluencia"]["valor"] == pytest.approx(100.0)
    assert sin_val["cobertura"] == pytest.approx(1.0)  # el componente sigue presente (solo familias)
    consultada = _indice(n_fam=4, n_val=0)
    assert consultada["detalle"]["confluencia"]["valor"] == pytest.approx(60.0)
    # el indice es el promedio ponderado con el valor reescalado
    comp = {"respaldo_13f_form4": sc.s_13f(2, 3, 500_000), "stock_act": sc.s_stock_act(7.5),
            "absorcion": sc.s_absorcion(0.263), "backtest": sc.s_backtest(0.669, 0.062, -0.084),
            "confluencia": sc.s_confluencia(4, None)}
    assert sin_val["indice"] == pytest.approx(_esperado(comp))
    # sin zona (n_familias None) el componente se excluye aunque n_valoracion sea None
    sin_zona = _indice(n_fam=None, n_val=None)
    assert "confluencia" not in sin_zona["detalle"]
    assert sin_zona["cobertura"] == pytest.approx(1 - P["confluencia"])


def test_indice_desde_modulos_default_n_valoracion_cero_no_cambia():
    por_defecto = sc.indice_desde_modulos(BT, R13_OK, SA_OK, None, 3, pesos=P)
    explicito = sc.indice_desde_modulos(BT, R13_OK, SA_OK, None, 3, 0, pesos=P)
    assert por_defecto == explicito
    assert por_defecto["detalle"]["confluencia"]["valor"] == pytest.approx(_confluencia_completa_previa(3, 0))


def test_main_n_valoracion_para_scoring_pasa_none_y_avisa_que_se_excluye():
    """Probe de main.py (solo import): con s_confluencia admitiendo None, main pasa None y deja de decir que puntua 0."""
    from main import n_valoracion_para_scoring
    val, aviso = n_valoracion_para_scoring(None)
    assert val is None
    assert aviso and "excluye" in aviso and "puntua 0" not in aviso
    assert n_valoracion_para_scoring(0) == (0, None)   # consultada sin anclas: dato valido, sin aviso
    assert n_valoracion_para_scoring(2) == (2, None)


# --- (d) calcular_indice_confiabilidad y su cobertura -----------------------------------------------------------
def test_calcular_indice_renormaliza_y_documenta_cobertura():
    r = sc.calcular_indice_confiabilidad({"absorcion": 40.0, "backtest": 80.0, "stock_act": None}, P)
    assert r["indice"] == pytest.approx((0.20 * 40 + 0.30 * 80) / 0.50)
    assert r["cobertura"] == pytest.approx(0.50)
    assert r["detalle"]["absorcion"]["peso_efectivo"] == pytest.approx(0.4)
    assert r["detalle"]["backtest"]["peso_efectivo"] == pytest.approx(0.6)
    doc = sc.calcular_indice_confiabilidad.__doc__
    assert "cobertura" in doc and "no disponible" in doc.lower()


def test_calcular_indice_valor_cero_es_dato_nan_y_claves_ajenas_no_cuentan():
    r = sc.calcular_indice_confiabilidad({"absorcion": 0.0, "backtest": 100.0, "stock_act": float("nan"),
                                          "inventada": 50.0}, P)
    assert r["indice"] == pytest.approx(60.0) and r["cobertura"] == pytest.approx(0.5)
    assert set(r["detalle"]) == {"absorcion", "backtest"}


def test_cobertura_se_mide_contra_la_suma_total_de_pesos():
    r = sc.calcular_indice_confiabilidad({"a": 10.0}, {"a": 2.0, "b": 2.0})  # pesos que no suman 1
    assert r["cobertura"] == pytest.approx(0.5) and r["indice"] == pytest.approx(10.0)


def test_calcular_indice_todo_none_o_pesos_cero():
    assert sc.calcular_indice_confiabilidad({k: None for k in sc.COMPONENTES}, P) == \
        {"indice": None, "cobertura": 0.0, "detalle": {}}
    r = sc.calcular_indice_confiabilidad({"a": 10.0}, {"a": 0.0, "b": 1.0})  # sin peso positivo no hay promedio definido
    assert r["indice"] is None and r["cobertura"] == 0.0


# --- Coherencia doc <-> codigo (evita que METODOLOGIA.md describa algo distinto de lo que hace el codigo) ------------
def _barras(n=120, k=60, cambios=None):
    """Serie plana (nunca toca el rango) con barras modificadas por indice: {j: dict(O=,H=,L=,C=,V=)}."""
    O, H, L, C, V = (np.full(n, v) for v in (99.2, 99.5, 99.0, 99.3, 1000.0))
    for j, c in (cambios or {}).items():
        O[j], H[j], L[j], C[j], V[j] = c.get("O", O[j]), c.get("H", H[j]), c.get("L", L[j]), c.get("C", C[j]), c.get("V", V[j])
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"Open": O, "High": H, "Low": L, "Close": C, "Volume": V}, index=idx), np.full(n, 1000.0)


I0, K = 50, 60
R_INF, R_SUP = 90.0, 95.0
TOQUE = {"O": 96.0, "H": 97.0, "L": 94.0, "C": 96.0, "V": 1000.0}      # llena (Low <= r_sup), sin absorcion
VELA_ABS = {"O": 93.5, "H": 96.0, "L": 93.0, "C": 95.8, "V": 5000.0}  # rechazo en rango, vol > 1.35 x SMA, cierre alto


def _evaluar(cambios):
    from src.validation import backtest_ranges as br
    h, sma = _barras(cambios=cambios)
    return br.evaluar_orden(h, I0, R_INF, R_SUP, sma)


@pytest.mark.parametrize("pos,cuenta", [(0, True), (1, True), (5, True), (6, False)])
def test_ventana_de_absorcion_son_la_vela_del_toque_mas_5(pos, cuenta):
    """Doc 2.2: 6 velas (k .. k+5), contando la del toque."""
    cambios = {K: dict(TOQUE), K + pos: dict(VELA_ABS)}
    r = _evaluar(cambios)
    assert r["llenada"] and r["idx_fill"] == K
    assert r["absorcion"] is cuenta


def test_absorcion_exige_interseccion_no_contencion_y_close_sobre_r_inf():
    cierra_sobre_rsup = dict(VELA_ABS, H=99.0, C=98.9)  # cierre por encima de r_sup: sigue siendo absorcion
    assert _evaluar({K: dict(TOQUE), K + 2: cierra_sobre_rsup})["absorcion"] is True
    cierra_bajo_rinf = dict(VELA_ABS, L=85.0, C=89.0, H=89.5)  # vela bajo r_inf (no intersecta): no cuenta
    assert _evaluar({K: dict(TOQUE), K + 2: cierra_bajo_rinf})["absorcion"] is False
    volumen_insuficiente = dict(VELA_ABS, V=1349.0)  # debe superar 1.35 x SMA(20) = 1350
    assert _evaluar({K: dict(TOQUE), K + 2: volumen_insuficiente})["absorcion"] is False


def test_bounce_invalida_con_el_minimo_intradia_no_con_el_cierre():
    """Doc 2.2: Low < r_inf x 0.97 invalida aunque la vela cierre sobre r_inf."""
    meta = {"H": 100.5}  # >= entrada (95) x 1.05 = 99.75
    toque = dict(TOQUE)
    mecha = {K + 1: {"L": R_INF * 0.96, "C": R_INF * 1.01}, K + 2: meta}  # mecha bajo el umbral, cierre sano
    assert _evaluar({K: toque, **mecha})["bounce"] is False
    sana = {K + 1: {"L": R_INF * 0.98, "C": R_INF * 1.01}, K + 2: meta}  # Low sobre el umbral
    assert _evaluar({K: toque, **sana})["bounce"] is True
    # misma barra con meta e invalidacion: gana la invalidacion
    misma = {K + 1: {"L": R_INF * 0.96, "H": 100.5}}
    assert _evaluar({K: toque, **misma})["bounce"] is False


def test_fill_profundo_solo_cuenta_el_dia_del_llenado():
    """Doc 2.2: Low[k] <= r_inf el dia del llenado; una perforacion posterior no cuenta."""
    perfora_despues = {K: dict(TOQUE), K + 3: {"L": 80.0}}
    assert _evaluar(perfora_despues)["profunda"] is False
    perfora_el_dia = {K: dict(TOQUE, L=89.0)}
    assert _evaluar(perfora_el_dia)["profunda"] is True


def _ordenes_sinteticas():
    base = dict(sin_rango=False, llenada=True, profunda=False, dias_hasta_fill=5, absorcion=False, bounce=True,
                ret60=0.05, ret20=0.01, ret120=0.1, ret252=0.2, mae60=-0.05, mae252=-0.1)
    filas = [dict(base, ticker="XXX", fecha=pd.Timestamp("2024-01-31"), fecha_fill=pd.Timestamp("2024-02-05"),
                  ventana_completa=True, n_familias=2, has_logreg=False),
             dict(base, ticker="XXX", fecha=pd.Timestamp("2024-02-29"), fecha_fill=pd.Timestamp("2024-03-04"),
                  ventana_completa=False, n_familias=3, has_logreg=True)]  # orden con vida incompleta pero con ret60
    return pd.DataFrame(filas)


def test_resumen_filtra_ventana_completa_y_evidencia_no():
    """Doc 2.4: 236 (resumen) frente a 237 (evidencia / tabla_calibracion)."""
    from src.validation import backtest_ranges as br
    o = _ordenes_sinteticas()
    assert br.resumen(o)["n_fills"] == 1
    assert sc.evidencia(o, None)["absorcion"]["n"] == 2


# --- docs/METODOLOGIA.md consistente con codigo y pesos -------------------------------------------------------------
@pytest.fixture(scope="module")
def doc() -> str:
    if not os.path.exists(RUTA_DOC):
        pytest.skip("docs/METODOLOGIA.md no disponible")
    with open(RUTA_DOC, encoding="utf-8") as f:
        return f.read()


def test_doc_describe_bounce_con_minimo_intradia(doc):
    assert "minimo intradia" in doc and "Low < r_inf x (1 - 0.03)" in doc
    assert "antes de cerrar bajo" not in doc  # frase anterior (incorrecta)


def test_doc_ventana_de_absorcion_coincide_con_el_codigo(doc):
    from src.validation import backtest_ranges as br
    assert br.VENTANA_ABS == 5 and br.BOUNCE_VENTANA == 20 and br.BOUNCE_PCT == 0.05 and br.BOUNCE_INVALIDA == 0.03
    assert "6 velas" in doc and "range(k, k + 5 + 1)" in doc


def test_doc_lista_de_pasivos_y_gestores_coincide_con_el_codigo(doc):
    pytest.importorskip("edgar")
    from src.data import sec_edgar
    from src.validation import case_studies as cs
    for nombre in cs.PASIVOS:
        assert nombre in doc, nombre
    for nombre in sec_edgar.GESTORES_13F:  # nomina real del screener en vivo
        assert nombre in doc, nombre
    n_total = len(cs._gestores_all())
    assert f"los {n_total} gestores" in doc and f"{n_total - len(cs.PASIVOS)} son activos" in doc
    assert f"**{len(sec_edgar.GESTORES_13F)} gestores**" in doc


def test_doc_peso_de_la_confluencia_en_el_ranking_sale_de_pesos_calibrados(doc):
    if not os.path.exists(sc.RUTA_PESOS):
        pytest.skip("pesos_calibrados.json no disponible")
    pesos = sc.cargar_pesos()
    w = pesos["confluencia"]
    total = 0.5 + 0.5 * w
    assert f"{total * 100:.1f} %" in doc and f"{total:.4f}" in doc and f"{w:.4f}" in doc
    sin_sa = pesos["confluencia"] / (1 - pesos["stock_act"])
    assert f"{(0.5 + 0.5 * sin_sa) * 100:.1f} %" in doc
    # coberturas citadas en 4.2
    assert f"{(1 - pesos['stock_act']) * 100:.1f} %" in doc
    assert f"{(pesos['confluencia'] + pesos['absorcion'] + pesos['backtest']) * 100:.1f} %" in doc  # sin SEC
    assert f"{(pesos['respaldo_13f_form4'] + pesos['stock_act'] + pesos['confluencia']) * 100:.1f} %" in doc  # sin backtest


def test_doc_documenta_la_valoracion_no_consultada(doc):
    assert "n_valoracion=None" in doc and "clip((n_familias - 1) / 3, 0, 1)" in doc
    assert "ii corregida" in doc and "tampoco es comparable entre tickers con valoracion consultada" in doc
    assert "n_valoracion = 0" not in doc  # descripcion anterior (la valoracion fallida puntuaba 0)


def test_doc_registra_las_correcciones_de_hoy(doc):
    assert "## 7. Cambios de correcciones (2026-11-26)" in doc
    assert "no son cero" in doc and "cobertura" in doc
    assert "los componentes sin dato se excluyen" in doc
    assert math.isfinite(sc.s_13f(1, 2, 0.0))  # el modulo sigue importable y operativo
