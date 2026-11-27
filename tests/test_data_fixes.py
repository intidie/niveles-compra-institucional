"""Contratos de src/data/stock_act.py y src/data/sec_edgar.py ante fallos (sin red: todo con mocks)."""
import json
import warnings

import pandas as pd
import pytest

from src.data import sec_edgar as se
from src.data import stock_act as sa

CLAVES_STOCK_ACT = {"pts", "disponible", "motivo", "compras", "ventas", "resumen", "ultimas_tx"}


# ------------------------------------------------------------------ STOCK Act
def _repo(tmp_path, archivos):
    """Crea <tmp>/repo/public/data/<ruta> con el contenido dado (dict/list -> JSON; str -> texto crudo)."""
    repo = tmp_path / "repo"
    data = repo / "public" / "data"
    data.mkdir(parents=True)
    for ruta, contenido in archivos.items():
        destino = data / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(contenido if isinstance(contenido, str) else json.dumps(contenido), encoding="utf-8")
    return str(repo)


@pytest.fixture
def sin_sync(monkeypatch):
    monkeypatch.setattr(sa, "sincronizar_repo", lambda *a, **k: None)


def _tx(ticker, tipo, fecha, politico="P1", monto="$1,001 - $15,000"):
    return {"ticker": ticker, "transaction_type": tipo, "transaction_date": fecha, "filer": politico,
            "amount": monto, "chamber": "house"}


def test_stock_act_fallo_de_descarga_no_es_neutro(monkeypatch, tmp_path):
    def falla(*a, **k):
        raise OSError("sin red")
    monkeypatch.setattr(sa, "sincronizar_repo", falla)
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=str(tmp_path / "x"))
    assert set(r) == CLAVES_STOCK_ACT
    assert r["pts"] is None and r["disponible"] is False
    assert "OSError" in r["motivo"] and r["compras"] == 0 and r["ventas"] == 0 and r["ultimas_tx"] == []


def test_stock_act_carpeta_ausente(sin_sync, tmp_path):
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=str(tmp_path / "no_existe"))
    assert r["pts"] is None and r["disponible"] is False and r["motivo"]


def test_stock_act_ticker_vacio(sin_sync, tmp_path):
    for t in ("", "   ", None):
        r = sa.consultar_flujo_congreso_stock_act(t, repo_dir=str(tmp_path))
        assert r["pts"] is None and r["disponible"] is False


def test_stock_act_consultado_sin_operaciones_es_cero_disponible(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": [_tx("MSFT", "Purchase", "2025-01-01")]})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert set(r) == CLAVES_STOCK_ACT
    assert r["pts"] == 0 and r["disponible"] is True and r["motivo"] is None
    assert r["compras"] == 0 and r["ventas"] == 0


def test_stock_act_con_operaciones(sin_sync, tmp_path):
    ops = [_tx("AAPL", "Purchase", "2025-03-01", "A"), _tx("AAPL", "Purchase", "2025-02-01", "B"),
           _tx("AAPL", "Sale (Full)", "2025-01-01", "C"), _tx("AAPL", "Sale (Partial)", "2024-12-01", "D"),
           _tx("MSFT", "Purchase", "2025-03-05", "E")]
    repo = _repo(tmp_path, {"ticker/AAPL.json": {"ticker": "AAPL", "trades": ops}})
    r = sa.consultar_flujo_congreso_stock_act("aapl", repo_dir=repo)
    assert set(r) == CLAVES_STOCK_ACT
    assert r["disponible"] is True and r["motivo"] is None
    assert (r["compras"], r["ventas"]) == (2, 2)
    assert r["pts"] == pytest.approx(7.5)          # 2 de 4 ultimas ops son compras -> 0.5 * 15
    assert len(r["ultimas_tx"]) == 4 and r["ultimas_tx"][0]["fecha"] == "2025-03-01"


def test_stock_act_solo_ventas_es_cero_con_datos(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": [_tx("AAPL", "Sale (Full)", "2025-01-01")]})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["disponible"] is True and r["pts"] == 0 and r["ventas"] == 1


def test_stock_act_json_ilegibles_no_disponible(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": "{ esto no es json"})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["pts"] is None and r["disponible"] is False and "legible" in r["motivo"]


def test_stock_act_esquema_irreconocible_no_disponible(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"otro.json": {"cosa": [1, 2, 3]}})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["pts"] is None and r["disponible"] is False and "ticker" in r["motivo"]


def test_stock_act_sin_ops_pero_con_archivo_ilegible_no_afirma_cero(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": [_tx("MSFT", "Purchase", "2025-01-01")], "roto.json": "{{{"})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["pts"] is None and r["disponible"] is False and "ilegible" in r["motivo"]


def test_stock_act_ops_con_archivo_ilegible_sigue_disponible(sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": [_tx("AAPL", "Purchase", "2025-01-01")], "roto.json": "{{{"})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["disponible"] is True and r["compras"] == 1 and r["pts"] == pytest.approx(15.0)


def test_stock_act_error_inesperado_no_lanza(monkeypatch, sin_sync, tmp_path):
    repo = _repo(tmp_path, {"trades.json": []})
    monkeypatch.setattr(sa.glob, "glob", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["pts"] is None and r["disponible"] is False and "RuntimeError" in r["motivo"]


# ------------------------------------------------------------------ Fakes de edgartools
class _Filing:
    def __init__(self, obj=None, error=None):
        self._obj, self._error = obj, error

    def obj(self):
        if self._error is not None:
            raise self._error
        return self._obj


class _Filings(list):
    def latest(self, n):
        return _Filings(self[:n])


class _Empresa:
    def __init__(self, filings=None, error=None):
        self._filings, self._error = filings, error

    def get_filings(self, form=None, filing_date=None, **kw):
        if self._error is not None:
            raise self._error
        return _Filings(self._filings or [])


class _F13:
    def __init__(self, filas, periodo="2025-12-31", infotable_ok=True):
        self.infotable = (pd.DataFrame(filas, columns=["Ticker", "SharesPrnAmount", "Value"])
                          if infotable_ok else None)
        self.report_period = periodo


class _F4:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


@pytest.fixture
def ident(monkeypatch):
    monkeypatch.setattr(se, "_asegurar_identidad", lambda: None)


def _parchear_empresas(monkeypatch, por_cik):
    """por_cik: {cik: _Empresa | Exception}; Company(cik) las devuelve o lanza."""
    def fabrica(ident_):
        valor = por_cik[ident_]
        if isinstance(valor, Exception):
            raise valor
        return valor
    monkeypatch.setattr(se, "Company", fabrica)


def _empresa13(actual, previo):
    return _Empresa([_Filing(actual), _Filing(previo)])


# ------------------------------------------------------------------ 13F
GESTORES = {"Berkshire Hathaway": 1, "Vanguard Group": 2, "Gestor Roto": 3, "Sin Posicion": 4}


def test_13f_columnas_pasivo_y_attrs(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {
        1: _empresa13(_F13([("AAPL", 150, 1500.0)]), _F13([("AAPL", 100, 900.0)])),   # activo acumula
        2: _empresa13(_F13([("AAPL", 2000, 2e4)]), _F13([("AAPL", 1500, 1.4e4)])),    # pasivo acumula
        3: RuntimeError("EDGAR no respondio\ncon detalle"),
        4: _empresa13(_F13([("MSFT", 10, 1.0)]), _F13([("MSFT", 10, 1.0)])),          # responde, sin AAPL
    })
    with pytest.warns(RuntimeWarning, match="Gestor Roto"):
        df = se.obtener_13f_ticker(" aapl ", GESTORES)
    assert list(df.columns) == ["gestor", "periodo", "acciones", "valor_usd", "acciones_previas", "cambio_pct",
                                "pasivo"]
    assert list(df["gestor"]) == ["Berkshire Hathaway", "Vanguard Group"]       # conserva activos y pasivos
    assert df["pasivo"].dtype == bool and list(df["pasivo"]) == [False, True]
    assert df.loc[0, "cambio_pct"] == pytest.approx(50.0)
    assert df.attrs["consultados"] == 4
    assert df.attrs["fallidos"] == ["Gestor Roto"]
    assert df.attrs["errores"]["Gestor Roto"].startswith("RuntimeError: EDGAR no respondio con detalle")
    assert df.attrs["sin_posicion"] == ["Sin Posicion"]                          # distinto de "fallido"
    assert df.attrs["respondieron"] == 3


def test_13f_todos_fallan_devuelve_vacio_con_fallidos(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {1: TimeoutError("t"), 2: ConnectionError("c"), 3: ValueError("v"),
                                     4: _Empresa([])})  # 4: sin 13F-HR -> tambien "no respondio"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = se.obtener_13f_ticker("AAPL", GESTORES)
    assert df.empty and "cambio_pct" in df.columns and "pasivo" in df.columns
    assert df.attrs["fallidos"] == list(GESTORES) and df.attrs["consultados"] == 4
    assert df.attrs["respondieron"] == 0 and df.attrs["sin_posicion"] == []
    assert df.attrs["errores"]["Berkshire Hathaway"].startswith("TimeoutError")
    assert df.attrs["errores"]["Sin Posicion"].startswith("LookupError")
    assert any("dato no disponible" in str(x.message) for x in w)
    r = se.resumir_13f(df)
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_acumulando"] is None
    assert r["fallidos"] == list(GESTORES) and r["respondieron_activos"] == 0


def test_13f_infotable_ilegible_es_fallo_no_posicion_nueva(monkeypatch, ident):
    # Si el 13F previo no es legible NO se debe leer como "posicion nueva" (falso acumulando).
    _parchear_empresas(monkeypatch, {1: _empresa13(_F13([("AAPL", 150, 1.0)]), _F13([], infotable_ok=False))})
    with pytest.warns(RuntimeWarning):
        df = se.obtener_13f_ticker("AAPL", {"Berkshire Hathaway": 1})
    assert df.empty and df.attrs["fallidos"] == ["Berkshire Hathaway"]
    assert "infotable" in df.attrs["errores"]["Berkshire Hathaway"]


def test_13f_posicion_nueva_y_salida_total(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {
        1: _empresa13(_F13([("AAPL", 50, 1.0)]), _F13([("MSFT", 1, 1.0)])),    # nueva
        2: _empresa13(_F13([("MSFT", 1, 1.0)]), _F13([("AAPL", 80, 1.0)])),    # salida total (-100 %)
    })
    df = se.obtener_13f_ticker("AAPL", {"Berkshire Hathaway": 1, "Bridgewater": 2})
    assert df.loc[df.gestor == "Berkshire Hathaway", "cambio_pct"].isna().all()
    assert df.loc[df.gestor == "Bridgewater", "cambio_pct"].iloc[0] == pytest.approx(-100.0)
    assert df.attrs["fallidos"] == [] and df["pasivo"].tolist() == [False, False]


def test_resumir_13f_excluye_pasivos_del_flag(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {
        1: _empresa13(_F13([("AAPL", 150, 1.0)]), _F13([("AAPL", 100, 1.0)])),   # activo acumula
        2: _empresa13(_F13([("AAPL", 2000, 1.0)]), _F13([("AAPL", 1500, 1.0)])),  # Vanguard acumula (pasivo)
        3: _empresa13(_F13([("AAPL", 5, 1.0)]), _F13([("AAPL", 10, 1.0)])),       # activo reduce
    })
    df = se.obtener_13f_ticker("AAPL", {"Berkshire Hathaway": 1, "Vanguard Group": 2, "Bridgewater": 3})
    r = se.resumir_13f(df)
    assert r["disponible"] is True
    assert (r["n_con_posicion"], r["n_acumulando"]) == (2, 1)                # solo activos
    assert (r["n_con_posicion_total"], r["n_pasivos_con_posicion"], r["n_pasivos_acumulando"]) == (3, 1, 1)


def test_resumir_13f_respondieron_sin_posicion_es_cero_real(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {1: _empresa13(_F13([("MSFT", 1, 1.0)]), _F13([("MSFT", 1, 1.0)]))})
    r = se.resumir_13f(se.obtener_13f_ticker("AAPL", {"Berkshire Hathaway": 1}))
    assert r["disponible"] is True and r["n_con_posicion"] == 0 and r["n_acumulando"] == 0


def test_pasivos_coinciden_con_case_studies():
    cs = pytest.importorskip("src.validation.case_studies")
    assert set(se.GESTORES_PASIVOS) == set(cs.PASIVOS)
    assert set(se.GESTORES_PASIVOS) <= set(se.GESTORES_13F)


# ------------------------------------------------------------------ 13F: cobertura activos vs pasivos
MIXTOS = {"Berkshire Hathaway": 1, "Bridgewater": 2, "Vanguard Group": 3, "State Street": 4}
ATTRS_COBERTURA = ("consultados", "respondieron", "consultados_activos", "respondieron_activos",
                   "consultados_pasivos", "respondieron_pasivos")


def _con_pos(n_act=150, n_prev=100):
    return _empresa13(_F13([("AAPL", n_act, 1.0)]), _F13([("AAPL", n_prev, 1.0)]))


def _sin_pos():
    return _empresa13(_F13([("MSFT", 1, 1.0)]), _F13([("MSFT", 1, 1.0)]))


def _cobertura(df):
    return tuple(df.attrs[k] for k in ATTRS_COBERTURA)


def test_13f_solo_fallan_los_activos(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {1: TimeoutError("t"), 2: ConnectionError("c"),
                                     3: _con_pos(), 4: _sin_pos()})
    with pytest.warns(RuntimeWarning):
        df = se.obtener_13f_ticker("AAPL", MIXTOS)
    assert _cobertura(df) == (4, 2, 2, 0, 2, 2)
    assert df.attrs["fallidos"] == ["Berkshire Hathaway", "Bridgewater"]
    assert df.attrs["sin_posicion"] == ["State Street"]
    assert df.attrs["sin_posicion_activos"] == [] and df.attrs["sin_posicion_pasivos"] == ["State Street"]
    assert list(df["gestor"]) == ["Vanguard Group"] and df["pasivo"].tolist() == [True]
    r = se.resumir_13f(df)
    # solo respondieron pasivos: sin dato de conviccion activa -> None, no 0
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_acumulando"] is None
    assert (r["consultados_activos"], r["respondieron_activos"]) == (2, 0)
    assert (r["consultados_pasivos"], r["respondieron_pasivos"]) == (2, 2)
    assert (r["n_con_posicion_total"], r["n_pasivos_con_posicion"], r["n_pasivos_acumulando"]) == (1, 1, 1)
    assert r["fallidos"] == ["Berkshire Hathaway", "Bridgewater"]


def test_13f_solo_fallan_los_pasivos(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {1: _con_pos(), 2: _sin_pos(),
                                     3: OSError("o"), 4: LookupError("l")})
    with pytest.warns(RuntimeWarning):
        df = se.obtener_13f_ticker("AAPL", MIXTOS)
    assert _cobertura(df) == (4, 2, 2, 2, 2, 0)
    assert df.attrs["sin_posicion_activos"] == ["Bridgewater"] and df.attrs["sin_posicion_pasivos"] == []
    r = se.resumir_13f(df)
    assert r["disponible"] is True and (r["n_con_posicion"], r["n_acumulando"]) == (1, 1)
    assert (r["respondieron_pasivos"], r["n_pasivos_con_posicion"]) == (0, 0)


def test_13f_todos_fallan_conserva_attrs_de_cobertura(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {i: RuntimeError("x") for i in (1, 2, 3, 4)})
    with pytest.warns(RuntimeWarning):
        df = se.obtener_13f_ticker("AAPL", MIXTOS)
    assert df.empty and _cobertura(df) == (4, 0, 2, 0, 2, 0)
    assert df.attrs["sin_posicion_activos"] == [] and df.attrs["sin_posicion_pasivos"] == []
    r = se.resumir_13f(df)
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_acumulando"] is None
    assert (r["consultados_activos"], r["respondieron_activos"],
            r["consultados_pasivos"], r["respondieron_pasivos"]) == (2, 0, 2, 0)
    assert (r["n_con_posicion_total"], r["n_pasivos_con_posicion"], r["n_pasivos_acumulando"]) == (0, 0, 0)


def test_13f_gestores_personalizados_clasifican_por_nombre(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {10: _con_pos(), 11: _sin_pos(), 12: _con_pos(), 13: RuntimeError("x")})
    custom = {"Pershing Square": 10, "Fondo X": 11, "State Street": 12, "Vanguard Group": 13}
    with pytest.warns(RuntimeWarning):
        df = se.obtener_13f_ticker("AAPL", custom)
    assert _cobertura(df) == (4, 3, 2, 2, 2, 1)
    assert df.attrs["sin_posicion_activos"] == ["Fondo X"] and df.attrs["fallidos"] == ["Vanguard Group"]
    r = se.resumir_13f(df)
    assert r["disponible"] is True and (r["n_con_posicion"], r["n_acumulando"]) == (1, 1)
    assert (r["n_pasivos_con_posicion"], r["n_pasivos_acumulando"]) == (1, 1)


def test_13f_gestores_personalizados_solo_pasivos_no_hay_dato_activo(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {12: _con_pos()})
    df = se.obtener_13f_ticker("AAPL", {"State Street": 12})
    assert _cobertura(df) == (1, 1, 0, 0, 1, 1)
    r = se.resumir_13f(df)
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_pasivos_con_posicion"] == 1


def _df13(filas, attrs=None, con_pasivo=True):
    """DataFrame 13F armado a mano: filas = (gestor, acciones, acciones_previas)."""
    reg = [{"gestor": g, "periodo": "2025-12-31", "acciones": a, "valor_usd": 1.0, "acciones_previas": p,
            "cambio_pct": ((a - p) / p * 100.0) if p > 0 else None, "pasivo": g in se.GESTORES_PASIVOS}
           for g, a, p in filas]
    df = pd.DataFrame(reg, columns=se.COLUMNAS_13F)
    if not con_pasivo:
        df = df.drop(columns=["pasivo"])
    df.attrs.update(attrs or {})
    return df


def test_resumir_13f_con_attrs_nuevos():
    df = _df13([("Vanguard Group", 10, 5)],
               {"consultados": 3, "respondieron": 1, "fallidos": ["A", "B"], "sin_posicion": [],
                "consultados_activos": 2, "respondieron_activos": 0,
                "consultados_pasivos": 1, "respondieron_pasivos": 1})
    r = se.resumir_13f(df)
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_acumulando"] is None
    assert (r["consultados"], r["respondieron"]) == (3, 1)
    assert (r["consultados_activos"], r["respondieron_activos"]) == (2, 0)
    assert (r["consultados_pasivos"], r["respondieron_pasivos"]) == (1, 1)
    assert (r["n_pasivos_con_posicion"], r["n_pasivos_acumulando"]) == (1, 1)


def test_resumir_13f_con_attrs_antiguos_infiere_cobertura():
    # Formato previo: solo consultados / respondieron / fallidos / errores / sin_posicion.
    antiguos = {"consultados": 3, "respondieron": 1, "fallidos": ["Berkshire Hathaway", "Bridgewater"],
                "errores": {}, "sin_posicion": []}
    r = se.resumir_13f(_df13([("Vanguard Group", 10, 5)], antiguos))
    assert r["disponible"] is False and r["n_con_posicion"] is None and r["n_acumulando"] is None
    assert (r["respondieron_activos"], r["respondieron_pasivos"]) == (0, 1)
    assert (r["consultados_activos"], r["consultados_pasivos"]) == (2, 1)
    # Un activo respondio sin posicion (consta en sin_posicion): cero real, no "no disponible".
    antiguos2 = {"consultados": 2, "respondieron": 1, "fallidos": ["Vanguard Group"], "errores": {},
                 "sin_posicion": ["Berkshire Hathaway"]}
    r2 = se.resumir_13f(_df13([], antiguos2))
    assert r2["disponible"] is True and (r2["n_con_posicion"], r2["n_acumulando"]) == (0, 0)
    assert (r2["respondieron_activos"], r2["consultados_pasivos"], r2["respondieron_pasivos"]) == (1, 1, 0)


def test_resumir_13f_sin_ningun_attr_no_falla():
    r = se.resumir_13f(_df13([("Berkshire Hathaway", 150, 100), ("Vanguard Group", 10, 5)]))
    assert r["disponible"] is True and (r["n_con_posicion"], r["n_acumulando"]) == (1, 1)
    assert (r["respondieron_activos"], r["respondieron_pasivos"]) == (1, 1)
    solo_pasivo = se.resumir_13f(_df13([("State Street", 10, 5)]))
    assert solo_pasivo["disponible"] is False and solo_pasivo["n_con_posicion"] is None
    assert solo_pasivo["n_pasivos_con_posicion"] == 1
    vacio = se.resumir_13f(_df13([]))
    assert vacio["disponible"] is False and vacio["n_con_posicion"] is None and vacio["n_con_posicion_total"] == 0
    # DataFrame heredado sin columna 'pasivo': se clasifica por nombre.
    sin_col = se.resumir_13f(_df13([("Berkshire Hathaway", 150, 100), ("Vanguard Group", 10, 5)], con_pasivo=False))
    assert (sin_col["n_con_posicion"], sin_col["n_pasivos_con_posicion"]) == (1, 1)


def test_resumir_13f_fila_activa_prevalece_sobre_attr_contradictorio():
    df = _df13([("Berkshire Hathaway", 150, 100)], {"consultados_activos": 1, "respondieron_activos": 0})
    r = se.resumir_13f(df)
    assert r["disponible"] is True and r["respondieron_activos"] == 1 and r["n_con_posicion"] == 1


# ------------------------------------------------------------------ Form 4
def _f4(filas):
    return _Filing(_F4(pd.DataFrame(filas, columns=["Code", "Shares", "Value", "Ticker"])))


def test_form4_solo_codigo_p(monkeypatch, ident):
    filings = [_f4([("P", 100, 5000.0, "AAPL"), ("S", 50, 2500.0, "AAPL"), ("A", 10, None, "AAPL")]),
               _f4([("P", 40, 2000.0, "AAPL"), ("M", 5, None, "AAPL")])]
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa(filings)})
    compras, res = se.compras_insiders("AAPL")
    assert res["compras"] == 2 and res["acciones"] == 140 and res["valor_usd"] == 7000.0
    assert res["transacciones_analizadas"] == 5 and res["disponible"] is True
    assert res["filings_total"] == 2 and res["filings_fallidos"] == 0
    assert set(compras["Code"]) == {"P"}


def test_form4_sin_filings_es_cero_real(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa([])})
    compras, res = se.compras_insiders("AAPL")
    assert compras.empty and res["compras"] == 0 and res["disponible"] is True and res["filings_total"] == 0


def test_form4_error_de_consulta_se_propaga(monkeypatch, ident):
    _parchear_empresas(monkeypatch, {"AAPL": ConnectionError("sin red")})
    with pytest.raises(ConnectionError):
        se.compras_insiders("AAPL")
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa(error=TimeoutError("lento"))})
    with pytest.raises(TimeoutError):
        se.compras_insiders("AAPL")


def test_form4_todos_los_filings_ilegibles_lanza(monkeypatch, ident):
    filings = [_Filing(error=ValueError("xml roto")) for _ in range(3)]
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa(filings)})
    with pytest.raises(RuntimeError, match="3 de 3"):
        se.compras_insiders("AAPL")


def test_form4_mayoria_ilegible_lanza_y_minoria_avisa(monkeypatch, ident):
    ok = _f4([("P", 10, 100.0, "AAPL")])
    mal = _Filing(error=ValueError("xml roto"))
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa([ok, mal, mal])})        # 2/3 fallan
    with pytest.raises(RuntimeError):
        se.compras_insiders("AAPL")
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa([ok, ok, ok, mal])})     # 1/4 falla: tolerado con aviso
    with pytest.warns(RuntimeWarning, match="1 de 4"):
        compras, res = se.compras_insiders("AAPL")
    assert res["compras"] == 3 and res["filings_total"] == 4 and res["filings_fallidos"] == 1


def test_form4_sin_columna_code_es_cero_real(monkeypatch, ident):
    # edgartools omite 'Code' cuando ningun Form 4 trae transacciones: cero real, no error.
    solo_meta = _Filing(_F4(pd.DataFrame([{"Insider": "X", "Ticker": "AAPL"}])))
    _parchear_empresas(monkeypatch, {"AAPL": _Empresa([solo_meta])})
    compras, res = se.compras_insiders("AAPL")
    assert compras.empty and res["compras"] == 0 and res["disponible"] is True and res["filings_total"] == 1


# ------------------------------------------------------------------ STOCK Act: de-duplicado y campos reales
def _reg(id_, ticker="AAPL", tipo="Purchase", fecha="2025-03-01", nombre="Ed Case", fid="house_ed_case",
         label="$1,001 - $15,000", owner="SP"):
    return {"id": id_, "ticker": ticker, "transaction_type": tipo, "transaction_date": fecha,
            "filer_id": fid, "filer_name": nombre, "amount_range_label": label, "owner": owner,
            "chamber": "house"}


def test_stock_act_duplicado_real_en_varios_json_se_colapsa(sin_sync, tmp_path):
    t = _reg("house_1_g0")
    repo = _repo(tmp_path, {"ticker/AAPL.json": {"ticker": "AAPL", "trades": [t]},
                            "trades.json": [dict(t)], "filer/house_ed_case.json": {"filer": {}, "trades": [dict(t)]}})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert (r["compras"], r["ventas"]) == (1, 0)


def test_stock_act_operaciones_distintas_mismo_dia_se_conservan(sin_sync, tmp_path):
    ops = [_reg("a"), _reg("b"),                                         # mismo filer/fecha/tipo/monto, ids distintos
           _reg("c", nombre="Otro", fid="house_otro"),                  # otro filer
           _reg("d", label="$15,001 - $50,000"),                        # otro monto
           _reg("e", tipo="Sale (Full)")]
    repo = _repo(tmp_path, {"ticker/AAPL.json": {"ticker": "AAPL", "trades": ops}, "trades.json": [dict(o) for o in ops]})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert (r["compras"], r["ventas"]) == (4, 1)


def test_stock_act_sin_id_usa_clave_compuesta(sin_sync, tmp_path):
    a = _reg(None)
    b = _reg(None, owner="JT")
    repo = _repo(tmp_path, {"trades.json": [a, dict(a), b]})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["compras"] == 2


def test_stock_act_nombres_y_montos_reales_y_pts_en_rango(sin_sync, tmp_path):
    ops = [_reg(f"x{i}", tipo="Purchase" if i % 3 else "Sale (Partial)", fecha=f"2025-03-{i + 1:02d}",
                nombre=f"Persona {i}", label=f"$1,001 - ${i + 2}0,000") for i in range(20)]
    repo = _repo(tmp_path, {"ticker/AAPL.json": {"ticker": "AAPL", "trades": ops}})
    r = sa.consultar_flujo_congreso_stock_act("AAPL", repo_dir=repo)
    assert r["compras"] + r["ventas"] == 20
    assert r["ultimas_tx"][0]["politico"] == "Persona 19" and r["ultimas_tx"][0]["monto"].startswith("$1,001 - $")
    assert all(t["politico"] != "Funcionario" and t["monto"] for t in r["ultimas_tx"])
    assert all(not k.startswith("_") for t in r["ultimas_tx"] for k in t)
    assert 0 <= r["pts"] <= 15
