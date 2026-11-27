"""Watchlist personalizada, rutas de datos, modo nube y /health (sin red, sin servidor)."""
import asyncio
import os

import pytest

import app_ui as ui_mod
from src.reporting import userdata


# ---------------------------------------------------------------- funciones puras
def test_normalizar_mayusculas_sin_duplicados_e_invalidos():
    r = userdata.normalizar([" nvda ", {"ticker": "NVDA", "nota": "x"}, "aapl", "no valido!", "", None, "BRK-B"])
    assert [i["ticker"] for i in r] == ["NVDA", "AAPL", "BRK-B"]


def test_guardar_y_leer_roundtrip_atomico(tmp_path):
    ruta = userdata.guardar([{"ticker": "nvda", "nota": "IA, chips"}, "AAPL"], str(tmp_path))
    assert os.path.basename(ruta) == "mi_watchlist.csv"
    assert userdata.leer(str(tmp_path)) == [{"ticker": "NVDA", "nota": "IA; chips"}, {"ticker": "AAPL", "nota": ""}]
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
    with open(ruta, encoding="utf-8") as f:
        assert f.read().startswith("ticker,nota")


def test_leer_tolerante_a_archivo_faltante_o_corrupto(tmp_path):
    assert userdata.leer(str(tmp_path)) == [] and not userdata.existe(str(tmp_path))
    (tmp_path / "mi_watchlist.csv").write_bytes(b"\xff\xfe\x00\x81garbage\x00\x00")
    assert userdata.leer(str(tmp_path)) == []
    (tmp_path / "mi_watchlist.csv").write_text("ticker,nota\n??,x\nmsft,ok\n#c\n", encoding="utf-8")
    assert userdata.leer(str(tmp_path)) == [{"ticker": "MSFT", "nota": "ok"}]


def test_anadir_quitar_fusionar():
    l, ok = userdata.anadir([], "nvda", "nota")
    assert ok and l == [{"ticker": "NVDA", "nota": "nota"}]
    l2, ok2 = userdata.anadir(l, "NVDA")
    assert not ok2 and l2 == l
    assert userdata.anadir(l, "!!")[1] is False
    l3, q = userdata.quitar(l, "nvda")
    assert q and l3 == [] and userdata.quitar(l3, "NVDA")[1] is False
    f, n = userdata.fusionar([{"ticker": "AAPL", "nota": ""}], ["AAPL", "MSFT"])
    assert n == 1 and userdata.tickers(f) == ["AAPL", "MSFT"]


def test_directorio_datos_por_entorno(monkeypatch, tmp_path):
    monkeypatch.delenv("NIVELES_DATA_DIR")
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert userdata.directorio_datos() == str(tmp_path / "NivelesInstitucionales")
    monkeypatch.setenv("PORT", "10000")
    assert userdata.directorio_datos() != str(tmp_path / "NivelesInstitucionales")  # nube: directorio temporal
    assert userdata.directorio_datos().endswith("NivelesInstitucionales")
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "x"))
    assert userdata.directorio_datos() == str(tmp_path / "x")


def test_recurso_congelado_usa_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)
    assert userdata.recurso("watchlist.csv") == str(tmp_path / "watchlist.csv")


def test_watchlist_de_ejemplo_existe_y_es_legible():
    assert userdata.leer_csv(ui_mod.RUTA_WATCHLIST)


# ---------------------------------------------------------------- modo nube
def test_calcular_ticker_en_nube_usa_hilo_y_nunca_run_cpu_bound(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    from nicegui import run

    async def prohibido(*a, **k):
        raise AssertionError("cpu_bound crearia procesos")
    monkeypatch.setattr(run, "cpu_bound", prohibido)
    llamadas = []
    monkeypatch.setattr(ui_mod.nucleo, "analizar_ticker",
                        lambda t, c, **kw: llamadas.append((t, c, kw)) or {"ticker": t, "precio_actual": 1.0})
    r = asyncio.run(ui_mod.calcular_ticker("AAA", 500.0, backtest=False, sec=True))
    assert r["ticker"] == "AAA" and llamadas == [("AAA", 500.0, {"periodo": "3y", "backtest": False, "sec": True, "verbose": False})]   # nube: 3 anios


def test_cache_lru_limita_y_resumen_liviano_quita_pesados():
    c = ui_mod.CacheLRU(3)
    for t in "ABCD":
        c[t] = {"x": t}
    assert list(c) == ["B", "C", "D"]
    c["B"] = {"x": 1}  # reutilizar lo mueve al final
    c["E"] = {}
    assert list(c) == ["D", "B", "E"]
    r = {"ticker": "A", "df": object(), "form4": [1], "r13f": [2], "zonas": []}
    assert ui_mod.resumen_liviano(r) == {"ticker": "A", "zonas": []}


def _correr(cliente, coro_fn):
    async def interno():
        with cliente:  # notify/UI dentro de una tarea nueva necesitan un slot activo
            await coro_fn()
    asyncio.run(interno())


@pytest.fixture
def cliente():
    from nicegui.client import Client
    from nicegui.page import page
    with Client(page("/")) as c:
        yield c


def test_escaneo_en_nube_secuencial_libera_memoria_y_ranking_liviano(cliente, monkeypatch):
    userdata.guardar(["AAA", "BBB", "CCC", "DDD", "EEE"])
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(3))
    vistos, gc_llamadas = [], []

    pedidos_vj = []

    async def falso(t, cap, bt, sec, vj=True):
        vistos.append((t, bt))
        pedidos_vj.append(vj)  # se afirma fuera: un assert aqui dentro lo tragaria el except de escanear
        return {"ticker": t, "precio_actual": 10.0, "df": object(), "form4": None, "r13f": None, "zonas": [],
                "errores": [], "confluencia_score": 50.0, "confiabilidad": {"indice": 50.0, "cobertura": 1.0}}
    monkeypatch.setattr(ui_mod, "calcular_ticker", falso)
    monkeypatch.setattr(ui_mod.gc, "collect", lambda *a: gc_llamadas.append(1) or 0)
    capturado = {}
    orig = ui_mod.filas_ranking

    def espia(rs):
        capturado["rs"] = rs
        return orig(rs)
    monkeypatch.setattr(ui_mod, "filas_ranking", espia)
    v = ui_mod.Vista()
    v.construir()
    assert v.sw_bt_wl.value is False  # backtest desactivado por defecto en la nube
    _correr(cliente, v.escanear)
    assert [t for t, _ in vistos] == ["AAA", "BBB", "CCC", "DDD", "EEE"] and all(bt is False for _, bt in vistos)
    assert len(gc_llamadas) == 5
    assert pedidos_vj == [False] * 5  # en la nube el escaneo NO calcula el valor justo (columna Upside opcional)
    assert len(capturado["rs"]) == 5 and all("df" not in r for r in capturado["rs"])
    assert len(ui_mod.CACHE) == 3 and "AAA" not in ui_mod.CACHE


# ---------------------------------------------------------------- UI watchlist
def test_boton_alterna_agregar_y_quitar_con_persistencia(cliente):
    v = ui_mod.Vista()
    v.construir()
    assert v.items == [] and v.btn_wl.text == "Agregar a mi watchlist" and not v.btn_wl.enabled
    v._ticker_cambiado("nvda")
    assert v.ticker == "NVDA" and v.btn_wl.text == "Agregar NVDA a mi watchlist" and v.btn_wl.enabled
    v.alternar_watchlist()
    assert userdata.tickers(userdata.leer()) == ["NVDA"] and v.btn_wl.text == "Quitar NVDA de mi watchlist"
    assert "NVDA" in v.sel_ticker.options
    v.alternar_watchlist()
    assert userdata.leer() == [] and v.btn_wl.text == "Agregar NVDA a mi watchlist"
    v.alternar_watchlist()
    assert userdata.tickers(userdata.leer()) == ["NVDA"]


def test_pestana_watchlist_manual_ejemplo_vaciar_y_seleccion(cliente):
    from nicegui import ui
    v = ui_mod.Vista()
    v.construir()
    textos = [e.text for e in cliente.elements.values() if isinstance(e, ui.label)]
    assert all(l in textos for l in ui_mod.TEXTO_WATCHLIST_INTRO) and len(ui_mod.TEXTO_WATCHLIST_INTRO) == 3
    v.in_wl_ticker.value = "!!"
    v.anadir_manual()
    assert v.items == []
    v.in_wl_ticker.value = "msft"
    v.in_wl_nota.value = "mega cap"
    v.anadir_manual()
    assert userdata.leer() == [{"ticker": "MSFT", "nota": "mega cap"}] and v.in_wl_ticker.value == ""
    v.cargar_ejemplo()
    ts = userdata.tickers(userdata.leer())
    assert ts[0] == "MSFT" and ts.count("MSFT") == 1 and len(ts) > 3
    assert v.sel_ticker.options == ts and v.watchlist == ts
    v.quitar_ticker("MSFT")
    assert "MSFT" not in userdata.tickers(userdata.leer())
    ts = v.watchlist
    v._marcar(ts[0], False)
    assert ts[0] not in v._seleccionados() and v.btn_scan.text.startswith("Escanear seleccionados")
    v.vaciar_lista()
    assert userdata.leer() == [] and v.btn_scan.text == "Escanear mi watchlist (0)"


def test_escanear_solo_procesa_los_seleccionados_y_lista_vacia_avisa(cliente, monkeypatch):
    userdata.guardar(["AAA", "BBB", "CCC"])
    vistos = []

    async def falso(t, cap, bt, sec, vj=True):
        vistos.append(t)
        return {"ticker": t, "precio_actual": 5.0, "zonas": [], "errores": []}
    monkeypatch.setattr(ui_mod, "calcular_ticker", falso)
    v = ui_mod.Vista()
    v.construir()
    v._marcar("BBB", False)
    _correr(cliente, v.escanear)
    assert vistos == ["AAA", "CCC"]
    v.vaciar_lista()
    vistos.clear()
    _correr(cliente, v.escanear)
    assert vistos == []


def test_autocompletado_toma_la_lista_personalizada(cliente):
    userdata.guardar(["ZZZ", "YYY"])
    v = ui_mod.Vista()
    v.construir()
    assert v.sel_ticker.options == ["ZZZ", "YYY"] and v.ticker == "ZZZ"


def test_modelo_explica_la_watchlist():
    t = " ".join(ui_mod.texto_seccion(s) for s in ui_mod.contenido_modelo() if s["id"] == "watchlist")
    assert "POR TICKER" in t and "no persiste" in t and "SOLO" in t


# ---------------------------------------------------------------- health y config de ejecucion
def test_health_registrado_y_devuelve_ok():
    from nicegui import app
    assert ui_mod.health() == {"status": "ok"}
    assert any(getattr(r, "path", None) == "/health" for r in app.routes)


def test_config_ejecucion_nube_y_local():
    n = ui_mod.config_ejecucion({"PORT": "10000"})
    assert n["host"] == "0.0.0.0" and n["port"] == 10000 and n["native"] is False and n["show"] is False
    assert "window_size" not in n and n["reload"] is False and n["dark"] is True
    loc = ui_mod.config_ejecucion({"APP_UI_PORT": "9001", "APP_UI_SHOW": "0"}, pywebview_ok=True)
    assert loc["host"] == "127.0.0.1" and loc["port"] == 9001 and loc["native"] is True and "show" not in loc
    assert loc["window_size"] == (1440, 900)
    assert ui_mod.config_ejecucion({}, pywebview_ok=True)["port"] == 8080


def test_config_sin_pywebview_cae_a_navegador(capsys):
    c = ui_mod.config_ejecucion({}, pywebview_ok=False)
    assert c["native"] is False and c["show"] is True and "window_size" not in c and "pywebview" in capsys.readouterr().out
    assert ui_mod.config_ejecucion({"APP_UI_NATIVE": "0"})["native"] is False


def test_escaneo_en_nube_no_pasa_valor_justo_a_analizar_ticker_por_defecto(cliente, monkeypatch):
    """Camino real calcular_ticker -> analizar_ticker: en la nube el kw valor_justo llega como False."""
    userdata.guardar(["AAA", "BBB"])
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setattr(ui_mod, "CACHE", ui_mod.CacheLRU(3))
    kws = []
    v = ui_mod.Vista()
    v.construir()   # antes de parchear: la pestana Modelo lee la firma real de analizar_ticker
    monkeypatch.setattr(ui_mod.nucleo, "analizar_ticker",
                        lambda t, c, **kw: kws.append(kw) or {"ticker": t, "precio_actual": 1.0, "zonas": [], "errores": []})
    _correr(cliente, v.escanear)
    assert len(kws) == 2 and all(kw.get("valor_justo") is False for kw in kws)
    v.sw_vj_wl.value = True                      # opt-in explicito: ya no se pasa el kw (analizar_ticker usa True)
    kws.clear()
    _correr(cliente, v.escanear)
    assert len(kws) == 2 and all("valor_justo" not in kw for kw in kws)
