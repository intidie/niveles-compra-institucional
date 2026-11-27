"""Cache SQLite local (src/reporting/cache_local.py): TTL, version, corrupcion, concurrencia, purga y tamano. Todo en tmp_path."""
import json
import os
import sqlite3
import threading
import time

import numpy as np
import pytest

from src.reporting import cache_local as cl
from src.reporting.cache_local import CacheLocal


@pytest.fixture
def cache(tmp_path):
    return CacheLocal(str(tmp_path / "c" / "cache.sqlite"))


@pytest.fixture(autouse=True)
def _entorno(monkeypatch, tmp_path):
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "datos"))
    monkeypatch.delenv("PORT", raising=False)


# ------------------------------------------------------------------------------------------ basico
def test_guardar_y_obtener_roundtrip_json(cache):
    v = {"a": 1, "b": [1.5, None, "x"], "c": {"d": True}}
    assert cache.guardar("k", v, "entradas", "AAPL") is True
    assert cache.obtener("k") == v
    assert cache.obtener("no-existe") is None and cache.obtener("no-existe", defecto=7) == 7
    assert os.path.exists(cache.ruta)        # el directorio de datos se crea al escribir


def test_conversion_de_numpy_nan_y_fechas(cache):
    import pandas as pd
    cache.guardar("k", {"n": np.float64(2.5), "i": np.int64(3), "nan": float("nan"), "inf": float("inf"),
                        "arr": np.array([1.0, 2.0]), "f": pd.Timestamp("2024-05-06"), "t": (1, 2)})
    assert cache.obtener("k") == {"n": 2.5, "i": 3, "nan": None, "inf": None, "arr": [1.0, 2.0],
                                  "f": "2024-05-06T00:00:00", "t": [1, 2]}


def test_valor_no_serializable_no_rompe(cache):
    class Raro:
        def __repr__(self):
            return "raro"
    assert cache.guardar("k", {"x": Raro()}) is True and cache.obtener("k") == {"x": "raro"}


def test_reemplaza_la_misma_clave(cache):
    cache.guardar("k", 1)
    cache.guardar("k", 2)
    assert cache.obtener("k") == 2 and cache.estadisticas()["entradas"] == 1


# ------------------------------------------------------------------------------------------ TTL
def test_ttl_explicito_y_por_tipo(cache):
    assert cl.TTL_POR_TIPO == {"entradas": 24 * 3600.0, "rf": 24 * 3600.0, "info": 6 * 3600.0, "serie_vj": 24 * 3600.0}
    cache.guardar("corta", "x", ttl=0.05)
    cache.guardar("larga", "y", "info")
    time.sleep(0.12)
    assert cache.obtener("corta") is None            # expiro (y se elimina)
    assert cache.obtener("larga") == "y"
    assert cache.estadisticas()["entradas"] == 1


def test_ttl_por_tipo_se_guarda_en_la_fila(cache):
    t0 = time.time()
    cache.guardar("i", 1, "info")
    cache.guardar("e", 1, "entradas")
    con = sqlite3.connect(cache.ruta)
    filas = dict(con.execute("SELECT tipo, expira - creado FROM cache").fetchall())
    con.close()
    assert filas["info"] == pytest.approx(6 * 3600, abs=1) and filas["entradas"] == pytest.approx(24 * 3600, abs=1)
    assert time.time() - t0 < 5


def test_purgar_expirados(cache):
    cache.guardar("a", 1, ttl=0.3)
    cache.guardar("b", 2, ttl=3600)
    time.sleep(0.4)
    assert cache.purgar(expirados=True) == 1 and cache.obtener("b") == 2


# ------------------------------------------------------------------------------------------ version
def test_invalidacion_por_version_de_codigo(cache):
    cache.guardar("k", {"v": 1}, version="fv-1")
    assert cache.obtener("k", version="fv-1") == {"v": 1}
    assert cache.obtener("k", version="fv-2") is None          # otro codigo: se descarta
    assert cache.obtener("k", version="fv-1") is None          # y ya no esta
    cache.guardar("k2", 5, version="v1")
    assert cache.obtener("k2") == 5                            # sin version pedida: no filtra


def test_cambio_de_esquema_recrea_la_tabla(tmp_path):
    ruta = str(tmp_path / "e.sqlite")
    a = CacheLocal(ruta, esquema=1)
    a.guardar("k", 1)
    b = CacheLocal(ruta, esquema=2)
    assert b.obtener("k") is None and b.recreaciones == 1
    b.guardar("k", 2)
    assert CacheLocal(ruta, esquema=2).obtener("k") == 2


# ------------------------------------------------------------------------------------------ ticker y purga
def test_invalidacion_por_ticker_y_tipo(cache):
    cache.guardar("a1", 1, "entradas", "AAPL")
    cache.guardar("a2", 2, "serie_vj", "aapl")          # el ticker se normaliza a mayusculas
    cache.guardar("n1", 3, "entradas", "NVDA")
    assert cache.invalidar_ticker("AAPL") == 2
    assert cache.obtener("a1") is None and cache.obtener("a2") is None and cache.obtener("n1") == 3
    cache.guardar("n2", 4, "serie_vj", "NVDA")
    assert cache.purgar(ticker="NVDA", tipo="serie_vj") == 1 and cache.obtener("n1") == 3
    assert cache.purgar(clave="n1") == 1
    cache.guardar("z", 1)
    assert cache.purgar() == 0          # sin criterio no borra nada
    assert cache.purgar(todo=True) == 1 and cache.estadisticas()["entradas"] == 0


# ------------------------------------------------------------------------------------------ tamano acotado
def test_tamano_acotado_purga_las_mas_antiguas(tmp_path):
    c = CacheLocal(str(tmp_path / "t.sqlite"), max_entradas=5)
    for i in range(12):
        c.guardar(f"k{i}", i)
        time.sleep(0.002)
    st = c.estadisticas()
    assert st["entradas"] == 5
    assert [c.obtener(f"k{i}") for i in range(7, 12)] == [7, 8, 9, 10, 11] and c.obtener("k0") is None


def test_tamano_en_bytes_acotado(tmp_path):
    c = CacheLocal(str(tmp_path / "b.sqlite"), max_bytes=3000)
    for i in range(10):
        c.guardar(f"k{i}", "x" * 800)
        time.sleep(0.002)
    assert c.estadisticas()["bytes"] <= 3000 and c.obtener("k9") == "x" * 800
    c.guardar("enorme", "y" * 10000)      # una sola entrada mayor que el limite: se conserva (n > 1 es la condicion)
    assert c.obtener("enorme") == "y" * 10000 and c.estadisticas()["entradas"] == 1


def test_expiradas_se_purgan_antes_que_las_vigentes(tmp_path):
    c = CacheLocal(str(tmp_path / "x.sqlite"), max_entradas=3)
    c.guardar("vieja", 1, ttl=0.3)
    c.guardar("a", 1, ttl=3600)
    c.guardar("b", 2, ttl=3600)
    time.sleep(0.4)
    c.guardar("c", 3, ttl=3600)
    assert c.obtener("a") == 1 and c.obtener("b") == 2 and c.obtener("c") == 3


# ------------------------------------------------------------------------------------------ corrupcion
def test_bd_corrupta_se_recrea_y_conserva_copia(tmp_path):
    ruta = tmp_path / "corrupta.sqlite"
    ruta.write_bytes(b"esto no es una base sqlite " * 100)
    c = CacheLocal(str(ruta))
    assert c.obtener("k") is None                      # no lanza
    assert c.guardar("k", {"ok": 1}) is True and c.obtener("k") == {"ok": 1}
    assert c.recreaciones >= 1 and os.path.exists(str(ruta) + ".corrupta")


def test_bd_se_corrompe_con_la_instancia_abierta(cache):
    cache.guardar("k", 1)
    with open(cache.ruta, "r+b") as f:
        f.write(b"\x00" * 4096)                         # destruye la cabecera
    for suf in ("-wal", "-shm"):
        if os.path.exists(cache.ruta + suf):
            os.remove(cache.ruta + suf)
    assert cache.obtener("k") is None                  # se recrea sola
    assert cache.guardar("k", 2) is True and cache.obtener("k") == 2


def test_json_ilegible_en_una_fila_se_trata_como_ausente(cache):
    cache.guardar("k", 1)
    con = sqlite3.connect(cache.ruta)
    con.execute("UPDATE cache SET valor = '{no es json' WHERE clave = 'k'")
    con.commit()
    con.close()
    assert cache.obtener("k", defecto="d") == "d" and cache.obtener("k") is None
    assert cache.estadisticas()["entradas"] == 0       # y se elimino


def test_directorio_no_escribible_no_lanza(tmp_path):
    bloqueo = tmp_path / "archivo"
    bloqueo.write_text("x")
    c = CacheLocal(str(bloqueo / "sub" / "c.sqlite"))      # el "directorio" padre es un archivo
    assert c.guardar("k", 1) is False and c.obtener("k") is None and c.purgar(todo=True) == 0


# ------------------------------------------------------------------------------------------ concurrencia
def test_concurrencia_de_hilos_sin_perdidas_ni_excepciones(cache):
    errores = []

    def trabajo(n):
        try:
            for i in range(25):
                assert cache.guardar(f"h{n}-{i}", {"n": n, "i": i}, "entradas", f"T{n}")
                assert cache.obtener(f"h{n}-{i}") == {"n": n, "i": i}
        except Exception as ex:  # noqa: BLE001
            errores.append(ex)
    hilos = [threading.Thread(target=trabajo, args=(n,)) for n in range(8)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert errores == [] and cache.estadisticas()["entradas"] == 8 * 25
    assert cache.obtener("h3-7") == {"n": 3, "i": 7}


def test_lecturas_y_escrituras_concurrentes_sobre_la_misma_clave(cache):
    cache.guardar("k", 0)
    visto, errores = set(), []

    def escribe():
        for i in range(1, 60):
            cache.guardar("k", i)

    def lee():
        try:
            for _ in range(120):
                v = cache.obtener("k")
                assert v is None or isinstance(v, int)
                visto.add(v)
        except Exception as ex:  # noqa: BLE001
            errores.append(ex)
    hs = [threading.Thread(target=escribe)] + [threading.Thread(target=lee) for _ in range(3)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    assert errores == [] and cache.obtener("k") == 59 and len(visto) >= 1


def test_dos_instancias_misma_ruta_comparten_datos(tmp_path):
    ruta = str(tmp_path / "s.sqlite")
    a, b = CacheLocal(ruta), CacheLocal(ruta)
    a.guardar("k", 1)
    assert b.obtener("k") == 1
    b.guardar("k", 2)
    assert a.obtener("k") == 2


# ------------------------------------------------------------------------------------------ API de modulo y decorador
def test_api_de_modulo_usa_el_directorio_de_datos(tmp_path):
    assert cl.habilitado() and cl.guardar("m", {"x": 1}, "rf", "KO", version="v") is True
    assert cl.obtener("m", "v") == {"x": 1}
    assert os.path.exists(tmp_path / "datos" / cl.NOMBRE_ARCHIVO)
    assert cl.purgar(ticker="ko") == 1 and cl.obtener("m") is None


def test_api_de_modulo_sigue_los_cambios_de_directorio(monkeypatch, tmp_path):
    cl.guardar("k", 1)
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "otro"))
    assert cl.obtener("k") is None                       # otro directorio -> otra base
    cl.guardar("k", 2)
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "datos"))
    assert cl.obtener("k") == 1


def test_deshabilitado_en_la_nube(monkeypatch, tmp_path):
    monkeypatch.setenv("PORT", "8080")
    assert not cl.habilitado() and cl.cache_por_defecto() is None
    assert cl.guardar("k", 1) is False and cl.obtener("k", defecto="d") == "d" and cl.purgar(todo=True) == 0
    assert not (tmp_path / "datos").exists()              # nada se escribe en disco


def test_decorador_cacheado(monkeypatch):
    llamadas = []

    @cl.cacheado("entradas", clave=lambda t, x=1: f"f:{t}:{x}", ticker=lambda t, x=1: t, version="v1")
    def costosa(t, x=1):
        llamadas.append((t, x))
        return {"t": t, "x": x}
    assert costosa("A") == {"t": "A", "x": 1} and costosa("A") == {"t": "A", "x": 1} and len(llamadas) == 1
    assert costosa("A", x=2) == {"t": "A", "x": 2} and len(llamadas) == 2
    cl.purgar(ticker="A")
    costosa("A")
    assert len(llamadas) == 3
    monkeypatch.setenv("PORT", "1")                       # nube: no cachea, siempre llama
    costosa("A")
    costosa("A")
    assert len(llamadas) == 5


def test_decorador_no_guarda_none(monkeypatch):
    n = []

    @cl.cacheado("info", clave=lambda t: f"i:{t}")
    def f(t):
        n.append(1)
        return None
    f("Z")
    f("Z")
    assert len(n) == 2


def test_archivo_es_sqlite_valido_y_wal(cache):
    cache.guardar("k", 1)
    con = sqlite3.connect(cache.ruta)
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert con.execute("PRAGMA user_version").fetchone()[0] == cl.ESQUEMA
    assert json.loads(con.execute("SELECT valor FROM cache").fetchone()[0]) == 1
    con.close()
