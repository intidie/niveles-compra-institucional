"""Cache local en disco (SQLite) clave -> JSON, para ejecucion LOCAL (en la nube no se usa: visor ligero).

Diseno
------
* Una sola tabla ``cache(clave, tipo, ticker, valor, creado, expira, version, esquema, tam)``. ``valor`` es JSON
  (``json.dumps`` con conversion de numpy/pandas y NaN/inf -> null). Nada se guarda en pickle.
* Ubicacion: ``userdata.directorio_datos()`` (``NIVELES_DATA_DIR`` para tests) + ``cache_local.sqlite``.
* TTL por tipo (``TTL_POR_TIPO``): entradas XBRL/fundamentales 24 h, ^TNX 24 h, info de mercado 6 h, serie historica
  de valor justo 24 h. Un TTL explicito en ``guardar`` prevalece.
* Invalidacion: por ``version`` (la del codigo que genero el valor; si no coincide, la entrada se descarta), por
  ``esquema`` (version de la tabla; si cambia se recrea la BD) y por ticker (``purgar(ticker=...)``).
* Robustez: escritura atomica (una transaccion ``BEGIN IMMEDIATE`` por operacion, modo WAL); si la BD esta corrupta
  (archivo ilegible, ``DatabaseError``) se renombra a ``.corrupta`` y se recrea vacia; ninguna funcion lanza por fallos
  del cache (devuelven None/False): el cache es una optimizacion, nunca un requisito.
* Thread-safe: un ``RLock`` por instancia y conexion nueva por operacion (NiceGUI atiende desde varios hilos);
  ``timeout`` de SQLite para varios procesos.
* Tamano acotado: tras cada ``guardar`` se purgan primero las expiradas y luego las mas antiguas hasta cumplir
  ``max_bytes`` y ``max_entradas``.
* ``habilitado()`` es False en la nube (``PORT`` en el entorno): ``obtener`` devuelve None y ``guardar`` no escribe.
"""
from __future__ import annotations

import functools
import json
import math
import os
import sqlite3
import threading
import time
from typing import Any, Callable

from src.reporting import userdata

ESQUEMA = 1
NOMBRE_ARCHIVO = "cache_local.sqlite"
H = 3600.0
TTL_POR_TIPO: dict[str, float] = {
    "entradas": 24 * H,      # fundamentales XBRL / EntradasValoracion
    "rf": 24 * H,            # ^TNX
    "info": 6 * H,           # info de mercado
    "serie_vj": 24 * H,      # serie historica de valor justo
}
TTL_DEFECTO = 24 * H
MAX_BYTES_DEFECTO = 64 * 1024 * 1024
MAX_ENTRADAS_DEFECTO = 2000

_SQL_CREAR = """
CREATE TABLE IF NOT EXISTS cache (
    clave TEXT PRIMARY KEY, tipo TEXT NOT NULL, ticker TEXT, valor TEXT NOT NULL,
    creado REAL NOT NULL, expira REAL NOT NULL, version TEXT, esquema INTEGER NOT NULL, tam INTEGER NOT NULL)
"""


def habilitado() -> bool:
    """False en la nube (visor ligero: sin cache en disco)."""
    return not userdata.en_nube()


def _limpio(x: Any) -> Any:
    """Convierte a tipos JSON puros (numpy, pandas, NaN/inf -> None)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_limpio(v) for v in x]
    if hasattr(x, "isoformat"):
        return x.isoformat()
    if hasattr(x, "tolist"):          # arreglos y escalares de numpy / pandas
        try:
            return _limpio(x.tolist())
        except Exception:  # noqa: BLE001
            return None
    return str(x)


class CacheLocal:
    """Cache SQLite clave -> JSON. Ver el docstring del modulo."""

    def __init__(self, ruta: str | None = None, max_bytes: int = MAX_BYTES_DEFECTO,
                 max_entradas: int = MAX_ENTRADAS_DEFECTO, esquema: int = ESQUEMA):
        self.ruta = ruta or os.path.join(userdata.directorio_datos(), NOMBRE_ARCHIVO)
        self.max_bytes, self.max_entradas, self.esquema = int(max_bytes), int(max_entradas), int(esquema)
        self._lock = threading.RLock()
        self._listo = False
        self.recreaciones = 0      # cuantas veces se recreo la BD por corrupcion/esquema (para tests y diagnostico)

    # ---- conexion y recuperacion ------------------------------------------------------------------------------
    def _conectar(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.ruta) or ".", exist_ok=True)
        con = sqlite3.connect(self.ruta, timeout=10.0, isolation_level=None, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute(_SQL_CREAR)
            ver = con.execute("PRAGMA user_version").fetchone()[0]
            if ver not in (0, self.esquema):            # esquema distinto: se recrea la tabla
                con.execute("DROP TABLE cache")
                con.execute(_SQL_CREAR)
                self.recreaciones += 1
            con.execute(f"PRAGMA user_version={self.esquema}")
            con.execute("SELECT COUNT(*) FROM cache").fetchone()   # fuerza la lectura: detecta archivo corrupto
        except Exception:
            con.close()
            raise
        return con

    def _recrear(self) -> None:
        """Descarta una BD corrupta (la conserva como ``.corrupta`` si puede) y empieza de cero."""
        for suf in ("", "-wal", "-shm"):
            ruta = self.ruta + suf
            try:
                if suf == "" and os.path.exists(ruta):
                    try:
                        os.replace(ruta, self.ruta + ".corrupta")
                    except OSError:
                        os.remove(ruta)
                elif os.path.exists(ruta):
                    os.remove(ruta)
            except OSError:
                pass
        self.recreaciones += 1

    def _ejecutar(self, fn: Callable[[sqlite3.Connection], Any], defecto: Any = None) -> Any:
        """Ejecuta ``fn(con)`` con reintento tras recrear la BD si estaba corrupta. Nunca lanza."""
        with self._lock:
            for intento in (0, 1):
                con = None
                try:
                    con = self._conectar()
                    return fn(con)
                except sqlite3.DatabaseError as ex:
                    if con is not None:
                        con.close()
                        con = None
                    if "locked" in str(ex).lower() or "busy" in str(ex).lower():
                        return defecto
                    if intento == 0:
                        self._recrear()
                        continue
                    return defecto
                except (OSError, sqlite3.Error):
                    return defecto
                finally:
                    if con is not None:
                        try:
                            con.close()
                        except sqlite3.Error:
                            pass
            return defecto

    # ---- API ------------------------------------------------------------------------------------------------
    def guardar(self, clave: str, valor: Any, tipo: str = "generico", ticker: str | None = None,
                ttl: float | None = None, version: str | None = None) -> bool:
        """Guarda ``valor`` (JSON) bajo ``clave``. True si se escribio. Purga si se excede el tamano maximo."""
        try:
            txt = json.dumps(_limpio(valor), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return False
        ahora = time.time()
        ttl = TTL_POR_TIPO.get(tipo, TTL_DEFECTO) if ttl is None else float(ttl)
        tam = len(txt.encode("utf-8"))

        def op(con):
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?,?)",
                            (clave, tipo, (ticker or "").upper() or None, txt, ahora, ahora + ttl, version, self.esquema, tam))
                self._acotar(con, ahora)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            return True
        return bool(self._ejecutar(op, False))

    def _acotar(self, con: sqlite3.Connection, ahora: float) -> None:
        con.execute("DELETE FROM cache WHERE expira <= ?", (ahora,))
        tot, n = con.execute("SELECT COALESCE(SUM(tam),0), COUNT(*) FROM cache").fetchone()
        while (tot > self.max_bytes or n > self.max_entradas) and n > 1:
            fila = con.execute("SELECT clave, tam FROM cache ORDER BY creado ASC LIMIT 1").fetchone()
            if fila is None:
                break
            con.execute("DELETE FROM cache WHERE clave = ?", (fila[0],))
            tot, n = tot - fila[1], n - 1

    def obtener(self, clave: str, version: str | None = None, defecto: Any = None) -> Any:
        """Valor guardado, o ``defecto`` si no existe, expiro o es de otra ``version`` (se elimina en esos casos)."""
        ahora = time.time()

        def op(con):
            fila = con.execute("SELECT valor, expira, version FROM cache WHERE clave = ?", (clave,)).fetchone()
            if fila is None:
                return defecto
            valor, expira, ver = fila
            if expira <= ahora or (version is not None and ver != version):
                con.execute("DELETE FROM cache WHERE clave = ?", (clave,))
                return defecto
            return json.loads(valor)
        try:
            return self._ejecutar(op, defecto)
        except ValueError:      # JSON ilegible: se trata como ausente
            self.purgar(clave=clave)
            return defecto

    def purgar(self, ticker: str | None = None, tipo: str | None = None, clave: str | None = None,
               expirados: bool = False, todo: bool = False) -> int:
        """Elimina entradas. ``todo`` borra todo; si no, las que cumplan ``ticker``/``tipo``/``clave`` (AND) o,
        con ``expirados``, las vencidas. Devuelve cuantas se borraron."""
        def op(con):
            con.execute("BEGIN IMMEDIATE")
            try:
                if todo:
                    cur = con.execute("DELETE FROM cache")
                else:
                    cond, args = [], []
                    if ticker:
                        cond.append("ticker = ?")
                        args.append(ticker.upper())
                    if tipo:
                        cond.append("tipo = ?")
                        args.append(tipo)
                    if clave:
                        cond.append("clave = ?")
                        args.append(clave)
                    if expirados:
                        cond.append("expira <= ?")
                        args.append(time.time())
                    if not cond:
                        con.execute("COMMIT")
                        return 0
                    cur = con.execute("DELETE FROM cache WHERE " + " AND ".join(cond), args)
                n = cur.rowcount
                con.execute("COMMIT")
                return n
            except Exception:
                con.execute("ROLLBACK")
                raise
        return int(self._ejecutar(op, 0) or 0)

    def invalidar_ticker(self, ticker: str) -> int:
        return self.purgar(ticker=ticker)

    def estadisticas(self) -> dict:
        def op(con):
            n, tam = con.execute("SELECT COUNT(*), COALESCE(SUM(tam),0) FROM cache").fetchone()
            return {"entradas": n, "bytes": tam, "ruta": self.ruta}
        return self._ejecutar(op, {"entradas": 0, "bytes": 0, "ruta": self.ruta})


# ---- Instancia por defecto (una por ruta; la ruta se resuelve en cada llamada para respetar NIVELES_DATA_DIR) -------
_instancias: dict[str, CacheLocal] = {}
_lock_instancias = threading.Lock()


def cache_por_defecto() -> CacheLocal | None:
    """Instancia compartida del directorio de datos actual; None en la nube."""
    if not habilitado():
        return None
    ruta = os.path.join(userdata.directorio_datos(), NOMBRE_ARCHIVO)
    with _lock_instancias:
        c = _instancias.get(ruta)
        if c is None:
            c = _instancias[ruta] = CacheLocal(ruta)
        return c


def obtener(clave: str, version: str | None = None, defecto: Any = None) -> Any:
    c = cache_por_defecto()
    return defecto if c is None else c.obtener(clave, version, defecto)


def guardar(clave: str, valor: Any, tipo: str = "generico", ticker: str | None = None, ttl: float | None = None,
            version: str | None = None) -> bool:
    c = cache_por_defecto()
    return False if c is None else c.guardar(clave, valor, tipo, ticker, ttl, version)


def purgar(ticker: str | None = None, tipo: str | None = None, clave: str | None = None, expirados: bool = False,
           todo: bool = False) -> int:
    c = cache_por_defecto()
    return 0 if c is None else c.purgar(ticker, tipo, clave, expirados, todo)


def cacheado(tipo: str, clave: Callable[..., str], ticker: Callable[..., str | None] | None = None,
             ttl: float | None = None, version: str | None = None):
    """Decorador: memoiza en disco el resultado (JSON-serializable) de una funcion, solo en local.

    ``clave(*args, **kw)`` construye la clave; ``ticker(*args, **kw)`` (opcional) la asocia a un ticker para
    ``purgar(ticker=...)``. Un resultado None no se guarda. En la nube llama siempre a la funcion."""
    def deco(fn):
        @functools.wraps(fn)
        def envoltorio(*a, **kw):
            if not habilitado():
                return fn(*a, **kw)
            k = clave(*a, **kw)
            previo = obtener(k, version)
            if previo is not None:
                return previo
            val = fn(*a, **kw)
            if val is not None:
                guardar(k, val, tipo, ticker(*a, **kw) if ticker else None, ttl, version)
            return val
        envoltorio.__wrapped_cache__ = True
        return envoltorio
    return deco
