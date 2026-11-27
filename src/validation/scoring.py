"""Indice de confiabilidad (0-100 %) de un rango de compra, con pesos calibrados empiricamente.

Componentes (cada uno normalizado a 0-100)
------------------------------------------
* respaldo_13f_form4 : gestores 13F que acumulan / con posicion + compras de insiders (Form 4, codigo P).
* stock_act          : ``pts`` (0-15) de ``consultar_flujo_congreso_stock_act`` reescalado a 0-100.
* absorcion          : tasa de absorcion (velas de rechazo con volumen > 1.35x SMA20, cierre en tercio superior).
* backtest           : bounce rate, retorno mediano a 60 ruedas y MAE mediano a 60 ruedas del backtest walk-forward.
* confluencia        : familias de ancla en el cluster (volumen/AVWAP/OB/swing/log-reg) + anclas de valoracion.

Calibracion
-----------
``calibrar_pesos`` parte de a-priori (``PESOS_PRIOR``) y los mezcla 50/50 con pesos empiricos
w ~ prior * exp(0.5 * clip(z, -3, 3)), donde z es el estadistico t (o z) del efecto de cada componente sobre el
retorno a 60 ruedas / coincidencia 13F (``backtest_ranges`` y ``case_studies``). Evidencia negativa reduce el
peso relativo y positiva lo aumenta. Componentes sin muestra (STOCK Act) usan z = 0 (conservan su a-priori). Resultado en ``pesos_calibrados.json``.
Ejecutar:  python -m src.validation.scoring   (requiere haber corrido backtest_ranges y case_studies).

Datos no disponibles != 0
-------------------------
Un dato que NO se pudo obtener (consulta fallida, sin respuesta) es ``None`` y se EXCLUYE: el componente (o la mitad
del componente) desaparece del promedio y los pesos restantes se renormalizan, de modo que la ``cobertura`` baja en
lugar de inflarse con ceros o con valores neutros. Un dato obtenido que vale cero (0 USD de compras de insiders,
0 gestores con posicion, 0 puntos STOCK Act, 0 anclas de valoracion en la zona) SI es dato y puntua cero. La
valoracion no consultada (``n_valoracion=None``) excluye solo la mitad de valoracion de ``s_confluencia`` y reescala
la de familias. Excepcion residual documentada en
``s_backtest``: si hay bounce pero faltan los medianos de retorno o MAE a 60 ruedas, se usan 0 (retorno) y 0.5 (MAE neutro).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_PESOS = os.path.join(AQUI, "pesos_calibrados.json")
CACHE = os.path.join(AQUI, "_cache")

COMPONENTES = ["respaldo_13f_form4", "stock_act", "absorcion", "backtest", "confluencia"]
PESOS_PRIOR = {"respaldo_13f_form4": 0.25, "stock_act": 0.10, "absorcion": 0.20, "backtest": 0.30,
               "confluencia": 0.15}
MEZCLA_EMPIRICA = 0.5


def cargar_pesos() -> dict:
    """Pesos calibrados de ``pesos_calibrados.json``; si el fichero no existe o es ilegible, los a-priori."""
    try:
        with open(RUTA_PESOS, encoding="utf-8") as f:
            return json.load(f)["pesos"]
    except Exception:  # noqa: BLE001
        return dict(PESOS_PRIOR)


# --- Normalizacion de componentes a 0-100 ---------------------------------------------------------------
def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _hay_dato(x) -> bool:
    """True si ``x`` es un dato utilizable: no es None ni NaN (cero SI es un dato valido)."""
    try:
        return not bool(pd.isna(x))
    except (TypeError, ValueError):  # no escalar: se deja pasar y la normalizacion posterior falla de forma visible
        return True


def s_13f(n_acumulando: int | None, n_con_posicion: int | None, compras_form4_usd: float | None = None):
    """Componente 13F/Form 4 (0-100): promedio 50/50 de las mitades DISPONIBLES.

    * Mitad gestores: fraccion de gestores ACTIVOS con posicion que aumentaron (``n_acumulando / n_con_posicion``,
      acotada a [0, 1]). Disponible si ``n_con_posicion`` no es None; ``n_con_posicion == 0`` es el dato valido
      "0 gestores con posicion" (fraccion 0). Si ``n_con_posicion > 0`` pero ``n_acumulando`` es None no se puede
      calcular la fraccion y la mitad se trata como no disponible.
    * Mitad Form 4: compras de insiders (codigo P) en USD, saturadas en 1 M USD. Disponible si
      ``compras_form4_usd`` no es None (0.0 = "consulta correcta, sin compras" = dato valido).

    Una mitad no disponible (None/NaN) se EXCLUYE y el score es el promedio de las disponibles, reponderado: con una
    sola disponible esta vale el 100 % del componente (no el 50 %). Sin ninguna mitad disponible devuelve None
    (el componente se excluye del indice y los pesos se renormalizan).
    """
    mitades = []
    if _hay_dato(n_con_posicion):
        if n_con_posicion <= 0:
            mitades.append(0.0)
        elif _hay_dato(n_acumulando):
            mitades.append(_clip01(n_acumulando / n_con_posicion))
    if _hay_dato(compras_form4_usd):
        mitades.append(_clip01(compras_form4_usd / 1_000_000))
    if not mitades:
        return None
    return 100 * sum(mitades) / len(mitades)


def s_stock_act(pts: float | None):
    """``pts`` (0-15) de STOCK Act a 0-100. None/NaN (consulta fallida) -> None: el componente se excluye."""
    return 100 * _clip01(pts / 15.0) if _hay_dato(pts) else None


def s_absorcion(tasa: float | None):
    return None if tasa is None or np.isnan(tasa) else 100 * _clip01(tasa)


def s_backtest(bounce, ret60_mediana, mae60_mediana):
    """50 % bounce, 30 % retorno mediano 60d (satura en +15 %), 20 % control de MAE (0 % = 1, -25 % = 0).

    Sin bounce (None/NaN) el componente se excluye (None). Excepcion residual: con bounce pero sin retorno mediano
    se usa 0 para esa parte, y sin MAE mediano se usa 0.5 (neutro)."""
    if bounce is None or np.isnan(bounce):
        return None
    r = 0.0 if ret60_mediana is None or np.isnan(ret60_mediana) else _clip01(ret60_mediana / 0.15)
    m = 0.5 if mae60_mediana is None or np.isnan(mae60_mediana) else 1 - _clip01(abs(min(mae60_mediana, 0)) / 0.25)
    return 100 * (0.5 * _clip01(bounce) + 0.3 * r + 0.2 * m)


def s_confluencia(n_familias: int | None, n_valoracion: int | None = 0):
    """Confluencia (0-100): 60 % familias de ancla de la zona + 40 % anclas de valoracion en la zona.

    Con valoracion consultada (``n_valoracion`` entero, incluido 0):
        ``100 * clip01(0.6 * (n_familias - 1) / 3 + 0.4 * min(1, n_valoracion / 2))``
    (1 familia aporta 0; 4 familias dan el maximo de la parte de familias; 2 o mas anclas de valoracion dan el maximo de
    la parte de valoracion). ``n_valoracion = 0`` es un dato valido: "se consulto y no hay anclas en la zona".

    Con ``n_valoracion`` None/NaN (valoracion NO consultada, p. ej. fallo de ``calcular_pisos_valoracion``) la mitad
    de valoracion se EXCLUYE y la parte de familias se reescala a 0-100 (se divide por su peso 0.6):
        ``100 * clip01((n_familias - 1) / 3)``      (4 o mas familias -> 100; 1 familia -> 0)
    ``n_familias`` None/NaN -> None (sin zona no hay confluencia).
    """
    if not _hay_dato(n_familias):
        return None
    familias = (n_familias - 1) / 3
    if not _hay_dato(n_valoracion):
        return 100 * _clip01(familias)
    return 100 * _clip01(0.6 * familias + 0.4 * min(1, n_valoracion / 2))


def calcular_indice_confiabilidad(comp: dict, pesos: dict | None = None) -> dict:
    """Indice de confiabilidad 0-100 como promedio ponderado de los componentes CON dato.

    ``comp``: {componente: valor 0-100 o None}. Los None/NaN (dato no disponible) y las claves sin peso se excluyen;
    los pesos de los componentes restantes se renormalizan para sumar 1. Un valor 0 es un dato y SI entra.

    Devuelve un dict con:
    * ``indice``: float 0-100, o None si ningun componente tiene dato.
    * ``cobertura``: float en [0, 1] = (suma de los pesos de los componentes con dato) / (suma de todos los pesos).
      1.0 = los 5 componentes presentes; baja cuando falta alguno (sin STOCK Act: 1 - peso de stock_act). Mide
      cuanta parte del modelo tuvo dato, no la calidad de esos datos. Con cobertura baja el indice es el promedio de lo
      poco que hay y no es comparable con el de otro ticker con cobertura distinta.
    * ``detalle``: {componente_con_dato: {"valor", "peso_efectivo"}}; ``peso_efectivo`` = peso / suma de pesos con
      dato (suma 1 sobre los presentes).
    Si ninguno tiene dato: ``{"indice": None, "cobertura": 0.0, "detalle": {}}``.
    """
    pesos = pesos or cargar_pesos()
    disp = {k: v for k, v in comp.items() if _hay_dato(v) and k in pesos}
    wsum = sum(pesos[k] for k in disp)
    if not disp or wsum <= 0:
        return {"indice": None, "cobertura": 0.0, "detalle": {}}
    idx = sum(pesos[k] * v for k, v in disp.items()) / wsum
    return {"indice": float(idx), "cobertura": float(wsum / sum(pesos.values())),
            "detalle": {k: {"valor": float(v), "peso_efectivo": float(pesos[k] / wsum)} for k, v in disp.items()}}


def indice_desde_modulos(backtest_resumen: dict | None, respaldo13f: dict | None = None, stock_act: dict | None = None,
                         absorcion: float | None = None, n_familias: int | None = None,
                         n_valoracion: int | None = 0, pesos: dict | None = None) -> dict:
    """Atajo para el screener: usa las salidas de ``backtest_ranges.resumen``, ``sec_edgar`` y ``stock_act``.

    Un dato no disponible EXCLUYE su componente (y baja la ``cobertura``); no se sustituye por 0 ni por neutro:
    * ``respaldo13f`` (``n_acumulando``, ``n_con_posicion``, ``compras_form4_usd``): cada clave puede ser None
      (13F sin respuesta / Form 4 fallido); ver ``s_13f``. ``respaldo13f=None`` equivale a todas en None.
    * ``stock_act``: dict de ``consultar_flujo_congreso_stock_act``. Con ``pts=None`` o ``disponible=False``
      (consulta fallida) el componente se excluye. ``stock_act=None`` (no consultado) tambien lo excluye.
    * ``backtest_resumen`` None o vacio: se excluyen backtest y absorcion (salvo que se pase ``absorcion``).
    * ``n_valoracion``: se pasa tal cual a ``s_confluencia``. None = valoracion no consultada (se excluye la mitad de
      valoracion de la confluencia y se reescala la de familias; el componente sigue presente, la cobertura no cambia);
      0 (por defecto) = consultada sin anclas en la zona (dato valido que puntua 0).
    """
    r13 = respaldo13f or {}
    bt = backtest_resumen or {}
    sa = stock_act or {}
    pts = None if sa.get("disponible") is False else sa.get("pts")
    comp = {"respaldo_13f_form4": s_13f(r13.get("n_acumulando"), r13.get("n_con_posicion"), r13.get("compras_form4_usd")),
            "stock_act": s_stock_act(pts),
            "absorcion": s_absorcion(absorcion if absorcion is not None else bt.get("absorcion")),
            "backtest": s_backtest(bt.get("bounce"), bt.get("ret60_mediana"), bt.get("mae60_mediana")),
            "confluencia": s_confluencia(n_familias, n_valoracion)}
    return calcular_indice_confiabilidad(comp, pesos)


# --- Calibracion empirica ---------------------------------------------------------------------------------
def _t_dos_grupos(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return 0.0
    return float(stats.ttest_ind(a, b, equal_var=False).statistic)


def _t_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    m = x.notna() & y.notna()
    if m.sum() < 10 or x[m].nunique() < 2:
        return 0.0, 0.0
    rho = stats.spearmanr(x[m], y[m]).statistic
    t = rho * np.sqrt((m.sum() - 2) / max(1e-12, 1 - rho ** 2))
    return float(rho), float(t)


def _historial_pit(llen: pd.DataFrame) -> pd.Series:
    """Bounce rate previo del ticker usando solo fills ya resueltos (fecha_fill + 30 dias <= fecha_fill actual)."""
    out = pd.Series(np.nan, index=llen.index)
    for _, g in llen.groupby("ticker"):
        g = g.sort_values("fecha_fill")
        for i, row in g.iterrows():
            prev = g[(g["fecha_fill"] + pd.Timedelta(days=30) <= row["fecha_fill"]) & g["bounce"].notna()]
            if len(prev) >= 5:
                out[i] = prev["bounce"].astype(float).mean()
    return out


def evidencia(ordenes: pd.DataFrame, casos_agregado: dict | None) -> dict:
    llen = ordenes[ordenes["llenada"].astype(bool) & ordenes["ret60"].notna()].copy()
    llen["absorcion"] = llen["absorcion"].astype(bool)
    llen["bounce"] = llen["bounce"].astype(float)
    ev = {}
    ab = llen[llen["absorcion"]]
    no = llen[~llen["absorcion"]]
    ev["absorcion"] = {"z": _t_dos_grupos(ab["ret60"], no["ret60"]), "n": len(llen),
                       "detalle": f"ret60 media con absorcion {ab['ret60'].mean():+.3%} (n={len(ab)}) vs sin {no['ret60'].mean():+.3%} "
                                  f"(n={len(no)}); bounce {ab['bounce'].mean():.1%} vs {no['bounce'].mean():.1%}"}
    prior = _historial_pit(llen)
    rho, t = _t_spearman(prior, llen["bounce"])
    ev["backtest"] = {"z": t, "n": int(prior.notna().sum()),
                      "detalle": f"Spearman(bounce previo PIT del ticker, bounce actual) = {rho:+.3f} (n={int(prior.notna().sum())})"}
    rho, t = _t_spearman(llen["n_familias"], llen["ret60"])
    tl = _t_dos_grupos(llen.loc[llen["has_logreg"].astype(bool), "ret60"], llen.loc[~llen["has_logreg"].astype(bool), "ret60"])
    ev["confluencia"] = {"z": max(t, tl), "n": len(llen),
                         "detalle": f"Spearman(n_familias, ret60) = {rho:+.3f} (t={t:+.2f}); t(has_logreg, ret60) = {tl:+.2f}"}
    if casos_agregado and casos_agregado.get("n_trimestres"):
        n, obs, ctrl = casos_agregado["n_trimestres"], casos_agregado["solapa_rodante"], casos_agregado["p_control_rodante"]
        z = (obs - ctrl) / np.sqrt(max(1e-9, ctrl * (1 - ctrl) / n))
        ev["respaldo_13f_form4"] = {"z": float(z), "n": n,
                                    "detalle": f"solape rodante con zona 13F {obs:.1%} vs control aleatorio {ctrl:.1%} (n={n}); z={z:+.2f}"}
    else:
        ev["respaldo_13f_form4"] = {"z": None, "n": 0, "detalle": "sin casos 13F disponibles: se mantiene a-priori"}
    ev["stock_act"] = {"z": None, "n": 0, "detalle": "sin serie historica PIT en este repo: se mantiene a-priori"}
    return ev


def calibrar_pesos(ev: dict) -> dict:
    """w_emp ~ prior * exp(0.5 * clip(z, -3, 3)) (z de cada componente; sin muestra z = 0), renormalizado;
    peso final = 50 % a-priori + 50 % empirico."""
    emp = {k: PESOS_PRIOR[k] * (1.0 if ev[k]["z"] is None else float(np.exp(0.5 * np.clip(ev[k]["z"], -3, 3))))
           for k in COMPONENTES}
    se = sum(emp.values())
    pesos = {k: (1 - MEZCLA_EMPIRICA) * PESOS_PRIOR[k] + MEZCLA_EMPIRICA * emp[k] / se for k in COMPONENTES}
    s = sum(pesos.values())
    return {k: v / s for k, v in pesos.items()}


def tabla_calibracion(ordenes: pd.DataFrame, pesos: dict) -> pd.DataFrame:
    """Confiabilidad del modelo: score PIT (absorcion + historial PIT del ticker + confluencia) por tercil
    contra bounce, retorno 60d y MAE60 realizados."""
    llen = ordenes[ordenes["llenada"].astype(bool) & ordenes["ret60"].notna()].copy()
    llen["bounce"] = llen["bounce"].astype(float)
    prior = _historial_pit(llen)
    sc = []
    for i, r in llen.iterrows():
        comp = {"absorcion": 100.0 if r["absorcion"] else 0.0,
                "backtest": None if np.isnan(prior[i]) else 100 * prior[i],
                "confluencia": s_confluencia(int(r["n_familias"]), int(bool(r["has_logreg"])))}
        sc.append(calcular_indice_confiabilidad(comp, pesos)["indice"])
    llen["score"] = sc
    llen = llen.dropna(subset=["score", "bounce"])
    llen["tercil"] = pd.qcut(llen["score"], 3, labels=["bajo", "medio", "alto"], duplicates="drop")
    tab = llen.groupby("tercil", observed=True).agg(n=("score", "size"), score_medio=("score", "mean"),
                                                   bounce=("bounce", "mean"), ret60_media=("ret60", "mean"),
                                                   ret60_mediana=("ret60", "median"), mae60_mediana=("mae60", "median"))
    rho = stats.spearmanr(llen["score"], llen["bounce"]).statistic
    rho60 = stats.spearmanr(llen["score"], llen["ret60"]).statistic
    tab.attrs["spearman_bounce"], tab.attrs["spearman_ret60"], tab.attrs["n"] = float(rho), float(rho60), len(llen)
    return tab


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ruta_o = os.path.join(CACHE, "ordenes_todas.pkl")
    if not os.path.exists(ruta_o):
        print("Falta _cache/ordenes_todas.pkl: ejecute primero  python -m src.validation.backtest_ranges")
        return
    ordenes = pd.read_pickle(ruta_o)
    casos_ag = None
    ruta_c = os.path.join(CACHE, "casos_resultados.json")
    if os.path.exists(ruta_c):
        from src.validation.case_studies import agregado
        with open(ruta_c, encoding="utf-8") as f:
            casos_ag = agregado(json.load(f))
    ev = evidencia(ordenes, casos_ag)
    pesos = calibrar_pesos(ev)
    print("=== CALIBRACION DE PESOS DEL INDICE DE CONFIABILIDAD ===")
    print(f"{'componente':22}{'z/t':>8}{'a-priori':>10}{'calibrado':>11}  evidencia")
    for k in COMPONENTES:
        z = ev[k]["z"]
        print(f"{k:22}{'n/d' if z is None else f'{z:+.2f}':>8}{PESOS_PRIOR[k]:>10.1%}{pesos[k]:>11.1%}  {ev[k]['detalle']}")
    with open(RUTA_PESOS, "w", encoding="utf-8") as f:
        json.dump({"pesos": pesos, "prior": PESOS_PRIOR, "mezcla_empirica": MEZCLA_EMPIRICA,
                   "evidencia": {k: {"z": v["z"], "n": v["n"], "detalle": v["detalle"]} for k, v in ev.items()}},
                  f, indent=1)
    tab = tabla_calibracion(ordenes, pesos)
    print(f"\n=== CALIBRACION DEL MODELO: score PIT por tercil vs resultados realizados (n={tab.attrs['n']}) ===")
    t = tab.copy()
    for c in ("bounce", "ret60_media", "ret60_mediana", "mae60_mediana"):
        t[c] = t[c].map(lambda x: f"{x * 100:6.1f}%")
    t["score_medio"] = t["score_medio"].map(lambda x: f"{x:5.1f}")
    print(t.to_string())
    print(f"Spearman(score, bounce)={tab.attrs['spearman_bounce']:+.3f}  Spearman(score, ret60)={tab.attrs['spearman_ret60']:+.3f}")
    print(f"Pesos guardados en {RUTA_PESOS}")
    ejemplo = calcular_indice_confiabilidad({"respaldo_13f_form4": 50.0, "stock_act": 50.0, "absorcion": 26.0,
                                             "backtest": 60.0, "confluencia": 50.0}, pesos)
    print(f"Ejemplo de uso (componentes ilustrativos, NO datos reales): indice={ejemplo['indice']:.1f}")


if __name__ == "__main__":
    main()
