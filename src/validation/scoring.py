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
    try:
        return json.load(open(RUTA_PESOS, encoding="utf-8"))["pesos"]
    except Exception:  # noqa: BLE001
        return dict(PESOS_PRIOR)


# --- Normalizacion de componentes a 0-100 ---------------------------------------------------------------
def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def s_13f(n_acumulando: int | None, n_con_posicion: int | None, compras_form4_usd: float | None = None):
    """50 % fraccion de gestores con posicion que aumentaron + 50 % compras Form 4 (saturan en USD 1 M)."""
    if n_con_posicion is None and compras_form4_usd is None:
        return None
    frac = (n_acumulando or 0) / n_con_posicion if n_con_posicion else 0.0
    f4 = _clip01((compras_form4_usd or 0.0) / 1_000_000)
    return 100 * (0.5 * frac + 0.5 * f4)


def s_stock_act(pts: float | None):
    return None if pts is None else 100 * _clip01(pts / 15.0)


def s_absorcion(tasa: float | None):
    return None if tasa is None or np.isnan(tasa) else 100 * _clip01(tasa)


def s_backtest(bounce, ret60_mediana, mae60_mediana):
    """50 % bounce, 30 % retorno mediano 60d (satura en +15 %), 20 % control de MAE (0 % = 1, -25 % = 0)."""
    if bounce is None or np.isnan(bounce):
        return None
    r = 0.0 if ret60_mediana is None or np.isnan(ret60_mediana) else _clip01(ret60_mediana / 0.15)
    m = 0.5 if mae60_mediana is None or np.isnan(mae60_mediana) else 1 - _clip01(abs(min(mae60_mediana, 0)) / 0.25)
    return 100 * (0.5 * _clip01(bounce) + 0.3 * r + 0.2 * m)


def s_confluencia(n_familias: int | None, n_valoracion: int = 0):
    if n_familias is None:
        return None
    return 100 * _clip01(0.6 * (n_familias - 1) / 3 + 0.4 * min(1, n_valoracion / 2))


def calcular_indice_confiabilidad(comp: dict, pesos: dict | None = None) -> dict:
    """``comp``: {componente: valor 0-100 o None}. Los None se excluyen y los pesos se renormalizan."""
    pesos = pesos or cargar_pesos()
    disp = {k: v for k, v in comp.items() if v is not None and k in pesos}
    if not disp:
        return {"indice": None, "cobertura": 0.0, "detalle": {}}
    wsum = sum(pesos[k] for k in disp)
    idx = sum(pesos[k] * v for k, v in disp.items()) / wsum
    return {"indice": float(idx), "cobertura": float(wsum / sum(pesos.values())),
            "detalle": {k: {"valor": float(v), "peso_efectivo": float(pesos[k] / wsum)} for k, v in disp.items()}}


def indice_desde_modulos(backtest_resumen: dict, respaldo13f: dict | None = None, stock_act: dict | None = None,
                         absorcion: float | None = None, n_familias: int | None = None, n_valoracion: int = 0,
                         pesos: dict | None = None) -> dict:
    """Atajo para el screener: usa las salidas de ``backtest_ranges.resumen``, ``sec_edgar`` y ``stock_act``."""
    r13 = respaldo13f or {}
    comp = {"respaldo_13f_form4": s_13f(r13.get("n_acumulando"), r13.get("n_con_posicion"), r13.get("compras_form4_usd")),
            "stock_act": s_stock_act(None if stock_act is None else stock_act.get("pts")),
            "absorcion": s_absorcion(absorcion if absorcion is not None else backtest_resumen.get("absorcion")),
            "backtest": s_backtest(backtest_resumen.get("bounce"), backtest_resumen.get("ret60_mediana"),
                                   backtest_resumen.get("mae60_mediana")),
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
        casos_ag = agregado(json.load(open(ruta_c, encoding="utf-8")))
    ev = evidencia(ordenes, casos_ag)
    pesos = calibrar_pesos(ev)
    print("=== CALIBRACION DE PESOS DEL INDICE DE CONFIABILIDAD ===")
    print(f"{'componente':22}{'z/t':>8}{'a-priori':>10}{'calibrado':>11}  evidencia")
    for k in COMPONENTES:
        z = ev[k]["z"]
        print(f"{k:22}{'n/d' if z is None else f'{z:+.2f}':>8}{PESOS_PRIOR[k]:>10.1%}{pesos[k]:>11.1%}  {ev[k]['detalle']}")
    json.dump({"pesos": pesos, "prior": PESOS_PRIOR, "mezcla_empirica": MEZCLA_EMPIRICA,
               "evidencia": {k: {"z": v["z"], "n": v["n"], "detalle": v["detalle"]} for k, v in ev.items()}},
              open(RUTA_PESOS, "w", encoding="utf-8"), indent=1)
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
