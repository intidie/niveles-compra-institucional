"""Regresion: series XBRL con ``period_end`` duplicado (KO: el mismo trimestre aparece con inicio 2010-07-02 y
2010-07-03 en filings distintos; 10-K/10-Q; reexpresiones) rompian ``reindex`` con
``ValueError: cannot reindex on an axis with duplicate labels`` y se perdian los pisos de valoracion.

Todo con datos sinteticos, sin red."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.levels import valuation_floors as vf

OCF = "us-gaap:NetCashProvidedByUsedInOperatingActivities"
CAPEX = "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
SH = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
CASH = "us-gaap:CashAndCashEquivalentsAtCarryingValue"
DEBT = "us-gaap:LongTermDebt"

RESTATED_SH = 1.10e9
ORIG_SH = 1.00e9


def _mk(rows, con_filed=True) -> pd.DataFrame:
    """Replica el esquema que produce ``vf._facts`` (dur, ref_filing y, opcionalmente, filed)."""
    cols = ["concept", "period_type", "period_start", "period_end", "numeric_value", "fiscal_year",
            "fiscal_period", "filed"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("period_start", "period_end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["dur"] = (df["period_end"] - df["period_start"]).dt.days
    df["ref_filing"] = df.groupby(["fiscal_year", "fiscal_period"])["period_end"].transform("max")
    return df if con_filed else df.drop(columns="filed")


def _trimestres(desde=2017, hasta=2024):
    fin = pd.date_range(f"{desde}-03-31", f"{hasta}-12-31", freq="QE")
    return [(e - pd.Timedelta(days=89), e) for e in fin]


def _facts_sinteticos(con_filed=True) -> pd.DataFrame:
    rows = []
    for ini, fin in _trimestres():
        fy, fp = fin.year, f"Q{(fin.month - 1) // 3 + 1}"
        filed = fin + pd.Timedelta(days=30)
        rows.append([OCF, "duration", ini, fin, 2.0e9, fy, fp, filed])
        rows.append([CAPEX, "duration", ini, fin, 0.5e8, fy, fp, filed])
        rows.append([SH, "duration", ini, fin, ORIG_SH, fy, fp, filed])
        # Duplicado del mismo trimestre en la presentacion del anio siguiente (comparativo), inicio +1 dia.
        rows.append([SH, "duration", ini + pd.Timedelta(days=1), fin, ORIG_SH, fy + 1, fp,
                     filed + pd.Timedelta(days=365)])
        # Duplicado exacto (mismo inicio y fin) de OCF reportado en 10-K y 10-Q.
        rows.append([OCF, "duration", ini, fin, 2.0e9, fy + 1, fp, filed + pd.Timedelta(days=365)])
        rows.append([CASH, "instant", pd.NaT, fin, 5.0e9, fy, fp, filed])
        rows.append([CASH, "instant", pd.NaT, fin, 5.0e9, fy + 1, fp, filed + pd.Timedelta(days=365)])
        rows.append([DEBT, "instant", pd.NaT, fin, 1.0e10, fy, fp, filed])
        rows.append([DEBT, "instant", pd.NaT, fin, 1.0e10, fy + 1, fp, filed + pd.Timedelta(days=365)])
    # Reexpresion del ultimo trimestre de acciones con valor DISTINTO, presentada despues y con inicio +1 dia.
    ini, fin = _trimestres()[-1]
    rows.append([SH, "duration", ini + pd.Timedelta(days=1), fin, RESTATED_SH, 2025, "Q4",
                 pd.Timestamp("2026-06-01")])  # posterior a todo comparativo
    return _mk(rows, con_filed)


def _precios() -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-02", "2025-01-31")
    rng = np.random.default_rng(0)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    return pd.DataFrame({"Close": close}, index=idx)


TK = SimpleNamespace(splits=None)


# --- Helpers ------------------------------------------------------------------------------------------
def test_dedup_periodo_gana_la_ultima_presentada_aunque_este_antes_en_el_orden():
    f = _mk([
        [SH, "duration", "2024-07-01", "2024-09-30", 1.10e9, 2024, "Q3", "2025-06-01"],   # reexpresion (vigente)
        [SH, "duration", "2024-07-01", "2024-09-30", 1.00e9, 2024, "Q3", "2024-10-30"],   # original
        [SH, "duration", "2024-07-02", "2024-09-30", 1.00e9, 2025, "Q3", "2025-01-15"],   # inicio +1 dia
    ])
    u = vf._dedup_periodo(f, por_inicio=False)
    assert len(u) == 1 and u["numeric_value"].iloc[0] == 1.10e9
    s = vf._serie_concepto(f, SH, "duration")
    assert len(s) == 1 and s["numeric_value"].iloc[0] == 1.10e9  # inicios +-1 dia colapsan en un periodo


def test_dedup_sin_filed_conserva_la_ultima_fila():
    f = _mk([
        [SH, "duration", "2024-07-01", "2024-09-30", 1.00e9, 2024, "Q3", None],
        [SH, "duration", "2024-07-02", "2024-09-30", 1.20e9, 2024, "Q3", None],
    ], con_filed=False)
    assert "filed" not in f.columns
    assert vf._serie_concepto(f, SH, "duration")["numeric_value"].tolist() == [1.20e9]


def test_serie_concepto_no_fusiona_ytd_con_trimestre_y_respeta_as_of():
    f = _mk([
        [OCF, "duration", "2024-01-01", "2024-03-31", 1.0e9, 2024, "Q1", "2024-05-01"],
        [OCF, "duration", "2024-01-01", "2024-06-30", 2.5e9, 2024, "Q2", "2024-08-01"],   # YTD 6M
        [OCF, "duration", "2024-04-01", "2024-06-30", 1.5e9, 2024, "Q2", "2024-08-01"],   # trimestre aislado
        [OCF, "duration", "2024-04-01", "2024-06-30", 1.6e9, 2025, "Q2", "2025-08-01"],   # reexpresion posterior
    ])
    s = vf._serie_concepto(f, OCF, "duration")
    assert len(s) == 3  # Q1, 6M YTD y Q2 aislado siguen siendo periodos distintos
    q = vf._flujo_trimestral(f, OCF)
    assert q[pd.Timestamp("2024-06-30")] == 1.6e9  # vigente
    q_pit = vf._flujo_trimestral(f[f["filed"] <= "2024-12-31"], OCF)
    assert q_pit[pd.Timestamp("2024-06-30")] == 1.5e9
    s_pit = vf._serie_concepto(f, OCF, "duration", as_of="2024-12-31")  # sin look-ahead
    assert s_pit[s_pit["dur"] <= 110]["numeric_value"].tolist() == [1.0e9, 1.5e9]


def test_flujo_trimestral_ytd_con_inicio_variable_un_dia():
    """Q2 = 6M - Q1 aunque el 6M venga con inicio distinto en un dia (calendario 52/53 semanas)."""
    f = _mk([
        [OCF, "duration", "2010-01-01", "2010-04-02", 1.0e9, 2010, "Q1", "2010-05-01"],
        [OCF, "duration", "2010-01-02", "2010-07-02", 2.6e9, 2010, "Q2", "2010-08-01"],
    ])
    q = vf._flujo_trimestral(f, OCF)
    assert q[pd.Timestamp("2010-07-02")] == pytest.approx(1.6e9)


def test_puntual_y_ttm_diario_toleran_indice_duplicado():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2023-06-30", "2023-09-30", "2023-09-30"]))
    out = vf._puntual_diario(s, idx)
    assert out.iloc[-1] == 3.0 and out.index.is_unique
    q = pd.Series(np.arange(1.0, 7.0), index=pd.to_datetime(
        ["2023-03-31", "2023-06-30", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]))
    assert vf._ttm_diario(q, idx).index.is_unique


# --- Integracion sobre el modulo -----------------------------------------------------------------------
@pytest.mark.parametrize("con_filed", [True, False])
def test_ev_fcf_con_period_end_duplicados_no_lanza_y_usa_reexpresion(con_filed):
    facts = _facts_sinteticos(con_filed)
    assert facts.duplicated(["concept", "period_end"]).any()  # la fixture realmente trae duplicados
    r = vf.ev_fcf_historico("SYN", TK, _precios(), facts)
    assert r is not None
    assert r["acciones"] == RESTATED_SH  # gana la reexpresion presentada despues (o la ultima sin 'filed')
    assert r["fcf_ttm"] == pytest.approx(4 * (2.0e9 - 0.5e8))
    assert all(np.isfinite(v) for v in r["pisos"].values())


def test_calcular_pisos_valoracion_completo_con_duplicados(monkeypatch):
    eps = [["us-gaap:EarningsPerShareDiluted", "duration", ini, fin, 0.4, fin.year, f"Q{(fin.month - 1) // 3 + 1}",
            fin + pd.Timedelta(days=30)] for ini, fin in _trimestres()]
    eps += [["us-gaap:EarningsPerShareDiluted", "duration", ini + pd.Timedelta(days=1), fin, 0.4, fin.year + 1,
             f"Q{(fin.month - 1) // 3 + 1}", fin + pd.Timedelta(days=395)] for ini, fin in _trimestres()]
    monkeypatch.setattr(vf, "_facts", lambda t: pd.concat([_facts_sinteticos(), _mk(eps)], ignore_index=True))
    r = vf.calcular_pisos_valoracion("SYN", TK, _precios(), {})
    tipos = {a["tipo"] for a in r["anclas"]}
    assert {"PE_P25", "EVFCF_P25", "LOG_-2.0SD"} <= tipos
    assert r["ev_fcf"] is not None and r["pe"] is not None
