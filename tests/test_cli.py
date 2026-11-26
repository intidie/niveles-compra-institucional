import json

import pytest

import main as m


def test_modo_individual_defaults():
    a = m.parse_args(["--ticker", "nvda", "--capital", "1000"])
    assert a.ticker == "NVDA" and a.capital == 1000.0 and a.watchlist is None
    assert a.formato == "both" and a.top == 10


def test_modo_batch():
    a = m.parse_args(["--watchlist", "watchlist.csv", "--top", "5"])
    assert a.watchlist == "watchlist.csv" and a.top == 5 and a.ticker is None


def test_modos_mutuamente_excluyentes_y_requeridos(capsys):
    with pytest.raises(SystemExit):
        m.parse_args(["--ticker", "A", "--watchlist", "w.csv"])
    with pytest.raises(SystemExit):
        m.parse_args([])


@pytest.mark.parametrize("extra", [["--capital", "0"], ["--capital", "-5"], ["--top", "0"], ["--capital", "abc"]])
def test_valores_invalidos(extra):
    with pytest.raises(SystemExit):
        m.parse_args(["--ticker", "AAPL"] + extra)


def test_leer_watchlist(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("ticker,nota\nnvda,x\n# comentario\nAAPL,y\nNVDA,dup\n\n", encoding="utf-8")
    assert m.leer_watchlist(str(p)) == ["NVDA", "AAPL"]


def test_leer_watchlist_sin_cabecera_y_vacia(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("msft\nmeta\n", encoding="utf-8")
    assert m.leer_watchlist(str(p)) == ["MSFT", "META"]
    q = tmp_path / "v.csv"
    q.write_text("", encoding="utf-8")
    assert m.leer_watchlist(str(q)) == []


def test_watchlist_ejemplo_del_repo():
    t = m.leer_watchlist("watchlist.csv")
    assert "NVDA" in t and len(t) == len(set(t))


def test_asignar_capital():
    zs = [{"n_familias": 3, "r_sup": 100.0}, {"n_familias": 1, "r_sup": 50.0}]
    m.asignar_capital(zs, 1000)
    assert zs[0]["capital_asignado"] == pytest.approx(750)
    assert zs[1]["capital_asignado"] == pytest.approx(250)
    assert zs[0]["acciones"] == pytest.approx(7.5)
    assert m.asignar_capital([], 1000) == []


def test_ranking_orden():
    rs = [{"ticker": "A", "score_ranking": 40.0}, {"ticker": "B", "score_ranking": None},
          {"ticker": "C", "score_ranking": 80.0}]
    assert [r["ticker"] for r in m.ordenar_ranking(rs)] == ["C", "A", "B"]
    assert m.puntaje_ranking(80, 40) == 60
    assert m.puntaje_ranking(None, 40) == 40
    assert m.puntaje_ranking(None, None) is None


def _falso(ticker, capital, **kw):
    ok = ticker != "BAD"
    sc = {"AAA": 30.0, "BBB": 70.0}.get(ticker)
    return {"ticker": ticker, "fecha": "2026-01-01", "capital": capital, "df": None, "form4": None, "r13f": None,
            "precio_actual": 10.0 if ok else None, "errores": [] if ok else ["mercado: sin datos"],
            "zonas": [{"nivel": 1, "r_inf": 8.0, "r_sup": 9.0, "p_centro": 8.5, "dist_pct": -10.0, "n_familias": 2,
                       "familias": ["volumen", "ob"], "tipos": ["POC"], "capital_asignado": capital, "acciones": 1.0}]
            if ok else [],
            "confluencia_score": 50.0 if ok else None, "confiabilidad": {"indice": sc, "cobertura": 0.5},
            "score_ranking": sc}


def test_main_individual(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "analizar_ticker", _falso)
    rc = m.main(["--ticker", "AAA", "--capital", "500", "--sin-grafico", "--salida", str(tmp_path)])
    assert rc == 0
    d = json.loads((tmp_path / "AAA_niveles.json").read_text(encoding="utf-8"))
    assert d["resultados"][0]["ticker"] == "AAA"
    assert (tmp_path / "AAA_niveles.csv").exists()


def test_main_batch_top_y_fallidos(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "analizar_ticker", _falso)
    w = tmp_path / "w.csv"
    w.write_text("ticker\nAAA\nBBB\nBAD\n", encoding="utf-8")
    rc = m.main(["--watchlist", str(w), "--top", "1", "--salida", str(tmp_path / "o")])
    assert rc == 0
    d = json.loads((tmp_path / "o" / "ranking_watchlist.json").read_text(encoding="utf-8"))
    assert [r["ticker"] for r in d["resultados"]] == ["BBB"]
    assert "BAD" in capsys.readouterr().out


def test_main_watchlist_inexistente(tmp_path):
    assert m.main(["--watchlist", str(tmp_path / "no.csv")]) == 2


def test_analizar_ticker_sin_red_no_lanza(monkeypatch):
    import src.data.market_data as md

    def boom(*a, **k):
        raise ConnectionError("sin red")
    monkeypatch.setattr(md, "descargar_historial", boom)
    r = m.analizar_ticker("AAA", 100, verbose=False)
    assert r["precio_actual"] is None and r["errores"]
