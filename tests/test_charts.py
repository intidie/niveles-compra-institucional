import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.reporting import charts


@pytest.fixture(autouse=True)
def _cerrar():
    yield
    plt.close("all")


def _form4(df):
    return pd.DataFrame({"Code": ["P", "S", "P"], "Date": [df.index[100], df.index[120], df.index[300]],
                         "Price": [np.nan, 50.0, 110.0], "Shares": [10, 5, 7], "Value": [1000, 250, 770]})


def _r13f(df):
    return pd.DataFrame({"gestor": ["A", "B", "C"],
                         "periodo": [df.index[200].strftime("%Y-%m-%d")] * 2 + ["1999-12-31"],
                         "acciones": [10.0, 5.0, 3.0], "acciones_previas": [5.0, 8.0, 0.0],
                         "cambio_pct": [100.0, -37.5, None]})


def test_grafico_completo_con_marcadores(df_sintetico, zonas, tmp_path):
    ruta = tmp_path / "g.png"
    fig = charts.graficar_niveles(df_sintetico, "TEST", zonas, form4=_form4(df_sintetico),
                                  r13f=_r13f(df_sintetico), ruta_guardado=str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 5000
    assert len(fig.axes) == 2
    etiquetas = fig.axes[0].get_legend_handles_labels()[1]
    assert any("Form 4" in e and "x2" in e for e in etiquetas)  # solo codigo P
    assert any("13F" in e for e in etiquetas)


def test_marcadores_form4_filtra_p_y_rellena_precio(df_sintetico):
    f4 = charts.marcadores_form4(_form4(df_sintetico), df_sintetico)
    assert len(f4) == 2
    assert f4["precio"].notna().all() and (f4["precio"] > 0).all()


def test_trimestres_13f_solo_acumulacion_en_rango(df_sintetico):
    q = charts.trimestres_acumulacion(_r13f(df_sintetico), df_sintetico)
    assert len(q) == 1 and q[0] == df_sintetico.index[200]


def test_sin_datos_degrada(zonas, tmp_path):
    for df in (None, pd.DataFrame()):
        fig = charts.graficar_niveles(df, "X", zonas, ruta_guardado=str(tmp_path / "v.png"))
        assert fig is not None
        plt.close(fig)
    assert (tmp_path / "v.png").exists()


def test_sin_form4_ni_13f_ni_zonas(df_sintetico):
    fig = charts.graficar_niveles(df_sintetico, "X")
    assert len(fig.axes) == 2
    for f4, r13 in ((pd.DataFrame(), pd.DataFrame()), (pd.DataFrame({"Code": ["P"]}), pd.DataFrame({"x": [1]}))):
        charts.graficar_niveles(df_sintetico, "X", [], form4=f4, r13f=r13)


def test_indice_con_zona(df_sintetico):
    d = df_sintetico.copy()
    d.index = d.index.tz_localize("America/New_York")
    fig = charts.graficar_niveles(d, "TZ", [{"r_inf": 70, "r_sup": 75}], form4=_form4(df_sintetico))
    assert fig is not None


def test_perfil_histograma(df_sintetico):
    c, v, paso = charts.perfil_histograma(df_sintetico)
    assert len(c) == len(v) == 70 and paso > 0 and v.sum() > 0
    assert len(charts.perfil_histograma(pd.DataFrame())[0]) == 0
