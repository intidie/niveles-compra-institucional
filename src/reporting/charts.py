"""Grafico dual (Precio + Volume Profile lateral) en tema oscuro con marcadores Form 4 y 13F.

Degrada con gracia: sin datos de precio devuelve una figura con aviso; sin Form 4 / 13F omite los marcadores.
No fija backend de matplotlib (compatible con Colab inline y con Agg en tests/servidores).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

BG_FIG, BG_PANEL, GRID = "#0d1117", "#131722", "#232936"
TXT, MUTED = "#e6edf3", "#8b949e"
PALETA = ["#2ea043", "#1f883d", "#00b4d8", "#a371f7"]
COLS_FECHA = ("Date", "TransactionDate", "Transaction Date", "Transaction_Date", "date", "fecha", "Filing Date")
COLS_PRECIO = ("Price", "price", "precio")

# Referencia UNICA de distancia en todo el proyecto (consola, grafico, UI, export): r_sup, el precio del limite GTC.
ETIQUETA_DIST_RSUP = "dist. a r_sup"


def dist_a_r_sup_pct(r_sup: float, precio: float) -> float | None:
    """Distancia % de ``r_sup`` (precio de la orden limite GTC) al precio actual: ``(r_sup / precio - 1) * 100``.

    Negativo = la zona esta por debajo del precio. Es la UNICA formula de distancia que usan ``main.py``, el grafico
    y ``app_ui.py``; ``None`` si el precio no es valido.
    """
    try:
        r, p = float(r_sup), float(precio)
    except (TypeError, ValueError):
        return None
    return (r / p - 1.0) * 100.0 if np.isfinite(r) and np.isfinite(p) and p > 0 else None


def _col(df: pd.DataFrame, candidatas):
    return next((c for c in candidatas if c in df.columns), None)


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def perfil_histograma(df: pd.DataFrame, anios: int = 5, n_bins: int = 70):
    """(centros, volumen_suavizado, ancho_bin) del perfil lateral; vacios si no hay datos."""
    if df is None or df.empty:
        return np.array([]), np.array([]), 0.0
    d = df.copy()
    d.index = _naive(d.index)
    d = d[d.index >= d.index[-1] - pd.DateOffset(years=anios)]
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if not np.isfinite(lo) or hi <= lo:
        return np.array([]), np.array([]), 0.0
    paso = (hi - lo) / n_bins
    hist = np.zeros(n_bins)
    for l, h, v in zip(d["Low"].values, d["High"].values, d["Volume"].values):
        i0 = min(int((l - lo) // paso), n_bins - 1)
        i1 = min(int((h - lo) // paso), n_bins - 1)
        hist[i0:i1 + 1] += v / (i1 - i0 + 1)
    try:
        from scipy.ndimage import gaussian_filter1d
        hist = gaussian_filter1d(hist, 1.5)
    except Exception:  # noqa: BLE001
        pass
    return lo + paso * (np.arange(n_bins) + 0.5), hist, paso


def marcadores_form4(form4: pd.DataFrame | None, df: pd.DataFrame) -> pd.DataFrame:
    """Compras (codigo P) dentro del rango del grafico: columnas fecha, precio. Vacio si no hay datos usables."""
    vacio = pd.DataFrame(columns=["fecha", "precio"])
    if form4 is None or not isinstance(form4, pd.DataFrame) or form4.empty or df is None or df.empty:
        return vacio
    f = form4
    if "Code" in f.columns:
        f = f[f["Code"] == "P"]
    cf, cp = _col(f, COLS_FECHA), _col(f, COLS_PRECIO)
    if f.empty or cf is None:
        return vacio
    fechas = _naive(pd.to_datetime(f[cf], errors="coerce")).normalize()
    precios = pd.to_numeric(f[cp], errors="coerce").values if cp else np.full(len(f), np.nan)
    out = pd.DataFrame({"fecha": fechas, "precio": precios}).dropna(subset=["fecha"])
    idx = _naive(df.index).normalize()
    out = out[(out["fecha"] >= idx[0]) & (out["fecha"] <= idx[-1])].reset_index(drop=True)
    if out.empty:
        return vacio
    cierre = pd.Series(df["Close"].values, index=idx)
    cierre = cierre[~cierre.index.duplicated()]
    respaldo = cierre.reindex(pd.DatetimeIndex(out["fecha"]), method="nearest").values
    out["precio"] = np.where(out["precio"] > 0, out["precio"], respaldo)
    return out


def trimestres_acumulacion(r13f: pd.DataFrame | None, df: pd.DataFrame, solo_activos: bool = True) -> list[pd.Timestamp]:
    """Fines de trimestre 13F con compra neta (cambio_pct > 0 o posicion nueva) dentro del rango del grafico.

    ``solo_activos`` (def.) ignora las filas con ``pasivo`` verdadero (Vanguard / State Street: flujos de indice, no
    conviccion), igual que el conteo de gestores del screener.
    """
    if r13f is None or not isinstance(r13f, pd.DataFrame) or r13f.empty or "periodo" not in r13f.columns:
        return []
    if df is None or df.empty:
        return []
    if solo_activos and "pasivo" in r13f.columns:
        r13f = r13f[~r13f["pasivo"].fillna(False).astype(bool)]
        if r13f.empty:
            return []
    z = pd.Series(0.0, index=r13f.index)
    cambio = pd.to_numeric(r13f.get("cambio_pct", z), errors="coerce")
    previas = pd.to_numeric(r13f.get("acciones_previas", z), errors="coerce").fillna(0)
    acciones = pd.to_numeric(r13f.get("acciones", z), errors="coerce").fillna(0)
    acumula = (cambio > 0) | ((previas <= 0) & (acciones > 0))
    fechas = pd.to_datetime(r13f.loc[acumula, "periodo"], errors="coerce").dropna()
    idx = _naive(df.index)
    return sorted({pd.Timestamp(x) for x in fechas if idx[0] <= pd.Timestamp(x) <= idx[-1]})


def _figura_vacia(titulo: str, mensaje: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_PANEL)
    ax.text(0.5, 0.5, mensaje, color=MUTED, ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_title(titulo, color=TXT, loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _cerrar(fig, ruta, mostrar):
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=BG_FIG)
    if mostrar:
        plt.show()


def graficar_niveles(df: pd.DataFrame | None, ticker: str, zonas: list[dict] | None = None,
                     precio_actual: float | None = None, form4: pd.DataFrame | None = None,
                     r13f: pd.DataFrame | None = None, subtitulo: str | None = None,
                     ruta_guardado: str | None = None, mostrar: bool = False):
    """Devuelve la Figure. ``zonas``: dicts con r_inf, r_sup (y opcional fuente, nivel).

    La linea discontinua y la distancia de cada etiqueta usan ``r_sup`` (limite GTC), igual que consola y UI.
    """
    titulo = f"{ticker} - Mapa de niveles institucionales"
    if df is None or df.empty or not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        fig = _figura_vacia(titulo, "Sin datos de mercado disponibles (red o ticker invalido)")
        _cerrar(fig, ruta_guardado, mostrar)
        return fig

    d = df.copy()
    d.index = _naive(d.index)
    zonas = zonas or []
    px = float(precio_actual if precio_actual is not None else d["Close"].iloc[-1])

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(16, 9), sharey=True,
                                     gridspec_kw={"width_ratios": [4.3, 1.0], "wspace": 0.025})
    fig.patch.set_facecolor(BG_FIG)
    for ax in (ax_p, ax_v):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, color=GRID, alpha=0.7)
        for s in ax.spines.values():
            s.set_color(GRID)

    ax_p.plot(d.index, d["Close"], color="#58a6ff", linewidth=1.4, label="Precio cierre (ajustado)", zorder=4)
    ax_p.fill_between(d.index, d["Low"], d["High"], color="#58a6ff", alpha=0.10, zorder=2)
    if len(d) >= 200:
        ax_p.plot(d.index, d["Close"].rolling(200).mean(), color=MUTED, linestyle=":", linewidth=1.1,
                  label="SMA 200D", zorder=3)

    # Trimestres 13F de acumulacion
    q13 = trimestres_acumulacion(r13f, d)
    for i, q in enumerate(q13):
        ax_p.axvspan(q - pd.DateOffset(months=3), q, color="#a371f7", alpha=0.10, zorder=1,
                     label="Trimestre de acumulacion 13F (gestores activos)" if i == 0 else None)
        y = float(d["Close"].reindex([q], method="nearest").iloc[0])
        ax_p.scatter([q], [y], marker="D", s=60, color="#a371f7", edgecolor="white", linewidth=0.6, zorder=8,
                     label="Cierre trimestre 13F (acumula)" if i == 0 else None)

    # Compras Form 4 (codigo P)
    f4 = marcadores_form4(form4, d)
    if not f4.empty:
        ax_p.scatter(f4["fecha"], f4["precio"], marker="^", s=70, color="#3fb950", edgecolor="white",
                     linewidth=0.6, zorder=9, label=f"Compra insider Form 4 (P) x{len(f4)}")

    # Zonas de compra
    x_lbl = d.index[max(0, int(len(d) * 0.015))]
    for i, z in enumerate(zonas):
        c = PALETA[i % len(PALETA)]
        lo, hi = float(z["r_inf"]), float(z["r_sup"])
        ax_p.axhspan(lo, hi, color=c, alpha=0.16, zorder=1)
        # La linea discontinua marca r_sup (el limite GTC) y la distancia se mide siempre hasta ella.
        ax_p.axhline(hi, color=c, linestyle="--", linewidth=1.2, alpha=0.9, zorder=3,
                     label="Limite GTC en r_sup (techo de cada zona)" if i == 0 else None)
        ax_v.axhline(hi, color=c, linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)
        fuente = str(z.get("fuente", ""))
        fuente = fuente if len(fuente) <= 48 else fuente[:45] + "..."
        dist = dist_a_r_sup_pct(hi, px)
        txt_dist = "n/d" if dist is None else f"{dist:+.1f}%"
        # '\$' literal: dos '$' sin escapar en el mismo texto los interpreta matplotlib como formula (mathtext).
        ax_p.text(x_lbl, hi, f" N{z.get('nivel', i + 1)}: \\${lo:.2f}-\\${hi:.2f} | {ETIQUETA_DIST_RSUP} {txt_dist} | "
                             f"{fuente} ",
                  color="#ffffff", fontsize=8.1, fontweight="bold", va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.22", facecolor=BG_FIG, edgecolor=c, alpha=0.88), zorder=7)
    ax_p.axhline(px, color="#f0f6fc", linewidth=1.2, alpha=0.85, label=f"Cotizacion actual (${px:.2f})", zorder=5)

    # Volume Profile lateral
    centros, vol, paso = perfil_histograma(d)
    if len(centros):
        cols_vp = ["#f59e0b" if v >= 0.7 * vol.max() else "#30363d" for v in vol]
        ax_v.barh(centros, vol, height=paso * 0.88, color=cols_vp, edgecolor="none", alpha=0.85, zorder=2)
    ax_v.xaxis.set_visible(False)
    ax_v.set_xlabel("Volume Profile 5A\n(naranja = nodos de alto volumen)", color=MUTED, fontsize=8.5)

    precios = [float(d["Low"].min()), float(d["High"].max()), px] + [v for z in zonas for v in (z["r_inf"], z["r_sup"])]
    ax_p.set_ylim(min(precios) * 0.95, max(precios) * 1.04)
    ax_p.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax_p.set_ylabel("Precio (USD)", color=TXT, fontsize=10, fontweight="bold")
    ax_p.set_title(titulo + (f"\n{subtitulo}" if subtitulo else ""), color=TXT, fontsize=10.6,
                   fontweight="bold", pad=14, loc="left")
    leg = ax_p.legend(loc="upper left", facecolor=BG_FIG, edgecolor=GRID, labelcolor=TXT, fontsize=8.5,
                      framealpha=0.9)
    leg.set_zorder(10)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.07)
    fig.text(0.06, 0.012, f"{ETIQUETA_DIST_RSUP}: distancia % del limite GTC (r_sup) al precio actual; negativo = por debajo.",
             color=MUTED, fontsize=8, ha="left", va="bottom")
    _cerrar(fig, ruta_guardado, mostrar)
    return fig
