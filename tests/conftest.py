import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def df_sintetico():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=600)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(idx))))
    return pd.DataFrame({"Open": close * 0.998, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float)}, index=idx)


@pytest.fixture
def zonas():
    return [{"nivel": 1, "r_inf": 90.0, "r_sup": 95.0, "n_familias": 3, "familias": ["volumen", "avwap", "ob"],
             "fuente": "POC+AVWAP+OB"},
            {"nivel": 2, "r_inf": 80.0, "r_sup": 84.0, "n_familias": 2, "familias": ["swing", "logreg"]}]


@pytest.fixture(autouse=True)
def _datos_usuario_aislados(tmp_path, monkeypatch):
    """Nunca tocar el directorio de datos real del usuario (watchlist personalizada) durante los tests."""
    monkeypatch.setenv("NIVELES_DATA_DIR", str(tmp_path / "datos_usuario"))
