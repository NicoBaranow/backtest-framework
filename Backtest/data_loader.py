"""Carga y filtrado del parquet master."""

import numpy as np
import pandas as pd

from config import BacktestConfig


def load_data(config: BacktestConfig) -> pd.DataFrame:
    """Carga el dataset, filtra filas operables y marca límites de sesión."""
    # Lee el parquet “master” ya enriquecido (OHLC + features + metadata de sesión).
    # Se asume que `timestamp` viene como datetime64 y que existen columnas:
    # - session (label de sesión)
    # - excluded (bool: filas no operables / a ignorar)
    df: pd.DataFrame = pd.read_parquet(config.data_path)

    # Filtro base:
    # - Solo sesiones de interés (config.sessions)
    # - Quitar filas marcadas como “excluded” (data quality, holidays, outliers, etc.)
    mask = df["session"].isin(config.sessions) & ~df["excluded"]
    df = df.loc[mask].reset_index(drop=True)

    # Fecha de trading (date) para:
    # - agrupar PnL por día
    # - aplicar límites de riesgo diarios (max_daily_loss_pct)
    df["trade_date"] = df["timestamp"].dt.date

    # Session ID:
    # Convertimos (trade_date + session) en un entero contiguo para acelerar:
    # - detección de inicios/fines de sesión
    # - reseteos de estado en la state machine
    # Importante: se factoriza en el orden del dataframe ya filtrado.
    session_label = df["trade_date"].astype(str) + "_" + df["session"]
    df["session_id"] = session_label.factorize()[0].astype(np.int32)

    # Day ID: entero contiguo por fecha de trading.
    # Permite que el motor busque exits a lo largo de todo el día,
    # no solo dentro de una sesión (trades de open_window sobreviven hasta close_window).
    df["day_id"] = df["trade_date"].factorize()[0].astype(np.int32)

    return df
