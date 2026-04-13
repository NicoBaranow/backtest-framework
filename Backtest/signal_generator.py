"""Extracción de señales unificadas — infraestructura compartida.

Recibe las máscaras crudas de la estrategia activa, las combina
en un entry_mask + directions unificado, y aplica el filtro temporal.
Cada estrategia puede saltearse escribir esto porque este módulo lo hace.
"""

from __future__ import annotations

import datetime
from typing import Callable

import numpy as np
import pandas as pd

from strategies.base import BaseStrategyConfig, SignalData


def generate_signals(
    df: pd.DataFrame,
    strategy_config: BaseStrategyConfig,
    generate_masks: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray] | SignalData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Combina máscaras long/short en señales unificadas.

    Args:
        df: DataFrame con datos OHLC + features.
        strategy_config: Config de la estrategia (para ventana temporal).
        generate_masks: Callable que devuelve (entry_long, entry_short)
                        o SignalData con stops/targets dinámicos.

    Devuelve:
        entry_mask   – bool array, True donde hay señal de entrada.
        directions   – int array, 1=long, -1=short.
        stop_pts     – float array por barra (o None si la estrategia usa fijos).
        target_pts   – float array por barra (o None si la estrategia usa fijos).
    """
    result = generate_masks(df)

    if isinstance(result, SignalData):
        entry_long = result.entry_long
        entry_short = result.entry_short
        stop_pts_arr = result.stop_pts
        target_pts_arr = result.target_pts
    else:
        entry_long, entry_short = result
        stop_pts_arr = None
        target_pts_arr = None

    entry_mask: np.ndarray = entry_long | entry_short
    directions: np.ndarray = np.where(
        entry_long, 1, np.where(entry_short, -1, 0)
    )

    # ── Filtro temporal: solo señales dentro de la ventana UTC ──
    utc_time = df["timestamp"].dt.time
    if strategy_config.signal_start_utc:
        h, m = map(int, strategy_config.signal_start_utc.split(":"))
        too_early: np.ndarray = (utc_time < datetime.time(h, m)).values
        entry_mask = entry_mask & ~too_early
    if strategy_config.signal_end_utc:
        h, m = map(int, strategy_config.signal_end_utc.split(":"))
        too_late: np.ndarray = (utc_time >= datetime.time(h, m)).values
        entry_mask = entry_mask & ~too_late

    return entry_mask, directions, stop_pts_arr, target_pts_arr
