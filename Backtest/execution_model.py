"""Modelo de costos: slippage dinámico y comisiones."""

import numpy as np

from config import BacktestConfig
from strategies.base import BaseStrategyConfig


def compute_slippage(
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    session_start: int,
    engine_config: BacktestConfig,
    strategy_config: BaseStrategyConfig,
) -> float:
    """Slippage dinámico basado en el rolling range LOB de los últimos 5 bars.

    - lb se clampea a session_start para no cruzar fronteras de sesión.
    - El rango usa max(High) - min(Low) sobre [lb, idx) (causal, excluye idx).
    - Si lb == idx (primeras barras de sesión), el rango es 0.
    - base_slippage_ticks viene de la estrategia (EFD usa 3, gamma_bounce usa 1).
    """
    lb: int = max(session_start, idx - 5)
    if lb < idx:
        local_range: float = float(highs[lb:idx].max() - lows[lb:idx].min())
    else:
        local_range = 0.0
    return (
        strategy_config.base_slippage_ticks
        + local_range * engine_config.slippage_penalty_factor
    ) * engine_config.tick_size


def compute_entry_price(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    session_start: int,
    direction: int,
    engine_config: BacktestConfig,
    strategy_config: BaseStrategyConfig,
) -> tuple[float, float]:
    """Precio de entrada con slippage adverso.

    Devuelve (entry_price, slippage_aplicado).
    """
    slippage: float = compute_slippage(
        highs, lows, idx, session_start, engine_config, strategy_config
    )
    if direction == 1:
        entry_price = float(opens[idx]) + slippage
    else:
        entry_price = float(opens[idx]) - slippage
    return entry_price, slippage


def compute_exit_price(
    raw_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    exit_idx: int,
    session_start: int,
    direction: int,
    engine_config: BacktestConfig,
    strategy_config: BaseStrategyConfig,
) -> tuple[float, float]:
    """Precio de salida con slippage adverso.

    Devuelve (exit_price, slippage_aplicado).
    """
    slippage: float = compute_slippage(
        highs, lows, exit_idx, session_start, engine_config, strategy_config
    )
    if direction == 1:
        exit_price = raw_price - slippage
    else:
        exit_price = raw_price + slippage
    return exit_price, slippage


def compute_commission(config: BacktestConfig, n_contracts: int) -> float:
    """Comisión round-trip total."""
    return config.commission_rt * n_contracts