"""Orquestador: sparse loop vectorizado sobre índices de señal.

Conecta signal_generator, execution_model, risk_manager y portfolio
para simular path dependency (un trade activo bloquea nuevas señales).
"""

import importlib
from functools import partial

import numpy as np
import pandas as pd

from config import BacktestConfig
from signal_generator import generate_signals
from execution_model import (
    compute_entry_price,
    compute_exit_price,
    compute_commission,
)
from risk_manager import (
    compute_position_size,
    compute_stop_target,
    check_daily_loss_limit,
    check_max_drawdown,
)
from portfolio import Portfolio, Trade


def _precompute_session_ends(session_ids: np.ndarray) -> np.ndarray:
    """Último índice de cada sesión contigua, ordenado ascendentemente."""
    changes = np.where(np.diff(session_ids) != 0)[0]
    return np.append(changes, len(session_ids) - 1)


def _precompute_session_starts(session_ids: np.ndarray) -> np.ndarray:
    """Primer índice de cada sesión contigua, ordenado ascendentemente."""
    changes = np.where(np.diff(session_ids) != 0)[0] + 1
    return np.insert(changes, 0, 0)


def _find_session_end(idx: int, session_end_indices: np.ndarray) -> int:
    """Índice de fin de sesión más cercano >= idx."""
    pos: int = int(np.searchsorted(session_end_indices, idx, side="left"))
    if pos < len(session_end_indices):
        return int(session_end_indices[pos])
    return int(session_end_indices[-1])


def _find_session_start(idx: int, session_start_indices: np.ndarray) -> int:
    """Índice de inicio de sesión correspondiente a idx."""
    pos: int = int(np.searchsorted(session_start_indices, idx, side="right")) - 1
    return int(session_start_indices[max(pos, 0)])


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> Portfolio:
    """Ejecuta el backtest completo con sparse loop sobre señales."""

    # ── 0. Cargar estrategia dinámica ──
    strategy_mod = importlib.import_module(f"strategies.{config.strategy}")
    strat_cfg = strategy_mod.StrategyConfig()
    masks_fn = partial(strategy_mod.generate_entry_masks, config=strat_cfg)

    # ── 1. Generar señales vectorizadas ──
    entry_mask, directions, stop_pts_arr, target_pts_arr = generate_signals(
        df, strat_cfg, masks_fn
    )

    # ── 2. Extraer arrays de NumPy ──
    opens: np.ndarray = df["open"].values
    highs: np.ndarray = df["high"].values
    lows: np.ndarray = df["low"].values
    closes: np.ndarray = df["close"].values
    timestamps: np.ndarray = df["timestamp"].values
    sessions: np.ndarray = df["session"].values
    session_ids: np.ndarray = df["session_id"].values
    # sum_gex_oi[t-1]: régimen de gamma causal para sizing dinámico.
    # None si la columna no existe en el dataset.
    sum_gex_oi_prev: np.ndarray | None = (
        df["sum_gex_oi"].shift(1).values if "sum_gex_oi" in df.columns else None
    )
    day_ids: np.ndarray = df["day_id"].values
    trade_dates: np.ndarray = df["trade_date"].values

    # ── 3. Precomputar límites ──
    # Sesión (per date+session): para slippage LOB (no mezclar volatilidad entre sesiones).
    session_start_indices: np.ndarray = _precompute_session_starts(session_ids)
    # Día (per date): para búsqueda de exit (trades cruzan open_window → close_window).
    day_end_indices: np.ndarray = _precompute_session_ends(day_ids)

    # ── 4. Índices de señal ──
    signal_indices: np.ndarray = np.flatnonzero(entry_mask)

    # ── 5. Inicializar portfolio ──
    portfolio = Portfolio(config=config)

    # ── 6. Variables de estado de riesgo ──
    current_equity: float = config.account_size
    peak_equity: float = current_equity
    current_day = None            # fecha del trade anterior (date object)
    daily_pnl: float = 0.0
    halt_day = None               # día suspendido por daily loss (date object)
    next_available_idx: int = 0

    # ── 7. Sparse loop ──
    for idx in signal_indices:
        if idx < next_available_idx:
            continue  # Bloqueado por posición activa

        direction: int = int(directions[idx])
        if direction == 0:
            continue

        signal_day = trade_dates[idx]  # datetime.date object

        # ── Pre-trade: ¿día suspendido por daily loss? ──
        if halt_day is not None and signal_day == halt_day:
            continue

        # ── Límites: día (exit search) y sesión (slippage) ──
        day_end: int = _find_session_end(idx, day_end_indices)
        session_start: int = _find_session_start(idx, session_start_indices)

        # ── Stop/target: dinámico por señal o fijo de config ──
        if stop_pts_arr is not None:
            signal_stop: float = float(stop_pts_arr[idx])
        else:
            signal_stop = strat_cfg.stop_pts
        if target_pts_arr is not None:
            signal_target: float = float(target_pts_arr[idx])
        else:
            signal_target = strat_cfg.target_pts

        # ── Position sizing (desacoplado del stop de ejecución) ──
        sizing_stop: float = config.sizing_stop_pts if config.sizing_stop_pts > 0 else signal_stop
        n_contracts: int = compute_position_size(config, sizing_stop, current_equity)
        if n_contracts <= 0:
            continue

        # ── Régimen de gamma: escalar contratos en gamma negativa ──
        if (
            sum_gex_oi_prev is not None
            and strat_cfg.regime_size_multiplier != 1.0
            and sum_gex_oi_prev[idx] < 0
        ):
            n_contracts = min(
                int(n_contracts * strat_cfg.regime_size_multiplier),
                config.max_contracts,
            )

        # ── Entry ──
        entry_price, entry_slip = compute_entry_price(
            opens, highs, lows, idx, session_start, direction, config, strat_cfg
        )

        # ── Stop y target ──
        stop_price, target_price = compute_stop_target(
            entry_price, direction, signal_stop, signal_target
        )

        # ── Buscar salida vectorialmente (a lo largo de todo el día) ──
        search_end: int = day_end + 1

        if direction == 1:
            stop_hits = np.where(lows[idx:search_end] <= stop_price)[0]
            target_hits = np.where(highs[idx:search_end] >= target_price)[0]
        else:
            stop_hits = np.where(highs[idx:search_end] >= stop_price)[0]
            target_hits = np.where(lows[idx:search_end] <= target_price)[0]

        stop_idx: int | None = (
            idx + int(stop_hits[0]) if len(stop_hits) > 0 else None
        )
        target_idx: int | None = (
            idx + int(target_hits[0]) if len(target_hits) > 0 else None
        )

        # ── Determinar salida: elegir el primer candidato cronológicamente ──
        # stop tiene prioridad sobre target/time si coinciden en la misma barra.
        candidates: list[tuple[int, int, str, float]] = []
        if stop_idx is not None:
            candidates.append((stop_idx, 0, "stop", float(stop_price)))
        if target_idx is not None:
            candidates.append((target_idx, 1, "target", float(target_price)))
        if strat_cfg.max_trade_duration_bars > 0:
            time_limit: int = min(idx + strat_cfg.max_trade_duration_bars, day_end)
            candidates.append((time_limit, 2, "time", float(closes[time_limit])))
        candidates.append((day_end, 3, "eod", float(closes[day_end])))
        candidates.sort()
        exit_idx, _, exit_reason, raw_exit_price = candidates[0]

        # ── Exit price con slippage (mismo session_start, LOB contiguo) ──
        exit_price, exit_slip = compute_exit_price(
            raw_exit_price, highs, lows, exit_idx, session_start, direction, config, strat_cfg
        )

        # ── PnL ──
        commission: float = compute_commission(config, n_contracts)
        if direction == 1:
            pnl_pts: float = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        pnl_usd: float = pnl_pts * config.point_value * n_contracts
        net_pnl: float = pnl_usd - commission

        # ── Duración ──
        entry_ts = pd.Timestamp(timestamps[idx])
        exit_ts = pd.Timestamp(timestamps[exit_idx])
        duration: int = int((exit_ts - entry_ts).total_seconds())

        trade_date_str: str = str(signal_day)

        # ── Registrar trade ──
        trade = Trade(
            entry_idx=idx,
            exit_idx=exit_idx,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_slippage=entry_slip,
            exit_slippage=exit_slip,
            contracts=n_contracts,
            pnl_pts=pnl_pts,
            pnl_usd=pnl_usd,
            commission=commission,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            entry_time=str(entry_ts),
            exit_time=str(exit_ts),
            session=str(sessions[idx]),
            setup="long" if direction == 1 else "short",
            duration_seconds=duration,
            trade_date=trade_date_str,
        )
        portfolio.record_trade(trade)
        next_available_idx = max(exit_idx + 1, idx + strat_cfg.cooldown_seconds)

        # ═══════════════════════════════
        # VALIDACIÓN DE RIESGO POST-TRADE
        # ═══════════════════════════════

        # a) Actualizar equity y peak
        current_equity += net_pnl
        peak_equity = max(peak_equity, current_equity)

        # b) Reset daily_pnl si cambió de día calendario
        if current_day is not None and signal_day != current_day:
            daily_pnl = 0.0
        current_day = signal_day

        # c) Acumular PnL diario
        daily_pnl += net_pnl

        # d) Max drawdown → halt total del backtest
        if check_max_drawdown(current_equity, peak_equity, config):
            break

        # e) Daily loss → suspender el día actual
        if check_daily_loss_limit(daily_pnl, config, current_equity):
            halt_day = signal_day

    return portfolio
