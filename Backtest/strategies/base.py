"""Contrato base que toda estrategia debe cumplir."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignalData:
    """Output enriquecido de generate_entry_masks.

    stop_pts y target_pts son arrays float con valores por barra.
    Cuando son None, el engine usa los valores fijos de StrategyConfig.

    Estrategias simples (gamma_bounce) pueden devolver un tuple (long, short).
    Estrategias con exits dinámicos (EFD) devuelven SignalData.
    signal_generator maneja ambos casos.
    """

    entry_long: np.ndarray
    entry_short: np.ndarray
    stop_pts: np.ndarray | None = None
    target_pts: np.ndarray | None = None


@dataclass
class BaseStrategyConfig:
    """Parámetros que el engine necesita de cualquier estrategia.

    Cada estrategia hereda de esta clase y agrega sus propios campos.
    El engine accede a estos campos para stop/target, cooldown y filtro temporal.
    """

    target_pts: float = 0.0
    stop_pts: float = 0.0
    cooldown_seconds: int = 0
    base_slippage_ticks: int = 1
    max_trade_duration_bars: int = 0  # 0 = sin límite de tiempo

    # ── Ventana de señales (UTC, formato "HH:MM") ──
    # Solo se generan señales dentro de este rango horario.
    # "" = sin límite en ese extremo.
    # Ejemplos:
    #   gamma_bounce: ("", "19:15")      → señales hasta las 19:15 UTC
    #   GRM:          ("19:30", "20:30")  → solo última hora
    #   VPC:          ("18:00", "19:30")  → solo 14:00-15:30 ET
    signal_start_utc: str = ""
    signal_end_utc: str = ""

    # ── Sizing por régimen de gamma ──
    # Si sum_gex_oi < 0 (gamma negativa → feedback procíclico amplificado),
    # multiplicar n_contracts por este factor.
    # 1.0 = sin efecto (default). NQ WFD: 2.0
    regime_size_multiplier: float = 1.0
