"""Configuración del engine de backtesting.

Parámetros de infraestructura: cuenta, costos, riesgo, instrumento.
Los parámetros de la estrategia viven en strategies/<nombre>.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class BacktestConfig:
    """Parámetros del engine — independientes de la estrategia."""

    # ── Estrategia activa ──
    strategy: str = "flow_whale_detection"  # Nombre del archivo en strategies/ sin .py

    # ── Cuenta ──
    account_size: float = 1000_000.0
    risk_per_trade_pct: float = 0.1
    # Techo de contratos por restricción de broker.
    # El sizing real es dinámico: floor(equity × risk% / (stop_pts × point_value)),
    # capped a este valor.
    max_contracts: int = 100
    
    # Stop de referencia fijo para sizing (0.0 = usar stop dinámico de la señal).
    # Rompe la anti-correlación entre vol baja → muchos contratos → más stops.
    sizing_stop_pts: float = 40.0

    # ── Costos ──
    commission_rt: float = 4.06
    slippage_penalty_factor: float = 0.05

    # ── Límites de riesgo ──
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0

    # ── Sesiones RTH ──
    sessions: list[str] = field(
        default_factory=lambda: ["open_window"]
    )

    # ── Constantes del instrumento NQ ──
    tick_size: float = 0.25 
    tick_value: float = 5.0 
    point_value: float = 20.0 
    
    # # ── Constantes del instrumento ES ──
    # tick_size: float = 0.25
    # tick_value: float = 12.5
    # point_value: float = 50.0

    # ── Rutas ──
    data_path: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent
            / "data"
            / "NQ_0DTE_Master_v3.parquet"
            # / "ES_SPX_Master_v2.parquet"
        )
    )
    results_path: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent / "results"
        )
    )
