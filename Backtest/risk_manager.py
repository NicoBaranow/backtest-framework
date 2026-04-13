"""Cálculo de stop, target, sizing dinámico y límites de cuenta."""

from config import BacktestConfig


def compute_position_size(
    engine_config: BacktestConfig,
    stop_pts: float,
    current_equity: float,
) -> int:
    """Sizing dinámico: contratos basados en riesgo por trade.

    n = min(floor(equity × risk% / (stop_pts × point_value)), max_contracts)

    - equity: equity actual (se actualiza tras cada trade)
    - stop_pts: stop dinámico por barra de la estrategia
    - max_contracts: techo de broker en config.max_contracts
    """
    risk_amount: float = current_equity * engine_config.risk_per_trade_pct / 100.0
    risk_per_contract: float = stop_pts * engine_config.point_value
    if risk_per_contract <= 0:
        return 0
    dynamic_contracts: int = int(risk_amount / risk_per_contract)
    return min(max(dynamic_contracts, 0), engine_config.max_contracts)


def compute_stop_target(
    entry_price: float, direction: int, stop_pts: float, target_pts: float
) -> tuple[float, float]:
    """Devuelve (stop_price, target_price) según dirección."""
    if direction == 1:
        stop = entry_price - stop_pts
        target = entry_price + target_pts
    else:
        stop = entry_price + stop_pts
        target = entry_price - target_pts
    return stop, target


def check_daily_loss_limit(
    daily_pnl: float, config: BacktestConfig, current_equity: float
) -> bool:
    """True si se alcanzó el límite de pérdida diaria."""
    limit: float = current_equity * config.max_daily_loss_pct / 100.0
    return daily_pnl <= -limit


def check_max_drawdown(
    equity: float, peak_equity: float, config: BacktestConfig
) -> bool:
    """True si se alcanzó el drawdown máximo."""
    if peak_equity <= 0:
        return False
    dd_pct: float = (peak_equity - equity) / peak_equity * 100.0
    return dd_pct >= config.max_drawdown_pct
