"""Tracking de posición activa, equity curve y drawdown."""

from dataclasses import dataclass, field

from config import BacktestConfig


@dataclass
class Trade:
    """Registro inmutable de un trade ejecutado."""

    entry_idx: int
    exit_idx: int
    direction: int          # 1=long, -1=short
    entry_price: float
    exit_price: float
    entry_slippage: float
    exit_slippage: float
    contracts: int
    pnl_pts: float
    pnl_usd: float
    commission: float
    net_pnl: float
    exit_reason: str        # 'stop' | 'target' | 'eod'
    entry_time: str
    exit_time: str
    session: str
    setup: str              # 'long' | 'short'
    duration_seconds: int
    trade_date: str


@dataclass
class Portfolio:
    """Estado de la cuenta durante el backtest."""

    config: BacktestConfig
    equity: float = 0.0
    peak_equity: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.equity = self.config.account_size
        self.peak_equity = self.equity
        self.equity_curve.append(self.equity)

    def record_trade(self, trade: Trade) -> None:
        """Registra un trade y actualiza equity, peak y PnL diario."""
        self.trades.append(trade)
        self.equity += trade.net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

        # Acumular PnL diario
        date_key: str = trade.trade_date
        self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0.0) + trade.net_pnl

    def get_daily_pnl(self, date_key: str) -> float:
        """PnL acumulado del día."""
        return self.daily_pnl.get(date_key, 0.0)
