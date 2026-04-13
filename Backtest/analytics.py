"""Métricas de performance, reportes y exportación de resultados."""

import numpy as np
import pandas as pd

from config import BacktestConfig
from portfolio import Portfolio


def compute_analytics(portfolio: Portfolio, config: BacktestConfig) -> dict:
    """Calcula y muestra todas las métricas del backtest.

    Exporta equity curve y trade log a config.results_path.
    """
    trades = portfolio.trades
    if not trades:
        print("No se ejecutaron trades.")
        return {}

    df = pd.DataFrame([vars(t) for t in trades])

    # ── Conteos ──
    total: int = len(df)
    longs = df[df["setup"] == "long"]
    shorts = df[df["setup"] == "short"]
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]

    # ── Win rates ──
    wr_total: float = len(wins) / total * 100
    wr_long: float = (
        len(longs[longs["net_pnl"] > 0]) / len(longs) * 100
        if len(longs) > 0
        else 0.0
    )
    wr_short: float = (
        len(shorts[shorts["net_pnl"] > 0]) / len(shorts) * 100
        if len(shorts) > 0
        else 0.0
    )

    # ── IC del win rate (Wald, 99%) ──
    p_hat: float = len(wins) / total
    z99: float = 2.576
    wr_se: float = float(np.sqrt(p_hat * (1 - p_hat) / total))
    wr_ci_lo: float = (p_hat - z99 * wr_se) * 100
    wr_ci_hi: float = (p_hat + z99 * wr_se) * 100

    # ── Expectancy ──
    total_pnl: float = df["net_pnl"].sum()
    avg_pnl_pts: float = df["pnl_pts"].mean()
    avg_pnl_usd: float = df["net_pnl"].mean()

    # ── Profit factor ──
    gross_profit: float = float(wins["net_pnl"].sum()) if len(wins) > 0 else 0.0
    gross_loss: float = float(abs(losses["net_pnl"].sum())) if len(losses) > 0 else 0.0
    profit_factor: float = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # ── Sharpe ratio anualizado (sobre retornos diarios, incluye días sin trades) ──
    # trade_date se almacena como string "YYYY-MM-DD"; se alinea explícitamente
    # en el mismo formato para evitar mismatch string/Timestamp en reindex.
    daily_pnl = df.groupby("trade_date")["net_pnl"].sum()
    first_date = pd.to_datetime(daily_pnl.index.min())
    last_date = pd.to_datetime(daily_pnl.index.max())
    all_biz_days_str: pd.Index = pd.Index(
        pd.bdate_range(start=first_date, end=last_date).strftime("%Y-%m-%d")
    )
    daily_returns: pd.Series = (
        daily_pnl.reindex(all_biz_days_str, fill_value=0.0) / config.account_size
    )
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe: float = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # ── Max drawdown ──
    equity_arr = np.array(portfolio.equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    dd = peak - equity_arr
    max_dd_usd: float = float(dd.max())
    max_dd_pct: float = float((dd / peak).max() * 100) if peak.max() > 0 else 0.0

    # ── Duración de trades ──
    durations = df["duration_seconds"] / 60.0  # minutos

    # ── Breakdown por sesión ──
    session_stats = df.groupby("session").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    )

    # ── Breakdown por setup ──
    setup_stats = df.groupby("setup").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    )

    # ── Breakdown por tipo de salida ──
    exit_stats = df.groupby("exit_reason").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
    )

    # ── Exportar resultados ──
    results_dir = config.results_path
    results_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame({"equity": portfolio.equity_curve})
    equity_df.to_csv(results_dir / "equity_curve.csv", index=False)
    df.to_csv(results_dir / "trade_log.csv", index=False)

    # ── Reporte ──
    print("=" * 60)
    print("  GAMMALITO — Backtest Report")
    print("=" * 60)

    print(f"\n{'Total trades:':<30} {total}")
    print(f"  {'Long:':<28} {len(longs)}")
    print(f"  {'Short:':<28} {len(shorts)}")

    print(f"\n{'Win Rate Total:':<30} {wr_total:.1f}%  IC99% [{wr_ci_lo:.1f}%, {wr_ci_hi:.1f}%]")
    print(f"  {'Long:':<28} {wr_long:.1f}%")
    print(f"  {'Short:':<28} {wr_short:.1f}%")

    print(f"\n{'Expectancy (pts):':<30} {avg_pnl_pts:.2f}")
    print(f"{'Expectancy (USD):':<30} ${avg_pnl_usd:.2f}")
    print(f"{'Total Net PnL:':<30} ${total_pnl:,.2f}")

    print(f"\n{'Profit Factor:':<30} {profit_factor:.2f}")
    print(f"{'Sharpe Ratio (anual):':<30} {sharpe:.2f}")

    print(f"\n{'Max Drawdown (USD):':<30} ${max_dd_usd:,.2f}")
    print(f"{'Max Drawdown (%):':<30} {max_dd_pct:.2f}%")

    print(f"\n{'Duración media (min):':<30} {durations.mean():.1f}")
    print(f"{'Duración mediana (min):':<30} {durations.median():.1f}")
    print(f"{'Duración max (min):':<30} {durations.max():.1f}")

    print(f"\n{'─' * 60}")
    print("  Breakdown por Sesión")
    print(f"{'─' * 60}")
    print(session_stats.to_string())

    print(f"\n{'─' * 60}")
    print("  Breakdown por Setup")
    print(f"{'─' * 60}")
    print(setup_stats.to_string())

    print(f"\n{'─' * 60}")
    print("  Breakdown por Tipo de Salida")
    print(f"{'─' * 60}")
    print(exit_stats.to_string())

    print(f"\n  Equity curve → {results_dir / 'equity_curve.csv'}")
    print(f"  Trade log    → {results_dir / 'trade_log.csv'}")
    print("=" * 60)

    return {
        "total_trades": total,
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "win_rate": wr_total,
        "win_rate_ci99_lo": wr_ci_lo,
        "win_rate_ci99_hi": wr_ci_hi,
        "win_rate_long": wr_long,
        "win_rate_short": wr_short,
        "expectancy_pts": avg_pnl_pts,
        "expectancy_usd": avg_pnl_usd,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
    }
