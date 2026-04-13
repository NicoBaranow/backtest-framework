"""Entry point del motor de backtesting Gammalito."""

import time

from config import BacktestConfig
from data_loader import load_data
from backtest_engine import run_backtest
from analytics import compute_analytics


def main() -> None:
    config = BacktestConfig()

    print("Cargando datos...")
    df = load_data(config)
    print(f"  Filas operables: {len(df):,}")

    print("Ejecutando backtest...")
    t0 = time.perf_counter()
    portfolio = run_backtest(df, config)
    elapsed = time.perf_counter() - t0
    print(f"  Completado en {elapsed:.2f}s — {len(portfolio.trades)} trades")

    compute_analytics(portfolio, config)


if __name__ == "__main__":
    main()
