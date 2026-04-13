"""Validación estadística del backtest: Monte Carlo, bootstrap, régimen.

Responde tres preguntas:
  1. ¿El PnL es distinguible de azar? (Monte Carlo)
  2. ¿Las métricas son estables? (Bootstrap CI)
  3. ¿El edge depende de un régimen específico? (Análisis de régimen)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════
#  1. MONTE CARLO — ¿ES DISTINGUIBLE DE AZAR?
# ═══════════════════════════════════════════

def monte_carlo_permutation(
    trade_pnls: np.ndarray,
    n_simulations: int = 100_000,
    initial_equity: float = 500_000.0,
    seed: int = 42,
) -> dict:
    """Permuta el orden de los trade PnLs y compara con el resultado real.

    Hipótesis nula: el orden de los trades no importa, el PnL total
    es robusto al path. Si el PnL real está en el percentil >95 de las
    permutaciones, el resultado podría depender del orden (path-dependence
    por sizing dinámico, daily loss limits, etc.).

    También compara contra un null model de retornos shuffled con signo
    aleatorio (destruye la correlación señal-dirección).
    """
    rng = np.random.default_rng(seed)
    n_trades: int = len(trade_pnls)
    real_total_pnl: float = float(trade_pnls.sum())
    real_equity_curve: np.ndarray = initial_equity + np.concatenate(
        [[0], np.cumsum(trade_pnls)]
    )
    real_max_dd: float = _max_drawdown_from_curve(real_equity_curve)

    # ── Permutación de orden (mismos trades, distinto path) ──
    perm_pnls: np.ndarray = np.zeros(n_simulations)
    perm_max_dd: np.ndarray = np.zeros(n_simulations)
    perm_sharpe: np.ndarray = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled: np.ndarray = rng.permutation(trade_pnls)
        perm_pnls[i] = shuffled.sum()
        curve: np.ndarray = initial_equity + np.concatenate(
            [[0], np.cumsum(shuffled)]
        )
        perm_max_dd[i] = _max_drawdown_from_curve(curve)
        if shuffled.std() > 0:
            perm_sharpe[i] = shuffled.mean() / shuffled.std() * np.sqrt(252)

    # ── Null model: signo aleatorio (¿hay edge direccional?) ──
    null_pnls: np.ndarray = np.zeros(n_simulations)
    abs_pnls: np.ndarray = np.abs(trade_pnls)
    for i in range(n_simulations):
        signs: np.ndarray = rng.choice([-1, 1], size=n_trades)
        null_pnls[i] = (abs_pnls * signs).sum()

    # ── p-value: fracción de simulaciones que superan el PnL real ──
    p_value_perm: float = float((perm_pnls >= real_total_pnl).mean())
    p_value_null: float = float((null_pnls >= real_total_pnl).mean())

    return {
        "real_total_pnl": real_total_pnl,
        "real_max_dd": real_max_dd,
        "perm_pnl_mean": float(perm_pnls.mean()),
        "perm_pnl_std": float(perm_pnls.std()),
        "perm_pnl_p5": float(np.percentile(perm_pnls, 5)),
        "perm_pnl_p50": float(np.percentile(perm_pnls, 50)),
        "perm_pnl_p95": float(np.percentile(perm_pnls, 95)),
        "perm_max_dd_mean": float(perm_max_dd.mean()),
        "perm_max_dd_p95": float(np.percentile(perm_max_dd, 95)),
        "perm_sharpe_mean": float(perm_sharpe.mean()),
        "perm_sharpe_p95": float(np.percentile(perm_sharpe, 95)),
        "p_value_permutation": p_value_perm,
        "null_pnl_mean": float(null_pnls.mean()),
        "null_pnl_p95": float(np.percentile(null_pnls, 95)),
        "p_value_null": p_value_null,
        "n_simulations": n_simulations,
    }


# ═══════════════════════════════════════════
#  2. BOOTSTRAP — ¿LAS MÉTRICAS SON ESTABLES?
# ═══════════════════════════════════════════

def bootstrap_metrics(
    trade_pnls: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap con reemplazo sobre los trade PnLs.

    Genera intervalos de confianza para: expectancy, win rate,
    profit factor, sharpe ratio. Si el CI incluye 0 (expectancy)
    o 1.0 (profit factor), no hay edge estadístico.
    """
    rng = np.random.default_rng(seed)
    n_trades: int = len(trade_pnls)
    alpha: float = (1 - ci_level) / 2

    boot_expectancy: np.ndarray = np.zeros(n_bootstrap)
    boot_wr: np.ndarray = np.zeros(n_bootstrap)
    boot_pf: np.ndarray = np.zeros(n_bootstrap)
    boot_sharpe: np.ndarray = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample: np.ndarray = rng.choice(trade_pnls, size=n_trades, replace=True)
        boot_expectancy[i] = sample.mean()
        boot_wr[i] = (sample > 0).mean()
        wins_sum: float = sample[sample > 0].sum()
        losses_sum: float = abs(sample[sample <= 0].sum())
        boot_pf[i] = wins_sum / losses_sum if losses_sum > 0 else 10.0
        if sample.std() > 0:
            boot_sharpe[i] = sample.mean() / sample.std() * np.sqrt(252)

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.percentile(arr, alpha * 100)),
            float(np.median(arr)),
            float(np.percentile(arr, (1 - alpha) * 100)),
        )

    return {
        "expectancy_ci": _ci(boot_expectancy),
        "win_rate_ci": _ci(boot_wr),
        "profit_factor_ci": _ci(boot_pf),
        "sharpe_ci": _ci(boot_sharpe),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


# ═══════════════════════════════════════════
#  3. RÉGIMEN — ¿EL EDGE DEPENDE DE GAMMA?
# ═══════════════════════════════════════════

def regime_analysis(trade_log: pd.DataFrame, df_data: pd.DataFrame) -> dict:
    """Cruza cada trade con el régimen de gamma al momento de entrada.

    Usa sum_gex_oi de la barra de entrada para clasificar el trade
    como gamma positiva o negativa, y calcula métricas por régimen.
    También hace un breakdown temporal (primeros 50% vs últimos 50%).
    """
    # ── Régimen de gamma por trade ──
    gex_at_entry: list[float] = []
    for _, row in trade_log.iterrows():
        idx: int = int(row["entry_idx"])
        if idx < len(df_data):
            gex_at_entry.append(float(df_data.iloc[idx]["sum_gex_oi"]))
        else:
            gex_at_entry.append(0.0)

    trade_log = trade_log.copy()
    trade_log["gex_regime"] = [
        "neg_gamma" if g < 0 else "pos_gamma" for g in gex_at_entry
    ]

    regime_stats: dict = {}
    for reg_name in ["neg_gamma", "pos_gamma"]:
        subset = trade_log[trade_log["gex_regime"] == reg_name]
        if len(subset) == 0:
            continue
        regime_stats[reg_name] = {
            "trades": len(subset),
            "total_pnl": float(subset["net_pnl"].sum()),
            "avg_pnl": float(subset["net_pnl"].mean()),
            "win_rate": float((subset["net_pnl"] > 0).mean() * 100),
            "pct_of_total_trades": float(len(subset) / len(trade_log) * 100),
        }

    # ── Análisis temporal: primera mitad vs segunda mitad ──
    mid: int = len(trade_log) // 2
    first_half = trade_log.iloc[:mid]
    second_half = trade_log.iloc[mid:]

    temporal_stats: dict = {}
    for label, subset in [("first_half", first_half), ("second_half", second_half)]:
        temporal_stats[label] = {
            "trades": len(subset),
            "total_pnl": float(subset["net_pnl"].sum()),
            "avg_pnl": float(subset["net_pnl"].mean()),
            "win_rate": float((subset["net_pnl"] > 0).mean() * 100),
            "neg_gamma_pct": float(
                (subset["gex_regime"] == "neg_gamma").mean() * 100
            ) if "gex_regime" in subset.columns else 0.0,
        }

    # ── Breakdown por semana ──
    trade_log["week"] = pd.to_datetime(trade_log["trade_date"]).dt.isocalendar().week
    weekly = trade_log.groupby("week").agg(
        trades=("net_pnl", "count"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    )

    return {
        "by_regime": regime_stats,
        "temporal": temporal_stats,
        "weekly": weekly,
    }


# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════

def _max_drawdown_from_curve(equity_curve: np.ndarray) -> float:
    """Max drawdown en USD desde un array de equity."""
    peak: np.ndarray = np.maximum.accumulate(equity_curve)
    dd: np.ndarray = peak - equity_curve
    return float(dd.max())


# ═══════════════════════════════════════════
#  REPORTE
# ═══════════════════════════════════════════

def print_validation_report(
    mc: dict, boot: dict, regime_data: dict
) -> None:
    """Imprime el reporte completo de validación."""

    print("\n" + "=" * 60)
    print("  VALIDACIÓN ESTADÍSTICA")
    print("=" * 60)

    # ── Monte Carlo ──
    print("\n─── Monte Carlo Permutation Test ───")
    print(f"  PnL real:              ${mc['real_total_pnl']:>12,.2f}")
    print(f"  PnL medio (permut.):   ${mc['perm_pnl_mean']:>12,.2f}")
    print(f"  PnL P5–P95 (permut.):  ${mc['perm_pnl_p5']:>12,.2f} — ${mc['perm_pnl_p95']:>12,.2f}")
    print(f"  Max DD medio (permut.):${mc['perm_max_dd_mean']:>12,.2f}")
    print(f"  Max DD P95 (permut.):  ${mc['perm_max_dd_p95']:>12,.2f}")
    print(f"  Max DD real:           ${mc['real_max_dd']:>12,.2f}")
    print(f"  p-value (permutación): {mc['p_value_permutation']:.4f}")
    print(f"     → {'PnL NO depende significativamente del path ✓' if mc['p_value_permutation'] > 0.05 else '⚠ PnL depende del orden de trades (path-dependent)'}")
    print(f"\n  PnL medio (null/azar): ${mc['null_pnl_mean']:>12,.2f}")
    print(f"  PnL P95 (null):        ${mc['null_pnl_p95']:>12,.2f}")
    print(f"  p-value (null):        {mc['p_value_null']:.4f}")
    print(f"     → {'Edge direccional significativo ✓' if mc['p_value_null'] < 0.05 else '⚠ No se puede descartar azar en la dirección'}")
    print(f"  ({mc['n_simulations']:,} simulaciones)")

    # ── Bootstrap ──
    ci: float = boot["ci_level"] * 100
    print(f"\n─── Bootstrap CI ({ci:.0f}%) ───")
    exp_lo, exp_med, exp_hi = boot["expectancy_ci"]
    wr_lo, wr_med, wr_hi = boot["win_rate_ci"]
    pf_lo, pf_med, pf_hi = boot["profit_factor_ci"]
    sh_lo, sh_med, sh_hi = boot["sharpe_ci"]

    print(f"  Expectancy:     ${exp_lo:>9,.2f}  —  ${exp_med:>9,.2f}  —  ${exp_hi:>9,.2f}")
    edge_exp = "✓" if exp_lo > 0 else "⚠ CI incluye 0"
    print(f"     → {edge_exp}")

    print(f"  Win Rate:       {wr_lo*100:>8.1f}% —  {wr_med*100:>8.1f}% —  {wr_hi*100:>8.1f}%")
    edge_wr = "✓" if wr_lo > 0.5 else "⚠ CI incluye 50%"
    print(f"     → {edge_wr}")

    print(f"  Profit Factor:  {pf_lo:>8.2f}  —  {pf_med:>8.2f}  —  {pf_hi:>8.2f}")
    edge_pf = "✓" if pf_lo > 1.0 else "⚠ CI incluye 1.0"
    print(f"     → {edge_pf}")

    print(f"  Sharpe (anual): {sh_lo:>8.2f}  —  {sh_med:>8.2f}  —  {sh_hi:>8.2f}")
    print(f"  ({boot['n_bootstrap']:,} resamples)")

    # ── Régimen ──
    print(f"\n─── Análisis de Régimen ───")
    for regime, stats in regime_data.get("by_regime", {}).items():
        label = "Γ−" if regime == "neg_gamma" else "Γ+"
        print(f"  {label}: {stats['trades']} trades ({stats['pct_of_total_trades']:.1f}%), "
              f"WR {stats['win_rate']:.1f}%, "
              f"Avg ${stats['avg_pnl']:,.2f}, "
              f"Total ${stats['total_pnl']:,.2f}")

    # ── Temporal ──
    print(f"\n─── Análisis Temporal (1ra mitad vs 2da mitad) ───")
    for half, stats in regime_data.get("temporal", {}).items():
        label = "1ra mitad" if half == "first_half" else "2da mitad"
        print(f"  {label}: {stats['trades']} trades, "
              f"WR {stats['win_rate']:.1f}%, "
              f"Avg ${stats['avg_pnl']:,.2f}, "
              f"Total ${stats['total_pnl']:,.2f}, "
              f"Γ− {stats['neg_gamma_pct']:.1f}%")

    t1 = regime_data["temporal"].get("first_half", {})
    t2 = regime_data["temporal"].get("second_half", {})
    if t1 and t2:
        ratio = t2.get("avg_pnl", 0) / t1["avg_pnl"] if t1.get("avg_pnl", 0) != 0 else float("inf")
        if abs(ratio) > 3:
            print(f"  ⚠ La 2da mitad tiene {ratio:.1f}x la expectancy de la 1ra — posible dependencia de régimen")
        if t2.get("neg_gamma_pct", 0) > t1.get("neg_gamma_pct", 0) + 15:
            print(f"  ⚠ La 2da mitad tiene más Γ− ({t2['neg_gamma_pct']:.0f}% vs {t1['neg_gamma_pct']:.0f}%) — el edge puede ser específico de régimen")

    # ── Semanal ──
    print(f"\n─── PnL Semanal ───")
    weekly = regime_data.get("weekly")
    if weekly is not None:
        print(weekly.to_string())
        losing_weeks: int = int((weekly["total_pnl"] < 0).sum())
        total_weeks: int = len(weekly)
        print(f"\n  Semanas perdedoras: {losing_weeks}/{total_weeks} ({losing_weeks/total_weeks*100:.0f}%)")

    print("\n" + "=" * 60)


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════

def main() -> None:
    """Ejecuta el backtest y la validación completa."""
    # Agregar Backtest/ al path para imports relativos
    backtest_dir = Path(__file__).resolve().parent
    if str(backtest_dir) not in sys.path:
        sys.path.insert(0, str(backtest_dir))

    from config import BacktestConfig
    from data_loader import load_data
    from backtest_engine import run_backtest
    from analytics import compute_analytics

    config = BacktestConfig()

    print("Cargando datos...")
    df = load_data(config)
    print(f"  Filas operables: {len(df):,}")

    print("Ejecutando backtest...")
    portfolio = run_backtest(df, config)
    print(f"  {len(portfolio.trades)} trades")

    metrics = compute_analytics(portfolio, config)

    if len(portfolio.trades) < 10:
        print("\n⚠ Muy pocos trades para validación estadística.")
        return

    # ── Preparar datos ──
    trade_pnls: np.ndarray = np.array([t.net_pnl for t in portfolio.trades])
    trade_log = pd.DataFrame([vars(t) for t in portfolio.trades])

    # ── Ejecutar validaciones ──
    print("\nEjecutando validación estadística (10,000 simulaciones)...")
    mc = monte_carlo_permutation(trade_pnls, initial_equity=config.account_size)
    boot = bootstrap_metrics(trade_pnls)
    reg = regime_analysis(trade_log, df)

    print_validation_report(mc, boot, reg)


if __name__ == "__main__":
    main()
