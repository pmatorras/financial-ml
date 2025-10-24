"""
Portfolio backtesting orchestration.
"""

import pandas as pd
import numpy as np
from financial_ml.utils.paths import get_market_file, get_prediction_file
from financial_ml.models import get_models
from financial_ml.portfolio.diagnostics import print_model_agreement, compare_model_performance_by_period, analyze_prediction_stability, analyze_turnover, analyze_sector_concentration, analyze_beta_exposure, compare_drawdowns_to_spy, test_sharpe_significance
from financial_ml.portfolio.performance import aggregate_portfolio_return, include_benchmark_return
from financial_ml.portfolio.visualization import draw_cumulative_drawdown
from financial_ml.portfolio.construction import construct_portfolio, smooth_predictions

def run_backtest(args, per_top=90, per_bot=10):
    """
    Execute complete portfolio backtesting pipeline with diagnostics and visualization.
    
    This function orchestrates the full end-to-end backtesting workflow.
        Args:
        args: Namespace with attributes:
        per_top: Percentage of top stocks to long
        per_bot: Percentage of bottom stocks to short
    """
    # Load Data for predictions, stocks, and SPY benchmark
    model = args.best_model if 'all' in args.model else args.model
    print("making calculations for", model)
    preds_nm = get_prediction_file(args)
    preds = pd.read_csv(preds_nm)

    doEnsemble=False
    if doEnsemble:
        # Create ensemble predictions by averaging LogReg_L2 + RF_cal
        logreg_preds = preds[preds['model'] == 'logreg_l2'][['date', 'ticker', 'y_prob', 'y_true']].copy()
        rf_preds = preds[preds['model'] == 'rf_cal'][['date', 'ticker', 'y_prob', 'y_true']].copy()
        
        # Merge and average
        ensemble = logreg_preds.merge(rf_preds, on=['date', 'ticker'], suffixes=('_lr', '_rf'))
        ensemble['y_prob'] = 0.5 * ensemble['y_prob_lr'] + 0.5 * ensemble['y_prob_rf']
        ensemble['y_true'] = ensemble['y_true_lr']  # Same labels
        ensemble['model'] = 'ensemble_A'
        ensemble = ensemble[['date', 'ticker', 'y_prob', 'y_true', 'model']]
        
        # Append ensemble to predictions
        preds = pd.concat([preds, ensemble], ignore_index=True)
        print(f"Added ensemble_A predictions: {len(ensemble)} rows")
        model = "ensemble_A"

    preds['date'] = pd.to_datetime(preds['date'])
    prices_file = get_market_file(args)
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True).ffill()
    spy = prices['SPY'] if 'SPY' in prices.columns else None
    prices = prices.drop(columns=['SPY'], errors='ignore')

    # Calculate monthly returns
    returns_realized = prices.pct_change(periods=1).stack().reset_index()
    returns_realized.columns = ['date', 'ticker', 'return']

    # Merge Predictions with Returns
    df = preds.merge(returns_realized, on=['date', 'ticker'], how='inner')

    models =get_models()
    print_model_agreement(df, models)
    compare_model_performance_by_period(preds, returns_realized, models=['logreg_l2', model])
    # Filter to best model (Logistic L2)
    df = df[df['model'] == model].copy()
    df = smooth_predictions(df, window=3)
    stocks_per_date = df.groupby('date')['ticker'].nunique()
    median_stocks = stocks_per_date.median()
    
    # Filter out dates with less than 50% of typical stock count
    complete_dates = stocks_per_date[stocks_per_date > (median_stocks * 0.5)].index
    df = df[df['date'].isin(complete_dates)]
    
    print(f"\n=== DATA FILTERING ===")
    print(f"Removed {len(stocks_per_date) - len(complete_dates)} incomplete dates")
    print(f"Backtest period: {df['date'].min().date()} to {df['date'].max().date()}")

    print("\n=== PREDICTIONS DATA ===")
    print(f"Total prediction rows: {len(preds)}")
    print(f"Unique dates in predictions: {preds['date'].nunique()}")
    print(f"Date range: {preds['date'].min()} to {preds['date'].max()}")

    # Check 2025-09-30 specifically
    preds_sep30 = preds[preds['date'] == pd.Timestamp('2025-09-30')]
    print(f"\nPredictions on 2025-09-30:")
    print(f"  Number of rows: {len(preds_sep30)}")
    print(f"  Unique tickers: {preds_sep30['ticker'].nunique()}")
    print(f"  Models: {preds_sep30['model'].unique()}")
    # Check prediction distribution
    print("\nPrediction Distribution:")
    print(df['y_prob'].describe())
    print(f"\nMin: {df['y_prob'].min():.3f}")
    print(f"Max: {df['y_prob'].max():.3f}")
    print(f"Mean: {df['y_prob'].mean():.3f}")
    print(f"Median: {df['y_prob'].median():.3f}")

    # Check if predictions are all <0.5
    if df['y_prob'].max() < 0.5:
        print("\n❌ BUG: All predictions are <50%! Your model is broken.")
    elif df['y_prob'].min() > 0.5:
        print("\n⚠️  All predictions are >50%. Model may be too confident.")
    else:
        print("\n✅ Predictions span 0-1 range (normal)")

    pred_col = 'y_prob_smooth'
    df =construct_portfolio(df, per_top=per_top, per_bot=per_bot, pred_col=pred_col)
    analyze_prediction_stability(df)
    turnover = analyze_turnover(df) #turnover analysis
    analyze_sector_concentration(df, pred_col) 

    # Calculate Portfolio Returns
    portfolio_returns = df.groupby('date').apply(aggregate_portfolio_return,  portfolio_type=args.type, include_groups=False).reset_index()
    portfolio_returns.columns = ['date', 'portfolio_return']

    # Cumulative returns
    portfolio_returns['cum_return'] = (1 + portfolio_returns['portfolio_return']).cumprod()

    portfolio_returns = include_benchmark_return(spy, portfolio_returns)

    beta, alpha = analyze_beta_exposure(portfolio_returns)
    compare_drawdowns_to_spy(portfolio_returns)

    # Sharpe Ratio (annualized)
    avg_return = portfolio_returns['portfolio_return'].mean() * 12  # Monthly to annual
    volatility = portfolio_returns['portfolio_return'].std() * np.sqrt(12)
    risk_free_rate = 0.04  # Assume 4% risk-free rate
    sharpe_ratio = (avg_return - risk_free_rate) / volatility

    # Maximum Drawdown
    running_max = portfolio_returns['cum_return'].cummax()
    drawdown = (portfolio_returns['cum_return'] - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_rate = (portfolio_returns['portfolio_return'] > 0).sum() / len(portfolio_returns)


    test_sharpe_significance(portfolio_returns=portfolio_returns, sharpe_ratio=sharpe_ratio)


    print("=" * 60)
    print(f"PORTFOLIO PERFORMANCE METRICS {args.type}")
    print("=" * 60)
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.1%}")
    print(f"Win Rate:            {win_rate:.1%}")
    print(f"Annual Return:       {avg_return:.1%}")
    print(f"Annual Volatility:   {volatility:.1%}")
    print(f"Total Return:        {(portfolio_returns['cum_return'].iloc[-1] - 1):.1%}")
    print("=" * 60)

    draw_cumulative_drawdown(portfolio_returns=portfolio_returns, spy=spy, drawdown=drawdown, max_drawdown=max_drawdown, model=model, portfolio_type=args.type)