"""
Portfolio visualization functions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from financial_ml.utils.config import FIGURE_DIR, SP500_NAMES_FILE
from financial_ml.models import get_model_name

def plot_sector_concentration_over_time(df_portfolio, sector_file=SP500_NAMES_FILE):
    """
    Plot how sector concentration changes over time.
    Shows if model has stable sector bias or is sector-timing.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Load and prepare sector data
    sectors = pd.read_csv(sector_file)
    if 'Symbol' in sectors.columns:
        sectors = sectors.rename(columns={'Symbol': 'ticker'})
    
    # Merge sector info
    df = df_portfolio.merge(sectors[['ticker', 'GICS Sector']], on='ticker', how='left')
    
    # Get long positions only
    longs = df[df['position'] == 1].copy()
    
    # Calculate sector weights per month
    sector_weights = []
    for date in longs['date'].unique():
        month_data = longs[longs['date'] == date]
        sector_counts = month_data['GICS Sector'].value_counts()
        total = len(month_data)
        
        for sector, count in sector_counts.items():
            sector_weights.append({
                'date': date,
                'sector': sector,
                'weight': count / total * 100
            })
    
    weights_df = pd.DataFrame(sector_weights)
    
    # Pivot for plotting
    weights_pivot = weights_df.pivot(index='date', columns='sector', values='weight').fillna(0)
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    weights_pivot.plot.area(ax=ax, alpha=0.8, linewidth=0)
    
    ax.set_title('Sector Allocation Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    figname = FIGURE_DIR/ 'sector_drift_over_time.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved sector drift chart: {figname}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SECTOR STABILITY ANALYSIS")
    print("="*60)
    
    # Calculate mean and std for each sector
    sector_stats = weights_pivot.describe().T[['mean', 'std', 'min', 'max']]
    sector_stats = sector_stats.sort_values('mean', ascending=False)
    
    print("\nAverage Sector Weights Over Time:")
    print(f"{'Sector':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*60)
    
    for sector in sector_stats.index:
        mean = sector_stats.loc[sector, 'mean']
        std = sector_stats.loc[sector, 'std']
        min_val = sector_stats.loc[sector, 'min']
        max_val = sector_stats.loc[sector, 'max']
        
        # Flag high volatility sectors
        flag = "⚠️⚠️" if std > 10 else "⚠️" if std > 5 else ""
        print(f"{sector:<30} {mean:>7.1f}% {std:>7.1f}% {min_val:>7.1f}% {max_val:>7.1f}% {flag}")
    
    print("\n💡 Interpretation:")
    print("  - Low Std = Stable sector preference (structural bias)")
    print("  - High Std = Sector timing attempts (may be luck)")
    print("="*60)
    
    return weights_pivot

def draw_cumulative_drawdown_all(portfolio_returns, spy, equal_weight_returns, random_returns,
                            drawdown, max_drawdown, model, portfolio_type, per_top):
    """
    Create 2-panel chart with honest performance attribution.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    model_name = get_model_name(model_key=model)
    model_name_short = get_model_name(model_key=model, short_name=True)

    # Chart 1: Cumulative Returns with Multiple Benchmarks
    axes[0].plot(portfolio_returns['date'], portfolio_returns['cum_return'], 
                label=f'Your Model (Top {per_top}%)', linewidth=2.5, color='#2E7D32', zorder=4)
    
    # Handle random_returns - might be None or DataFrame
    if random_returns is not None:
        if isinstance(random_returns, pd.DataFrame):
            # It's already a DataFrame with 'date' and 'cum_return'
            axes[0].plot(random_returns['date'], random_returns['cum_return'], 
                        label=f'Random Selection (Top {per_top}%)', linewidth=2, 
                        color='#FFA726', linestyle='--', alpha=0.8, zorder=3)
        else:
            # It's a Series - compute cumulative on the fly
            cum_random = (1 + random_returns).cumprod()
            axes[0].plot(portfolio_returns['date'], cum_random, 
                        label=f'Random Selection (Top {per_top}%)', linewidth=2, 
                        color='#FFA726', linestyle='--', alpha=0.8, zorder=3)
    
    # Handle equal_weight_returns - might be None or DataFrame
    if equal_weight_returns is not None:
        if isinstance(equal_weight_returns, pd.DataFrame):
            axes[0].plot(equal_weight_returns['date'], equal_weight_returns['cum_return'], 
                        label='Equal-Weight Benchmark (100%)', linewidth=2, 
                        color='#1976D2', linestyle='-.', alpha=0.8, zorder=2)
        else:
            cum_equal = (1 + equal_weight_returns).cumprod()
            axes[0].plot(portfolio_returns['date'], cum_equal, 
                        label='Equal-Weight Benchmark (100%)', linewidth=2, 
                        color='#1976D2', linestyle='-.', alpha=0.8, zorder=2)
    
    # SPY (your original code)
    if spy is not None:
        axes[0].plot(portfolio_returns['date'], portfolio_returns['spy_cum_return'], 
                    label='SPY (Cap-Weighted)', linewidth=2, color='#757575', 
                    linestyle=':', alpha=0.7, zorder=1)
    
    axes[0].set_title(f'Performance Attribution: {model_name} Strategy', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(alpha=0.3)

    # Chart 2: Drawdown (unchanged)
    axes[1].fill_between(portfolio_returns['date'], drawdown * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
    axes[1].axhline(max_drawdown * 100, color='darkred', linestyle='--', 
                    label=f'Max Drawdown: {max_drawdown:.1%}')
    axes[1].set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    figname = FIGURE_DIR / f"portfolio_backtest_{model}_{portfolio_type}_top{per_top}.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    
    print(f"\nChart saved to {figname}")

