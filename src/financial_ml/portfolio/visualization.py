"""
Portfolio visualization functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from financial_ml.utils.config import FIGURE_DIR, SP500_NAMES_FILE, SEPARATOR_WIDTH
from financial_ml.utils.paths import get_fig_name
from financial_ml.models import get_model_name
from financial_ml.portfolio.diagnostics import calculate_model_agreement_correlations



def plot_correlation_matrix(preds_df, models_dict, fig_dir=FIGURE_DIR):
    """
    Plot model prediction correlation heatmap
    
    Args:
        preds_df: DataFrame with columns ['date', 'ticker', 'y_prob', 'model']
        models_dict: Dict of model names (keys) to use
        fig_dir: Directory to save figure
    
    Returns:
        Path to saved figure
    """
    
    # Get correlation data from diagnostics
    results = calculate_model_agreement_correlations(preds_df, models_dict)
    correlations = results['correlations']
    
    # Build correlation matrix
    models = list(models_dict.keys())
    n = len(models)
    corr_matrix = np.ones((n, n))  # Diagonal = 1
    
    # Fill matrix (symmetric)
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                key1 = f"{model1}_vs_{model2}"
                key2 = f"{model2}_vs_{model1}"
                
                if key1 in correlations:
                    corr_matrix[i, j] = correlations[key1]['correlation']
                elif key2 in correlations:
                    corr_matrix[i, j] = correlations[key2]['correlation']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',  # Red (high) to Yellow to Green (low)
                vmin=0, 
                vmax=1,
                xticklabels=models,
                yticklabels=models,
                cbar_kws={'label': 'Spearman Correlation'},
                square=True,
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    ax.set_title('Model Prediction Correlation Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = fig_dir / get_fig_name('correlation', models)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved correlation matrix: {output_path}")

    return output_path


def plot_performance_vs_correlation(preds_df, models_dict, performance_dict, reference_model='rf', fig_dir=FIGURE_DIR):
    """
    Plot model performance vs correlation with reference model
    
    Args:
        preds_df: DataFrame with predictions
        models_dict: Dict of model names
        performance_dict: Dict mapping model names to Sharpe ratios
                         e.g., {'rf': 0.93, 'logreg_l2': 0.79, ...}
        reference_model: Model to compare correlations against (default: 'rf')
        fig_dir: Directory to save figure
    
    Returns:
        Path to saved figure
    """
    
    # Get correlation data
    results = calculate_model_agreement_correlations(preds_df, models_dict)
    correlations = results['correlations']
    
    # Extract correlations with reference model
    ref_corr = {}
    for model in models_dict.keys():
        if model == reference_model:
            ref_corr[model] = 1.0
        else:
            key1 = f"{reference_model}_vs_{model}"
            key2 = f"{model}_vs_{reference_model}"
            
            if key1 in correlations:
                ref_corr[model] = correlations[key1]['correlation']
            elif key2 in correlations:
                ref_corr[model] = correlations[key2]['correlation']
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models_dict.keys():
        if model not in performance_dict or model not in ref_corr:
            continue
            
        x = ref_corr[model]
        y = performance_dict[model]
        
        if model == reference_model:
            ax.scatter(x, y, s=300, c='green', marker='*', 
                      label=f'{model} (Reference)', zorder=3, edgecolors='black')
        else:
            ax.scatter(x, y, s=150, alpha=0.7, edgecolors='black')
            ax.annotate(model, (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
    
    # Reference lines
    ref_sharpe = performance_dict.get(reference_model, 0.93)
    ax.axhline(y=ref_sharpe, color='green', linestyle='--', alpha=0.3, 
              label=f'{reference_model} Sharpe')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3,
              label='Correlation threshold (0.5)')
    
    # Quadrant labels
    ax.text(0.25, ref_sharpe + 0.02, 'Low corr,\nHigh perf', 
            ha='center', fontsize=9, alpha=0.5, style='italic')
    ax.text(0.75, ref_sharpe + 0.02, 'High corr,\nHigh perf', 
            ha='center', fontsize=9, alpha=0.5, style='italic')
    
    ax.set_xlabel(f'Correlation with {reference_model}', fontsize=12)
    ax.set_ylabel('Portfolio Sharpe Ratio', fontsize=12)
    ax.set_title(f'Model Performance vs Correlation with Best Model ({reference_model})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = fig_dir / get_fig_name('performance_vs_correlation', model)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved performance vs correlation plot: {output_path}")
    
    return output_path


def plot_sector_concentration_over_time(df_portfolio, sector_file=SP500_NAMES_FILE, fig_dir=FIGURE_DIR, model=''):
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

    figname = fig_dir/ get_fig_name(fig_type='concentration', model_name=model)
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved sector drift chart: {figname}")
    
    # Summary statistics
    print("\n" + "="* SEPARATOR_WIDTH)
    print("SECTOR STABILITY ANALYSIS")
    print("="* SEPARATOR_WIDTH)
    
    # Calculate mean and std for each sector
    sector_stats = weights_pivot.describe().T[['mean', 'std', 'min', 'max']]
    sector_stats = sector_stats.sort_values('mean', ascending=False)
    
    print("\nAverage Sector Weights Over Time:")
    print(f"{'Sector':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"* SEPARATOR_WIDTH)
    
    for sector in sector_stats.index:
        mean = sector_stats.loc[sector, 'mean']
        std = sector_stats.loc[sector, 'std']
        min_val = sector_stats.loc[sector, 'min']
        max_val = sector_stats.loc[sector, 'max']
        
        # Flag high volatility sectors
        flag = "⚠️ ⚠️" if std > 10 else "⚠️" if std > 5 else ""
        print(f"{sector:<30} {mean:>7.1f}% {std:>7.1f}% {min_val:>7.1f}% {max_val:>7.1f}% {flag}")
    
    print("\n💡 Interpretation:")
    print("  - Low Std = Stable sector preference (structural bias)")
    print("  - High Std = Sector timing attempts (may be luck)")
    print("="* SEPARATOR_WIDTH)
    
    return weights_pivot

def plot_cumulative_drawdown_all(portfolio_returns, spy, equal_weight_returns, random_returns,
                            drawdown, max_drawdown, model, portfolio_type, per_top, fig_dir=FIGURE_DIR):
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
    #figname = fig_dir / f"portfolio_backtest_{model}_{portfolio_type}_top{per_top}.png"
    figname = fig_dir / get_fig_name("performance", model_name=model, p_type=portfolio_type, per_top=per_top)

    plt.savefig(figname, dpi=300, bbox_inches='tight')
    
    print(f"\nChart saved to {figname}")

