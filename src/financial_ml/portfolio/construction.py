"""
Portfolio construction from model predictions.
"""

def construct_portfolio(df, per_top=10, per_bot=10, pred_col='y_prob'):
    """
    Construct the portfolio using the  per_top% top companies, 
    and the short per_bot% companies. Keeps all columns including date
    """
    df['position'] = 0
    for date in df['date'].unique():
        mask = df['date'] == date
        date_data = df[mask].copy()
        top_subset = (100-per_top)/100.
        bot_subset = (per_bot/100.)
        top = date_data[pred_col].quantile(top_subset)
        bottom = date_data[pred_col].quantile(bot_subset)
        
        df.loc[mask & (df[pred_col] >= top), 'position'] = 1
        df.loc[mask & (df[pred_col] <= bottom), 'position'] = -1
    return df

def smooth_predictions(df, window=3):
    """
    Apply rolling average to predictions to reduce noise.
    
    Args:
        df: DataFrame with 'ticker', 'date', 'y_prob'
        window: Number of months to average (3 = quarterly smoothing)
    
    Returns:
        DataFrame with added 'y_prob_smooth' column
    """
    df = df.sort_values(['ticker', 'date'])
    # For each ticker, smooth y_prob over time
    df['y_prob_smooth'] = df.groupby('ticker')['y_prob'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    
    # Calculate how much smoothing reduced volatility
    df['y_prob_smooth_prev'] = df.groupby('ticker')['y_prob_smooth'].shift(1)
    df['change_smooth'] = (df['y_prob_smooth'] - df['y_prob_smooth_prev']).abs()
    
    df['y_prob_prev_raw'] = df.groupby('ticker')['y_prob'].shift(1)
    df['change_raw'] = (df['y_prob'] - df['y_prob_prev_raw']).abs()
    mean_change_before = df['change_raw'].mean()
    print("\n" + "="*60)
    print(f"SMOOTHING IMPACT (window={window} months)")
    print("="*60)
    print(f"\nBefore smoothing:")
    print(f"  Mean change: {mean_change_before:.2%}")
    print(f"\nAfter smoothing:")
    print(f"  Mean change: {df['change_smooth'].mean():.4f} ({df['change_smooth'].mean():.2%})")
    print(f"  Reduction:   {(0.0467 - df['change_smooth'].mean()) / 0.0467:.1%}")
    print("="*60)
    
    return df

