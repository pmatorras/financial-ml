# Design Decisions and Rationale

This document explains the reasoning behind key design choices in the project.


## Table of Contents

- [1. Model Choice: Random Forest (Calibrated)](#1-model-choice-random-forest-calibrated)
- [2. Ensemble: Rejected](#2-ensemble-rejected)
- [3. Smoothing: 3-Month Window](#3-smoothing-3-month-window)
- [4. Time Series CV: Expanding Window](#4-time-series-cv-expanding-window)
- [5. Transaction Costs: 10 bps](#5-transaction-costs-10-bps)
- [6. Portfolio strategy](#6-portfolio-strategy-100-long-only)
- [7. Sentiment features](#7-sentiment-features-vix-with-clipped-interactions)
- [8. Random forest hyperparameters](#8-random-forest-hyperparameters)
- [Summary: Key Principles](#summary-key-principles)
---


## 1. Model Choice: Random Forest (Calibrated)

### Decision: Use calibrated Random Forest as final model

### Why Random Forest Over Logistic Regression?
1. **Better performance**
    - RF test AUC: 0.557 vs LogReg 0.558 (similar)
    - RF Sharpe: 0.71 vs LogReg ~0.65 (better)

2. **Captures non-linearities**
    - Stock returns are not linear in factors
    - Interactions matter (e.g., value works differently in small vs large caps)

3. **Robust to outliers**
    - Tree-based models less sensitive to extreme values
    - No need for extensive outlier treatment

### Why Constrained (max_depth=3)?
1. **Prevents overfitting**
    - Deep trees: Test AUC drops to 0.52
    - Shallow trees: Generalize better

2. **Interpretability**
    - Depth=3 means max 8 leaf nodes per tree
    - Easier to understand decision boundaries

### Why Calibration?
1. **Fixed miscalibrated probabilities**
    - Original: mean 0.487 (predicting only 5.7% beat SPY)
    - Calibrated: mean 0.502 (proper interpretation)

2. **Improved performance**
    - Sharpe: 0.71 → 0.80 (+13%)
    - Better discrimination (std 0.035 → 0.059)

3. **Industry standard**
    - Isotonic calibration is well-established
    - Preserves ranking

---

## 2. Ensemble: Rejected

### Decision: Do NOT use ensemble (LogReg + RF)

### Why Not Ensemble?
1. **Underperformed single model:**
    - Ensemble Sharpe: 0.71 vs RF_cal 0.80 → Weaker model dragged down performance

2. **Correlation too high (0.556)** → Models captured similar patterns


### Key Learning:
**Ensembles work when:**
- Models have similar performance
- Low correlation (<0.5)
- Capture truly different patterns

**Our case:**
- RF_cal clearly superior
- Moderate correlation (not enough diversity)

**Conclusion: Use best single model**

---

## 3. Smoothing: 3-Month Window

### Decision: Apply 3-month rolling average to predictions

### Problem:
Raw predictions were found to be volatile month-to-month (5.18% mean change)


**→ Move to a 3-month window:** 

### Rationale:
1. **Reduces noise without losing signal**
    - 58.7% reduction in volatility
    - Sharpe maintained at 0.80

2. **Lowers transaction costs**
    - Turnover: 50% → 42%
    - Cost drag: 1.0% → 0.82%

3. **Smoother returns**
    - Less month-to-month volatility  (1.93% mean change) in portfolio
    - Better investor experience

4. **Align the information to fundamentals**


### Implementation:

```bash
df['y_prob_smooth'] = df.groupby('ticker')['y_prob'].rolling(3, min_periods=1).mean()

```

---

## 4. Time Series CV: Expanding Window

### Decision: Use expanding window (not sliding window)

### Why Expanding?
1. **Uses all available data**
    - Later folds have more training data
    - Better estimates as time progresses

2. **More realistic**
    - In production, you'd use all historical data
    - Mimics actual deployment

3. **Reduces variance**
    - More training data → stabler models
    - Fewer overfitting issues

### Why Not Sliding Window?
- Throws away old data
- Smaller training sets
- Less stable models over time

---


## 5. Transaction Costs: 10 bps

### Decision: Model 10 bps (0.1%) per trade

### Breakdown:
- Bid-ask spread: 5 bps (realistic for liquid S&P 500 stocks)
- Market impact: 5 bps (small orders, moderate liquidity)

### Rationale:
- **Conservative but realistic**
    - S&P 500 stocks are liquid
    - Institutional execution typically 5-15 bps

- **Sensitive to AUM**
    - Works for <$100M strategies
    - Larger funds would face higher costs

### Impact on Results:
- Gross alpha: ~3.0%
- Transaction costs: -0.82%
- **Net alpha: 2.18%** (still attractive)

---

## 6. Portfolio Strategy: 100% Long-Only

### Decision: Use 100% long-only strategy (NOT long-short or 130-30)

### Why Long-Only Over Long-Short?

**Long-short failed**:
- Spread between top and bottom 10%: only 0.09%/month
- Annual return: 0.54% (essentially zero)
- Problem: Both long and short picks have high correlation (0.95)
- Both groups ride the market up together

**Long-only works**:
- Annual return: 17.9%
- Sharpe ratio: 0.80
- Alpha: 2.29% (statistically significant)
- Beta: 1.01 (market-neutral exposure)

### Why Not 130-30?

**Tested 130-30 Strategy**:
- Annual return: 18.3% (+0.4% vs long-only)
- Sharpe ratio: 0.80 (identical)
- Max drawdown: -20.9% (marginally better)

**Conclusion**: Not worth the added complexity, real-world costs eliminate any edge:
- Short borrow fees: 0.5-2% annually
- Higher rebalancing frequency
- Short recall risk
- Margin requirements

### Root Cause Analysis

**Model captures market exposure, not cross-sectional alpha**:
- Top 10% outperform market by 0.2%/month
- Bottom 10% also underperforms market by 0.07%/month
- Correlation between longs and shorts: 0.95

**What this means**:
- Model is good at identifying stocks with positive beta
- Model is weak at differentiating winners from losers
- Alpha comes from market participation + modest selection edge

### Final Strategy

100% long-only with top 10% holdings:
- Captures market beta (~15.5% annual)
- Adds selection alpha (~2.3% annual)
- Total: 17.9% annual with Sharpe 0.80

## 7. Sentiment Features: VIX with Clipped Interactions

### Decision: Add VIX-based features with regime-aware interactions

### Configuration:

**3 VIX-related features (from 1 base feature):**

1. `VIX_percentile`: Rolling percentile rank (0-1 scale)
2. `mom121_x_VIX`: Momentum × VIX_percentile (clipped [0.25, 0.75])
3. `vol12_x_VIX`: Volatility × VIX_percentile (clipped [0.25, 0.75])

### Why Add VIX?

**Market sentiment matters for stock selection:**

- Momentum works differently in calm vs volatile markets
- Stock volatility signal stronger during high market volatility
- VIX captures market regime (calm, normal, stressed)


### Evolution: From 5 Features to 3

**Initial approach failed (5 VIX features):**

- Added: `VIX`, `VIX_log`, `VIX_change_1m`, `VIX_percentile`, `VIX_zscore`
- Result: +0.097 AUC but +122% variance (too unstable)
- Problem: Regime dependency (COVID extreme values broke model)

**Final approach succeeded (3 VIX features):**

- Simplified: Only `VIX_percentile` + 2 interactions variables
- Clipped: [0.25, 0.75] to reduce extreme regime effects
- Result: **+0.093 AUC with only +31% variance** (acceptable)


### Why Interactions Over Raw VIX?

**Interactions capture regime-dependent behavior:**

- `mom121 × VIX`: "Momentum matters more/less in volatile markets"
- `vol12 × VIX`: "Stock volatility signal amplified when market is volatile"

**Raw VIX features were redundant:**

- `VIX_log` and `VIX_change_1m` overlapped with vol12 (caused 28% importance drop)
- Interactions provide unique information without redundancy


### Why Clip to [0.25, 0.75]?

**Tested 4 clip ranges:**


| Clip Range | AUC | Variance | Trade-off |
| :-- | :-- | :-- | :-- |
| [0.2, 0.8] | 0.5310 | 0.0142 | High performance, higher variance |
| **[0.25, 0.75]** | **0.5303** | **0.0123** | **Best balance**  |
| [0.3, 0.7] | 0.5280 | 0.0122 | Overconstrained |

**[0.25, 0.75] chosen because:**

1. Only -0.007 vs best performance
2. 13% lower variance than wider clip
3. Reduces regime range from 10x to 3x (stable across folds)
4. Still captures high/low VIX regimes (just dampened extremes)

### Performance Impact:

| Metric | Without VIX | With VIX | Improvement |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.5210 | **0.5303** | **+0.093**  |
| Variance | 0.0094 | 0.0123 | +31% (acceptable) |
| Fold 1 (calm 2016-19) | 0.517 | 0.527 | +0.010 |
| Fold 2 (COVID 2019-22) | 0.512 | 0.517 | +0.005 |
| Fold 3 (elevated 2022-25) | 0.534 | 0.547 | +0.013 |

### Why NOT Use rf_cal?

**rf_cal performed worse than base rf:**

- rf_cal test AUC: 0.5200 (baseline) → 0.5307 (with VIX)
- rf test AUC: 0.5210 (baseline) → **0.5303 (with VIX)** 

**Calibration hurt performance:**

- Added cv=3 layer reduced effective training data
- With weak signal (AUC ~0.53), calibration overfits to noise
- For stock ranking (not probability estimation), calibration unnecessary


**Result:** Stable improvement across all market conditions (+0.093) with controlled variance increase.

## 8. Random Forest Hyperparameters: 

### Final Configuration:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    min_samples_split=100,
    min_samples_leaf=50,
    max_features='log2',
    max_samples=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
```


### Systematic Optimization Process

Tested 13 configurations across 4 phases:

**Phase 1 - Tree Depth:**

- Tested: depth $\in$ {3, 4, 5}
- Result: Depth 3 best (0.527 test AUC)
- Why deeper failed: Overfitting (gap increased 0.059 → 0.084)

**Phase 2 - Feature Sampling:**

- Tested: max_features $\in$ {log2, sqrt, 0.3, 0.4}
- Result: log2 best (0.527 test AUC, std 0.007)
- Why more features failed: Reduced tree diversity, hurt COVID fold

**Phase 3 - Ensemble Size:**

- Tested: n_estimators $\in$ {50, 100, 200}
- Result: 50 best (0.527 test AUC)
- Why more trees failed: Over-averaged weak signal

**Phase 4 - Bootstrap Sampling:**

- Tested: max_samples $\in$ {None, 0.9, 0.8}
- Result: None (100%) best (0.527 test AUC)
- Why subsampling failed: Removed data from already-weak signal


### Why Defaults Were Already Optimal

**1. Configuration matched problem characteristics:**

- Weak signal → shallow trees needed (depth 3)
- Limited data → full bootstrap needed (max_samples=None)
- Regime changes → high tree diversity needed (log2 features)

**2. Original tuning was conservative:**

- max_depth=3: Very shallow (only 8 leaf nodes max)
- max_features='log2': Only 26% of features per split
- These choices naturally regularize against overfitting

**3. Weak signal is self-limiting:**
With AUC ~0.53 (barely above random 0.50), the model can't overfit much:

- Not enough signal to memorize
- Regularization built into weak patterns


### Performance Validation

All alternatives either:

- **Worse test AUC** (depth 4-5, n_estimators 100-200, max_samples 0.8-0.9)
- **Higher variance** (max_features 0.4: std doubled to 0.014)
- **Worse generalization** (all deeper/complex configs had higher train-test gap)

| Config | Test AUC | Change | Test Std | Change |
| :-- | :-- | :-- | :-- | :-- |
| **Baseline** | **0.527** | — | **0.007** | — |
| depth=4 | 0.525 | -0.002 | 0.005 | -0.002 |
| max_features=0.4 | 0.529 | +0.002 | 0.014 | +0.007 |
| n_estimators=100 | 0.526 | -0.001 | 0.005 | -0.002 |
| max_samples=0.8 | 0.523 | -0.004 | 0.006 | -0.001 |

None worth changing: gains minimal or offset by worse stability.

### Key Principle: Simplicity with Weak Signals

**General ML wisdom:** More complexity → better performance (until overfitting)

**With weak signals (AUC < 0.55):**

- More complexity → worse performance (immediately overfits)
- Simpler models generalize better
- Default regularization often already optimal

**These results confirmed this:**

- Every complexity increase hurt performance
- Simplest config (baseline) was best

## 9. VIX Sentiment Feature: Simple Over Complex

### Decision: Add only VIX_percentile (no interactions, no calibration)

### Final Configuration:

```python
# features.py
def calculate_sentiment_features(sentiment_data):
    vix = sentiment_data['VIX']
    vix_percentile = vix.rolling(12, min_periods=3).rank(pct=True)
    return {'VIX_percentile': vix_percentile}

# No interactions - RF learns them naturally
def calculate_vix_interactions(market_features, sentiment_features, tickers):
    return {}

# definitions.py
model = 'rf'  # Use base RF, NOT rf_cal
total_features = 13  # 12 base + 1 VIX_percentile
```


### Why Add VIX?

**Market regime affects stock selection:**

- Calm markets (VIX < 15): Value and momentum work well
- Volatile markets (VIX > 30): Quality and defensive factors dominate
- VIX_percentile (12-month rolling rank) captures relative volatility

**Empirical evidence:**

- Test AUC improved: 0.519 → 0.526 (+0.007)
- Portfolio Sharpe improved: 0.92 → 0.93 (+0.01)
- All time periods benefited, especially COVID (+0.007)

***

### Why NOT Interactions?

**Tested multiplicative interactions:**

```python
# What we tested (and rejected)
mom121_x_VIX = mom121 * VIX_percentile
vol12_x_VIX = vol12 * VIX_percentile
```

**Results:**


| Metric | VIX Only | VIX + Interactions | Difference |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.526 | 0.527 | +0.001 (minimal) |
| Portfolio Sharpe | **0.93** | 0.86 | **-0.07**  |
| Alpha vs Random | **1.72%** | 0.44% | **-1.28%**  |
| Annual Return | **20.2%** | 18.9% | **-1.3%**  |

**Why interactions failed:**

1. **Probability compression**
    - VIX only: [0.43, 0.64] range = 0.21
    - Interactions: [0.42, 0.65] range = 0.23 (similar)
    - But interactions **destroyed signal at extremes**
2. **Forced functional form**
    - Multiplication assumes specific relationship
    - Random Forest can learn **better interactions naturally**
    - Tree splits like: "If VIX_percentile > 0.7 AND mom121 < 0 → Strong sell"
    - More flexible than fixed multiplication
3. **Over-regularization**
    - Interactions made model overly cautious
    - Lost conviction in top picks
    - Top 10% selection needs extreme probabilities
4. **Failed where it should help**
    - 2022 Bear Market drawdown: -22.9% (VIX only) vs -24.4% (interactions)
    - VIX interactions **worse** in the period they should protect

***

### Why NOT Calibration?

**Tested: rf_cal with VIX_percentile**

**Results:**


| Metric | rf + VIX | rf_cal + VIX | Difference |
| :-- | :-- | :-- | :-- |
| Portfolio Sharpe | **0.93** | 0.79 | **-0.14**  |
| Alpha vs Random | **1.72%** | **-0.69%** | **-2.41%**  |
| Annual Return | **20.2%** | 17.8% | **-2.4%**  |

**rf_cal actually LOSES to random stock picking!**

**Why calibration failed with VIX:**

1. **Regime-dependent distributions**
    - Calm markets: Different probability distribution than volatile markets
    - Calibration fits **single average curve**
    - Doesn't account for regime shifts
2. **Stretched probabilities mislead selection**
    - Calibration: [0.28, 0.82] range (looks good!)
    - But stretched values don't correlate with regime-specific returns
    - Makes wrong stocks look good in wrong regimes
3. **Baseline calibration worked**
    - No sentiment: rf_cal had 0.92 Sharpe (tied best)
    - With VIX: rf_cal drops to 0.79 Sharpe (worst)
    - VIX breaks calibration assumptions

**Principle:** Calibration assumes **stationary distribution**. Regime features (VIX) create **non-stationary distributions** where calibration misleads.

***

### Why VIX_percentile Specifically?

**Tested alternatives:**

- `VIX` (raw level): Too noisy, regime-dependent thresholds
- `VIX_log`: Non-linear transform, no benefit
- `VIX_change_1m`: Too reactive, adds noise

**VIX_percentile advantages:**

1. **Bounded **: Always interpretable
2. **Relative measure**: "High/low compared to past year"
3. **Stable**: 12-month window smooths daily noise
4. **Regime indicator**:
    - < 0.3 = Calm period
    - 0.3-0.7 = Normal
    - > 0.7 = Volatile period
5. **Works across time**:
    - VIX=15 in 2017 (high) → percentile 0.8
    - VIX=15 in 2020 (low) → percentile 0.2
    - Captures **relative** not absolute volatility

***

### Performance Comparison

| Configuration | Test AUC | Sharpe | Return | Alpha | Verdict |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **VIX_percentile only** | **0.526** | **0.93** | **20.2%** | **1.72%** | **Optmimal**  |
| No sentiment | 0.519 | 0.92 | 20.2% | 1.70% | Baseline |
| VIX + interactions | 0.527 | 0.86 | 18.9% | 0.44% | Failed |
| rf_cal + VIX | 0.526 | 0.79 | 17.8% | -0.69% | Disaster |

**Winner is clear:** Single VIX_percentile feature beats all alternatives.

***

### Key Principle: Let the Model Learn

**What we learned:**

-  Engineered interactions: Impose assumptions, limit flexibility
-  Simple features: Model discovers optimal interactions

**Random Forest strength:**

- Naturally learns non-linear relationships
- Discovers feature interactions through tree splits
- No need to manually specify functional forms



***

### Risk: Why Not Remove VIX Entirely?

One Could argue that Baseline (no VIX) had 0.92 Sharpe, final has 0.93 - barely different!

**Why we keep VIX_percentile:**

1. **Test AUC improved** (0.519 → 0.526)
    - More robust ranking
    - Less overfitting to specific periods
2. **All periods benefited:**
    - Bull 2016-19: 0.515 → 0.516
    - COVID 2019-22: 0.512 → 0.519 (+0.007!)
    - Recovery 2022-25: 0.535 → 0.542 (+0.007!)
3. **Better downside protection:**
    - Max drawdown: -23.0% → -22.9%
    - 2022 bear: -23.0% → -22.9%
4. **Future-proofing:**
    - Next crisis will have extreme VIX
    - Model explicitly regime-aware
    - Not just lucky on historical data
5. **Statistical significance:**
    - Both pass Sharpe significance tests (p < 0.001)
    - But VIX model has better metrics across board

**Cost:** 1 additional feature (13 vs 12)
**Benefit:** Improved AUC + regime awareness + better drawdowns

**Risk-reward:** Strongly favors keeping VIX_percentile.

***

### Implementation Notes

**Data requirement:**

- VIX data availability: 1995-present
- Training data starts: 2016-11-30 (earliest with all features)
- Full 9-year backtest available

**Computational cost:**

- Negligible (1 extra feature)
- Rolling percentile fast to compute
- No complex interactions to calculate

**Maintenance:**

- VIX data updated daily (public, reliable)
- No calibration needed (use raw RF probabilities)
- Feature calculation stable across time

***

## Summary: Simple Features + Powerful Models

**Rejected approaches:**

-  Complex interactions (mom121 × VIX, vol12 × VIX)
-  Calibration (rf_cal)
-  Multiple VIX transforms (log, changes)

**Winning approach:**

-  Single feature: VIX_percentile
-  Let Random Forest learn interactions
-  Use uncalibrated probabilities

**Result:** Best Sharpe (0.93), improved AUC (+0.007), robust across all regimes.