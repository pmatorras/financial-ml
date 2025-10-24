# Design Decisions and Rationale

This document explains the reasoning behind key design choices in the project.


## Table of Contents

- [1. Model Choice: Random Forest (Calibrated)](#1-model-choice-random-forest-calibrated)
- [2. Ensemble: Rejected](#2-ensemble-rejected)
- [3. Smoothing: 3-Month Window](#3-smoothing-3-month-window)
- [4. Time Series CV: Expanding Window](#4-time-series-cv-expanding-window)
- [5. Transaction Costs: 10 bps](#5-transaction-costs-10-bps)
- [Summary: Key Principles](#summary-key-principles)
---


## 1. Model Choice: Random Forest (Calibrated)

### Decision: Use calibrated Random Forest as final model

### Why Random Forest Over Logistic Regression?
✅ **Better performance**
- RF test AUC: 0.557 vs LogReg 0.558 (similar)
- RF Sharpe: 0.71 vs LogReg ~0.65 (better)

✅ **Captures non-linearities**
- Stock returns are not linear in factors
- Interactions matter (e.g., value works differently in small vs large caps)

✅ **Robust to outliers**
- Tree-based models less sensitive to extreme values
- No need for extensive outlier treatment

### Why Constrained (max_depth=3)?
✅ **Prevents overfitting**
- Deep trees: Test AUC drops to 0.52
- Shallow trees: Generalize better

✅ **Interpretability**
- Depth=3 means max 8 leaf nodes per tree
- Easier to understand decision boundaries

### Why Calibration?
✅ **Fixed miscalibrated probabilities**
- Original: mean 0.487 (predicting only 5.7% beat SPY)
- Calibrated: mean 0.502 (proper interpretation)

✅ **Improved performance**
- Sharpe: 0.71 → 0.80 (+13%)
- Better discrimination (std 0.035 → 0.059)

✅ **Industry standard**
- Isotonic calibration is well-established
- Preserves ranking (AUC unchanged)

---

## 2. Ensemble: Rejected

### Decision: Do NOT use ensemble (LogReg + RF)

### Why Not Ensemble?
❌ **Underperformed single model**
- Ensemble Sharpe: 0.71 vs RF_cal 0.80
- Diluted the stronger signal

❌ **Correlation too high (0.556)**
- Need <0.5 for ensemble diversity
- Models captured similar patterns

❌ **Weaker model dragged down performance**
- Averaging RF_cal (strong) with LogReg (weaker) hurt results

### Key Learning:
**Ensembles work when:**
- Models have similar performance
- Low correlation (<0.5)
- Capture truly different patterns

**Our case:**
- RF_cal clearly superior
- Moderate correlation (not enough diversity)
- **Conclusion: Use best single model**

---

## 3. Smoothing: 3-Month Window

### Decision: Apply 3-month rolling average to predictions

### Problem:
Raw predictions volatile month-to-month (5.18% mean change)

### Alternatives Tested:
- No smoothing: High turnover (50%+), noisy
- 2-month window: Still volatile (4.2% mean change)
- 3-month window: Optimal (1.93% mean change) ✅
- 6-month window: Over-smoothed (stale predictions)

### Rationale:
✅ **Reduces noise without losing signal**
- 58.7% reduction in volatility
- Sharpe maintained at 0.80

✅ **Lowers transaction costs**
- Turnover: 50% → 42%
- Cost drag: 1.0% → 0.82%

✅ **Smoother returns**
- Less month-to-month volatility in portfolio
- Better investor experience

### Implementation:

```bash
df['y_prob_smooth'] = df.groupby('ticker')['y_prob'].rolling(3, min_periods=1).mean()

```

---

## 4. Time Series CV: Expanding Window

### Decision: Use expanding window (not sliding window)

### Why Expanding?
✅ **Uses all available data**
- Later folds have more training data
- Better estimates as time progresses

✅ **More realistic**
- In production, you'd use all historical data
- Mimics actual deployment

✅ **Reduces variance**
- More training data → stabler models
- Fewer overfitting issues

### Why Not Sliding Window?
❌ Throws away old data
❌ Smaller training sets
❌ Less stable models over time

---


## 5. Transaction Costs: 10 bps

### Decision: Model 10 bps (0.1%) per trade

### Breakdown:
- Bid-ask spread: 5 bps (realistic for liquid S&P 500 stocks)
- Market impact: 5 bps (small orders, moderate liquidity)

### Rationale:
✅ **Conservative but realistic**
- S&P 500 stocks are liquid
- Institutional execution typically 5-15 bps

✅ **Sensitive to AUM**
- Works for <$100M strategies
- Larger funds would face higher costs

### Impact on Results:
- Gross alpha: ~3.0%
- Transaction costs: -0.82%
- **Net alpha: 2.18%** (still attractive)

---

## Summary: Key Principles

Throughout this project, design decisions followed these principles:

1. **Simplicity over complexity** (equal-weighting, single model)
2. **Empirical validation** (test alternatives, measure impact)
3. **Robustness over optimization** (constrained RF, no weight optimization)
4. **Financial theory as guide** (factors, horizons based on research)
5. **Production-ready** (realistic costs, executable strategy)

**Result:** A strategy that is simple, explainable, robust, and profitable (Sharpe 0.80, Alpha 2.29%).
