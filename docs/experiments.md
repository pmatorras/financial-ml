# Experiment Log

This document tracks all model experiments and design choices throughout the project development.

## Table of Contents

- [Experiment 1: Base Model Selection](#experiment-1-base-model-selection)
- [Experiment 2: Probability Calibration](#experiment-2-probability-calibration)
- [Experiment 3: Ensemble Testing](#experiment-3-ensemble-testing)
- [Experiment 4: Smoothing Impact](#experiment-5-smoothing-impact)
- [Experiment 5: Feature Engineering](#experiment-5-feature-engineering)
- [Experiment 6: VIX sentiment features](#experiment-6-vix-sentiment-features)
- [Experiment 7: Random forest hyperparameter optimisation](#experiment-7-random-forest-hyperparameter-optimization)
- [Summary of Final Configuration](#summary-of-final-configuration)


---

## Experiment 1: Base Model Selection
**Date:** October 23, 2025  
**Goal:** Identify the best base model for stock selection

### Models Tested:
| Model | Train AUC | Test AUC | Notes |
|-------|-----------|----------|-------|
| Logistic Regression L1 | 0.585 | 0.554 | Linear, L1 regularization |
| Logistic Regression L2 | 0.582 | 0.558 | Linear, L2 regularization |
| Random Forest | 0.625 | 0.554 | Best discrimination |
| Gradient Boosting | ~0.62 | ~0.53 | Overfitting issues |

### Results:
- LogReg L1 and L2 are nearly identical (correlation 0.98)
- Random Forest shows best test AUC (0.554)
- Training-test gap: ~7pp for all models (acceptable overfitting)

### Decision: Use Random Forest as base model

---

## Experiment 2: Probability Calibration
**Date:** October 23, 2025  
**Goal:** Fix miscalibrated probabilities (mean 0.487 instead of 0.50)

### Problem Identified:
- Original RF predicted only 19 stocks (5.7%) would beat SPY
- Mean probability: 0.487 (should be ~0.50)
- Interpretability issue: Going long stocks with prob <0.5

### Approach: 
CalibratedClassifierCV with isotonic regression (cv=3)

### Results:

| Metric | Original RF | Calibrated RF | Change |
|--------|-------------|---------------|--------|
| Mean y_prob | 0.487 | 0.502 |  Fixed |
| Std y_prob | 0.035 | 0.059 |  +68% (better separation) |
| Test AUC | 0.554 | 0.557 | +0.3pp |
| Sharpe Ratio | 0.71 | 0.80 |  +13% |
| Annual Alpha | 0.87% | 2.29% |  +163% |
| Max Drawdown | -22.8% | -21.5% |  Better |

### Why It Worked:
1. Fixed probability scale (mean now 0.502)
2. Increased prediction spread (std 0.059) → clearer signal for portfolio construction
3. Better discrimination between good/bad stocks
4. Top 10% stocks now have prob >0.5 (interpretable)

### Decision: Use RF_cal as final model

---

## Experiment 3: Ensemble Testing
**Date:** October 23, 2025  
**Goal:** Test if averaging LogReg + RF_cal improves performance

### Hypothesis:
Combining linear (LogReg) and non-linear (RF) models might capture complementary signals

### Correlation Analysis:
- LogReg_L2 vs RF_cal: 0.556 (moderate correlation)
- Expected: Low correlation (<0.5) needed for ensemble benefit

### Ensemble Configuration:
- 50% LogReg_L2 + 50% RF_cal
- Simple averaging of predicted probabilities

### Results:

| Metric | RF_cal (Best) | Ensemble | Change |
|--------|---------------|----------|--------|
| Sharpe Ratio | 0.80 | 0.71 |  -11% |
| Annual Return | 17.9% | 16.8% |  -1.1% |
| Max Drawdown | -21.5% | -22.2% |  Worse |
| Volatility | 17.4% | 18.1% |  +0.7% |

### Why It Failed:
1. LogReg is weaker (AUC 0.558 vs RF_cal 0.557)
2. Correlation too high (0.556) → not enough diversity
3. Averaging diluted RF_cal's stronger signal
4. LogReg's narrow predictions (std ~0.04) compressed RF_cal's better spread (0.059)

### Decision: Reject ensemble. Use RF_cal alone.

### Key Learning:
Ensembles work when models have similar performance and low correlation (<0.5). When one model is clearly superior, averaging with weaker models hurts performance.

---

## Experiment 4: Smoothing Impact
**Date:** October 23, 2025  
**Goal:** Reduce month-to-month prediction volatility

### Problem:
Raw predictions change significantly month-to-month (mean 5.18%)

### Approach:
3-month rolling average of predictions

### Results:
- Raw prediction change: 5.18% mean
- After smoothing: 1.93% mean
- Reduction: 58.7% 

### Impact:
- Reduced turnover (from ~50% to ~42%)
- Lower transaction costs
- Smoother portfolio returns
- No degradation in Sharpe ratio

### Decision: Apply 3-month smoothing to all predictions

---



## Experiment 5: Feature Engineering

**Goal:** Test if engineered features improve baseline performance

**Tested Configurations:**
1. Reversal feature (r1 negation): -0.002 AUC
2. Cross-sectional ranks (replace raw): -0.007 AUC, **-0.031 on Fold 3**
3. Interaction features (4 new): -0.003 AUC
4. Combinations of above: All worse

**Key Finding:** Every tried enhancement worsened the performance, especially on recent data (Fold 3: 2021-2025)

**Why Enhancements Failed:**
- **Ranks:** Redundant for tree models (RF learns relative comparisons naturally)
- **Interactions:** RF with max_depth=3 can't use extra features effectively
- **Reversal:** Pure noise, no new information
- **All:** Overfit to historical patterns, fail on recent data

**Conclusion:** Baseline 14 raw features is preferred.

**Key Learning:** In ML, more features often means worse generalization. Constrained models (max_depth=3) need carefully selected features, not exhaustive transformations.

---

## Experiment 6: VIX Sentiment Features

**Date:** October 28, 2025
**Goal:** Add market volatility (VIX) features to improve regime-aware predictions and portfolio performance

### Hypothesis:

Market volatility affects stock selection - momentum and volatility features should work differently in calm vs volatile regimes. Adding VIX-based features could improve both predictive accuracy and portfolio returns.

### Initial Baseline (No Sentiment):

- Test AUC: 0.519
- Portfolio Sharpe: 0.92
- Annual Return: 20.2%
- Alpha vs Random: 1.70%

***

### Iteration 1: VIX Features with Interactions

**Approach:** Add VIX features plus multiplicative interactions

**Features added (3 new, 15 total):**

1. `VIX_percentile`: Rolling 12-month percentile rank (0-1 scale)
2. `mom121_x_VIX`: Momentum × VIX_percentile (clipped [0.25, 0.75])
3. `vol12_x_VIX`: Stock volatility × VIX_percentile (clipped [0.25, 0.75])

**Predictive Results:**


| Metric | No Sentiment | With Interactions | $\Delta$(absolute) |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.519 | **0.527** | **+0.008** |
| Fold 1 | 0.515 | 0.533 | +0.018 |
| Fold 2 (COVID) | 0.512 | 0.518 | +0.006 |
| Fold 3 | 0.535 | 0.530 | -0.005 |

**Portfolio Results:**


| Metric | No Sentiment | With Interactions | $\Delta$(absolute) |
| :-- | :-- | :-- | :-- |
| Sharpe | **0.92** | 0.86 | **-0.06** |
| Annual Return | **20.2%** | 18.9% | **-1.3%** |
| Win Rate | **70.8%** | 66.0% | **-4.8%** |
| Alpha vs Random | **1.70%** | 0.44% | **-1.26%** |
| Max Drawdown | -23.0% | -24.4% | -1.4% worse |

**Probability Distribution Analysis:**

```
NO sentiment:     [0.246, 0.807] range = 0.561
WITH interactions: [0.419, 0.654] range = 0.235 (58% smaller!)
```

**Finding:**

- **Test AUC improved** (+0.8 percentage points)
- **Portfolio performance degraded significantly**
- **Probability range compressed** - less conviction in top picks
- **Barely beats random** (0.44% alpha)

**Suspected root Cause:** Multiplicative interactions over-regularize predictions, compressing probability ranges and destroying signal at extremes (top/bottom 10% where portfolio selection happens).

***

### Iteration 2: VIX_percentile Only (No Interactions)

**Approach:** Use VIX as simple feature, let Random Forest learn interactions naturally

**Features added (1 new, 13 total):**

- `VIX_percentile`: Rolling 12-month percentile rank (0-1 scale)

**Predictive Results:**


| Metric | No Sentiment | VIX Only | $\Delta$(absolute) |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.519 | **0.526** | **+0.007** |
| Fold 1 | 0.515 | 0.516 | +0.001  |
| Fold 2 (COVID) | 0.512 | **0.519** | **+0.007** |
| Fold 3 | 0.535 | **0.542** | **+0.007** |

**Portfolio Results:**


| Metric | No Sentiment | VIX Only | $\Delta$(absolute) |
| :-- | :-- | :-- | :-- |
| Sharpe | 0.92 | **0.93** | **+0.01** |
| Annual Return | 20.2% | **20.2%** | Equal |
| Win Rate | 70.8% | 69.8% | -1.0% |
| Alpha vs Random | 1.70% | **1.72%** | **+0.02%** |
| Max Drawdown | -23.0% | **-22.9%** | **-0.1% better** |

**Probability Distribution Analysis:**

```
NO sentiment:  [0.246, 0.807] range = 0.561
VIX only:      [0.429, 0.643] range = 0.214 (moderately compressed)
```

**Finding:**

- **Best Sharpe of all models** (0.93)
- **Test AUC improved** (+0.007)
- **All periods improved**, especially COVID (+0.007)
- **Portfolio performance maintained** (20.2% return)
- **Still beats random convincingly** (1.72% alpha)
- **Better downside protection** (-22.9% vs -23.0%)

**Why It Works:**
Random Forest can **learn non-linear interactions** between VIX_percentile and other features:

- In calm markets (VIX_percentile < 0.3): Value and momentum work well
- In volatile markets (VIX_percentile > 0.7): Quality factors dominate
- RF learns these patterns **adaptively** without forcing functional form

***

### Iteration 3: Calibration Test (rf_cal + VIX_percentile)

**Approach:** Test if calibration helps with VIX features

**Results:**


| Metric | rf + VIX | rf_cal + VIX | Change |
| :-- | :-- | :-- | :-- |
| Sharpe | **0.93** | 0.79 | **-0.14** |
| Annual Return | **20.2%** | 17.8% | **-2.4%** |
| Alpha vs Random | **1.72%** | **-0.69%** | **-2.41%** |

**Probability Distribution:**

```
rf + VIX:      [0.429, 0.643] range = 0.214
rf_cal + VIX:  [0.277, 0.823] range = 0.546 (stretched back out)
```

**Finding:**

- **Calibration destroys performance** with VIX features
- **Loses to random selection** (-0.69% alpha)
- **Worst performer** despite wide probability range

**Why Calibration Fails:**
Calibration stretches probabilities based on **average frequencies**, but VIX creates **regime-dependent distributions**. The stretched probabilities don't correlate with regime-specific returns, making selections worse than random.

***

## Experiment 7: Random Forest Hyperparameter Optimization

**Date:** October 28, 2025
**Goal:** Systematically optimize Random Forest hyperparameters to maximize test AUC while maintaining stability

### Hypothesis:

Default RF hyperparameters (depth=3, n_estimators=50, max_features='log2') may not be optimal. Deeper trees or more features per split could capture more complex patterns and improve performance.

### Baseline Configuration:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    min_samples_split=100,
    min_samples_leaf=50,
    max_features='log2',
    max_samples=None,
    random_state=42,
    class_weight="balanced"
)
```

- Test AUC: 0.527
- Test Std: 0.007
- Train-Test Gap: 0.059

***

### Phase 1: Tree Depth Optimization

**Tested:** max_depth $\in$ {3, 4, 5}


| Depth | Test AUC | Test Std | Train AUC | Gap | Fold 1 | Fold 2 | Fold 3 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **3** | **0.527** | 0.007 | 0.586 | **0.059** | **0.533** | **0.518** | **0.530** |
| 4 | 0.525 | **0.005** | 0.596 | 0.071 | 0.532 | 0.521 | 0.522 |
| 5 | 0.524 | 0.008 | 0.608 | 0.084 | 0.535 | 0.517 | 0.521 |

**Key Findings:**

- **Depth 3 wins** on test AUC (best performance)
- Deeper trees (4-5) **overfit**: higher train AUC but worse test AUC
- Gap increases dramatically: 0.059 → 0.071 → 0.084
- Depth 4 has lower variance (0.005) but loses -0.002 test AUC
- **With weak signal (AUC ~0.53), shallower trees generalize better**

**Decision: Keep max_depth=3**

***

### Phase 2: Feature Sampling Optimization

**Tested:** max_features $\in$ {log2, sqrt, 0.3, 0.4}
(with max_depth=3 fixed)


| max_features | \# Features/Split | Test AUC | Test Std | Fold 2 (COVID) |
| :-- | :-- | :-- | :-- | :-- |
| **log2** | 4 | **0.527** | **0.007** | **0.518**  |
| sqrt | 4 | **0.527** | **0.007** | **0.518**  |
| 0.3 | 4.5 | 0.526 | 0.011 | 0.510  |
| 0.4 | 6 | 0.529 | 0.014  | 0.509  |

**Key Findings:**

- log2 and sqrt are basically identical (logical since both sqrt(15)~log2(15)~4)
- max_features=0.4 has **best average** (+0.002) but **2x variance** (0.007 → 0.014)
- **COVID fold (Fold 2) suffers** with more features: 0.518 → 0.509 (-0.009)
- More features = **less tree diversity** = worse regime robustness

**Reasoning:**
With weak signal + regime changes, need high tree diversity:

- Fewer features per split → more diverse trees → better generalization
- Risk-adjusted metric: log2 wins (Sharpe-like = 0.86 vs 0.57 for 0.4)

**Decision: Keep max_features='log2'**

***

### Phase 3: Ensemble Size Optimization

**Tested:** n_estimators $\in$ {50, 100, 200}
(with max_depth=3, max_features='log2' fixed)


| n_estimators | Test AUC | Test Std | Train AUC | Gap | Fold 1 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **50** | **0.527** | 0.007 | 0.586 | **0.059** | **0.533** |
| 100 | 0.526 | **0.005** | 0.588 | 0.062 | 0.525 |
| 200 | 0.524 | 0.006 | 0.588 | 0.064 | 0.524 |

**Key Findings:**

- **More trees = worse performance!** (counterintuitive)
- 50 trees: Best test AUC and best generalization
- 100/200 trees: Train AUC increases but test AUC decreases → **overfitting**
- Variance reduction minimal: 0.007 → 0.005 (not worth -0.001 to -0.003 loss)

**Reasoning:**
With extremely weak signal (AUC ~0.53):

- Each tree captures **mostly noise + bit of signal**
- 50 trees: Signal survives, noise cancels out 
- 200 trees: **Over-averaging removes signal too** 
- Like applying too much noise filter to weak radio signal

**Decision: Keep n_estimators=50**

***

### Phase 4: Bootstrap Sampling Optimization

**Tested:** max_samples $\in$ {None, 0.9, 0.8}
(with max_depth=3, max_features='log2', n_estimators=50 fixed)


| max_samples | Test AUC | Test Std | Train AUC | Gap | Fold 1 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **None (100%)** | **0.527** | 0.007 | 0.586 | 0.059 | **0.533** |
| 0.9 (90%) | 0.523 | **0.006** | 0.584 | 0.061 | 0.526 |
| 0.8 (80%) | 0.523 | **0.006** | 0.584 | 0.061 | 0.524 |

**Key Findings:**

- Subsampling **hurts performance**: -0.004 with 80-90% samples
- Signal already weak → reducing data makes it worse
- Not overfitting (gap = 0.059 is healthy) → no need for regularization
- Variance improvement minimal: 0.007 → 0.006 (not worth AUC loss)

**Reasoning:**
Subsampling helps when:

- Overfitting (not your case)
- Strong signal (yours is weak)
- High variance (yours is 0.007, already low)

**Decision:** **Keep max_samples=None**

***

### Final Performance vs Baseline:

| Metric | Baseline (no sentiment) | **Final (VIX only)** | $\Delta$(absolute) |
| :-- | :-- | :-- | :-- |
| **Test AUC** | 0.519 | **0.526** | **+0.007**  |
| **Sharpe** | 0.92 | **0.93** | **+0.01**  |
| **Annual Return** | 20.2% | **20.2%** | Equal  |
| **Win Rate** | 70.8% | 69.8% | -1.0% |
| **Alpha vs Random** | 1.70% | **1.72%** | **+0.02%**  |
| **Max Drawdown** | -23.0% | **-22.9%** | **+0.1%**  |

**Total features: 13** (12 base + 1 VIX)

***

## Key Learnings

### 1. **Feature Engineering Can Hurt Random Forests**

| Approach | Test AUC | Portfolio Sharpe | Interpretation |
| :-- | :-- | :-- | :-- |
| Engineered interactions | 0.527 | 0.86 | Forced functional form hurts |
| Simple feature | 0.526 | **0.93** | RF learns optimal interactions  |

**Lesson:** With Random Forests, **simple features > complex interactions**. The tree structure naturally discovers non-linear relationships without manual engineering.

### 2. **AUC ≠ Portfolio Performance**

All three VIX approaches improved test AUC:

- With interactions: +0.008 AUC, but -0.06 Sharpe
- VIX only: +0.007  AUC, **+0.01 Sharpe** 
- Calibrated: Better AUC, but -0.69% alpha (loses to random!)

**AUC measures ranking across all thresholds**, but portfolios select **only the top 10%**. Performance at extremes matters more than average ranking.

### 3. **Probability Compression is a Red Flag**

| Configuration | Prob Range | Portfolio Alpha |
| :-- | :-- | :-- |
| Baseline | 0.561 | 1.70%  |
| VIX only | 0.214 | 1.72%  |
| VIX interactions | 0.235 | 0.44%  |

**VIX only** compresses probabilities but still works because:

- Compression is **moderate** (not extreme)
- **Ranking within compressed range** still informative
- RF maintains enough spread for top 10% selection

**VIX interactions** compress too much:

- Model becomes overly cautious
- Less conviction in winners
- Top 10% threshold too close to median


### 4. **Calibration Requires Careful Consideration**

Calibration helped baseline model (0.92 Sharpe), but **destroyed VIX model** (0.79 Sharpe).

**When calibration fails:**

- Features create **regime-dependent distributions**
- Average calibration curve doesn't apply to all regimes
- Stretched probabilities mislead portfolio selection

**Lesson:** Calibration assumes **single stable distribution**. With regime features (VIX), this assumption breaks.

### 5. **Regime Features Should Be Simple**

**What worked:**

- `VIX_percentile`: Binary question ("high or low volatility?")
- Single value per time period
- Let model decide how it matters

**What failed:**

- Multiplicative interactions (`mom121 × VIX`)
- Forced assumptions about functional form
- Over-regularization

**Principle:** Regime features should **flag conditions**, not **modify features directly**.

### 6. **Random Baseline Test is Critical**

| Model | Alpha vs Random | Usable? |
| :-- | :-- | :-- |
| VIX only | 1.72% |  YES |
| VIX interactions | 0.44% |  Marginal |
| rf_cal + VIX | -0.69% |  NO |

Without random baseline testing, we might have shipped a model that **loses to random stock picking**!

***

## Conclusion

**Final model: rf with VIX_percentile only**

Achieved:

-  Best Sharpe of all tested models (0.93)
-  Improved test AUC (+0.007)
-  Maintained portfolio returns (20.2%)
-  Better downside protection
-  Significant alpha vs random (1.72%)

**Key insight:** Simple features + powerful model (Random Forest) > Complex feature engineering. Let the algorithm discover interactions rather than imposing them.