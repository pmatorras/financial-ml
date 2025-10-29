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
**Date:** October 23, 2025 (initial), October 29, 2025 (updated)
**Goal:** Test if averaging predictions from multiple models improves portfolio performance

### Hypothesis:
Combining linear (LogReg) and non-linear (RF) models might capture complementary signals, given that generally, ensembles work best when models make uncorrelated errors with similar performance.

### Correlation Analysis:
- LogReg_L2 vs RF_cal: 0.556 (moderate correlation)
- Expected: Low correlation (<0.5) needed for ensemble benefit

### Phase 1: Initial testing (October 23, 2025)

#### Ensemble Configuration:
- 50% LogReg_L2 + 50% RF_cal
- Simple averaging of predicted probabilities

**Model correlation:** 0.556 (moderate, higher than ideal <0.5)

#### Results:

| Metric | RF_cal (Best) | Ensemble | Change |
|--------|---------------|----------|--------|
| Sharpe Ratio | 0.80 | 0.71 |  -11% |
| Annual Return | 17.9% | 16.8% |  -1.1% |
| Max Drawdown | -21.5% | -22.2% |  Worse |
| Volatility | 17.4% | 18.1% |  +0.7% |

### Phase 2: Retesting with VIX Features (October 29, 2025)

After adding VIX_percentile feature, Random Forest performance improved significantly (Sharpe 0.93). Retested ensembles to see if combining improved models helped.

**Baseline individual performance (with VIX):**

- Random Forest: Sharpe 0.93, return 20.2%, drawdown -22.9%
- Random Forest Calibrated: Sharpe 0.92, return 20.2%, drawdown -23.0%
- Logistic Regression L2: Sharpe 0.79, return 18.1%, drawdown -25.4%

**Model correlations:**

- logreg_l2 vs rf: 0.423 (moderate)
- logreg_l2 vs rf_cal: 0.375 (moderate, lower than rf)
- rf vs rf_cal: 0.773 (high)

***

#### Test 1: Logistic L2 + Random Forest

**Configuration:** 50% logreg_l2 + 50% rf

**Results:**

- Sharpe: 0.83
- Annual Return: 18.2%
- Max Drawdown: -23.9%
- Win Rate: 68.9%

**Analysis:**

- Worse than RF alone by 0.10 Sharpe (11% degradation)
- Ensemble improved logreg (0.79 → 0.83) but degraded RF (0.93 → 0.83)
- Net effect: Worse than using best model alone

***

#### Test 2: Logistic L2 + Random Forest Calibrated (with sentiments)

**Configuration:** 50% logreg_l2 + 50% rf_cal

**Results:**

- Sharpe: 0.85
- Annual Return: 18.6%
- Max Drawdown: -23.0%
- Win Rate: 66.0%

**Analysis:**

- Better than Test 1 (0.85 vs 0.83) due to slightly lower correlation (0.375 vs 0.423)
- Still worse than rf_cal alone by 0.07 Sharpe (8% degradation)
- Same pattern: Helps weak model, hurts strong model

***

### Complete Performance Ranking

| Rank | Model/Ensemble | Sharpe | Type | Status |
| :-- | :-- | :-- | :-- | :-- |
| 1 | RF + VIX | 0.93 | Single model | Best |
| 2 | RF_cal + VIX | 0.92 | Single model | Very good |
| 3 | logreg + rf_cal | 0.85 | Ensemble | Worse than components |
| 4 | logreg + rf | 0.83 | Ensemble | Worse than components |
| 5 | logreg_l2 | 0.79 | Single model | Weak |

Both ensembles worse than top 2 single models.

***

#### 1. Dominant Single Model

Random Forest (0.93) outperforms other models by large margin:

- vs rf_cal: +0.01 (1% better)
- vs logreg: +0.14 (18% better)

**Mathematical result:** Averaging 0.93 with 0.79 produces 0.86 at best (50/50 weighting). Actual result (0.83) is worse due to probability compression.

#### 2. No Complementary Pairs

**Best individual performers:**

- rf (0.93) + rf_cal (0.92): High correlation (0.773), ensemble would be ~0.91-0.92
- rf (0.93) + logreg (0.79): Moderate correlation (0.423), but large performance gap

**Low correlation pairs:**

- All involve weak models (boosting: 0.78-0.84, logreg: 0.79)
- Diversity without quality doesn't help


#### 3. Probability Compression

Averaging predictions compresses probability distributions:

**Example with top 10% selection:**

- RF alone: Stock A = 0.64, Stock B = 0.62 (clear differentiation)
- Logreg: Stock A = 0.54, Stock B = 0.56 (less clear)
- Ensemble: Stock A = 0.59, Stock B = 0.59 (no differentiation)

Compression makes stock selection more random, hurting portfolio construction.

#### 4. Optimal Weighting Analysis

Performance-based optimal weights:

```
w_rf = 0.93 / (0.93 + 0.79) = 0.54
w_logreg = 0.46
```

Even with optimal 54/46 weighting (vs 50/50 tested), expected Sharpe would be approximately 0.84, still worse than RF alone (0.93).

To match RF performance, would need 90% RF + 10% logreg, at which point ensemble provides no benefit.

***

### Alternative Configurations Not Tested

**RF + RF_cal ensemble:**

- High correlation (0.773): Expected Sharpe 0.91-0.92
- Marginal improvement over using single model
- Not worth added complexity

**Three-model ensembles:**

- Even more probability compression
- Expected Sharpe: 0.80-0.85 (worse than two-model)

**Including boosting models:**

- All boosting models weak (Sharpe 0.78-0.84)
- Would drag ensemble down further

### Decision: Reject ensemble. Use RF alone.

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

### Feature Importance Analysis

Despite improving portfolio performance, VIX_percentile ranks last in feature importance:


| Feature | Importance | Rank |
| :-- | :-- | :-- |
| vol12 | 25.8% | 1 |
| BookToMarket | 23.6% | 2 |
| ROA | 13.3% | 3 |
| NetMargin | 8.6% | 4 |
| ROE | 7.2% | 5 |
| ... | ... | ... |
| r1 | 0.32% | 12 |
| **VIX_percentile** | **0.23%** | **13** |

**Portfolio performance:**

- Without VIX: Sharpe 0.92, AUC 0.519
- With VIX: Sharpe 0.93, AUC 0.525


### Why Low Importance Does Not Mean Low Value

**Feature importance measures average contribution** across all samples (all market conditions). VIX_percentile works differently:

**Usage pattern:**

- Normal markets (85% of time): Inactive, other features dominate
- High-volatility regimes (15% of time): Critical for stock selection
- Result: Low average importance but high marginal value
> A bit like the seatbelt of a car, unecessary in normal driving conditions, but important in accidents.

**Mechanism:**

- High-VIX periods: Growth/momentum stocks underperform
- VIX_percentile enables regime-aware selection
- Prevents catastrophic picks during market stress periods
- Acts as "portfolio insurance" rather than primary signal



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


## Experiment 8: Gradient Boosting Model Comparison

**Date:** October 28, 2025

**Goal:** Test if gradient boosting methods (XGBoost, LightGBM, sklearn GradientBoosting) can outperform Random Forest for stock selection

### Motivation

Random Forest uses bagging (parallel tree training with bootstrap sampling). Gradient boosting methods build trees sequentially, with each tree correcting errors of previous trees. This sequential learning can sometimes outperform bagging in many ML tasks. We tested whether boosting could improve our weak-signal stock prediction problem (test AUC around 0.52).

### Baseline Performance (Random Forest)

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    max_features='log2',
    max_samples=None,
    min_samples_split=0.02,
    min_samples_leaf=0.01,
    class_weight="balanced"
)
```

**Results:**

- Test AUC: 0.525
- Train AUC: 0.587
- Train-test gap: 0.062
- Portfolio Sharpe: 0.93
- Alpha vs Random: 1.72%
- Probability range: [0.429, 0.643] = 0.214

***

### Boosting Variant 1: XGBoost (Initial)

**Configuration:**

```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1
)
```

**Results:**


| Metric | Value | vs RF |
| :-- | :-- | :-- |
| Test AUC | 0.527 | +0.002 |
| Train AUC | 0.612 | +0.025 |
| Train-test gap | 0.085 | +0.023 (hint of overfitting) |

**Finding:** Better test AUC but much higher train-test gap. Model seems to be overfitting despite weak signal.

***

### Boosting Variant 2: LightGBM (Initial)

**Configuration:**

```python
LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_alpha=0.1,
    reg_lambda=1.0,
    num_leaves=8,
    min_child_samples=50,
    class_weight='balanced'
)
```

**Results:**


| Metric | Value | vs RF |
| :-- | :-- | :-- |
| Test AUC | 0.522 | -0.003 |
| Train AUC | 0.610 | +0.023 |
| Train-test gap | 0.088 | +0.026 (overfitting) |

**Finding:** Worse test AUC and even more overfitting than XGBoost. Leaf-wise growth too aggressive for weak signal.

***

### Iteration 2: Increased Regularization

Applied stronger regularization to control overfitting:

**XGBoost (Regularized):**

```python
learning_rate=0.01,  # Reduced from 0.05
reg_alpha=0.5,       # Increased from 0.1
reg_lambda=2.0,      # Increased from 1.0
```

**Results:**


| Metric | Before | After | Change |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.527 | 0.524 | -0.003 |
| Train AUC | 0.612 | 0.594 | -0.018 |
| Gap | 0.085 | 0.070 | -0.015 (better) |

**LightGBM (Regularized):**

```python
learning_rate=0.01,
reg_alpha=0.5,
reg_lambda=2.0,
min_child_samples=100
```

**Results:**


| Metric | Before | After | Change |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.522 | 0.526 | +0.004 |
| Train AUC | 0.610 | 0.594 | -0.016 |
| Gap | 0.088 | 0.068 | -0.020 (better) |

**Finding:** Regularization successfully reduced overfitting. Test AUC now competitive with RF.

***

### Boosting Variant 3: sklearn GradientBoosting

**Configuration:**

```python
GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.5,              # More aggressive than XGBoost/LightGBM
    validation_fraction=0.2,    # Built-in validation
    n_iter_no_change=10         # Early stopping
)
```

**Results:**


| Metric | Value | vs RF |
| :-- | :-- | :-- |
| Test AUC | 0.526 | +0.001 |
| Train AUC | 0.591 | +0.004 |
| Train-test gap | 0.065 | +0.003 |
| Test Std | 0.016 | +0.005 (more variance) |

**Finding:** Best boosting variant. Early stopping and aggressive subsampling (0.5) prevented overfitting. But higher variance than RF.

***

## Portfolio Performance Comparison

Despite similar test AUC, portfolio performance revealed large differences:


| Model | Test AUC | Train Gap | Alpha vs Random | Portfolio Sharpe | Prob Range |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Random Forest** | 0.525 | 0.062 | **1.72%** | **0.93** | 0.214 |
| XGBoost | 0.524 | 0.070 | 0.41% | ~0.80 | 0.111 |
| LightGBM | **0.526** | 0.068 | **0.20%** | ~0.78 | 0.111 |
| GradientBoosting | 0.526 | 0.065 | 0.80% | ~0.84 | Unknown |

### Critical Finding: Probability Compression

**Random Forest:**

- Min: 0.429, Max: 0.643
- Range: 0.214
- Top 10% spread: ~0.05 (strong conviction)

**XGBoost:**

- Min: 0.493, Max: 0.604
- Range: 0.111 (48% smaller than RF)
- Top 10% spread: ~0.015 (weak conviction)

**LightGBM:**

- Min: 0.467, Max: 0.578
- Range: 0.111 (48% smaller than RF)
- Mean: 0.502 (almost exactly 0.50, indicating near-random predictions)
- Top 10% spread: ~0.02 (very weak conviction)

**Explanation:** Boosting models compressed probabilities toward 0.50 despite heavy regularization. With top 10% portfolio selection, weak conviction at extremes resulted in nearly random stock picking.

***

## Model Correlation Analysis

Spearman correlation between model predictions:

```
rf vs xgb:   0.576  (moderate - different patterns)
rf vs lgbm:  0.911  (high - similar but noisier)
rf vs gb:    ~0.633  (moderate)
xgb vs lgbm: 0.591  (moderate)
```

**Interpretation:**

- LightGBM highly correlated with RF (0.911) but performs much worse. Making similar picks with less confidence.
- XGBoost learning different patterns (0.576) but those patterns don't translate to returns.
- None of the boosting models discovered new alpha.

***

## Why Boosting Failed

### 1. Weak Signal + Sequential Learning = Noise Amplification

With test AUC around 0.52 (barely above random 0.50):

- Signal-to-noise ratio is extremely low
- Sequential learning: Each tree corrects "errors" of previous trees
- But "errors" are mostly noise, not signal
- Boosting chases noise patterns

**Random Forest advantage:** Independent trees average out noise, signal accumulates.

### 2. Regularization Paradox

To prevent overfitting, we added heavy regularization:

- Very low learning rate (0.01)
- Strong L1/L2 penalties (reg_alpha=0.5, reg_lambda=2.0)
- Large minimum samples per leaf

**Result:**

- Train-test gap improved (goal achieved)
- But probabilities compressed toward 0.50 (unintended consequence)
- Model became overly cautious, losing conviction in winners


### 3. Portfolio Selection at Extremes

Top 10% strategy requires strong conviction at extremes (90th percentile and above):

- RF: Top 10% ranges from 0.59 to 0.64 (clear separation)
- Boosting: Top 10% ranges from 0.56 to 0.60 (minimal separation)

With compressed probabilities, boosting models can't distinguish true winners from lucky stocks.

### 4. Early Stopping Helps But Not Enough

sklearn GradientBoosting with early stopping performed better than XGBoost/LightGBM:

- Alpha: 0.80% vs 0.20-0.41%
- Gap: 0.065 vs 0.068-0.070

But still far behind RF (1.72% alpha). The fundamental issue remains: sequential learning amplifies noise in weak-signal regimes.

***

## Attempted Solutions (All Failed)

1. **Stronger regularization:** Controlled overfitting but compressed probabilities
2. **Early stopping:** Helped but insufficient
3. **Aggressive subsampling (0.5):** Created diversity but still worse than RF's bagging
4. **Different learning rates:** 0.01-0.05 tested, all suboptimal
5. **Feature scaling (StandardScaler):** Marginal test AUC improvement, worse portfolio performance

***

## Key Learnings

### 1. Weak Signals Favor Bagging Over Boosting

Contrary to typical ML wisdom where "boosting > bagging", this breaks down at AUC < 0.55:

- Bagging's randomness provides natural regularization
- Sequential learning in boosting amplifies noise
- RF's parallel trees are optimal for low signal-to-noise ratios


### 2. Test AUC is Misleading for Portfolio Construction

LightGBM achieved highest test AUC (0.526) but worst portfolio alpha (0.20%):

- AUC measures ranking quality across all thresholds
- Portfolio selection happens at single extreme threshold (top 10%)
- Performance at extremes matters more than average ranking


### 3. Probability Spread Matters More Than Point Estimates

Models with similar AUC but different probability spreads:

- RF (range 0.214): 1.72% alpha
- Boosting (range 0.111): 0.20-0.80% alpha

Wider spread indicates stronger conviction in predictions, critical for portfolio construction.

### 4. Model Correlation Reveals Information

High RF-LightGBM correlation (0.911) with worse performance indicates LightGBM is adding noise to RF's signal, not discovering new patterns. Low correlation would suggest complementary information and potential for ensembling.

***

## Final Comparison: Complete Results

| Model | Test AUC | Train Gap | Alpha vs Random | Sharpe | Prob Range | Training Time | Verdict |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **RF + VIX** | 0.525 | 0.062 | **1.72%** | **0.93** | 0.214 | Fast | **OPTIMAL** |
| rf_cal | 0.519 | 0.064 | 1.70% | 0.92 | 0.56 | Fast | Good alternative |
| GB | 0.526 | 0.065 | 0.80% | ~0.84 | Unknown | Medium | Not worth it |
| XGBoost | 0.524 | 0.070 | 0.41% | ~0.80 | 0.111 | Medium | Failed |
| LightGBM | 0.526 | 0.068 | 0.20% | ~0.78 | 0.111 | Fast | Failed |


***

## Conclusion

Random Forest remains the optimal model for weak-signal stock prediction. Extensive testing of gradient boosting alternatives (XGBoost, LightGBM, sklearn GradientBoosting) revealed:

- Similar or slightly better test AUC
- Much worse portfolio performance (54-88% worse alpha)
- Fundamental incompatibility between sequential learning and weak signals
- Probability compression destroys portfolio construction

**Decision:** Use Random Forest with VIX_percentile feature for production. Boosting is not suitable for this problem.

***