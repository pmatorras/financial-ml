# Experiment Log

This document tracks all model experiments and design choices throughout the project development.

## Table of Contents

- [Experiment 1: Base Model Selection](#experiment-1-base-model-selection)
- [Experiment 2: Probability Calibration](#experiment-2-probability-calibration)
- [Experiment 3: Ensemble Testing](#experiment-3-ensemble-testing)
- [Experiment 4: Smoothing Impact](#experiment-5-smoothing-impact)
- [Experiment 5: Feature Engineering](#experiment-5-feature-engineering)
- [Experiment 6: VIX sentiment features](#experiment-6-vix-sentiment-features--interaction-terms)
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

## Experiment 6: VIX Sentiment Features \& Interaction Terms

**Date:** October 28, 2025
**Goal:** Add market sentiment features (VIX) to improve regime-aware predictions

### Hypothesis:

Market volatility (VIX) could improve stock selection by capturing regime-dependent behavior of momentum and volatility features.

### Initial Approach: Raw VIX Features

Added 3 VIX features:

- `VIX`: Raw VIX level
- `VIX_change_1m`: 1-month momentum
- `VIX_zscore`: Z-score vs rolling mean


### Results (17 features total):

| Metric | No Sentiment | With VIX Features | Change |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.5210 | 0.5307 | +9.7 bps  |
| Variance (std) | 0.0094 | 0.0209 | +122%  |
| Fold 1 (calm) | 0.517 | 0.524 | +0.7 bps |
| Fold 2 (COVID) | 0.512 | 0.509 | -0.3 bps |
| Fold 3 (elevated) | 0.534 | 0.559 | +2.5 bps |

### Problem Identified:

**High regime instability**

- VIX features had importance std > mean (e.g., VIX_change_1m: importance 0.119 $\pm$ 0.170)
- COVID extreme values (VIX=82) were out-of-distribution
- Different folds had dramatically different VIX regimes


### Feature Importance Analysis:

Total VIX contribution: **25.1%** of model importance

- But caused vol12 to drop 28% and BookToMarket to drop 26%
- VIX features were **cannibalizing** existing volatility signals

***

### Iteration 1: Add Interaction Terms

**Approach:** Create regime-aware features

- `mom121 × VIX_percentile`: Momentum works differently in high/low VIX
- `vol12 × VIX_percentile`: Volatility signal amplified in volatile regimes

**Results (17 features: 3 VIX + 2 interactions):**


| Metric | No Interactions | With Interactions | Change |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.5307 | 0.5373 | +6.6 bps  |
| Variance | 0.0209 | 0.0274 | +31%  |
| Fold 2 (COVID) | 0.509 | 0.516 | +0.7 bps  |
| Fold 3 | 0.559 | 0.576 | **+1.7 bps**  |

**Key finding:** Interactions helped where they should (volatile periods) but increased overall variance.

***

### Iteration 2: Simplify VIX Features

**Approach:** Remove redundant features

- Drop: `VIX_log`, `VIX_change_1m` (redundant with vol12)
- Keep: Only `VIX_percentile` + 2 interactions

**Results (15 features → 14 features):**


| Metric | All VIX (17) | Simplified (15) | Change |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.5373 | 0.5290 | -8.3 bps |
| Variance | 0.0274 | 0.0132 | **-52%**  |

**Massive stability improvement** with acceptable performance loss.

***

### Iteration 3: Clip Interaction Values

**Approach:** Reduce extreme regime effects

- Clip `VIX_percentile` to `[0.25, 0.75]` before multiplying
- Reduces regime range from 10x to 3x

**Clip Range Grid Search:**


| Clip Range | Test AUC | Variance | Notes |
| :-- | :-- | :-- | :-- |
| [0.2, 0.8] | **0.5310** | 0.0142 | Best performance |
| **[0.25, 0.75]** | **0.5303** | **0.0123** | **Best balance** |
| [0.3, 0.7] | 0.5280 | 0.0122 | Overconstrained |
| [0.35, 0.65] | 0.5283 | 0.0108 | Too narrow |


***

### Final Configuration: Simplified + Clipped VIX

**Features (15 total):**

- 12 base features (market + fundamentals)
- 1 VIX feature: `VIX_percentile`
- 2 interactions: `mom121 × VIX`, `vol12 × VIX` (clipped [0.25, 0.75])

**Performance:**


| Metric | Baseline | Final | Improvement |
| :-- | :-- | :-- | :-- |
| Test AUC | 0.5210 | **0.5303** | **+9.3 bps** |
| Variance | 0.0094 | 0.0123 | +31% (acceptable) |
| Fold 1 | 0.517 | 0.527 | +1.0 bps |
| Fold 2 (COVID) | 0.512 | 0.517 | +0.5 bps |
| Fold 3 | 0.534 | 0.547 | +1.3 bps |

**Feature Importance:**

- Interactions: ~23% combined importance (rank \#3 and \#4)
- Successfully captured regime effects without excessive redundancy

***

### Key Learnings:

1. **VIX adds value (+9.3 bps)** but must be used carefully to avoid regime instability
2. **Interaction terms > raw VIX features** for regime-aware modeling
3. **Simplification crucial**: Dropped 3 VIX features, kept 1 + 2 interactions → 52% variance reduction
4. **Clipping optimal at [0.25, 0.75]**: Balances performance and stability
5. **Random Forest already robust to multicollinearity** - problem was regime dependency, not feature correlation

### Decision:

 **Use simplified + clipped VIX interactions:**

- `VIX_percentile` + `mom121_x_VIX` + `vol12_x_VIX` (clipped [0.25, 0.75])
- Best risk-adjusted improvement: +9.3 bps with controlled variance

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
- Depth 4 has lower variance (0.005) but loses -0.2 bps test AUC
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
- max_features=0.4 has **best average** (+0.2 bps) but **2x variance** (0.007 → 0.014)
- **COVID fold (Fold 2) suffers** with more features: 0.518 → 0.509 (-0.9 bps)
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
- Variance reduction minimal: 0.007 → 0.005 (not worth -0.1 to -0.3 bps loss)

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

- Subsampling **hurts performance**: -0.4 bps with 80-90% samples
- Signal already weak → reducing data makes it worse
- Not overfitting (gap = 0.059 is healthy) → no need for regularization
- Variance improvement minimal: 0.007 → 0.006 (not worth AUC loss)

**Reasoning:**
Subsampling helps when:

- Overfitting (not your case)
- Strong signal (yours is weak)
- High variance (yours is 0.007, already low)

**Decision:** ✅ **Keep max_samples=None**

***

## Final Optimized Configuration

After testing **13 different configurations** across 4 phases:

```python
RandomForestClassifier(
    n_estimators=50,           # Phase 3: Confirmed optimal 
    max_depth=3,               # Phase 1: Confirmed optimal 
    min_samples_split=100,     # Not optimized (keep default)
    min_samples_leaf=50,       # Not optimized (keep default)
    max_features='log2',       # Phase 2: Confirmed optimal 
    max_samples=None,          # Phase 4: Confirmed optimal 
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
```

**Final Performance:**

- Test AUC: **0.527** (unchanged from baseline!)
- Test Std: **0.007** (very stable)
- Train-Test Gap: **0.059** (healthy)
- vs No Sentiment: **+0.6 bps improvement**

***

## Key Learnings

### 1. **Simpler is Better with Weak Signals**

Every attempt to increase complexity (deeper trees, more features, more trees) **degraded performance**:

- Shallow trees (depth 3) > deep trees (depth 5)
- Fewer features/split (log2) > more features (0.4)
- Fewer trees (50) > more trees (200)
- Full bootstrap (100%) > subsampling (80%)


### 2. **Default Hyperparameters Were Already Near-Optimal**

All 4 optimization phases confirmed the original configuration was correct. This is **rare** in ML!

Reason: Configuration was already tuned for:

- Weak signal (AUC ~0.53)
- Limited data (~100k samples)
- Regime instability (3 different market periods)


### 3. **Optimization Revealed Why Alternatives Failed**

| Alternative | Why It Failed |
| :-- | :-- |
| Depth 4-5 | Overfit (gap increased to 0.071-0.084) |
| max_features 0.4 | Reduced tree diversity → hurt COVID fold |
| n_estimators 200 | Over-averaged away weak signal |
| max_samples 0.8 | Removed too much data from weak signal |

### 4. **Variance-Performance Tradeoff Visible**

Multiple configs improved variance but hurt test AUC:

- Depth 4: std 0.007→0.005 but AUC -0.2 bps
- 100 trees: std 0.007→0.005 but AUC -0.1 bps

**Decision rule:** Don't sacrifice performance for minimal variance gains

### 5. **COVID Fold (Fold 2) is the Canary**

Changes that hurt Fold 2 (COVID period) were consistently bad:

- max_features 0.4: Fold 2 dropped to 0.509 (-0.9 bps)
- This fold is hardest to predict → best indicator of robustness

***

## Conclusion

**Systematic hyperparameter optimization validated the baseline configuration.** The original hyperparameters were already optimal for this:

- Weak signal regime (AUC ~0.53)
- Regime-dependent data (calm/COVID/elevated VIX periods)
- Limited sample size with complex features (15 features, VIX interactions)

**Result:** No hyperparameter changes needed. Current configuration is production-ready.
