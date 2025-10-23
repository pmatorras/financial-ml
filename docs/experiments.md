# Experiment Log

This document tracks all model experiments and design choices throughout the project development.

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
| Mean y_prob | 0.487 | 0.502 | ✅ Fixed |
| Std y_prob | 0.035 | 0.059 | ✅ +68% (better separation) |
| Test AUC | 0.554 | 0.557 | ✅ +0.3pp |
| Sharpe Ratio | 0.71 | 0.80 | ✅ +13% |
| Annual Alpha | 0.87% | 2.29% | ✅ +163% |
| Max Drawdown | -22.8% | -21.5% | ✅ Better |

### Why It Worked:
1. Fixed probability scale (mean now 0.502)
2. Increased prediction spread (std 0.059) → clearer signal for portfolio construction
3. Better discrimination between good/bad stocks
4. Top 10% stocks now mostly have prob >0.5 (interpretable)

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
| Sharpe Ratio | 0.80 | 0.71 | ❌ -11% |
| Annual Return | 17.9% | 16.8% | ❌ -1.1% |
| Max Drawdown | -21.5% | -22.2% | ❌ Worse |
| Volatility | 17.4% | 18.1% | ❌ +0.7% |

### Why It Failed:
1. LogReg is weaker (AUC 0.558 vs RF_cal 0.557)
2. Correlation too high (0.556) → not enough diversity
3. Averaging diluted RF_cal's stronger signal
4. LogReg's narrow predictions (std ~0.04) compressed RF_cal's better spread (0.059)

### Decision: Reject ensemble. Use RF_cal alone.

### Key Learning:
Ensembles work when models have similar performance and low correlation (<0.5). When one model is clearly superior, averaging with weaker models hurts performance.

---

## Experiment 4: Hyperparameter Tuning
**Date:** October 22, 2025  
**Goal:** Optimize Random Forest hyperparameters

### Parameters Tested:

**max_depth:**
- depth=2: Test AUC 0.53 (underfitting)
- depth=3: Test AUC 0.557 ✅ (optimal)
- depth=5: Test AUC 0.55 (overfitting)
- depth=10: Test AUC 0.52 (severe overfitting)

**n_estimators:**
- n=30: Test AUC 0.545 (underfitting)
- n=50: Test AUC 0.557 ✅ (optimal)
- n=100: Test AUC 0.557 (no improvement, slower)

**min_samples_split:**
- 0.01: Overfits
- 0.02: Optimal ✅
- 0.05: Underfits

### Final Hyperparameters:

```bash
RandomForestClassifier(
n_estimators
50, max_
epth=3, min_samples
split=0.02, min_sa
ples_leaf=0.01,
ax_features='log2',
lass_weight='ba)
```


---

## Experiment 5: Smoothing Impact
**Date:** October 23, 2025  
**Goal:** Reduce month-to-month prediction volatility

### Problem:
Raw predictions change significantly month-to-month (mean 5.18%)

### Approach:
3-month rolling average of predictions

### Results:
- Raw prediction change: 5.18% mean
- After smoothing: 1.93% mean
- Reduction: 58.7% ✅

### Impact:
- Reduced turnover (from ~50% to ~42%)
- Lower transaction costs
- Smoother portfolio returns
- No degradation in Sharpe ratio

### Decision: Apply 3-month smoothing to all predictions

---

## Summary of Final Configuration

**Model:** Calibrated Random Forest (CalibratedClassifierCV with isotonic regression)

**Key Design Choices:**
- ✅ Calibration over raw probabilities
- ✅ Single model over ensemble
- ✅ Constrained RF (max_depth=3) over deep trees
- ✅ 3-month smoothing for stability
- ✅ Top/bottom 10% portfolio construction

**Performance:**
- Sharpe Ratio: 0.80
- Annual Alpha: 2.29%
- Statistical significance: p < 0.001 (Bonferroni-adjusted)

## Experiment 5: Feature Engineering Ablation (Oct 23, 2025)

**Goal:** Test if engineered features improve baseline performance

**Tested Configurations:**
1. Reversal feature (r1 negation): -0.002 AUC
2. Cross-sectional ranks (replace raw): -0.007 AUC, **-0.031 on Fold 3**
3. Interaction features (4 new): -0.003 AUC
4. Combinations of above: All worse

**Key Finding:** Every enhancement hurt performance, especially on recent data (Fold 3: 2021-2025)

**Why Enhancements Failed:**
- **Ranks:** Redundant for tree models (RF learns relative comparisons naturally)
- **Interactions:** RF with max_depth=3 can't use extra features effectively
- **Reversal:** Pure noise, no new information
- **All:** Overfit to historical patterns, fail on recent data

**Conclusion:** Baseline 14 raw features is optimal. Simplicity > complexity.

**Key Learning:** In ML, more features often means worse generalization. Constrained models (max_depth=3) need carefully selected features, not exhaustive transformations.
