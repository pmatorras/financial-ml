# Technical Methodology

Detailed documentation of the technical approach used in this project.

---

## 1. Data Pipeline

### 1.1 Universe
- **Source:** S&P 500 constituents
- **Period:** 2010-2025 (training), 2016-2025 (backtesting)
- **Frequency:** Monthly rebalancing
- **Average stocks per month:** ~335 (due to missing data/fundamentals)

### 1.2 Data Sources
- **Market data:** Yahoo Finance (prices, volumes)
- **Fundamental data:** SEC EDGAR API (quarterly filings)
- **Benchmark:** SPY ETF for market returns

### 1.3 Data Quality
- Remove stocks with <24 months history
- Remove stocks with missing fundamental data
- Forward-fill quarterly fundamentals to monthly
- Filter out extreme outliers (>3 sigma in returns)

---

## 2. Feature Engineering

### 2.1 Market Features (Technical)

**Price-based:**
- `ClosePrice`: Raw closing price
- `LogMktCap`: log(price × shares outstanding)

**Momentum:**
- `r1`: 1-month return
- `r12`: 12-month return
- `mom121`: 12-month return skipping most recent month

**Volatility:**
- `vol3`: 3-month return standard deviation
- `vol12`: 12-month return standard deviation

### 2.2 Fundamental Features

**Valuation:**
- `BookToMarket`: Book value / Market cap

**Profitability:**
- `ROA`: Net Income / Total Assets
- `ROE`: Net Income / Shareholders Equity
- `NetMargin`: Net Income / Revenue

**Financial Health:**
- `Leverage`: Total Debt / Total Assets
- `AssetGrowth`: Year-over-year change in total assets
- `NetShareIssuance`: Year-over-year change in shares outstanding

### 2.3 Feature Scaling
All features standardized using `StandardScaler` (fit on training set only)

---

## 3. Target Variable

### 3.1 Definition
Binary classification: Will each stock beat SPY over next 12 months?

```bash
forward_return_stock = (price_t+12 / price_t) - 1
forward_return_spy = (spy_t+12 / spy_t) - 1
y = 1 if forward_return_stock > forward_return_spy else 0
```


### 3.2 Label Distribution
- Approximately 50/50 split (by construction)
- Varies by regime (bear markets: <50% beat SPY)

---

## 4. Time Series Cross-Validation

### 4.1 Strategy
Expanding window to prevent lookahead bias:

```bash
Fold 1: Train [2010-2016] → Test [2016-2018]
Fold 2: Train [2010-2018] → Test [2018-2021]
Fold 3: Train [2010-2021] → Test [2021-2025]
```


### 4.2 Critical Safeguards
- No data leakage: features computed using only past data
- Forward returns computed at prediction time (never look ahead)
- Scaler fit on training set only, then applied to test
- No shuffling (maintains temporal order)

### 4.3 Evaluation Metrics
- **Primary:** AUC (area under ROC curve)
- **Secondary:** Sharpe ratio, alpha, drawdown from portfolio backtest

---

## 5. Model Architecture

### 5.1 Base Model: Random Forest

```bash
RandomForestClassifier(
n_estimators=50, # Balance: enough trees, not too slow
max_depth=3, # Constrained to prevent overfitting
min_samples_split=0.02, # 2% of samples (regularization)
min_samples_leaf=0.01, # 1% of samples (regularization)
max_features='log2', # Feature sampling for diversity
class_weight='balanced', # Handle any class imbalance
random_state=42 # Reproducibility
)
```

**Why Random Forest?**
- Captures non-linear relationships (vs linear LogReg)
- Handles feature interactions naturally
- Robust to outliers
- Provides feature importance

**Why Constrained (max_depth=3)?**
- Deep trees (10+) overfit badly (test AUC drops to 0.52)
- Shallow trees (3) generalize better
- Reduces training-test gap from 12pp to 7pp

### 5.2 Calibration: Isotonic Regression

```bash
CalibratedClassifierCV(
estimator=RandomForestClassifier(...),
method='isotonic', # Non-parametric, flexible
cv=3 # Internal cross-validation
)
```

**Why Calibration?**
- Original RF: mean probability 0.487, which even though is not a problem in terms of the portfolio decision (as it takes the top/bottom stocks) it is still miscalibrated.
- After calibration: mean probability 0.502 (correct)
- Increased std from 0.035 to 0.059 (i.e. better discrimination)

**How It Works:**
1. Train base a RF on training data
2. Use internal 3-fold CV to learn calibration mapping
3. Apply isotonic regression: monotonic function that maps raw probabilities to calibrated ones
4. Preserves AUC (ranking unchanged), fixes probability scale

**Impact:**
- Sharpe improved from 0.71 to 0.80 (+13%)
- Alpha improved from 0.87% to 2.29% (+163%)
- Interpretability: now >45% of predictions have prob >0.5

---

## 6. Portfolio Construction

### 6.1 Signal Generation
For each stock on each month:
1. Generate probability p(beat SPY)
2. Apply 3-month rolling average (smoothing)
3. Rank stocks by smoothed probability

### 6.2 Position Assignment
- **Long:** Top 10% (highest probabilities)
- **Short:** Bottom 10% (lowest probabilities)
- **Neutral:** Middle 80%

Typical portfolio:
- ~33 long positions
- ~33 short positions
- ~66 total positions

### 6.3 Weighting
Equal-weighted within long and short buckets:
- Each long position: 1/33 of long capital
- Each short position: 1/33 of short capital
- Total exposure: 100% long, 100% short (dollar-neutral)

### 6.4 Rebalancing
- **Frequency:** Monthly (last trading day)
- **Turnover:** ~42% average
- **Transaction cost:** 10 bps per trade (0.1%)

---

## 7. Performance Attribution

### 7.1 Risk-Adjusted Returns
- **Sharpe Ratio:** 0.80
  - Calculation: (Annual Return - Risk Free) / Annual Volatility
  - Risk-free rate: 0% (for simplicity)
  - Annualization: √12 for monthly data

### 7.2 Alpha Calculation
Linear regression: `Portfolio_Return = α + β × SPY_Return + ε`

Results:
- **Beta:** 1.006 (market-neutral)
- **Alpha:** 2.29% annual (0.191% monthly × 12)
- **R²:** 0.859 (high correlation with market)

### 7.3 Statistical Significance
**Sharpe Ratio T-test:**
- T-statistic: 6.53
- P-value: <0.001
- Bonferroni adjustment: p < 0.0167 (tested 3 models)
- Result: ✅ Statistically significant

---

## 8. Risk Management

### 8.1 Drawdown Analysis
- **Max Drawdown:** -21.5%
- **Comparison:** SPY -24.0% (same period)
- **Recovery time:** [calculate from data]

### 8.2 Regime Analysis
| Period | Portfolio DD | SPY DD | Difference |
|--------|--------------|--------|------------|
| COVID (2020) | -21.5% | -24.0% | +2.5% better |
| Bear (2022) | -20.5% | -24.0% | +3.5% better |

### 8.3 Position Limits
- Max 10% in any single stock (naturally enforced by equal-weighting)
- Sector concentration: Monitor but not explicitly constrained

---

## 9. Transaction Cost Modeling

### 9.1 Cost Assumptions
- **Bid-ask spread:** 5 bps (0.05%)
- **Market impact:** 5 bps (0.05%)
- **Total cost per trade:** 10 bps (0.1%)

### 9.2 Annual Drag Calculation

```
Annual turnover = 42% monthly × 12 = 504% annual
Cost = 504% × 0.1% = 0.50% drag (one-sided)
Dollar-neutral: 2 × 0.50% = 1.0% drag
Actual measured: 0.82% (from diagnostics)
```

### 9.3 Net Alpha
- Gross alpha: ~3.0% (from IC)
- Transaction costs: -0.82%
- **Net alpha: 2.18%** ✅

---

## 10. Production Considerations

### 10.1 Data Latency
- Fundamental data: Quarterly lag (up to 45 days after quarter-end)
- Price data: Daily (1-day lag acceptable)
- Rebalancing: End of month (flexibility in exact timing)

### 10.2 Execution
- Portfolio construction: <1 minute (Python script)
- Order generation: Equal-weighted positions
- Execution window: Last day of month (spread over multiple hours)

### 10.3 Monitoring
- Track prediction distribution (mean ~0.50)
- Monitor turnover (alert if >60%)
- Check sector concentration (alert if any >30%)
- Verify data completeness (alert if <300 stocks)

---

## 11. Known Limitations

### 11.1 Survivorship Bias
- Using current S&P 500 constituents may introduce bias
- Mitigation: Historical constituent data would improve accuracy

### 11.2 Market Impact
- Large AUM would increase transaction costs beyond 10 bps
- Strategy works best for <$100M AUM

### 11.3 Regime Dependency
- Strategy may underperform in:
  - Extreme momentum markets (2020-2021 tech rally)
  - Market dislocations (2008 crisis not in training data)

### 11.4 Factor Crowding
- Heavy reliance on size factor (37% of importance)
- Size premium has weakened in recent years
- Risk: If many strategies use similar factors, alpha erodes

---

## 12. Future Improvements

### 12.1 Feature Engineering
- Add sentiment features (news, social media)
- Include sector-relative features (demean within industry)
- Test alternative fundamental ratios

### 12.2 Portfolio Construction
- Test different percentiles (top/bottom 5%, 15%)
- Explore dynamic weighting (by conviction/probability)
- Add sector neutralization

### 12.3 Modeling
- Test alternative models (LightGBM, XGBoost)
- Explore deep learning (LSTM for time series)
- Multi-horizon predictions (1M, 3M, 6M, 12M)

### 12.4 Risk Management
- Implement max sector exposure limits
- Add volatility targeting (scale positions by risk)
- Dynamic rebalancing (respond to market volatility)
