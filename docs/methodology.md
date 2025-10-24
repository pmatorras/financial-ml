# Technical Methodology

Detailed documentation of the technical approach used in this project.

---
## Table of Contents

- [Project structure](#project-structure)
    - [Source Code](#source-code)
    - [Generated Directories](#generated-directories)
    - [Module overview](#module-overview)
- [Data Pipeline](#data-pipeline)
    - [Universe](#universe)
    - [Data Sources](#data-sources)
    - [Data Quality](#data-quality)
- [Feature Engineering](#feature-engineering)
    - [Market Features (Technical)](#market-features-technical)
    - [Discriminating variables](#discriminating-variables)
    - [Feature Scaling](#feature-scaling)
- [Target Variable](#target-variable)
    - [Definition](#definition)
    - [Label Distribution](#label-distribution)
- [Time Series Cross-Validation](#time-series-cross-validation)
    - [Strategy](#strategy)
    - [Critical Safeguards](#critical-safeguards)
    - [Evaluation Metrics](#evaluation-metrics)
- [Model Architecture](#model-architecture)
    - [Base Model: Random Forest](#base-model-random-forest)
    - [Calibration: Isotonic Regression](#calibration-isotonic-regression)
- [Portfolio Construction](#portfolio-construction)
    - [Signal Generation](#signal-generation)
    - [Position Assignment](#position-assignment)
    - [Weighting](#weighting)
    - [Rebalancing](#rebalancing)
- [Performance Attribution](#performance-attribution)
    - [Risk-Adjusted Returns](#risk-adjusted-returns)
    - [Alpha Calculation](#alpha-calculation)
    - [Statistical Significance](#statistical-significance)
- [Risk Management](#risk-management)
    - [Drawdown Analysis](#drawdown-analysis)
    - [Regime Analysis](#regime-analysis)
    - [Position Limits](#position-limits)
- [Transaction Cost Modeling](#transaction-cost-modeling)
    - [Cost Assumptions](#cost-assumptions)
    - [Annual Drag Calculation](#annual-drag-calculation)
    - [Net Alpha](#net-alpha)
- [Production Considerations](#production-considerations)
    - [Data Latency](#data-latency)
    - [Execution](#execution)
    - [Monitoring](#monitoring)
- [Known Limitations](#known-limitations)
    - [Survivorship Bias](#survivorship-bias)
    - [Market Impact](#market-impact)
    - [Regime Dependency](#regime-dependency)
    - [Factor Crowding](#factor-crowding)
- [Future Improvements](#future-improvements)


## Project structure

- Entrypoint and CLI flags live in the main module (`src/financial_ml/`) that dispatches data collection, training, and fundamentals jobs.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.

### Source Code
The current structure of `src/financial_ml` is as follows:
```bash
src/financial_ml/
│
├── __init__.py                      # Package initialization
├── __main__.py                      # Entry point: python -m financial_ml
├── main.py                          # CLI command routing
│
├── data/                            # Data loading and processing
│   ├── __init__.py                  # (Optional) Public API for loaders
│   ├── loaders.py                   # Load market/fundamental data from CSV
│   ├── features.py                  # Feature engineering (market features, ratios)
│   ├── validation.py                # Data quality checks (require_non_empty, etc.)
│   └── collectors/                  # External data collection
│       ├── __init__.py              # Exports: collect_market_data, collect_fundamentals
│       ├── market_data.py           # Download stock prices from yfinance
│       └── fundamental_data.py      # Download fundamentals from SEC EDGAR
│
├── models/                          # Model training and definitions
│   ├── __init__.py                  # Exports: train, get_models, get_model_name
│   ├── training.py                  # Train models with time-series CV
│   └── definitions.py               # Model pipeline definitions (LogReg, RF)
│
├── evaluation/                      # Model analysis and evaluation
│   ├── __init__.py                  # Exports: analyze_models
│   ├── analyze.py                   # Load models and run analysis
│   └── feature_analysis.py          # Feature importance, coefficients
│
├── portfolio/                       # Backtesting and portfolio construction
│   ├── __init__.py                  # Exports: run_backtest
│   ├── backtest.py                  # Main backtesting orchestration
│   ├── construction.py              # Portfolio construction (positions, smoothing)
│   ├── performance.py               # Return calculation and metrics
│   ├── diagnostics.py               # Model agreement, turnover, beta analysis
│   └── visualization.py             # Plotting (cumulative returns, drawdown)
│
└── utils/                           # Utilities and configuration
    ├── config.py                    # Constants (DATA_DIR, MARKET_KEYS, etc.)
    ├── paths.py                     # Path helpers (get_prediction_file, etc.)
    └── helpers.py                   # Common utilities (safe_div, etc.)
```
***
### Generated Directories

These directories are created automatically during execution:

```bash
data/                                # Raw and processed data
├── market/                          # Stock price data (CSV)
├── fundamentals/                    # SEC EDGAR fundamental data (CSV)
└── predictions/                     # Model predictions (CSV)
    ├── production/                  # Full dataset predictions
    └── debug/                       # Test subset predictions

models/                              # Trained model artifacts
├── production/                      # Models trained on full dataset
│   ├── logreg_l1.pkl
│   ├── logreg_l2.pkl
│   ├── rf.pkl
│   └── feature_names.txt
└── debug/                           # Models trained on test subset

figures/                             # Generated plots and visualizations
├── portfolio_backtest_*.png         # Portfolio performance charts
└── feature_importance_*.png         # Feature analysis charts

debug/                               # Debug outputs (when --debug flag used)
├── funda.csv                        # Raw fundamentals snapshot
├── monthly.csv                      # Monthly resampled data
└── *.csv                            # Other debug CSVs

logs/                                # Application logs (if implemented)
```
This structure keeps the logic compartmentalized:
- `core/` centralises configuration and utilities. General paths are centralised through these scripts, with `data/`, `figures/`, and `logs/` auto-created on first run.
- `data/` manages loading, cleaning, and labeling.
- `features/` handles transformation and feature engineering.
- `models/` encapsulates model workflows and portfolio logic.
- `viz/` contains both charts and dashboards for performance and exploration.The 
- `cli/` directory aligns with the earlier design’s modular entrypoint for reproducible workflows (market, fundamentals, train, plot...).

This makes the repository scalable for predictive modeling, portfolio backtesting, and visualization, while keeping all paths consistent with professional ML deployment standards
***
### Module overview

#### Data Module

Load, validate, and engineer features from market and fundamental data.

**Key Components:**

- **Loaders** - Read processed CSV data (market prices, fundamentals)
- **Features** - Calculate market indicators and fundamental ratios
- **Collectors** - Download data from yfinance and SEC EDGAR APIs
- **Validation** - Data quality checks and assertions


#### Models Module

Define and train ML models with time-series cross-validation.

**Supported Models:**

- Logistic Regression (L1/L2 regularization)
- Random Forest Classifier

**Features:**

- Time-series CV to prevent lookahead bias
- Hyperparameter-tuned pipelines
- Model persistence with joblib


#### Evaluation Module

Analyze trained model performance and feature importance.

**Capabilities:**

- Load saved models without retraining
- Extract feature importance/coefficients
- Visualize model behavior


#### Portfolio Module

Construct portfolios from predictions and run comprehensive backtests.

**Features:**

- Long/short portfolio construction
- Prediction smoothing (reduce noise)
- Performance metrics (Sharpe, drawdown, returns)
- Diagnostic analysis (turnover, beta, model agreement)
- Compare to benchmark (SPY)


#### Utils Module

Shared configuration, paths, and utility functions that can be accesible from all over the repository.

***


## Data Pipeline

 ### Universe
- **Source:** S&P 500 constituents
- **Period:** 2010-2025 (training), 2016-2025 (backtesting)
- **Frequency:** Monthly rebalancing
- **Average stocks per month:** ~335 (due to missing data/fundamentals)

 ### Data Sources
- **Market data:** Yahoo Finance (prices, volumes)
- **Fundamental data:** SEC EDGAR API (quarterly filings)
- **Benchmark:** SPY ETF for market returns

 ### Data Quality
- Remove stocks with <24 months history
- Remove stocks with missing fundamental data
- Forward-fill quarterly fundamentals to monthly
- Filter out extreme outliers (>3 sigma in returns)

---

## Feature Engineering

 ### Market Features (Technical)
### Discriminating variables

Currently, the model takes information from both the market stock information (monthly basis), and (quaterly) fundamentals:

#### Variables from market behaviours:
- `ClosePrice`: Raw stock closing price
**Momentum:**
- `r1` (1m return): Captures the most recent monthly price move, providing a highly responsive but noisy signal that helps models account for short‑term dynamics and potential reversal pressure.

- `r12` (12m return): Summarizes the past year’s trend including the latest month, offering a strong baseline momentum proxy that can be tempered with risk controls for stability.

- `mom121` (12m − 1m momentum): Focuses on medium‑term trend by excluding the most recent month, reducing short‑term reversal effects and typically improving persistence out of sample.

**Volatility:**
- `vol3` (3m rolling std): Fast‑moving realized volatility over three months that reacts to recent shocks, useful for volatility‑managed scaling and down‑weighting unstable names.

- `vol12` (12m rolling std): Slower, more structural risk estimate over a full year that complements vol3 by distinguishing transient turbulence from persistent volatility regimes.
***


#### Variables from fundamentals

The following variables are taken from the stock fundamentals:

- Book-to-Market (`B/M`):  captures valuation relative to book value and is a canonical factor in asset pricing and cross-sectional models. $B/M=\frac{Equity}{Price\times Shares}$
- Return on Equity (`ROE`):  measures profitability to equity holders and proxies the profitability factor component in five-factor frameworks. $ROE=\frac{Net Income_{TTM}}{Equity}$
- Return on Assets (`ROA`):  complements ROE by controlling for capital structure and overall asset base. $ROA=\frac{Net Income_{TTM}}{Assets}$
- `Net Margin`: gauges earnings efficiency and is routinely used in fundamental screens and profitability diagnostics.  $Net Margin=\frac{Net Income_{TTM}}{Revenues_{TTM}}$
- `Leverage`:  captures balance-sheet risk and interacts with profitability and value in expected return models. $Leverage=\frac{Liabilities}{Assets}$
- `Asset Growth` (Investment):  maps to the investment factor where higher investment has been associated with lower average returns. $Inv=\frac{Assets_{t}-Assets_{t-4q}}{Assets_{t-4q}}$
- `Net Share Issuance`:  tracks dilution/buybacks and has documented predictive power for subsequent returns. $Issuance=\frac{Shares_{t}-Shares_{t-4q}}{Shares_{t-4q}}$
- Size (`marketCap`):  provides a standard size control that stabilizes cross-sectional comparisons. $\log(Market Cap)=\log(Price\times Shares) $

 This set targets value, profitability, investment, leverage, size, and dilution, which align with widely used multi-factor models and documented cross-sectional return predictors.

#### Combination of both features:
- `LogMktCap`: log(price × shares outstanding)
***


 ### Feature Scaling
All features standardized using `StandardScaler` (fit on training set only)

---

## Target Variable

 ### Definition
Binary classification: Will each stock beat SPY over next 12 months?

```bash
forward_return_stock = (price_t+12 / price_t) - 1
forward_return_spy = (spy_t+12 / spy_t) - 1
y = 1 if forward_return_stock > forward_return_spy else 0
```


 ### Label Distribution
- Approximately 50/50 split (by construction)
- Varies by regime (bear markets: <50% beat SPY)

---

## Time Series Cross-Validation

 ### Strategy
Expanding window to prevent lookahead bias:

```bash
Fold 1: Train [2010-2016] → Test [2016-2018]
Fold 2: Train [2010-2018] → Test [2018-2021]
Fold 3: Train [2010-2021] → Test [2021-2025]
```


 ### Critical Safeguards
- No data leakage: features computed using only past data
- Forward returns computed at prediction time (never look ahead)
- Scaler fit on training set only, then applied to test
- No shuffling (maintains temporal order)

 ### Evaluation Metrics
- **Primary:** AUC (area under ROC curve)
- **Secondary:** Sharpe ratio, alpha, drawdown from portfolio backtest

---

## Model Architecture

 ### Base Model: Random Forest

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

 ### Calibration: Isotonic Regression

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

## Portfolio Construction

 ### Signal Generation
For each stock on each month:
1. Generate probability p(beat SPY)
2. Apply 3-month rolling average (smoothing)
3. Rank stocks by smoothed probability

 ### Position Assignment
- **Long:** Top 10% (highest probabilities)
- **Short:** Bottom 10% (lowest probabilities)
- **Neutral:** Middle 80%

Typical portfolio:
- ~33 long positions
- ~33 short positions
- ~66 total positions

 ### Weighting
Equal-weighted within long and short buckets:
- Each long position: 1/33 of long capital
- Each short position: 1/33 of short capital
- Total exposure: 100% long, 100% short (dollar-neutral)

 ### Rebalancing
- **Frequency:** Monthly (last trading day)
- **Turnover:** ~42% average
- **Transaction cost:** 10 bps per trade (0.1%)

---

## Performance Attribution

 ### Risk-Adjusted Returns
- **Sharpe Ratio:** 0.80
  - Calculation: (Annual Return - Risk Free) / Annual Volatility
  - Risk-free rate: 0% (for simplicity)
  - Annualization: √12 for monthly data

 ### Alpha Calculation
Linear regression: `Portfolio_Return = α + β × SPY_Return + ε`

Results:
- **Beta:** 1.006 (market-neutral)
- **Alpha:** 2.29% annual (0.191% monthly × 12)
- **R²:** 0.859 (high correlation with market)

 ### Statistical Significance
**Sharpe Ratio T-test:**
- T-statistic: 6.53
- P-value: <0.001
- Bonferroni adjustment: p < 0.0167 (tested 3 models)
- Result: Statistically significant

---

## Risk Management

 ### Drawdown Analysis
- **Max Drawdown:** -21.5%
- **Comparison:** SPY -24.0% (same period)
- **Recovery time:** [calculate from data]

 ### Regime Analysis
| Period | Portfolio DD | SPY DD | Difference |
|--------|--------------|--------|------------|
| COVID (2020) | -21.5% | -24.0% | +2.5% better |
| Bear (2022) | -20.5% | -24.0% | +3.5% better |

 ### Position Limits
- Max 10% in any single stock (naturally enforced by equal-weighting)
- Sector concentration: Monitor but not explicitly constrained

---

## Transaction Cost Modeling

 ### Cost Assumptions
- **Bid-ask spread:** 5 bps (0.05%)
- **Market impact:** 5 bps (0.05%)
- **Total cost per trade:** 10 bps (0.1%)

 ### Annual Drag Calculation

```
Annual turnover = 42% monthly × 12 = 504% annual
Cost = 504% × 0.1% = 0.50% drag (one-sided)
Dollar-neutral: 2 × 0.50% = 1.0% drag
Actual measured: 0.82% (from diagnostics)
```

 ### Net Alpha
- Gross alpha: ~3.0% (from IC)
- Transaction costs: -0.82%
- **Net alpha: 2.18%** ✅

---

## Production Considerations

 ### Data Latency
- Fundamental data: Quarterly lag (up to 45 days after quarter-end)
- Price data: Daily (1-day lag acceptable)
- Rebalancing: End of month (flexibility in exact timing)

 ### Execution
- Portfolio construction: <1 minute (Python script)
- Order generation: Equal-weighted positions
- Execution window: Last day of month (spread over multiple hours)

 ### Monitoring
- Track prediction distribution (mean ~0.50)
- Monitor turnover (alert if >60%)
- Check sector concentration (alert if any >30%)
- Verify data completeness (alert if <300 stocks)

---

## Known Limitations

 ### Survivorship Bias
- Using current S&P 500 constituents may introduce bias
- Mitigation: Historical constituent data would improve accuracy

 ### Market Impact
- Large AUM would increase transaction costs beyond 10 bps
- Strategy works best for <$100M AUM

 ### Regime Dependency
- Strategy may underperform in:
  - Extreme momentum markets (2020-2021 tech rally)
  - Market dislocations (2008 crisis not in training data)

 ### Factor Crowding
- Heavy reliance on size factor (37% of importance)
- Size premium has weakened in recent years
- Risk: If many strategies use similar factors, alpha erodes

---

## Future Improvements

### Historical Fundamentals Extension:
- **Goal:** Extend fundamental data before 2010
- **Challenge:** SEC EDGAR API only available from 2010+
- **Approach:** Commercial data providers (Compustat, Bloomberg) or manual parsing of historical filings
- **Expected Impact:** More training data, potentially stronger signals in earlier regimes

### Alternative Data Sources:
- **Sentiment Analysis:** News headlines, social media, analyst reports
- **Implementation:** NLP on earnings call transcripts, Twitter sentiment
- **Expected Impact:** +0.02-0.03 Sharpe from behavioral signals
- **Risk:** Sentiment data expensive, noisy, potential overfitting

###  International Markets:
- **Extend to:** European stocks (STOXX 600), Asian markets (Nikkei, HSI)
- **Challenge:** Different accounting standards, data availability
- **Benefit:** Test if signals generalize across regions
- **Implementation:** Country-specific models or unified multi-region approach

### Feature Engineering
- Add sentiment features (news, social media)
- Include sector-relative features (demean within industry)
- Test alternative fundamental ratios

### Portfolio Construction
- Test different percentiles (top/bottom 5%, 15%)
- Explore dynamic weighting (by conviction/probability)
- Add sector neutralization

### Modeling
- Test alternative models (LightGBM, XGBoost)
- Explore deep learning (LSTM for time series)
- Multi-horizon predictions (1M, 3M, 6M, 12M)

### Risk Management
- Implement max sector exposure limits
- Add volatility targeting (scale positions by risk)
- Dynamic rebalancing (respond to market volatility)


<br><hr>
[Back to top](#technical-methodology)
