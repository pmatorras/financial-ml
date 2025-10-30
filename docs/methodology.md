# Technical Methodology

Detailed documentation of the technical approach used in this project.

---
## Table of Contents

- [Project Structure](#project-structure)
    - [Repository Layout](#repository-layout)
    - [Source Code](#source-code)
    - [Generated Directories](#generated-directories)
    - [Module Overview](#module-overview)
- [Data Pipeline](#data-pipeline)
    - [Universe](#universe)
    - [Data Sources](#data-sources)
    - [Data Quality](#data-quality)
- [Feature Engineering](#feature-engineering)
    - [Discriminating Variables](#discriminating-variables)
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
    - [Portfolio Assignment](#portfolio-assignment)
    - [Rebalancing](#rebalancing)
- [Performance Attribution](#performance-attribution)
    - [Risk-Adjusted Returns](#risk-adjusted-returns)
    - [Alpha Calculation](#alpha-calculation)
    - [Statistical Significance](#statistical-significance)
- [Risk Management](#risk-management)
    - [Drawdown Analysis](#drawdown-analysis)
    - [Regime Analysis](#regime-analysis)
    - [Position Limits](#position-limits)
- [Production Considerations](#production-considerations)
    - [Data Latency](#data-latency)
    - [Execution](#execution)
    - [Monitoring](#monitoring)
- [Known Limitations](#known-limitations)
    - [Survivorship Bias](#survivorship-bias)
    - [Market Impact](#market-impact)
    - [Regime Dependency](#regime-dependency)
    - [Factor Crowding](#factor-crowding)
    - [Long-Only Constraints](#long-only-constraints)
- [Future Improvements](#future-improvements)

## Project structure

- Entrypoint and CLI flags live in the main module (`src/financial_ml/`) that dispatches data collection, training, and fundamentals jobs.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.

## Repository Layout
The repository follows standard Python project conventions with clear separation between source code, documentation, configuration, and generated outputs:
```python 
financial-ml/
│
├── src/financial_ml/              # Main package source code (detailed below)
├── docs/                          # Project documentation
│   ├── methodology.md             # Technical methodology and approach
│   ├── experiments.md             # Experimental results and analysis
│   ├── decisions.md               # Design decisions and rationale
│   ├── results.md                 # Summary of key findings
│   └── images/                    # Documentation figures
├── scripts/                       # Utility scripts
│   └── generate_final_figures.py  # Generate publication-ready figures
├── data/                          # Generated data (see Generated Directories)
├── models/                        # Generated models (see Generated Directories)
├── figures/                       # Generated figures (see Generated Directories)
├── README.md                      # Project overview and quick start
├── Requirements.txt               # Python dependencies
├── pyproject.toml                 # Build system configuration
├── Makefile                       # Common tasks automation
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

### Source Code
The current structure of `src/financial_ml` is as follows:
```python
src/financial_ml/
│
├── __init__.py                      # Package initialization
├── __main__.py                      # Entry point: python -m financial_ml
├── main.py                          # CLI command routing
│
├── cli/                             # Command-line interface components
│   ├── __init__.py                  # CLI exports
│   ├── parser.py                    # Argument parser configuration
│   └── validators.py                # Input validation and argument checks
├── data/                            # Data loading and processing
│   ├── __init__.py                  # (Optional) Public API for loaders
│   ├── loaders.py                   # Load market/fundamental data from CSV
│   ├── features.py                  # Feature engineering (market features, ratios)
│   ├── validation.py                # Data quality checks (require_non_empty, etc.)
│   └── collectors/                  # External data collection
│       ├── __init__.py              # Exports: collect_market_data, collect_fundamentals
│       ├── market_data.py           # Download stock prices from yfinance
│       ├── fundamental_data.py      # Download fundamentals from SEC EDGAR API
│       ├── sentiment_data.py        # Sentiment data collection
│       └── utils.py                 # Collector utility functions
├── models/                          # Model training and definitions
│   ├── __init__.py                  # Exports: train, get_models, get_model_name
│   ├── training.py                  # Train models with time-series CV
│   └── definitions.py               # Model pipeline definitions (LogReg, RF)
│
├── evaluation/                      # Model analysis and evaluation
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
    ├── helpers.py                   # Common utilities (safe_div, etc.)
    └── logging.py                   # Logging configuration and utilities

```
***
### Generated Directories

These directories are created automatically during execution:

```python
data/                                  # Raw and processed data
├── market/                            # Stock price data (CSV)
├── fundamentals/                      # SEC EDGAR fundamental data (CSV)
├── sentiment/                         # Sentiment data (if --do-sentiment used)
└── predictions/                       # Model predictions (CSV)
    ├── production/                    # Full dataset predictions
    └── debug/                         # Test subset predictions

models/                                # Trained model artifacts
├── model1.pkl                         # models pkl files
├── ...
└── feature_names.txt                  # Feature names for model interpretation
├── sentiment/                         # Models trained with sentiment data
│   ├── model_1.pkl                    # models pkl files
│   ├── ...
│   └── feature_names.txt              # Feature names for model interpretation
├── only_market/                       # Models trained on only market data
│   ├── model_1.pkl                    # models pkl files
│   ├── ...
│   └── feature_names.txt              # Feature names for model interpretation
└── debug/                             # Models trained on test subset

figures/                               # Generated plots and visualizations
├── portfolio_backtest_{model}.png     # Portfolio performance charts
├── coefficients_{model}.png           # Coefficients for logreg models
├── importance_{model}.png             # Feature importance charts
└── model_correlation_matrix.png       # Cross-model prediction correlation
└── sentiment/                         # Included sentiment information
    ├── portfolio_backtest_{model}.png # Portfolio performance charts
    ├── coefficients_{model}.png       # Coefficients for logreg models
    ├── importance_{model}.png         # Feature importance charts
    └── model_correlation_matrix.png   # Cross-model prediction correlation
└── market_only/                       # Market only information
    ├── portfolio_backtest_{model}.png # Portfolio performance charts
    ├── coefficients_{model}.png       # Coefficients for logreg models
    ├── importance_{model}.png         # Feature importance charts
    └── model_correlation_matrix.png   # Cross-model prediction correlation
debug/                                 # Debug outputs (when --debug flag used)
├── funda.csv                          # Raw fundamentals snapshot
├── monthly.csv                        # Monthly resampled data
└── *.csv                              # Other debug CSVs

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

#### Output File Details
**Data Files**

| File | Description | 
| :-- | :-- | 
| `data/sp500_list.csv` | S\&P 500 constituent tickers | 
| `data/sp500_prices.csv` | Monthly adjusted close prices | 
| `data/sentiment.csv` | Sentiment data | 
| `data/sp500_fundamentals.csv` | Company fundamentals from SEC | 

**Model Artifacts**

| File | Description |
|------|-------------|
| `models/rf.pkl` | Trained Random Forest model (joblib) |
| `models/feature_names.txt` | List of features used in training |
| `models/training_summary.txt` | Training metrics and performance summary |

**Prediction Files**
| File | Description |
| :-- | :-- |
| `data/predictions/production/predictions_{model}.csv` | Out-of-fold predictions with probabilities |
| `models/production/{model}.pkl` | Trained model artifacts (serialized with joblib) |
| `models/production/feature_names.txt` | List of features used in training |
| `models/production/training_summary.txt` | Summary of training AUC for each fold and each model, as well as list of training features |
**Visualizations:**

| File | Description |
|------|-------------|
| `importance_{model}.png` | Feature importance for a treee mdoel |
| `coefficients_logreg_{l1,l2}.png` | LogReg coefficient magnitudes |
| `portfolio_performance_{model}_{portfolio_type}_{per_top}.png` | Cumulative portfolio returns vs SPY for a given model, portfolio type, and percentage of top stocks|
| `model_correlation_matrix.png` | Correlation between all model predictions |
| `sector_drift_{model}.png` | Sector allocation over time vs SPY benchmark |

***
### Module overview
### CLI Module 
Handles command-line interface parsing, argument validation, and user input processing.​

Key Components:

- `parser.py` - Configures argument parser with subcommands (collect, train, backtest, analyze) and global flags (--debug, --do-sentiment, --market-only)
- `validators.py` - Validates user inputs, checks argument compatibility, and enforces constraints before workflow execution

#### Data Module

Load, validate, and engineer features from market and fundamental data.

**Key Components:**

- **Loaders** - Read processed CSV data (market prices, fundamentals, sentiment data)
- **Features** - Calculate market indicators and fundamental ratios
- **Collectors** - Download data from yfinance and SEC EDGAR APIs
- **Validation** - Data quality checks and assertions


#### Models Module

Define and train ML models with time-series cross-validation.

**Supported Models:**

- Logistic Regression (L1/L2 regularization, with tuned C parameters)
- Random Forest Classifier
- XGBoost, LightGBM, Gradient Boosting classifiers
- Isotonic calibration for probability refinement

**Features:**

- Time-series CV to prevent lookahead bias
- Hyperparameter-tuned scikit-learn pipelines
- Model persistence with joblib
- Feature scaling with StandardScaler




#### Evaluation Module

Analyze trained model performance and feature importance.

**Capabilities:**

- Load saved models without retraining
- Extract feature importance/coefficients
- Visualize model behaviour


#### Portfolio Module

Construct portfolios from predictions and run comprehensive backtests.

**Features:**

- Long/short portfolio construction
- Prediction smoothing (reduce noise)
- Monthly rebalancing with transaction cost modeling
- Performance metrics (Sharpe, drawdown, returns)
- Sector concentration comparisons over time
- Diagnostic analysis (turnover, beta, model agreement)
- Benchmark comparison against SPY and equal-weight baseline


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
- **Sentiment data** Yahoo Finance `^VIX` index
- **Benchmark:** SPY ETF for market returns

 ### Data Quality
- Remove stocks with <24 months history
- Remove stocks with missing fundamental data
- Forward-fill quarterly fundamentals to monthly
- Filter out extreme outliers (>3 sigma in returns)

---

## Feature Engineering

### Discriminating variables

Currently, the model takes information from both the market stock information (monthly basis), and (quaterly) fundamentals:

#### Variables from market behaviours:
- ~~`ClosePrice`: Raw stock closing price~~ (Not included - provides no predictive value beyond returns)

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

#### Variables from Sentiment Information

Market sentiment indicators capture regime-dependent behavior and risk appetite that affect stock selection strategies.

**Currently Implemented:**

- **VIXpercentile** - Rolling 12-month percentile rank of VIX (0-1 scale), acts as regime indicator for market stress periods

**Alternative Approaches Tested:**

The following sentiment-based features were tested but **not included** in the final model due to performance degradation:

- **Multiplicative Interactions** - `mom12 × VIXpercentile` and `vol12 × VIXpercentile` (clipped to [0.25, 0.75])
    - Test AUC improved (+0.8pp) but portfolio Sharpe degraded significantly (0.92 → 0.86)
    - Probability range compressed 58% (0.561 → 0.235), destroying signal at portfolio selection extremes
    - Alpha vs random collapsed from 1.70 to 0.44

**Why VIXpercentile Works:**

VIXpercentile functions as portfolio insurance rather than a primary signal, with low average feature importance (ranked 13th of 13) but high marginal value during market stress:[^1]

- Normal markets (85% of time): Inactive, other features dominate
- High-volatility regimes (15% of time): Critical for regime-aware stock selection
- Prevents catastrophic picks during market stress without overfitting to historical patterns

The Random Forest learns non-linear regime-dependent interactions naturally:

- In calm markets (VIX percentile < 0.3): Value and momentum factors work well
- In volatile markets (VIX percentile > 0.7): Quality factors dominate

#### Combination of both features:
- ~~`LogMktCap`: log(price × shares outstanding)~~
> `LogMktCap` was originally considered but is not included in the final model. Market capitalization information is already captured through the fundamental variables (Equity, Assets) and price-based features (returns). Including raw log market cap provided no additional predictive value and was therefore excluded during feature selection.
***

### Feature Scaling
All features standardized using `StandardScaler` (fit on training set only)

---

## Target Variable

 ### Definition
Binary classification: Will each stock beat SPY over next 12 months?

```python
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

```python
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

```python
RandomForestClassifier(
    n_estimators=50,        # Balance: enough trees, not too slow
    max_depth=3,            # Constrained to prevent overfitting
    min_samples_split=0.02, # 2% of samples (regularization)
    min_samples_leaf=0.01,  # 1% of samples (regularization)
    max_features='log2',    # Feature sampling for diversity
    class_weight='balanced',# Handle any class imbalance
    random_state=42         # Reproducibility
)
```

**Why Random Forest?**
- Captures non-linear relationships (vs linear LogReg)
- Handles feature interactions naturally
- Robust to outliers
- Provides feature importance
- Superior to gradient boosting methods (XGBoost, LightGBM) for weak-signal problems

**Why Constrained (max_depth=3)?**
- Deep trees (10+) overfit badly (test AUC drops to 0.52)
- Shallow trees (3) generalize better
- Reduces training-test gap from 12pp to 7pp
- With weak signals (AUC ~0.53), constrained trees prevent overfitting to noise

 ### ~~Calibration: Isotonic Regression~~ (Not used in the final model)

```python
# NOT INCLUDED - Calibration degrades performance with VIX features
CalibratedClassifierCV(
estimator=RandomForestClassifier(...),
method='isotonic', # Non-parametric, flexible
cv=3 # Internal cross-validation
)
```
*Why Calibration Was Removed:**

While calibration improved the baseline model (without sentiment features), it **significantly degrades performance** when VIX features are included:


| Metric | RF + VIX | RF_cal + VIX | Change |
| :-- | :-- | :-- | :-- |
| Sharpe Ratio | 0.93 | 0.79 | -0.14 (-15%) |
| Annual Return | 20.2% | 17.8% | -2.4pp |
| Alpha vs Random | 1.72 | -0.69 | -2.41 (fails) |
| Prob Range | 0.214 | 0.546 | Over-stretched |

**Root Cause:**

Isotonic calibration stretches probabilities based on average frequencies, but VIX creates regime-dependent probability distributions. The stretched probabilities don't correlate with regime-specific returns, making stock selections worse than random during market stress periods.

**Historical Context (Baseline Model Without VIX):**

For reference, calibration *did* improve the baseline model before VIX features were added:

- Mean probability: 0.487 → 0.502 (correct)
- Probability std: 0.035 → 0.059 (better discrimination)
- Sharpe ratio: 0.71 → 0.80 (+13%)
- Alpha: 0.87% → 2.29% (+163%)

However, the addition of `VIXpercentile` as a regime indicator made calibration counterproductive, as the model now naturally learns regime-dependent probability distributions that should not be uniformly stretched.

> **Final Decision:** Using uncalibrated Random Forest (RF + `VIX`) as the production model.


## Portfolio Construction

 ### Signal Generation
For each stock on each month:
1. Generate probability p(beat SPY)
2. Apply 3-month rolling average (smoothing)
3. Rank stocks by smoothed probability

 ### Portfolio Assignment
The portfolio is currently configured as a **long-only** strategy with 100% equity exposure. This configuration differs from traditional long/short equity strategies (e.g., 130/30) but provides:
- Simpler implementation and lower regulatory complexity
- No short borrowing costs or constraints
- Alignment with long-term equity market returns plus alpha from stock selection

**Typical portfolio:**

- **Long:** Top 10% (~50 stocks)
- **No short positions**: Strategy focused on outperformance through stock selection
- **Benchmark** SPY (i.e, SP500 EFT) for performance comparison- Equal-weighted allocation
- 100% gross exposure, 100% net exposure

> Equal-weighted within long portfolio:
> - Each long position: 1/N of long capital (with N~50)


 ### Rebalancing
- **Frequency:** Monthly (last trading day)
- **Turnover:** ~42% average (reduced applying smoothing)
- **Transaction cost:** 10 bps per trade (0.1%)
- **3-month smoothing:** Reduces month-to-month prediction volatility by 58.7%.

---

## Performance Attribution

 ### Risk-Adjusted Returns
**Current Model (RF + VIX, Long-Only):**

- **Sharpe Ratio:** 0.93
- **Annual Return:** 20.2%
- **Volatility:** ~17.4% (monthly × √12)
- **Win Rate:** 69.8%
- **Max Drawdown:** -22.9%


  - Calculation: (Annual Return - Risk Free) / Annual Volatility
  - Risk-free rate: 0% (for simplicity)
  - Annualization: √12 for monthly data

 ### Alpha Calculation
**Alpha vs Random Portfolio:**

- **Alpha:** 1.72% annual (strategy beats random selection by 1.72pp)
- Comparison baseline: Random top-10% selection from same universe
- Demonstrates genuine stock-picking skill beyond market exposure

**Alpha vs SPY Benchmark:**
Linear regression: $\mathrm{Portfolio\_Return} = \alpha + \beta × \mathrm{SPY\_Return} + \varepsilon$

Expected results (long-only portfolio):

- **Beta:** ~1.0 (full market exposure, no hedging)
- **R**$^{2}$: High (strong correlation with market)
- **Alpha:** Excess return from stock selection skill

 ### Statistical Significance
**Sharpe Ratio T-test:**
**Sharpe Ratio T-test:**

- Based on monthly return series (2016-2025)
- Tests null hypothesis: Sharpe = 0
- Bonferroni adjustment for multiple model comparisons
- Result: Statistically significant outperformance

**Model Comparison:**

- RF + VIX (current): Sharpe 0.93, Alpha 1.72%
- Baseline (no sentiment): Sharpe 0.92, Alpha 1.70%
- Ensemble methods: Inferior performance (0.78-0.85 Sharpe)
---

## Risk Management

 ### Drawdown Analysis
**Current Model Performance:**

- **Max Drawdown:** -22.9%
- **COVID-19 Period (2020):** Improved regime awareness with VIX features
- **Bear Market (2022):** Better downside protection vs baseline (-22.9% vs -23.0%)

**Comparison to Benchmark:**

- Strategy drawdown: -22.9%
- SPY drawdown (same period): ~-24.0%
- Relative protection: +1.1pp better


 ### Regime Analysis

| Market Regime | VIX Percentile | Model Behaviour |
| :-- | :-- | :-- |
| Calm markets | < 0.3 (85% of time) | Value and momentum factors dominate |
| Volatile markets | > 0.7 (15% of time) | Quality factors emphasized |
| Crisis periods | > 0.9 | Defensive positioning, avoid growth traps |

**Fold-Level Performance:**

- Fold 1 (2016-2018): AUC 0.516 (stable period)
- Fold 2 (2018-2021, COVID): AUC 0.519 (+0.7pp with VIX vs baseline)
- Fold 3 (2021-2025): AUC 0.542 (strongest performance)

 ### Position Limits
**Current Implementation:**

- Max concentration per stock: ~2% (1/50 positions, equal-weighted)
- No explicit sector constraints (monitored via sector drift analysis)
- Universe constraint: S\&P 500 constituents only (quality filter)

**Risk Controls:**

- Equal weighting prevents concentration risk
- 10% universe selection  provides diversification
- Monthly rebalancing limits exposure drift
- 3-month smoothing reduces turnover and noise

---


## Production Considerations

### Data Latency

**Market Data (yfinance):**

- Frequency: Daily close prices
- Lag: 1 trading day (acceptable for monthly rebalancing)
- Reliability: High for S\&P 500 constituents

**Fundamental Data (SEC EDGAR API):**

- Frequency: Quarterly filings (10-Q, 10-K)
- Lag: Up to 45 days after quarter-end (regulatory filing deadlines)
- Forward-fill: Quarterly fundamentals interpolated to monthly frequency
- Impact: Fundamentals reflect trailing 1-2 quarters, not real-time

**Sentiment Data (VIX):**

- Frequency: Daily (continuous market data)
- Lag: 1 trading day
- Feature: 12-month rolling percentile calculated from historical VIX levels
- Reliability: High (exchange-traded, liquid index)

**Rebalancing Timing:**

- Target: Last trading day of month
- Flexibility: Can execute over final 2-3 trading days to manage market impact
- Data cutoff: Use data available as of month-end close


 ### Execution
**Portfolio Construction:**

- Computation time: <1 minute (Python script on standard hardware)
- Model inference: Batch prediction for SP500 stocks
- Position sizing: Simple equal-weight calculation (1/N)

**Order Generation:**

- Entry orders: ~23 new positions per month (45% turnover)
- Exit orders: ~23 positions closed per month
- Order type: Market-on-close or VWAP algorithms
- Execution window: Spread over final trading hours to minimize impact

**Transaction Costs:**

- Assumption: 10 bps per trade (includes bid-ask spread + market impact)
- Annual drag: ~0.5% from 42% monthly turnover
- Scalability: Realistic for portfolios <\$100M AUM


### Monitoring

**Model Health Checks:**

- Prediction distribution: Mean ~0.50, std ~0.06
- Probability range: Min ~0.43, Max ~0.64
> Alert if: Mean deviates >0.05 from 0.50 or std drops <0.04

**Portfolio Diagnostics:**

- Turnover: Monitor monthly, alert if >60% (excessive trading)
- Sector concentration: Alert if any sector >30% (sector bet risk)
- Universe coverage: Alert if <300 stocks available (data quality issue)
- Correlation with SPY: Expected ~0.85-0.90 for long-only

**Performance Tracking:**

- Daily: Portfolio NAV and returns
- Monthly: Sharpe ratio, alpha vs random, drawdown
- Quarterly: Feature importance stability, model drift detection
- Annual: Full performance attribution and strategy review

---

## Known Limitations

 ### Survivorship Bias
- Using current S&P 500 constituents may introduce bias
- Mitigation: Historical constituent data would improve accuracy
> Might be improved by using historical SP500 data, or commercial datasets

 ### Market Impact
- Large AUM would increase transaction costs beyond 10 bps
- Strategy works best for <$100M AUM

 ### Regime Dependency
This strategy may underperform in specific market regimes despite having VIX features in the training Possible cases:

1. **Extreme Momentum Markets:**
    - Example: 2020-2021 tech rally (low-quality growth stocks outperform)
    - Model emphasizes fundamentals, may miss pure momentum runs
    - VIX helps but doesn't fully capture speculative bubbles
2. **Market Dislocations Not in Training Data:**
    - Training period: 2010-2025 (post-GFC)
    - Missing: 2008 financial crisis, dot-com bubble, 1987 crash
    - Model may not generalize to unprecedented events
3. **Sector Rotation Events:**
    - Rapid sector rotations (e.g., energy spike 2022) may occur faster than monthly rebalancing
    - Model doesn't explicitly control sector exposure

**Mitigation:**

- `VIXpercentile` already provides some regime awareness
- Monthly rebalancing and the smoothing limits exposure to single-month anomalies
- Equal-weighting prevents concentration in mispriced sectors

 ### Factor Crowding
From the [feature importance](images/feature_importance.png) analysis:

- **Value factors:** BookToMarket (23.6% importance), ROA (13.3%)
- **Volatility:** vol12 (25.8% importance - highest)
- **Quality:** NetMargin (8.6%), ROE (7.2%)
- **Momentum:** Low importance (r1: 0.32%)

**Crowding Risks:**

1. **Volatility Factor Dominance:**
    - vol12 is top feature (25.8%)
    - Risk: Many quant funds use volatility-managed strategies
    - If crowded, alpha from volatility control may erode
2. **Value Factor Decay:**
    - BookToMarket still important (23.6%)
    - Value premium has weakened 2018-2021 (growth dominance)
    - Its premium may diminish further.
3. **Traditional Factor Exposure:**
    - Strategy uses well-documented factors (value, quality, low-vol)
    - These are widely known and traded by institutional investors
    - First-mover advantage has eroded since Fama-French research

**How this could be monitored:**

- Track correlation with factor ETFs (VLUE, QUAL, USMV)
- Monitor if strategy alpha declines as factors become crowded
- Consider adding alternative data (sentiment, alternative fundamentals) to differentiate

> One could also change the reliance on factor interactions and VIX-regime for other less-crowded factors, such as firm-specific sentiment, earnings call tone, supply chain data.

### Long-Only Constraints

Despite its simpler implementation and no short borrowing costs, tthe decision of constructing a 100% long portfolio carries the following limitations:

1. **No Short Side Alpha:**
    - Long-only captures alpha only from stock selection on long side
    - Traditional 130/30 strategies can harvest alpha from both longs and shorts
2. **Full Market Exposure:**
    - Beta ≈ 1.0 (cannot hedge market risk)
    - During bear markets, portfolio declines with market despite stock selection
    - Long/short strategies can maintain lower market beta (0.3-0.7)


## Future Improvements

1. **Extend fundamental data** before 2010
    - **Challenge:** SEC EDGAR API only available from 2010+
    - This could be achieved by using commercial data providers (Compustat, Bloomberg) or manual parsing of historical filings
    - **Expected Impact:** More training data, potentially stronger signals in earlier regimes

2. Include **alternative Data Sources**, such as News headlines, social media, analyst reports:
    - This could be achieved using NLP on earnings call transcripts, Twitter sentiment
    - **Expected Impact:** +0.02-0.03 Sharpe from behavioral signals
    - Sentiment data is difficult to obtain, noisy, and can produce potential overfitting

3. Extend to **International Markets**, such as European stocks (STOXX 600), or Asian markets (eg. Nikkei):
    - **Challenge:** Different accounting standards, data availability
    - **Benefit:** Test if signals generalize across regions
    - This could be implemented either via country-specific models or through a unified multi-region approach

4. Improve the **Risk Management**
    - Implement max sector exposure limits
    - Add volatility targeting (scale positions by risk)
    - Dynamic rebalancing (respond to market volatility)


<br><hr>
[Back to top](#technical-methodology)
