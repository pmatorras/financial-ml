# Financial-ML

This project ingests S\&P 500 constituents, downloads historical monthly close prices via a market data API, builds engineered features, and trains baseline classification models to predict whether a stock’s 12-month forward return will outperform a benchmark.
It can also fetch fundamentals for each company from public filings and persist tidy, time-aligned datasets for downstream modeling.

## Table of Contents
- [Features](#features)
- [Project structure](#project-structure)
    - [Source code](#source-code)
    - [Generated dfirectories](#generated-directories)
    - [Module overview](#module-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
    - [Global flags](#global-flags)
- [Data Collection](#data-collection)
    - [Usage](#data-usage)
    - [Data Sources](#data-sources)
    - [Output files](#data-output)
    - [Target variable](#target-variable)
- [Modelling](#modelling-and-evaluation)
    - [Usage](#modelling-usage)
    - [Discriminating variables](#discriminating-variables)
    - [ML Models](#ml-models)
    - [Cross-Validation](#cross-validation)
    - [Output files](#modelling-output)
- [Evaluation](#evaluation)
    - [Usage](#evaluation-usage)
    - [Output files](#evaluation-output)

- [Portfolio construction and backtesting](#portfolio-construction-and-backtesting)
    - [Usage](#portfolio-usage)
    - [Portfolio construction](#portfolio-construction)
    - [Metrics](#metrics)
    - [Diagnostics](#diagnostics)
    - [Output files](#portfolio-output)

- [Possible future extensions](#possible-future-extensions)
- [Notes and compliance](#Notes_and_compliance)




## Features

- Fetch the current S\&P 500 list, normalize tickers for the market data API, and persist the symbols table.
- Download monthly adjusted close prices for all S\&P 500 tickers and the benchmark instrument, writing tidy CSVs for full or test universes.
- Retrieve selected fundamentals (assets, liabilities, equity, revenues, net income, EPS, shares outstanding) from public filings and save as a long-format CSV.
- Combine fundamentals and market information to produce ticker features (e.g momentum, volatility, ROA, ROE...).
- Define an excess-return land evaluate baseline classifiers with expanding-window time series splits.
- Produce out-of-fold predictions per date and ticker for analysis and diagnostics.
- Use the predictions for the best and worst performing tickers to generate portfolios following different strategies, to try and beat the market.
- Assess the model robustness and the qualities of the portfolios (e.g. Sharpe ratios,  profits, risks, volatity vs SPY). 


## Project structure

- Entrypoint and CLI flags live in the main module (`src/financial_ml/`) that dispatches data collection, training, and fundamentals jobs.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.

### Source Code
The current structure of `src/financial_ml` is as follows:
```
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

```
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

#### 📊 Data Module

Load, validate, and engineer features from market and fundamental data.

**Key Components:**

- **Loaders** - Read processed CSV data (market prices, fundamentals)
- **Features** - Calculate market indicators and fundamental ratios
- **Collectors** - Download data from yfinance and SEC EDGAR APIs
- **Validation** - Data quality checks and assertions


#### 🤖 Models Module

Define and train ML models with time-series cross-validation.

**Supported Models:**

- Logistic Regression (L1/L2 regularization)
- Random Forest Classifier

**Features:**

- Time-series CV to prevent lookahead bias
- Hyperparameter-tuned pipelines
- Model persistence with joblib


#### 📈 Evaluation Module

Analyze trained model performance and feature importance.

**Capabilities:**

- Load saved models without retraining
- Extract feature importance/coefficients
- Visualize model behavior


#### 💼 Portfolio Module

Construct portfolios from predictions and run comprehensive backtests.

**Features:**

- Long/short portfolio construction
- Prediction smoothing (reduce noise)
- Performance metrics (Sharpe, drawdown, returns)
- Diagnostic analysis (turnover, beta, model agreement)
- Compare to benchmark (SPY)


#### 🔧 Utils Module

Shared configuration, paths, and utility functions that can be accesible from all over the repository.

## Installation

- Requires Python ≥ 3.10; core dependencies are declared in pyproject.toml (pandas, numpy, scikit-learn, yfinance, requests, pyarrow).
- Option A — Development (editable install): install the package from source so local changes are picked up.
- Option B — Reproducible (pinned): install exact versions from `Requirements.txt`, then install the package.
- It is recommended to create and activate a virtual environment and `upgrade pip/setuptools` before installing.
- If console scripts are defined in pyproject, they will be available on PATH after installation.

Editable install:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -e .
```

Reproducible install with pins:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r Requirements.txt
pip install -e .
```

## Quick start

```bash
#1. Run the complete ML pipeline:
python -m financial_ml market
python -m financial_ml fundamentals

#2. Train models
python -m financial_ml train

#3. Anayse training results
python -m financial_ml analyze

#4. Create portfolio and backtests
python -m financial_ml portfolio --model rf
```

### Global Flags

Available for all commands:

**`--test`**
- Uses subset of ~50 stocks instead of full S&P 500
- Appends `_test` suffix to data files

**`--debug`**
- Enables verbose logging
- Designed for pipeline validation with minimal data (1-2 stocks)
- Saves artifacts to `debug/` directories with `_debug` suffix

**Examples:**
```bash
#Development mode (small subset)
python -m financial_ml train --test

#Debug mode (verbose logging + debug artifacts)
python -m financial_ml train --debug

#Combined for troubleshooting
python -m financial_ml train --test --debug
```

> **Tip:** Use `python -m financial_ml --help` to see all available commands and flags.
  

## Data Collection

<a id="data-usage"></a>
### Usage

```bash
# 1. Download S&P 500 constituent list and prices
python -m financial_ml market

# 2. Download company fundamentals from SEC EDGAR
python -m financial_ml fundamentals

# Test mode (subset of ~50 stocks)
python -m financial_ml market --test
python -m financial_ml fundamentals --test
```
**Command specific flags**
```bash
- `--newtable, -nt` - Refresh S&P 500 constituent list from public source
- `--newinfo, -ni` - Redownload all historical price data (ignores cache)

```
**Caching behaviour:** By default, data files are only downloaded if they don't already exist. Use `--newtable` and `--newinfo` to force refresh.

> **Design note:** Market data collection and fundamentals collection are separate commands to enable independent execution. You can update constituents and prices without re-fetching fundamentals, or vice versa.

### Data sources 
Two types of data are currently considered: market and fundamental data:
1. **Market Data** (via yfinance)

    - **Universe:** S\&P 500 constituents scraped from public reference
    - **Prices:** Monthly adjusted close prices for all symbols + SPY benchmark
    - **Normalization:** Ticker symbols standardized for API compatibility

2. **Fundamental Data** (via SEC EDGAR API)

    - **Source:** Company 10-K/10-Q filings
    - **Tags:** Selected US-GAAP financial metrics (revenues, assets, equity, etc.)
    - **Processing:** Point-in-time series, de-duplicated by metric/unit/date
    - **Variants:** Multiple tag variants to handle company-specific reporting differences

> **Note:** Some fundamentals use different US-GAAP tag variants (e.g., `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`) to capture data across different reporting formats.

<a id="data-output"></a>
### Output Files

| File | Description | 
| :-- | :-- | 
| `data/market/sp500_list.csv` | S\&P 500 constituent tickers | 
| `data/market/sp500_prices.csv` | Monthly adjusted close prices | 
| `data/fundamentals/sp500_fundamentals.csv` | Company fundamentals from SEC | 

### Target Variable

**Binary classification:** Predict whether a stock will outperform the S\&P 500 benchmark.

**Label definition:** `y = 1` if stock's 12-month forward return exceeds SPY's 12-month forward return, else `y = 0`.

This creates a **relative momentum** signal focused on identifying stocks that beat the market, suitable for long/short portfolio construction.



## Modelling

<a id="modelling-usage"></a>
### Usage

```bash
# Train with fundamentals (default)
python -m financial_ml train

# Market data only (skip fundamentals)
python -m financial_ml train --no-fundamentals
```

**Command specific tags**
- `--no-fundamentals` - Train using only market data (excludes fundamental ratios)
### Discriminating variables

Currently, the model takes information from both the market stock information (monthly basis), and (quaterly) fundamentals:

#### Variables from market behaviours:

- `r1` (1m return): Captures the most recent monthly price move, providing a highly responsive but noisy signal that helps models account for short‑term dynamics and potential reversal pressure.

- `r12` (12m return): Summarizes the past year’s trend including the latest month, offering a strong baseline momentum proxy that can be tempered with risk controls for stability.

- `mom121` (12m − 1m momentum): Focuses on medium‑term trend by excluding the most recent month, reducing short‑term reversal effects and typically improving persistence out of sample.

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

***

### ML Models 

Three binary classifiers predict monthly stock outperformance vs. SPY benchmark:

- **Logistic Regression (L1)** - Lasso regularization, `C=0.5`
- **Logistic Regression (L2)** - Ridge regularization, `C=1.0`
- **Random Forest** - 50 trees, `max_depth=4`

> **Configuration:** See [`models/definitions.py`](src/financial_ml/models/definitions.py) for complete pipeline specifications.

**Preprocessing pipeline:**

- Sanitize infinite values → Replace with NaN
- Impute missing values → Median strategy
- Scale features → `StandardScaler` (linear models only)
- Balance classes → `class_weight='balanced'`


### Cross-Validation

Time-series split with **3 folds** and **36-month test windows** per fold. This expanding-window approach prevents lookahead bias and simulates realistic backtesting conditions.

**Metric:** AUC-ROC score logged for each model and fold.

<a id="modelling-output"></a>
### Output Files

| File | Description |
| :-- | :-- |
| `data/predictions/production/predictions_{model}.csv` | Out-of-fold predictions with `date`, `ticker`, `y_true`, `y_prob`, `y_pred`, `fold`, `model` |
| `models/production/{model}.pkl` | Trained model artifacts (serialized with joblib) |
| `models/production/feature_names.txt` | List of features used in training |


***

## Evaluation

One can analyse trained model feature importance and coefficients without retraining, simply by retrieving the models training information.

<a id="evaluation-usage"></a>
### Usage

```bash
# Analyse all trained models
python -m financial_ml analyze

# Loads models from models/production/ by default
# Use --debug to load models from debug directory
python -m financial_ml analyze --debug
```

<a id="evaluation-output"></a>
### Output

- **Feature importance plots** - Random Forest feature rankings saved to `figures/`
- **Coefficient plots** - Logistic Regression coefficients saved to `figures/`
- **CSV exports** - Feature importance/coefficients saved to `results/feature_importance/`


### What's Analysed

- **Random Forest** - Feature importance (Gini impurity reduction)
- **Logistic Regression** - Coefficient magnitudes and signs
- **Top features** - Ranked by contribution to predictions

> **Implementation:** See [`evaluation/feature_analysis.py`](src/financial_ml/evaluation/feature_analysis.py)
***

## Portfolio Construction and backtesting

Construct long/short portfolios from model predictions and evaluate performance against SPY benchmark.

<a id="portfolio-usage"></a>
### Usage

```bash
# Run backtest with specific model
python -m financial_ml portfolio --model rf
python -m financial_ml portfolio --model logreg_l2
python -m financial_ml portfolio --model logreg_l1
```


### Portfolio Construction

**Strategy:**

- Long top 10% of stocks by predicted probability
- Short bottom 10% of stocks
- Equal-weighted positions within each leg
- Monthly rebalancing

**Smoothing:** Optional exponential smoothing (`alpha=0.3`) to reduce prediction noise and turnover.

### Metrics

| Metric | Description |
| :-- | :-- |
| **Cumulative Return** | Total portfolio return over backtest period |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Turnover** | Average monthly portfolio churn |
| **Beta to SPY** | Market exposure and correlation |

### Diagnostics

- **Model agreement** - How often models agree on stock direction
- **Prediction stability** - Temporal consistency of signals
- **Beta exposure** - Long/short leg market sensitivity

<a id="portfolio-output"></a>
### Output

- **Performance charts** - Cumulative returns and drawdown plots saved to `figures/`
- **Backtest results** - Metrics printed to console

> **Implementation:** See [`portfolio/backtest.py`](src/financial_ml/portfolio/backtest.py)

***

### Possible future extensions
 
- Residual momentum: A stock’s trend after removing broad market/factor co‑movement, highlighting stock‑specific persistence rather than index‑driven moves.
- 12‑month drawdown: The percent distance of the current price from its highest level over the past year, summarizing recent loss severity and recovery state.
- Gross Profitability: $\frac{Sales-COGS}{Assets}$ (requires [COGS](https://en.wikipedia.org/wiki/Cost_of_goods_sold)) is a strong profitability proxy complementary to ROE/ROA in cross-sectional models.
- Accruals ([Sloan](https://quantpedia.com/strategies/accrual-anomaly)): requires cash flow from operations and current working-capital components to estimate accrual intensity, which is often predictive of returns.


## Notes and compliance

- Fundamentals ingestion uses a retry-enabled session and a descriptive User-Agent for responsible access to the filings API, and introduces a short sleep between requests.
- Tickers containing a dot are normalized with a dash for compatibility with the market data API, and the benchmark instrument is appended to the universe.
- Paths are resolved relative to a repository root two levels up from these modules, and output directories are created if missing.


<br><hr>
[Back to top](#financial-ml)
