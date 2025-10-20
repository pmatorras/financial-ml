# Financial-ML

This project ingests S\&P 500 constituents, downloads historical monthly close prices via a market data API, builds engineered features, and trains baseline classification models to predict whether a stock’s 12-month forward return will outperform a benchmark.
It can also fetch fundamentals for each company from public filings and persist tidy, time-aligned datasets for downstream modeling.

## Table of Contents
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data pipeline details](#data-pipeline-details)
- [Modelling](#modeling-and-evaluation)
    - [Discriminating variables](#discriminating-variables)
        - [From market behaviour](#variables-from-market-behaviours)
        - [From fundamentals](#variables-from-fundamentals)
    - [Modelling](#modelling)
- [Portfolio construction](#portfolio-construction)
- [Evaluation](#evaluation)
- [Outputs](#outputs)
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

- Entrypoint and CLI flags live in the main module (ˋsrc/financial_ml/ˋ) that dispatches data collection, training, and fundamentals jobs.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.
The current structure of ´src/financial_ml´ is as follows:
financial_ml/
│
├── core/
│   ├── utils.py                 # General helper functions common across modules
│   ├── config.py                # Global configuration loading (YAML/ENV)
│   └── logging_config.py        # Unified logging setup
│
├── data/
│   ├── loaders/
│   │   ├── price_data.py        # Market and price data loading
│   │   ├── fundamentals.py      # SEC/EDGAR or accounting data ingestion
│   │   └── macro.py             # IMF or macroeconomic dataset access
│   ├── preprocess/
│   │   ├── clean_data.py        # Cleaning and alignment of financial datasets
│   │   ├── merge_sources.py     # Merge multiple data feeds
│   │   └── label_data.py        # Create ML-ready labels (returns, regimes, etc.)
│   └── data_utils.py            # Common data manipulation utilities
│
├── features/
│   ├── build_features.py        # Generate technical & fundamental indicators
│   ├── feature_selection.py     # Statistical or ML-based feature filtering
│   └── transforms.py            # Normalization, scaling, and encoding
│
├── models/
│   ├── train_model.py           # Model training pipeline (RF, logistic, XGBoost)
│   ├── evaluate_model.py        # Cross-validation, scoring, diagnostics
│   ├── feature_importance.py    # Feature attribution extraction
│   └── portfolio.py             # Portfolio construction from predicted signals
│
├── viz/
│   ├── plots/
│   │   ├── performance_chart.py # Equity curves, ROC plots, feature plots
│   │   └── model_diagnostics.py # Residuals, SHAP, confusion matrices
│   └── dashboard/
│       ├── layout.py            # Dash layout components
│       ├── callbacks.py         # Dash interactivity logic
│       └── app.py               # Complete standalone dashboard entry point
│
├── cli/
│   ├── main.py                  # CLI entry: commands (market, fundamentals, train, plot)
│   └── arguments.py             # Argument parsing and command handling
│
├── wsgi.py                      # WSGI entry point for dashboard or API
└── __init__.py                  # Marks this directory as a package

This structure keeps the logic compartmentalized:
- ˋcore/ˋ centralises configuration and utilities. General paths are centralised through these scripts, with ˋdata/ˋ, ˋfigures/ˋ, and ˋlogs/ˋ auto-created on first run.
- ˋdata/ˋ manages loading, cleaning, and labeling.
- ˋfeatures/ˋ handles transformation and feature engineering.
- ˋmodels/ˋ encapsulates model workflows and portfolio logic.
- ˋviz/ˋ contains both charts and dashboards for performance and exploration.The 
- ˋcli/ˋ directory aligns with the earlier design’s modular entrypoint for reproducible workflows (market, fundamentals, train, plot...).

This makes the repository scalable for predictive modeling, portfolio backtesting, and visualization, while keeping all paths consistent with professional ML deployment standards


## Installation

- Requires Python ≥ 3.10; core dependencies are declared in pyproject.toml (pandas, numpy, scikit-learn, yfinance, requests, pyarrow).
- Option A — Development (editable install): install the package from source so local changes are picked up.
- Option B — Reproducible (pinned): install exact versions from ˋRequirements.txtˋ, then install the package.
- It is recommended to create and activate a virtual environment and ˋupgrade pip/setuptoolsˋ before installing.
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

## Usage


**Fetch S&P 500 constituents and market data:**
```bash
# Update constituents
python -m financial_ml markets --newtable

# Update historical market data
python -m financial_ml markets --newinfo

# Both together (with global flags placed before the subcommand)
python -m financial_ml --test -d info --newtable --newinfo
```
- The info subcommand separates refreshing the constituents list from downloading prices to make each step explicit and repeatable.

**Download fundamentals from SEC data:**
```bash
# Download/refresh fundamentals
python -m financial_ml fundamentals

# Test subset for faster iteration
python -m financial_ml --test fundamentals
```
- Fundamentals use the existing constituents file; refresh constituents first if needed via the info subcommand.

**Train models**:
```bash
# Train with market-only features
python -m financial_ml train

# Train including fundamentals-derived features
python -m financial_ml train --use-fundamentals

# Combine with test/debug
python -m financial_ml --test -d train --use-fundamentals
```

**Create portfolios**:
```bash
python -m financial_ml portfolio --m [Chosen model]
```



Notes:
- Global flags: 
    - `--test`: Run a reduced subset
    - `-d/--debug` for more verbose output.
- If the usage is unclear, run `--help` flag: `python -m financial_ml --help`
- One can also use `python -m financial_ml.main` to run it without callin __main__.py.  
- Constituents refresh (newtable) and market data refresh (newinfo) are intentionally separated from fundamentals and training so each step can be run independently.


## Data pipeline details


- Symbols: The S\&P 500 list is read from a public reference and saved to data/sp500_list.csv, with tickers normalized for downstream API compatibility.
- Prices: Monthly adjusted close prices are downloaded for all symbols and for a benchmark instrument, saving to data/sp500_values.csv (or data/sp500_values_test.csv in test mode).
- Fundamentals: Selected tags are retrieved from company filings, normalized as point-in-time series, de-duplicated by metric/unit/date, and written to data/sp500_fundamentals.csv (or the test variant). Some of the tags have different variants to account to each ticker reporting the information in slightly different ways.

- Label: Binary label indicates whether 12-month forward return exceeds the benchmark’s (S\&P 500) 12-month forward return.


## Modeling and evaluation

### Discriminating variables

Currently, the model takes information from both the market stock information (monthly basis), and (quaterly) fundamentals:

#### Variables from market behaviours:

- r1 (1m return): Captures the most recent monthly price move, providing a highly responsive but noisy signal that helps models account for short‑term dynamics and potential reversal pressure.

- r12 (12m return): Summarizes the past year’s trend including the latest month, offering a strong baseline momentum proxy that can be tempered with risk controls for stability.

- mom121 (12m − 1m momentum): Focuses on medium‑term trend by excluding the most recent month, reducing short‑term reversal effects and typically improving persistence out of sample.

- vol3 (3m rolling std): Fast‑moving realized volatility over three months that reacts to recent shocks, useful for volatility‑managed scaling and down‑weighting unstable names.

- vol12 (12m rolling std): Slower, more structural risk estimate over a full year that complements vol3 by distinguishing transient turbulence from persistent volatility regimes.
#### Variables from fundamentals

The following variables are taken from the stock fundamentals:

- Book-to-Market (B/M):  captures valuation relative to book value and is a canonical factor in asset pricing and cross-sectional models. $B/M=\frac{Equity}{Price\times Shares}$
- Return on Equity (ROE):  measures profitability to equity holders and proxies the profitability factor component in five-factor frameworks. $ROE=\frac{Net Income_{TTM}}{Equity}$
- Return on Assets (ROA):  complements ROE by controlling for capital structure and overall asset base. $ROA=\frac{Net Income_{TTM}}{Assets}$
- Net Margin: gauges earnings efficiency and is routinely used in fundamental screens and profitability diagnostics.  $Net Margin=\frac{Net Income_{TTM}}{Revenues_{TTM}}$
- Leverage:  captures balance-sheet risk and interacts with profitability and value in expected return models. $Leverage=\frac{Liabilities}{Assets}$
- Asset Growth (Investment):  maps to the investment factor where higher investment has been associated with lower average returns. $Inv=\frac{Assets_{t}-Assets_{t-4q}}{Assets_{t-4q}}$
- Net Share Issuance:  tracks dilution/buybacks and has documented predictive power for subsequent returns. $Issuance=\frac{Shares_{t}-Shares_{t-4q}}{Shares_{t-4q}}$
- Size (control):  provides a standard size control that stabilizes cross-sectional comparisons. $\log(Market Cap)=\log(Price\times Shares) $

 This set targets value, profitability, investment, leverage, size, and dilution, which align with widely used multi-factor models and documented cross-sectional return predictors.



### Modelling

- Models: Current considered models include:
    - L2 and L1 logistic regression with scaling and class weighting, with an option to extend to tree-based models (baseline).
    - Random forest
- Split: TimeSeriesSplit with n_splits=5 and a 36-month test window per fold provides an expanding-window backtest-like evaluation.
- Metrics and artifacts: For each model and fold, AUC is logged and out-of-fold predictions are written to data/oof_predictions.csv with columns [date, ticker, y_true, y_prob, y_pred, fold, model].

Each model is included as part of a pipeline, which sanitises invalid values, impute missing data, and scale features for linear models; trees use passthrough scaling by design.


### Portfolio construction

- Model scores are transformed into cross-sectional signals, assets are ranked each rebalance date, and long–short portfolios are formed from chosen quantiles (e.g., top/bottom deciles) using equal- or rank-weighting with position limits, neutrality constraints, and periodic rebalancing.[^3][^4][^1]
- Where appropriate, simple weights can be replaced by risk-aware optimization to target volatility and concentration constraints while preserving signal intent.


### Evaluation

- Model evaluation reports Information Coefficient and quantile hit rates for ranking quality, plus AUC/PR for directional classification, using walk-forward or expanding-window time-series cross-validation to prevent lookahead bias.
- Portfolio evaluation reports risk-adjusted performance (Sharpe/Sortino), drawdown profile (max drawdown/Calmar), turnover, and transaction-cost–adjusted returns, with rolling metrics and drawdown curves for diagnostics.

### Outputs

- data/sp500_list.csv: symbols table used to drive downstream tasks.
- data/sp500_values.csv and data/sp500_values_test.csv: monthly close prices per ticker plus benchmark.
- data/sp500_fundamentals.csv and data/sp500_fundamentals_test.csv: normalized point-in-time fundamentals across selected tags.
- data/oof_predictions.csv: stacked out-of-fold predictions for evaluation and analysis.


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
