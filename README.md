# Financial-ML

This project ingests S\&P 500 constituents, downloads historical monthly close prices via a market data API, builds engineered features, and trains baseline classification models to predict whether a stock’s 12-month forward return will outperform a benchmark.
It can also fetch fundamentals for each company from public filings and persist tidy, time-aligned datasets for downstream modeling.

## Table of Contents
- [Features <div id='Features'/>](#Features_<div_id='Features'/>)
- [Project structure <div id='Structure'/>](#Project_structure_<div_id='Structure'/>)
- [Installation <div id='Installation'/>](#Installation_<div_id='Installation'/>)
- [Usage](#Usage)
- [Data pipeline details](#Data_pipeline_details)
- [Modeling and evaluation](#Modeling_and_evaluation)
    - [Discriminating variables](#discriminating_vars)
        - [Market variables](#market_vars)
        - [Fundamental variables](#fundamental_vars)
        - [Optional future extensions](#future_extensions)
    - [Modelling](#ml_models)
- [Outputs](#Outputs)
- [Notes and compliance](#Notes_and_compliance)







## Features <div id='Features'/>
<div id="Features_<div_id='Features'/>"></div>

- Fetch the current S\&P 500 list, normalize tickers for the market data API, and persist the symbols table.
- Download monthly adjusted close prices for all S\&P 500 tickers and the benchmark instrument, writing tidy CSVs for full or test universes.
- Retrieve selected fundamentals (assets, liabilities, equity, revenues, net income, EPS, shares outstanding) from public filings and save as a long-format CSV.
- Engineer momentum and volatility features, define an excess-return label vs the benchmark, and evaluate baseline classifiers with expanding-window time series splits.
- Produce out-of-fold predictions per date and ticker for analysis and diagnostics.


## Project structure <div id='Structure'/>
<div id="Project_structure_<div_id='Structure'/>"></div>

- Entrypoint and CLI flags live in the main module that dispatches data collection, training, and fundamentals jobs.
- Paths and output locations are centralized, with data/, figures/, and logs/ auto-created on first run.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.

## Installation <div id='Installation'/>
<div id="Installation_<div_id='Installation'/>"></div>

- Requires Python ≥ 3.10; core dependencies are declared in pyproject.toml (pandas, numpy, scikit-learn, yfinance, requests, pyarrow).
- Option A — Development (editable install): install the package from source so local changes are picked up.
- Option B — Reproducible (pinned): install exact versions from Requirements.txt, then install the package.
- It is recommended to create and activate a virtual environment and upgrade pip/setuptools before installing.
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
<div id="Usage"></div>

The CLI supports flags to update the S\&P 500 list, download market data, run fundamentals ingestion, enable a smaller test universe, and launch model training.

- Update symbols and prices, then train:

```
python -m financial_ml -nt -ni --train
```

Replace <package> with the actual package name that contains the modules shown here.

- Fetch fundamentals for the full universe:

```
python -m financial_ml -f
```

- Run a quicker workflow on a 50-ticker subset:

```
python -m financial_ml -nt -ni -f --test
```

Flags summary:

- -nt/--newtable: refresh the S\&P 500 symbols table.
- -ni/--newinfo: download or refresh market price history CSVs.
- -f/--fundamentals: retrieve and store selected fundamentals from filings.
- --train: run the modeling pipeline on engineered features.
- --test: operate on a smaller universe to speed up iteration.


## Data pipeline details
<div id="Data_pipeline_details"></div>

- Symbols: The S\&P 500 list is read from a public reference and saved to data/sp500_list.csv, with tickers normalized for downstream API compatibility.
- Prices: Monthly adjusted close prices are downloaded for all symbols and for a benchmark instrument, saving to data/sp500_values.csv (or data/sp500_values_test.csv in test mode).
- Fundamentals: Selected tags are retrieved from company filings, normalized as point-in-time series, de-duplicated by metric/unit/date, and written to data/sp500_fundamentals.csv (or the test variant).

- Label: Binary label indicates whether 12-month forward return exceeds the benchmark’s (S\&P 500) 12-month forward return.


## Modeling and evaluation
<div id="Modeling_and_evaluation"></div>

### Discriminating variables <div id="discriminating_vars"></div>

Currently, the model takes information from both the market stock information (monthly basis), and (quaterly) fundamentals:

#### Variables from market behaviours: <div id="market_vars"></div>

- r1 (1m return): Captures the most recent monthly price move, providing a highly responsive but noisy signal that helps models account for short‑term dynamics and potential reversal pressure.

- r12 (12m return): Summarizes the past year’s trend including the latest month, offering a strong baseline momentum proxy that can be tempered with risk controls for stability.

- mom121 (12m − 1m momentum): Focuses on medium‑term trend by excluding the most recent month, reducing short‑term reversal effects and typically improving persistence out of sample.

- vol3 (3m rolling std): Fast‑moving realized volatility over three months that reacts to recent shocks, useful for volatility‑managed scaling and down‑weighting unstable names.

- vol12 (12m rolling std): Slower, more structural risk estimate over a full year that complements vol3 by distinguishing transient turbulence from persistent volatility regimes.
#### Variables from fundamentals <div id="fundamental_vars"></div>

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

 #### Possible future extensions <div id="future_extensions"></div>
 
- Residual momentum: A stock’s trend after removing broad market/factor co‑movement, highlighting stock‑specific persistence rather than index‑driven moves.
- 12‑month drawdown: The percent distance of the current price from its highest level over the past year, summarizing recent loss severity and recovery state.
- Gross Profitability: $\frac{Sales-COGS}{Assets}$ (requires [COGS](https://en.wikipedia.org/wiki/Cost_of_goods_sold)) is a strong profitability proxy complementary to ROE/ROA in cross-sectional models.
- Accruals ([Sloan](https://quantpedia.com/strategies/accrual-anomaly)): requires cash flow from operations and current working-capital components to estimate accrual intensity, which is often predictive of returns.

### Modelling <div id="ml_models"></div>

- Models: Current considered models include:
    - L2 and L1 logistic regression with scaling and class weighting, with an option to extend to tree-based models (baseline).
    - Random forest
- Split: TimeSeriesSplit with n_splits=5 and a 36-month test window per fold provides an expanding-window backtest-like evaluation.
- Metrics and artifacts: For each model and fold, AUC is logged and out-of-fold predictions are written to data/oof_predictions.csv with columns [date, ticker, y_true, y_prob, y_pred, fold, model].


### Outputs
<div id="Outputs"></div>

- data/sp500_list.csv: symbols table used to drive downstream tasks.
- data/sp500_values.csv and data/sp500_values_test.csv: monthly close prices per ticker plus benchmark.
- data/sp500_fundamentals.csv and data/sp500_fundamentals_test.csv: normalized point-in-time fundamentals across selected tags.
- data/oof_predictions.csv: stacked out-of-fold predictions for evaluation and analysis.


### Notes and compliance
<div id="Notes_and_compliance"></div>

- Fundamentals ingestion uses a retry-enabled session and a descriptive User-Agent for responsible access to the filings API, and introduces a short sleep between requests.
- Tickers containing a dot are normalized with a dash for compatibility with the market data API, and the benchmark instrument is appended to the universe.
- Paths are resolved relative to a repository root two levels up from these modules, and output directories are created if missing.


<br><hr>
[Back to top](#financial-ml)
