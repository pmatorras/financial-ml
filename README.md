## Financial-ML

### Overview

This project ingests S\&P 500 constituents, downloads historical monthly close prices via a market data API, builds engineered features, and trains baseline classification models to predict whether a stock’s 12-month forward return will outperform a benchmark.
It can also fetch fundamentals for each company from public filings and persist tidy, time-aligned datasets for downstream modeling.

### Features

- Fetch the current S\&P 500 list, normalize tickers for the market data API, and persist the symbols table.
- Download monthly adjusted close prices for all S\&P 500 tickers and the benchmark instrument, writing tidy CSVs for full or test universes.
- Retrieve selected fundamentals (assets, liabilities, equity, revenues, net income, EPS, shares outstanding) from public filings and save as a long-format CSV.
- Engineer momentum and volatility features, define an excess-return label vs the benchmark, and evaluate baseline classifiers with expanding-window time series splits.
- Produce out-of-fold predictions per date and ticker for analysis and diagnostics.


### Project structure

- Entrypoint and CLI flags live in the main module that dispatches data collection, training, and fundamentals jobs.
- Paths and output locations are centralized, with data/, figures/, and logs/ auto-created on first run.
- Market data ingestion and symbol management are encapsulated in the markets module, and fundamentals ingestion in the fundamentals module.
- The training pipeline, feature engineering, labeling, cross-validation, and metric reporting are implemented in the train module.


### Installation

- Python dependencies used by the code include pandas, numpy, scikit-learn, yfinance, requests, and urllib3.
- Install these packages in a virtual environment before running the commands below.

Example:

```
pip install pandas numpy scikit-learn yfinance requests urllib3
```


### Usage

The CLI supports flags to update the S\&P 500 list, download market data, run fundamentals ingestion, enable a smaller test universe, and launch model training.

- Update symbols and prices, then train:

```
python -m <package>.main -nt -ni --train
```

Replace <package> with the actual package name that contains the modules shown here.

- Fetch fundamentals for the full universe:

```
python -m <package>.main -f
```

- Run a quicker workflow on a 50-ticker subset:

```
python -m <package>.main -nt -ni -f --test
```

Flags summary:

- -nt/--newtable: refresh the S\&P 500 symbols table.
- -ni/--newinfo: download or refresh market price history CSVs.
- -f/--fundamentals: retrieve and store selected fundamentals from filings.
- --train: run the modeling pipeline on engineered features.
- --test: operate on a smaller universe to speed up iteration.


### Data pipeline details

- Symbols: The S\&P 500 list is read from a public reference and saved to data/sp500_list.csv, with tickers normalized for downstream API compatibility.
- Prices: Monthly adjusted close prices are downloaded for all symbols and for a benchmark instrument, saving to data/sp500_values.csv (or data/sp500_values_test.csv in test mode).
- Fundamentals: Selected tags are retrieved from company filings, normalized as point-in-time series, de-duplicated by metric/unit/date, and written to data/sp500_fundamentals.csv (or the test variant).
- Engineered features: r1 (1m return), r12 (12m return), mom121 (12m − 1m momentum), vol3 (3m rolling std of 1m returns), vol12 (12m rolling std).
- Label: Binary label indicates whether 12-month forward return exceeds the benchmark’s 12-month forward return.


### Modeling and evaluation

- Models: Baseline pipelines include L2 and L1 logistic regression with scaling and class weighting, with an option to extend to tree-based models.
- Split: TimeSeriesSplit with n_splits=5 and a 36-month test window per fold provides an expanding-window backtest-like evaluation.
- Metrics and artifacts: For each model and fold, AUC is logged and out-of-fold predictions are written to data/oof_predictions.csv with columns [date, ticker, y_true, y_prob, y_pred, fold, model].


### Outputs

- data/sp500_list.csv: symbols table used to drive downstream tasks.
- data/sp500_values.csv and data/sp500_values_test.csv: monthly close prices per ticker plus benchmark.
- data/sp500_fundamentals.csv and data/sp500_fundamentals_test.csv: normalized point-in-time fundamentals across selected tags.
- data/oof_predictions.csv: stacked out-of-fold predictions for evaluation and analysis.


### Notes and compliance

- Fundamentals ingestion uses a retry-enabled session and a descriptive User-Agent for responsible access to the filings API, and introduces a short sleep between requests.
- Tickers containing a dot are normalized with a dash for compatibility with the market data API, and the benchmark instrument is appended to the universe.
- Paths are resolved relative to a repository root two levels up from these modules, and output directories are created if missing.