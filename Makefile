# Set default values
TICKER ?= DPZ

.PHONY: test-dpz market fundamentals train clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  test-stock    - Run full pipeline for a given stock"
	@echo "  market        - Fetch market data"
	@echo "  fundamentals  - Process fundamentals"
	@echo "  train         - Train model"
	@echo "  clean         - Remove cache files"



market-debug:
	python -m financial_ml market -d --ticker $(TICKER) -ni

fundamentals-debug:
	python -m financial_ml fundamentals -d --ticker $(TICKER)

train-debug:
	python -m financial_ml train -d

# Full  test pipeline with debug mode
test: market-debug fundamentals-debug train-debug

#Full deployment
market:
	python -m financial_ml market

fundamentals:
	python -m financial_ml fundamentals

sentiment:
	python -m financial_ml sentiment

data: market, fundamentals, sentiment
train:
	python -m financial_ml train 

analyze:
	python -m financial_ml  analyze

portfolio:
	python -m financial_ml  portfolio

backtest: analyze, portfolio

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

optimise-depth:
	python -m financial_ml train --do-sentiment --model rf --tree-depth 3
	python -m financial_ml train --do-sentiment --model rf --tree-depth 4
	python -m financial_ml train --do-sentiment --model rf --tree-depth 5

optimise-max-features:
	python -m financial_ml train --do-sentiment --model rf --tree-max-features log2
	python -m financial_ml train --do-sentiment --model rf --tree-max-features sqrt
	python -m financial_ml train --do-sentiment --model rf --tree-max-features 0.3
	python -m financial_ml train --do-sentiment --model rf --tree-max-features 0.4

optimise-nestimators:
	python -m financial_ml train --do-sentiment --model rf --tree-nestimators 50
	python -m financial_ml train --do-sentiment --model rf --tree-nestimators 100
	python -m financial_ml train --do-sentiment --model rf --tree-nestimators 200

optimise-maxsamples:
	python -m financial_ml train --do-sentiment --model rf
	python -m financial_ml train --do-sentiment --model rf --tree-max-samples 0.8
	python -m financial_ml train --do-sentiment --model rf --tree-max-samples 0.9
