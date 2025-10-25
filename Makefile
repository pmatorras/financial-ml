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

train-all:
	python -m financial_ml train 

# Full  test pipeline
test-ticker: market-debug fundamentals-debug train-debug
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

