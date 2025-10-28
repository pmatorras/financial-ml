import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from financial_ml.utils.config import (
    DEBUG_DIR, 
    FUNDA_KEYS, 
    MARKET_KEYS, 
    CANONICAL_CONCEPTS, 
    N_SPLITS
    )
from financial_ml.utils.paths import (
    get_features_file,
    get_model_file,
    get_prediction_file
    ) 
from financial_ml.utils.logging import save_training_summary
from financial_ml.models.definitions import get_model_name, get_models
from financial_ml.data import (
    require_non_empty,
    load_market,
    load_fundamentals,
    load_sentiment,
    calculate_market_features,
    calculate_sentiment_features,
    compute_fundamental_ratios, 
    create_binary_labels,
    to_monthly_ffill,
    widen_by_canonical
    ) 

from financial_ml.data.features import (
    calculate_enhanced_features,
    broadcast_market_feature_to_stocks,
    calculate_vix_interactions
    )




def train(args):
    prices, spy_benchmark = load_market(args)
    require_non_empty(prices, "prices")
    tickers = prices.columns.tolist()

    market_features = calculate_market_features(prices,args)

    input_vars = list(market_features.values())
    market_keys = MARKET_KEYS 
    funda_keys = []
    df_keys = market_keys.copy()    
    # safe copy for concat keys

    sentiment_features = None
    sentiment_keys = []
    if args.do_sentiment:
        try:
            sentiment = load_sentiment(args)
            sentiment_features = calculate_sentiment_features(sentiment, args.trim_mode) 
            
            for name, series in sentiment_features.items():
                df = broadcast_market_feature_to_stocks(series, tickers)
                input_vars.append(df)
                sentiment_keys.append(name)
            
            if args.verbose or args.debug: print(f"Added sentiment features: {sentiment_keys}")
        except FileNotFoundError:
            print("⚠️ No sentiment data found")

    # Create interaction terms to capture regime-dependent effects
    interaction_keys = []
    if args.do_sentiment and sentiment_features is not None:
        
        interactions = calculate_vix_interactions(
            market_features, 
            sentiment_features, 
            tickers
        )
        
        for name, df in interactions.items():
            input_vars.append(df)
            interaction_keys.append(name)
        
        if interaction_keys and (args.verbose or args.debug):
            print(f"Added interaction features: {interaction_keys}")
    
    # Update df_keys to include interactions
    df_keys = market_keys + sentiment_keys + interaction_keys

    if not args.only_market:
        funda_keys = FUNDA_KEYS
        fundamentals = load_fundamentals(args)
        if args.debug: fundamentals.to_csv(DEBUG_DIR/"funda.csv")
        monthly = to_monthly_ffill(fundamentals, maxdate="2025-12-31", freq="BME")
        if args.debug: monthly.to_csv(DEBUG_DIR/"monthly.csv")
        monthly.name = "CommonStockSharesOutstanding"

        require_non_empty(monthly, "CommonStockSharesOutstanding")

        for tag in CANONICAL_CONCEPTS:
            wide = widen_by_canonical(prices=prices, monthly=monthly, canonical_key=tag)
            input_vars.append(wide)
            df_keys.append(tag)
        input_keys = market_keys +  funda_keys + sentiment_keys + interaction_keys

    if len(df_keys) != len(input_vars):
        print("keys and variables not the same size!")
        exit()

    if args.debug:
        for k, df in zip(df_keys, input_vars):
            print(k, type(df.index), df.index.min(), df.index.max(), df.shape)
    feat = pd.concat(input_vars, axis=1, keys=df_keys)
    if args.debug: feat.to_csv(DEBUG_DIR/"feat.csv")

    y = create_binary_labels(prices, spy_benchmark)

    # Align and stack panel to long format
    feat_long = feat.stack(level=1,future_stack=True)
    if "canonical_key" not in feat_long.columns:
        feat_long["canonical_key"] = ""
    require_non_empty(feat_long, "feat_long")
    if args.debug: print(feat_long.keys(), feat_long)

    if not args.only_market:
        compute_fundamental_ratios(feat_long, args)

    if hasattr(args, 'use_enhanced') and args.use_enhanced:
        print("\n=== CALCULATING ENHANCED FEATURES ===")
        
        feat_reset = feat_long.reset_index()
        enhanced_dict = calculate_enhanced_features(feat_reset)
        
        # Features that will be replaced by their ranks
        features_to_rank = ['BookToMarket', 'ROE', 'ROA', 'mom121', 'vol12', 'LogMktCap', 'r12']
        
        # Remove raw versions if we're adding ranked versions
        for feat in features_to_rank:
            rank_key = f'{feat}_rank'
            if rank_key in enhanced_dict and feat in input_keys:
                print(f"  Replacing {feat} with {rank_key}")
                input_keys.remove(feat)  # Remove raw
        
        # Add enhanced features
        for key, values in enhanced_dict.items():
            feat_long[key] = values.values
            if key not in input_keys:
                input_keys.append(key)

    #Add the Y to the datafrane
    y_long = y.stack().rename("y")
    df = feat_long.join(y_long, how="inner")
    require_non_empty(df, "feat_join_ylong")

    # Train/validation split: expanding window via TimeSeriesSplit
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]:"date",df.columns[1]:"ticker"})
    # Guards: types, sort, chronology
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    assert df["date"].is_monotonic_increasing, "Dates must be sorted ascending"  # guard
    require_non_empty(df, "df_sorted")

    # Show which features in input_keys have the most NaN
    if args.verbose or args.debug:
        print("\nNaN counts in input_keys:")
        nan_in_input_keys = df[input_keys].isnull().sum().sort_values(ascending=False)
        print(nan_in_input_keys[nan_in_input_keys > 0])

    # Show which tickers are getting dropped
    if 'ticker' in df.columns:
        tickers_before = set(df['ticker'].unique())
        df_after = df.dropna(subset=input_keys)
        tickers_after = set(df_after['ticker'].unique())
        dropped_tickers = tickers_before - tickers_after
        print(f"\nTickers dropped: {len(dropped_tickers)} out of {len(tickers_before)}")
        if (len(dropped_tickers)>0): print(f"Sample dropped tickers: {sorted(list(dropped_tickers))[:20]}")
        if (len(dropped_tickers)>20): print(f"Sample surviving tickers: {sorted(list(tickers_after))[:20]}")

        # For a few dropped tickers, show which features are NaN
        if dropped_tickers and (args.verbose or args.debug):
            sample_dropped = list(dropped_tickers)[:3]
            print(f"\nWhy these tickers were dropped:")
            for ticker in sample_dropped:
                ticker_rows = df[df['ticker'] == ticker]
                missing_features = ticker_rows[input_keys].isnull().sum()
                missing_features = missing_features[missing_features > 0]
                print(f"  {ticker}: missing {len(missing_features)} features")
                print(f"    {missing_features.head(10).to_dict()}")


    df = df.dropna(subset=input_keys)
    if args.verbose or args.debug: print(f"\nafter second dropna: {len(df)} rows")
    if not args.only_market:
        if 'ticker' in df.columns and (args.verbose or args.debug):
            print(f"  Unique tickers: {df['ticker'].nunique()}")
            print(f"  Sample tickers: {sorted(df['ticker'].unique())[:20]}")

    # Define features to exclude
    exclude_features = ['ClosePrice', 'LogMktCap']

    # Filter out excluded features
    input_keys = [key for key in input_keys if key not in exclude_features]
    print("Input variables to train", input_keys)

    X = df[input_keys].to_numpy()
    Y = df["y"].to_numpy()

    dates = pd.to_datetime(df["date"])
    order = np.argsort(dates.values)
    X, Y = X[order], Y[order]
    unique_dates = np.array(sorted(df["date"].unique()))

    tscv = TimeSeriesSplit(n_splits=N_SPLITS,test_size=36,gap=1)

    pred_rows = []
    models = get_models(args)
    trained_models = {}
    fold_results = {}
    print(f"Available models [{len(models)}]: {', '.join(sorted(models))}")
    models_to_run = models.items() if 'all' in args.model else [(args.model, models[args.model])]

    for name, pipe in models_to_run:
        print(f"Training {get_model_name(name)}")
        aucs_test = []
        aucs_train = []
        fold_results[name] = []
        for  split_id, (tr_d, te_d) in enumerate(tscv.split(unique_dates), 1):
            train_dates = set(unique_dates[tr_d])
            test_dates  = set(unique_dates[te_d])

            tr_mask = df["date"].isin(train_dates).to_numpy()
            te_mask = df["date"].isin(test_dates).to_numpy()

            Xtrain, Xtest = X[tr_mask], X[te_mask]
            Ytrain, Ytest = Y[tr_mask], Y[te_mask]
            # Log fold ranges and test class balance
            tr_start, tr_end = min(train_dates), max(train_dates)
            te_start, te_end = min(test_dates), max(test_dates)
            classes, counts = np.unique(Ytest, return_counts=True)
            if args.debug:
                print(f"[{name} | Fold {split_id}] "
                    f"Train {tr_start.date()} → {tr_end.date()} | "
                    f"Test {te_start.date()} → {te_end.date()} | "
                    f"Test class counts {dict(zip(classes, counts))}"
                    )
            print(f"[{name} | Fold {split_id}] "
                  f"Train {tr_start.date()} → {tr_end.date()} | "
                  f"Test {te_start.date()} → {te_end.date()} | "
                    )
            pipe.fit(Xtrain, Ytrain)
            p_train = pipe.predict_proba(Xtrain)[:,1]
            p_test = pipe.predict_proba(Xtest)[:,1]
            train_auc = roc_auc_score(Ytrain, p_train)
            test_auc = roc_auc_score(Ytest, p_test)
            aucs_train.append(train_auc)
            aucs_test.append(test_auc)
            fold_df = df.loc[te_mask, ["date","ticker"]].copy()
            fold_df["y_true"] = Ytest
            fold_df["y_prob"] = p_test
            fold_df["y_pred"] = (p_test >= 0.5).astype(int)
            fold_df["fold"] = split_id
            fold_df["model"] = name
            pred_rows.append(fold_df)
            
            # Collect fold metadata
            fold_info = {
                'fold': split_id,
                'train_start': tr_start.strftime('%Y-%m-%d'),
                'train_end': tr_end.strftime('%Y-%m-%d'),
                'test_start': te_start.strftime('%Y-%m-%d'),
                'test_end': te_end.strftime('%Y-%m-%d'),
                'train_auc': train_auc,
                'test_auc': test_auc,
                'train_samples': len(Ytrain),
                'test_samples': len(Ytest),
                'test_positive_rate': Ytest.mean()
            }
            fold_results[name].append(fold_info)

        trained_models[name] = pipe
        if 'rf' in name: print(f"Tree depth={args.tree_depth}, max features={args.tree_max_features}, nestimators: {args.tree_nestimators}, max samples: {args.tree_max_samples}")
        print("Baseline logistic train AUC ("+name+"):", np.round(aucs_train, 3).tolist(), f"mean: {np.round(np.mean(aucs_train),3)} +- {np.round(np.std(aucs_train),3)}")
        print("Baseline logistic test  AUC ("+name+"):", np.round(aucs_test, 3).tolist(), f"mean: {np.round(np.mean(aucs_test),3)} +- {np.round(np.std(aucs_test),3)}")


    if args.save:
        #Save prediction files
        pred_df = pd.concat(pred_rows, ignore_index=True)
        predpath = get_prediction_file(args)
        pred_df.to_csv(predpath, index=False)

        # Save trained models and information
        save_training_summary(fold_results, input_keys, args)
        for model_name, model_pipeline in trained_models.items():
            models_path = get_model_file(args, model_name)
            joblib.dump(model_pipeline, models_path)
            print(f"Saved {model_name} to {models_path}")
        feature_path = get_features_file(args)
        with open(feature_path, 'w') as f:
            f.write('\n'.join(input_keys))

    