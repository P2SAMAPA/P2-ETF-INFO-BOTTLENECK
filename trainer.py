import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from feature_engineering import compute_features
from info_bottleneck import VariationalIB

def prepare_data(features, targets):
    """Flatten across days and ETFs."""
    # features: (T, n_etfs, n_feat)
    # targets: (T, n_etfs)
    T, n_etfs, n_feat = features.shape
    X = features.reshape(-1, n_feat)
    y = targets.reshape(-1)
    # Remove NaN
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Information Bottleneck) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 60:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        macro_df = data_manager.get_macro_data(df)
        if macro_df.empty:
            macro_df = pd.DataFrame(0, index=returns.index, columns=config.MACRO_COLUMNS)

        # For each window, train a model and record predictions
        best_per_etf = {}  # ticker -> (best_pred_return, best_window)
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 60:
                print(f"  Skipping window {win}d (insufficient data)")
                continue

            # Build daily features and targets for this window
            daily_features = []
            daily_targets = []
            start_idx = max(0, len(returns) - win - 50)
            for i in range(start_idx, len(returns) - 1):
                window_returns = returns.iloc[:i+1]
                if len(window_returns) < 60:
                    continue
                features, _, _ = compute_features(window_returns, macro_df, window=config.FEATURE_WINDOW)
                target = returns.iloc[i+1].values
                daily_features.append(features)
                daily_targets.append(target)

            if len(daily_features) < 50:
                print(f"  Not enough daily samples for window {win}d")
                continue

            X_train_orig = np.array(daily_features)  # (T, n_etfs, n_feat)
            y_train_orig = np.array(daily_targets)   # (T, n_etfs)
            X_train, y_train = prepare_data(X_train_orig, y_train_orig)
            # split into train/val
            split = int(0.8 * len(X_train))
            X_tr, X_val = X_train[:split], X_train[split:]
            y_tr, y_val = y_train[:split], y_train[split:]

            # Model
            input_dim = X_train.shape[1]
            model = VariationalIB(input_dim, config.HIDDEN_DIM, config.LATENT_DIM, beta=config.BETA)
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

            # Training
            for epoch in range(config.EPOCHS):
                model.train()
                idx = torch.randperm(len(X_tr))
                total_loss = 0.0
                for i in range(0, len(idx), config.BATCH_SIZE):
                    batch_idx = idx[i:i+config.BATCH_SIZE]
                    Xb = X_tr[batch_idx]
                    yb = y_tr[batch_idx]
                    loss, _, _ = model.loss(Xb, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                # Validation
                model.eval()
                with torch.no_grad():
                    val_loss, _, _ = model.loss(X_val, y_val)
                if (epoch+1) % 10 == 0:
                    print(f"    Window {win}d, epoch {epoch+1}: train loss {total_loss/len(idx):.4f}, val loss {val_loss:.4f}")

            # Predict for the most recent day
            last_features, etf_names, _ = compute_features(returns, macro_df, window=config.FEATURE_WINDOW)
            last_tensor = torch.tensor(last_features, dtype=torch.float32)
            with torch.no_grad():
                pred_returns = model.predict(last_tensor).numpy()
            # Store prediction for each ETF
            for i, ticker in enumerate(etf_names):
                pred = pred_returns[i]
                if ticker not in best_per_etf or pred > best_per_etf[ticker][0]:
                    best_per_etf[ticker] = (pred, win)

            # Store window result for dashboard
            window_results[win] = {
                "n_samples": len(X_train),
                "val_loss": float(val_loss)
            }

        if not best_per_etf:
            print("  No valid windows")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort ETFs by best predicted return
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = []
        full_scores = {}
        for ticker, (pred, win) in sorted_etfs[:config.TOP_N]:
            top_etfs.append({
                "ticker": ticker,
                "pred_return": float(pred),
                "best_window": win
            })
            full_scores[ticker] = {
                "pred_return": float(pred),
                "best_window": win
            }
        print(f"  Top 3 ETFs by predicted return (best window): {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/info_bottleneck_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Information Bottleneck Engine complete ===")

if __name__ == "__main__":
    main()
