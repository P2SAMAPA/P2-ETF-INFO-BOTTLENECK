import pandas as pd
import numpy as np
import config

def compute_features(returns_df, macro_df, window=20):
    """
    For each ETF, compute features:
    - recent returns (1d, 5d, 21d)
    - volatility (21d std)
    - momentum (21d cumulative return)
    - max drawdown (21d)
    - skew, kurt (60d)
    - macro sensitivities (rolling correlation over 60d)
    - graph centralities (degree, eigenvector from correlation graph over 60d)
    Returns feature matrix (n_etfs, n_features) and feature names.
    """
    n_etfs = returns_df.shape[1]
    etf_names = returns_df.columns.tolist()
    features = []

    # Recent returns
    ret_1d = returns_df.iloc[-1].values
    ret_5d = returns_df.iloc[-5:].mean().values
    ret_21d = returns_df.iloc[-21:].mean().values
    vol_21d = returns_df.iloc[-21:].std().values
    mom_21d = (returns_df.iloc[-21:].sum()).values
    # Max drawdown (21d)
    drawdown = []
    for etf in etf_names:
        series = returns_df[etf].iloc[-21:]
        cumret = (1 + series).cumprod()
        peak = cumret.expanding().max()
        dd = (peak - cumret) / peak
        drawdown.append(dd.max())
    drawdown = np.array(drawdown)
    # Skew, kurt (60d)
    skew = returns_df.iloc[-60:].skew().values
    kurt = returns_df.iloc[-60:].kurt().values

    # Macro sensitivities (rolling 60d correlation)
    if len(returns_df) >= 60 and len(macro_df) >= 60:
        macro_diff = macro_df.diff().dropna()
        common = returns_df.index.intersection(macro_diff.index)
        if len(common) >= 30:
            ret_aligned = returns_df.loc[common]
            macro_aligned = macro_diff.loc[common]
            macro_sens = []
            for etf in etf_names:
                sens = [ret_aligned[etf].corr(macro_aligned[mcol]) if not np.isnan(ret_aligned[etf].corr(macro_aligned[mcol])) else 0.0 for mcol in config.MACRO_COLUMNS]
                macro_sens.append(sens)
        else:
            macro_sens = np.zeros((n_etfs, len(config.MACRO_COLUMNS)))
    else:
        macro_sens = np.zeros((n_etfs, len(config.MACRO_COLUMNS)))

    # Graph centralities (from correlation graph of last 60 days)
    if len(returns_df) >= 60:
        corr = returns_df.iloc[-60:].corr().values
        degree = np.sum(corr > 0.5, axis=1) / (n_etfs - 1)
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigenvector_centrality = np.abs(eigvecs[:, -1])
    else:
        degree = np.zeros(n_etfs)
        eigenvector_centrality = np.zeros(n_etfs)

    # Combine all features
    feature_matrix = np.column_stack([
        ret_1d, ret_5d, ret_21d, vol_21d, mom_21d, drawdown, skew, kurt,
        degree, eigenvector_centrality, macro_sens
    ])
    base_names = ['ret1d','ret5d','ret21d','vol21d','mom21d','drawdown','skew','kurt','degree','eigen_cent']
    macro_names = [f'sens_{m}' for m in config.MACRO_COLUMNS]
    feature_names = base_names + macro_names
    return feature_matrix, etf_names, feature_names
