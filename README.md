# P2-ETF-INFO-BOTTLENECK

# Information Bottleneck (IB) Engine

Uses the Variational Information Bottleneck to compress ETF features into a latent representation that retains maximum information about next‑day returns. Balances prediction accuracy (MSE) and compression (KL divergence) with a β hyperparameter.

- **Rolling windows:** 63, 126, 252 days (best per ETF)
- **Model:** Encoder (→ latent Gaussian) + Decoder (→ return)
- **Output:** top 3 ETFs per universe by predicted return, with the chosen window
- **Dashboard:** shows top ETFs, full ranking table, and validation losses per window

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
