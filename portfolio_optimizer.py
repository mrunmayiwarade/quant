import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")
st.title("Quant Portfolio Optimizer (Black-Litterman Lite)")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("1. Asset Selection")
default_tickers = "NVDA, AMD, INTC, MSFT, GOOG"
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.header("2. Market Views (Black-Litterman)")
st.sidebar.markdown("Adjust expected returns based on your analysis.")
view_return = st.sidebar.number_input("Shock to 1st Asset (e.g., -0.10 for -10%)", value=0.0, step=0.01)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
        return data
    except Exception as e:
        return pd.DataFrame()

def calculate_var(weights, mean_returns, cov_matrix, confidence_level=0.05):
    # Parametric VaR (Value at Risk)
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Z-score for 95% confidence is -1.645
    var_95 = portfolio_return - 1.645 * portfolio_std
    return var_95

# --- MAIN EXECUTION ---
with st.spinner("Fetching data..."):
    data = get_data(tickers, start_date, end_date)

if data.empty:
    st.error("Could not fetch data. Check ticker symbols.")
    st.stop()

# 1. Base Calculations (Historical)
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 2. Apply "Views" (The Quant Upgrade)
# We adjust the expected return of the FIRST asset in the list based on user input
# Real Black-Litterman is complex matrix math; this is a "Lite" version for demonstration
adjusted_mean_returns = mean_returns.copy()
if view_return != 0:
    target_asset = tickers[0]
    # We blend the historical mean with the user's view
    # In a real model, you'd use a confidence matrix (P and Omega)
    st.info(f"Applying view: Adjusting {target_asset} expected return by {view_return:.1%}")
    adjusted_mean_returns[target_asset] += (view_return / 252) # De-annualize for daily calculation

# 3. Optimization Engine
def portfolio_performance(weights, mean_rets, cov_mat):
    # Annualized
    returns = np.sum(mean_rets * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe(weights, mean_rets, cov_mat, risk_free_rate=0.0):
    p_ret, p_var = portfolio_performance(weights, mean_rets, cov_mat)
    return -(p_ret - risk_free_rate) / p_var if p_var != 0 else 0

num_assets = len(tickers)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0.0, 1.0) for asset in range(num_assets))

opt_result = optimize.minimize(
    negative_sharpe, 
    num_assets * [1. / num_assets], 
    args=(adjusted_mean_returns, cov_matrix), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

optimal_weights = opt_result.x
opt_ret, opt_vol = portfolio_performance(optimal_weights, adjusted_mean_returns, cov_matrix)
opt_sharpe = opt_ret / opt_vol
daily_var_95 = calculate_var(optimal_weights, adjusted_mean_returns, cov_matrix)

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Optimized Portfolio")
    
    # Weights Table
    df_weights = pd.DataFrame({"Asset": tickers, "Weight": optimal_weights})
    df_weights["Weight"] = df_weights["Weight"].map("{:.2%}".format)
    st.table(df_weights)
    
    st.subheader("Risk Metrics")
    st.metric("Expected Annual Return", f"{opt_ret:.2%}")
    st.metric("Annual Volatility", f"{opt_vol:.2%}")
    st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
    
    st.markdown("---")
    st.error(f"⚠️ Value at Risk (95%): {daily_var_95:.2%}")
    st.caption("This means there is a 5% chance you lose more than this percentage in a SINGLE DAY.")

with col2:
    st.subheader("Efficient Frontier (with Views)")
    
    # Monte Carlo
    results = np.zeros((3, 2000))
    for i in range(2000):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        p_ret, p_vol = portfolio_performance(w, adjusted_mean_returns, cov_matrix)
        results[0,i] = p_vol
        results[1,i] = p_ret
        results[2,i] = p_ret / p_vol
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.scatter(opt_vol, opt_ret, marker='*', color='r', s=300, label='Optimal (with Views)')
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Return')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    st.markdown("Returns approaching **+1.0** move together. Returns approaching **-1.0** move oppositely.")

    # Create a correlation matrix
    corr_matrix = returns.corr()

    # Set up the matplotlib figure
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Draw the heatmap with the mask and correct aspect ratio
    # annot=True adds the numbers inside the squares
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax2)
    
    st.pyplot(fig2)
