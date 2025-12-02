import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide")
st.title("Quant Portfolio Optimizer (Black-Litterman Model)")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("1. Asset Selection")
default_tickers = "NVDA, GLD, TLT, XLE, MSFT"
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)

# Parse and clean ticker input
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.header("2. Market Views")
st.sidebar.markdown("Adjust return expectations based on market views.")
# Input for Black-Litterman style view adjustment
view_return = st.sidebar.number_input("Shock to 1st Asset (e.g., -0.10 for -10%)", value=0.0, step=0.01)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# --- DATA ACQUISITION ---
@st.cache_data
def get_data(tickers, start, end):
    """
    Fetches historical adjusted close prices from Yahoo Finance.
    Results are cached to improve performance.
    """
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
        return data
    except Exception as e:
        return pd.DataFrame()

# --- RISK METRICS ---
def calculate_var(weights, mean_returns, cov_matrix, confidence_level=0.05):
    """
    Calculates Parametric Value at Risk (VaR) at a 95% confidence interval.
    Formula: VaR = Portfolio Mean - Z * Portfolio Std Dev
    """
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Z-score for 95% confidence is roughly 1.645
    z_score = 1.645
    var_95 = portfolio_return - z_score * portfolio_std
    return var_95

# --- MAIN EXECUTION BLOCK ---
with st.spinner("Fetching market data..."):
    data = get_data(tickers, start_date, end_date)

if data.empty:
    st.error("No data found. Please verify ticker symbols.")
    st.stop()

# Calculate daily returns and covariance matrix
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# --- EXPECTED RETURN ADJUSTMENT (Black-Litterman Lite) ---
# Adjusts the historical mean return of the target asset based on user input.
adjusted_mean_returns = mean_returns.copy()
if view_return != 0:
    target_asset = tickers[0]
    st.info(f"Applying view: Adjusting expected return for {target_asset} by {view_return:.1%}")
    # Adjustment is de-annualized for daily calculation compatibility
    adjusted_mean_returns[target_asset] += (view_return / 252)

# --- OPTIMIZATION FUNCTIONS ---
def portfolio_performance(weights, mean_rets, cov_mat):
    """
    Computes annualized portfolio return and volatility.
    Assumption: 252 trading days per year.
    """
    returns = np.sum(mean_rets * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe(weights, mean_rets, cov_mat, risk_free_rate=0.0):
    """
    Objective function for minimization. 
    Returns negative Sharpe Ratio since scipy.optimize minimizes functions.
    """
    p_ret, p_var = portfolio_performance(weights, mean_rets, cov_mat)
    return -(p_ret - risk_free_rate) / p_var if p_var != 0 else 0

# --- OPTIMIZATION ROUTINE ---
num_assets = len(tickers)
# Constraint: Weights must sum to 1.0 (fully invested)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Bounds: Weights must be between 0 and 1 (Long only, no leverage)
bounds = tuple((0.0, 1.0) for asset in range(num_assets))

# Sequential Least Squares Programming (SLSQP) for constrained optimization
opt_result = optimize.minimize(
    negative_sharpe, 
    num_assets * [1. / num_assets], # Initial guess: Equal weights
    args=(adjusted_mean_returns, cov_matrix), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

# Extract optimal results
optimal_weights = opt_result.x
opt_ret, opt_vol = portfolio_performance(optimal_weights, adjusted_mean_returns, cov_matrix)
opt_sharpe = opt_ret / opt_vol
daily_var_95 = calculate_var(optimal_weights, adjusted_mean_returns, cov_matrix)

# --- VISUALIZATION ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Optimal Allocation")
    
    # Display weights
    df_weights = pd.DataFrame({"Asset": tickers, "Weight": optimal_weights})
    df_weights["Weight"] = df_weights["Weight"].map("{:.2%}".format)
    st.table(df_weights)
    
    # Display metrics
    st.subheader("Portfolio Metrics")
    st.metric("Expected Annual Return", f"{opt_ret:.2%}")
    st.metric("Annual Volatility", f"{opt_vol:.2%}")
    st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
    
    st.markdown("---")
    st.error(f"Value at Risk (95%): {daily_var_95:.2%}")
    st.caption("Maximum expected daily loss at 95% confidence level.")

with col2:
    st.subheader("Efficient Frontier")
    
    # Monte Carlo Simulation
    num_simulations = 2000
    results = np.zeros((3, num_simulations))
    for i in range(num_simulations):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        p_ret, p_vol = portfolio_performance(w, adjusted_mean_returns, cov_matrix)
        results[0,i] = p_vol
        results[1,i] = p_ret
        results[2,i] = p_ret / p_vol
        
    # Plotting Efficient Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.scatter(opt_vol, opt_ret, marker='*', color='r', s=300, label='Optimal Portfolio')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    st.pyplot(fig)

    # Correlation Matrix Heatmap
    st.subheader("Asset Correlation Matrix")
    corr_matrix = returns.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax2)
    st.pyplot(fig2)
