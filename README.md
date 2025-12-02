Quant Portfolio Optimizer

A Python-based financial tool that implements Modern Portfolio Theory (MPT) and the Black-Litterman Model to construct optimized investment portfolios.

- Efficient Frontier Visualization: Monte Carlo simulation generating 2,000+ portfolios to find the optimal risk/reward balance.
- Black-Litterman "Lite": Allows users to inject subjective market views (e.g., "Tech will drop 5%") to adjust the mathematical optimization.
- Risk Metrics: Calculates Sharpe Ratio, Volatility, and parametric Value at Risk (VaR) at 95% confidence.
- Interactive Dashboard: Built with Streamlit for real-time ticker analysis and parameter tuning.
- Correlation Heatmap: visualizes asset dependency to ensure true diversification.

Tech Stack
- Python 3.9+
- Streamlit (UI/UX)
- YFinance (Real-time Market Data)
- Scipy (SLSQP Optimization)
- Pandas/Numpy (Statistical Analysis)
- Matplotlib/Seaborn (Data Visualization)
