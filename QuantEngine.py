import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class QuantEngine:
    """
    A quantitative risk engine that calculates Value at Risk (VaR) and 
    Conditional VaR (CVaR) using Monte Carlo simulations.
    """

    def __init__(self, portfolio_value: float, tickers: list, days: int = 5, simulations: int = 10000, confidence_interval: float = 0.95, start_date: str = None, end_date: str = None):
        # User defined parameters
        self.portfolio_value = portfolio_value
        self.tickers = tickers
        self.days = days
        self.simulations = simulations
        self.confidence_interval = confidence_interval
        self.start_date = start_date
        self.end_date = end_date

        # Initialize internal state variables
        self.adj_close_df = pd.DataFrame()
        self.weights = np.array([])
        self.expected_returns = 0.0
        self.std_dev = 0.0
        self.scenarioReturn = np.array([])

    def fetch_data(self) -> None:
        """
        Fetches historical 'Adjusted Close' price data from the Yahoo Finance API.
        Automatically removes invalid tickers to prevent downstream crashes.
        """
        # OPTIMIZATION: Loop over a copy of the list (self.tickers[:]) so we can safely remove bad tickers
        for ticker in self.tickers[:]:
            # Fetch data based on provided dates, or default to max history if dates are None
            if self.start_date and self.end_date:
                data = yf.download(ticker, start=self.start_date, end=self.end_date)
            else:
                data = yf.download(ticker, period="max")
            
            if data.empty:
                print(f"⚠️ Failed to fetch data for {ticker}. Removing from portfolio.")
                self.tickers.remove(ticker)
            else:
                self.adj_close_df[ticker] = data["Close"]

    def calculate_stats(self) -> None:
        """
        Calculates the statistical foundation for the Monte Carlo simulation.
        Aligns multi-asset dataframes by date to prevent covariance calculation errors.
        """
        if self.adj_close_df.empty:
            raise ValueError("No valid market data was downloaded. Please check your tickers or dates.")

        # OPTIMIZATION: Calculate weights here, AFTER invalid tickers have been removed
        num_tickers = len(self.tickers)
        self.weights = np.array([1.0 / num_tickers] * num_tickers)

        # Align all assets to start from the exact same date (the youngest stock's start date)
        # This prevents the .dropna() function from deleting decades of older stock data
        dates = [self.adj_close_df[col].first_valid_index() for col in self.adj_close_df.columns]
        max_date = max(dates)
        self.adj_close_df = self.adj_close_df.loc[max_date:]

        # Calculate daily logarithmic returns and covariance
        log_returns = np.log(self.adj_close_df / self.adj_close_df.shift(1)).dropna()
        cov_matrix = log_returns.cov()

        # Calculate expected portfolio return and standard deviation (volatility)
        self.expected_returns = np.sum(log_returns.mean() * self.weights)
        variance = self.weights.T @ cov_matrix @ self.weights
        self.std_dev = np.sqrt(variance)
    
    def run_var_simulation(self, seed: int = None) -> tuple:
        """
        Executes a vectorized Monte Carlo simulation to project future portfolio returns 
        and calculates downside risk metrics (VaR and CVaR).
        """
        if seed is not None:
            np.random.seed(seed)
    
        # Generate standard normal random variables (Z-scores)
        z_scores = np.random.normal(0, 1, self.simulations)

        # Vectorized array calculation for extreme speed
        self.scenarioReturn = (self.portfolio_value * float(self.expected_returns) * self.days) + \
                              (self.portfolio_value * float(self.std_dev) * z_scores * np.sqrt(self.days))
        
        # Value at Risk (VaR): The threshold at the specified confidence level
        VaR = -np.percentile(self.scenarioReturn, 100 * (1 - self.confidence_interval))

        # Conditional VaR (CVaR): Average loss in scenarios worse than the VaR threshold
        worst_cases = self.scenarioReturn[self.scenarioReturn <= -VaR]
        CVaR = -np.mean(worst_cases)

        return VaR, CVaR

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- QUANTITATIVE RISK ENGINE ---")
    
    # 1. Gather User Inputs (with error handling)
    while True:
        try:
            user_ticker = input("Enter stock ticker(s) separated by comma (eg. MSFT, AAPL): ").upper()
            # Clean up the list and ignore trailing commas
            ticker_list = [t.strip() for t in user_ticker.split(",") if t.strip()]
            
            user_portfolio = float(input("Enter your portfolio value ($): "))
            user_days = int(input("Enter timeframe in days (e.g., 5): "))
            
            print("\n(Optional) Press Enter to skip and use max available data.")
            user_start = input("Enter start date (YYYY-MM-DD): ").strip()
            user_end = input("Enter end date (YYYY-MM-DD): ").strip()
            
            # Convert empty strings to None so the fetch_data method knows to use period="max"
            user_start = user_start if user_start else None
            user_end = user_end if user_end else None
            
            break # Break out of the loop if all inputs are successfully captured
        except (ValueError, NameError, TypeError):
            print("❌ Invalid input format. Please try again.\n")

    # 2. Initialize Engine and Process Data
    engine = QuantEngine(
        portfolio_value=user_portfolio, 
        tickers=ticker_list, 
        days=user_days, 
        start_date=user_start, 
        end_date=user_end
    )
    
    print("\nFetching market data and computing variance...")
    engine.fetch_data()
    engine.calculate_stats()
    
    print(f"Running {engine.simulations:,} Monte Carlo simulations...")
    VaR, CVaR = engine.run_var_simulation(seed=42)

    # 3. Output Executive Summary
    print("\n" + "=" * 55)
    print(f"💰 RISK ANALYSIS REPORT ({engine.days}-Day Horizon)")
    print("=" * 55)
    print(f"Portfolio Value:           ${user_portfolio:,.2f}")
    print(f"Value at Risk (VaR):       ${float(VaR):,.2f}")
    print(f"Expected Shortfall (CVaR): ${float(CVaR):,.2f}")
    print("-" * 55)
    print("INTERPRETATION:")
    print(f"-> You can be {engine.confidence_interval:.0%} confident that your portfolio will")
    print(f"   NOT lose more than ${float(VaR):,.2f} over the next {engine.days} days.")
    print(f"-> However, in a 'Black Swan' market crash, your average")
    print(f"   loss past that threshold would be roughly ${float(CVaR):,.2f}.")
    print("=" * 55)