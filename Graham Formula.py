import yfinance as yf

def calculate_iv(eps, expected_growth_rate, current_bond_yield, pe_no_growth=7, multiplier=1):
    """Calculates the Intrinsic Value using the Revised Graham Formula."""
    historical_aaa_yield = 4.4
    base_calc = pe_no_growth + (multiplier * expected_growth_rate)
    return (eps * base_calc * historical_aaa_yield) / current_bond_yield

def evaluate_stock(ticker_symbol, consensus_growth_estimate):
    print(f"\nFetching live data for {ticker_symbol}...\n")
    
    # 1. Fetch stock and bond data
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    eps = info.get('trailingEps')
    current_price = info.get('currentPrice', info.get('previousClose'))
    
    if eps is None or current_price is None:
        print(f"Error: Could not retrieve valid EPS or Price data for '{ticker_symbol}'. Please check the ticker symbol.")
        return

    tnx = yf.Ticker("^TNX")
    current_bond_yield = tnx.history(period="1d")['Close'].iloc[-1]
    
    # 2. Determine the 3 Growth Rates (g)
    
    # Method A: Wall Street Consensus (Passed as parameter)
    g_consensus = consensus_growth_estimate
    
    # Method B: Fundamental Sustainable Growth (ROE * Retention Rate)
    roe = info.get('returnOnEquity', 0)
    payout_ratio = info.get('payoutRatio', 0)
    retention_rate = 1 - payout_ratio
    g_fundamental = (roe * retention_rate) * 100
    
    # Method C: Historical CAGR (Using Annual Income Statements)
    try:
        # yfinance typically returns the last 4 years of annual data
        eps_history = stock.income_stmt.loc['Diluted EPS'].dropna()
        years = len(eps_history) - 1
        if years > 0:
            latest_annual_eps = eps_history.iloc[0]
            oldest_annual_eps = eps_history.iloc[-1]
            # Handle negative EPS in history to avoid complex numbers
            if oldest_annual_eps > 0 and latest_annual_eps > 0:
                g_historical = (((latest_annual_eps / oldest_annual_eps) ** (1 / years)) - 1) * 100
            else:
                g_historical = 0
        else:
            g_historical = 0
    except Exception:
        print("Error fetching historical EPS. Defaulting historical growth to 0.")
        g_historical = 0
        years = 0

    # 3. Calculate Intrinsic Values
    iv_consensus = calculate_iv(eps, g_consensus, current_bond_yield)
    iv_fundamental = calculate_iv(eps, g_fundamental, current_bond_yield)
    iv_historical = calculate_iv(eps, g_historical, current_bond_yield)

    # 4. Output to Console
    print("="*45)
    print(f"--- {ticker_symbol} INTRINSIC VALUE (3 METHODS) ---")
    print("="*45)
    print(f"Current Price:      ${current_price:.2f}")
    print(f"TTM EPS:            ${eps}")
    print(f"Current Bond Yield: {current_bond_yield:.2f}%")
    print("="*45)
    
    # Formatting Method A
    print("METHOD 1: WALL STREET CONSENSUS")
    print(f"Expected Growth:    {g_consensus:.2f}%")
    print(f"Intrinsic Value:    ${iv_consensus:.2f}")
    print(f"% of IV:            {(current_price / iv_consensus) * 100:.2f}%")
    print("-" * 45)
    
    # Formatting Method B
    print("METHOD 2: FUNDAMENTAL SUSTAINABLE")
    print(f"Expected Growth:    {g_fundamental:.2f}%  (ROE: {roe*100:.1f}%)")
    print(f"Intrinsic Value:    ${iv_fundamental:.2f}")
    print(f"% of IV:            {(current_price / iv_fundamental) * 100:.2f}%")
    print("-" * 45)
    
    # Formatting Method C
    print("METHOD 3: HISTORICAL CAGR")
    print(f"Expected Growth:    {g_historical:.2f}%  ({years}-Year Span)")
    print(f"Intrinsic Value:    ${iv_historical:.2f}")
    if iv_historical > 0:
        print(f"% of IV:            {(current_price / iv_historical) * 100:.2f}%")
    else:
        print("% of IV:            N/A (Negative or Zero IV)")
    print("="*45)


def main():
    print("Starting Intrinsic Value Calculator...")
    print("Type 'exit' at any prompt to close the program.\n")
    
    while True:
        # 1. Get Ticker Symbol
        user_ticker = input("Enter ticker symbol (e.g., MSFT, AAPL): ").strip().upper()
        
        if user_ticker.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
        
        if not user_ticker:
            print("Ticker cannot be blank. Please try again.")
            continue
            
        # 2. Get Consensus Growth Rate
        growth_input = input(f"Enter consensus growth estimate for {user_ticker} (or press Enter for 10%): ").strip()
        
        if growth_input.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
            
        if growth_input == "":
            g_consensus = 10.0
        else:
            try:
                g_consensus = float(growth_input)
            except ValueError:
                print("Invalid number entered. Defaulting to 10.0%.")
                g_consensus = 10.0
                
        # 3. Run the evaluation
        evaluate_stock(user_ticker, g_consensus)
        print("\n" + "*"*50 + "\n")

if __name__ == "__main__":
    main()