import yfinance as yf
import numpy as np

def getBasicInfo(ticker):
    ticker = yf.Ticker(user_ticker)
    data = ticker.info

    name = data.get("shortName", "N/A")
    sector = data.get("sector", "N/A")
    currprice = data.get("currentPrice", "N/A")

    return name, sector, currprice

def gaapEps(ticker):
    ticker = yf.Ticker(ticker)
    q_financials = ticker.quarterly_income_stmt

    try:
        gaap_ttm_eps = q_financials.loc["Diluted EPS"].iloc[:4].sum()
        

        return np.round(gaap_ttm_eps, 2)
    
    except KeyError:
        print("Could not find 'Diluted EPS' in the financial statements.")        

while True:
    user_ticker = input("Enter a stock ticker (or 'exit' to quit): ").upper()

    if user_ticker == "EXIT":
        print("Exiting the program. Goodbye!")
        break

    if user_ticker == "":
        continue

    try:
        name, sector, currprice = getBasicInfo(user_ticker)

        print(f"\n----- {name} ({user_ticker}) -----")
        print(f"Sector: {sector}")
        print(f"Price: {currprice}")

        print("\nFetching financial data...")
        gaap_ttm_eps = gaapEps(user_ticker)
        print(f"GAAP TTM EPS: {gaap_ttm_eps}")

    except Exception as e:
        print(f"An error occurred: {e}. Please try a different ticker.")




# eps = data.get("trailingEps", "N/A")
# pe = currprice / eps

# wh52 = data.get("fiftyTwoWeekHigh", "N/A")
# wl52 = data.get("fiftyTwoWeekLow", "N/A")

# print(f"Corporation Name: {name}")
# print(f"Stock Ticker: {user_ticker}")
# print(f"Current Price: {currprice}")
# print(f"EPS (TTM): {eps}")
# print(f"Sector: {sector}")
# print(f"52 Week High: {wh52}")
# print(f"52 Week Low: {wl52}")
# print(f"PE Ratio: {pe}")