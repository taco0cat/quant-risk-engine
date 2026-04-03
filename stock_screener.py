import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

# Initialize the Rich Console
console = Console()

# --- 1. THE DATA ENGINE ---
def analyze_stock(ticker_symbol):
    """Pulls data from yfinance and calculates metrics."""
    stock = yf.Ticker(ticker_symbol)
    results = {"Ticker": ticker_symbol, "Metrics": {}, "Checks": {}, "Errors": [], "Raw": {}}

    try:
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        history_1y = stock.history(period="1y")
        sp500_1y = yf.Ticker("^GSPC").history(period="1y")

        has_3yr_data = len(financials.columns) >= 4
        
        # 1. Revenue CAGR
        if has_3yr_data and 'Total Revenue' in financials.index:
            rev_now = financials.loc['Total Revenue'].iloc[0]
            rev_3yr_ago = financials.loc['Total Revenue'].iloc[3]
            rev_cagr = ((rev_now / rev_3yr_ago) ** (1/3)) - 1
        else: rev_cagr = 0

        # 2 & 3. EPS Metrics
        if has_3yr_data and 'Diluted EPS' in financials.index:
            eps_now = financials.loc['Diluted EPS'].iloc[0]
            eps_1yr_ago = financials.loc['Diluted EPS'].iloc[1]
            eps_3yr_ago = financials.loc['Diluted EPS'].iloc[3]
            eps_cagr = ((eps_now / eps_3yr_ago) ** (1/3)) - 1 if eps_3yr_ago > 0 else 0
            eps_acceleration = (eps_now - eps_1yr_ago) / eps_1yr_ago if eps_1yr_ago > 0 else 0
        else: eps_cagr, eps_acceleration = 0, 0

        # 4. Gross Margin
        if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index:
            gross_margin = financials.loc['Gross Profit'].iloc[0] / financials.loc['Total Revenue'].iloc[0]
        else: gross_margin = 0

        # 5. ROE
        if 'Net Income' in financials.index and 'Stockholders Equity' in balance_sheet.index:
            roe = financials.loc['Net Income'].iloc[0] / balance_sheet.loc['Stockholders Equity'].iloc[0]
        else: roe = 0

        # 6. FCF
        fcf = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0

        # 7. Rel Strength
        if not history_1y.empty and not sp500_1y.empty:
            stock_return = (history_1y['Close'].iloc[-1] - history_1y['Close'].iloc[0]) / history_1y['Close'].iloc[0]
            sp500_return = (sp500_1y['Close'].iloc[-1] - sp500_1y['Close'].iloc[0]) / sp500_1y['Close'].iloc[0]
        else: stock_return, sp500_return = 0, 0

        # 8, 9, 10. Info Dict Metrics
        peg_ratio = info.get('pegRatio', 999) or 999
        market_cap = info.get('marketCap', 1)
        insider_ownership = info.get('heldPercentInsiders', 0) or 0
        
        if 'Total Debt' in balance_sheet.index and 'Stockholders Equity' in balance_sheet.index:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
            debt_to_equity = total_debt / equity if equity > 0 else 999
            shariah_debt_ratio = total_debt / market_cap
        else:
            debt_to_equity, shariah_debt_ratio = 0, 0

        # Save Metrics
        results["Metrics"] = {
            "Rev CAGR": rev_cagr, "EPS CAGR": eps_cagr, "EPS Accel": eps_acceleration,
            "Gross Margin": gross_margin, "ROE": roe, "FCF": fcf,
            "Stock Return": stock_return, "SP500 Return": sp500_return,
            "PEG Ratio": peg_ratio, "Debt/Equity": debt_to_equity,
            "Insider Own": insider_ownership, "Shariah Debt": shariah_debt_ratio
        }

        # Save Pass/Fail Bools
        results["Checks"] = {
            "Rev CAGR": rev_cagr > 0.15, "EPS CAGR": eps_cagr > 0.20,
            "EPS Accel": eps_acceleration > 0.15, "Gross Margin": gross_margin > 0.40,
            "ROE": roe > 0.15, "FCF": fcf > 0,
            "Rel Strength": stock_return > sp500_return, "PEG Ratio": peg_ratio < 2.5,
            "Debt/Equity": debt_to_equity < 1.5, "Insider Own": insider_ownership > 0.03,
            "Shariah Debt": shariah_debt_ratio < 0.30
        }
        
        results["Total Score"] = sum(1 for k, v in results["Checks"].items() if v == True and k != "Shariah Debt")
        results["Raw"]["Market Cap"] = market_cap

    except Exception as e:
        results["Errors"].append(str(e))

    return results

# --- 2. FORMATTING HELPERS ---
def format_large_number(value):
    if value >= 1_000_000_000_000: return f"${value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000: return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000: return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"

def format_status(passed):
    return "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"

# --- 3. THE CLI DASHBOARD ---
def main():
    console.clear()
    
    # Print a beautiful header panel
    console.print(Panel(
        "[bold cyan]Institutional Growth & Shariah Screener[/bold cyan]\n[dim]Powered by yfinance[/dim]", 
        expand=False, 
        border_style="blue"
    ))

    # 1. Get User Input
    try:
        ticker = console.input("\n[bold yellow]Enter Stock Ticker (e.g., NVDA):[/bold yellow] ").strip().upper()
        if not ticker:
            sys.exit()
    except KeyboardInterrupt:
        sys.exit()

    # 2. Loading Animation while engine runs
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Fetching 4 years of financial statements for {ticker}...", total=None)
        data = analyze_stock(ticker)

    # Handle errors gracefully
    if data["Errors"]:
        console.print(f"\n[bold red]Error pulling data for {ticker}. Check ticker symbol or internet connection.[/bold red]")
        console.print(f"[dim]{data['Errors']}[/dim]")
        return

    # 3. Build the output table
    m = data["Metrics"]
    c = data["Checks"]

    table = Table(title=f"Fundamental Analysis: {ticker}", show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Current Value", justify="right", style="white")
    table.add_column("Target", justify="center", style="dim")
    table.add_column("Status", justify="center")

    # Add Rows
    table.add_row("Revenue Growth (3yr)", f"{m['Rev CAGR']*100:.1f}%", "> 15%", format_status(c["Rev CAGR"]))
    table.add_row("EPS Growth (3yr)", f"{m['EPS CAGR']*100:.1f}%", "> 20%", format_status(c["EPS CAGR"]))
    table.add_row("Earnings Acceleration", f"{m['EPS Accel']*100:.1f}%", "> 15%", format_status(c["EPS Accel"]))
    table.add_row("Gross Margin", f"{m['Gross Margin']*100:.1f}%", "> 40%", format_status(c["Gross Margin"]))
    table.add_row("Return on Equity", f"{m['ROE']*100:.1f}%", "> 15%", format_status(c["ROE"]))
    table.add_row("Free Cash Flow", format_large_number(m['FCF']), "Positive", format_status(c["FCF"]))
    table.add_row("Rel. Strength (vs S&P)", f"Stock: {m['Stock Return']*100:.1f}%", "Outperform", format_status(c["Rel Strength"]))
    table.add_row("PEG Ratio", f"{m['PEG Ratio']:.2f}", "< 2.5", format_status(c["PEG Ratio"]))
    table.add_row("Debt-to-Equity", f"{m['Debt/Equity']:.2f}", "< 1.5", format_status(c["Debt/Equity"]))
    table.add_row("Insider Ownership", f"{m['Insider Own']*100:.1f}%", "> 3%", format_status(c["Insider Own"]))
    
    # Add a visual separator for the Shariah check
    table.add_section()
    table.add_row(
        "[bold white]Shariah Debt Limit[/bold white]", 
        f"{m['Shariah Debt']*100:.1f}%", 
        "< 30%", 
        format_status(c["Shariah Debt"])
    )

    # 4. Print the final dashboard
    console.print("\n")
    
    # Score Summary Banner
    score_color = "green" if data['Total Score'] >= 8 else "yellow" if data['Total Score'] >= 5 else "red"
    console.print(f"Market Cap: [bold]{format_large_number(data['Raw']['Market Cap'])}[/bold]  |  Growth Score: [bold {score_color}]{data['Total Score']} / 10[/bold {score_color}]")
    
    console.print(table)
    console.print("\n")

if __name__ == "__main__":
    main()