# tools/get_company_financials.py

import yfinance as yf

def get_company_financials(symbol):
    """
    Fetches key financial metrics using yfinance.
    Input: stock symbol (e.g., "AAPL")
    Output: dict with revenue, net income, EPS, PE ratio, etc.
    """
    try:
        stock = yf.Ticker(symbol)

        info = stock.info  # Main financial data

        # Optional: fallback if info is empty
        if not info:
            return f"No data found for {symbol}."

        financials = {
            "symbol": symbol,
            "revenue (TTM)": f"${info.get('totalRevenue', 'N/A'):,}" if info.get('totalRevenue') else "N/A",
            "net income (TTM)": f"${info.get('netIncomeToCommon', 'N/A'):,}" if info.get('netIncomeToCommon') else "N/A",
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Profit Margin": f"{round(info.get('profitMargins', 0) * 100, 2)}%" if info.get("profitMargins") else "N/A",
            "ROE": f"{round(info.get('returnOnEquity', 0) * 100, 2)}%" if info.get("returnOnEquity") else "N/A",
            "Debt to Equity": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else "N/A"
        }

        return financials

    except Exception as e:
        return {"error": f"Failed to fetch financials for {symbol}: {str(e)}"}

# Example usage
if __name__ == "__main__":
    symbol = "AAPL"
    data = get_company_financials(symbol)
    print(f"\n📊 Financials for {symbol}")
    for key, value in data.items():
        print(f"{key}: {value}")
