import yfinance as yf

def get_company_ceo(symbol: str) -> str:
    symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(symbol)
        ceo = stock.info.get("companyOfficers", [{}])[0].get("name", "Unknown")
        return f"The CEO of {symbol} is {ceo}."
    except Exception as e:
        return f"ERROR: Failed to get CEO for {symbol}: {e}"
    
#print(get_company_ceo("AAPL"))