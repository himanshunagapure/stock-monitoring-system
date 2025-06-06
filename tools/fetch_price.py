import yfinance as yf

def fetch_stock_price(ticker: str) -> float | None:
    try:
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        return price
    except Exception as e:
        print(f"Error: {e}")
        return None

#print(fetch_stock_price("INFY.NS"))