# tools.py
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any

def get_stock_price(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Fetch recent historical price data and current info for a stock ticker.
    Use period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        current = {
            "symbol": ticker.upper(),
            "name": info.get("longName", ticker),
            "currentPrice": info.get("currentPrice"),
            "previousClose": info.get("previousClose"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "marketCap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
        }
        
        end = datetime.now()
        start = end - timedelta(days=400)  # ~1.5 years buffer
        hist = stock.history(start=start, end=end, interval=interval)
        
        if not hist.empty:
            current["recentClose"] = round(hist["Close"].iloc[-1], 2)
            if len(hist) > 21:
                current["change_1mo_pct"] = round((hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) * 100, 2)
            if len(hist) > 63:
                current["change_3mo_pct"] = round((hist["Close"].iloc[-1] / hist["Close"].iloc[-63] - 1) * 100, 2)
            current["vol_3mo_ann_pct"] = round(hist["Close"].pct_change().std() * (252 ** 0.5) * 100, 2)
        
        return current
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def compute_simple_momentum(ticker: str, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Basic momentum: total and annualized return over lookback_days.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="max")
        
        if len(hist) < lookback_days:
            return {"error": "Insufficient history", "ticker": ticker}
        
        start_price = hist["Close"].iloc[-lookback_days]
        end_price   = hist["Close"].iloc[-1]
        total_ret   = (end_price / start_price - 1) * 100
        ann_ret     = ((end_price / start_price) ** (252 / lookback_days) - 1) * 100
        
        signal = "STRONG BUY" if total_ret > 25 else "MODERATE" if total_ret > 8 else "WEAK"
        
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "total_return_pct": round(total_ret, 2),
            "annualized_return_pct": round(ann_ret, 2),
            "momentum_signal": signal + " momentum"
        }
    except Exception as e:
        return {"error": str(e)}


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch key fundamental metrics from yfinance.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        fundamentals = {
            "ticker": ticker.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "debtToEquity": info.get("debtToEquity"),
            "dividendYield": info.get("dividendYield"),
            "beta": info.get("beta"),
        }
        # Clean NaN / None
        return {k: v if v is not None else "N/A" for k, v in fundamentals.items()}
    except Exception as e:
        return {"error": str(e), "ticker": ticker}