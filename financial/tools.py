# tools.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import io
import base64
import json

from .model import zai

# ── Core Data Tools ──────────────────────────────────────────────────────────

def get_stock_price(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
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
        start = end - timedelta(days=400)
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
        return {k: v if v is not None else "N/A" for k, v in fundamentals.items()}
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def compute_quality_factor(ticker: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker.upper()).info
        roe = info.get('returnOnEquity', 0) or 0
        debt_eq = info.get('debtToEquity', 999) or 999
        growth = info.get('earningsGrowth', 0) or 0
        score = (roe * 100 if roe else 0) - (debt_eq if debt_eq < 999 else 100) + (growth * 100 if growth else 0)
        quality = "HIGH" if score > 50 else "MEDIUM" if score > 0 else "LOW"
        return {
            "ticker": ticker.upper(),
            "roe": roe,
            "debt_to_equity": debt_eq,
            "earnings_growth": growth,
            "quality_score": round(score, 1),
            "quality_level": quality
        }
    except:
        return {"error": "Failed"}


# ── Backtesting ──────────────────────────────────────────────────────────────

def run_simple_momentum_backtest(
    ticker: str,
    momentum_lookback_days: int = 252,
    holding_period_days: int = 21,
    start_date: str = "2020-01-01",
    initial_capital: float = 100_000
) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(start=start_date, end=None)
        if len(df) < momentum_lookback_days + 100:
            return {"error": "Not enough history for backtest", "ticker": ticker}
        df["momentum"] = df["Close"].pct_change(momentum_lookback_days)
        df["signal"] = (df["momentum"] > 0).astype(int)
        df["forward_ret"] = df["Close"].pct_change(holding_period_days).shift(-holding_period_days)
        df["strategy_ret"] = df["signal"] * df["forward_ret"]
        df["equity"] = initial_capital * (1 + df["strategy_ret"]).cumprod().fillna(1)
        total_return = (df["equity"].iloc[-1] / initial_capital - 1) * 100
        ann_return = (df["equity"].iloc[-1] / initial_capital) ** (252 / len(df)) - 1
        ann_vol = df["strategy_ret"].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        max_dd = ((df["equity"] / df["equity"].cummax()) - 1).min() * 100
        trades = df["signal"].diff().abs().sum() / 2
        win_rate = (df["strategy_ret"] > 0).mean() * 100 if len(df) > 0 else 0
        return {
            "ticker": ticker.upper(),
            "strategy": f"Momentum({momentum_lookback_days}d hold {holding_period_days}d)",
            "period": f"{df.index[0].date()} → {df.index[-1].date()}",
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "num_trades": int(trades),
            "final_equity": round(df["equity"].iloc[-1], 2),
            "benchmark_buy_hold_return_pct": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# ── Charting ─────────────────────────────────────────────────────────────────
def generate_price_chart(ticker: str, period: str = "1y") -> Dict[str, str]:
    """
    Safer version: handles failures gracefully and always cleans up.
    """
    fig = None
    buf = None
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)
        
        if df.empty:
            return {"error": "No price data available for this period"}

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label='Close Price')
        ax.set_title(f"{ticker.upper()} Price Chart ({period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {
            "ticker": ticker.upper(),
            "chart_base64": img_base64,
            "description": "Price chart (PNG, base64 encoded)"
        }

    except Exception as e:
        return {"error": f"Failed to generate chart: {str(e)}"}

    finally:
        # Safe cleanup — only close if fig exists
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass  # ignore cleanup errors
        
        if buf is not None:
            try:
                buf.close()
            except Exception:
                pass


def compute_portfolio_metrics(tickers: list, weights: list = None) -> Dict[str, Any]:
    try:
        if not isinstance(tickers, list) or len(tickers) < 1:
            return {"error": "Provide list of tickers"}
        data = {}
        for t in tickers:
            stock = yf.Ticker(t.upper())
            hist = stock.history(period="2y")['Close']
            data[t.upper()] = hist
        df = pd.DataFrame(data).dropna()
        if df.empty:
            return {"error": "No overlapping data"}
        returns = df.pct_change().dropna()
        if weights is None:
            weights = np.array([1.0 / len(tickers)] * len(tickers))
        else:
            weights = np.array(weights)
            if abs(weights.sum() - 1.0) > 1e-6:
                return {"error": "Weights must sum to 1"}
        port_ret = np.dot(returns.mean() * 252, weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = port_ret / port_vol if port_vol != 0 else 0
        return {
            "tickers": [t.upper() for t in tickers],
            "weights": weights.tolist(),
            "annualized_return_pct": round(port_ret * 100, 2),
            "annualized_vol_pct": round(port_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3)
        }
    except Exception as e:
        return {"error": str(e)}


# ── News ─────────────────────────────────────────────────────────────────────

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

def get_recent_stock_news_enhanced(
    ticker: str,
    days_back: int = 45,
    min_items_per_day: int = 1,
    limit_per_day: int = 15
) -> Dict[str, Any]:
    if not FINNHUB_KEY:
        return {"error": "FINNHUB_API_KEY not set"}
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        url = (
            f"https://finnhub.io/api/v1/company-news?"
            f"symbol={ticker.upper()}&"
            f"from={start_date.strftime('%Y-%m-%d')}&"
            f"to={end_date.strftime('%Y-%m-%d')}&"
            f"token={FINNHUB_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()
        if not news:
            return {"error": "No news found", "ticker": ticker}
        df = pd.DataFrame(news)
        if 'datetime' not in df.columns:
            return {"error": "Invalid news format"}
        df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.date
        df = df.sort_values('date', ascending=False)
        grouped = df.groupby('date').agg(
            count=('headline', 'size'),
            headlines=('headline', list),
            summaries=('summary', list),
            sources=('source', list),
            urls=('url', list)
        ).reset_index()
        grouped = grouped[grouped['count'] >= min_items_per_day]
        return {
            "ticker": ticker.upper(),
            "period_days": days_back,
            "days_with_news": len(grouped),
            "daily_data": grouped.to_dict(orient='records'),
            "raw_count": len(df)
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def compute_news_sentiment_timeseries(
    daily_news_data: List[Dict],
    rolling_window: int = 7,
    delta_window_recent: int = 7,
    delta_window_prior: int = 14
) -> Dict[str, Any]:
    """
    Compute time-series sentiment using pure LLM prompting (no VADER, no transformers).
    
    Each day's news is summarized and scored independently by the LLM.
    Returns structured time-series with delta, rolling average, trend flags.
    """
    if not daily_news_data:
        return {"error": "No daily news data provided"}

    results = []

    for day in daily_news_data:
        date_str = str(day['date'])
        headlines = day.get('headlines', [])
        summaries = day.get('summaries', [])

        texts = []
        for h, s in zip(headlines, summaries):
            combined = (h or "").strip() + " " + (s or "").strip()
            if combined:
                texts.append(combined[:450])  # truncate to avoid context overflow

        if not texts:
            results.append({
                "date": date_str,
                "item_count": 0,
                "score": None,
                "label": "no_data",
                "explanation": "No usable news text"
            })
            continue

        # Build prompt for this day
        news_block = "\n".join([f"- {t}" for t in texts])

        prompt = f"""You are a financial sentiment analyst with deep domain knowledge.

Task:
Read the following recent news items about the company.
Determine the overall sentiment on a scale from -1.0 (very negative / bearish) to +1.0 (very positive / bullish).
Be precise, consider nuance, sarcasm, forward-looking statements, numbers, analyst tone, etc.

Return ONLY valid JSON with these exact keys:
{{
  "score": float,               // -1.0 to +1.0
  "label": "positive" | "negative" | "neutral",
  "confidence": float,          // 0.0 to 1.0
  "explanation": "one short sentence explaining the main drivers"
}}

News items from {date_str}:
{news_block}
"""

        try:
            # Adjust this line to match your actual LiteLlm calling pattern
            response = zai.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}  # if your model supports it
            )

            # Depending on your wrapper, response might be .content, .text, or raw dict
            # Adapt the next lines accordingly
            if hasattr(response, 'content'):
                text = response.content.strip()
            elif isinstance(response, str):
                text = response.strip()
            else:
                text = str(response)

            # Clean common JSON markdown wrappers
            if text.startswith("```json"):
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()

            parsed = json.loads(text)

            score = float(parsed.get("score", 0.0))
            label = parsed.get("label", "neutral").lower()
            explanation = parsed.get("explanation", "")

            results.append({
                "date": date_str,
                "item_count": len(texts),
                "score": round(score, 3),
                "label": label,
                "confidence": round(parsed.get("confidence", 0.7), 2),
                "explanation": explanation
            })

        except Exception as e:
            results.append({
                "date": date_str,
                "item_count": len(texts),
                "score": None,
                "label": "error",
                "explanation": f"LLM parsing failed: {str(e)}"
            })

    if not results:
        return {"error": "No days produced valid sentiment"}

    df = pd.DataFrame(results).sort_values("date")

    # Rolling average on numeric score
    df["rolling_avg"] = df["score"].rolling(window=rolling_window, min_periods=3).mean()

    # Delta between recent and prior period
    delta = None
    if len(df) >= delta_window_recent + delta_window_prior:
        recent = df["score"].tail(delta_window_recent).mean()
        prior_start = -delta_window_recent - delta_window_prior
        prior_end = -delta_window_recent
        prior = df["score"].iloc[prior_start:prior_end].mean()
        if pd.notna(prior):
            delta = recent - prior

    # Very simple slope estimate
    slope = None
    if len(df) >= 5:
        df["day_idx"] = range(len(df))
        valid = df[["day_idx", "score"]].dropna()
        if len(valid) >= 2:
            slope = valid["score"].corr(valid["day_idx"]) * (len(valid) - 1)

    flags = []
    if delta is not None and abs(delta) > 0.30:
        flags.append(f"Significant sentiment shift (Δ {round(delta, 3)})")
    if slope is not None and abs(slope) > 0.025:
        flags.append(f"Trend slope ≈ {round(slope, 3)}")

    return {
        "days_analyzed": len(df),
        "timeseries": df.to_dict(orient="records"),
        "current_rolling_avg": round(df["rolling_avg"].iloc[-1], 3) if "rolling_avg" in df.columns else None,
        "delta_recent_vs_prior": round(delta, 3) if delta is not None else None,
        "trend_slope": round(slope, 3) if slope is not None else None,
        "notable_flags": flags,
        "overall_trend": (
            "Strongly improving" if (delta or 0) > 0.30 else
            "Improving" if (delta or 0) > 0.12 else
            "Deteriorating" if (delta or 0) < -0.30 else
            "Stable / weak trend"
        ),
        "method": "pure LLM prompting (zai/glm-5)",
        "cost_note": "One LLM call per day with news"
    }

# ── Strategic Intelligence Tools ─────────────────────────────────────────────

def map_industry_value_chain(ticker: str) -> Dict[str, Any]:
    # Placeholder — in real use this would be filled by LLM / search
    return {
        "ticker": ticker.upper(),
        "upstream": ["Raw materials / suppliers"],
        "core_position": "Mid / downstream role",
        "downstream": ["Customers / end markets"],
        "competitors": ["Direct rivals"],
        "choke_points": ["Geopolitical exposure areas"]
    }


def assess_geopolitical_risks(ticker: str) -> Dict[str, Any]:
    # Placeholder — real version uses search / LLM
    return {
        "risk_level": "Medium",
        "active_tensions": ["US-China tech", "Taiwan risk"],
        "affected_parts": ["Supply chain"],
        "outlook": "Diversification reduces risk"
    }


def find_historical_analogs(ticker: str, current_context: str = None) -> Dict[str, Any]:
    # Placeholder
    return {
        "strongest_analog": "2018–2019 trade war",
        "outcome": "Diversified names outperformed",
        "implication": "Positive for non-China heavy exposure"
    }