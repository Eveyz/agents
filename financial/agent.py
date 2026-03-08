# agents.py
from google.adk.agents import Agent, SequentialAgent
from .tools import (
    get_stock_price, compute_simple_momentum, get_fundamentals, compute_quality_factor,
    run_simple_momentum_backtest, generate_price_chart, compute_portfolio_metrics,
    get_recent_stock_news_enhanced, compute_news_sentiment_timeseries,
    map_industry_value_chain, assess_geopolitical_risks, find_historical_analogs
)
from .model import zai

# Safety
from typing import Any, Dict, Optional

def safety_check_before_tool(tool: Any, args: Dict[str, Any], tool_context: Any = None) -> Optional[Dict[str, Any]]:
    tool_name = getattr(tool, 'name', getattr(tool, '__name__', 'unknown')).lower()
    blocked = ["buy", "sell", "order", "trade", "execute", "position", "broker"]
    if any(kw in tool_name for kw in blocked):
        raise ValueError("Trading actions blocked — research mode only.")
    return None


# Agents

data_agent = Agent(
    name="data_agent",
    model=zai,
    description="Fetches prices, momentum, fundamentals, quality.",
    instruction="Precise data provider. Return structured data only.",
    tools=[get_stock_price, compute_simple_momentum, get_fundamentals, compute_quality_factor],
    # before_tool_callback=safety_check_before_tool,
)

backtest_agent = Agent(
    name="backtest_agent",
    model=zai,
    description="Runs strategy backtests.",
    instruction="Expert backtester. Compare to benchmark.",
    tools=[run_simple_momentum_backtest],
    # before_tool_callback=safety_check_before_tool,
)

chart_agent = Agent(
    name="chart_agent",
    model=zai,
    description="Generates price charts.",
    instruction="Use generate_price_chart. Return base64 or describe.",
    tools=[generate_price_chart],
    # before_tool_callback=safety_check_before_tool,
)

portfolio_agent = Agent(
    name="portfolio_agent",
    model=zai,
    description="Portfolio risk/return metrics.",
    instruction="Analyze portfolios. Research only.",
    tools=[compute_portfolio_metrics],
    # before_tool_callback=safety_check_before_tool,
)

news_sentiment_agent = Agent(
    name="news_sentiment_agent",
    model=zai,
    description="News collection & time-series sentiment analysis.",
    instruction="""News & sentiment specialist.
1. Use get_recent_stock_news_enhanced (45 days default)
2. Compute compute_news_sentiment_timeseries on daily_data
3. Highlight deltas, trends, flags
4. Combine with other signals
Research only — not advice.""",
    tools=[get_recent_stock_news_enhanced, compute_news_sentiment_timeseries],
    # before_tool_callback=safety_check_before_tool,
)

strategic_intelligence_agent = Agent(
    name="strategic_intelligence_agent",
    model=zai,
    description="Holistic chain + geo + historical analysis.",
    instruction="""Senior macro-strategic analyst.
1. Map value chain
2. Assess geopolitical risks
3. Find historical analogs
4. Integrate news trends
5. Produce clear alpha thesis + conviction
Research only — not advice.""",
    tools=[map_industry_value_chain, assess_geopolitical_risks, find_historical_analogs],
    # before_tool_callback=safety_check_before_tool,
)

report_agent = Agent(
    name="report_agent",
    model=zai,
    description="Final synthesis & Markdown reports.",
    instruction="Synthesize into clear Markdown. Tables when useful. Disclaimer at end.",
)

# Define the pipeline (this assigns parents to the agents inside it)
# research_pipeline = SequentialAgent(
#     name="research_pipeline",
#     description="Full structured flow: data → strategy → backtest → report",
#     sub_agents=[
#         data_agent,
#         strategic_intelligence_agent,
#         news_sentiment_agent,       # or place it earlier if you prefer
#         backtest_agent,
#         chart_agent,
#         report_agent
#     ]
# )

# Root only knows about top-level agents / pipelines
root_agent = Agent(
    name="quant_research_coordinator",
    model=zai,
    description="Lead coordinator — delegates to specialists or full pipelines",
    instruction="""Lead quant research coordinator.
Delegation rules:
- For quick facts/prices/momentum/fundamentals → data_agent
- For news trend/change detection → news_sentiment_agent
- For deep chain/geopolitics/history → strategic_intelligence_agent
- For end-to-end structured analysis → research_pipeline
- For portfolio/multi-stock → portfolio_agent
- For charts → chart_agent
- For backtesting standalone → backtest_agent
- Always synthesize final answer yourself

Always prefix conclusions: "This is research/analysis only — not investment advice."
""",
    sub_agents=[
        data_agent,
        news_sentiment_agent,
        strategic_intelligence_agent,
        portfolio_agent,
        chart_agent,
        backtest_agent,
        # research_pipeline,          # ← the pipeline is now a child of root
    ],
    # before_tool_callback=safety_check_before_tool,
)