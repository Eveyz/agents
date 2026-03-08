from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from .tools import get_stock_price, compute_simple_momentum, get_fundamentals

zai = LiteLlm(
    model="zai/glm-5",
)

# ── Data Agent ───────────────────────────────────────────────────────────────
data_agent = Agent(
    name="data_agent",
    model=zai,
    description=(
        "Specialized agent for fetching and computing raw stock data, "
        "prices, momentum scores and fundamental metrics. "
        "Use this agent when factual market data or calculations are needed."
    ),
    instruction="""You are a precise, factual data provider.
Only return clean, structured data from tools — do not speculate or add opinions.
Always include the ticker and source (yfinance).
If input is unclear, ask for clarification (ticker, period, etc.).""",
    tools=[get_stock_price, compute_simple_momentum, get_fundamentals],
)


# ── Root / Coordinator Agent ─────────────────────────────────────────────────
root_agent = Agent(
    name="quant_research_coordinator",
    model=zai,
    description=(
        "Senior quantitative research coordinator. "
        "Delegates data tasks to data_agent, synthesizes insights, "
        "coordinates multi-step research workflows."
    ),
    instruction="""You are a senior quant research assistant focused on alpha discovery and stock analysis.
Never give direct investment advice — always prefix with:
"This is research/analysis only — not investment advice."

Strategy:
1. Understand user goal (screen, analyze ticker, compare, find momentum/value, etc.)
2. If data/facts needed → delegate to data_agent
3. If calculation or comparison needed → call data_agent multiple times if necessary
4. Synthesize results clearly, use tables when helpful
5. Suggest next logical questions/steps""",
    
    sub_agents=[data_agent],
)