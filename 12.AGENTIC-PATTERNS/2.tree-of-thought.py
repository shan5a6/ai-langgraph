from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
import os

# Load environment variables for GROQ_API_KEY
load_dotenv()

# Initialize Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile")  # or any other Groq model


# Define Agent State using TypedDict
class StrategyState(TypedDict):
    business_type: str
    expansion_options: List[str]
    strategy_analysis: Dict[str, str]
    best_strategy: str


# 🟢 Step 1: Generate Expansion Strategies
def generate_expansion_options(state: StrategyState) -> StrategyState:
    prompt = f"""
    The company specializes in {state['business_type']}. Suggest three possible expansion strategies:

    1. Entering a new geographical market.
    2. Launching a new product line.
    3. Partnering with an existing brand.

    Provide a brief overview of each strategy.
    """

    response = llm.invoke([
        SystemMessage(content="You are a business strategist."),
        HumanMessage(content=prompt)
    ])

    # Extract first three lines as options
    lines = response.content.split("\n")
    state["expansion_options"] = [l for l in lines if l.strip()][:3]

    return state


# 🟢 Step 2: Analyze Each Strategy (Thinking paths)
def analyze_strategy(state: StrategyState) -> StrategyState:
    strategy_analysis: Dict[str, str] = {}

    for strategy in state["expansion_options"]:
        prompt = f"""
        Analyze the following business expansion strategy:

        {strategy}

        Evaluate it based on:
        - Cost implications
        - Risk factors
        - Potential return on investment (ROI)

        Provide a structured breakdown.
        """

        response = llm.invoke([
            SystemMessage(content="You are a business analyst."),
            HumanMessage(content=prompt)
        ])

        strategy_analysis[strategy] = response.content

    state["strategy_analysis"] = strategy_analysis
    return state


# 🟢 Step 3: Select Best Strategy
def select_best_strategy(state: StrategyState) -> StrategyState:
    prompt = f"""
    Given the following business expansion strategies and their analysis:

    {state['strategy_analysis']}

    Rank these strategies based on:
    - Highest ROI
    - Lowest risk
    - Overall feasibility

    Select the BEST strategy and explain why.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert business strategist."),
        HumanMessage(content=prompt)
    ])

    state["best_strategy"] = response.content
    return state


# 🔵 Build LangGraph
workflow = StateGraph(StrategyState)

workflow.add_node("generate_expansion_options", generate_expansion_options)
workflow.add_node("analyze_strategy", analyze_strategy)
workflow.add_node("select_best_strategy", select_best_strategy)

workflow.set_entry_point("generate_expansion_options")
workflow.add_edge("generate_expansion_options", "analyze_strategy")
workflow.add_edge("analyze_strategy", "select_best_strategy")

graph = workflow.compile()


# 🟢 Run Example
input_data = {
    "business_type": "AI-based EdTech Startup"
}

result = graph.invoke(input_data)

print("🚀 AI-Generated Expansion Strategies:\n", result["expansion_options"])
print("\n🔍 Strategy Analysis:\n", result["strategy_analysis"])
print("\n🏆 Best Strategy Selected:\n", result["best_strategy"])
