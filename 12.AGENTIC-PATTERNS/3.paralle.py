from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables for GROQ_API_KEY
load_dotenv()

# ---------------------- State ----------------------
class MarketResearchState(TypedDict):
    query: str
    trends: str
    competitors: str
    sentiment: str
    summary: str


# ---------------------- Groq Model ----------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")  # or mixtral, gemma2, etc.


# ---------------------- Nodes ----------------------
def fetch_trends(state: MarketResearchState):
    prompt = f"What are the latest market trends for {state['query']}?"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"trends": response.content}


def analyze_competitors(state: MarketResearchState):
    prompt = f"List top competitors in the {state['query']} market."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"competitors": response.content}


def extract_sentiment(state: MarketResearchState):
    prompt = f"What is the customer sentiment toward products in the {state['query']} category?"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"sentiment": response.content}


def summarize(state: MarketResearchState):
    summary_prompt = f"""
    Provide a strategic market entry summary for the following product: {state['query']}

    Market Trends:
    {state.get('trends')}

    Competitor Landscape:
    {state.get('competitors')}

    Customer Sentiment:
    {state.get('sentiment')}

    Give an actionable, executive-level strategic conclusion.
    """
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"summary": response.content}


# ---------------------- Build the Graph ----------------------
graph_builder = StateGraph(MarketResearchState)

graph_builder.add_node("fetch_trends", fetch_trends)
graph_builder.add_node("analyze_competitors", analyze_competitors)
graph_builder.add_node("extract_sentiment", extract_sentiment)
graph_builder.add_node("summarize", summarize)

# Parallel execution: three nodes start at START
graph_builder.add_edge(START, "fetch_trends")
graph_builder.add_edge(START, "analyze_competitors")
graph_builder.add_edge(START, "extract_sentiment")

# All three feed into summarizer
graph_builder.add_edge("fetch_trends", "summarize")
graph_builder.add_edge("analyze_competitors", "summarize")
graph_builder.add_edge("extract_sentiment", "summarize")

graph = graph_builder.compile()


# ---------------------- Run Example ----------------------
inputs = {"query": "Smart Water Bottle"}
result = graph.invoke(inputs)


# ---------------------- Output ----------------------
print("\n=== Final Market Summary ===\n")
print(result["summary"])
