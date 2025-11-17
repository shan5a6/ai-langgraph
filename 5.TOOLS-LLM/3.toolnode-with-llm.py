from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from util.langgraph_util import display
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

@tool
def get_restaurant_recommendations(location: str):
    """Provides a single top restaurant recommendation for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(location.lower(), ["No recommendations available."])


@tool
def book_table(restaurant: str, time: str):
    """Books a table at a specified restaurant and time."""
    return f"Table booked at {restaurant} for {time}."


# Bind the tool to the model
tools = [get_restaurant_recommendations, book_table]
model = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)
tool_node = ToolNode(tools)


# TODO: Define functions for the workflow
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# TODO: Define Conditional Routing
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# TODO: Define the workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

display(graph)
config = {"configurable": {"thread_id": "1"}}

# First invoke - Get one restaurant recommendation
response = graph.invoke(
    {"messages": [HumanMessage(content="Can you recommend just one top restaurant in Paris? "
                                       "The response should contain just the restaurant name")]},
    config
)

# TODO: Extract the recommended restaurant
recommended_restaurant = response["messages"][-1].content
print(recommended_restaurant)

response = graph.invoke(
    {"messages": [HumanMessage(content=f"Book a table at this restaurant")]},
    config
)

# TODO: Extract the recommended restaurant
final_response = response["messages"][-1].content
print(final_response)
