from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class HelloWorldState(TypedDict):
    message: str

def hello(state: HelloWorldState):
    print(f"Hello Node: {state['message']}")
    return {"message": "Hello " + state["message"]}

def bye(state: HelloWorldState):
    print(f"Bye Node: {state['message']}")
    return {"message": "Bye " + state["message"]}

graph = StateGraph(HelloWorldState)
graph.add_node("hello", hello)
graph.add_node("bye", bye)

# graph.add_edge(START, "hello")
graph.set_entry_point("hello")
graph.add_edge("hello", "bye")
graph.add_edge("bye", END)

# Visualize
compiled = graph.compile()
# Grab internal graph representation
inner_graph = compiled.get_graph()  # or maybe graph.get_graph() depending on version

# Example: write out a PNG (if supported)
img_data = inner_graph.draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img_data)
print("Saved graph image to graph.png")

# Then run
output = compiled.invoke({"message": "Bharath"})
print(output)
