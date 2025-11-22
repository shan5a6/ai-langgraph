from typing import TypedDict
from langgraph.graph import START,END,StateGraph

class HelloWorldState(TypedDict):
    message: str

def hello(state: HelloWorldState):
    print(f"Hello Node: {state["message"]}")
    return {"message": "Hello" + state["message"]}

def bye(state: HelloWorldState):
    print(f"Bye Node: {state["message"]}")
    return {"message": "Bye" + state["message"]}

# define node 
graph = StateGraph(HelloWorldState)
graph.add_node("hello", hello)
graph.add_node("bye", bye)

graph.add_edge(START,"hello")
graph.add_edge("hello", "bye")
graph.add_edge("bye", END)

compiled = graph.compile()
inner_graph = compiled.get_graph()
inner_graph.dr
print(inner_graph)
output = compiled.invoke({"message": "aidevops"})
print(output)