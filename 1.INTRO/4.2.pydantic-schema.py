# # Running code with pydantic base model 
# from pydantic import BaseModel
# from langgraph.graph import END, START, StateGraph
# from util.langgraph_util import display


# class HelloWorldState(BaseModel):
#     message: str


# def hello(state: HelloWorldState):
#     print(f"Hello Node: {state.message}")
#     return {"message": "Hello "+state.message}


# def bye(state: HelloWorldState):
#     print(f"Bye Node: {state.message}")
#     return {"message": "Bye "+state.message}


# graph = StateGraph(HelloWorldState)
# graph.add_node("hello",hello)
# graph.add_node("bye",bye)

# graph.add_edge(START,"hello")
# graph.add_edge("hello","bye")
# graph.add_edge("bye",END)

# runnable = graph.compile()

# output = runnable.invoke({"message": "Bharath"})
# print(output)



# # Running code with extra parameters but with out passing the new one parameter & ignoring the new parameter value
# from pydantic import BaseModel
# from langgraph.graph import END, START, StateGraph
# from util.langgraph_util import display


# class HelloWorldState(BaseModel):
#     message: str
#     id: int


# def hello(state: HelloWorldState):
#     print(f"Hello Node: {state.message}")
#     return {"message": "Hello "+state.message}


# def bye(state: HelloWorldState):
#     print(f"Bye Node: {state.message}")
#     return {"message": "Bye "+state.message}


# graph = StateGraph(HelloWorldState)
# graph.add_node("hello",hello)
# graph.add_node("bye",bye)

# graph.add_edge(START,"hello")
# graph.add_edge("hello","bye")
# graph.add_edge("bye",END)

# runnable = graph.compile()

# output = runnable.invoke({"message": "Bharath","id": 123})
# print(output)

# Making parameter as optional 
from pydantic import BaseModel
from langgraph.graph import END, START, StateGraph
from util.langgraph_util import display
from typing import Optional

class HelloWorldState(BaseModel):
    message: str
    id: Optional[int] = None


def hello(state: HelloWorldState):
    print(f"Hello Node: {state.message}")
    return {"message": "Hello "+state.message}


def bye(state: HelloWorldState):
    print(f"Bye Node: {state.message}")
    return {"message": "Bye "+state.message}


graph = StateGraph(HelloWorldState)
graph.add_node("hello",hello)
graph.add_node("bye",bye)

graph.add_edge(START,"hello")
graph.add_edge("hello","bye")
graph.add_edge("bye",END)

runnable = graph.compile()

output = runnable.invoke({"message": "Bharath"})
print(output)

