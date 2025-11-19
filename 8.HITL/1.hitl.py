from langgraph.graph import END,START,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing import TypedDict


class TerraformState(TypedDict):
    inst: str
    code: str
    approve: bool
    response: str



def gen_code(state: TerraformState):
    code = f"Generated code for {state['inst']}"
    return Command(goto="approve_code", update={"code": code})

def approve_code(state: TerraformState):
    approve = interrupt(
        {
            "question": "Approve this code",
            "code": state['code']
        }
    )
    if approve:
       return Command(goto="apply_code")
    else:
       return Command(goto=END)

def apply_code(state: TerraformState):
    return {"response": "applied code successfully"}


def main():
    graph = StateGraph(TerraformState)
    graph.add_node("gen_code",gen_code)
    graph.add_node("approve_code",approve_code)
    graph.add_node("apply_code",apply_code)
    graph.set_entry_point("gen_code")
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


compiled = main()
config = {"configurable": {"thread_id": "1"}}
first_output = compiled.invoke({"inst": "Generate code for aws vpc creation"}, config=config)
interrupt_item = first_output['__interrupt__'][0].value
# Extract question and details
question = interrupt_item['question']
details = interrupt_item['code']

print("Question:", question)
print("code:", details)
user_input = input("Enter True / False: ").strip().lower() == "true"
print(user_input)
final_output = compiled.invoke(Command(resume=user_input), config=config)
print(final_output)
