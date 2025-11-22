import chainlit as cl
from claim_processing_agent import create_workflow  # Replace with your actual graph module
from langgraph.types import Command
from dotenv import load_dotenv
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "default"

# Load environment variables for GROQ_API_KEY
load_dotenv()

graph = create_workflow()

conversation_stage = "get_patient_id"
claim_info = {}
thread = {"configurable": {"thread_id": "101"}}


@cl.on_chat_start
async def on_start():
    global conversation_stage
    conversation_stage = "get_patient_id"
    await cl.Message("Welcome to the 💼 Claim Validation Assistant!").send()
    await cl.Message("🔹 Please enter the **Patient ID**:").send()


@cl.on_message
async def handle_message(message):
    global conversation_stage, claim_info

    if conversation_stage == "get_patient_id":
        claim_info["patient_id"] = message.content.strip()
        await cl.Message("🔹 Enter the **Treatment Code** (e.g., Z12.31):").send()
        conversation_stage = "get_treatment_code"
        return

    if conversation_stage == "get_treatment_code":
        claim_info["treatment_code"] = message.content.strip()
        await cl.Message("🔹 Enter the **Claim Details**:").send()
        conversation_stage = "get_claim_details"
        return

    if conversation_stage == "get_claim_details":
        claim_info["claim_details"] = message.content.strip()
        await cl.Message("🧰 Validating claim... Please wait.").send()

        state = graph.invoke(claim_info, config=thread)
        tasks = graph.get_state(config=thread).tasks

        if tasks:
            feedback = tasks[0].interrupts[0].value.get("feedback")
            await cl.Message(f"Human Review Needed: \n\n {feedback}").send()
            await cl.Message(f"Would you like to approve the claim? (yes/no)").send()
            conversation_stage = "await_approval"
            return

        # No interrupt, display results directly
        await show_results(state)
        conversation_stage = "restart"
        return

    if conversation_stage == "await_approval":
        if message.content.strip().lower() == "yes":
            state = graph.invoke(Command(resume="Approved"), config=thread)
        else:
            state = graph.invoke(Command(resume="Rejected"), config=thread)
        await cl.Message("Claim stored in database").send()
        await show_results(state)
        conversation_stage = "restart"
        return

    if conversation_stage == "restart" and message.content.strip().lower() == "restart":
        claim_info = {}
        conversation_stage = "get_patient_id"
        await cl.Message("🔄 Restarting process... Please enter the **Patient ID**:").send()
        return

    await cl.Message("⚠️ I did not understand that. Please follow the instructions.").send()


async def show_results(state):
    await cl.Message("📄 **Claim Summary:**").send()
    await cl.Message(f"👤 Patient Data:\n{state['patient_data']}").send()
    await cl.Message(f"💳 Insurance Data:\n{state['insurance_data']}").send()
    await cl.Message(f"📜 Policy Docs:\n{state['policy_docs']}").send()
    await cl.Message(f"🤖 AI Feedback:\n{state['ai_validation_feedback']}").send()
    await cl.Message(f"📌 Final Decision: **{state['final_decision']}**").send()
    await cl.Message("🔄 Type `restart` to process another claim.").send()


if __name__ == "__main__":
    # Run Chainlit server on 0.0.0.0:8080
    cl.run(host="0.0.0.0", port=8080)
