from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
import psycopg
import requests
from langgraph.types import interrupt
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ---------------------- Define State ----------------------
class ClaimState(TypedDict):
    patient_id: str
    treatment_code: str
    claim_details: str
    patient_data: dict
    insurance_data: dict
    policy_docs: List[str]
    ai_validation_feedback: str
    final_decision: str
    _next: str  # For decision branching

# ---------------------- Constants ----------------------
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"
DB_CONFIG = {
    "dbname": "claims_db",
    "user": "myuser",
    "password": "mypassword",
    "host": "localhost"
}

# ---------------------- GROQ LLM ----------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")  # GROQ model

# ---------------------- Load Policy Documents ----------------------
loader = TextLoader("insurance_data.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# ---------------------- Qdrant Vector Store ----------------------
embedding_model = FastEmbedEmbeddings()
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    location="http://localhost:6333",
    collection_name="insurance_policies"
)

retriever = vector_store.as_retriever()

# ---------------------- Step 1: Fetch Patient Data ----------------------
def fetch_patient_data(state: ClaimState):
    patient_id = state["patient_id"]
    response = requests.get(f"{FHIR_BASE_URL}/Patient/{patient_id}")
    state["patient_data"] = response.json() if response.status_code == 200 else {"error": "Patient Not Found"}
    return state

# ---------------------- Step 2: Fetch Insurance Data ----------------------
def fetch_patient_insurance(state: ClaimState):
    patient_id = state["patient_id"]
    response = requests.get(f"{FHIR_BASE_URL}/Coverage?patient={patient_id}")
    state["insurance_data"] = response.json() if response.status_code == 200 else {"error": "Insurance Not Found"}
    return state

# ---------------------- Step 3: Retrieve Policy Documents ----------------------
def retrieve_policy_docs(state: ClaimState):
    query = f"Retrieve insurance policy details for {state['treatment_code']}"
    docs = retriever.invoke(query)
    state["policy_docs"] = [doc.page_content for doc in docs]
    return state

# ---------------------- Step 4: AI-Based Claim Validation ----------------------
def validate_claim(state: ClaimState):
    claim_text = f"""
    Claim Details: {state["claim_details"]}
    Patient Data: {state["patient_data"]}
    Insurance Coverage: {state["insurance_data"]}
    Retrieved Policies: {state["policy_docs"]}
    """
    response = llm.invoke(f"Validate the following claim. Should it be Approved, Rejected or need more info? {claim_text}")
    state["ai_validation_feedback"] = response.content
    return state

# ---------------------- Step 5: Decision Node ----------------------
def claim_decision(state: ClaimState):
    decision_text = state["ai_validation_feedback"].lower()
    if "more info" in decision_text:
        state["final_decision"] = "Request More Info"
        state["_next"] = "human_review"
    elif "approve" in decision_text:
        state["final_decision"] = "Approved"
        state["_next"] = "store_claim"
    elif "reject" in decision_text:
        state["final_decision"] = "Rejected"
        state["_next"] = "store_claim"
    return state

# ---------------------- Step 6: Store Decision in Database ----------------------
def store_claim(state: ClaimState):
    conn = psycopg.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO claims (patient_id,status,decision_details) VALUES (%s,%s,%s)",
        (state["patient_id"], state["final_decision"], state["ai_validation_feedback"])
    )
    conn.commit()
    cur.close()
    conn.close()
    return state

# ---------------------- Step 7: Human Review ----------------------
def human_review(state: ClaimState):
    state["final_decision"] = interrupt({"feedback": state["ai_validation_feedback"]})
    return state

# ---------------------- Build LangGraph Workflow ----------------------
def create_workflow():
    graph = StateGraph(ClaimState)
    graph.add_node("fetch_patient_data", fetch_patient_data)
    graph.add_node("fetch_patient_insurance", fetch_patient_insurance)
    graph.add_node("retrieve_policy_docs", retrieve_policy_docs)
    graph.add_node("validate_claim", validate_claim)
    graph.add_node("claim_decision", claim_decision)
    graph.add_node("store_claim", store_claim)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("fetch_patient_data")
    graph.add_edge("fetch_patient_data", "fetch_patient_insurance")
    graph.add_edge("fetch_patient_insurance", "retrieve_policy_docs")
    graph.add_edge("retrieve_policy_docs", "validate_claim")
    graph.add_edge("validate_claim", "claim_decision")
    graph.add_edge("human_review", "store_claim")

    # Conditional edges based on AI feedback
    graph.add_conditional_edges(
        "claim_decision",
        lambda state: state["_next"],
        {"store_claim": "store_claim", "human_review": "human_review"}
    )

    # In-memory checkpointer
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    workflow = create_workflow()
    inputs = {
        "patient_id": "12345",
        "treatment_code": "TREAT-001",
        "claim_details": "Patient underwent a routine checkup and minor surgery."
    }
    result = workflow.invoke(inputs)
    print("FINAL DECISION:", result["final_decision"])
