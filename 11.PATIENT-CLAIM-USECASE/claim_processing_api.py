from fastapi import FastAPI
from pydantic import BaseModel
from claim_processing_agent import create_workflow

from dotenv import load_dotenv
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "default"

# Load environment variables for GROQ_API_KEY
load_dotenv()

app = FastAPI()
graph = create_workflow()


class ClaimRequest(BaseModel):
    patient_id: str
    treatment_code: str
    claim_details: str


class ClaimResponse(BaseModel):
    final_decision: str
    ai_feedback: str


@app.post("/process-claim", response_model=ClaimResponse)
async def process_claim(request: ClaimRequest):
    input_state = {
        "patient_id": request.patient_id,
        "treatment_code": request.treatment_code,
        "claim_details": request.claim_details
    }
    result = graph.invoke(input_state,config={"configurable":{"thread_id":"api-thread"}})
    return {
        "final_decision": result.get("final_decision"),
        "ai_feedback": result.get("ai_validation_feedback")
    }

