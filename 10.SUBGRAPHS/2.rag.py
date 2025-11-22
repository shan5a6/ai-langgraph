# 2.rag_manual.py - FINAL WORKING VERSION (Manual Data)

from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Changed: Using simple TextLoader instead of complex web scrapers
from langchain_community.document_loaders import TextLoader 
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

# Load environment variables for GROQ_API_KEY
load_dotenv()

# --- PREREQUISITES ---
# Ensure qdrant-client is pinned for compatibility: pip install qdrant-client==0.11.0
# ---------------------

# -----------------------------
# 1️⃣ Data Source (Manual File)
# -----------------------------
NEWS_FILE_PATH = "news.txt"

# -----------------------------
# 2️⃣ Load Articles from File
# -----------------------------
def load_articles_from_file(file_path: str) -> List[Document]:
    """Loads text from a local file using TextLoader."""
    try:
        print(f"---LOADING DATA from {file_path}---")
        # TextLoader is robust and returns a List[Document]
        loader = TextLoader(file_path)
        docs_list = loader.load()
        print(f"Loaded {len(docs_list)} initial document(s).")
        return docs_list
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}. Please create news.txt.")
        return []

docs_list = load_articles_from_file(NEWS_FILE_PATH)

# -----------------------------
# 3️⃣ Split Articles into Chunks
# -----------------------------
# Use appropriate chunks for the structured text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Slightly smaller to ensure specific topics are retrieved
    chunk_overlap=50,
    separators=["\n---\n", "\n\n", "\n", " "] # Added the manual separator
)
doc_splits = text_splitter.split_documents(docs_list)
print(f"Data split into {len(doc_splits)} manageable chunks for Qdrant.")

# -----------------------------
# 4️⃣ Store News in Qdrant
# -----------------------------
embedding_model = FastEmbedEmbeddings()
qdrant = QdrantVectorStore.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    location="http://localhost:6333",
    collection_name="manual_current_affairs", # New collection name
    force_recreate=True # Ensures we use only the new data
)
retriever = qdrant.as_retriever(search_kwargs={"k": 3}) # Retrieve 3 top results

# -----------------------------
# 5️⃣ RAG Graph: Retrieve
# -----------------------------
class RAGGraphState(TypedDict):
    input: str
    data: List[Document] 

def retrieve_data(state: RAGGraphState):
    print("---Retrieve Data---")
    query = state["input"]
    docs = retriever.invoke(query)
    return {"data": docs}

def create_rag_workflow():
    workflow = StateGraph(RAGGraphState)
    workflow.add_node("retrieve_data", retrieve_data)
    workflow.add_edge(START, "retrieve_data")
    workflow.add_edge("retrieve_data", END)
    return workflow.compile()

rag_workflow = create_rag_workflow()

# -----------------------------
# 6️⃣ GROQ Model + Prompt for Summary
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a news analyst summarizing current affairs.
Use the retrieved news articles to answer the user's question. If the articles do not contain the answer, state that clearly.

Question:
{question}

News Articles:
{context}

Provide a concise, clear summary.
""")

llm = ChatGroq(model="llama-3.3-70b-versatile")
rag_chain = prompt | llm | StrOutputParser()

# -----------------------------
# 7️⃣ Full RAG Pipeline
# -----------------------------
class CurrentAffairsGraphState(TypedDict):
    question: str
    retrieved_news: List[Document]
    generation: str

def generate_current_affairs_summary(state: CurrentAffairsGraphState):
    print("---Generate News Summary---")
    question = state["question"]

    # 1) Retrieve news
    retrieved = rag_workflow.invoke({"input": question})
    retrieved_docs = retrieved["data"]
    
    # 2) Extract page_content for the LLM context
    context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # 3) LLM Summary
    summary = rag_chain.invoke({
        "question": question,
        "context": context_str
    })

    return {
        "question": question,
        "retrieved_news": retrieved_docs,
        "generation": summary
    }

def create_current_affairs_workflow():
    workflow = StateGraph(CurrentAffairsGraphState)
    workflow.add_node("generate_current_affairs_summary", generate_current_affairs_summary)
    workflow.add_edge(START, "generate_current_affairs_summary")
    workflow.add_edge("generate_current_affairs_summary", END)
    return workflow.compile()

current_affairs_graph = create_current_affairs_workflow()

# -----------------------------
# 8️⃣ Run Query
# -----------------------------
inputs = {"question": "Summarize the major geopolitical news, the latest on the Eurozone economy, and the status of Apex Dynamics."}
response = current_affairs_graph.invoke(inputs)

print("\n--- CURRENT AFFAIRS SUMMARY ---")
print(response["generation"])