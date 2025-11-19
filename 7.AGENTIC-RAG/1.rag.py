from typing import List, TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables (optional for LLM keys)
load_dotenv()

# -------------------------------
# 1️⃣ Current Affairs News Sources
# -------------------------------
news_urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com/world",
    "https://www.nytimes.com/section/world",
    "https://www.reuters.com/world/",
    "https://www.aljazeera.com/news/"
]

# -------------------------------
# 2️⃣ Load Documents
# -------------------------------
print("⏳ Loading web documents...")
docs = [WebBaseLoader(url).load() for url in news_urls]
docs_list = [item for sublist in docs for item in sublist]
print(f"✅ Loaded {len(docs_list)} documents.")

# -------------------------------
# 3️⃣ Split into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)
doc_chunks = text_splitter.split_documents(docs_list)
print(f"✅ Split into {len(doc_chunks)} chunks.")

# -------------------------------
# 4️⃣ Setup embeddings & Qdrant
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

qdrant_client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "current_affairs_news"

# Initialize Qdrant vectorstore
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

# Add chunks to Qdrant (if not already added)
vectorstore.add_texts([chunk.page_content for chunk in doc_chunks])
retriever = vectorstore.as_retriever()
print("✅ Documents added to Qdrant and retriever initialized.")

# -------------------------------
# 5️⃣ Prompt template
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a news analyst summarizing the latest current affairs.
Use the retrieved articles to provide a concise summary.
Highlight key global events and developments.

Question: {question}
News Articles: {context}
Summary:
"""
)

# -------------------------------
# 6️⃣ Graph state definition
# -------------------------------
class CurrentAffairsGraphState(TypedDict):
    question: str
    retrieved_news: List[str]
    generation: str

# -------------------------------
# 7️⃣ Graph nodes
# -------------------------------
def retrieve_current_affairs(state: CurrentAffairsGraphState):
    print("---RETRIEVE CURRENT AFFAIRS---")
    question = state["question"]
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_news = [doc.page_content for doc in retrieved_docs]
    print(f"✅ Retrieved {len(retrieved_news)} documents.")
    return {"question": question, "retrieved_news": retrieved_news}

def generate_current_affairs_summary(state: CurrentAffairsGraphState):
    print("---GENERATE CURRENT AFFAIRS SUMMARY---")
    question = state["question"]
    retrieved_news = state["retrieved_news"]
    context_text = "\n\n--- Source Split ---\n\n".join(retrieved_news)
    generation = prompt.format(question=question, context=context_text)
    return {"question": question, "retrieved_news": retrieved_news, "generation": generation}

# -------------------------------
# 8️⃣ Graph workflow
# -------------------------------
def create_current_affairs_workflow():
    workflow = StateGraph(CurrentAffairsGraphState)
    workflow.add_node("retrieve_current_affairs", retrieve_current_affairs)
    workflow.add_node("generate_current_affairs_summary", generate_current_affairs_summary)
    workflow.add_edge(START, "retrieve_current_affairs")
    workflow.add_edge("retrieve_current_affairs", "generate_current_affairs_summary")
    workflow.add_edge("generate_current_affairs_summary", END)
    return workflow.compile()

# -------------------------------
# 9️⃣ Execute workflow
# -------------------------------
if __name__ == "__main__":
    current_affairs_graph = create_current_affairs_workflow()
    inputs = {"question": "What are the top global headlines today?"}
    print(f"\n--- Starting RAG for: '{inputs['question']}' ---\n")
    response = current_affairs_graph.invoke(inputs)
    print("\n--- CURRENT AFFAIRS SUMMARY ---")
    print(response["generation"])
