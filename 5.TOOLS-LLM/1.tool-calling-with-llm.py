from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# ============================================================
# 1️⃣ Load environment variables
# ============================================================
load_dotenv()  # Loads values from .env into environment

# Ensure your .env file contains:
# GROQ_API_KEY=your_groq_api_key_here

# Optional safety check
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in environment. Please set it in your .env file.")


# ============================================================
# 2️⃣ Define a tool
# ============================================================
@tool
def get_restaurant_recommendations(location: str):
    """Provides a list of top restaurant recommendations for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(location.lower(), ["No recommendations available for this location."])


# ============================================================
# 3️⃣ Initialize Groq model and bind the tool
# ============================================================
tools = [get_restaurant_recommendations]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Fast and powerful Groq model
    temperature=0.7
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4️⃣ Send a test message and invoke model
# ============================================================
messages = [
    HumanMessage(content="Recommend some restaurants in Munich.")
]

llm_output = llm_with_tools.invoke(messages)

# ============================================================
# 5️⃣ Print output
# ============================================================
print("🤖 Model Response:\n", llm_output)
