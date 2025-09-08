# rag_chatbot_app_improved.py

import os
from dotenv import load_dotenv
import streamlit as st
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import openai
import copy

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat")

# Azure Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# ----------------------------
# Initialize OpenAI API
# ----------------------------
openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-12-01-preview"

# ----------------------------
# Initialize Azure Search client
# ----------------------------
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# ----------------------------
# Helper: Retrieve top-k relevant chunks
# ----------------------------
def get_top_chunks(query, top_k=10):
    results = search_client.search(
        search_text=query,
        top=top_k
    )
    chunks = []
    for r in results:
        content = r.get("content")
        if content:
            chunks.append(content)
    return chunks

# ----------------------------
# Generate answer using chat model
# ----------------------------
def generate_answer(user_query, chat_history):
    # Retrieve relevant chunks
    context_chunks = get_top_chunks(user_query, top_k=10)
    context_text = "\n".join(context_chunks) if context_chunks else "" #python ternary operator

# Makes a deep copy of previous conversation.
# We use deepcopy so that modifications here do not accidentally change the original list structure elsewhere.
#Adding a system level instuction to guide the model's behavior.

    messages = copy.deepcopy(chat_history)
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context to answer questions accurately. "
            "If the context is insufficient, give the most careful, concise answer possible. "
            "Do not hallucinate beyond the context."
        )
    })
    #Adding the user query and context as the final message.
    #We are asking AI to answer the question based on the retrieved context.
    messages.append({
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"
    })
    # Call Azure OpenAI chat completion
    try:
        response = openai.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.2,#Makes the output more focused and deterministic.   
            max_tokens=1000
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {e}"

    # Update chat history
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Citi", page_icon="🏦")
st.title("Client Manual Chatbot 🏦")

if "chat_history" not in st.session_state: #remembering the state between user interactions
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="input")#takes the user input

if user_input: #if user has entered something
    answer, st.session_state.chat_history = generate_answer(user_input, st.session_state.chat_history)

# Display chat history
for msg in st.session_state.chat_history:
    role = "You" if msg["role"] == "user" else "Bot"
    st.markdown(f"**{role}:** {msg['content']}")

#python -m streamlit run c:/Projects/ragg-app/rag_chatbot_app.py