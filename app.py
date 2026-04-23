import os
import streamlit as st

# 🔑 API Key
os.environ["GROQ_API_KEY"] = "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"

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

# Load PDF directly (no vector DB)
loader = PyPDFLoader("data/knowledge.pdf")
docs = loader.load()

# LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0
)

st.title("🤖 Simple Support Assistant")

query = st.text_input("Ask your question:")

if st.button("Submit"):
    if query:
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer the question using this context:

{context}

Question: {query}
If not found, say I don't know.
"""

        response = llm.invoke(prompt)

        st.success(response.content)
