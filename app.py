import os
os.environ["GROQ_API_KEY"] = "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"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ------------------ LLM ------------------
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0
)


# ------------------ Load / Create DB ------------------
def load_db():
    if not os.path.exists("db"):
        loader = PyPDFLoader("data/knowledge.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = Chroma.from_documents(
            chunks,
            embedding,
            persist_directory="db"
        )
        db.persist()

    return Chroma(
        persist_directory="db",
        embedding_function=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    )


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="RAG Support Assistant")

st.title("🤖 Customer Support Assistant")

query = st.text_input("Ask your question:")

if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            db = load_db()
            retriever = db.as_retriever(search_kwargs={"k": 3})

            docs = retriever.get_relevant_documents(query)

            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are a helpful customer support assistant.

Use ONLY the context below:
{context}

Question: {query}

If answer is not found, say: I don't know.
"""

            response = llm.invoke(prompt)

        st.success(response.content)
    else:
        st.warning("Please enter a question.")