import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import tempfile
import os
from dotenv import load_dotenv

# Load .env and API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Ask the Docs", page_icon="📄")
st.title("📄 Ask the Docs – RAG App")

uploaded_file = st.file_uploader("Upload your PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("Ask a question about the document:")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith(".pdf") else TextLoader(tmp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    response = chain(
        {"input_documents": relevant_docs, "question": question},
        return_only_outputs=True
    )

    st.markdown("### 🧠 Answer:")
    st.write(response["output_text"])