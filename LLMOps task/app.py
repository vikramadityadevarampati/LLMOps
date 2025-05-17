import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Ask the Docs", page_icon="📄")
st.title("📄 Ask the Docs – Mini RAG App")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("Ask a question about the document:")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load document
    loader = PyPDFLoader(file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(file_path)
    documents = loader.load()

    # Chunk text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retrieve relevant docs
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)

    # Run LLM
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    response = chain({"input_documents": relevant_docs, "question": question}, return_only_outputs=True)

    st.markdown("### 🧠 Answer:")
    st.write(response["output_text"])
