# 📄 Ask the Docs – RAG Application

A mini RAG (Retrieval-Augmented Generation) app that lets you ask questions about any `.pdf` or `.txt` document using LLMs + vector search.

## 🎯 Features
- Upload PDF/TXT file
- Ask any question about its content
- Uses FAISS for vector similarity search
- Answers generated using OpenAI GPT-3.5
- Built using Streamlit + LangChain
- Docker & AWS EC2 deploy-ready

## ⚙️ How It Works
1. Document is uploaded
2. Text is split into 1000-character chunks
3. Embeddings are generated using OpenAI
4. Chunks stored in FAISS vector DB
5. User enters a question
6. Top similar chunks are retrieved
7. OpenAI LLM generates an answer using context

## 🧠 LLM Details
- Embedding Model: OpenAIEmbedding
- LLM: OpenAI GPT-3.5
- Frameworks: LangChain, FAISS, Streamlit

## ▶️ Run Locally
```bash
git clone https://github.com/vikramadityadevarampati/LLMOps.git
cd LLMOps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
streamlit run app.py
```

## 🐳 Run with Docker
```bash
docker build -t ask-the-docs .
docker run -p 8501:8501 ask-the-docs
```

## ☁️ Deploy to AWS EC2
1. Launch EC2 instance (Ubuntu 22.04, t2.micro)
2. SSH into instance:
```bash
sudo apt update
sudo apt install git python3-pip -y
git clone https://github.com/vikramadityadevarampati/LLMOps.git
cd LLMOps
pip3 install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
streamlit run app.py --server.port=8501 --server.enableCORS false --server.enableXsrfProtection false
```
3. Open port 8501 in EC2 Security Group

## 📬 Submission Format
✅ Public URL: http://your-ec2-ip:8501  
✅ GitHub repo: https://github.com/vikramadityadevarampati/LLMOps  
Notes:
- OpenAI GPT-3.5 used
- Streamlit + LangChain + FAISS
- Docker support included

## 👨‍💻 Author
Built with ❤️ by Vikram for the GenAI/LLMOps internship challenge at Convolution Engineering.