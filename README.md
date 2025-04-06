
# 📄 Chat with Multiple PDF Files using Gemini Pro & FAISS

This project allows you to upload and interact with multiple PDF documents using Google's Gemini Pro (ChatGoogleGenerativeAI) and FAISS for efficient vector-based document search. Built using LangChain, Streamlit, and Google Generative AI APIs, this app lets you ask context-aware questions from the uploaded PDFs.


## 🚀 Features

- 📂 Upload and process multiple PDFs
- 📄 Extract text from PDF files
- 🔍 Chunk the text and create vector embeddings using Google's Embedding Model
- 🧠 Semantic search with FAISS vector store
- 💬 Chat with documents using Gemini 1.5 Pro
- ⚡️ Real-time Q&A via Streamlit


## 🧠 Tech Stack

- Streamlit
- LangChain
- Google Generative AI



## 📦 Installation

1. Clone the repository:

```bash
  git clone https://github.com/yourusername/pdf-chat-gemini.git
  cd pdf-chat-gemini
```
2. Install dependencies:

```bash
  pip install -r requirements.txt
```
3. Set up your environment:

```bash
  GOOGLE_API_KEY=your_google_api_key_here
```
## 🛠️ How it Works

1. Upload multiple PDF files from the sidebar.
2. The app extracts text using PyPDF2.
3. Text is split into chunks using RecursiveCharacterTextSplitter.
4. Embeddings are generated using GoogleGenerativeAIEmbeddings.
5. A FAISS index is created and stored locally.
6. When a user asks a question, the app:
    - Performs similarity search in the FAISS vector DB.
    - Uses Gemini Pro to answer based on the relevant chunks.
## Authors

Built by Dilshan Upasena (GD)

- [LinkedIn](https://www.linkedin.com/in/dilshan-upasena-98gdu)
- [GitHub](https://www.github.com/G-Dilshan)


## License

[MIT](https://choosealicense.com/licenses/mit/)

