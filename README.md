# Web-Content-Q-A-Tool

This is a Streamlit-based web application that allows users to:

✅ Enter URLs to extract and process webpage content.

✅ Ask questions based only on the extracted information.

✅ Retrieve relevant content using FAISS embeddings.

✅ Generate responses using the LLaMA 2 GGUF model with CTransformers (optimized for CPU).

Features
Extracts text from webpages using BeautifulSoup.
Splits text into manageable chunks with LangChain’s RecursiveCharacterTextSplitter.
Embeds text using HuggingFaceEmbeddings (MiniLM-L6-v2).
Stores embeddings in a FAISS vector database for efficient retrieval.
Uses TheBloke/Llama-2-7B-Chat-GGUF for answer generation.
Runs entirely on CPU, no need for a GPU!

🛠️ Installation
Follow these steps to set up and run the tool on your local machine.

Install Dependencies

pip install streamlit requests beautifulsoup4 langchain_huggingface faiss-cpu ctransformers


Run the code
streamlit run app.py
The app will open in your browser at http://localhost:8501 


<img width="1426" alt="image" src="https://github.com/user-attachments/assets/45fc7a6b-7e2f-44a4-810c-aba638f8b0b5" />

