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

📌 How It Works
1️⃣ Extracting Information from a URL
When a user enters a URL, the tool:
🔹 Sends a request to the webpage using requests with a user-agent header to prevent blocking.
🔹 Parses the webpage's HTML content using BeautifulSoup.
🔹 Extracts the text inside <p> tags (paragraphs) to avoid irrelevant content like menus or scripts.

def fetch_clean_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()  # Ensure a successful response
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract only paragraph text
        text_content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return text_content if text_content else None
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"
💡 Why this approach?

Ensures only meaningful text is retrieved.
Ignores unnecessary content (navigation bars, ads, footers).
2️⃣ Storing & Retrieving Relevant Information
Once the content is extracted, we:
🔹 Split the text into small chunks (512 tokens each) using LangChain’s RecursiveCharacterTextSplitter to handle large documents.
🔹 Convert text chunks into LangChain Document objects.
🔹 Generate vector embeddings using HuggingFace’s sentence-transformers/all-MiniLM-L6-v2.
🔹 Store embeddings in FAISS (a fast, in-memory vector search database).
🔹 Retrieve relevant content when a user asks a question.


text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
texts = text_splitter.split_text(all_text)

docs = [Document(page_content=txt) for txt in texts]

# Create FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
💡 Why FAISS?

It allows fast similarity search across thousands of text chunks.
Efficient for retrieving the most relevant context before answering.
3️⃣ Generating Answers with LLaMA 2 (GGUF Model)
When a user submits a question:
🔹 The tool retrieves the most relevant text chunks using FAISS.
🔹 The retrieved content is fed into the LLaMA 2 model to generate an answer.
🔹 We format the input as a structured prompt for better responses.
🔹 The LLaMA-2-7B-Chat-GGUF model processes the prompt and generates a response.



💡 Why LLaMA 2 GGUF?

GGUF models are optimized for low memory usage.
Temperature = 0.01 ensures factual and stable responses.

