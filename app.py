import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from ctransformers import AutoModelForCausalLM
import os

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to fetch and clean webpage content
def fetch_clean_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text_content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return text_content if text_content else None

    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"

# Streamlit UI
st.title("Web Content Q&A Tool")
st.write("Enter URLs and ask questions based on their content.")

# Input URLs
urls = st.text_area("Enter one or more URLs (separated by new lines):")
urls = urls.split("\n") if urls else []

# Input question
question = st.text_input("Ask a question based on the content:")

if st.button("Submit"):
    if not urls or not question:
        st.error("Please enter URLs and a question.")
    else:
        try:
            all_text = ""
            for url in urls:
                content = fetch_clean_content(url)
                if content:
                    all_text += content + "\n\n"

            if not all_text.strip():
                st.error("No valid text found after processing. The webpage might be empty or JavaScript-rendered.")
                st.stop()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
            texts = text_splitter.split_text(all_text)

            if not texts:
                st.error("Text splitting failed. No valid chunks generated.")
                st.stop()

            # Convert text chunks to LangChain Document objects
            docs = [Document(page_content=txt) for txt in texts]

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Load GGUF model using CTransformers (Optimized for CPU)
            llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_type="llama",
    max_new_tokens=100,  
    temperature=0.01  
)

            # Define QA function with context truncation
            def generate_answer(question, context):
                if not context.strip():
                    return "No relevant content found in the provided URLs."

                # Truncate context to 400 tokens (to fit within 512-token limit)
                context = " ".join(context.split()[:400])

                # Proper prompt formatting for GGUF models
                prompt = f"""### Instruction:
You are a helpful assistant. Answer the user's question based only on the provided context.

### Context:
{context}

### Question:
{question}

### Answer:
"""

                output = llm(prompt)
                return output.strip()

            # Retrieve relevant content
            retriever = vectorstore.as_retriever()
            context_docs = retriever.invoke(question)

            if not context_docs:
                st.warning("No relevant information found in the provided URLs.")
            else:
                context = "\n".join([doc.page_content for doc in context_docs])  
                answer = generate_answer(question, context)
                st.success(f"Answer: {answer}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

