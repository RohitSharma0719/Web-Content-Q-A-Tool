Web Content Q&A Tool – How It Works

Introduction
The Web Content Q&A Tool is designed to extract text from webpages, store and retrieve relevant content, and generate intelligent answers based on the extracted information. It is built using Streamlit for the user interface, BeautifulSoup for web scraping, FAISS for fast content retrieval, and LLaMA 2 GGUF for answering user queries.

1️⃣ Extracting Information from a Webpage
When a user provides one or more URLs, the tool sends an HTTP request to each webpage and retrieves its content. However, web pages contain a mix of useful and irrelevant content, such as navigation bars, ads, and scripts. To extract meaningful text, the tool:

Parses the HTML structure of the webpage.
Extracts only paragraph (<p>) elements where most meaningful content resides.
Filters out empty or redundant text segments.
If the webpage relies on JavaScript to load content dynamically, the extraction might be incomplete, as basic web scraping does not execute JavaScript.

2️⃣ Storing & Retrieving Relevant Information
Once the webpage content is extracted, it is often too large to process at once. To handle this:

The text is divided into smaller segments (chunks) to make retrieval efficient.
These chunks are converted into vector representations using a pre-trained Hugging Face sentence-transformer model.
The vectors are stored in FAISS, an optimized in-memory database for searching similar text efficiently.
When a user asks a question, the system:

Finds the most relevant text chunks by comparing the query’s vector representation with stored vectors.
Retrieves the top-matching text sections as the context for answering the query.
3️⃣ Generating Answers Using LLaMA 2 GGUF
Once relevant text is retrieved, it is used as context for answering the user’s question. However, language models require well-structured input to generate coherent responses. The system:

Formats the input as a structured instruction for the LLaMA 2 model.
Limits the context size to prevent exceeding the model’s token limit.
Uses a GGUF-optimized LLaMA 2 model, which is lightweight and runs efficiently on a CPU.
The model then analyzes the retrieved text and generates a relevant answer.

4️⃣ User Interaction via Streamlit UI
The entire process is integrated into a Streamlit-based user interface, where users can:

Enter one or more URLs to fetch content.
Type a question based on the extracted content.
Click Submit to receive an AI-generated answer.
If relevant content is found, the tool provides a response based only on the retrieved information. If no relevant text is available, it informs the user that no useful content was found.



Install Dependencies

pip install streamlit requests beautifulsoup4 langchain_huggingface faiss-cpu ctransformers


Run the code
streamlit run app.py
The app will open in your browser at http://localhost:8501 


<img width="1426" alt="image" src="https://github.com/user-attachments/assets/45fc7a6b-7e2f-44a4-810c-aba638f8b0b5" />



