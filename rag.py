import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

def main(pdf_path):
    # Load environment variables from .env file
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please add your API key.")

    # Set the GOOGLE_API_KEY environment variable required by langchain-google-genai
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    # 1. Load PDF into documents
    print(f"Loading and extracting text from PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split docs into text chunks (with overlap to retain context)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(documents)
    print(f"Total text chunks after splitting: {len(docs_split)}")

    # 3. Create Google Gemini embeddings via LangChain wrapper
    print("Initializing Gemini embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

    # 4. Build Chroma vector store with embedded docs
    print("Building Chroma vectorstore (embedding documents)...")
    vectorstore = Chroma.from_documents(docs_split, embeddings, collection_name="pdf_chunks")

    # 5. Initialize LangChain Gemini chat model for generation
    print("Loading Gemini chat model for generation...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 6. Build RetrievalQA chain with retriever and Gemini LLM
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    print("\nRAG system ready. You can now ask questions about the PDF content. (type 'exit' to quit)")

    # 7. Interactive query loop
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = qa_chain.invoke({"query": query})['result']
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG with LangChain + Google Gemini API + .env")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    args = parser.parse_args()

    main(args.pdf)
