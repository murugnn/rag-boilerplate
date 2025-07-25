import os
import sys
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pickle
from typing import List, Dict, Tuple, Optional
import re
from dotenv import load_dotenv
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

class RAGSystem:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the RAG system with embedding model, Gemini client, and configurable chunking parameters.
        Args:
            embedding_model_name (str): Name of the SentenceTransformer model to use.
            chunk_size (int): The maximum number of words per text chunk.
            chunk_overlap (int): The number of overlapping words between chunks.
        """
        logging.info("Initializing RAG System...")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logging.info(f"Loaded embedding model: {embedding_model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
        
        # Configure Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            logging.error("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize Gemini model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Initialized Google Gemini model.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Configure generation parameters
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=800, # Increased for potentially more detailed answers
        )
        
        # Permissive safety settings for document analysis
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Initialize FAISS vector store
        self.index: Optional[faiss.IndexFlatIP] = None
        self.text_chunks: List[str] = []
        self.metadata: List[Dict] = []
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logging.info("RAG System initialized successfully.")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with metadata."""
        logging.info(f"Extracting text from: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text_data = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_data.append({
                        'text': text.strip(),
                        'page': page_num + 1,
                        'source': os.path.basename(pdf_path)
                    })
            doc.close()
            logging.info(f"Extracted text from {len(text_data)} pages.")
            return text_data
        except Exception as e:
            logging.error(f"An error occurred during PDF extraction from {pdf_path}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = re.findall(r'\S+', text)
        if not words:
            return []
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def process_and_embed_pdf(self, pdf_path: str):
        """Process a single PDF, create chunks, and generate vector embeddings."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logging.info(f"Processing PDF: {pdf_path}")
        text_data = self.extract_text_from_pdf(pdf_path)
        
        all_chunks = []
        all_metadata = []
        
        logging.info("Chunking text...")
        for page_data in text_data:
            chunks = self.chunk_text(page_data['text'])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'page': page_data['page'],
                    'source': page_data['source'],
                    'chunk_id': i,
                })
        
        if not all_chunks:
            logging.warning(f"No text chunks were generated from {pdf_path}. The PDF might be empty or image-based.")
            self.index = None
            return

        logging.info(f"Created {len(all_chunks)} text chunks.")
        
        logging.info("Generating embeddings for all chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        
        logging.info("Building FAISS index (using Cosine Similarity)...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings) # Normalize for Cosine Similarity
        self.index.add(embeddings.astype('float32'))
        
        self.text_chunks = all_chunks
        self.metadata = all_metadata
        
        logging.info("PDF processing and embedding complete.")
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Retrieve most relevant text chunks for a given query."""
        if self.index is None:
            raise ValueError("Vector index is not available. Please process a PDF first.")
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding) # Normalize query vector
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = [
            (self.text_chunks[idx], self.metadata[idx], float(score))
            for score, idx in zip(scores[0], indices[0]) if idx != -1
        ]
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, Dict, float]]) -> str:
        """Generate an answer using Gemini based strictly on the retrieved context."""
        if not retrieved_chunks:
            return "I could not find any relevant information in the document to answer this question."

        context_parts = []
        for chunk, metadata, score in retrieved_chunks:
            context_parts.append(f"--- Context from {metadata['source']}, Page {metadata['page']} ---\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an expert Question-Answering assistant. Your task is to answer the user's question based *ONLY* on the provided context below.

Follow these rules strictly:
1.  Base your answer entirely on the information within the "CONTEXT" section. Do not use any external knowledge.
2.  If the answer is not found in the context, you must state: "Based on the provided context, I cannot answer this question."
3.  For each piece of information you use, cite the source and page number in parentheses, like this: (Source: document.pdf, Page 5).
4.  Synthesize the information from different parts of the context into a coherent, well-written answer. Do not just copy-paste the chunks.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error during Gemini generation: {e}")
            return f"An error occurred while generating the answer: {e}"

    def ask(self, question: str, k: int = 7) -> Dict:
        """Main method to ask a question and get a complete response."""
        if self.index is None:
            return {"error": "No document has been processed. Please provide a PDF."}
        
        logging.info(f"\nReceived question: '{question}'")
        
        retrieved_chunks = self.retrieve_relevant_chunks(question, k)
        
        if not retrieved_chunks:
            logging.warning("No relevant chunks found for the query.")
            return {
                "question": question,
                "answer": "I could not find relevant information in the document to answer your question.",
                "sources": []
            }
        
        logging.info(f"Retrieved {len(retrieved_chunks)} relevant chunks for context.")
        answer = self.generate_answer(question, retrieved_chunks)
        
        # Get unique sources from the retrieved chunks for display
        unique_sources = {}
        for chunk, metadata, score in retrieved_chunks:
            source_key = f"{metadata['source']}_p{metadata['page']}"
            if source_key not in unique_sources:
                unique_sources[source_key] = {
                    "source": metadata['source'],
                    "page": metadata['page'],
                    "relevance_score": round(score, 4)
                }
        
        return {
            "question": question,
            "answer": answer,
            "sources": list(unique_sources.values())
        }

def main():
    """Main function to run the RAG system from the command line."""
    # --- MODIFIED: Check for PDF path from command-line arguments ---
    if len(sys.argv) != 2:
        print("=" * 70)
        print("Usage: python your_script_name.py <path_to_your_pdf_file>")
        print("Example: python rag_script.py documents/annual_report.pdf")
        print("=" * 70)
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        print(f"Error: The file '{pdf_path}' was not found or is not a PDF.")
        sys.exit(1)
    
    print("=" * 60)
    print("      Welcome to the Document Q&A System powered by Gemini")
    print("=" * 60)

    try:
        rag = RAGSystem(chunk_size=700, chunk_overlap=70)
        
        # --- MODIFIED: Process the provided PDF directly ---
        rag.process_and_embed_pdf(pdf_path)

    except (ValueError, Exception) as e:
        print(f"\nAn error occurred during initialization or processing: {e}")
        sys.exit(1)

    # Interactive question-answering loop
    print("\n" + "=" * 60)
    print(f"Document '{os.path.basename(pdf_path)}' is ready. Ask your questions!")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = rag.ask(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print("\n--- Answer ---")
                print(result['answer'])
                print("\n--- Sources ---")
                if result['sources']:
                    for source in result['sources']:
                        print(f"  - Document: {source['source']}, Page: {source['page']} (Top relevance: {source['relevance_score']:.4f})")
                else:
                    print("  No specific sources were cited.")
                print("-" * 15)
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.exception("An unexpected error occurred in the main loop.")
            print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    if not os.getenv('GEMINI_API_KEY'):
        print("\n" + "=" * 70)
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable not found.")
        print("Please create a '.env' file in this directory and add the key:")
        print("GEMINI_API_KEY=YOUR_API_KEY_HERE")
        print("=" * 70)
        sys.exit(1)
    
    main()