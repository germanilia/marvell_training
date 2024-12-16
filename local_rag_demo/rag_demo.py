import PyPDF2
from pathlib import Path
import ollama
import chromadb
from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Global configurations
EMBEDDING_MODEL = "mxbai-embed-large"
ANSWER_MODEL = "gemma"
CHROMA_CLIENT = chromadb.HttpClient(host="http://127.0.0.1:8000")

def load_pdf(pdf_path: str | Path) -> list[str]:
    """
    Load a PDF and split it into pages.
    
    Args:
        pdf_path (str | Path): Path to the PDF file
    
    Returns:
        list[str]: List of strings, where each string is the text content of a page
    """
    # Convert the path to Path object and check if file exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    pages = []
    
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get total number of pages
        num_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            
            # Extract text from the page
            text = page.extract_text()
            
            # Add the page text to our list
            pages.append(text)
    
    return pages

def split_text_into_chunks(text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Find the end of the chunk
        end = start + chunk_size
        
        # If this isn't the last chunk, try to find a good breaking point
        if end < text_length:
            # Try to find the last period or space within the chunk
            last_period = text[start:end].rfind('.')
            last_space = text[start:end].rfind(' ')
            
            # Use the latest good breaking point
            if last_period > 0:
                end = start + last_period + 1
            elif last_space > 0:
                end = start + last_space + 1
        
        # Add the chunk to our list
        chunks.append(text[start:end].strip())
        
        # Move the start position, accounting for overlap
        start = end - overlap

    return chunks

def create_embeddings_from_pdf(pdf_path: str | Path, collection_name: str, chunk_size: int = 1024, overlap: int = 50) -> None:
    """
    Create embeddings from PDF chunks and store them in ChromaDB collection.
    Each page is processed separately, and chunks maintain their page association.
    """
    # Load PDF pages
    pages = load_pdf(pdf_path)
    
    # Delete collection if it exists
    try:
        CHROMA_CLIENT.delete_collection(name=collection_name)
    except:
        pass
    
    collection = CHROMA_CLIENT.create_collection(name=collection_name)
    
    chunk_id = 0
    batch_size = 10  # Process in smaller batches
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    # Process each page separately
    for page_num, page_content in enumerate(pages, start=1):
        # Split page content into chunks
        chunks = split_text_into_chunks(page_content, chunk_size, overlap)
        
        # Process each chunk from this page
        for chunk_num, chunk in enumerate(chunks, start=1):
            # Skip empty chunks
            if not chunk or chunk.isspace():
                print(f"Skipping empty chunk {chunk_id}")
                continue
            
            try:
                # Generate embedding using Ollama
                response = ollama.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=chunk,
                )
                embedding = response["embedding"]
                
                # Validate embedding
                if not embedding or len(embedding) == 0:
                    print(f"Skipping chunk {chunk_id} due to empty embedding")
                    continue
                    
                # Add to batch
                ids.append(f"page_{page_num}_chunk_{chunk_num}")
                embeddings.append(embedding)
                documents.append(chunk)
                metadatas.append({
                    "page_number": page_num,
                    "chunk_number": chunk_num,
                    "total_chunks_in_page": len(chunks)
                })
                
                chunk_id += 1
                
                # If batch is full or this is the last chunk, add to collection
                if len(ids) >= batch_size or (page_num == len(pages) and chunk_num == len(chunks)):
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    # Clear batch
                    ids = []
                    embeddings = []
                    documents = []
                    metadatas = []
                    
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {str(e)}")
                continue

def query_similar_content(query: str, collection_name: str, n_results: int = 3) -> List[Dict]:
    """
    Query the collection for similar content using KNN search.
    
    Args:
        query (str): The query text
        collection_name (str): Name of the ChromaDB collection
        n_results (int): Number of results to return
    
    Returns:
        List[Dict]: List of results with content and metadata
    """
    collection = CHROMA_CLIENT.get_collection(collection_name)
    
    # Get all embeddings and documents from the collection
    all_results = collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )
    
    # Generate embedding for the query
    query_embedding = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query
    )["embedding"]
    
    # Convert embeddings to numpy array
    X = np.array(all_results['embeddings'])
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=min(n_results, len(X)), metric='cosine')
    knn.fit(X)
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors([query_embedding])
    
    # Format results
    formatted_results = []
    for idx, distance in zip(indices[0], distances[0]):
        formatted_results.append({
            'content': all_results['documents'][idx],
            'page_number': all_results['metadatas'][idx]['page_number'],
            'distance': distance
        })
    
    return formatted_results

def get_ai_response(query: str, context: List[Dict]) -> str:
    """
    Get AI-generated response based on the query and retrieved context.
    
    Args:
        query (str): The user's question
        context (List[Dict]): Retrieved similar content
    
    Returns:
        str: AI-generated response
    """
    # Prepare context for the prompt
    context_text = "\n\n".join([f"Page {r['page_number']}: {r['content']}" for r in context])
    
    # Construct the prompt
    prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

    # Generate response using Ollama
    print("----------------------------------------")
    print(prompt)
    print("----------------------------------------")
    response = ollama.generate(
        model=ANSWER_MODEL,
        prompt=prompt
    )
    
    return response['response']

if __name__ == "__main__":
    try:
        pdf_path = "rye.pdf"
        collection_name = "catcher_in_the_rye_new_2"
        
        print("Welcome to the PDF Chat Assistant!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("----------------------------------------")
        
        while True:
            # Get user input
            query = input("\nYour question: ").strip()
            
            # Check for exit command
            if query.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            # Skip empty questions
            if not query:
                print("Please ask a question!")
                continue
            
            try:
                # Get similar content
                results = query_similar_content(query, collection_name)
                
                # Get AI response
                ai_response = get_ai_response(query, results)
                
                # Print results
                print("\nAI Response:")
                print(ai_response)
                
                # Print the pages used for the response
                used_pages = sorted(set(result['page_number'] for result in results))
                print(f"\nThis response was generated using content from page(s): {', '.join(map(str, used_pages))}")
                
                print("\n----------------------------------------")
                
            except Exception as e:
                print(f"\nError processing your question: {e}")
                print("Please try asking another question.")
                
    except Exception as e:
        print(f"Error: {e}")
