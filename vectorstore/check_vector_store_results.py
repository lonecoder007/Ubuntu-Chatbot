from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Path to the ChromaDB directory
persist_directory = "chroma_db"

# Function to load embeddings (same as in the main code)
def load_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load and return the HuggingFace Embeddings model."""
    print(f"Loading embeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Function to query the ChromaDB
def query_chromadb(query, k=3):
    """Query the ChromaDB with a given search query and print results."""
    embeddings = load_embeddings()  # Load the embeddings model

    # Load the existing ChromaDB
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    print(f"🔍 Querying ChromaDB for: '{query}'")

    # Perform the similarity search
    results = db.similarity_search(query, k=k)  # Retrieve the top k results

    # Print the results
    if results:
        print(f"Found {len(results)} result(s):")
        for idx, result in enumerate(results):
            # Get the score from the metadata
            score = result.metadata.get('score', 'N/A')
            print(f"Result {idx + 1}:")
            print(f"Page content: {result.page_content[:200]}...")  # Print first 200 chars of the result
            print(f"Score: {score}")
    else:
        print(f"No results found for the query '{query}'.")

# Main function to run the query
if __name__ == "__main__":
    query = input("Enter your query to search the ChromaDB: ")
    query_chromadb(query)
