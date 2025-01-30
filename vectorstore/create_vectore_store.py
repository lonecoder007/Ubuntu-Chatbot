import os
import shutil
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Function to load the HuggingFace Embeddings model
def load_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load and return the HuggingFace Embeddings model."""
    print(f"Loading embeddings model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


# Function to load markdown files from a directory
def load_markdown_files(directory):
    """Loads all markdown files from the specified directory."""
    print("Loading markdown files...")
    loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    return documents


# Function to split documents into chunks
def split_documents(documents):
    """Splits the documents into smaller chunks."""
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


# Function to initialize or load ChromaDB
def load_or_create_chromadb(persist_directory="chroma_db", embeddings=None):
    """Load the existing ChromaDB or create a new one if it doesn't exist."""
    if os.path.exists(persist_directory):
        print("Loaded existing ChromaDB.")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("No existing ChromaDB found. A new one will be created.")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db


# Function to create a new ChromaDB from documents
def create_new_chromadb(directory, persist_directory="chroma_db"):
    """Create a new ChromaDB using markdown files from a specified directory."""
    documents = load_markdown_files(directory)
    chunks = split_documents(documents)

    embeddings = load_embeddings()  # Load the embeddings model
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    db.persist()  # Save to disk
    print("New ChromaDB created and persisted successfully!")


# Function to add new data to an existing ChromaDB
def add_new_data_to_chromadb(directory, persist_directory="chroma_db"):
    """Add new markdown data to an existing ChromaDB."""
    embeddings = load_embeddings()  # Load the embeddings model

    # Load or create the ChromaDB
    db = load_or_create_chromadb(persist_directory, embeddings)

    # Load and split new documents
    new_documents = load_markdown_files(directory)
    new_chunks = split_documents(new_documents)

    # Add the new data to the existing ChromaDB
    print("Adding new data to ChromaDB...")
    db.add_documents(new_chunks)
    db.persist()  # Save to disk
    print("New data added and persisted successfully!")


# Function to delete the ChromaDB
def delete_chromadb(persist_directory="chroma_db"):
    """Delete the ChromaDB and all its data."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)  # Deletes the directory and all its contents
        print("ChromaDB deleted successfully.")
    else:
        print("ChromaDB not found, nothing to delete.")


# Main execution
if __name__ == "__main__":
    print("Welcome to ChromaDB management!")

    # Menu to choose operation
    choice = input(
        "Select an operation:\n1. Create a new ChromaDB\n2. Add new data to existing ChromaDB\n3. Delete ChromaDB\nYour choice: ")

    if choice == '1':
        # Create new ChromaDB
        directory = input("Enter the directory containing markdown files to create a new ChromaDB: ")
        create_new_chromadb(directory)
    elif choice == '2':
        # Add new data to existing ChromaDB
        directory = input("Enter the directory containing markdown files to add to ChromaDB: ")
        add_new_data_to_chromadb(directory)
    elif choice == '3':
        # Delete ChromaDB
        delete_chromadb()
    else:
        print("Invalid choice. Please choose a valid option.")
