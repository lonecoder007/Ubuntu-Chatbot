import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"  # Set timeout to 180 seconds

# Load all markdown files from a parent directory recursively
def load_markdown_files(directory):
    loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    return documents


# Split text into smaller chunks for better embedding
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)


# Create embeddings using a Hugging Face model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Store embeddings in ChromaDB
def store_in_chromadb(documents, persist_directory="chroma_db"):
    embeddings = get_embedding_model()
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    db.persist()  # Save to disk
    print("✅ Vector store created and persisted successfully!")


# Main execution
if __name__ == "__main__":
    parent_directory = "demo_bot_data/ubuntu-docs"
    print("Loading markdown files...")
    documents = load_markdown_files(parent_directory)

    print(f"Loaded {len(documents)} markdown files.")

    print("✂Ssplitting documents into smaller chunks...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} document chunks.")

    print("Storing embeddings in ChromaDB...")
    store_in_chromadb(chunks)

    print("Done!")
