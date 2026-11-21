import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all the text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # Check if docs directoy exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    # Load all .txt files from the docs directoy
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc, in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content.strip()[:100]}...")
        print(f"metadate: {doc.metadata}")

    return documents

def split_documents(documents, chuck_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chuck_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print('-'*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) -5} more chunks")

        return chunks

def create_vector_store(chunks, persist_directory='db/chromadb'):
    """Crate and persist ChromaDB vector store"""
    print("\nCreating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metada={"hsnw:space": "cosine"}
    )
    print("--- Finished creating vector store")

    print(f"Vectore store created and saved to {persist_directory}")
    return vector_store

def main():
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path='docs')

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and Storing in Vector DB
    vector_store = create_vector_store(chunks)


if __name__ == "__main__":
    main()