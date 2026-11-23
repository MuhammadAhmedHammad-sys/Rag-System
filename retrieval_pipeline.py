import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persistent_directory = 'db/chromadb'

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hsnw:space":"cosine"}
)

query = "Which island does SpaceX lease for its launches in the Pacific?"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k":5,
#                    "score_threshold:":0.3}
# )

relevant_docs=retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
