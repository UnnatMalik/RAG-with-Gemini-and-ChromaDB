from math import e

from google import genai
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

api_key = os.getenv("API_KEY")

# client = genai.Client(api_key=api_key)

response = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",api_key_env_var="API_KEY",
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=response
)

client = genai.Client(api_key=api_key)

def load_text_document(path):
    print("==== Loading Document ====")
    document = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(path,filename), "r",encoding="utf-8"
            ) as file:
                document.append({"id":filename,"text":file.read()})
    return document

def split_text(text,chunk_size=1000,chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./news_articles"
documents = load_text_document(directory_path)
print(f"Loaded {len(documents)} documents")    

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== splitting documents into chunks ====")
    for i,chunk in enumerate(chunks):
        chunked_documents.append(
            {"id" : f"{doc['id']}_chunk{i+1}","text" : chunk}
        )

print(f"split documents into {len(chunked_documents)} chunks")

def get_gemini_embeddings(text):
    response = client.models.embed_content(
        contents=text,
        model="gemini-embedding-001"
    )
    embedding = response.embeddings[0].values
    print("==== Generating embeddings... ====")
    return embedding

for doc in chunked_documents:
    doc["embedding"] = get_gemini_embeddings(doc["text"])

print(doc["embedding"])

for doc in chunked_documents:
    print("==== Inserting Chunks into db;; ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

def query_documents(question,n_results=2):
    results = collection.query(query_texts=question,n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevent chunks ====")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    Response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return Response.text

if __name__ == "__main__":
    # question = "tell me about databricks ?"
    # relevant_chunks = query_documents(question)
    # answer = generate_response(question, relevant_chunks)
    # print(answer)
    while True:
        user_question = input("Ask a question (or type 'exit'): ").strip()
        if not user_question or user_question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        relevant_chunks = query_documents(user_question)
        answer = generate_response(user_question, relevant_chunks)
        print(answer)