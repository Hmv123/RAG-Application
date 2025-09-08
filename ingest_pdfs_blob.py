# os, dotenv → load your secrets (keys, endpoints) from .env.
# BlobServiceClient → connects to Azure Blob Storage (where PDFs live).
# SearchClient → connects to Azure Cognitive Search index.
# openai → calls Azure OpenAI to generate embeddings.
# PdfReader → extracts text from PDF files.
# uuid → generates unique IDs for each chunk.
# BytesIO → allows reading blobs in memory (no need to save locally).

import os
import uuid
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import openai
from PyPDF2 import PdfReader
from io import BytesIO

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# Azure OpenAI setup
openai.api_type = os.getenv("OPENAI_API_TYPE")           # 'azure'
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")    # e.g., https://xxxx.openai.azure.com/
openai.api_version = os.getenv("OPENAI_API_VERSION")    # e.g., 2024-12-01-preview

embedding_model = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")  # embedding deployment name

# Azure Search setup
#search_client → lets us push documents into Azure AI Search index.
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

# Azure Blob Storage setup
#container_client → lets us list and fetch files from Blob Storage.
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_name = os.getenv("AZURE_STORAGE_CONTAINER")  # e.g., 'pdfs'
container_client = blob_service_client.get_container_client(container_name)

# ----------------------------
# PDF reading & chunking
# ----------------------------
def read_pdf_from_blob(blob_client):
    blob_data = blob_client.download_blob().readall() #donwload the blob data to memory
    reader = PdfReader(BytesIO(blob_data))
    text = ""
    #Read all pages and return a single string
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ----------------------------
# Process PDFs from Blob Storage
# ----------------------------
for blob in container_client.list_blobs():
    if blob.name.endswith(".pdf"):
        print(f"Processing {blob.name} from storage...")
        blob_client = container_client.get_blob_client(blob)
        
        text = read_pdf_from_blob(blob_client)
        chunks = chunk_text(text)
        docs_to_upload = []
#Sends each chunk to Azure OpenAI → gets a vector (list of 3072 numbers),This vector captures the semantic meaning of that chunk.
        for chunk in chunks:
            # Generate embedding
            embedding_response = openai.embeddings.create(
                model=embedding_model,
                input=chunk
            )
            #embeddings is a Json object with a data field that contains a list of #. Each embedding is itself a list of numbers.
            #data is a list of dictionary with the key for vector as embedding
            vector = embedding_response.data[0].embedding 

            docs_to_upload.append({
                "id": str(uuid.uuid4()),
                "content": chunk,
                "embedding": vector
            })
            
#----Optimized way to upload all chunks at once instead of one by one.-----
# Generate embeddings for all chunks in one request
# embedding_response = openai.embeddings.create(
#     model=embedding_model,
#     input=chunks
# )

# docs_to_upload = []
# for i, chunk in enumerate(chunks):
#     vector = embedding_response.data[i].embedding  # each chunk's embedding
#     docs_to_upload.append({
#         "id": str(uuid.uuid4()),
#         "content": chunk,
#         "embedding": vector
#     })


        # Upload chunks to Azure Search
        if docs_to_upload:
            result = search_client.upload_documents(documents=docs_to_upload)
            print(f"Uploaded {len(docs_to_upload)} chunks from {blob.name}")
        else:
            print(f"No chunks to upload for {blob.name}")
