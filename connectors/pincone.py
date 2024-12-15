import os
from pinecone import Pinecone

pinecone_api_key = os.getenv(pinecone_api_key)

# Initialize Pinecone client
def initialize_pinecone_client(pinecone_api_key):
    try:
        pc = Pinecone(pinecone_api_key)
        return pc
    except Exception as e:
        error = f"Error initializing Pinecone client: {e}")
        return error
