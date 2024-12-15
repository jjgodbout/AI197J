import os
import cohere

cohere_api_key = os.getenv(cohere_api_key)

#Initialize Cohere client
def initialize_cohere_client(cohere_api_key):

    try:
        co = cohere.Client(cohere_api_key)
        return co
    except Exception as e:
        error = f"Error initializing Cohere client: {e}")
        return error

