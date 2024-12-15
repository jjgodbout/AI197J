import boto3
import json
import os
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_secret():
    secret_name = "document_pipeline"
    region_name = "us-east-1"
    aws_access_key_id = os.getenv('aws_key_id')
    aws_secret_access_key = os.getenv('aws_sak')


    if aws_access_key_id is None or aws_secret_access_key is None:
        raise ValueError("AWS credentials are not set in environment variables")

    # Create a Secrets Manager client
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # The operation fails due to a client error.
        raise e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)  # Assuming the secret is stored in JSON format