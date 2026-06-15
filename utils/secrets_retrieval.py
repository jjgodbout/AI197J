import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_secret():
    secret_name = "document_pipeline"
    region_name = "us-east-1"

    # Credentials are resolved via boto3's default credential chain
    # (~/.aws/credentials, environment, IAM role, etc.).
    session = boto3.session.Session()
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