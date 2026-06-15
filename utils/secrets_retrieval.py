import boto3
import json
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _resolve_aws_credentials():
    """Resolve AWS credentials from the environment or Streamlit secrets.

    On Streamlit Cloud, secrets are not always exported as environment
    variables (nested keys never are), so boto3's default credential chain
    can come up empty. We therefore read them explicitly and pass them to
    the client. Locally, these come back as None and boto3 falls back to its
    default chain (~/.aws/credentials, env vars, IAM role, etc.).
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"

    try:
        import streamlit as st

        if not access_key and "AWS_ACCESS_KEY_ID" in st.secrets:
            access_key = st.secrets["AWS_ACCESS_KEY_ID"]
        if not secret_key and "AWS_SECRET_ACCESS_KEY" in st.secrets:
            secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
        if "AWS_DEFAULT_REGION" in st.secrets:
            region = st.secrets["AWS_DEFAULT_REGION"]
    except Exception:
        # Streamlit not available or no secrets file configured (local dev).
        pass

    return access_key, secret_key, region


def get_secret():
    secret_name = "document_pipeline"

    access_key, secret_key, region = _resolve_aws_credentials()

    # Passing None for the keys lets boto3 fall back to its default
    # credential chain (used for local development).
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # The operation fails due to a client error.
        raise e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)  # Assuming the secret is stored in JSON format
