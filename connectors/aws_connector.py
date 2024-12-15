import pymysql
import time
from typing import Any, Dict
from utils.secrets_retrieval import get_secret
import boto3
import json


# Retrieve AWS credentials from Secrets Manager
credentials = get_secret()

#Watchtower Connection Credentials
WATCHTOWER_HOST = credentials['WATCHTOWER_HOST']
WATCHTOWER_PASSWORD = credentials['WATCHTOWER_PASSWORD']
WATCHTOWER_USERNAME = credentials['WATCHTOWER_USERNAME']
WATCHTOWER_PORT = credentials['WATCHTOWER_PORT']
WATCHTOWER_DB = credentials['WATCHTOWER_DB']

class MySQLAuroraConnection:
    """
    This class is used to establish a connection to MySQL Aurora on AWS.

    Attributes
    ----------
    connection_parameters : Dict[str, Any]
        A dictionary containing the connection parameters for MySQL Aurora.
    connection : pymysql.Connection
        A MySQL connection object.

    Methods
    -------
    get_connection()
        Establishes and returns the MySQL Aurora connection.

    """

    def __init__(self):
        self.connection_parameters = self._get_connection_parameters_from_env()
        self.connection = None

    @staticmethod
    def _get_connection_parameters_from_env() -> Dict[str, Any]:
        connection_parameters = {
            "host": WATCHTOWER_HOST,
            "user": WATCHTOWER_USERNAME,
            "password": WATCHTOWER_PASSWORD,
            "database": WATCHTOWER_DB,
            "port": 3306
        }
        return connection_parameters

    def get_connection(self, max_retries=3, retry_delay=2):
        """
        Establishes and returns the MySQL Aurora connection.

        Parameters:
            max_retries (int): Maximum number of connection retries.
            retry_delay (int): Delay in seconds between retries.

        Returns:
            connection: MySQL Aurora connection.
        """
        retries = 0
        while retries < max_retries:
            try:
                if self.connection is None or not self.connection.open:
                    self.connection = pymysql.connect(**self.connection_parameters)
                return self.connection
            except pymysql.MySQLError as e:
                print(f"Connection error: {e}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Unable to establish connection.")
                    raise

def initialize_s3_client(aws_access_key_id: str, aws_secret_access_key: str, region_name='us-east-1') -> boto3.client:
    """
    Initializes and returns an AWS S3 client.

    Parameters:
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.
        region_name (str): AWS region to connect to.

    Returns:
        boto3.client: Initialized S3 client.
    """
    return boto3.client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

def initialize_lambda_client(aws_access_key_id: str, aws_secret_access_key: str, region_name='us-east-1') -> boto3.client:
    """
    Initializes and returns an AWS Lambda client.

    Parameters:
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.
        region_name (str): AWS region to connect to.

    Returns:
        boto3.client: Initialized Lambda client.
    """
    return boto3.client(
        'lambda',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

def invoke_lambda_function(lambda_function_name: str, payload: Dict[str, Any], aws_access_key_id: str, aws_secret_access_key: str):
    """
    Invokes an AWS Lambda function.

    Parameters:
        lambda_function_name (str): Name of the Lambda function.
        payload (Dict[str, Any]): Payload to send to the Lambda function.
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.

    Returns:
        Response from the Lambda function.
    """
    lambda_client = initialize_lambda_client(aws_access_key_id, aws_secret_access_key)
    response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    return response

def initialize_sqs_client(aws_access_key_id: str, aws_secret_access_key: str, region_name='us-east-1') -> boto3.client:
    """
    Initializes and returns an AWS SQS client.
    ...
    """
    return boto3.client(
        'sqs',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

def send_message_to_sqs(queue_url: str, message_body: str, aws_access_key_id: str, aws_secret_access_key: str):
    """
    Sends a message to an AWS SQS queue.

    Parameters:
        queue_url (str): URL of the SQS queue.
        message_body (str): The message body to send.
        aws_access_key_id (str): AWS Access Key ID.
        aws_secret_access_key (str): AWS Secret Access Key.

    Returns:
        Response from the SQS service.
    """
    sqs_client = initialize_sqs_client(aws_access_key_id, aws_secret_access_key)
    response = sqs_client.send_message(
        QueueUrl=queue_url,
        MessageBody=message_body
    )
    return response

class ElastiCacheInspector:
    """
    This class is used to inspect ElastiCache clusters on AWS.

    Methods
    -------
    list_elasticache_clusters()
        Lists all ElastiCache clusters and prints basic information about each.
    """

    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, region_name='us-east-1'):
        self.client = boto3.client(
            'elasticache',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def list_elasticache_clusters(self):
        """
        Lists all ElastiCache clusters and prints basic information about each.
        """
        try:
            response = self.client.describe_cache_clusters(ShowCacheNodeInfo=True)

            for cluster in response['CacheClusters']:
                print(f"Cluster ID: {cluster['CacheClusterId']}")
                print(f"Engine: {cluster['Engine']}")
                print(f"Status: {cluster['CacheClusterStatus']}")
                print("Cache Nodes:")
                for node in cluster['CacheNodes']:
                    print(f"  - Node ID: {node['CacheNodeId']}, Status: {node['CacheNodeStatus']}")
                print("\n")

        except Exception as e:
            print(f"An error occurred: {e}")