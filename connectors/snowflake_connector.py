from snowflake.snowpark.session import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import *
from typing import Any, Dict
from utils.secrets_retrieval import get_secret
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Retrieve AWS credentials from Secrets Manager
credentials = get_secret()

#Snowflake connection credentials
SNOWFLAKE_ACCOUNT = credentials['SNOWFLAKE_ACCOUNT']
SNOWFLAKE_USER = credentials['SNOWFLAKE_USER']
SNOWFLAKE_PASSWORD = credentials['SNOWFLAKE_PASSWORD']
SNOWFLAKE_WAREHOUSE = 'COLBY_AI197J'
SNOWFLAKE_DATABASE = 'COLBY'
SNOWFLAKE_SCHEMA = credentials['SNOWFLAKE_SCHEMA']
SNOWFLAKE_ROLE = 'COLBY'

class SnowflakeConnection:
    """
    This class is used to establish a connection to Snowflake.

    Attributes
    ----------
    connection_parameters : Dict[str, Any]
        A dictionary containing the connection parameters for Snowflake.
    session : snowflake.snowpark.Session
        A Snowflake session object.

    Methods
    -------
    get_session()
        Establishes and returns the Snowflake connection session.

    """

    def __init__(self):
        self.connection_parameters = self._get_connection_parameters_from_env()
        self.session = None

    @staticmethod
    def _get_connection_parameters_from_env() -> Dict[str, Any]:
        connection_parameters = {
            "account": SNOWFLAKE_ACCOUNT,
            "user": SNOWFLAKE_USER,
            "password": SNOWFLAKE_PASSWORD,
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA ,
            "role": SNOWFLAKE_ROLE,
        }
        return connection_parameters

    def get_session(self):
        """
        Establishes and returns the Snowflake connection session.
        Returns:
            session: Snowflake connection session.
        """
        if self.session is None:
            self.session = Session.builder.configs(self.connection_parameters).create()
            self.session.sql_simplifier_enabled = True
        return self.session



