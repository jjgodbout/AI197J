from snowflake.snowpark.session import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import *
from typing import Any, Dict, Optional
from utils.secrets_retrieval import get_secret
from dotenv import load_dotenv
import streamlit as st
import os
import logging
import secrets
import string

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Retrieve AWS credentials from Secrets Manager
credentials = get_secret()

# Snowflake connection credentials (shared config)
SNOWFLAKE_ACCOUNT = credentials['SNOWFLAKE_ACCOUNT']
SNOWFLAKE_WAREHOUSE = 'COLBY_AI197J'
SNOWFLAKE_DATABASE = 'COLBY'
SNOWFLAKE_SCHEMA = credentials['SNOWFLAKE_SCHEMA']

# Admin credentials (for user provisioning)
SNOWFLAKE_ADMIN_USER = credentials['SNOWFLAKE_USER']
SNOWFLAKE_ADMIN_TOKEN = os.getenv('SNOWFLAKE_ADMIN_TOKEN')
SNOWFLAKE_ADMIN_PASSWORD = credentials['SNOWFLAKE_PASSWORD']
SNOWFLAKE_ADMIN_ROLE = 'ACCOUNTADMIN'

# Default role for app users
SNOWFLAKE_USER_ROLE = os.getenv('SNOWFLAKE_USER_ROLE', 'AI197J_USER')


class SnowflakeConnection:
    """
    Establishes a connection to Snowflake using a programmatic access token (PAT).

    If no access_token is provided, attempts to retrieve one from
    st.session_state.snowflake_token.
    """

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or self._get_token_from_session()
        if not self.access_token:
            raise ValueError(
                "No Snowflake access token available. "
                "Please log in or set SNOWFLAKE_ADMIN_TOKEN in .env."
            )
        self.connection_parameters = self._build_connection_parameters()
        self.session = None

    @staticmethod
    def _get_token_from_session() -> Optional[str]:
        """Retrieve the current user's Snowflake PAT from session state."""
        try:
            return st.session_state.get('snowflake_token')
        except Exception:
            # Not running inside Streamlit (e.g. tests, scripts)
            return None

    def _build_connection_parameters(self) -> Dict[str, Any]:
        """Build connection parameters using the access token as password."""
        return {
            "account": SNOWFLAKE_ACCOUNT,
            "user": st.session_state.get('snowflake_user', SNOWFLAKE_ADMIN_USER),
            "password": self.access_token,
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA,
            "role": st.session_state.get('snowflake_role', SNOWFLAKE_USER_ROLE),
        }

    def get_session(self) -> Session:
        """Establishes and returns the Snowflake connection session."""
        if self.session is None:
            try:
                self.session = Session.builder.configs(self.connection_parameters).create()
                self.session.sql_simplifier_enabled = True
            except Exception as e:
                logger.error(f"Failed to connect to Snowflake: {str(e)}")
                raise
        return self.session

    def close_session(self):
        """Closes the Snowflake session if it exists."""
        if self.session:
            self.session.close()
            self.session = None


class AdminSnowflakeConnection:
    """
    Admin connection used for provisioning Snowflake users and generating
    programmatic access tokens. Uses the admin PAT from environment.
    """

    def __init__(self):
        token = SNOWFLAKE_ADMIN_TOKEN or SNOWFLAKE_ADMIN_PASSWORD
        if not token:
            raise ValueError("No admin credentials available (set SNOWFLAKE_ADMIN_TOKEN in .env)")
        self.connection_parameters = {
            "account": SNOWFLAKE_ACCOUNT,
            "user": SNOWFLAKE_ADMIN_USER,
            "password": token,
            "warehouse": SNOWFLAKE_WAREHOUSE,
            "database": SNOWFLAKE_DATABASE,
            "schema": SNOWFLAKE_SCHEMA,
            "role": SNOWFLAKE_ADMIN_ROLE,
        }
        self.session = None

    def get_session(self) -> Session:
        if self.session is None:
            self.session = Session.builder.configs(self.connection_parameters).create()
            self.session.sql_simplifier_enabled = True
        return self.session

    def close_session(self):
        if self.session:
            self.session.close()
            self.session = None

    def create_snowflake_user(self, username: str, email: str, first_name: str, last_name: str) -> Dict[str, str]:
        """
        Create a Snowflake user and generate a programmatic access token.

        Returns dict with 'snowflake_user' and 'access_token' keys.
        """
        session = self.get_session()
        sf_username = self._sanitize_username(username)

        try:
            # Generate a temporary password (won't be used directly by the app)
            temp_password = self._generate_temp_password()

            # Create the Snowflake user
            session.sql(f"""
                CREATE USER IF NOT EXISTS {sf_username}
                    PASSWORD = '{temp_password}'
                    LOGIN_NAME = '{sf_username}'
                    DISPLAY_NAME = '{first_name} {last_name}'
                    EMAIL = '{email}'
                    DEFAULT_WAREHOUSE = '{SNOWFLAKE_WAREHOUSE}'
                    DEFAULT_ROLE = '{SNOWFLAKE_USER_ROLE}'
                    MUST_CHANGE_PASSWORD = FALSE
            """).collect()
            logger.info(f"Created Snowflake user: {sf_username}")

            # Grant the app role to the user
            session.sql(f"GRANT ROLE {SNOWFLAKE_USER_ROLE} TO USER {sf_username}").collect()
            logger.info(f"Granted role {SNOWFLAKE_USER_ROLE} to {sf_username}")

            # Generate a programmatic access token for the user
            token_name = f"ai197j_app_{sf_username}"
            result = session.sql(f"""
                ALTER USER {sf_username}
                    ADD PROGRAMMATIC ACCESS TOKEN
                    TOKEN_NAME = '{token_name}'
                    TOKEN_COMMENT = 'AI197J app access for {email}'
            """).collect()

            # Extract the token from the result
            access_token = str(result[0][0]) if result else None

            if not access_token:
                raise ValueError(f"Failed to generate access token for {sf_username}")

            logger.info(f"Generated PAT for user: {sf_username}")

            return {
                'snowflake_user': sf_username,
                'access_token': access_token,
                'token_name': token_name,
            }

        except Exception as e:
            logger.error(f"Error creating Snowflake user {sf_username}: {str(e)}")
            raise
        finally:
            self.close_session()

    def revoke_user_token(self, sf_username: str, token_name: str):
        """Revoke a user's programmatic access token."""
        session = self.get_session()
        try:
            session.sql(f"""
                ALTER USER {sf_username}
                    REMOVE PROGRAMMATIC ACCESS TOKEN
                    TOKEN_NAME = '{token_name}'
            """).collect()
            logger.info(f"Revoked token '{token_name}' for user {sf_username}")
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            raise
        finally:
            self.close_session()

    @staticmethod
    def _sanitize_username(username: str) -> str:
        """Convert email/username to a valid Snowflake username."""
        # Replace @ and . with underscores, uppercase
        sanitized = username.replace('@', '_').replace('.', '_').upper()
        # Remove any remaining invalid characters
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        return sanitized

    @staticmethod
    def _generate_temp_password() -> str:
        """Generate a secure temporary password."""
        alphabet = string.ascii_letters + string.digits + '!@#$%'
        return ''.join(secrets.choice(alphabet) for _ in range(24))
