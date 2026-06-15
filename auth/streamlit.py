import streamlit as st
import hashlib
import os
from datetime import datetime
from connectors.snowflake_connector import AdminSnowflakeConnection
from utils.query_handler import execute_sql
import logging

logger = logging.getLogger(__name__)

USERS_TABLE = "COLBY.AI197J.USERS"


class AuthManager:
    def __init__(self):
        """Initialize AuthManager with Snowflake-backed user management"""
        if "authentication_status" not in st.session_state:
            st.session_state.authentication_status = None
        if "username" not in st.session_state:
            st.session_state.username = None
        if "name" not in st.session_state:
            st.session_state.name = None
        if "user_id" not in st.session_state:
            st.session_state.user_id = None
        if "snowflake_token" not in st.session_state:
            st.session_state.snowflake_token = None
        if "snowflake_user" not in st.session_state:
            st.session_state.snowflake_user = None
        if "snowflake_role" not in st.session_state:
            st.session_state.snowflake_role = None

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _get_user_by_email(self, email: str):
        """Look up a user row from the USERS table via the admin connection."""
        try:
            admin = AdminSnowflakeConnection()
            session = admin.get_session()
            rows = session.sql(f"""
                SELECT * FROM {USERS_TABLE}
                WHERE EMAIL = '{email}'
            """).collect()
            admin.close_session()
            if rows:
                return dict(rows[0].asDict())
            return None
        except Exception as e:
            logger.error(f"Error looking up user {email}: {e}")
            return None

    def _create_user_row(self, email: str, first_name: str, last_name: str,
                         password_hash: str, sf_user: str = None,
                         sf_token: str = None, sf_token_name: str = None):
        """Insert a new user into the USERS table via the admin connection."""
        try:
            admin = AdminSnowflakeConnection()
            session = admin.get_session()
            session.sql(f"""
                INSERT INTO {USERS_TABLE}
                    (EMAIL, FIRST_NAME, LAST_NAME, PASSWORD_HASH,
                     SNOWFLAKE_USER, SNOWFLAKE_TOKEN, SNOWFLAKE_TOKEN_NAME)
                VALUES (
                    '{email}',
                    '{first_name.replace("'", "''")}',
                    '{last_name.replace("'", "''")}',
                    '{password_hash}',
                    '{sf_user or ""}',
                    '{(sf_token or "").replace("'", "''")}',
                    '{sf_token_name or ""}'
                )
            """).collect()
            admin.close_session()
            logger.info(f"Created user row for {email}")
        except Exception as e:
            logger.error(f"Error creating user row: {e}")
            raise

    def _provision_snowflake_user(self, email: str, first_name: str, last_name: str) -> dict:
        """Create a Snowflake user and generate a PAT."""
        try:
            admin = AdminSnowflakeConnection()
            result = admin.create_snowflake_user(
                username=email,
                email=email,
                first_name=first_name,
                last_name=last_name,
            )
            logger.info(f"Provisioned Snowflake user for {email}")
            return result
        except Exception as e:
            logger.error(f"Error provisioning Snowflake user: {e}")
            raise

    def _load_snowflake_credentials(self, user_row: dict):
        """Load Snowflake credentials from the user row into session state."""
        sf_token = user_row.get("SNOWFLAKE_TOKEN")
        sf_user = user_row.get("SNOWFLAKE_USER")
        if sf_token and sf_user:
            st.session_state.snowflake_token = sf_token
            st.session_state.snowflake_user = sf_user
            st.session_state.snowflake_role = os.getenv("SNOWFLAKE_USER_ROLE", "AI197J_USER")
            logger.info(f"Loaded Snowflake credentials for {sf_user}")
        else:
            logger.warning("No Snowflake credentials found for user")

    def render_login(self):
        """Render login form"""
        st.subheader("Login")

        if st.session_state.get("authentication_status"):
            st.write("Authentication successful")
            st.write(f"Username (email): {st.session_state.get('username')}")
            return

        email = st.text_input("Email", key="login_email").lower()
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            if not email or not password:
                st.error("Please enter both email and password")
                return

            user_row = self._get_user_by_email(email)
            if not user_row:
                st.error("User not found. Please register first.")
                return

            # Verify password
            if user_row["PASSWORD_HASH"] != self._hash_password(password):
                st.session_state.authentication_status = False
                st.error("Username/password is incorrect")
                return

            # Load Snowflake credentials into session state
            self._load_snowflake_credentials(user_row)

            # Set session state
            st.session_state.authentication_status = True
            st.session_state.username = email
            st.session_state.name = user_row.get("FIRST_NAME", email)
            st.session_state.user_id = str(user_row.get("ID", ""))

            st.success("Logged in successfully!")
            st.rerun()

    def render_registration(self):
        """Render registration form"""
        st.subheader("Register")

        name = st.text_input("Full Name", key="register_name")
        email = st.text_input("Email Address", key="register_email").lower()
        password = st.text_input("Password", type="password", key="register_password")
        password_repeat = st.text_input("Repeat Password", type="password", key="register_password_repeat")

        if st.button("Register", key="register_button"):
            if not all([name, email, password, password_repeat]):
                st.error("Please fill in all fields")
                return

            if password != password_repeat:
                st.error("Passwords do not match")
                return

            # Check if user already exists
            existing = self._get_user_by_email(email)
            if existing:
                st.error("An account with this email already exists. Please log in.")
                return

            try:
                name_parts = name.split()
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

                # Step 1: Provision Snowflake user + PAT
                with st.spinner("Setting up your account..."):
                    sf_credentials = self._provision_snowflake_user(email, first_name, last_name)

                # Step 2: Create the user row in USERS table
                self._create_user_row(
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    password_hash=self._hash_password(password),
                    sf_user=sf_credentials["snowflake_user"],
                    sf_token=sf_credentials["access_token"],
                    sf_token_name=sf_credentials["token_name"],
                )

                st.success("Registration successful! Please log in.")
                st.session_state.authentication_status = None

            except Exception as e:
                logger.error(f"Registration failed: {e}")
                st.error(f"Registration failed: {str(e)}")

    def render_logout(self):
        """Render logout button"""
        if st.button("Logout", key="logout_button"):
            for key in ["authentication_status", "username", "name", "user_id",
                        "snowflake_token", "snowflake_user", "snowflake_role"]:
                st.session_state[key] = None
            # Clear managers so they re-init on next login
            for key in ["chat_manager", "context_manager", "audio_creator"]:
                st.session_state.pop(key, None)
            st.rerun()

    def show_user_details(self):
        """Show user profile details"""
        if not st.session_state.authentication_status:
            st.error("Please log in to view profile")
            return

        st.header("User Profile")
        user_row = self._get_user_by_email(st.session_state.username)
        if user_row:
            st.write(f"**Name:** {user_row.get('FIRST_NAME', '')} {user_row.get('LAST_NAME', '')}")
            st.write(f"**Email:** {user_row.get('EMAIL', '')}")
            st.write(f"**Snowflake User:** {user_row.get('SNOWFLAKE_USER', 'Not configured')}")
            sf_status = "Connected" if st.session_state.get("snowflake_token") else "Not connected"
            st.write(f"**Snowflake Status:** {sf_status}")
