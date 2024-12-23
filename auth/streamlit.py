import streamlit as st
import hashlib
import os
from literalai import LiteralClient
from datetime import datetime


class AuthManager:
    def __init__(self):
        """Initialize AuthManager with LiteralAI client"""
        literal_api_key = os.getenv('LITERAL_API_KEY')
        if not literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")

        self.client = LiteralClient(api_key=literal_api_key)

        # Initialize session state variables if they don't exist
        if "authentication_status" not in st.session_state:
            st.session_state.authentication_status = None
        if "username" not in st.session_state:
            st.session_state.username = None
        if "name" not in st.session_state:
            st.session_state.name = None
        if "user_id" not in st.session_state:
            st.session_state.user_id = None

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _get_or_create_literal_user(self, email: str, name: str = None) -> dict:
        """Get or create a user in LiteralAI"""
        try:
            # Try to get existing user
            user = self.client.api.get_user(identifier=email)

            if user:
                return {
                    'id': getattr(user, 'id', None),
                    'identifier': getattr(user, 'identifier', None),
                    'metadata': getattr(user, 'metadata', {})
                }

            # Create new user if doesn't exist
            first_name = name.split()[0] if name else email
            last_name = ' '.join(name.split()[1:]) if name and len(name.split()) > 1 else ''

            metadata = {
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'created_at': datetime.utcnow().isoformat()
            }

            user = self.client.api.create_user(
                identifier=email,
                metadata=metadata
            )

            return {
                'id': getattr(user, 'id', None),
                'identifier': getattr(user, 'identifier', None),
                'metadata': getattr(user, 'metadata', {})
            }

        except Exception as e:
            st.error(f"Error managing LiteralAI user: {str(e)}")
            raise

    def render_login(self):
        """Render login form"""
        st.subheader("Login")

        if st.session_state.get("authentication_status"):
            st.write("Authentication successful")
            st.write(f"Username (email): {st.session_state.get('username')}")
            st.write(f"User ID: {st.session_state.get('user_id')}")

        email = st.text_input("Email", key="login_email").lower()
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            if email and password:
                try:
                    # Get or create LiteralAI user
                    literal_user = self._get_or_create_literal_user(email)

                    if literal_user and literal_user['id']:
                        # Set session state variables
                        st.session_state.authentication_status = True
                        st.session_state.username = email
                        st.session_state.name = literal_user['metadata'].get('name', email)
                        st.session_state.user_id = literal_user['id']

                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to retrieve user information")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
            else:
                st.error("Please enter both email and password")

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

            try:
                # Create user in LiteralAI
                literal_user = self._get_or_create_literal_user(email, name)

                if literal_user and literal_user['id']:
                    st.success("Registration successful! Please log in.")
                    st.session_state.authentication_status = None  # Reset auth status
                else:
                    st.error("Failed to create user")
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")

    def render_logout(self):
        """Render logout button"""
        if st.button("Logout", key="logout_button"):
            for key in ['authentication_status', 'username', 'name', 'user_id']:
                st.session_state[key] = None
            st.rerun()

    def show_user_details(self):
        """Show user profile details"""
        if not st.session_state.authentication_status:
            st.error("Please log in to view profile")
            return

        st.header("User Profile")
        try:
            user = self._get_or_create_literal_user(st.session_state.username)
            if user:
                st.write(f"**Name:** {user['metadata'].get('name', 'N/A')}")
                st.write(f"**Email:** {user['identifier']}")
                st.write(f"**Created:** {user['metadata'].get('created_at', 'N/A')}")
        except Exception as e:
            st.error(f"Error loading user profile: {str(e)}")