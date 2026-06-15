import streamlit as st
from dotenv import load_dotenv
import os
from auth.streamlit import AuthManager
from pages.context_files import ContextFileManager
from pages.audio_creator import AudioCreator
from pages.chatbot import render_chat_interface
from utils.query_handler import execute_sql
from chatbot.chatbot_manager import ChatManager

import logging
import sys


def setup_logging():
    """Configure application-wide logging"""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

    logger = logging.getLogger('main')
    logger.info("Logging configured successfully")


# Define page functions for navigation
def chatbot_page():
    if 'chat_manager' in st.session_state:
        render_chat_interface(st.session_state.chat_manager)
    else:
        st.error("Chat manager not properly initialized. Please refresh the page.")


def context_files_page():
    if 'context_manager' in st.session_state:
        st.session_state.context_manager.render_interface()
    else:
        st.error("Context manager not properly initialized.")


def audio_creator_page():
    if 'audio_creator' in st.session_state:
        st.session_state.audio_creator.render_interface()
    else:
        st.error("Audio creator not properly initialized.")


def user_profile_page():
    if 'auth_manager' in st.session_state:
        st.session_state.auth_manager.show_user_details()
    else:
        st.error("Auth manager not properly initialized.")


def initialize_session_state():
    """Initialize authentication-related session state (no Snowflake connection yet)"""
    try:
        # Initialize authentication related states
        if 'authentication_status' not in st.session_state:
            st.session_state.authentication_status = None
        if 'name' not in st.session_state:
            st.session_state.name = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'snowflake_token' not in st.session_state:
            st.session_state.snowflake_token = None
        if 'snowflake_user' not in st.session_state:
            st.session_state.snowflake_user = None
        if 'snowflake_role' not in st.session_state:
            st.session_state.snowflake_role = None

        print("Session state initialization complete")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        st.error(f"Error initializing application: {str(e)}")
        st.stop()


def initialize_managers():
    """Initialize managers that require Snowflake (called after authentication)"""
    try:
        if 'chat_manager' not in st.session_state:
            st.session_state.chat_manager = ChatManager(query_handler=execute_sql)
            print("Chat manager initialized")

        if 'context_manager' not in st.session_state:
            st.session_state.context_manager = ContextFileManager()
            print("Context manager initialized")

        if 'audio_creator' not in st.session_state:
            st.session_state.audio_creator = AudioCreator()
            print("Audio creator initialized")

    except Exception as e:
        print(f"Error during manager initialization: {str(e)}")
        st.error(f"Error initializing application: {str(e)}")
        if hasattr(e, '__cause__'):
            st.error(f"Caused by: {str(e.__cause__)}")
        st.stop()


def render_authenticated_interface(auth: AuthManager):
    """Render the interface for authenticated users"""
    try:
        print("Rendering authenticated interface...")

        # Sidebar welcome message and logout
        with st.sidebar:
            st.write(f'Welcome *{st.session_state["name"]}*')
            auth.render_logout()
            st.divider()

        # Define pages for navigation
        pages = {
            "Main": [
                st.Page(chatbot_page, title="Chatbot", icon="💬"),
                st.Page(context_files_page, title="Context Files", icon="📁"),
                st.Page(audio_creator_page, title="Audio Creator", icon="🎵")
            ]
        }

        # Setup navigation
        current_page = st.navigation(pages, position="sidebar")
        current_page.run()

        print("Rendered authenticated interface")
    except Exception as e:
        print(f"Error in authenticated interface: {str(e)}")
        st.error(f"Error rendering interface: {str(e)}")


def render_auth_pages(auth: AuthManager):
    """Render authentication pages"""
    try:
        print("Rendering auth pages...")
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            auth.render_login()
            if st.session_state.get("authentication_status") is False:
                st.error('Username/password is incorrect')
            elif st.session_state.get("authentication_status") is None:
                st.warning('Please enter your credentials')

        with tab2:
            auth.render_registration()

        print("Auth pages rendered")
    except Exception as e:
        print(f"Error in auth pages: {str(e)}")
        st.error(f"Error rendering auth pages: {str(e)}")


def main():
    """Main application entry point"""
    try:
        logger = logging.getLogger('main')
        logger.info("Starting AI197J System")
        st.title("AI Chatbot Tools")

        # Initialize session state and managers
        initialize_session_state()

        # Initialize authentication manager
        auth = AuthManager()
        st.session_state.auth_manager = auth
        print("Auth manager initialized")

        # Render appropriate interface based on authentication status
        if st.session_state.get("authentication_status"):
            # Initialize Snowflake-dependent managers only after login
            initialize_managers()
            render_authenticated_interface(auth)
        else:
            render_auth_pages(auth)

        print("Main function completed")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if hasattr(e, '__cause__'):
            st.error(f"Caused by: {str(e.__cause__)}")


if __name__ == "__main__":
    # Load environment variables at startup
    load_dotenv()

    # Configure page settings
    st.set_page_config(
        page_title="AI Chatbot Tools",
        page_icon="AI197J_logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Call setup_logging before any other initialization
    setup_logging()

    main()