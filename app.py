import streamlit as st
from dotenv import load_dotenv
import os
from auth.streamlit import AuthManager
from pages.context_files import ContextFileManager
from pages.audio_creator import AudioCreator
from pages.chatbot import render_chat_interface
from utils.query_handler import execute_sql
from chatbot.chatbot_manager import ChatManager


# Load environment variables at startup
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="AI197J System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables and managers"""
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

        # Initialize authentication related states if not present
        if 'authentication_status' not in st.session_state:
            st.session_state.authentication_status = None
        if 'name' not in st.session_state:
            st.session_state.name = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        print("Session state initialization complete")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        st.error(f"Error initializing application: {str(e)}")
        if hasattr(e, '__cause__'):
            st.error(f"Caused by: {str(e.__cause__)}")
        st.stop()

def render_authenticated_interface(auth: AuthManager):
    """Render the interface for authenticated users"""
    try:
        print("Rendering authenticated interface...")
        # Sidebar navigation
        with st.sidebar:
            st.write(f'Welcome *{st.session_state["name"]}*')
            auth.render_logout()
            st.divider()

            # Navigation menu
            selected = st.selectbox(
                "Navigate to",
                ["User Profile", "Chatbot", "Context Files", "Audio Creator"]
            )

        # Main content area
        if selected == "User Profile":
            auth.show_user_details()

        elif selected == "Chatbot":
            if 'chat_manager' in st.session_state:
                # Use the new render_chat_interface function
                render_chat_interface(st.session_state.chat_manager)
            else:
                st.error("Chat manager not properly initialized. Please refresh the page.")

        elif selected == "Context Files":
            st.session_state.context_manager.render_interface()

        elif selected == "Audio Creator":
            st.session_state.audio_creator.render_interface()

        print(f"Rendered {selected} interface")
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
        print("Starting main function...")
        st.title("AI197J System")

        # Initialize session state and managers
        initialize_session_state()

        # Initialize authentication manager
        auth = AuthManager()
        print("Auth manager initialized")

        # Render appropriate interface based on authentication status
        if st.session_state.get("authentication_status"):
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
    main()