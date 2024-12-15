import streamlit as st
from dotenv import load_dotenv
import os
from auth.streamlit import AuthManager
from pages.chatbot import ChatManager
from pages.context_files import ContextFileManager
from pages.audio_creator import AudioCreator

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
        # Only initialize if not already present
        if 'chat_manager' not in st.session_state:
            st.session_state.chat_manager = ChatManager()

        if 'context_manager' not in st.session_state:
            st.session_state.context_manager = ContextFileManager()

        if 'audio_creator' not in st.session_state:
            st.session_state.audio_creator = AudioCreator()

    except ValueError as e:
        st.error(f"Error initializing application: {str(e)}")
        st.stop()


def render_authenticated_interface(auth: AuthManager):
    """Render the interface for authenticated users"""
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
        st.session_state.chat_manager.render_chat_interface()
    elif selected == "Context Files":
        st.session_state.context_manager.render_interface()
    elif selected == "Audio Creator":
        st.session_state.audio_creator.render_interface()


def render_auth_pages(auth: AuthManager):
    """Render authentication pages"""
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        auth.render_login()
        if st.session_state.get("authentication_status") is False:
            st.error('Username/password is incorrect')
        elif st.session_state.get("authentication_status") is None:
            st.warning('Please enter your credentials')

    with tab2:
        auth.render_registration()


def main():
    """Main application entry point"""
    st.title("AI197J System")

    # Initialize session state and managers
    initialize_session_state()

    # Initialize authentication manager
    auth = AuthManager()

    # Render appropriate interface based on authentication status
    if st.session_state.get("authentication_status"):
        render_authenticated_interface(auth)
    else:
        render_auth_pages(auth)


if __name__ == "__main__":
    main()