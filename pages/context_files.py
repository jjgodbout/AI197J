import streamlit as st
from literalai import LiteralClient
import os
from datetime import datetime


class ContextFileManager:
    def __init__(self):
        """Initialize ContextFileManager with LiteralAI client"""
        literal_api_key = os.getenv('LITERAL_API_KEY')
        if not literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")

        self.client = LiteralClient(api_key=literal_api_key)

    def upload_file(self, file_data, file_name: str, thread_id: str = None):
        """Upload a file to LiteralAI"""
        try:
            response = self.client.api.upload_file(
                content=file_data,
                thread_id=thread_id,
                mime="application/octet-stream"
            )
            return response
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return None

    def list_thread_files(self, thread_id: str):
        """List all files associated with a thread"""
        try:
            thread = self.client.api.get_thread(thread_id)
            return thread.get('files', []) if thread else []
        except Exception as e:
            st.error(f"Error listing files: {str(e)}")
            return []

    def render_interface(self):
        """Render the context files interface"""
        st.header("Context Files")

        # File upload section
        uploaded_file = st.file_uploader("Upload a context file", type=['txt', 'pdf', 'csv'])
        if uploaded_file:
            file_data = uploaded_file.read()

            # Get current thread ID if in a llm session
            thread_id = st.session_state.get('current_thread')

            if st.button("Upload File"):
                with st.spinner("Uploading file..."):
                    response = self.upload_file(
                        file_data=file_data,
                        file_name=uploaded_file.name,
                        thread_id=thread_id
                    )
                    if response:
                        st.success("File uploaded successfully!")

        # Display existing files
        st.subheader("Your Context Files")
        if thread_id := st.session_state.get('current_thread'):
            files = self.list_thread_files(thread_id)
            if files:
                for file in files:
                    with st.expander(f"{file.get('name', 'Unnamed file')}"):
                        st.write(f"Uploaded: {file.get('created_at', 'Unknown date')}")
                        # Add more file details or actions as needed
            else:
                st.info("No files uploaded yet.")
        else:
            st.info("Start a llm to upload context files.")