from openai import OpenAI
from typing import Optional, Dict, List
from datetime import datetime
import os
import os
from literalai import LiteralClient

import streamlit as st


class ChatManager:
    def __init__(self):
        """Initialize ChatManager with LiteralAI and OpenAI clients"""
        # Initialize LiteralAI client
        literal_api_key = os.getenv('LITERAL_API_KEY')
        if not literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")
        self.client = LiteralClient(api_key=literal_api_key)

        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.openai_client = OpenAI(api_key=openai_api_key)

    def get_or_create_thread(self, name: str, participant_id: str) -> Dict:
        """Create a new thread or get existing one"""
        try:
            # Get all threads and filter manually
            result = self.client.api.get_threads(first=100)  # Adjust limit as needed

            if hasattr(result, 'data'):
                # Look for existing thread with matching metadata
                for thread in result.data:
                    if (getattr(thread, 'name', '') == name and
                            getattr(thread, 'metadata', {}).get('user_id') == participant_id):
                        return thread

            # Create new thread if none exists
            thread = self.client.api.create_thread(
                name=name,
                metadata={
                    "user_id": participant_id,
                    "user_email": st.session_state.get('username'),
                    "created_at": datetime.utcnow().isoformat(),
                    "created_by_name": f"{st.session_state.get('name', 'User')}"
                }
            )

            return thread

        except Exception as e:
            st.error(f"Error in thread management: {str(e)}")
            raise

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get message history for a thread"""
        try:
            # Get thread to verify it exists
            thread = self.client.api.get_thread(id=thread_id)
            if not thread:
                return []

            # Get steps for the thread
            steps = getattr(thread, 'steps', [])
            if not steps:
                return []

            return [
                {
                    'input': {
                        'role': getattr(step, 'input', {}).get('role', 'user'),
                        'content': getattr(step, 'input', {}).get('content', '')
                    }
                }
                for step in steps
            ]

        except Exception as e:
            st.error(f"Error retrieving thread history: {str(e)}")
            return []

    def add_message(self, thread_id: str, content: str, role: str):
        """Add a new message to the thread"""
        try:
            # Add message as a step
            step = self.client.api.create_step(
                thread_id=thread_id,
                input={
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Update thread metadata
            self.client.api.update_thread(
                id=thread_id,
                metadata={
                    "last_updated": datetime.utcnow().isoformat(),
                    "last_message": content[:100] + "..." if len(content) > 100 else content
                }
            )

            return step

        except Exception as e:
            st.error(f"Error adding message: {str(e)}")
            raise

    def get_user_threads(self, participant_id: str) -> List[Dict]:
        """Get all threads for a user"""
        try:
            result = self.client.api.get_threads(first=100)  # Adjust limit as needed

            if not hasattr(result, 'data'):
                return []

            # Filter threads by user_id in metadata
            user_threads = []
            for thread in result.data:
                metadata = getattr(thread, 'metadata', {})
                if metadata.get('user_id') == participant_id:
                    user_threads.append({
                        'node': {
                            'id': getattr(thread, 'id', ''),
                            'name': getattr(thread, 'name', 'Untitled Chat'),
                            'metadata': metadata
                        }
                    })

            return sorted(
                user_threads,
                key=lambda x: x['node']['metadata'].get('last_updated', ''),
                reverse=True
            )

        except Exception as e:
            st.error(f"Error retrieving user threads: {str(e)}")
            return []

    def render_chat_interface(self):
        """Render the chat interface"""
        st.header("AI Chat")

        # Verify user is authenticated
        if not st.session_state.get("authentication_status"):
            st.error("Please log in to use the chat")
            return

        user_id = st.session_state.get("user_id")
        if not user_id:
            st.error("User ID not found in session")
            return

        # Get or create chat thread
        if 'current_thread' not in st.session_state:
            try:
                thread_name = f"Chat with {st.session_state.get('name', 'User')}"
                thread = self.get_or_create_thread(
                    name=thread_name,
                    participant_id=user_id
                )
                st.session_state.current_thread = getattr(thread, 'id', None)
                if not st.session_state.current_thread:
                    raise ValueError("Failed to get thread ID")
            except Exception as e:
                st.error(f"Error initializing chat: {str(e)}")
                return

        # Display thread selection sidebar
        with st.sidebar:
            st.subheader("Your Conversations")

            thread_edges = self.get_user_threads(user_id)
            if thread_edges:
                thread_names = []
                thread_mapping = {}

                for edge in thread_edges:
                    thread = edge['node']
                    name = thread.get('name', 'Untitled Chat')
                    thread_id = thread.get('id')
                    if thread_id:
                        thread_names.append(name)
                        thread_mapping[name] = thread_id

                if thread_names:
                    selected_thread = st.selectbox(
                        "Select Conversation",
                        thread_names,
                        key="thread_selector"
                    )

                    if selected_thread in thread_mapping:
                        st.session_state.current_thread = thread_mapping[selected_thread]

            if st.button("New Chat", key="new_chat_button"):
                try:
                    new_thread_name = f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    thread = self.get_or_create_thread(
                        name=new_thread_name,
                        participant_id=user_id
                    )
                    st.session_state.current_thread = getattr(thread, 'id', None)
                    if not st.session_state.current_thread:
                        raise ValueError("Failed to get thread ID")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating new chat: {str(e)}")

        # Display chat interface
        if current_thread_id := st.session_state.get('current_thread'):
            # Display chat messages
            messages = self.get_thread_history(current_thread_id)
            for message in messages:
                with st.chat_message(message['input']['role']):
                    st.write(message['input']['content'])

            # Chat input
            if prompt := st.chat_input("Type your message here...", key="chat_input"):
                # Add user message
                with st.chat_message("user"):
                    st.write(prompt)

                try:
                    # Add to thread
                    self.add_message(
                        thread_id=current_thread_id,
                        content=prompt,
                        role="user"
                    )

                    # Get AI response
                    with st.chat_message("assistant"):
                        response = self.openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            stream=True
                        )
                        response_text = st.write_stream(response)

                    # Add AI response to thread
                    self.add_message(
                        thread_id=current_thread_id,
                        content=response_text,
                        role="assistant"
                    )
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")