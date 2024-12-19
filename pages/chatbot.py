from typing import Optional, Dict, List, Any
from datetime import datetime
import os
import streamlit as st
from literalai import LiteralClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManager
import uuid
from enum import Enum
from langchain_core.callbacks import BaseCallbackHandler
from llm.types import LLMConfig
from llm.repository import LLMRepository
from llm.factory import LLMFactory
from utils.query_handler import execute_sql
from literalai.observability.generation import ChatGeneration

class StepType(Enum):
    LLM = "llm"
    TOOL = "tool"
    RUN = "run"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    UNDEFINED = "undefined"

class ChatManager:
    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant. You aim to be accurate, 
    informative, and engaging while maintaining a natural conversational style."""

    def __init__(self, query_handler):
        """Initialize ChatManager with LangChain integration"""
        # Initialize session state for chat settings
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = self.DEFAULT_SYSTEM_MESSAGE
        if 'use_history' not in st.session_state:
            st.session_state.use_history = True
        # Initialize LiteralAI client
        literal_api_key = os.getenv('LITERAL_API_KEY')
        if not literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")
        self.literal_client = LiteralClient(api_key=literal_api_key)

        # Setup Langchain callback
        callback = self.literal_client.langchain_callback()

        # Convert the callback to a proper BaseCallbackHandler
        if not isinstance(callback, BaseCallbackHandler):
            from langchain_core.callbacks import LangChainCallback
            callback = LangChainCallback(callback)

        # Create callback manager with proper type
        self.callback_manager = CallbackManager([callback])

        # Setup repository
        self.repository = LLMRepository(query_handler)

        # Initialize with default model (gpt-3.5-turbo)
        self.current_config = None
        self.llm = None
        self.initialize_default_model()

    def initialize_default_model(self):
        """Initialize with gpt-3.5-turbo"""
        try:
            model_config = self.repository.get_model("gpt-3.5-turbo")
            if not model_config:
                raise ValueError("Default model gpt-3.5-turbo not found in database")

            self.current_config = model_config
            self.llm = LLMFactory.create_model(model_config, self.callback_manager)
        except Exception as e:
            raise ValueError(f"Failed to initialize default model: {str(e)}")

    def _create_step_metadata(self, step_type: StepType, additional_metadata: Optional[Dict] = None) -> Dict:
        """Create standardized metadata for steps"""
        user_email = st.session_state.get('username')
        metadata = {
            "step_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "step_type": step_type.value,
            "session_id": st.session_state.get('session_id'),
            "user_id": st.session_state.get('user_id'),
            "user_email": user_email,
            "model_info": {
                "model_id": self.current_config.model_id,
                "model_name": self.current_config.model_name,
                "provider": self.current_config.provider,
                "temperature": self.current_config.temperature
            } if self.current_config else None,
            "client_info": {
                "platform": "streamlit",
                "version": "1.0.0"
            }
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        return metadata

    def _log_step(self, thread_id: str, step_type: StepType, content: Dict[str, Any],
                  metadata: Optional[Dict] = None) -> Dict:
        """Create a step with enhanced logging"""
        try:
            step_metadata = self._create_step_metadata(step_type, metadata)
            return self.literal_client.api.create_step(
                thread_id=thread_id,
                type=step_type.value,
                input={
                    "content": content,
                    "metadata": step_metadata
                }
            )
        except Exception as e:
            st.error(f"Error logging step: {str(e)}")
            raise

    def get_or_create_thread(self, name: str, participant_id: str) -> Dict:
        """Create a new thread with LangChain integration"""
        try:
            user_email = st.session_state.get('username')

            # Debug logging
            print(f"Creating thread with name: {name}")
            print(f"User Email: {user_email}")
            print(f"Current session state: {st.session_state}")

            # First, get user by identifier (email)
            try:
                literal_user = self.literal_client.api.get_user(identifier=user_email)
                print(f"Found LiteralAI user: {literal_user.id}")

                if not literal_user:
                    raise ValueError("User not found in LiteralAI")

            except Exception as e:
                print(f"Error getting LiteralAI user: {str(e)}")
                raise ValueError(f"Failed to get LiteralAI user: {str(e)}")

            # Create thread using the LiteralAI user ID
            thread = self.literal_client.api.create_thread(
                name=name,
                participant_id=literal_user.id,  # Use LiteralAI user ID instead of email
                metadata={
                    "user_email": user_email,
                    "literal_user_id": literal_user.id,
                    "created_at": datetime.utcnow().isoformat(),
                    "model_id": self.current_config.model_id,
                    "model_name": self.current_config.model_name,
                    "provider": self.current_config.provider,
                    "temperature": self.current_config.temperature,
                    "thread_type": "langchain_chat"
                }
            )

            print(f"Created thread: {thread.id}")
            print(f"Thread metadata: {thread.metadata}")

            # Initialize with system message
            self._log_step(
                thread_id=thread.id,
                step_type=StepType.LLM,
                content={
                    "role": "system",
                    "content": self.DEFAULT_SYSTEM_MESSAGE
                },
                metadata={
                    "message_type": "system",
                    "is_system_message": True,
                    "user_email": user_email,
                    "literal_user_id": literal_user.id
                }
            )

            return thread

        except Exception as e:
            print(f"Detailed error in thread management: {str(e)}")
            st.error(f"Error in thread management: {str(e)}")
            raise

    def process_message(self, thread_id: str, prompt: str):
        try:
            # Get conversation history based on use_history setting
            conversation_history = (
                self.get_thread_history(thread_id)
                if st.session_state.get('use_history', True)
                else []
            )

            user_email = st.session_state.get('username')
            literal_user = self.literal_client.api.get_user(identifier=user_email)

            # Log user message
            self._log_step(
                thread_id=thread_id,
                step_type=StepType.LLM,
                content={
                    "role": "user",
                    "content": prompt
                },
                metadata={
                    "user_email": user_email,
                    "literal_user_id": literal_user.id if literal_user else None
                }
            )

            # Build message chain
            messages = []

            # Create system message that includes both instructions and context
            system_content = st.session_state.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE)
            context_text = st.session_state.get('context_text', "").strip()

            if context_text:
                system_content += f"\n\nContext to consider in your responses:\n{context_text}"

            messages.append(SystemMessage(content=system_content))

            # Add conversation history if enabled
            if st.session_state.get('use_history', True):
                for msg in conversation_history:
                    role = msg.get('input', {}).get('role')
                    content = msg.get('input', {}).get('content')
                    if role == 'user':
                        messages.append(HumanMessage(content=content))
                    elif role == 'assistant':
                        messages.append(AIMessage(content=content))

            # Add new message
            messages.append(HumanMessage(content=prompt))

            # Update model configuration with current settings
            if hasattr(self.current_config, 'temperature'):
                self.current_config.temperature = st.session_state.get('chat_temperature',
                                                                       self.current_config.temperature)
                self.llm = LLMFactory.create_model(self.current_config, self.callback_manager)

            # Get streaming response
            response = self.llm.stream(messages)
            response_chunks = []

            for chunk in response:
                if isinstance(chunk, AIMessage):
                    response_chunks.append(chunk.content)
                    yield chunk.content

            # Log complete response
            full_response = ''.join(response_chunks)

            # Create generation metadata including context info
            generation = ChatGeneration(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                message_completion={
                    "role": "assistant",
                    "content": full_response
                },
                model=self.current_config.model_name,
                provider=self.current_config.provider
            )

            # Log generation
            self.literal_client.api.create_generation(generation)

            # Log assistant response with context metadata
            self._log_step(
                thread_id=thread_id,
                step_type=StepType.LLM,
                content={
                    "role": "assistant",
                    "content": full_response
                },
                metadata={
                    "token_count": len(full_response.split()),
                    "completion_type": "stream",
                    "user_email": user_email,
                    "literal_user_id": literal_user.id if literal_user else None,
                    "model": self.current_config.model_name,
                    "provider": self.current_config.provider,
                    "system_prompt": st.session_state.get('system_prompt'),
                    "temperature": self.current_config.temperature,
                    "use_history": st.session_state.get('use_history', True),
                    "has_context": bool(context_text)
                }
            )

        except Exception as e:
            error_data = {
                "error_type": "MessageProcessingError",
                "error_message": str(e),
                "context": {
                    "prompt": prompt,
                    "thread_id": thread_id,
                    "user_email": st.session_state.get('username')
                }
            }
            self._log_step(
                thread_id=thread_id,
                step_type=StepType.RUN,
                content=error_data,
                metadata={"is_error": True}
            )
            raise

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get message history for a thread"""
        try:
            # Get thread with its steps
            thread = self.literal_client.api.get_thread(id=thread_id)
            if not thread:
                print(f"Thread not found: {thread_id}")
                return []

            messages = []
            steps = getattr(thread, 'steps', [])

            # Process each step in the thread
            for step in steps:
                input_data = getattr(step, 'input', {})
                if isinstance(input_data, dict) and input_data.get('content', {}).get('content', '').strip():
                    messages.append({
                        'input': input_data.get('content', {})
                    })

            print(f"Retrieved {len(messages)} messages from thread {thread_id}")
            return messages

        except Exception as e:
            print(f"Error retrieving thread history: {str(e)}")
            st.error(f"Error retrieving thread history: {str(e)}")
            return []

    def get_user_threads(self, user_email: str) -> List[Dict]:
        """Get all threads for a specific user"""
        try:
            # First get the LiteralAI user ID
            literal_user = self.literal_client.api.get_user(identifier=user_email)
            if not literal_user:
                raise ValueError("User not found in LiteralAI")

            print(f"Getting threads for LiteralAI user ID: {literal_user.id}")

            # Get threads filtering by participant_id (using LiteralAI user ID)
            threads = self.literal_client.api.get_threads(
                filters={
                    "field": "participantId",
                    "operator": "eq",
                    "value": literal_user.id
                },
                order_by={
                    "column": "createdAt",
                    "direction": "DESC"
                }
            )

            print(f"Found {len(threads.data) if hasattr(threads, 'data') else 0} threads")
            return threads.data if hasattr(threads, 'data') else []

        except Exception as e:
            print(f"Error getting user threads: {str(e)}")
            st.error(f"Error getting user threads: {str(e)}")
            return []

    def switch_model(self, model_id: str) -> None:
        """Switch to a different model"""
        try:
            model_config = self.repository.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in database")

            print(f"Switching to model: {model_config.model_name}")
            self.current_config = model_config
            self.llm = LLMFactory.create_model(model_config, self.callback_manager)
            print("Model switch successful")

        except Exception as e:
            print(f"Error switching model: {str(e)}")
            raise ValueError(f"Failed to switch model: {str(e)}")

    def get_thread_stats(self, thread_id: str) -> Dict:
        """Get statistics for a thread"""
        try:
            thread = self.literal_client.api.get_thread(id=thread_id)
            if not thread:
                return {}

            # Get generations for the thread
            generations = self.literal_client.api.get_generations(
                filters={
                    "field": "threadId",
                    "operator": "eq",
                    "value": thread_id
                }
            )

            # Calculate statistics
            stats = {
                "total_messages": len(getattr(thread, 'steps', [])),
                "total_generations": len(generations.data) if hasattr(generations, 'data') else 0,
                "created_at": thread.created_at,
                "last_message": max([step.created_at for step in getattr(thread, 'steps', [])], default=None),
                "model": thread.metadata.get('model_name') if thread.metadata else None,
            }

            return stats

        except Exception as e:
            print(f"Error getting thread stats: {str(e)}")
            return {}


    def render_chat_interface(self):
        """Render chat interface with settings dialog and system prompt display"""
        st.header("AI Chat")

        if not st.session_state.get("authentication_status"):
            st.error("Please log in to use the chat")
            return

        user_id = st.session_state.get("user_id")
        user_email = st.session_state.get("username")

        if not user_id or not user_email:
            st.error("User ID or email not found in session")
            return

        try:
            # Add settings dialog
            @st.dialog("Chat Settings", width="large")
            def show_settings_dialog():
                """Dialog for chat settings"""
                # Initialize context in session state if not present
                if 'context_text' not in st.session_state:
                    st.session_state.context_text = ""

                tabs = st.tabs(["General Settings", "Context & Instructions"])

                with tabs[0]:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Temperature slider
                        new_temp = st.slider(
                            "Temperature",
                            min_value=0.0,
                            max_value=2.0,
                            value=st.session_state.get('chat_temperature', self.current_config.temperature),
                            step=0.1,
                            help="Higher values make the output more random"
                        )

                        # History toggle
                        new_history = st.toggle(
                            "Include Chat History",
                            value=st.session_state.get('use_history', True),
                            help="Toggle whether to include previous messages in the context"
                        )

                    with col2:
                        # Model selection
                        models = self.repository.get_all_active_models()
                        model_options = {m.model_name: m.model_id for m in models}
                        current_model_name = st.session_state.get('current_model_name', self.current_config.model_name)
                        new_model = st.selectbox(
                            "Select Model",
                            options=list(model_options.keys()),
                            index=list(model_options.keys()).index(current_model_name)
                        )

                with tabs[1]:
                    # System prompt input
                    new_prompt = st.text_area(
                        "System Instructions",
                        value=st.session_state.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE),
                        height=150,
                        help="Define the AI assistant's behavior and role"
                    )

                    # Context input
                    new_context = st.text_area(
                        "Context",
                        value=st.session_state.get('context_text', ""),
                        height=150,
                        help="Provide background information or context that should be considered in responses"
                    )

                    # Example context usage
                    with st.expander("Context Usage Tips"):
                        st.markdown("""
                        - Add reference materials, documentation, or background information
                        - Include relevant data or facts that should inform responses
                        - Context will be included in every message exchange
                        - Keep context concise and relevant to maintain quality responses
                        """)

                # Save button
                if st.button("Save Settings"):
                    # Update all session state variables
                    st.session_state.system_prompt = new_prompt
                    st.session_state.chat_temperature = new_temp
                    st.session_state.use_history = new_history
                    st.session_state.context_text = new_context

                    # Update model if changed
                    if new_model != current_model_name:
                        self.switch_model(model_options[new_model])
                        st.session_state.current_model_name = new_model

                    st.rerun()

            # Sidebar content
            with st.sidebar:
                # Settings button
                if st.button("⚙️ Chat Settings"):
                    show_settings_dialog()

                st.divider()
                st.subheader("Your Chats")

                # Get user's threads
                user_threads = self.get_user_threads(user_email)

                # Create new chat button
                if st.button("New Chat"):
                    st.session_state.current_thread = None
                    st.rerun()

                # Show existing threads
                for thread in user_threads:
                    thread_name = thread.name or f"Chat from {thread.created_at.strftime('%Y-%m-%d %H:%M')}"
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(thread_name, key=thread.id):
                            st.session_state.current_thread = thread.id
                            st.rerun()
                    with col2:
                        stats = self.get_thread_stats(thread.id)
                        if stats:
                            st.caption(f"Messages: {stats['total_messages']}")

            # Main chat interface
            if 'current_thread' not in st.session_state:
                thread_name = f"Chat - {user_email} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                thread = self.get_or_create_thread(thread_name, user_id)
                thread_id = getattr(thread, 'id', None)

                if not thread_id:
                    raise ValueError("Failed to get thread ID")

                st.session_state.current_thread = thread_id

            # Display current settings
            with st.expander("Current Chat Settings", expanded=False):
                st.info(f"🤖 System Instructions:\n{st.session_state.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE)}")
                st.info(f"🌡️ Temperature: {self.current_config.temperature}")
                st.info(f"📚 History Enabled: {st.session_state.get('use_history', True)}")
                st.info(f"📝 Model: {self.current_config.model_name}")

            # Chat display
            if current_thread_id := st.session_state.get('current_thread'):
                messages = self.get_thread_history(current_thread_id)

                # Always show system message first
                st.chat_message("system").write(st.session_state.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE))

                # Display chat history
                for message in messages:
                    role = message.get('input', {}).get('role')
                    if role != 'system':
                        with st.chat_message(role):
                            content = message.get('input', {}).get('content', '')
                            st.write(content)

                # Handle new message input
                if prompt := st.chat_input("Type your message here...", key="chat_input"):
                    with st.chat_message("user"):
                        st.write(prompt)

                    try:
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = []

                            for chunk in self.process_message(current_thread_id, prompt):
                                full_response.append(chunk)
                                message_placeholder.markdown(''.join(full_response) + "▌")

                            final_response = ''.join(full_response)
                            message_placeholder.markdown(final_response)

                    except Exception as e:
                        st.error(f"Error processing message: {str(e)}")

        except Exception as e:
            st.error(f"Error in chat interface: {str(e)}")
            if hasattr(e, '__cause__'):
                st.error(f"Caused by: {str(e.__cause__)}")