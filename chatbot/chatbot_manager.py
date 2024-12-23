# file: chat_manager.py
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
from context.raw_text import RawDocumentText
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


executor = ThreadPoolExecutor(max_workers=10)


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

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _get_literal_client(api_key: str) -> LiteralClient:
        """Cached creation of LiteralAI client"""
        return LiteralClient(api_key=api_key)

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False, hash_funcs={LiteralClient: lambda _: None})
    def _get_user_threads(_literal_client, user_email: str) -> List[Dict]:
        """Cached version of getting user threads"""
        try:
            literal_user = _literal_client.api.get_user(identifier=user_email)
            if not literal_user:
                raise ValueError("User not found in LiteralAI")

            threads = _literal_client.api.get_threads(
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
            return threads.data if hasattr(threads, 'data') else []
        except Exception as e:
            print(f"Error getting user threads: {str(e)}")
            return []

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False, hash_funcs={LiteralClient: lambda _: None})
    def _get_thread_history(_literal_client, thread_id: str) -> List[Dict]:
        """Cached version of getting thread history"""
        try:
            thread = _literal_client.api.get_thread(id=thread_id)
            if not thread:
                return []

            messages = []
            steps = getattr(thread, 'steps', [])
            for step in steps:
                input_data = getattr(step, 'input', {})
                if isinstance(input_data, dict) and input_data.get('content', {}).get('content', '').strip():
                    messages.append({
                        'input': input_data.get('content', {})
                    })
            return messages
        except Exception as e:
            print(f"Error retrieving thread history: {str(e)}")
            return []

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _create_llm_model(_config: LLMConfig, _callback_manager: Optional[CallbackManager] = None):
        """Cached creation of LLM model"""
        return LLMFactory.create_model(_config, _callback_manager)

    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables with defaults"""
        defaults = {
            'system_prompt': ChatManager.DEFAULT_SYSTEM_MESSAGE,
            'use_history': True,
            'chat_temperature': 0.7,
            'context_text': "",
            'current_model_name': None,
            'current_thread': None,
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _get_cached_settings(_thread_id: str):
        """Get cached settings for a thread (stubbed)"""
        return {
            'system_prompt': st.session_state.system_prompt,
            'chat_temperature': st.session_state.chat_temperature,
            'use_history': st.session_state.use_history,
            'context_text': st.session_state.context_text,
            'selected_doc_parts': st.session_state.selected_doc_parts,
            'document_context': st.session_state.document_context,
            'current_model_name': st.session_state.current_model_name
        }

    def __init__(self, query_handler):
        """Initialize ChatManager with LangChain integration"""
        ChatManager.initialize_session_state()

        literal_api_key = os.getenv('LITERAL_API_KEY')
        if not literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")

        self.literal_client = ChatManager._get_literal_client(literal_api_key)

        # Setup Langchain callback
        callback = self.literal_client.langchain_callback()
        # Convert the callback to a proper BaseCallbackHandler if needed
        if not isinstance(callback, BaseCallbackHandler):
            from langchain_core.callbacks import LangChainCallback
            callback = LangChainCallback(callback)

        self.callback_manager = CallbackManager([callback])
        self.repository = LLMRepository(query_handler)
        self.current_config = None
        self.llm = None
        self.preloaded_models = {}
        self.initialize_default_model()

    def initialize_default_model(self):
        """Initialize with default model using cached creation"""
        try:
            model_config = self.repository.get_model("gpt-3.5-turbo")
            if not model_config:
                raise ValueError("Default model not found in database")
            self.current_config = model_config
            self.llm = ChatManager._create_llm_model(model_config, self.callback_manager)

            # Preload other commonly used models
            common_models = ["gpt-4"]  # Add other model names as needed
            for model_name in common_models:
                model = self.repository.get_model(model_name)
                if model:
                    self.preloaded_models[model.model_id] = ChatManager._create_llm_model(model, self.callback_manager)
        except Exception as e:
            raise ValueError(f"Failed to initialize default model: {str(e)}")

    def _validate_model_config(self, model_config: LLMConfig) -> None:
        """Validate model configuration"""
        if not model_config:
            raise ValueError("Model configuration is None")
        if not model_config.model_id:
            raise ValueError("Model ID is missing")
        if not model_config.model_name:
            raise ValueError("Model name is missing")
        if not model_config.context_length:
            raise ValueError("Context length is missing")

        print(f"""
            Model Validation:
            - ID: {model_config.model_id}
            - Name: {model_config.model_name}
            - Provider: {model_config.provider}
            - Context Length: {model_config.context_length}
            - Temperature: {model_config.temperature}
            - Currently Loaded Model: {self.current_config.model_name if self.current_config else 'None'}
            """)

    def switch_model(self, model_id: str) -> None:
        """Switch to a different model with enhanced validation"""
        try:
            print(f"\n=== Model Switch Request ===")
            print(f"Requested model ID: {model_id}")
            print(f"Current model: {self.current_config.model_name if self.current_config else 'None'}")

            # Get fresh model config
            model_config = self.repository.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in database")

            # Validate configuration
            self._validate_model_config(model_config)

            # Create new model instance
            print(f"Creating new model instance for: {model_config.model_name}")
            self.llm = self._create_llm_model(model_config, self.callback_manager)
            self.current_config = model_config

            print(f"""
                Model Switch Complete:
                - Previous: {self.current_config.model_name if self.current_config else 'None'}
                - New: {model_config.model_name}
                - Context Length: {model_config.context_length}
                ====================
                """)

        except Exception as e:
            print(f"ERROR switching model: {str(e)}")
            raise ValueError(f"Failed to switch model: {str(e)}")

    def get_or_create_thread(self, name: str, participant_id: str, thread_settings: Optional[Dict] = None) -> Dict:
        """Create a new thread with model validation"""
        print("\n=== Creating New Thread ===")

        if thread_settings and thread_settings.get('model_id'):
            print(f"""
    Thread Creation Settings:
    - Model Name: {thread_settings.get('model_name')}
    - Model ID: {thread_settings.get('model_id')}
    - Context Length: {thread_settings.get('context_length')}
    """)
            # Verify model exists in database
            model_config = self.repository.get_model(thread_settings['model_id'])
            if not model_config:
                raise ValueError(f"Model {thread_settings['model_id']} not found in database")

            # Switch to specified model before thread creation
            print(f"Switching to model {model_config.model_name} for new thread")
            self.switch_model(thread_settings['model_id'])

            # Verify switch was successful
            if self.current_config.model_id != thread_settings['model_id']:
                raise ValueError(f"Failed to switch to requested model {thread_settings['model_name']}")

        print(f"Creating thread with model: {self.current_config.model_name}")
        return asyncio.run(self.get_or_create_thread_async(name, participant_id, thread_settings))

    async def get_or_create_thread_async(self, name: str, participant_id: str, thread_settings: Optional[Dict] = None) -> Dict:
        try:
            user_email = st.session_state.get('username')
            loop = asyncio.get_event_loop()

            # Get or create user
            literal_user = await loop.run_in_executor(
                executor, self.literal_client.api.get_user, user_email
            )

            if not literal_user:
                try:
                    literal_user = await loop.run_in_executor(
                        executor, self.literal_client.api.get_user, participant_id
                    )
                except Exception:
                    user_metadata = {
                        'first_name': st.session_state.get('name', '').split()[0],
                        'last_name': ' '.join(st.session_state.get('name', '').split()[1:])
                    }
                    literal_user = await loop.run_in_executor(
                        executor,
                        self.literal_client.api.create_user,
                        user_email,
                        user_metadata
                    )

            if not literal_user:
                raise ValueError(f"Unable to find or create user for {user_email}")

            # If thread_settings contains model info, switch to that model
            if thread_settings and thread_settings.get('model_id'):
                await loop.run_in_executor(
                    executor,
                    self.switch_model,
                    thread_settings['model_id']
                )

            # Create thread metadata with settings
            thread_metadata = {
                "user_email": user_email,
                "literal_user_id": literal_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "model_id": self.current_config.model_id,
                "model_name": self.current_config.model_name,
                "provider": self.current_config.provider,
                "settings": thread_settings or {
                    "temperature": st.session_state.get('chat_temperature', 0.7),
                    "system_prompt": st.session_state.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE),
                    "use_history": st.session_state.get('use_history', True),
                    "context_text": st.session_state.get('context_text', ""),
                    "model_name": self.current_config.model_name,
                    "model_id": self.current_config.model_id
                }
            }

            # Create the thread
            thread = await loop.run_in_executor(
                executor,
                lambda: self.literal_client.api.create_thread(
                    name=name,
                    participant_id=literal_user.id,
                    metadata=thread_metadata
                )
            )

            if not thread:
                raise ValueError("Failed to create thread")

            # Update session state with thread settings
            self._update_session_state_from_thread(thread)

            # Log system message
            await self._log_step_async(
                thread_id=thread.id,
                step_type=StepType.LLM,
                content={
                    "role": "system",
                    "content": thread_metadata["settings"]["system_prompt"]
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
            st.error(f"Error creating thread: {str(e)}")
            raise

    def _update_session_state_from_thread(self, thread) -> None:
        """Update session state with thread settings"""
        if hasattr(thread, 'metadata') and thread.metadata:
            settings = thread.metadata.get('settings', {})
            st.session_state.system_prompt = settings.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE)
            st.session_state.chat_temperature = settings.get('temperature', 0.7)
            st.session_state.use_history = settings.get('use_history', True)
            st.session_state.context_text = settings.get('context_text', '')
            st.session_state.current_model_name = settings.get('model_name', self.current_config.model_name)

    def get_thread_settings(self, thread_id: str) -> Dict:
        """Get settings for a specific thread"""
        try:
            thread = self.literal_client.api.get_thread(id=thread_id)
            if thread and hasattr(thread, 'metadata'):
                return thread.metadata.get('settings', {})
            return {}
        except Exception as e:
            print(f"Error getting thread settings: {str(e)}")
            return {}

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get message history for a thread (cached)"""
        return ChatManager._get_thread_history(self.literal_client, thread_id)

    def get_user_threads(self, user_email: str) -> List[Dict]:
        """Get all threads for a specific user (cached)"""
        return ChatManager._get_user_threads(self.literal_client, user_email)

    async def _log_step_async(self, thread_id: str, step_type: StepType, content: Dict[str, Any],
                              metadata: Optional[Dict] = None) -> Dict:
        """Asynchronous version of logging step"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, self._log_step, thread_id, step_type, content, metadata
        )

    def _log_step(self, thread_id: str, step_type: StepType, content: Dict[str, Any],
                  metadata: Optional[Dict] = None) -> Dict:
        """Create a step with enhanced logging"""
        step_metadata = self._create_step_metadata(step_type, metadata)
        return self.literal_client.api.create_step(
            thread_id=thread_id,
            type=step_type.value,
            input={
                "content": content,
                "metadata": step_metadata
            }
        )

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

    def process_message(self, thread_id: str, prompt: str):
        """Process user message and return AI response (blocking)"""
        return asyncio.run(self.process_message_async(thread_id, prompt))

    async def process_message_async(self, thread_id: str, prompt: str) -> str:
        """Process message with enhanced model tracking"""
        try:
            print("\n=== Message Processing Debug ===")

            # Get thread settings and log current state
            thread_settings = self.get_thread_settings(thread_id)
            print(f"""
                Thread and Model State:
                - Thread ID: {thread_id}
                - Current Model Config: {self.current_config.model_name if self.current_config else 'None'} (ID: {self.current_config.model_id if self.current_config else 'None'})
                - Thread Settings Model: {thread_settings.get('model_name')} (ID: {thread_settings.get('model_id')})
                """)

            # Ensure we're using the correct model from thread settings
            if thread_settings and thread_settings.get('model_id'):
                if not self.current_config or self.current_config.model_id != thread_settings['model_id']:
                    print(f"Switching model to match thread settings: {thread_settings['model_name']}")
                    # Get fresh model config
                    model_config = self.repository.get_model(thread_settings['model_id'])
                    if not model_config:
                        raise ValueError(f"Model {thread_settings['model_id']} not found in database")

                    # Switch to the correct model
                    await asyncio.get_event_loop().run_in_executor(
                        executor,
                        self.switch_model,
                        thread_settings['model_id']
                    )
                    print(f"Model switched successfully to: {self.current_config.model_name}")

            # Verify current model configuration
            if not self.current_config:
                raise ValueError("No active model configuration")

            print(f"""
                Active Model Configuration:
                - Name: {self.current_config.model_name}
                - ID: {self.current_config.model_id}
                - Context Length: {self.current_config.context_length}
                - Provider: {self.current_config.provider}
                """)


            # Get model's context length limit
            model_config = self.repository.get_model(self.current_config.model_id)
            max_context_length = model_config.context_length if model_config else 4096  # Default fallback

            # Reserve tokens for system message and new prompt
            reserved_tokens = 500  # For system message
            prompt_tokens = len(prompt.split())
            available_tokens = max_context_length - reserved_tokens - prompt_tokens

            user_email = st.session_state.get('username')
            loop = asyncio.get_event_loop()
            literal_user = await loop.run_in_executor(
                executor, self.literal_client.api.get_user, user_email
            )

            # Build message chain
            messages = []

            # System + context
            system_content = thread_settings.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE)
            context_text = thread_settings.get('context_text', '').strip()
            if context_text:
                system_content += f"\n\nContext to consider:\n{context_text}"
            messages.append(SystemMessage(content=system_content))

            # Add chat history if enabled, with token limit consideration
            if thread_settings.get('use_history', True):
                conversation_history = await loop.run_in_executor(
                    executor, self.get_thread_history, thread_id
                )

                # Process messages from newest to oldest
                history_messages = []
                total_tokens = 0

                for msg in reversed(conversation_history[-20:]):  # Limit to last 20 messages
                    role = msg.get('input', {}).get('role')
                    content = msg.get('input', {}).get('content', '')
                    if role not in ('user', 'assistant'):
                        continue

                    msg_tokens = len(content.split())
                    if total_tokens + msg_tokens > available_tokens:
                        break

                    total_tokens += msg_tokens
                    if role == 'user':
                        history_messages.append(HumanMessage(content=content))
                    else:
                        history_messages.append(AIMessage(content=content))

                # Add history messages in correct order
                messages.extend(reversed(history_messages))

            # Add current prompt
            messages.append(HumanMessage(content=prompt))

            # Log user message
            await self._log_step_async(
                thread_id=thread_id,
                step_type=StepType.LLM,
                content={"role": "user", "content": prompt},
                metadata={
                    "user_email": user_email,
                    "literal_user_id": literal_user.id if literal_user else None,
                    "model": self.current_config.model_name,
                    "temperature": self.current_config.temperature
                }
            )

            # Get response
            try:
                response = self.llm.stream(messages)
                response_chunks = []
                for chunk in response:
                    if isinstance(chunk, AIMessage):
                        response_chunks.append(chunk.content)
                full_response = ''.join(response_chunks)

                # Log generation and response
                await self._log_generation_async(thread_id, prompt, full_response)
                await self._log_step_async(
                    thread_id=thread_id,
                    step_type=StepType.LLM,
                    content={"role": "assistant", "content": full_response},
                    metadata={
                        "token_count": len(full_response.split()),
                        "completion_type": "stream",
                        "user_email": user_email,
                        "literal_user_id": literal_user.id if literal_user else None,
                        "model": self.current_config.model_name,
                        "provider": self.current_config.provider
                    }
                )

                return full_response
            except Exception as e:
                # Log specific error for context length issues
                if "context_length_exceeded" in str(e):
                    error_msg = "Message history too long for this model. Try starting a new chat or using a model with larger context window."
                    st.error(error_msg)
                raise

        except Exception as e:
            error_data = {
                "error_type": "MessageProcessingError",
                "error_message": str(e),
                "prompt": prompt
            }
            await self._log_step_async(thread_id, StepType.RUN, error_data, {"is_error": True})
            raise

    def _log_generation(self, thread_id: str, prompt: str, response: str):
        """Log a generation with proper metadata"""
        input_tokens = len(prompt.split())
        output_tokens = len(response.split())
        total_tokens = input_tokens + output_tokens
        generation = ChatGeneration(
            messages=[{"role": "user", "content": prompt}],
            message_completion={"role": "assistant", "content": response},
            model=self.current_config.model_name,
            provider=self.current_config.provider,
            token_count=total_tokens,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            metadata={
                "thread_id": thread_id,
                "temperature": self.current_config.temperature,
                "created_at": datetime.utcnow().isoformat(),
                "user_email": st.session_state.get('username'),
                "use_history": st.session_state.get('use_history', True),
                "has_context": bool(st.session_state.get('context_text', "").strip())
            },
            settings={
                "temperature": self.current_config.temperature
            }
        )
        self.literal_client.api.create_generation(generation)

    async def _log_generation_async(self, thread_id: str, prompt: str, response: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor, self._log_generation, thread_id, prompt, response
        )

    def get_thread_stats(self, thread_id: str) -> Dict:
        """Get some statistics about a thread"""
        try:
            with self.literal_client.thread(thread_id=thread_id) as thread:
                if not thread:
                    return {}
                steps = getattr(thread, 'steps', [])
                thread_metadata = getattr(thread, 'metadata', {})
                created_at = thread_metadata.get('created_at') if thread_metadata else None

                # Get up to 100 generations in descending order
                generations = self.literal_client.api.get_generations(
                    first=100,
                    order_by={"column": "createdAt", "direction": "DESC"}
                )
                gen_data = []
                if hasattr(generations, 'data'):
                    gen_data = [
                        gen for gen in generations.data
                        if hasattr(gen, 'metadata')
                        and gen.metadata
                        and gen.metadata.get('thread_id') == thread_id
                    ]

                # Compute token usage
                total_tokens = sum(gen.token_count or 0 for gen in gen_data if hasattr(gen, 'token_count'))
                input_tokens = sum(gen.input_token_count or 0 for gen in gen_data if hasattr(gen, 'input_token_count'))
                output_tokens = sum(gen.output_token_count or 0 for gen in gen_data if hasattr(gen, 'output_token_count'))

                step_times = [step.created_at for step in steps if hasattr(step, 'created_at')]
                last_message = max(step_times) if step_times else None

                return {
                    "total_messages": len(steps),
                    "total_generations": len(gen_data),
                    "created_at": created_at,
                    "last_message": last_message,
                    "model": thread_metadata.get('model_name'),
                    "temperature": thread_metadata.get('temperature'),
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
        except Exception as e:
            print(f"Error getting thread stats: {str(e)}")
            return {}
