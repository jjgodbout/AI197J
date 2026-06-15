from typing import Optional, Dict, List, Any, Generator
from datetime import datetime
import os
import json
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid
import logging
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from llm.repository import LLMRepository
from context.raw_text import RawDocumentText
from context.markdown import MarkdownDocumentText
from context.cortex import CortexSearchHandler
from chatbot.chat_repository import ChatRepository

logger = logging.getLogger('chat_manager')


class ChatManager:
    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant. You aim to be accurate, informative, and engaging while maintaining a natural conversational style."""

    def __init__(self, query_handler):
        """Initialize ChatManager with Snowflake-backed chat storage"""
        self._initialize_session_state()
        self.repository = LLMRepository(query_handler)
        self.chat_repo = ChatRepository(query_handler)
        self.current_config = None
        self.llm = None
        self.preloaded_models = {}
        self._model_initialized = False

    def _initialize_session_state(self):
        """Initialize session state with defaults"""
        defaults = {
            'system_prompt': self.DEFAULT_SYSTEM_MESSAGE,
            'use_history': True,
            'chat_temperature': 0.7,
            'context_text': "",
            'current_model_name': None,
            'current_thread': None,
            'messages': [],
            'previous_thread': None
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def ensure_model_initialized(self):
        """Ensure the default model is initialized (lazy, called on first use)."""
        if self._model_initialized:
            return
        self.initialize_default_model()

    def initialize_default_model(self):
        """Initialize the default model configuration and LLM instance"""
        try:
            available_models = self.repository.get_all_active_models()

            if not available_models:
                raise ValueError("No active models available")

            default_model = available_models[0]
            self.current_config = default_model

            self.llm = self._initialize_llm(
                model_id=default_model.model_id,
                model_name=default_model.model_name,
                provider=default_model.provider,
                temperature=st.session_state.get('chat_temperature', 0.7)
            )

            st.session_state.current_model_name = default_model.model_name
            self._model_initialized = True

        except Exception as e:
            logger.error(f"Error initializing default model: {str(e)}")
            raise

    def _initialize_llm(self, model_id: str, model_name: str, provider: str, temperature: float):
        """Initialize an LLM instance with given configuration"""
        try:
            if self.current_config and abs(self.current_config.temperature - temperature) > 0.001:
                self.preloaded_models.clear()

            cache_key = f"{provider}_{model_name}_{model_id}_{temperature}"
            logger.debug(f"Initializing LLM with cache key: {cache_key}")

            if cache_key in self.preloaded_models:
                return self.preloaded_models[cache_key]

            if provider.lower() == "anthropic":
                llm = ChatAnthropic(
                    model_name=model_name,
                    temperature=temperature,
                )
            elif provider.lower() == "openai":
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            if not hasattr(self.current_config, 'temperature'):
                setattr(self.current_config, 'temperature', temperature)

            self.preloaded_models[cache_key] = llm
            return llm

        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    # ---- Thread operations (Snowflake-backed) ----

    def create_thread(self, name: str, user_email: str,
                      settings: Optional[Dict] = None) -> Dict:
        """Create a new chat thread stored in Snowflake."""
        self.ensure_model_initialized()
        return self.chat_repo.create_thread(
            user_email=user_email,
            name=name,
            model_name=self.current_config.model_name,
            model_id=self.current_config.model_id,
            provider=self.current_config.provider,
            settings=settings,
        )

    def get_thread(self, thread_id: str) -> Optional[Dict]:
        return self.chat_repo.get_thread(thread_id)

    def get_thread_settings(self, thread_id: str) -> Dict:
        thread = self.chat_repo.get_thread(thread_id)
        if thread:
            return thread.get("settings", {})
        return {}

    def get_user_threads(self, user_email: str) -> List[Dict]:
        return self.chat_repo.get_user_threads(user_email)

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get message history for a thread."""
        messages = self.chat_repo.get_thread_messages(thread_id)
        return [
            {
                "role": m["role"],
                "content": m["content"],
                "created_at": m.get("created_at"),
            }
            for m in messages
            if m["role"] in ("user", "assistant")
        ]

    def get_thread_stats(self, thread_id: str) -> Dict:
        return self.chat_repo.get_thread_stats(thread_id)

    def update_thread_settings(self, thread_id: str, new_settings: Dict) -> bool:
        return self.chat_repo.update_thread_settings(thread_id, new_settings)

    def delete_thread(self, thread_id: str) -> bool:
        result = self.chat_repo.delete_thread(thread_id)
        if st.session_state.get('current_thread') == thread_id:
            st.session_state.current_thread = None
            st.session_state.messages = []
        return result

    # ---- Message processing ----

    def process_message(self, thread_id: str, prompt: str) -> Generator:
        """Process a message with context from documents and vector search."""
        try:
            self.ensure_model_initialized()

            thread = self.chat_repo.get_thread(thread_id)
            if not thread:
                raise ValueError(f"Thread {thread_id} not found")

            thread_settings = thread.get("settings", {})

            # Build system message with merged context
            system_parts = [thread_settings.get('system_prompt', self.DEFAULT_SYSTEM_MESSAGE)]

            # Add custom context
            if thread_settings.get('custom_context'):
                system_parts.append(f"\nContext:\n{thread_settings['custom_context'].strip()}")

            # Add vector search results
            vector_context = self._perform_vector_search(prompt, thread_settings)
            if vector_context:
                system_parts.append(vector_context)

            # Process selected documents
            doc_context = self._process_documents(thread_settings)
            if doc_context:
                system_parts.append(doc_context)

            # Build messages array
            system_message = SystemMessage(content="\n\n".join(system_parts))
            messages = [system_message]

            # Add chat history
            if thread_settings.get('use_history', True):
                history = self.chat_repo.get_thread_messages(thread_id)
                context_limit = int(self.current_config.context_length * 0.8)
                current_tokens = len(system_message.content.split())

                for msg in history:
                    if msg['role'] not in ('user', 'assistant'):
                        continue
                    content = msg['content']
                    tokens = len(content.split())
                    if current_tokens + tokens > context_limit:
                        break
                    current_tokens += tokens
                    message_class = HumanMessage if msg['role'] == 'user' else AIMessage
                    messages.append(message_class(content=content))

            messages.append(HumanMessage(content=prompt))

            # Log user message to Snowflake
            self.chat_repo.add_message(
                thread_id=thread_id,
                role="user",
                content=prompt,
                model_name=self.current_config.model_name,
                token_count=len(prompt.split()),
            )

            # Stream response
            full_response = ""
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_text = chunk.content
                    full_response += chunk_text
                    yield chunk_text

            # Log assistant response to Snowflake
            self.chat_repo.add_message(
                thread_id=thread_id,
                role="assistant",
                content=full_response,
                model_name=self.current_config.model_name,
                token_count=len(full_response.split()),
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def _process_documents(self, thread_settings: Dict) -> str:
        """Process traditional document selections (raw and markdown)."""
        selected_docs = thread_settings.get('selected_documents', [])
        if not selected_docs:
            return ""

        try:
            raw_doc = RawDocumentText()
            markdown_doc = MarkdownDocumentText()

            raw_pairs = []
            markdown_pairs = []
            for doc in selected_docs:
                doc_id = str(doc.get('document_id'))
                part_num = int(doc.get('part_number'))
                doc_type = doc.get('type', 'raw')
                if doc_type == 'raw':
                    raw_pairs.append((doc_id, part_num))
                elif doc_type == 'markdown':
                    markdown_pairs.append((doc_id, part_num))

            docs_text = []

            if raw_pairs:
                raw_contents = raw_doc.get_multiple_raw_texts(raw_pairs)
                for doc_id, part_num in raw_pairs:
                    cache_key = (str(doc_id), part_num)
                    if cache_key in raw_contents:
                        content = raw_contents[cache_key]
                        text = ' '.join(str(item) for item in content if item) if isinstance(content, list) else str(content)
                        if text.strip():
                            docs_text.append(f"Raw Document {doc_id} Part {part_num}:\n{text.strip()}")

            if markdown_pairs:
                markdown_contents = markdown_doc.get_multiple_raw_texts(markdown_pairs)
                for doc_id, part_num in markdown_pairs:
                    cache_key = (str(doc_id), part_num)
                    if cache_key in markdown_contents:
                        content = markdown_contents[cache_key]
                        text = ' '.join(str(item) for item in content if item) if isinstance(content, list) else str(content)
                        if text.strip():
                            docs_text.append(f"Markdown Document {doc_id} Part {part_num}:\n{text.strip()}")

            if docs_text:
                return "\nReference Documents:\n" + "\n\n".join(docs_text)
            return ""

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return ""

    def _perform_vector_search(self, prompt: str, thread_settings: Dict) -> str:
        """Perform vector search using CortexSearchHandler."""
        try:
            vector_search = thread_settings.get('vector_search', {})
            if not vector_search.get('enabled'):
                return ""

            documents = vector_search.get('documents', [])
            if not documents:
                return ""

            search_handler = CortexSearchHandler()
            context_parts = []

            for doc in documents:
                doc_id = str(doc['document_id'])
                try:
                    results = search_handler.search(
                        query=prompt,
                        document_id=doc_id,
                        chunk_type=vector_search.get('chunk_type', 'LLAMA_PARSE'),
                        limit=vector_search.get('results_per_doc', 3)
                    )
                except Exception as search_error:
                    logger.error(f"Error searching document {doc_id}: {search_error}")
                    continue

                if results:
                    doc_context = []
                    for result in results:
                        text_chunk = result.get('text_chunk', '')
                        if text_chunk:
                            page = result.get('page', 'Unknown')
                            chunk_num = result.get('chunk_number', 'Unknown')
                            doc_context.append(f"Page {page}, Chunk {chunk_num}:\n{text_chunk.strip()}")
                    if doc_context:
                        context_parts.append(f"\nFrom {doc['name']}:\n" + "\n\n".join(doc_context))

            if context_parts:
                return "\n\n".join(["Relevant context from vector search:", "\n\n".join(context_parts)])
            return ""

        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return ""

    # ---- Model management ----

    def change_model(self, model_name: str) -> bool:
        """Change the active model."""
        try:
            self.ensure_model_initialized()
            available_models = self.repository.get_all_active_models()
            selected_model = next((m for m in available_models if m.model_name == model_name), None)

            if not selected_model:
                raise ValueError(f"Model {model_name} not found")

            self.current_config = selected_model
            self.llm = self._initialize_llm(
                model_id=selected_model.model_id,
                model_name=selected_model.model_name,
                provider=selected_model.provider,
                temperature=st.session_state.get('chat_temperature', 0.7)
            )

            st.session_state.current_model_name = model_name
            return True

        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            return False
