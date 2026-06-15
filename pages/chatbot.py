import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from chatbot.chatbot_manager import ChatManager
from context.raw_text import RawDocumentText
from context.markdown import MarkdownDocumentText
from pages.context_files import ContextFileManager

import logging

logger = logging.getLogger('chatbot_app')


def render_chat_interface(manager: ChatManager):
    """
    Renders the AI Chat interface with Snowflake-backed history and streaming.
    """
    # Initialize key state variables if not present
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.previous_thread = None
        st.session_state.current_thread = None

    if 'show_new_chat_form' not in st.session_state:
        st.session_state.show_new_chat_form = False

    if 'thread_list' not in st.session_state:
        st.session_state.thread_list = []

    if 'refresh_needed' not in st.session_state:
        st.session_state.refresh_needed = True

    st.header("AI Chat")

    # Authentication check
    if not st.session_state.get("authentication_status"):
        st.error("Please log in to use the chat")
        return

    user_id = st.session_state.get("user_id")
    user_email = st.session_state.get("username")

    if not user_id or not user_email:
        st.error("User ID or email not found in session")
        return

    # Create main page layout
    main_col = st.container()

    # Sidebar: Thread Management
    with st.sidebar:
        st.title("Chat Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("New Chat", type="primary", key="new_chat_button"):
                st.session_state.show_new_chat_form = True
                st.session_state.current_thread = None
                st.session_state.messages = []
                st.session_state.previous_thread = None
                st.session_state.initialized = False
                st.rerun()

        with col2:
            if st.button("Refresh", key="refresh_threads"):
                st.session_state.pop('thread_list', None)
                st.session_state.refresh_needed = True
                st.rerun()

        st.divider()

        # Fetch threads from Snowflake
        if st.session_state.refresh_needed or not st.session_state.thread_list:
            try:
                threads = manager.get_user_threads(user_email)
                thread_options = []
                for thread in threads:
                    thread_id = thread.get("id")
                    name = thread.get("name", "")
                    created_at = thread.get("created_at")
                    if thread_id:
                        display_name = name or f"Chat {created_at}"
                        thread_options.append((thread_id, display_name))

                st.session_state.thread_list = thread_options
                st.session_state.refresh_needed = False
            except Exception as e:
                logger.error(f"Error fetching threads: {str(e)}")
                st.error("Failed to load chat history")
                st.session_state.thread_list = []
                st.session_state.refresh_needed = False

        # Thread selection dropdown
        if st.session_state.thread_list:
            selected_thread = st.selectbox(
                "Select Conversation",
                options=[t[0] for t in st.session_state.thread_list],
                format_func=lambda x: next(
                    (t[1] for t in st.session_state.thread_list if t[0] == x), x
                ),
                index=0,
                key="thread_selector"
            )
            if selected_thread and not st.session_state.show_new_chat_form:
                st.session_state.current_thread = selected_thread
                if not st.session_state.initialized:
                    st.session_state.initialized = True
                    st.rerun()
        else:
            st.info("No conversations yet")

    # Main Chat Area
    with main_col:
        if st.session_state.get("show_new_chat_form", False):
            _render_new_chat_form(manager, user_id, user_email)
        else:
            _render_existing_chat(manager)


def _render_new_chat_form(manager: ChatManager, user_id: str, user_email: str):
    """Render the new chat creation form."""
    with st.form("new_chat_form"):
        st.subheader("Start New Chat")

        # Get available models
        manager.ensure_model_initialized()
        models = manager.repository.get_all_active_models()
        model_options = {m.model_name: m.model_id for m in models}
        model_info = {
            m.model_name: {
                "id": m.model_id,
                "context_length": m.context_length,
                "provider": m.provider
            }
            for m in models
        }

        # Basic settings
        col1, col2 = st.columns(2)
        with col1:
            chat_name = st.text_input(
                "Chat Name",
                value=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            selected_model = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                format_func=lambda x: f"{x} ({model_info[x]['provider']})"
            )

        with col2:
            temperature = st.slider(
                "Temperature", min_value=0.0, max_value=2.0,
                value=0.7, step=0.1,
                help="Higher values make output more creative but less precise"
            )
            use_history = st.checkbox(
                "Use Chat History", value=True,
                help="Include previous messages as context"
            )

        system_prompt = st.text_area(
            "System Instructions",
            value=manager.DEFAULT_SYSTEM_MESSAGE,
            height=100,
            help="Instructions that guide the AI's behavior"
        )

        custom_context = st.text_area(
            "Custom Context", value="", height=100,
            help="Add any specific context for this conversation"
        )

        # Document selection
        file_manager = ContextFileManager()
        raw_doc = RawDocumentText()
        markdown_doc = MarkdownDocumentText()

        st.divider()
        doc_tabs = st.tabs(["Raw/Markdown Documents", "Vector Search Documents"])

        selected_doc_pairs = []
        selected_vector_docs = []
        use_vector_search = False
        results_per_doc = 3
        chunk_type = "LLAMA_PARSE"

        with doc_tabs[0]:
            st.subheader("Traditional Document Selection")
            try:
                if user_email:
                    raw_docs_df = raw_doc.get_user_documents(user_email)
                    markdown_docs_df = markdown_doc.get_user_documents(user_email)

                    if not raw_docs_df.empty or not markdown_docs_df.empty:
                        doc_tab1, doc_tab2 = st.tabs(["Raw Text", "Markdown"])

                        with doc_tab1:
                            if not raw_docs_df.empty:
                                raw_doc_options = raw_docs_df.apply(
                                    lambda x: f"{x['DOCUMENT_NAME']} (Part {x['PART_NUMBER']} - Raw)",
                                    axis=1
                                ).tolist()
                                selected_raw_docs = st.multiselect(
                                    "Select Raw Text Documents", options=raw_doc_options
                                )
                                for selection in selected_raw_docs:
                                    doc_info = raw_docs_df[
                                        raw_docs_df.apply(
                                            lambda x: f"{x['DOCUMENT_NAME']} (Part {x['PART_NUMBER']} - Raw)",
                                            axis=1
                                        ) == selection
                                    ].iloc[0]
                                    selected_doc_pairs.append({
                                        "document_id": int(doc_info['DOCUMENT_ID']),
                                        "part_number": int(doc_info['PART_NUMBER']),
                                        "type": "raw"
                                    })
                            else:
                                st.info("No raw text documents available")

                        with doc_tab2:
                            if not markdown_docs_df.empty:
                                markdown_doc_options = markdown_docs_df.apply(
                                    lambda x: f"{x['DOCUMENT_NAME']} (Part {x['PART_NUMBER']} - Markdown)",
                                    axis=1
                                ).tolist()
                                selected_markdown_docs = st.multiselect(
                                    "Select Markdown Documents", options=markdown_doc_options
                                )
                                for selection in selected_markdown_docs:
                                    doc_info = markdown_docs_df[
                                        markdown_docs_df.apply(
                                            lambda x: f"{x['DOCUMENT_NAME']} (Part {x['PART_NUMBER']} - Markdown)",
                                            axis=1
                                        ) == selection
                                    ].iloc[0]
                                    selected_doc_pairs.append({
                                        "document_id": int(doc_info['DOCUMENT_ID']),
                                        "part_number": int(doc_info['PART_NUMBER']),
                                        "type": "markdown"
                                    })
                            else:
                                st.info("No markdown documents available")
                    else:
                        st.info("No documents found for your account")
            except Exception as e:
                logger.error(f"Error loading traditional documents: {str(e)}")
                st.error("Failed to load traditional documents")

        with doc_tabs[1]:
            st.subheader("Vector Search Documents")
            try:
                if user_email:
                    docs_df = file_manager.get_user_documents(user_email)
                    if not docs_df.empty:
                        doc_options = docs_df[['id', 'name', 'source', 'token_count']].copy()
                        selected_docs = st.multiselect(
                            "Select documents for vector search",
                            options=doc_options['name'].tolist(),
                            help="Choose documents to search through during chat"
                        )

                        if selected_docs:
                            for doc_name in selected_docs:
                                doc_info = docs_df[docs_df['name'] == doc_name].iloc[0]
                                selected_vector_docs.append({
                                    "document_id": int(doc_info['id']),
                                    "name": doc_name,
                                    "source": doc_info['source']
                                })
                                st.info(
                                    f"📄 {doc_name}\n\n"
                                    f"Source: {doc_info['source']}\n"
                                    f"Tokens: {int(doc_info.get('token_count', 0)):,}"
                                )

                        use_vector_search = st.checkbox(
                            "Enable Vector Search",
                            value=bool(selected_docs),
                            help="Use semantic search to find relevant information"
                        )

                        if use_vector_search and selected_docs:
                            vc1, vc2 = st.columns(2)
                            with vc1:
                                results_per_doc = st.number_input(
                                    "Results per document",
                                    min_value=1, max_value=200, value=100
                                )
                            with vc2:
                                chunk_type = st.selectbox(
                                    "Content type",
                                    options=["LLAMA_PARSE", "RAW_TEXT"],
                                    format_func=lambda x: "Markdown" if x == "LLAMA_PARSE" else "Raw Text"
                                )
                    else:
                        st.info("No documents available for vector search")
            except Exception as e:
                logger.error(f"Error loading vector search documents: {str(e)}")
                st.error("Failed to load vector search documents")

        # Submit button
        submitted = st.form_submit_button("Start Chat", type="primary")

        if submitted:
            try:
                model_changed = manager.change_model(selected_model)
                if not model_changed:
                    raise ValueError(f"Failed to change model to {selected_model}")

                thread_settings = {
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "use_history": use_history,
                    "model_name": selected_model,
                    "model_id": model_options[selected_model],
                    "context_length": model_info[selected_model]['context_length'],
                    "custom_context": custom_context,
                    "selected_documents": selected_doc_pairs,
                    "vector_search": {
                        "enabled": use_vector_search,
                        "documents": selected_vector_docs,
                        "results_per_doc": results_per_doc,
                        "chunk_type": chunk_type,
                    }
                }

                thread = manager.create_thread(
                    name=chat_name,
                    user_email=st.session_state.get("username"),
                    settings=thread_settings,
                )

                st.session_state.show_new_chat_form = False
                st.session_state.current_thread = thread["id"]
                st.session_state.messages = []
                st.session_state.refresh_needed = True
                st.session_state.pop('thread_list', None)
                st.rerun()

            except Exception as e:
                logger.error(f"Error creating chat: {str(e)}")
                st.error(f"Error creating chat: {str(e)}")


def _render_existing_chat(manager: ChatManager):
    """Render the existing chat conversation view."""
    current_thread_id = st.session_state.current_thread

    # Load history when thread changes
    if current_thread_id and current_thread_id != st.session_state.previous_thread:
        thread_history = manager.get_thread_history(current_thread_id)
        st.session_state.messages = [
            {'role': msg['role'], 'content': msg['content'], 'created_at': msg.get('created_at')}
            for msg in thread_history
            if msg['role'] in ('user', 'assistant')
        ]
        st.session_state.previous_thread = current_thread_id

    if not current_thread_id:
        st.info("Select a conversation or start a new chat.")
        return

    # Chat settings expander
    with st.expander("Chat Settings"):
        thread_settings = manager.get_thread_settings(current_thread_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Model Settings")
            st.info(f"Model: {thread_settings.get('model_name', 'Unknown')}")
            st.info(f"Temperature: {thread_settings.get('temperature', 0.7)}")

        with col2:
            st.markdown("##### Context Settings")
            st.info(f"Context Length: {thread_settings.get('context_length', 0):,} tokens")
            st.info(f"History: {'Enabled' if thread_settings.get('use_history', True) else 'Disabled'}")

        st.markdown("##### System Instructions")
        st.text_area(
            "System Prompt",
            value=thread_settings.get('system_prompt', ''),
            disabled=True, height=100
        )

        if thread_settings.get('custom_context'):
            st.markdown("##### Additional Context")
            st.text_area(
                "Custom Context",
                value=thread_settings['custom_context'],
                disabled=True, height=100
            )

        col1, col2 = st.columns(2)
        with col1:
            selected_docs = thread_settings.get('selected_documents', [])
            if selected_docs:
                st.markdown("##### Traditional Documents")
                for doc in selected_docs:
                    st.info(
                        f"Document ID: {doc['document_id']}\n"
                        f"Part: {doc['part_number']}\n"
                        f"Type: {doc['type']}"
                    )
        with col2:
            vector_search = thread_settings.get('vector_search', {})
            if vector_search.get('enabled'):
                st.markdown("##### Vector Search Documents")
                for doc in vector_search.get('documents', []):
                    st.info(f"📄 {doc['name']}\nSource: {doc['source']}")
                st.success(
                    f"Vector Search Enabled\n"
                    f"Results per doc: {vector_search.get('results_per_doc', 3)}\n"
                    f"Content type: {vector_search.get('chunk_type', 'LLAMA_PARSE')}"
                )

    # Chat messages display
    chat_container = st.container()
    with chat_container:
        messages_container = st.container(height=500, border=True)

        with messages_container:
            col1, = st.columns(1)
            with col1:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Type your message here", key="chat_input"):
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    try:
                        for chunk in manager.process_message(current_thread_id, prompt):
                            if chunk:
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        st.error(f"Error generating response: {str(e)}")

            # Refresh history from Snowflake
            thread_history = manager.get_thread_history(current_thread_id)
            st.session_state.messages = [
                {'role': msg['role'], 'content': msg['content'], 'created_at': msg.get('created_at')}
                for msg in thread_history
                if msg['role'] in ('user', 'assistant')
            ]
