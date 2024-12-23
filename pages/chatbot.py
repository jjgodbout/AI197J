import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from context.raw_text import RawDocumentText
from concurrent.futures import ThreadPoolExecutor

# Import your ChatManager
from chatbot.chatbot_manager import ChatManager

executor = ThreadPoolExecutor(max_workers=4)


def render_chat_interface(manager: ChatManager):
    """Render the full chat interface with initial settings form and thread management."""

    st.header("AI Chat")

    if not st.session_state.get("authentication_status"):
        st.error("Please log in to use the chat")
        return

    user_id = st.session_state.get("user_id")
    user_email = st.session_state.get("username")

    if not user_id or not user_email:
        st.error("User ID or email not found in session")
        return

    # Document handler for user documents
    doc_handler = RawDocumentText()
    try:
        user_docs = executor.submit(doc_handler.get_user_documents, user_email).result()
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        user_docs = pd.DataFrame()

    # ---- SIDEBAR ----
    with st.sidebar:
        # Get all user threads
        user_threads = manager.get_user_threads(user_email)

        # Create thread selection
        thread_options = []
        if user_threads:
            thread_options = [(thread.id, thread.name or f"Chat from {thread.created_at or 'Unknown Date'}")
                              for thread in user_threads]

        col1, col2 = st.columns([4, 1])
        with col1:
            if thread_options:
                selected_thread = st.selectbox(
                    "Select Chat",
                    options=[t[0] for t in thread_options],
                    format_func=lambda x: next((t[1] for t in thread_options if t[0] == x), x),
                    index=None
                )
                if selected_thread:
                    st.session_state.current_thread = selected_thread
                    print(f"\n=== Selected Thread: {selected_thread} ===")

        with col2:
            if st.button("New Chat", type="primary"):
                st.session_state.current_thread = None
                st.session_state.new_chat = True
                print("\n=== Starting New Chat ===")
                st.rerun()

    # ---- MAIN CHAT AREA ----
    # Show settings form for new chat
    if not st.session_state.get('current_thread'):
        with st.form("chat_settings"):
            st.subheader("New Chat Settings")

            # Get model information
            models = manager.repository.get_all_active_models()
            model_options = {m.model_name: m.model_id for m in models}
            model_info = {m.model_name: {
                "id": m.model_id,
                "context_length": m.context_length,
                "provider": m.provider
            } for m in models}

            # Layout form in columns
            col1, col2 = st.columns(2)
            with col1:
                chat_name = st.text_input(
                    "Chat Name",
                    value=f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )

                # Model selection with context length info
                default_model = next(iter(model_options.keys())) if model_options else "gpt-3.5-turbo"
                selected_model = st.selectbox(
                    "Select Model",
                    options=list(model_options.keys()),
                    index=0,
                    format_func=lambda x: f"{x} (Context: {model_info[x]['context_length']:,} tokens)"
                )

                # Show model details
                if selected_model:
                    st.caption(f"""Model Details:
                    - Provider: {model_info[selected_model]['provider']}
                    - Context Length: {model_info[selected_model]['context_length']:,} tokens
                    - Model ID: {model_info[selected_model]['id']}""")

            with col2:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    help="Higher = more creative"
                )
                use_history = st.checkbox(
                    "Include Chat History",
                    value=True
                )

            # System instructions and context
            system_prompt = st.text_area(
                "System Instructions",
                value=manager.DEFAULT_SYSTEM_MESSAGE,
                height=100
            )

            context_text = st.text_area(
                "Additional Context",
                value="",
                height=100,
                help="Add any additional context for the AI"
            )

            # Document selection if available
            if not user_docs.empty:
                st.subheader("Document Context")
                doc_options = []
                for doc_id in user_docs['DOCUMENT_ID'].unique():
                    doc_parts = user_docs[user_docs['DOCUMENT_ID'] == doc_id]
                    doc_name = doc_parts['DOCUMENT_NAME'].iloc[0]
                    for _, row in doc_parts.iterrows():
                        option = {
                            'id': f"{row['DOCUMENT_ID']}_{row['PART_NUMBER']}",
                            'label': f"{doc_name} - Part {row['PART_NUMBER']} ({row['TOKEN_COUNT']} tokens)"
                        }
                        doc_options.append(option)

                selected_docs = st.multiselect(
                    "Select Document Parts to Include",
                    options=[opt['id'] for opt in doc_options],
                    format_func=lambda x: next((opt['label'] for opt in doc_options if opt['id'] == x), x)
                )

            submitted = st.form_submit_button("Start Chat")

            if submitted:
                try:
                    print("\n=== Form Submitted ===")
                    print(f"Selected Model: {selected_model}")
                    print(f"Model ID: {model_options[selected_model]}")

                    # Create thread settings
                    thread_settings = {
                        "temperature": temperature,
                        "system_prompt": system_prompt,
                        "use_history": use_history,
                        "context_text": context_text,
                        "model_name": selected_model,
                        "model_id": model_options[selected_model],
                        "context_length": model_info[selected_model]['context_length']
                    }
                    print(f"Thread Settings: {thread_settings}")

                    # Handle selected documents
                    if not user_docs.empty and selected_docs:
                        combined_context = []
                        total_tokens = 0
                        for selection in selected_docs:
                            doc_id, part_number = selection.split('_')
                            try:
                                text_content = executor.submit(doc_handler.get_raw_text, doc_id,
                                                               int(part_number)).result()
                                combined_context.extend(text_content)
                                # Rough token estimation
                                total_tokens += sum(len(text.split()) for text in text_content)
                            except Exception as e:
                                st.error(f"Error fetching content for {selection}: {str(e)}")

                        if combined_context:
                            doc_context = "\n\n".join(combined_context)
                            thread_settings["context_text"] = context_text + "\n\n" + doc_context
                            print(f"Added document context. Estimated total tokens: {total_tokens}")

                            # Warn about potential context length issues
                            if total_tokens > model_info[selected_model]['context_length'] * 0.75:
                                st.warning(
                                    f"Selected documents may use a large portion of the model's context length. Consider using a model with larger context window or selecting fewer documents.")

                    # Switch to selected model before creating thread
                    print("Switching to selected model...")
                    manager.switch_model(model_options[selected_model])

                    # Create new thread with settings
                    print("Creating new thread...")
                    thread = manager.get_or_create_thread(chat_name, user_id, thread_settings)
                    print(f"Thread created with ID: {thread.id}")
                    st.session_state.current_thread = thread.id
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating thread: {str(e)}")
                    print(f"Error details: {str(e)}")

    # Show chat interface if thread exists
    else:
        current_thread_id = st.session_state.current_thread
        # Get thread settings for display
        thread_settings = manager.get_thread_settings(current_thread_id)

        # Show current chat info in sidebar
        with st.sidebar:
            with st.expander("Current Chat Settings", expanded=False):
                st.info(f"🤖 Model: {thread_settings.get('model_name', 'Unknown')}")
                st.info(f"🌡️ Temperature: {thread_settings.get('temperature', 0.7)}")
                st.info(f"📚 History: {'Enabled' if thread_settings.get('use_history', True) else 'Disabled'}")
                if thread_settings.get('context_text', '').strip():
                    st.info("📄 Has Context: Yes")
                context_length = thread_settings.get('context_length', 0)
                st.info(f"📏 Context Length: {context_length:,} tokens")

        # Display message history
        messages = manager.get_thread_history(current_thread_id)

        # System message
        st.chat_message("system").write(thread_settings.get('system_prompt', manager.DEFAULT_SYSTEM_MESSAGE))

        # Chat messages
        for message in messages:
            role = message.get('input', {}).get('role')
            if role != 'system':
                with st.chat_message(role):
                    content = message.get('input', {}).get('content', '')
                    st.write(content)

        # New message input
        if prompt := st.chat_input("Type your message here...", key="chat_input"):
            with st.chat_message("user"):
                st.write(prompt)
            try:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = manager.process_message(current_thread_id, prompt)
                    message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                print(f"Error details: {str(e)}")