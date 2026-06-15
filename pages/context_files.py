import streamlit as st
import os
from datetime import datetime
import io
import base64
import PyPDF2
import pandas as pd
import tiktoken
import asyncio
import re
from snowflake.snowpark.session import Session
from context.markdown import PDFMarkdownProcessor
from connectors.snowflake_connector import SnowflakeConnection
from utils.query_handler import execute_sql
from logging import getLogger
from context.cortex import CortexSearchHandler
from utils.chunker import TextChunkProcessor
from interfaces.analysis_interface import AnalysisInterface
from interfaces.kg_interface import render_graph_management_tab

logger = getLogger(__name__)


def to_snake_case(filename: str) -> str:
    """Convert filename to snake case format"""
    name, ext = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return f"{name}{ext.lower()}"


class ContextFileManager:
    def __init__(self):
        """Initialize the ContextFileManager (defers Snowflake connection until needed)"""
        self._snowflake = None
        self._session = None
        self.database = "COLBY"
        self.schema = "AI197J"
        self.search_handler = CortexSearchHandler()
        self.analysis_interface = AnalysisInterface(self.get_user_documents)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self._markdown_processor = None

    @property
    def snowflake(self):
        """Lazy-initialize Snowflake connection using the user's PAT."""
        if self._snowflake is None:
            self._snowflake = SnowflakeConnection()
        return self._snowflake

    @property
    def session(self):
        """Lazy-initialize Snowflake session."""
        if self._session is None:
            self._session = self.snowflake.get_session()
            self._set_database_context()
        return self._session

    @property
    def markdown_processor(self):
        """Lazy-initialize markdown processor."""
        if self._markdown_processor is None:
            self._markdown_processor = PDFMarkdownProcessor(self.snowflake)
        return self._markdown_processor

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))

    async def process_text_chunks(self, document_id: int) -> bool:
        """Process text chunks after markdown extraction"""
        try:
            chunk_processor = TextChunkProcessor()
            result = chunk_processor.run()
            if "Success" in result:
                logger.info(f"Successfully processed chunks for document {document_id}")
                return True
            else:
                logger.error(f"Failed to process chunks: {result}")
                return False
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            return False

    def _set_database_context(self):
        """Ensure database and schema are set for the session"""
        try:
            self.session.sql(f"USE DATABASE {self.database}").collect()
            self.session.sql(f"USE SCHEMA {self.schema}").collect()
        except Exception as e:
            st.error(f"Error setting database context: {str(e)}")
            raise

    def get_user_documents(self, user_email: str) -> pd.DataFrame:
        """Get documents for a specific user with presigned URLs"""
        try:
            # First get base document info using custom query to include path
            query = f"""
                SELECT DISTINCT
                    d.id as document_id,
                    d.name as document_name,
                    d.source,
                    d.uploaded_by,
                    d.path,
                    t.token_count
                FROM colby.ai197j.documents d
                LEFT JOIN (
                    SELECT document_id, SUM(token_count) as token_count
                    FROM colby.ai197j.raw_text
                    GROUP BY document_id
                ) t ON d.id = t.document_id
                WHERE d.uploaded_by = '{user_email}' or d.uploaded_by = 'jgodbout@colby.edu'
                ORDER BY d.id DESC
            """

            doc_data = execute_sql(query, 'snowflake')
            if not doc_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(doc_data)

            # Generate presigned URLs for each document
            def get_presigned_url(path):
                try:
                    relative_path = path.replace('@source_documents/', '')
                    query = f"""
                    SELECT GET_PRESIGNED_URL('@source_documents', '{relative_path}', 3600)
                    """
                    result = self.session.sql(query).collect()
                    return result[0][0] if result else None
                except Exception as e:
                    logger.error(f"Error generating presigned URL for {path}: {str(e)}")
                    return None

            # Add presigned URLs to DataFrame
            df['presigned_url'] = df['PATH'].apply(get_presigned_url)

            # Clean up column names and order
            df = df.rename(columns={
                'DOCUMENT_ID': 'id',
                'DOCUMENT_NAME': 'name',
                'SOURCE': 'source',
                'UPLOADED_BY': 'uploaded_by',
                'PATH': 'path',
                'TOKEN_COUNT': 'token_count'
            })

            return df

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            st.error(f"Error retrieving documents: {str(e)}")
            return pd.DataFrame()

    def extract_text_from_pdf(self, file_data: bytes, document_id: int) -> bool:
        """Extract raw text from PDF using PyPDF2"""
        try:
            self._set_database_context()

            pdf_file = io.BytesIO(file_data)
            reader = PyPDF2.PdfReader(pdf_file)

            text_data = []
            for page_num in range(len(reader.pages)):
                text = reader.pages[page_num].extract_text()
                token_count = self.count_tokens(text)
                text_data.append({
                    'DOCUMENT_ID': document_id,
                    'PAGE_NUMBER': page_num + 1,
                    'TEXT_CONTENT': text,
                    'EXTRACTION_METHOD': 'PyPDF2',
                    'TOKEN_COUNT': token_count
                })

            df = pd.DataFrame(text_data)
            snowdf = self.session.create_dataframe(df)
            snowdf.write.mode("append").save_as_table("raw_text")
            return True

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            st.error(f"Error extracting text: {str(e)}")
            st.exception(e)
            return False

    async def extract_markdown_from_pdf(self, document_id: int) -> bool:
        """Extract markdown text from PDF using LlamaParse"""
        try:
            success = await self.markdown_processor.process_and_load_markdown(document_id)
            return success
        except Exception as e:
            logger.error(f"Error extracting markdown text: {str(e)}")
            st.error(f"Error extracting markdown text: {str(e)}")
            st.exception(e)
            return False

    def _ensure_stage_exists(self):
        """Ensure the stage exists for file uploads"""
        try:
            self._set_database_context()
            create_stage_query = """
            CREATE STAGE IF NOT EXISTS source_documents
            DIRECTORY = (ENABLE = TRUE)
            """
            self.session.sql(create_stage_query).collect()
        except Exception as e:
            logger.error(f"Error creating stage: {str(e)}")
            st.error(f"Error creating stage: {str(e)}")
            raise

    def upload_to_stage(self, file_data: bytes, file_name: str) -> str:
        """Upload file to Snowflake stage"""
        try:
            self._set_database_context()
            self._ensure_stage_exists()

            snake_case_filename = to_snake_case(file_name)
            stage_path = f'@source_documents/{snake_case_filename}'

            temp_path = f'/tmp/{snake_case_filename}'
            with open(temp_path, 'wb') as f:
                f.write(file_data)

            put_query = f"PUT 'file://{temp_path}' @source_documents AUTO_COMPRESS=FALSE"
            self.session.sql(put_query).collect()

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return stage_path

        except Exception as e:
            logger.error(f"Error uploading to stage: {str(e)}")
            st.error(f"Error uploading to stage: {str(e)}")
            st.exception(e)
            return None

    def insert_document_metadata(self, name: str, path: str, source: str, uploaded_by: str) -> int:
        """Insert document metadata into Snowflake"""
        try:
            insert_query = f"""
                INSERT INTO colby.ai197j.documents (name, path, source, uploaded_by)
                VALUES ('{name}', '{path}', '{source}', '{uploaded_by}')
            """
            insert_result = execute_sql(insert_query, 'snowflake')

            if insert_result is None or insert_result == 0:
                st.error("Failed to insert document metadata")
                return None

            get_id_query = f"""
                SELECT id FROM colby.ai197j.documents 
                WHERE name = '{name}' AND path = '{path}' 
                ORDER BY id DESC 
                LIMIT 1
            """
            data = execute_sql(get_id_query, 'snowflake')

            if data is None or len(data) == 0:
                st.error("Failed to retrieve document ID")
                return None

            df = pd.DataFrame(data)
            if 'ID' in df.columns:
                return df['ID'].iloc[0]
            elif 'id' in df.columns:
                return df['id'].iloc[0]
            else:
                st.error("Could not find ID column in response")
                return None

        except Exception as e:
            logger.error(f"Error inserting metadata: {str(e)}")
            st.error(f"Error inserting metadata: {str(e)}")
            return None

    def display_documents(self):
        """Display user documents with download links"""
        try:
            self._set_database_context()

            # Get user email from session
            user_email = st.session_state.get("username")
            if not user_email:
                st.error("Please log in to view documents")
                return

            # Get documents with presigned URLs
            docs_df = self.get_user_documents(user_email)

            if not docs_df.empty:
                st.subheader("Your Documents")

                # Create a DataFrame view with specific columns
                view_df = docs_df[['name', 'source', 'token_count', 'presigned_url']].copy()

                # Format token counts
                view_df['token_count'] = view_df['token_count'].fillna(0).astype(int).apply(lambda x: f"{x:,}")

                # Create display version of DataFrame
                for idx, row in view_df.iterrows():
                    with st.expander(f"📄 {row['name']}"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"Source: {row['source']}")
                            st.write(f"Tokens: {row['token_count']}")

                        with col2:
                            if row['presigned_url']:
                                st.link_button("Download PDF", row['presigned_url'], type="primary")
                            else:
                                st.warning("Download unavailable")
            else:
                st.info("No documents found")

        except Exception as e:
            logger.error(f"Error displaying documents: {str(e)}")
            st.error(f"Error displaying documents: {str(e)}")
            st.exception(e)

    def render_interface(self):
        """Render the document upload and management interface"""
        st.header("Context Files")

        tab1, tab2, tab3, tab4 = st.tabs(["Document Manager", "Document Search","Document Analysis","Knowledge Graphs"])

        with tab1:

            if 'form_key' not in st.session_state:
                st.session_state.form_key = 0

            uploaded_file = st.file_uploader(
                "Upload a document",
                type=['pdf'],
                key=f"file_uploader_{st.session_state.form_key}"
            )

            if uploaded_file:
                file_data = uploaded_file.read()
                doc_name = st.text_input(
                    "Document Name",
                    value=uploaded_file.name,
                    key=f"doc_name_{st.session_state.form_key}"
                )
                doc_source = st.text_input(
                    "Document Source",
                    key=f"doc_source_{st.session_state.form_key}"
                )

                if st.button("Upload Document", key=f"upload_btn_{st.session_state.form_key}"):
                    if not doc_source:
                        st.warning("Please provide a document source")
                        return

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        status_text.text("Uploading document to stage...")
                        stage_path = self.upload_to_stage(file_data, uploaded_file.name)
                        progress_bar.progress(20)

                        if stage_path and st.session_state.get('username'):
                            status_text.text("Creating document record...")
                            document_id = self.insert_document_metadata(
                                name=doc_name,
                                path=stage_path,
                                source=doc_source,
                                uploaded_by=st.session_state['username']
                            )
                            progress_bar.progress(40)

                            if document_id:
                                status_text.text("Extracting text from PDF...")
                                raw_text_success = self.extract_text_from_pdf(file_data, document_id)
                                progress_bar.progress(60)

                                if raw_text_success:
                                    status_text.text("Extracting markdown from PDF...")
                                    markdown_success = asyncio.run(self.extract_markdown_from_pdf(document_id))
                                    progress_bar.progress(80)

                                    if markdown_success:
                                        status_text.text("Processing text chunks...")
                                        chunk_success = asyncio.run(self.process_text_chunks(document_id))
                                        progress_bar.progress(90)

                                        if chunk_success:
                                            progress_bar.progress(100)
                                            status_text.text("Processing complete!")
                                            st.success("Document uploaded and processed successfully!")
                                            st.session_state.form_key += 1
                                            st.rerun()
                                        else:
                                            status_text.text("Chunk processing failed")
                                            st.warning("Document uploaded but chunk processing failed")
                                    else:
                                        status_text.text("Markdown extraction failed")
                                        st.warning("Document uploaded but markdown extraction failed")
                                else:
                                    status_text.text("Text extraction failed")
                                    st.error("Document uploaded but text extraction failed")
                            else:
                                progress_bar.progress(40)
                                status_text.text("Document record creation failed")
                                st.error("Failed to create document record")
                        else:
                            if not st.session_state.get('username'):
                                progress_bar.progress(0)
                                status_text.text("Login required")
                                st.error("Please log in to upload documents")
                            else:
                                progress_bar.progress(25)
                                status_text.text("Upload failed")
                                st.error("Upload failed")

                    except Exception as e:
                        logger.error(f"Error in upload process: {str(e)}")
                        st.error(f"An error occurred: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

            # Display existing documents
            st.divider()
            self.display_documents()

        with tab2:

            # Document Search Interface
            st.divider()
            with st.expander("🔍 Search Documents", expanded=True):
                # Get user email from session
                user_email = st.session_state.get("username")
                if not user_email:
                    st.error("Please log in to search documents")
                    return

                # Get documents
                docs_df = self.get_user_documents(user_email)
                if not docs_df.empty:
                    # Create document selection options
                    doc_options = docs_df[['id', 'name']].copy()
                    doc_dict = dict(zip(doc_options['name'], doc_options['id']))

                    # Document selection
                    selected_doc = st.selectbox(
                        "Select a document to search",
                        options=list(doc_dict.keys()),
                        key="doc_selector"
                    )

                    if selected_doc:
                        document_id = str(doc_dict[selected_doc])

                        # Create two columns for better layout
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            # Prompt input
                            user_prompt = st.text_area(
                                "Enter your search prompt",
                                placeholder="What would you like to know about this document?",
                                key="search_prompt",
                                height=100
                            )

                        with col2:
                            # Number of results selector
                            num_results = st.number_input(
                                "Number of results",
                                min_value=1,
                                max_value=10,
                                value=3,
                                key="num_results"
                            )

                        # Search button
                        if st.button("Search", key="search_button", type="primary"):
                            if user_prompt:
                                try:
                                    with st.spinner("Searching document..."):
                                        results = self.search_handler.search(
                                            query=user_prompt,
                                            document_id=document_id,
                                            chunk_type="LLAMA_PARSE",
                                            limit=num_results
                                        )

                                        if results:
                                            st.success("Search completed successfully!")
                                            st.json(results, expanded=True)
                                        else:
                                            st.info("No results found for your query.")
                                except Exception as e:
                                    logger.error(f"Search error: {str(e)}")
                                    st.error(f"Error performing search: {str(e)}")
                            else:
                                st.warning("Please enter a search prompt.")
                else:
                    st.info("No documents available for search.")

        with tab3:
            # Use the new analysis interface
            self.analysis_interface.render()

        with tab4:
            render_graph_management_tab()