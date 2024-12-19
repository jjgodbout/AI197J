import streamlit as st
from literalai import LiteralClient
import os
from datetime import datetime
import io
import base64
import PyPDF2
import pandas as pd
from snowflake.snowpark.session import Session
from connectors.snowflake_connector import SnowflakeConnection
from utils.query_handler import execute_sql

class ContextFileManager:
    def __init__(self):
        self.literal_api_key = os.getenv('LITERAL_API_KEY')
        if not self.literal_api_key:
            raise ValueError("LITERAL_API_KEY environment variable is not set")

        self.client = LiteralClient(api_key=self.literal_api_key)
        self.snowflake = SnowflakeConnection()
        self.session = self.snowflake.get_session()
        self.database = "COLBY"
        self.schema = "AI197J"

        # Set the database and schema immediately after session creation
        self._set_database_context()

    def _set_database_context(self):
        """Ensure database and schema are set for the session"""
        try:
            self.session.sql(f"USE DATABASE {self.database}").collect()
            self.session.sql(f"USE SCHEMA {self.schema}").collect()
        except Exception as e:
            st.error(f"Error setting database context: {str(e)}")
            raise

    def extract_text_from_pdf(self, file_data: bytes, document_id: int):
        try:
            # Ensure database context is set before operations
            self._set_database_context()

            pdf_file = io.BytesIO(file_data)
            reader = PyPDF2.PdfReader(pdf_file)

            text_data = []
            for page_num in range(len(reader.pages)):
                text = reader.pages[page_num].extract_text()
                text_data.append({
                    'DOCUMENT_ID': document_id,
                    'PAGE_NUMBER': page_num + 1,
                    'TEXT_CONTENT': text,
                    'EXTRACTION_METHOD': 'PyPDF2'
                })

            df = pd.DataFrame(text_data)
            snowdf = self.session.create_dataframe(df)
            snowdf.write.mode("append").save_as_table("raw_text")  # No need for full qualification
            return True
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            st.exception(e)
            return False

    def _ensure_stage_exists(self):
        """Ensure the stage exists for file uploads"""
        try:
            self._set_database_context()
            create_stage_query = f"""
            CREATE STAGE IF NOT EXISTS source_documents
            DIRECTORY = (ENABLE = TRUE)
            """
            self.session.sql(create_stage_query).collect()
        except Exception as e:
            st.error(f"Error creating stage: {str(e)}")
            raise

    def upload_to_stage(self, file_data: bytes, file_name: str):
        try:
            self._set_database_context()
            self._ensure_stage_exists()
            stage_path = f'@source_documents/{file_name}'  # Simplified stage path

            temp_path = f'/tmp/{file_name}'
            with open(temp_path, 'wb') as f:
                f.write(file_data)

            put_query = f"PUT 'file://{temp_path}' @source_documents AUTO_COMPRESS=FALSE"
            result = self.session.sql(put_query).collect()

            if os.path.exists(temp_path):
                os.remove(temp_path)

            return stage_path
        except Exception as e:
            st.error(f"Error uploading to stage: {str(e)}")
            st.exception(e)
            return None

    def insert_document_metadata(self, name: str, path: str, source: str, uploaded_by: str):
        try:
            # Insert the document using parameterized query with proper VALUES syntax
            insert_query = f"""
                INSERT INTO colby.ai197j.documents (name, path, source, uploaded_by)
                VALUES ('{name}', '{path}', '{source}', '{uploaded_by}')
            """
            insert_result = execute_sql(insert_query, 'snowflake')

            # Check if insert was successful
            if insert_result is None or insert_result == 0:
                st.error("Failed to insert document metadata")
                return None

            # Get the ID of the inserted document
            get_id_query = f"""
                SELECT id FROM colby.ai197j.documents 
                WHERE name = '{name}' AND path = '{path}' 
                ORDER BY id DESC 
                LIMIT 1
            """
            data = execute_sql(get_id_query, 'snowflake')

            # Check if we got data back
            if data is None or len(data) == 0:
                st.error("Failed to retrieve document ID")
                return None

            # Convert to DataFrame and handle potential column access
            df = pd.DataFrame(data)
            if 'ID' not in df.columns and 'id' in df.columns:
                return df['id'].iloc[0]  # Use lowercase if uppercase not found
            elif 'ID' in df.columns:
                return df['ID'].iloc[0]  # Use uppercase if found
            else:
                st.error("Could not find ID column in response")
                return None

        except Exception as e:
            st.error(f"Error inserting metadata: {str(e)}")
            return None

    def display_documents(self):
        try:
            self._set_database_context()

            query = """
                SELECT name, source, uploaded_by, path 
                FROM documents 
                ORDER BY id DESC
            """
            result = self.session.sql(query).collect()

            if result:
                st.subheader("Uploaded Documents")
                for doc in result:
                    with st.expander(f"{doc['NAME']}"):
                        st.write(f"Source: {doc['SOURCE']}")
                        st.write(f"Uploaded by: {doc['UPLOADED_BY']}")
                        st.write(f"Path: {doc['PATH']}")
            else:
                st.info("No documents uploaded yet.")

        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")


    def render_interface(self):
        st.header("Context Files")

        # Initialize form_key in session state if not present
        if 'form_key' not in st.session_state:
            st.session_state.form_key = 0

        # File uploader with unique key
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

                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Upload to stage (25%)
                    status_text.text("Uploading document to stage...")
                    stage_path = self.upload_to_stage(file_data, uploaded_file.name)
                    progress_bar.progress(25)

                    if stage_path and st.session_state.get('username'):
                        # Step 2: Insert metadata (50%)
                        status_text.text("Creating document record...")
                        document_id = self.insert_document_metadata(
                            name=doc_name,
                            path=stage_path,
                            source=doc_source,
                            uploaded_by=st.session_state['username']
                        )
                        progress_bar.progress(50)

                        if document_id:
                            # Step 3: Extract text (75%)
                            status_text.text("Extracting text from PDF...")
                            if self.extract_text_from_pdf(file_data, document_id):
                                # Step 4: Complete (100%)
                                progress_bar.progress(100)
                                status_text.text("Processing complete!")
                                st.success("Document uploaded and text extracted successfully!")

                                # Clear the form by incrementing the key
                                st.session_state.form_key += 1
                                st.rerun()
                            else:
                                progress_bar.progress(75)
                                status_text.text("Text extraction failed")
                                st.error("Document uploaded but text extraction failed")
                        else:
                            progress_bar.progress(50)
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
                    st.error(f"An error occurred: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

        # Display existing documents
        self.display_documents()