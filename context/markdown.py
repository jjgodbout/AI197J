import pandas as pd
from llama_parse import LlamaParse
from typing import Union, List, Dict, Any,Dict, List, Optional, Tuple
import tiktoken
import os
import io
import tempfile
import nest_asyncio
from connectors.snowflake_connector import SnowflakeConnection
import requests
import logging
from utils.query_handler import execute_sql

# Initialize logger after imports
logger = logging.getLogger(__name__)

class MarkdownDocumentText:
    """Handles retrieval and management of document text from Snowflake database."""

    def __init__(self):
        self._markdown_text_cache: Dict[tuple, List[str]] = {}

    def get_user_documents(self, user_email: str) -> pd.DataFrame:

        if not user_email or '@' not in user_email:
            raise ValueError("Invalid email address provided")

        try:
            query = f"""
                SELECT DISTINCT
                    t.document_id,
                    t.document_name,
                    t.part_number,
                    t.tokens_in_part as token_count,
                    d.path,
                    d.uploaded_by
                FROM colby.ai197j.doc_parts_markdown t 
                LEFT JOIN colby.ai197j.documents d ON t.document_id = d.id
                WHERE d.uploaded_by = '{user_email}' or d.uploaded_by = 'jgodbout@colby.edu'
                ORDER BY t.document_id, t.part_number ASC
            """

            document_parts_data = execute_sql(query, 'snowflake')

            return pd.DataFrame(document_parts_data)

        except Exception as e:
            logger.error(f"Error retrieving documents for user {user_email}: {str(e)}")
            raise

    def get_markdown_text(self, doc_id: str, part_number: int) -> List[str]:
        if not doc_id:
            raise ValueError("Document ID cannot be empty")
        if part_number < 0:
            raise ValueError("Part number must be non-negative")

        cache_key = (str(doc_id), part_number)

        if cache_key not in self._markdown_text_cache:
            try:
                query = f"""
                    SELECT 
                        text_content,
                        document_id,
                        document_name,
                        part_number,
                        part_start_page,
                        part_end_page,
                        tokens_in_part as token_count,
                        token_split
                    FROM colby.ai197j.doc_parts_markdown
                    WHERE document_id = {doc_id}
                    AND part_number = {part_number}
                """

                markdown_text_data = execute_sql(query, 'snowflake')

                if not markdown_text_data:
                    raise ValueError(f"No data found for document {doc_id}, part {part_number}")

                markdown_text_df = pd.DataFrame(markdown_text_data)
                self._markdown_text_cache[cache_key] = markdown_text_df['TEXT_CONTENT'].tolist()

            except Exception as e:
                logger.error(f"Error retrieving markdown text for doc {doc_id}, part {part_number}: {str(e)}")
                raise

        return self._markdown_text_cache[cache_key]  # Fixed from _raw_text_cache to _markdown_text_cache

    def get_multiple_raw_texts(self, doc_part_pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], List[str]]:
        """Retrieves text content for multiple document parts efficiently."""
        if not doc_part_pairs:
            raise ValueError("Must provide at least one document-part pair")

        result = {}
        uncached_pairs = []

        # Check cache first
        for doc_id, part_number in doc_part_pairs:
            cache_key = (str(doc_id), part_number)
            if cache_key in self._markdown_text_cache:  # Fixed from _raw_text_cache
                result[cache_key] = self._markdown_text_cache[cache_key]  # Fixed from _raw_text_cache
            else:
                uncached_pairs.append((doc_id, part_number))

        # Fetch uncached documents
        if uncached_pairs:
            conditions = " OR ".join([
                f"(document_id = '{doc_id}' AND part_number = {part_number})"
                for doc_id, part_number in uncached_pairs
            ])

            try:
                query = f"""
                    SELECT 
                        text_content,
                        document_id,
                        document_name,
                        part_number
                    FROM colby.ai197j.doc_parts_markdown
                    WHERE {conditions}
                """

                markdown_text_data = execute_sql(query, 'snowflake')

                if markdown_text_data:
                    markdown_text_df = pd.DataFrame(markdown_text_data)
                    grouped = markdown_text_df.groupby(['DOCUMENT_ID', 'PART_NUMBER'])

                    for (doc_id, part_number), group in grouped:
                        cache_key = (str(doc_id), part_number)
                        text_content = group['TEXT_CONTENT'].tolist()
                        self._markdown_text_cache[cache_key] = text_content  # Fixed from _raw_text_cache
                        result[cache_key] = text_content

            except Exception as e:
                logger.error(f"Error retrieving multiple markdown texts: {str(e)}")
                raise

        return result

    def clear_cache(self):
        """Clears the internal text cache."""
        self._markdown_text_cache.clear()
class PDFMarkdownProcessor:
    def __init__(self, snowflake_connection: SnowflakeConnection = None):
        """Initialize PDFMarkdownProcessor with optional snowflake connection"""
        # Apply nest_asyncio at initialization
        nest_asyncio.apply()

        if snowflake_connection is None:
            snowflake_connection = SnowflakeConnection()

        # Setup Snowflake connection
        self.snowflake_connection = snowflake_connection
        self.session = self.snowflake_connection.get_session()
        self.session.sql("USE DATABASE COLBY").collect()
        self.session.sql("USE SCHEMA AI197J").collect()
        self.session.sql("USE WAREHOUSE COLBY_AI197J").collect()
        self.session.sql_simplifier_enabled = True

        # Setup LlamaParse
        llama_api_key = os.getenv('LLAMA_INDEX_API_KEY')
        if not llama_api_key:
            raise ValueError("LLAMA_INDEX_API_KEY not found in environment variables")

        self.parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            verbose=True
        )

        # Setup token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))

    def get_document_info(self, document_id: int) -> Dict[str, Any]:
        """Get document information including stage path"""
        query = f"""
            SELECT ID, NAME, PATH, SOURCE, UPLOADED_BY
            FROM COLBY.AI197J.DOCUMENTS
            WHERE ID = {document_id}
        """

        result = self.session.sql(query).collect()
        if not result:
            raise ValueError(f"No document found with ID {document_id}")

        doc_info = result[0].as_dict()
        print(f"Retrieved document info: {doc_info}")
        return doc_info

    def get_presigned_url(self, stage_path: str) -> str:
        """Get a presigned URL for accessing the staged file"""
        relative_path = stage_path.replace('@source_documents/', '')
        print(f"Getting presigned URL for path: {relative_path}")

        query = f"""
        SELECT GET_PRESIGNED_URL('@source_documents', '{relative_path}', 3600)
        """
        result = self.session.sql(query).collect()
        if not result or not result[0][0]:
            raise ValueError(f"Could not generate presigned URL for {stage_path}")

        url = result[0][0]
        print(f"Generated presigned URL successfully")
        return url

    async def process_pdf_content(self, pdf_bytes: bytes, file_name: str) -> List[Dict]:
        """Process PDF content using LlamaParse"""
        try:
            print(f"Processing PDF content for file: {file_name}")

            # Create a temporary file to store the PDF content
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name
                print(f"Created temporary file at: {temp_path}")

            try:
                # Directly use LlamaParse instead of SimpleDirectoryReader
                print("Starting LlamaParse processing...")
                parsed_content = await self.parser.aload_data(
                    temp_path,
                    extra_info={"file_name": file_name}
                )
                print(f"LlamaParse processing complete. Content type: {type(parsed_content)}")

                if not parsed_content:
                    print("LlamaParse returned no content")
                    return []

                print(f"Successfully parsed PDF content")
                return parsed_content

            except Exception as e:
                print(f"Error during LlamaParse processing: {str(e)}")
                raise

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    print(f"Cleaning up temporary file: {temp_path}")
                    os.unlink(temp_path)
                    print("Temporary file cleaned up")

        except Exception as e:
            print(f"Error in process_pdf_content: {str(e)}")
            raise

    async def process_and_load_markdown(self, document_id: int) -> bool:
        """Process PDF using presigned URL and load markdown into Snowflake"""
        try:
            # Get document information
            doc_info = self.get_document_info(document_id)
            stage_path = doc_info['PATH']
            file_name = doc_info['NAME']

            print(f"Processing document: {file_name} (ID: {document_id})")

            # Get presigned URL for the staged file
            presigned_url = self.get_presigned_url(stage_path)

            # Download PDF content
            print(f"Downloading PDF from presigned URL")
            response = requests.get(presigned_url)
            response.raise_for_status()

            # Process the PDF content
            parsed_documents = await self.process_pdf_content(response.content, file_name)

            if not parsed_documents:
                print(f"No content extracted from PDF for document {document_id}")
                return False

            # Prepare rows for Snowflake
            rows = []
            print(f"Processing {len(parsed_documents)} document sections")

            for idx, doc in enumerate(parsed_documents):
                if hasattr(doc, 'text') and doc.text:
                    text = doc.text
                    token_count = self.count_tokens(text)
                    print(f"Section {idx + 1}: {token_count} tokens")

                    rows.append({
                        'DOCUMENT_ID': document_id,
                        'PAGE_NUMBER': idx + 1,
                        'TEXT_CONTENT': text,
                        'EXTRACTION_METHOD': 'LLAMA_PARSE',
                        'TOKEN_COUNT': token_count
                    })
                else:
                    print(f"Skipping section {idx + 1} - no valid text content")

            if not rows:
                print(f"No valid text content found in parsed PDF for document {document_id}")
                return False

            # Create DataFrame and load to Snowflake
            df = pd.DataFrame(rows)
            df = df.sort_values('PAGE_NUMBER').reset_index(drop=True)
            print(f"Created DataFrame with {len(df)} rows")

            snow_df = self.session.create_dataframe(df)
            snow_df.write.mode("append").save_as_table("COLBY.AI197J.MARKDOWN_TEXT")

            print(f"Successfully processed and loaded document {document_id} with {len(df)} pages")
            return True

        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")
            raise