import pandas as pd
from typing import List, Dict, Any, Optional
from snowflake.snowpark.session import Session
from connectors.snowflake_connector import SnowflakeConnection
from utils.query_handler import execute_sql
import tiktoken
from logging import getLogger

logger = getLogger(__name__)


class ContextualChunkProcessor:
    def __init__(self):
        """Initialize the ContextualChunkProcessor with necessary connections and configurations"""
        self.snowflake = SnowflakeConnection()
        self.session = self.snowflake.get_session()
        self.database = "COLBY"
        self.schema = "AI197J"
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 400
        self._set_database_context()

    def _set_database_context(self):
        """Set the database and schema context for Snowflake operations"""
        try:
            self.session.sql(f"USE DATABASE {self.database}").collect()
            self.session.sql(f"USE SCHEMA {self.schema}").collect()
        except Exception as e:
            logger.error(f"Error setting database context: {str(e)}")
            raise

    def get_document_text(self, document_id: int) -> Optional[str]:
        """Retrieve the full document text from raw_text table"""
        try:
            query = f"""
                SELECT TEXT_CONTENT 
                FROM {self.database}.{self.schema}.raw_text 
                WHERE DOCUMENT_ID = {document_id} 
                ORDER BY PAGE_NUMBER
            """
            result = execute_sql(query, 'snowflake')
            if not result:
                return None

            # Combine all text content
            full_text = ' '.join([row[0] for row in result if row[0]])
            return full_text
        except Exception as e:
            logger.error(f"Error retrieving document text: {str(e)}")
            return None

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))

    def create_chunks(self, text: str, document_id: int) -> List[Dict[str, Any]]:
        """Create chunks from text with proper size and overlap"""
        chunks = []
        tokens = self.encoding.encode(text)
        current_pos = 0
        chunk_number = 1

        while current_pos < len(tokens):
            # Get chunk of appropriate size
            chunk_end = min(current_pos + self.chunk_size, len(tokens))
            chunk_tokens = tokens[current_pos:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Get context for the chunk using Claude (simulated here)
            chunk_with_context = self.add_context(chunk_text, text)

            chunks.append({
                'TEXT_CHUNK': chunk_with_context,
                'DOCUMENT_ID': document_id,
                'CHUNK_NUMBERR': chunk_number,  # Note: Using CHUNK_NUMBERR as per table definition
                'PAGE': 1,  # Default to 1 since we're processing combined text
                'TEXT_TYPE': 'CONTEXTUAL'
            })

            # Move position with some overlap
            current_pos = chunk_end - 100  # 100 token overlap
            chunk_number += 1

        return chunks

    def add_context(self, chunk_text: str, full_document: str) -> str:
        """
        Add context to a chunk using the format specified in the article
        Note: In a production environment, this would call Claude API
        """
        # Simplified context generation - in production, use Claude API
        context = f"This chunk is from document section {chunk_text[:50]}..."
        return f"{context}\n\n{chunk_text}"

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Insert chunks into the TEXT_CHUNKS table"""
        try:
            # Create DataFrame from chunks
            df = pd.DataFrame(chunks)

            # Convert to Snowpark DataFrame and write to table
            snow_df = self.session.create_dataframe(df)
            snow_df.write.mode("append").save_as_table("TEXT_CHUNKS")

            logger.info(f"Successfully inserted {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error inserting chunks: {str(e)}")
            return False

    def process_document(self, document_id: int) -> bool:
        """Process a document to create and store contextual chunks"""
        try:
            # Get full document text
            document_text = self.get_document_text(document_id)
            if not document_text:
                logger.error(f"No text found for document {document_id}")
                return False

            # Create chunks with context
            chunks = self.create_chunks(document_text, document_id)
            if not chunks:
                logger.error(f"No chunks created for document {document_id}")
                return False

            # Insert chunks into database
            success = self.insert_chunks(chunks)
            return success

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return False

    def clear_existing_chunks(self, document_id: int) -> bool:
        """Clear existing chunks for a document before reprocessing"""
        try:
            query = f"""
                DELETE FROM {self.database}.{self.schema}.TEXT_CHUNKS 
                WHERE DOCUMENT_ID = {document_id}
            """
            execute_sql(query, 'snowflake')
            return True
        except Exception as e:
            logger.error(f"Error clearing existing chunks: {str(e)}")
            return False