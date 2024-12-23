from typing import List, Dict, Optional, Tuple
import pandas as pd
from utils.query_handler import execute_sql
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class DocumentPart:
    document_id: str
    document_name: str
    part_number: int
    token_count: int
    text_content: Optional[str] = None
    part_start_page: Optional[int] = None
    part_end_page: Optional[int] = None
    token_split: Optional[bool] = None


class RawDocumentText:
    """Handles retrieval and management of document text from Snowflake database."""

    def __init__(self):
        self._raw_text_cache: Dict[tuple, List[str]] = {}

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
                    d.uploaded_by
                FROM colby.ai197j.doc_parts_basic t 
                LEFT JOIN colby.ai197j.documents d ON t.document_id = d.id
                WHERE d.uploaded_by = '{user_email}'
                ORDER BY t.document_id, t.part_number ASC
            """

            document_parts_data = execute_sql(query, 'snowflake')

            return pd.DataFrame(document_parts_data)

        except Exception as e:
            logger.error(f"Error retrieving documents for user {user_email}: {str(e)}")
            raise

    def get_raw_text(self, doc_id: str, part_number: int) -> List[str]:

        if not doc_id:
            raise ValueError("Document ID cannot be empty")
        if part_number < 0:
            raise ValueError("Part number must be non-negative")

        cache_key = (doc_id, part_number)

        if cache_key not in self._raw_text_cache:
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
                    FROM colby.ai197j.doc_parts_basic
                    WHERE document_id = {doc_id}
                    AND part_number = {part_number}
                """

                raw_text_data = execute_sql(query,'snowflake')

                if not raw_text_data:
                    raise ValueError(f"No data found for document {doc_id}, part {part_number}")

                raw_text_df = pd.DataFrame(raw_text_data)
                self._raw_text_cache[cache_key] = raw_text_df['TEXT_CONTENT'].tolist()

            except Exception as e:
                logger.error(f"Error retrieving raw text for doc {doc_id}, part {part_number}: {str(e)}")
                raise

        return self._raw_text_cache[cache_key]

    def get_multiple_raw_texts(self, doc_part_pairs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], List[str]]:
        """
        Retrieves raw text content for multiple document-part combinations efficiently.
        Args:doc_part_pairs: List of tuples, each containing (document_id, part_number)
        Returns:Dictionary mapping (doc_id, part_number) to list of text content
        Raises:ValueError: If input list is empty or contains invalid pairs
        """
        if not doc_part_pairs:
            raise ValueError("Must provide at least one document-part pair")

        result = {}
        uncached_pairs = []

        # First check cache
        for doc_id, part_number in doc_part_pairs:
            cache_key = (doc_id, part_number)
            if cache_key in self._raw_text_cache:
                result[cache_key] = self._raw_text_cache[cache_key]
            else:
                uncached_pairs.append((doc_id, part_number))

        # If we have uncached pairs, fetch them all at once
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
                        part_number,
                        part_start_page,
                        part_end_page,
                        tokens_in_part as token_count,
                        token_split
                    FROM colby.ai197j.doc_parts_basic
                    WHERE {conditions}
                """

                raw_text_data = execute_sql(query, 'snowflake')

                if raw_text_data:
                    raw_text_df = pd.DataFrame(raw_text_data)

                    # Group by document_id and part_number
                    grouped = raw_text_df.groupby(['DOCUMENT_ID', 'PART_NUMBER'])

                    for (doc_id, part_number), group in grouped:
                        cache_key = (doc_id, part_number)
                        text_content = group['TEXT_CONTENT'].tolist()
                        self._raw_text_cache[cache_key] = text_content
                        result[cache_key] = text_content

            except Exception as e:
                logger.error(f"Error retrieving multiple raw texts: {str(e)}")
                raise

        return result

    def clear_cache(self):
        """Clears the internal text cache."""
        self._raw_text_cache.clear()