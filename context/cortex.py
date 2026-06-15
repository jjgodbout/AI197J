from typing import List, Dict, Optional, Any, Callable
import json
from utils.query_handler import execute_sql
from context.raw_text import RawDocumentText
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import logging

logger = logging.getLogger('cortex_search')

class CortexSearchHandler:
    def __init__(self):
        import logging
        self.logger = logging.getLogger('cortex_search')
        self.service_name = 'prospectordocuments.external_documents.EXTERNAL_TEXT_CHUNK_SEARCH'
        self.default_columns = [
            "text_chunk",
            "document_id",
            "document_name",
            "chunk_type",
            "page",
            "chunk_number",
            "path",
            "source",
            "user_email"
        ]

    def build_filter(self, document_id: Optional[str] = None, chunk_type: Optional[str] = None) -> Dict[str, Any]:
        """Build the filter dictionary for the Cortex Search query."""
        filters = []

        if document_id:
            filters.append({"@eq": {"DOCUMENT_ID": document_id}}) # Note: Changed to uppercase
        if chunk_type:
            filters.append({"@eq": {"CHUNK_TYPE": chunk_type}}) # Note: Changed to uppercase

        if len(filters) == 0:
            return {}
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"@and": filters}

    def search(self,
               query: str,
               document_id: Optional[str] = None,
               chunk_type: Optional[str] = None,
               limit: int = 10,
               columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Execute a search query with the specified filters."""
        try:
            # Build the search parameters
            search_params = {
                "query": query,
                "columns": columns or self.default_columns,
                "limit": limit
            }

            # Add filters if any are specified
            filters = self.build_filter(document_id, chunk_type)
            if filters:
                search_params["filter"] = filters

            # Convert search params to JSON and escape single quotes
            search_params_json = json.dumps(search_params).replace("'", "''")

            self.logger.debug(f"Search params: {search_params_json}")

            # Construct the SQL query
            query = f"""
            SELECT PARSE_JSON(
                SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                    '{self.service_name}',
                    '{search_params_json}'
                )
            )['results'] as results;
            """

            self.logger.debug(f"Executing Snowflake query: {query}")

            # Execute the query
            results = execute_sql(query, "snowflake")
            self.logger.debug(f"Query results: {results}")

            if results and len(results) > 0:
                try:
                    # Results is returned as a single row with a string that needs to be parsed
                    if isinstance(results[0]['RESULTS'], str):
                        return json.loads(results[0]['RESULTS'])
                    return results[0]['RESULTS']
                except (KeyError, json.JSONDecodeError) as e:
                    self.logger.error(f"Error parsing results: {e}")
                    return []
            return []

        except Exception as e:
            self.logger.error(f"Error executing Cortex Search: {str(e)}")
            self.logger.exception("Full traceback:")
            raise