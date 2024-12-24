from typing import List, Optional, Dict, Any
from datetime import datetime
from .types import LLMConfig, Provider


class LLMRepository:
    """Repository for accessing LLM configurations from Snowflake"""

    def __init__(self, query_handler):
        self.table_name = "COLBY.AI197J.LLM_MODELS"
        self.query_handler = query_handler

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        try:
            print(f"Executing query: {query}")
            print(f"With params: {params}")

            result = self.query_handler(query, "snowflake", params)
            # Convert Snowpark Row objects to dictionaries
            if result:
                return [dict(row.asDict()) for row in result]
            return []
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise

    def get_model(self, model_id: str) -> Optional[LLMConfig]:
        """Get model configuration by ID"""
        query = f"""
        SELECT *
        FROM {self.table_name}
        WHERE MODEL_ID = '{model_id}'
        AND IS_ACTIVE = TRUE
        """

        result = self._execute_query(query)
        return LLMConfig.from_snowflake(result[0]) if result else None

    def get_models_by_provider(self, provider: Provider) -> List[LLMConfig]:
        """Get all active models for a provider"""
        query = f"""
        SELECT *
        FROM {self.table_name}
        WHERE PROVIDER = '{provider.value}'
        AND IS_ACTIVE = TRUE
        ORDER BY MODEL_NAME
        """

        results = self._execute_query(query)
        return [LLMConfig.from_snowflake(row) for row in results]

    def get_models_by_group(self, group: str) -> List[LLMConfig]:
        """Get all active models in a group"""
        query = f"""
        SELECT *
        FROM {self.table_name}
        WHERE MODEL_GROUP = '{group}'
        AND IS_ACTIVE = TRUE
        ORDER BY MODEL_NAME
        """

        results = self._execute_query(query)
        return [LLMConfig.from_snowflake(row) for row in results]

    def get_all_active_models(self) -> List[LLMConfig]:
        """Get all active models"""
        query = f"""
        SELECT *
        FROM {self.table_name}
        WHERE IS_ACTIVE = TRUE
        ORDER BY PROVIDER, MODEL_GROUP, MODEL_NAME
        """

        results = self._execute_query(query)
        return [LLMConfig.from_snowflake(row) for row in results]

    def get_model_groups(self) -> List[str]:
        """Get all available model groups"""
        query = f"""
        SELECT DISTINCT MODEL_GROUP
        FROM {self.table_name}
        WHERE IS_ACTIVE = TRUE
        ORDER BY MODEL_GROUP
        """

        results = self._execute_query(query)
        return [row['MODEL_GROUP'] for row in results if row['MODEL_GROUP']]

    def search_models(self, search_term: str,
                      provider: Optional[Provider] = None,
                      group: Optional[str] = None) -> List[LLMConfig]:
        """Search for models with optional filters"""
        conditions = ["IS_ACTIVE = TRUE"]

        if provider:
            conditions.append(f"PROVIDER = '{provider.value}'")

        if group:
            conditions.append(f"MODEL_GROUP = '{group}'")

        search_pattern = f"%{search_term}%"
        conditions.append(f"""
            (LOWER(MODEL_NAME) LIKE LOWER('{search_pattern}')
            OR LOWER(DESCRIPTION) LIKE LOWER('{search_pattern}')
            OR LOWER(MODEL_TYPE) LIKE LOWER('{search_pattern}'))
        """)

        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT *
        FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY PROVIDER, MODEL_NAME
        """

        results = self._execute_query(query)
        return [LLMConfig.from_snowflake(row) for row in results]

    def get_provider_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about models by provider"""
        query = f"""
        SELECT 
            PROVIDER,
            COUNT(*) as MODEL_COUNT,
            AVG(CONTEXT_LENGTH) as AVG_CONTEXT_LENGTH,
            MIN(CREATED_AT) as FIRST_ADDED,
            MAX(UPDATED_AT) as LAST_UPDATED
        FROM {self.table_name}
        WHERE IS_ACTIVE = TRUE
        GROUP BY PROVIDER
        ORDER BY MODEL_COUNT DESC
        """

        return self._execute_query(query)