from typing import List, Optional, Dict, Any
import pandas as pd
import json
from utils.query_handler import execute_sql


class TextAnalyzer:
    def __init__(self, page_group_size: int = 10):
        """Initialize TextAnalyzer with specified page group size"""
        self.page_group_size = page_group_size
        self.base_query = """
            WITH 
            document_info AS (
                SELECT ID, NAME, PATH, SOURCE
                FROM COLBY.AI197J.DOCUMENTS
                WHERE ID = '{doc_id}'
            ),
            page_groups AS (
                SELECT 
                    *,
                    FLOOR(page_number / {page_group_size}) * {page_group_size} as group_start,
                    FLOOR(page_number / {page_group_size}) * {page_group_size} + {page_group_size} - 1 as group_end
                FROM colby.ai197j.doc_parts_basic 
                WHERE document_id = '{doc_id}'
            ),
            analysis_result AS (
                SELECT
                    {analysis_function} as result_text,
                    MIN(t.page_number) as start_page,
                    MAX(t.page_number) as end_page,
                    t.group_start,
                    t.group_end,
                    d.ID as document_id,
                    d.NAME as document_name,
                    d.PATH as document_path,
                    d.SOURCE as document_source,
                    COUNT(*) as pages_in_group,
                    ARRAY_TO_STRING(ARRAY_AGG(t.text_content || ' page:' || t.page_number), ' ') as raw_text
                FROM page_groups t
                CROSS JOIN document_info d
                GROUP BY 
                    t.group_start,
                    t.group_end,
                    d.ID,
                    d.NAME,
                    d.PATH,
                    d.SOURCE
            )
            SELECT 
                result_text as analysis_result,
                raw_text,
                start_page,
                end_page,
                group_start,
                group_end,
                document_id,
                document_name,
                document_path,
                document_source,
                pages_in_group
            FROM analysis_result
            ORDER BY group_start ASC
        """

    def _clean_value(self, value: Any, debug: bool = False) -> Any:
        """Clean and validate value for JSON serialization"""
        if debug:
            print(f"Cleaning value: {type(value)} - {value}")

        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                if value.strip().startswith('{') or value.strip().startswith('['):
                    return json.loads(value)
                return value
            except:
                return value
        return str(value)

    def _safe_get_value(self, row: pd.Series, column: str, default: Any = None, debug: bool = False) -> Any:
        """Safely get value from DataFrame row with JSON handling"""
        try:
            value = row.get(column, default)
            cleaned = self._clean_value(value, debug)
            if debug:
                print(f"Column {column}: {type(value)} -> {type(cleaned)}")
            return cleaned
        except Exception as e:
            if debug:
                print(f"Error getting {column}: {str(e)}")
            return default

    def _dataframe_to_json(self, df: pd.DataFrame, analysis_type: str, debug: bool = False) -> Dict:
        """Convert DataFrame to structured JSON format with error handling"""
        if debug:
            print(f"\nDataFrame columns: {df.columns.tolist()}")
            if not df.empty:
                print(f"First row: {df.iloc[0].to_dict()}")

        if df.empty:
            return {
                'document': None,
                'analysis': {
                    'type': analysis_type,
                    'parts': []
                },
                'metadata': {
                    'row_count': 0,
                    'status': 'no_results',
                    'page_group_size': self.page_group_size
                }
            }

        try:
            # Get document info from first row
            first_row = df.iloc[0]
            document = {
                'id': self._safe_get_value(first_row, 'DOCUMENT_ID', debug=debug),
                'name': self._safe_get_value(first_row, 'DOCUMENT_NAME', debug=debug),
                'path': self._safe_get_value(first_row, 'DOCUMENT_PATH', debug=debug),
                'source': self._safe_get_value(first_row, 'DOCUMENT_SOURCE', debug=debug)
            }

            # Process analysis parts
            parts = []
            for _, row in df.iterrows():
                part = {
                    'start_page': self._safe_get_value(row, 'START_PAGE', debug=debug),
                    'end_page': self._safe_get_value(row, 'END_PAGE', debug=debug),
                    'pages_in_group': self._safe_get_value(row, 'PAGES_IN_GROUP', debug=debug),
                    'result': self._safe_get_value(row, 'ANALYSIS_RESULT', debug=debug)
                }
                parts.append(part)

            result = {
                'document': document,
                'analysis': {
                    'type': analysis_type,
                    'parts': parts
                },
                'metadata': {
                    'row_count': len(df),
                    'status': 'success',
                    'page_group_size': self.page_group_size
                }
            }

            return result

        except Exception as e:
            if debug:
                print(f"Error in _dataframe_to_json: {str(e)}")
            return {
                'document': None,
                'analysis': {
                    'type': analysis_type,
                    'parts': []
                },
                'metadata': {
                    'row_count': 0,
                    'status': 'error',
                    'error': str(e),
                    'page_group_size': self.page_group_size
                }
            }

    def _execute_query(self, query: str, analysis_type: str, debug: bool = True) -> Dict:
        """Execute query and return results as JSON"""
        try:
            if debug:
                print(f"\nExecuting query for {analysis_type}...")

            results = execute_sql(query, "snowflake")

            if debug and results is not None:
                print(f"Got results: {len(results)} rows")
                if results:
                    print(f"First result: {results[0]}")

            if results is not None:
                df = pd.DataFrame(results)
                return self._dataframe_to_json(df, analysis_type, debug)

            if debug:
                print("No results returned from query")

            return self._dataframe_to_json(pd.DataFrame(), analysis_type, debug)

        except Exception as e:
            if debug:
                print(f"Error in query execution: {str(e)}")
            return {
                'error': str(e),
                'query': query,
                'document': None,
                'analysis': {
                    'type': analysis_type,
                    'parts': []
                },
                'metadata': {
                    'row_count': 0,
                    'status': 'error',
                    'page_group_size': self.page_group_size
                }
            }

    def get_summary(self, doc_id: str) -> Dict:
        """Get text summaries for document sections"""
        analysis_function = """
            SNOWFLAKE.CORTEX.SUMMARIZE(
                ARRAY_TO_STRING(ARRAY_AGG(t.text_content || ' page:' || t.page_number), ' ')
            )
        """
        query = self.base_query.format(
            doc_id=doc_id,
            page_group_size=self.page_group_size,
            analysis_function=analysis_function
        )
        return self._execute_query(query, 'summary')

    def classify_text(self, doc_id: str, categories: List[str]) -> Dict:
        """Classify text sections into provided categories"""
        categories_sql = "ARRAY_CONSTRUCT(" + ",".join(f"'{cat}'" for cat in categories) + ")"

        analysis_function = f"""
            SNOWFLAKE.CORTEX.classify_text(
                ARRAY_TO_STRING(ARRAY_AGG(t.text_content || ' page:' || t.page_number), ' '),
                {categories_sql}
            )
        """
        query = self.base_query.format(
            doc_id=doc_id,
            page_group_size=self.page_group_size,
            analysis_function=analysis_function
        )
        return self._execute_query(query, 'classification')

    def extract_answer(self, doc_id: str, question: str) -> Dict:
        """Extract answers to specific questions from text"""
        question = question.replace("'", "''")
        analysis_function = f"""
            SNOWFLAKE.CORTEX.extract_answer(
                ARRAY_TO_STRING(ARRAY_AGG(t.text_content || ' page:' || t.page_number), ' '),
                '{question}'
            )
        """
        query = self.base_query.format(
            doc_id=doc_id,
            page_group_size=self.page_group_size,
            analysis_function=analysis_function
        )
        return self._execute_query(query, 'question_answering')

    def complete_analysis(self, doc_id: str, prompt: str, model: str = 'mixtral-8x7b') -> Dict:
        """Perform completion-based analysis using specified model"""
        prompt = prompt.replace("'", "''")
        analysis_function = f"""
            SNOWFLAKE.CORTEX.complete(
                '{model}',
                ARRAY_TO_STRING(
                    ARRAY_AGG('{prompt}\n' || t.text_content || ' page:' || t.page_number)
                , ' ')
            )
        """
        query = self.base_query.format(
            doc_id=doc_id,
            page_group_size=self.page_group_size,
            analysis_function=analysis_function
        )
        return self._execute_query(query, 'completion')