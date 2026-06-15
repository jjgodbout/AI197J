import pandas as pd
import tiktoken
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from snowflake.snowpark import Session
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, LongType
from connectors.snowflake_connector import SnowflakeConnection
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class TextChunkProcessor:
    def __init__(self, max_workers=2):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.sf_connection = SnowflakeConnection()
        self.session = self.sf_connection.get_session()
        # Set the schema explicitly
        self.session.sql('USE SCHEMA "COLBY"."AI197J"').collect()
        self.max_workers = max_workers

    def count_tokens(self, text):
        if not isinstance(text, str):
            return 0
        return len(self.tokenizer.encode(text))

    def preprocess_text(self, text):
        if isinstance(text, str):
            return text.replace('\n', ' ').replace('"', '')
        return str(text)

    def split_text_into_chunks(self, text, chunk_size=380, chunk_overlap=50):
        if not isinstance(text, str):
            return []
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = Document(text=text)
        return splitter.split_text(doc.text)

    def process_row(self, row):
        chunks = self.split_text_into_chunks(row['TEXT_CONTENT'])
        return [{
            'text_chunk': chunk,
            'document_id': row['DOCUMENT_ID'],
            'page': row['PAGE_NUMBER'],
            'chunk_number': idx + 1,  # Start numbering from 1
            'text_type': row['EXTRACTION_METHOD']
        } for idx, chunk in enumerate(chunks)]

    def process_dataframe(self, df):
        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_row, row) for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(df), desc="Processing rows"):
                chunks.extend(future.result())
        return pd.DataFrame(chunks)

    def get_text_query(self):
        query = """
            SELECT 
                m.DOCUMENT_ID,
                m.PAGE_NUMBER,
                m.TEXT_CONTENT,
                m.EXTRACTION_METHOD
            FROM "COLBY"."AI197J"."MARKDOWN_TEXT" m
            LEFT JOIN "COLBY"."AI197J"."TEXT_CHUNKS" t 
                ON m.DOCUMENT_ID = t.DOCUMENT_ID 
                AND m.PAGE_NUMBER = t.PAGE
            WHERE t.DOCUMENT_ID IS NULL
            ORDER BY m.DOCUMENT_ID, m.PAGE_NUMBER
        """
        return query

    def run(self):
        try:
            text_query = self.get_text_query()
            text_df = self.session.sql(text_query).to_pandas()

            if text_df.empty:
                return "No new documents found for processing."

            text_df['TEXT_CONTENT'] = text_df['TEXT_CONTENT'].apply(self.preprocess_text)
            df_exploded = self.process_dataframe(text_df)

            schema = StructType([
                StructField("text_chunk", StringType()),
                StructField("document_id", LongType()),
                StructField("chunk_number", IntegerType()),
                StructField("page", IntegerType()),
                StructField("text_type", StringType())
            ])

            # Create dataframe with explicit schema=None to avoid warning
            snowpark_df = self.session.create_dataframe(df_exploded, schema=None)
            # Use fully qualified table name
            snowpark_df.write.mode("append").save_as_table('"COLBY"."AI197J"."TEXT_CHUNKS"')

            success_message = (f"Success: Processed {len(text_df)} documents into {len(df_exploded)} chunks "
                             f"using {self.max_workers} workers.")
            self.logger.info(success_message)
            return success_message
        except Exception as e:
            error_message = f"Error occurred during processing: {str(e)}"
            self.logger.error(error_message)
            return error_message
        finally:
            self.session.close()