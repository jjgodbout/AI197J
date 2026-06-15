"""
Snowflake-backed repository for chat threads and messages.
Replaces all LiteralAI thread/step/generation operations.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import json
import logging

logger = logging.getLogger(__name__)

THREADS_TABLE = "COLBY.AI197J.CHAT_THREADS"
MESSAGES_TABLE = "COLBY.AI197J.CHAT_MESSAGES"


class ChatRepository:
    """CRUD operations for chat threads and messages stored in Snowflake."""

    def __init__(self, query_handler):
        self.query_handler = query_handler

    def _execute(self, query: str) -> list:
        result = self.query_handler(query, "snowflake")
        if result:
            return [dict(row.asDict()) for row in result]
        return []

    # ---- threads ----

    def create_thread(
        self,
        user_email: str,
        name: str,
        model_name: str = None,
        model_id: str = None,
        provider: str = None,
        settings: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        thread_id = str(uuid.uuid4())
        settings_json = json.dumps(settings or {}).replace("'", "''")
        name_escaped = (name or "").replace("'", "''")

        self._execute(f"""
            INSERT INTO {THREADS_TABLE}
                (ID, USER_EMAIL, NAME, MODEL_NAME, MODEL_ID, PROVIDER, SETTINGS)
            VALUES (
                '{thread_id}',
                '{user_email}',
                '{name_escaped}',
                '{model_name or ""}',
                '{model_id or ""}',
                '{provider or ""}',
                PARSE_JSON('{settings_json}')
            )
        """)

        logger.info(f"Created thread {thread_id} for {user_email}")
        return {
            "id": thread_id,
            "user_email": user_email,
            "name": name,
            "model_name": model_name,
            "model_id": model_id,
            "provider": provider,
            "settings": settings or {},
            "created_at": datetime.utcnow().isoformat(),
        }

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute(f"""
            SELECT * FROM {THREADS_TABLE}
            WHERE ID = '{thread_id}'
        """)
        if not rows:
            return None
        return self._thread_row_to_dict(rows[0])

    def get_user_threads(self, user_email: str) -> List[Dict[str, Any]]:
        rows = self._execute(f"""
            SELECT * FROM {THREADS_TABLE}
            WHERE USER_EMAIL = '{user_email}'
            ORDER BY CREATED_AT DESC
        """)
        return [self._thread_row_to_dict(r) for r in rows]

    def update_thread_settings(self, thread_id: str, new_settings: Dict) -> bool:
        thread = self.get_thread(thread_id)
        if not thread:
            return False
        merged = {**thread.get("settings", {}), **new_settings}
        settings_json = json.dumps(merged).replace("'", "''")
        self._execute(f"""
            UPDATE {THREADS_TABLE}
            SET SETTINGS = PARSE_JSON('{settings_json}'),
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE ID = '{thread_id}'
        """)
        return True

    def delete_thread(self, thread_id: str) -> bool:
        # Delete messages first, then thread
        self._execute(f"DELETE FROM {MESSAGES_TABLE} WHERE THREAD_ID = '{thread_id}'")
        self._execute(f"DELETE FROM {THREADS_TABLE} WHERE ID = '{thread_id}'")
        logger.info(f"Deleted thread {thread_id}")
        return True

    # ---- messages ----

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        model_name: str = None,
        token_count: int = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        content_escaped = (content or "").replace("'", "''")
        meta_json = json.dumps(metadata or {}).replace("'", "''")

        self._execute(f"""
            INSERT INTO {MESSAGES_TABLE}
                (THREAD_ID, ROLE, CONTENT, MODEL_NAME, TOKEN_COUNT, METADATA)
            VALUES (
                '{thread_id}',
                '{role}',
                '{content_escaped}',
                '{model_name or ""}',
                {token_count or 0},
                PARSE_JSON('{meta_json}')
            )
        """)
        return {
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
        }

    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        rows = self._execute(f"""
            SELECT * FROM {MESSAGES_TABLE}
            WHERE THREAD_ID = '{thread_id}'
            ORDER BY CREATED_AT ASC
        """)
        return [self._message_row_to_dict(r) for r in rows]

    def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        rows = self._execute(f"""
            SELECT
                COUNT(*) AS TOTAL_MESSAGES,
                SUM(TOKEN_COUNT) AS TOTAL_TOKENS,
                MIN(CREATED_AT) AS FIRST_MESSAGE,
                MAX(CREATED_AT) AS LAST_MESSAGE
            FROM {MESSAGES_TABLE}
            WHERE THREAD_ID = '{thread_id}'
        """)
        if not rows:
            return {}
        r = rows[0]
        return {
            "total_messages": r.get("TOTAL_MESSAGES", 0),
            "total_tokens": r.get("TOTAL_TOKENS", 0),
            "first_message": r.get("FIRST_MESSAGE"),
            "last_message": r.get("LAST_MESSAGE"),
        }

    # ---- helpers ----

    @staticmethod
    def _thread_row_to_dict(row: Dict) -> Dict[str, Any]:
        settings = row.get("SETTINGS", {})
        if isinstance(settings, str):
            try:
                settings = json.loads(settings)
            except json.JSONDecodeError:
                settings = {}
        return {
            "id": row["ID"],
            "user_email": row["USER_EMAIL"],
            "name": row.get("NAME", ""),
            "model_name": row.get("MODEL_NAME", ""),
            "model_id": row.get("MODEL_ID", ""),
            "provider": row.get("PROVIDER", ""),
            "settings": settings,
            "created_at": row.get("CREATED_AT"),
            "updated_at": row.get("UPDATED_AT"),
        }

    @staticmethod
    def _message_row_to_dict(row: Dict) -> Dict[str, Any]:
        metadata = row.get("METADATA", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return {
            "id": row.get("ID"),
            "thread_id": row["THREAD_ID"],
            "role": row["ROLE"],
            "content": row.get("CONTENT", ""),
            "model_name": row.get("MODEL_NAME", ""),
            "token_count": row.get("TOKEN_COUNT", 0),
            "metadata": metadata,
            "created_at": row.get("CREATED_AT"),
        }
