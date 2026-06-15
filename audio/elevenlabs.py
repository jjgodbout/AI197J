import requests
from io import BytesIO
from typing import List, Dict, Optional
from connectors.snowflake_connector import SnowflakeConnection
from utils.query_handler import execute_sql
import logging


class AudioManager:
    def __init__(self, eleven_labs_api_key: str):
        """Initialize AudioManager with API key and establish connections."""
        self.api_key = eleven_labs_api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.sf_connection = SnowflakeConnection()
        self.session = self.sf_connection.get_session()
        self.logger = logging.getLogger(__name__)

        # Ensure table exists with correct schema
        self._init_table()

    def _init_table(self):
        """Initialize the audio_files table with the correct schema."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS colby.ai197j.audio_files (
            audio_id NUMBER AUTOINCREMENT,
            file_name VARCHAR,
            creator_email VARCHAR,
            voice_id VARCHAR,
            model_id VARCHAR,
            stability FLOAT,
            similarity_boost FLOAT,
            style FLOAT,
            use_speaker_boost BOOLEAN,
            text_content TEXT,
            stage_file_path VARCHAR,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (audio_id)
        )
        """
        try:
            execute_sql(create_table_query, "snowflake")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    def get_available_voices(self) -> List[Dict]:
        """
        Get list of all available voices from ElevenLabs.
        Returns voices with their metadata including descriptions.
        """
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers={"xi-api-key": self.api_key},
                params={"show_legacy": False}  # Exclude legacy voices
            )
            response.raise_for_status()
            voices = response.json().get("voices", [])

            # Filter and format voice information
            available_voices = []
            for voice in voices:
                # Include premade, professional, and high_quality voices
                if voice.get("category") in ["premade", "professional", "high_quality"]:
                    # Create description including category and labels if available
                    description = voice.get("description", "No description available")
                    if voice.get("labels"):
                        labels_text = ", ".join(f"{k}: {v}" for k, v in voice["labels"].items())
                        description = f"{description}\n\nCharacteristics: {labels_text}"

                    voice_info = {
                        "voice_id": voice["voice_id"],
                        "name": voice.get("name", "Unnamed Voice"),
                        "description": description,
                        "category": voice.get("category", "unknown").title(),
                        "preview_url": voice.get("preview_url"),
                        "display_name": f"{voice.get('name', 'Unnamed Voice')} ({voice.get('category', 'unknown').title()})"
                    }
                    available_voices.append(voice_info)

            return sorted(available_voices, key=lambda x: (x["category"], x["name"]))
        except Exception as e:
            self.logger.error(f"Error fetching voices: {str(e)}")
            raise

    def _clean_filename(self, filename: str) -> str:
        """
        Clean the filename to ensure it's valid (alphanumeric, underscore, or dash).
        """
        return ''.join(e for e in filename if e.isalnum() or e in ['_', '-'])

    def _get_stage_location(self, creator_email: str) -> str:
        """
        Get the stage location path for a given creator email.
        Replaces '@' with '_at_' in the email to build a valid folder name.
        """
        return f"@COLBY.AI197J.AUDIO_FILES/{creator_email.replace('@', '_at_')}/"

    def generate_audio(
        self,
        text: str,
        voice_id: str,
        file_name: str,
        creator_email: str,
        voice_settings: Optional[Dict] = None,
        model_id: str = "eleven_multilingual_v2"
    ) -> int:
        """
        Generate audio using the ElevenLabs API and store it in Snowflake stage.
        Returns the audio_id of the generated file.
        """
        # Set default voice settings if none provided
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }

        try:
            # Call the ElevenLabs API to generate audio
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": voice_settings
                }
            )
            response.raise_for_status()

            # Clean the filename and prepare paths
            file_name = self._clean_filename(file_name)
            stage_location = self._get_stage_location(creator_email)

            # Create a BytesIO stream from the response content
            audio_stream = BytesIO(response.content)

            # Upload directly to Snowflake stage
            self.logger.info(f"Uploading file to stage: {file_name}.mp3")
            put_result = self.session.file.put_stream(
                input_stream=audio_stream,
                stage_location=f"{stage_location}{file_name}.mp3",
                auto_compress=False,
                overwrite=True,
                source_compression='NONE'
            )

            # Verify successful upload
            if not put_result or 'UPLOADED' not in str(put_result).upper():
                raise Exception(f"Failed to upload file: {put_result}")

            # -------------------------------------------------------------------
            # IMPORTANT FIX: Use the same folder structure in stage_file_path
            # that you used in `_get_stage_location()`. That means replacing
            # '@' with '_at_' in the email portion.
            # -------------------------------------------------------------------
            stage_file_path = f"{creator_email.replace('@','_at_')}/{file_name}.mp3"

            # (1) Escape any single quotes in user-provided fields for f-string
            escaped_file_name = file_name.replace("'", "''")
            escaped_creator_email = creator_email.replace("'", "''")
            escaped_voice_id = voice_id.replace("'", "''")
            escaped_model_id = model_id.replace("'", "''")
            escaped_text = text.replace("'", "''")
            escaped_stage_file_path = stage_file_path.replace("'", "''")

            # (2) Convert boolean to Snowflake literal
            use_speaker_boost_str = "true" if voice_settings["use_speaker_boost"] else "false"

            # Build the INSERT query with f-strings (use caution for injection)
            insert_query = f"""
            INSERT INTO colby.ai197j.audio_files (
                file_name, creator_email, voice_id, model_id,
                stability, similarity_boost, style, use_speaker_boost,
                text_content, stage_file_path
            ) VALUES (
                '{escaped_file_name}',
                '{escaped_creator_email}',
                '{escaped_voice_id}',
                '{escaped_model_id}',
                {voice_settings["stability"]},
                {voice_settings["similarity_boost"]},
                {voice_settings["style"]},
                {use_speaker_boost_str},
                '{escaped_text}',
                '{escaped_stage_file_path}'
            )
            """

            self.logger.info(f"Inserting audio file metadata: {file_name}")
            execute_sql(insert_query, "snowflake")

            # Retrieve the inserted record's ID
            select_query = f"""
            SELECT audio_id 
            FROM colby.ai197j.audio_files 
            WHERE file_name = '{escaped_file_name}'
            ORDER BY created_at DESC 
            LIMIT 1
            """
            result = execute_sql(select_query, "snowflake")
            if not result:
                raise Exception(f"Failed to retrieve audio_id for file {file_name}")

            audio_id = result[0]['AUDIO_ID']
            self.logger.info(f"Successfully inserted audio file '{file_name}' with ID {audio_id}")
            return audio_id

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in ElevenLabs API request: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating audio file: {str(e)}")
            raise

    def get_audio_content(self, audio_id: int) -> bytes:
        """
        Get audio file content from the Snowflake stage using streaming.
        """
        try:
            query = f"""
            SELECT stage_file_path, file_name
            FROM colby.ai197j.audio_files
            WHERE audio_id = {audio_id}
            """
            result = execute_sql(query, "snowflake")
            if not result:
                raise Exception(f"Audio file with ID {audio_id} not found")

            stage_file_path = result[0]['STAGE_FILE_PATH']
            self.logger.info(f"Downloading file from stage: {stage_file_path}")

            # Use get_stream for efficient streaming download
            stream = self.session.file.get_stream(
                stage_location=f"@COLBY.AI197J.AUDIO_FILES/{stage_file_path}",
                parallel=4
            )
            return stream.read()

        except Exception as e:
            self.logger.error(f"Error retrieving audio file: {str(e)}")
            raise

    def get_audio_metadata(self, audio_id: int) -> Dict:
        """
        Get metadata for an audio file by ID.
        """
        query = f"""
        SELECT 
            audio_id, file_name, creator_email, voice_id, model_id,
            stability, similarity_boost, style, use_speaker_boost,
            text_content, created_at, stage_file_path
        FROM colby.ai197j.audio_files 
        WHERE audio_id = {audio_id}
        """
        result = execute_sql(query, "snowflake")
        if not result:
            raise Exception(f"Audio file with ID {audio_id} not found")
        return result[0]

    def get_user_audio_files(self, creator_email: str) -> List[Dict]:
        """
        Get the list of audio files for a specific user by creator_email.
        """
        escaped_creator_email = creator_email.replace("'", "''")
        query = f"""
        SELECT 
            audio_id, file_name, text_content, created_at, stage_file_path
        FROM colby.ai197j.audio_files 
        WHERE creator_email = '{escaped_creator_email}'
        ORDER BY created_at DESC
        """
        return execute_sql(query, "snowflake")
