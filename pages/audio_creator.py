import streamlit as st
import os
from audio.elevenlabs import AudioManager
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import List, Dict


def count_words(text: str) -> int:
    """Count the number of words in a text string"""
    return len(text.split())


class AudioCreator:
    def __init__(self):
        """Initialize AudioCreator with AudioManager"""
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.MAX_WORDS = 1800

        # Load environment variables
        load_dotenv()
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            self.logger.error("ELEVENLABS_API_KEY not found in environment variables")
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

        self.audio_manager = AudioManager(api_key)

    def render_interface(self):
        """Render the audio creator interface"""
        st.header("Audio Creator")

        # Get available voices
        try:
            voices = self.audio_manager.get_available_voices()
            self.logger.info(f"Successfully fetched {len(voices)} voices")
        except Exception as e:
            self.logger.error(f"Error fetching voices: {str(e)}")
            st.error("Failed to fetch available voices. Please try again later.")
            return

        # Voice selection outside the form for dynamic updates
        voice_options = {voice["display_name"]: voice for voice in voices}

        selected_voice = st.selectbox(
            "Select a voice",
            options=list(voice_options.keys()),
            help="Choose a voice for your audio. Hover over options to see descriptions and characteristics.",
            format_func=lambda x: x,
            key="voice_selector"
        )

        # Show preview if available (outside the form)
        if selected_voice:
            voice = voice_options[selected_voice]
            selected_voice_id = voice["voice_id"]
            selected_voice_name = voice["name"]

            if voice.get("preview_url"):
                st.audio(voice["preview_url"], format='audio/mp3')

        # Input form
        with st.form("audio_creation_form"):
            text_input = st.text_area(
                f"Enter text to convert to speech (max {self.MAX_WORDS:,} words)",
                max_chars=10000,
                help=f"Enter the text you want to convert to speech (limit: {self.MAX_WORDS:,} words)"
            )

            # Show word count below text area
            if text_input:
                word_count = count_words(text_input)
                count_color = "red" if word_count > self.MAX_WORDS else "green"
                st.markdown(
                    f'<p style="color: {count_color}">Word count: {word_count:,}/{self.MAX_WORDS:,}</p>',
                    unsafe_allow_html=True
                )

            # Voice settings
            st.subheader("Voice Settings")
            col1, col2 = st.columns(2)
            with col1:
                stability = st.slider(
                    "Stability", 0.0, 1.0, 0.5,
                    help="Higher stability makes the voice more consistent"
                )
                style = st.slider(
                    "Style Exaggeration", 0.0, 1.0, 0.6,
                    help="Higher style increases expressiveness"
                )
            with col2:
                similarity_boost = st.slider(
                    "Similarity Boost", 0.0, 1.0, 0.75,
                    help="Higher similarity makes the voice more similar to the original"
                )
                speaker_boost = st.checkbox(
                    "Use Speaker Boost", value=True,
                    help="Enhance voice clarity and reduce background noise"
                )

            file_name = st.text_input(
                "File name (without extension)",
                help="Enter the name for your audio file"
            )

            submit_button = st.form_submit_button("Generate Audio")

        if submit_button:
            if not text_input:
                st.warning("Please enter some text to convert to speech.")
                return
            if not file_name:
                st.warning("Please enter a file name.")
                return
            if not selected_voice:
                st.warning("Please select a voice.")
                return

            # Check word count before processing
            word_count = count_words(text_input)
            if word_count > self.MAX_WORDS:
                st.error(
                    f"Text exceeds maximum word limit. Please reduce text to {self.MAX_WORDS:,} words or less. Current count: {word_count:,} words")
                return

            try:
                with st.spinner("Generating audio..."):
                    voice_settings = {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style,
                        "use_speaker_boost": speaker_boost
                    }

                    # Generate audio
                    self.logger.info(f"Generating audio for file: {file_name}")
                    audio_id = self.audio_manager.generate_audio(
                        text=text_input,
                        voice_id=selected_voice_id,
                        file_name=file_name,
                        creator_email=st.session_state.get("username", ""),
                        voice_settings=voice_settings
                    )

                    # Get metadata
                    metadata = self.audio_manager.get_audio_metadata(audio_id)
                    audio_content = self.audio_manager.get_audio_content(audio_id)

                    st.success("Audio generated successfully!")

                    # Display audio player and download button
                    st.subheader("Generated Audio")

                    # Audio player
                    st.audio(audio_content, format='audio/mp3')

                    # Download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_content,
                        file_name=f"{file_name}.mp3",
                        mime="audio/mp3",
                        key=f"download_{audio_id}"
                    )

                    # Display metadata
                    st.markdown("**File Details:**")
                    st.write(f"- Name: {metadata['FILE_NAME']}")
                    st.write(f"- Voice: {selected_voice_name}")
                    st.write(f"- Created: {metadata['CREATED_AT'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- Words: {word_count:,}")

                    self.logger.info(f"Successfully generated and displayed audio for {file_name}")

            except Exception as e:
                self.logger.error(f"Error in audio generation process: {str(e)}")
                st.error(f"Error generating audio: {str(e)}")
                return

        # Show user's previous audio files
        try:
            st.subheader("Your Audio Files")
            user_files = self.audio_manager.get_user_audio_files(
                st.session_state.get("username", "")
            )

            if not user_files:
                st.info("No previous audio files found.")
                return

            for file in user_files:
                with st.expander(f"📁 {file['FILE_NAME']} - {file['CREATED_AT'].strftime('%Y-%m-%d %H:%M')}"):
                    try:
                        audio_content = self.audio_manager.get_audio_content(file['AUDIO_ID'])
                        st.audio(audio_content, format='audio/mp3')
                        st.download_button(
                            label=f"Download {file['FILE_NAME']}",
                            data=audio_content,
                            file_name=f"{file['FILE_NAME']}.mp3",
                            mime="audio/mp3",
                            key=f"download_existing_{file['AUDIO_ID']}"
                        )
                        if file['TEXT_CONTENT']:
                            st.markdown("**Text:**")
                            st.text(file['TEXT_CONTENT'])
                    except Exception as e:
                        self.logger.error(f"Error displaying audio file {file['FILE_NAME']}: {str(e)}")
                        st.warning(f"Unable to load audio for {file['FILE_NAME']}")

        except Exception as e:
            self.logger.error(f"Error displaying user's audio files: {str(e)}")
            st.error("Failed to load previous audio files. Please try again later.")