from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManager
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere, ChatSnowflakeCortex
import os

from .types import LLMConfig, Provider
from connectors.snowflake_connector import SnowflakeConnection  # Import your connection class


class LLMFactory:
    """Factory for creating LangChain model instances"""

    @staticmethod
    def validate_environment(provider: Provider) -> None:
        """Validate required environment variables"""
        required_vars = {
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
            Provider.COHERE: ["COHERE_API_KEY"],
            # Remove Snowflake validation since we're using the connection class
        }

        if provider not in [Provider.SNOWFLAKE, *required_vars.keys()]:
            raise ValueError(f"Unknown provider: {provider}")

        if provider != Provider.SNOWFLAKE:  # Skip Snowflake validation
            missing = [var for var in required_vars[provider] if not os.getenv(var)]
            if missing:
                raise ValueError(f"Missing environment variables for {provider}: {', '.join(missing)}")

    @staticmethod
    def _get_base_kwargs(config: LLMConfig, callback_manager: Optional[CallbackManager] = None) -> Dict[str, Any]:
        """Get base keyword arguments for model initialization"""
        base_kwargs = {
            "temperature": config.temperature,
            "streaming": config.streaming,
            "callbacks": callback_manager.handlers if callback_manager else None
        }

        if config.max_tokens:
            base_kwargs["max_tokens"] = config.max_tokens

        return base_kwargs

    @staticmethod
    def create_model(config: LLMConfig, callback_manager: Optional[CallbackManager] = None) -> BaseChatModel:
        """Create a LangChain chat model instance"""
        try:
            # Convert provider string to enum if needed
            provider = Provider(config.provider) if isinstance(config.provider, str) else config.provider

            # Validate environment variables
            LLMFactory.validate_environment(provider)

            # Get base configuration
            base_kwargs = LLMFactory._get_base_kwargs(config, callback_manager)

            # Create model based on provider
            if provider == Provider.OPENAI:
                return ChatOpenAI(
                    model_name=config.model_name,
                    **base_kwargs
                )

            elif provider == Provider.ANTHROPIC:
                return ChatAnthropic(
                    model=config.model_name,
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                    max_retries=3,
                    model_kwargs={"top_p": config.top_p} if config.top_p else None,
                    **base_kwargs
                )

            elif provider == Provider.COHERE:
                return ChatCohere(
                    model=config.model_name,
                    cohere_api_key=os.getenv("COHERE_API_KEY"),
                    model_kwargs={"top_p": config.top_p} if config.top_p else None,
                    **base_kwargs
                )

            elif provider == Provider.SNOWFLAKE:
                # Use your Snowflake connection class
                snowflake_conn = SnowflakeConnection()
                session = snowflake_conn.get_session()

                return ChatSnowflakeCortex(
                    model=config.model_name,
                    cortex_function="complete",
                    session=session,  # Use the session from your connection class
                    model_kwargs={"top_p": config.top_p} if config.top_p else None,
                    **base_kwargs
                )

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            raise ValueError(f"Error creating model: {str(e)}")