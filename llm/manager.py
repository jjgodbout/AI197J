from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManager

from .types import LLMConfig, Provider
from .repository import LLMRepository
from .factory import LLMFactory

class LLMManager:
    """Manager for handling LLM configurations and instances"""

    def __init__(self, query_handler, callback_manager: Optional[CallbackManager] = None):
        self.repository = LLMRepository(query_handler)
        self.callback_manager = callback_manager
        self.current_config: Optional[LLMConfig] = None
        self.current_model: Optional[BaseChatModel] = None

    def initialize_model(self, model_id: str) -> BaseChatModel:
        """Initialize a model by its ID"""
        config = self.repository.get_model(model_id)
        if not config:
            raise ValueError(f"Model {model_id} not found or not active")

        try:
            self.current_config = config
            self.current_model = LLMFactory.create_model(
                config,
                self.callback_manager
            )
            return self.current_model
        except Exception as e:
            raise ValueError(f"Failed to initialize model: {str(e)}")

    def get_available_models(self, provider: Optional[Provider] = None) -> Dict[str, Dict[str, Any]]:
        """Get available models, optionally filtered by provider"""
        if provider:
            models = self.repository.get_models_by_provider(provider)