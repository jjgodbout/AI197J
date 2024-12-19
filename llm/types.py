from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    SNOWFLAKE = "snowflake"


class LLMConfig(BaseModel):
    """Configuration class matching Snowflake table schema"""
    model_id: str
    model_name: str
    provider: Provider
    model_type: str
    temperature: float = Field(ge=0.0, le=2.0)
    streaming: bool = True
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    description: Optional[str] = None
    model_group: Optional[str] = None
    context_length: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_snowflake(cls, row: Dict[str, Any]) -> 'LLMConfig':
        """Create configuration from Snowflake row"""
        # Handle both Row object and dictionary input
        if hasattr(row, 'asDict'):
            row = row.asDict()

        return cls(
            model_id=str(row['MODEL_ID']),  # Ensure string type
            model_name=str(row['MODEL_NAME']),
            provider=str(row['PROVIDER']),
            model_type=str(row['MODEL_TYPE']),
            temperature=float(row['TEMPERATURE']),  # Ensure float type
            streaming=bool(row['STREAMING']),  # Ensure boolean type
            max_tokens=int(row['MAX_TOKENS']) if row.get('MAX_TOKENS') is not None else None,
            top_p=float(row['TOP_P']) if row.get('TOP_P') is not None else None,
            frequency_penalty=float(row['FREQUENCY_PENALTY']) if row.get('FREQUENCY_PENALTY') is not None else None,
            presence_penalty=float(row['PRESENCE_PENALTY']) if row.get('PRESENCE_PENALTY') is not None else None,
            description=str(row['DESCRIPTION']) if row.get('DESCRIPTION') else None,
            model_group=str(row['MODEL_GROUP']) if row.get('MODEL_GROUP') else None,
            context_length=int(row['CONTEXT_LENGTH']) if row.get('CONTEXT_LENGTH') is not None else None,
            created_at=row.get('CREATED_AT'),
            updated_at=row.get('UPDATED_AT'),
            is_active=bool(row.get('IS_ACTIVE', True)),
            metadata=row.get('METADATA')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Snowflake operations"""
        return {
            'MODEL_ID': self.model_id,
            'MODEL_NAME': self.model_name,
            'PROVIDER': self.provider,
            'MODEL_TYPE': self.model_type,
            'TEMPERATURE': self.temperature,
            'STREAMING': self.streaming,
            'MAX_TOKENS': self.max_tokens,
            'TOP_P': self.top_p,
            'FREQUENCY_PENALTY': self.frequency_penalty,
            'PRESENCE_PENALTY': self.presence_penalty,
            'DESCRIPTION': self.description,
            'MODEL_GROUP': self.model_group,
            'CONTEXT_LENGTH': self.context_length,
            'IS_ACTIVE': self.is_active,
            'METADATA': self.metadata
        }

    @property
    def model_kwargs(self) -> Optional[Dict[str, Any]]:
        """Get model-specific kwargs for LangChain"""
        kwargs = {}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        return kwargs if kwargs else None