from dataclasses import dataclass
from typing import Optional

from src.enums.token_type_enum import TokenType


@dataclass
class TokenData:
    token_id: int
    lemma_id: Optional[str]
    token: str
    infinitive: Optional[str]
    reading: Optional[str]
    token_type: TokenType
