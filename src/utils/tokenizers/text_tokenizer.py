from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.enums.language_enum import Languages
from src.models.token_data import TokenData

class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, dataset_path: str, line_count: int, language: Languages, enhancement_variations: int) -> Tuple[List[List[str]], Dict[str, TokenData]]:
        pass

    @abstractmethod
    def save_tokens(self, token_dict: Dict[str, TokenData], file_name: str) -> None:
        pass
