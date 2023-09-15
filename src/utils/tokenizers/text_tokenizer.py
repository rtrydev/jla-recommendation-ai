from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.enums.language_enum import Languages

class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, dataset_path: str, line_count: int, language: Languages, enhancement_variations: int) -> Tuple[List[List[str]], Dict[str, int]]:
        pass
