from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, dataset_path: str, line_count: int) -> Tuple[List[List[str]], Dict[str, int]]:
        pass
