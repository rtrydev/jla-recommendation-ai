from abc import ABC, abstractmethod
from typing import List

class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, dataset_path: str, line_count: int) -> List[List[str]]:
        pass
