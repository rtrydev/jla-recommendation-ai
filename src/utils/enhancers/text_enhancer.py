from typing import List

import numpy as np

class TextEnhancer:
    def randomize_token_sequences(self, lines: List[List[str]], variations: int) -> List[List[str]]:
        result = []
        for line in lines:
            for _ in range(variations):
                np.random.shuffle(line)
                result.append(list(line))

        return result
