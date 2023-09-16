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

    def enhance_neighborhood(self, lines: List[List[str]], neighborhood_size: int) -> List[List[str]]:
        result = []
        for idx in range(neighborhood_size, len(lines) - neighborhood_size):
            current_enhanced_line = []
            for neighbor_idx in range(idx - neighborhood_size, idx + neighborhood_size):
                current_enhanced_line += lines[neighbor_idx]

            result.append(current_enhanced_line)

        return result
