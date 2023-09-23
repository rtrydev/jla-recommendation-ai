# pylint: disable=C0413
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from random import random
from typing import Any

import numpy as np
from keras.models import load_model

from src.utils.tokenizers.token_loader import TokenLoader
from src.enums.token_type_enum import TokenType

TOKENS_TO_GENERATE = 1
TOKEN_CANDIDATES = 1
CANDIDATES_TO_DISPLAY = 10
REQUIRE_INFINITIVE = True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python {sys.argv[0]} <TOKEN_DUMP> <MODEL>')
        sys.exit(1)

    TOKEN_DUMP = sys.argv[1]
    MODEL = sys.argv[2]

    token_dict = TokenLoader().load(TOKEN_DUMP)
    num_tokens = len(token_dict.keys())

    print(f'Known tokens: {num_tokens}')

    tokens = {
        entry.token_id: entry
        for entry in token_dict.values()
    }
    model: Any = load_model(MODEL)

    generated_sequence = [
        token_dict['配信'].token_id
    ]

    for _ in range(TOKENS_TO_GENERATE):
        input_sequence = np.array([generated_sequence])
        predicted_token_probs = model.predict(input_sequence)

        candidate_probabilities = [
            {
                'index': idx,
                'probability': probability
            }
            for idx, probability in enumerate(predicted_token_probs[0][-1])
            if TokenType(tokens[idx].token_type) not in [
                    TokenType.BLANK,
                    TokenType.PARTICLE,
                    TokenType.META,
                    TokenType.PUNCTUATION,
                    TokenType.AUXILARY_VERB
                ]
            and (tokens[idx].infinitive is not None and (tokens[idx].infinitive != 'None')) or not REQUIRE_INFINITIVE
        ]
        candidate_probabilities.sort(key=lambda element: element['probability'], reverse=True)

        selected = candidate_probabilities[:TOKEN_CANDIDATES][int(random() * TOKEN_CANDIDATES)]

        for idx, candidate in enumerate(candidate_probabilities[:CANDIDATES_TO_DISPLAY]):
            candidate_data = tokens[candidate["index"]]
            print(
                f'''
                    candidate {idx + 1}:
                    token: {candidate_data.token},
                    inf: {candidate_data.infinitive},
                    reading: {candidate_data.reading},
                    type: {TokenType(candidate_data.token_type).name};
                    Probability: {candidate["probability"]}
                '''
            )

        if selected == 1:
            break

        generated_sequence.append(selected['index'])
