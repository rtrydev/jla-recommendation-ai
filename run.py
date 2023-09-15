# pylint: disable=C0413
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from random import random
from typing import Any

import numpy as np
from keras.models import load_model

from src.utils.tokenizers.token_loader import TokenLoader

TOKENS_TO_GENERATE = 1
TOKEN_CANDIDATES = 1
CANDIDATES_TO_DISPLAY = 50

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
        token_dict['アメリカ'].token_id
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
        ]
        candidate_probabilities.sort(key=lambda element: element['probability'], reverse=True)

        selected = candidate_probabilities[:TOKEN_CANDIDATES][int(random() * TOKEN_CANDIDATES)]

        for candidate in candidate_probabilities[:CANDIDATES_TO_DISPLAY]:
            print(f'candidate: {tokens[candidate["index"]]}; Probability: {candidate["probability"]}')

        if selected == 1:
            break

        generated_sequence.append(selected['index'])

    generated_tokens = [tokens[token_index].token for token_index in generated_sequence]
    print('Generated Sequence:', ''.join(generated_tokens))
