# pylint: disable=C0413
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from random import random
from typing import Any

import numpy as np
from keras.models import load_model

from src.utils.factories.tokenizer_factory import create_tokenizer

DATA_LINES = 50000
TOKENS_TO_GENERATE = 1
TOKEN_CANDIDATES = 1

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python {sys.argv[0]} <DATASET> <MODEL>')
        sys.exit(1)

    DATASET = sys.argv[1]
    MODEL = sys.argv[2]

    text_tokenizer = create_tokenizer(DATASET)
    token_sequences, token_to_index = text_tokenizer.tokenize(DATASET, DATA_LINES)

    tokens = list(token_to_index.keys())
    tokens = sorted(tokens, key=lambda x: token_to_index[x])
    num_tokens = len(tokens)

    numerical_sequences = [[token_to_index[token] for token in sequence] for sequence in token_sequences]

    model: Any = load_model(MODEL)

    generated_sequence = [
        token_to_index['<|start|>']
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

        for candidate in candidate_probabilities[:30]:
            print(f'candidate: {tokens[candidate["index"]]}: {candidate["probability"]}')

        if selected == 1:
            break

        generated_sequence.append(selected['index'])

    generated_tokens = [tokens[token_index] for token_index in generated_sequence]
    print('Generated Sequence:', ''.join(generated_tokens))
