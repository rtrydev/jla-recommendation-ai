# pylint: disable=C0413
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from random import random
from typing import Any

import numpy as np
from keras.models import load_model

from src.utils.tokenizers.srt_text_tokenizer import SrtTextTokenizer
from src.utils.tokenizers.text_tokenizer import TextTokenizer

DATA_LINES = 10000
TOKENS_TO_GENERATE = 1
TOKEN_CANDIDATES = 10

if __name__ == '__main__':
    MODEL = sys.argv[1]
    DATASET = sys.argv[2]
    text_tokenizer: TextTokenizer = SrtTextTokenizer()

    token_sequences, token_to_index = text_tokenizer.tokenize(DATASET, DATA_LINES)

    tokens = list(token_to_index.keys())
    tokens = sorted(tokens, key=lambda x: token_to_index[x])
    num_tokens = len(tokens)

    numerical_sequences = [[token_to_index[token] for token in sequence] for sequence in token_sequences]

    model: Any = load_model(MODEL)

    generated_sequence = [
        token_to_index['お']
    ]

    for _ in range(TOKENS_TO_GENERATE):
        input_sequence = np.array([generated_sequence])
        predicted_token_probs = model.predict(input_sequence)

        candidates = np.argpartition(predicted_token_probs[0][-1], -TOKEN_CANDIDATES)
        selected = candidates[-TOKEN_CANDIDATES:][int(random() * TOKEN_CANDIDATES)]

        for candidate in candidates[-10:]:
            print(f'candidate: {tokens[candidate]}')

        if selected == 1:
            break

        generated_sequence.append(selected)

    generated_tokens = [tokens[token_index] for token_index in generated_sequence]
    print('Generated Sequence:', ''.join(generated_tokens))
