# pylint: disable=C0413
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from typing import Any

import numpy as np
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.models import load_model

from src.utils.factories.tokenizer_factory import create_tokenizer

DATA_LINES = 50000
EPOCHS = 20

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python {sys.argv[0]} <DATASET> <RESULT_MODEL.h5>')
        sys.exit(1)

    DATASET = sys.argv[1]
    RESULT_MODEL = sys.argv[2]
    BASE_MODEL = sys.argv[3] if len(sys.argv) > 3 else None

    text_tokenizer = create_tokenizer(DATASET)
    token_sequences, token_to_index = text_tokenizer.tokenize(DATASET, DATA_LINES)

    tokens = list(token_to_index.keys())
    tokens = sorted(tokens, key=lambda x: token_to_index[x])
    num_tokens = len(tokens)

    numerical_sequences = [[token_to_index[token] for token in sequence] for sequence in token_sequences]

    X = [sequence[:-1] for sequence in numerical_sequences]
    Y = [sequence[1:] for sequence in numerical_sequences]

    EMBEDDING_DIM = 300
    HIDDEN_UNITS = 256

    if BASE_MODEL is not None:
        model: Any = load_model(BASE_MODEL)

        model.fit(np.array(X), np.array(Y), epochs=EPOCHS)
        model.save(RESULT_MODEL)

    else:
        model = Sequential([
            Embedding(input_dim=num_tokens, output_dim=EMBEDDING_DIM),
            LSTM(HIDDEN_UNITS, return_sequences=True),
            Dense(num_tokens, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='Adam'
        )
        model.summary()
        a = np.array(X)
        b = np.array(Y)
        model.fit(np.array(X), np.array(Y), epochs=EPOCHS)
        model.save(RESULT_MODEL)

    print(f'Training finished! Saved weights to {RESULT_MODEL}')
