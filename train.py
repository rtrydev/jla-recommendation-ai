import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from random import random

import numpy as np

from keras import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.models import load_model

from text_tokenizer import get_tokens_from_srt

DATA_LINES = 10000
EPOCHS = 20
TOKENS_TO_GENERATE = 64
TOKEN_CANDIDATES = 1
RESUME = False
SKIP_TRAIN = False

if __name__ == '__main__':
    dataset = sys.argv[1]
    # Example token sequences
    token_sequences, token_to_index = get_tokens_from_srt(dataset, DATA_LINES)

    # Create vocabulary
    tokens = [token for token in token_to_index.keys()]
    tokens = sorted(tokens, key=lambda x: token_to_index[x])
    num_tokens = len(tokens)

    # Convert token sequences to numerical sequences
    numerical_sequences = [[token_to_index[token] for token in sequence] for sequence in token_sequences]

    # Generate training data
    X = [sequence[:-1] for sequence in numerical_sequences]
    y = [sequence[1:] for sequence in numerical_sequences]

    # Model architecture
    embedding_dim = 300
    hidden_units = 256

    if RESUME:
        model = load_model('trained-model-20.h5')

        if not SKIP_TRAIN:
            model.fit(np.array(X), np.array(y), epochs=EPOCHS)
            model.save('trained-model-40.h5')
    else:
        model = Sequential([
            Embedding(input_dim=num_tokens, output_dim=embedding_dim),
            LSTM(hidden_units, return_sequences=True),
            Dense(num_tokens, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='Adam'
        )
        model.summary()
        a = np.array(X)
        b = np.array(y)
        model.fit(np.array(X), np.array(y), epochs=EPOCHS)
        model.save('trained-model-20.h5')

    # Generate new sequences
    generated_sequence = [
        token_to_index["お"]
    ]

    for _ in range(TOKENS_TO_GENERATE):
        input_sequence = np.array([generated_sequence])
        predicted_token_probs = model.predict(input_sequence)

        ind = np.argpartition(predicted_token_probs[0][-1], -TOKEN_CANDIDATES)[-TOKEN_CANDIDATES:]
        selected = ind[int(random() * TOKEN_CANDIDATES)]

        if selected == 1:
            break

        generated_sequence.append(selected)

    generated_tokens = [tokens[token_index] for token_index in generated_sequence]
    print("Generated Sequence:", ''.join(generated_tokens))
