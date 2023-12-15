import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense

from prepare_model_input import prepare_bilstm_input

# Retrieve training data and labels and testing data and labels
train_data, test_data, train_labels, test_labels, tokenizer, vocab_size, train_sequences, max_sequence_length, \
    padded_train_sequences = prepare_bilstm_input()

test_sequences = tokenizer.texts_to_sequences(test_data)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')


def build_bi_lstm_model():
    bi_lstm_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dense(1, activation='sigmoid')  # Adjust the number of units based on your task
    ])
    bi_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return bi_lstm_model


if __name__ == "__main__":
    model = build_bi_lstm_model()

    # Train the model
    model.fit(padded_train_sequences, train_labels, epochs=10, batch_size=1)
    model.save("saved_models/bi_lstm_model.keras")

    # Ensure labels are reshaped to match the output shape of the model
    test_labels = np.expand_dims(test_labels, axis=-1)

    loss, accuracy = model.evaluate(padded_test_sequences, test_labels)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
