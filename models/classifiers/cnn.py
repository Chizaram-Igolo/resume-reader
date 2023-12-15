import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

from prepare_model_input import prepare_cnn_input

# Retrieve training data
train_data, test_data, train_labels, test_labels, tokenizer, vocab_size, train_sequences, max_sequence_length, \
    padded_train_sequences = prepare_cnn_input()

# Prepare testing data
test_sequences = tokenizer.texts_to_sequences(test_data)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')

# Define the CNN model
embedding_dim = 50  # You can adjust this based on your specific task and vocabulary size
filter_sizes = [3, 4, 5]  # Convolutional filters of different sizes
num_filters = 128  # Number of filters in each convolutional layer


def build_cnn_model():
    cnn_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length, name='cnn'),
        Conv1D(filters=num_filters, kernel_size=filter_sizes[0], activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')  # Adjust the number of units based on your task
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_model


if __name__ == "__main__":
    model = build_cnn_model()

    # Train the model
    model.fit(padded_train_sequences, train_labels, epochs=10, batch_size=64)
    model.save("saved_models/cnn_model.keras")

    # Ensure labels are reshaped to match the output shape of the model
    test_labels = np.expand_dims(test_labels, axis=-1)

    loss, accuracy = model.evaluate(padded_test_sequences, test_labels)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
