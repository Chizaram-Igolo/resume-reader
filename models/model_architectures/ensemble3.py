from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Input, Average
import tf2crf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten

from prepare_model_input import prepare_bilstm_input, prepare_cnn_input, prepare_crf_input

# Bi-LSTM inputs
train_data_bilstm, test_data_bilstm, train_labels_bilstm, test_labels_bilstm, tokenizer_bilstm, vocab_size_bilstm, \
train_sequences_bilstm, max_sequence_length_bilstm, padded_train_sequences_bilstm = prepare_bilstm_input()

test_sequences = tokenizer_bilstm.texts_to_sequences(test_data_bilstm)
padded_test_sequences_bilstm = pad_sequences(test_sequences, maxlen=max_sequence_length_bilstm, padding='post')

# CNN inputs
train_data_cnn, test_data_cnn, train_labels_cnn, test_labels_cnn, tokenizer_cnn, vocab_size_cnn, \
train_sequences_cnn, max_sequence_length_cnn, padded_train_sequences_cnn = prepare_cnn_input()

test_sequences = tokenizer_cnn.texts_to_sequences(test_data_cnn)
padded_test_sequences_cnn = pad_sequences(test_sequences, maxlen=max_sequence_length_bilstm, padding='post')

# CRF inputs
X_train_crf, y_train_crf, X_test_crf, y_test_crf = prepare_crf_input()


# Define the Bi-LSTM Model
def create_bilstm_model(input_dim, output_dim):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=input_length),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dense(output_dim, activation='softmax')
    ])
    return model


# Define the CNN Model
def create_cnn_model(input_dim, output_dim):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=input_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(output_dim, activation='softmax')
    ])
    return model


# Define the CRF Model
def create_crf_model(input_dim, output_dim):
    input_layer = Input(shape=(None,))  # Variable sequence length
    embedding_layer = Embedding(input_dim, 64)(input_layer)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    crf_layer = tf2crf.CRF(output_dim)(lstm_layer)

    model = Model(inputs=input_layer, outputs=crf_layer)
    return model


# Create the ensemble by combining individual models
def create_ensemble(models):
    inputs = [model.input for model in models]
    outputs = [model.output for model in models]

    print(len(outputs))
    print(outputs[2])

    # Flatten each output individually
    # flattened_outputs = [Flatten()(output) for output in outputs]
    flattened_outputs = [Flatten()(outputs[2])]

    print('got here')

    # Use tf.keras.layers.Average to average the flattened predictions
    averaged = Average()(flattened_outputs)
    ensemble_model = Model(inputs=inputs, outputs=averaged)
    return ensemble_model


# Set your input and output dimensions and input_length
input_dim = 10000  # Example vocabulary size
output_dim = 10  # Number of output classes
input_length = 100  # Maximum input sequence length

# Create individual models
bilstm_model = create_bilstm_model(input_dim, output_dim)
cnn_model = create_cnn_model(input_dim, output_dim)
crf_model = create_crf_model(input_dim, output_dim)

# Compile the individual models
bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
crf_model.compile(optimizer='adam')

# Combine the models into an ensemble
ensemble_models = [bilstm_model, cnn_model, crf_model]
ensemble_model = create_ensemble(ensemble_models)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the BiLSTM model
bilstm_model.fit(padded_train_sequences_bilstm, train_labels_bilstm, epochs=8, batch_size=1)

# Train the CNN model
cnn_model.fit(padded_train_sequences_cnn, train_labels_cnn, epochs=10, batch_size=64)

# Train the CRF model
crf_model.fit(X_train_crf, y_train_crf)

# Train the ensemble model on the training data and labels
# ensemble_model.fit(
#     [padded_train_sequences_bilstm, padded_train_sequences_cnn, X_train_crf],  # Input for each individual model
#     train_labels_bilstm,  # Target labels
#     epochs=5,  # Adjust as needed
#     batch_size=32  # Adjust as needed
# )
#
# # Evaluate the ensemble model on the test set
# ensemble_loss, ensemble_accuracy = ensemble_model.evaluate(
#     [padded_test_sequences_bilstm[:, 0, :], padded_test_sequences_cnn, X_test_crf],  # Input for each individual model
#     test_labels_bilstm  # Target labels
# )
# print(f'Ensemble Model - Test Loss: {ensemble_loss:.4f}, Test Accuracy: {ensemble_accuracy:.4f}')
#
# # Finally, evaluate the ensemble model on your test data
# ensemble_model.evaluate(test_data_bilstm, test_labels_bilstm)
