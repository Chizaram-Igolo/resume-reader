from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Input
import tf2crf

from prepare_model_input import prepare_bilstm_input, prepare_cnn_input, prepare_crf_input


X_train_bilstm, X_test_bilstm, y_train_bilstm, y_test_bilstm, tokenizer_bilstm, vocab_size_bilstm, \
    train_sequences_bilstm, max_sequence_length_bilstm, padded_train_sequences_bilstm = prepare_bilstm_input()

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, tokenizer_cnn, vocab_size_cnn, train_sequences_cnn, \
    max_sequence_length_cnn, padded_train_sequences_cnn = prepare_cnn_input()

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

    # Add a Dense layer to each output before applying GlobalAveragePooling1D
    dense_outputs = [Dense(64, activation='relu')(output) for output in outputs]

    # Use GlobalAveragePooling1D to ensure consistent shapes before averaging
    pooled_outputs = [GlobalAveragePooling1D()(output) for output in dense_outputs]

    # Use tf.keras.layers.Average to average the pooled predictions
    averaged = Average()(pooled_outputs)
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

# Train individual models
bilstm_model.fit(X_train_bilstm, y_train_bilstm, epochs=10, batch_size=32, validation_data=(X_test_bilstm,
                                                                                            y_test_bilstm))
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test_cnn))
crf_model.fit(X_train_crf, y_train_crf, epochs=10, batch_size=32, validation_data=(X_test_crf, y_test_crf))

# Combine the models into an ensemble
ensemble_models = [bilstm_model, cnn_model, crf_model]
ensemble_model = create_ensemble(ensemble_models)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train ensemble model
ensemble_model.fit([X_train_cnn]*len(ensemble_models), y_train_cnn, epochs=10, batch_size=32,
                   validation_data=([X_test_cnn]*len(ensemble_models), y_test_cnn))

# Finally, evaluate the ensemble model on your test data
ensemble_model.evaluate([X_test_cnn]*len(ensemble_models), y_test_cnn)
