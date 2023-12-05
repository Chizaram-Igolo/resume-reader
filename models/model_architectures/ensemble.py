from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, \
    Input, Average
import tf2crf


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

# Combine the models into an ensemble
ensemble_models = [bilstm_model, cnn_model, crf_model]
ensemble_model = create_ensemble(ensemble_models)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train each individual model and the ensemble model on your dataset
# You will need to prepare your data and labels accordingly

# Finally, evaluate the ensemble model on your test data
# ensemble_model.evaluate(test_data, test_labels)
