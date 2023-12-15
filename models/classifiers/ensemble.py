from keras.models import Model, load_model
from keras.layers import Dense, Concatenate, ZeroPadding1D

from bi_lstm import padded_train_sequences as train_data_bi_lstm, train_labels
from cnn import padded_train_sequences as train_data_cnn

bi_model = load_model("saved_models/bi_lstm_model.keras")
c_model = load_model("saved_models/cnn_model.keras")


# Create the ensemble by combining individual models
def create_ensemble(models):
    bi_lstm_model = models[0]
    cnn_model = models[1]

    input_shape = cnn_model.layers[1].output.shape
    desired_shape = bi_lstm_model.layers[1].output.shape

    # Pad the tensor to the desired shape
    padding_size = desired_shape[1] - input_shape[1] - 1
    padded_cnn_output = ZeroPadding1D(padding=padding_size)(cnn_model.layers[1].output)

    # Remove the output layer of each model
    models_no_output = [Model(bi_lstm_model.input, bi_lstm_model.layers[1].output),
                        Model(cnn_model.input, padded_cnn_output)]

    # Concatenate the outputs of the models
    concatenated = Concatenate()([model.output for model in models_no_output])

    # Add a dense layer for classification
    dense = Dense(1, activation='sigmoid')(concatenated)

    # Create the ensemble model
    model = Model(inputs=[model.input for model in models], outputs=dense)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Combine the models into an ensemble
ensemble_model = create_ensemble([bi_model, c_model])

# Train the ensemble model
ensemble_model.fit(
    x=[train_data_bi_lstm, train_data_cnn],
    y=train_labels,
    epochs=10,  # Adjust the number of epochs
    batch_size=64  # Adjust the batch size
)

ensemble_model.save("saved_models/ensemble_model.keras")
