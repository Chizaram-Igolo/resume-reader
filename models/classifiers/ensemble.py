import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from keras.models import Model, load_model
from keras.layers import Dense, Concatenate, ZeroPadding1D

from bi_lstm import padded_train_sequences as train_data_bi_lstm, train_labels, \
                    padded_test_sequences as test_data_bi_lstm, test_labels as test_labels_bi_lstm
from cnn import padded_train_sequences as train_data_cnn, padded_test_sequences as test_data_cnn
from crf import X_test as X_test_crf, y_test_flat

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

# Join the crf_model predictions to the bi-lstm and cnn predictions.
crf_model = joblib.load("saved_models/crf_model.joblib")

# Fit the MultiLabelBinarizer instance
mlb = MultiLabelBinarizer()
y_test_binary = mlb.fit_transform(y_test_flat)

y_pred_bi_lstm_cnn = ensemble_model.predict([test_data_bi_lstm, test_data_bi_lstm])

# Get evaluation metrics for CRF
y_pred_crf = crf_model.predict(X_test_crf)

# Flatten the nested lists
y_pred_flat = [label for sublist in y_pred_crf for label in sublist]
y_pred_binary = mlb.transform(y_pred_flat)

# Compute the micro-averaged F1 score, Hamming Loss, and Jaccard Similarity Score
micro_f1 = f1_score(y_test_binary, y_pred_binary, average='micro')
hamming_loss_value = hamming_loss(y_test_binary, y_pred_binary)
jaccard_score_value = jaccard_score(y_test_binary, y_pred_binary, average='samples')

print("Micro-Averaged F1 Score:", micro_f1)
print("Hamming Loss:", hamming_loss_value)
print("Jaccard Similarity Score:", jaccard_score_value)

# Finally, evaluate the ensemble model on your test data
loss_bi_lstm_cnn, accuracy_bi_lstm_cnn = ensemble_model.evaluate([test_data_bi_lstm, test_data_cnn], test_labels_bi_lstm)
print(f'Test Loss: {loss_bi_lstm_cnn:.4f}, Test Accuracy: {accuracy_bi_lstm_cnn:.4f}')

# Find the maximum length
max_length = max(len(seq) for seq in y_pred_crf)

# Pad each list to the maximum length
padded_crf = [np.pad(seq, (0, max_length - len(seq)))[:, np.newaxis] for seq in y_pred_crf]

# Convert the list of lists to a NumPy array
y_pred_crf = np.array(padded_crf)
# Combine predictions from all models
y_pred_crf = np.array(y_pred_crf)

ensemble_predictions = np.concatenate([y_pred_bi_lstm_cnn, y_pred_crf], axis=1)

# Combine and print overall evaluation summary
combined_accuracy = (accuracy_bi_lstm_cnn + micro_f1) / 2
combined_loss = (loss_bi_lstm_cnn + hamming_loss_value) / 2

print(f'Combined Accuracy: {combined_accuracy:.4f}')
print(f'Combined Loss: {combined_loss:.4f}')
