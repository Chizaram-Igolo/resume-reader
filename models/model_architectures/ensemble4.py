import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from keras.models import Model, load_model
from keras.layers import Dense, Concatenate, concatenate, Input
import tf2crf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten

from bi_lstm import padded_test_sequences as padded_test_sequences_bi_lstm
from cnn import padded_test_sequences as padded_test_sequences_cnn
from crf import X_test as X_test_crf

bi_lstm_model = load_model("saved_models/bi_lstm_model.h5")
cnn_model = load_model("saved_models/cnn_model.h5")
crf_model = joblib.load("saved_models/crf_model.joblib")


# Create the ensemble by combining individual models
def create_ensemble(models):

    # Remove the output layer of each model except CRF
    models_no_output = [Model(model.input, model.layers[-2].output) for model in models[:-1]]

    print(concatenate([model.output for model in models_no_output]))
    print(Concatenate([model.output for model in models_no_output]))
    print(Concatenate((model.output for model in models_no_output)))
    print([model.output for model in models_no_output])

    # Concatenate the outputs of the models
    concatenated = concatenate([model.output for model in models_no_output])

    # Add a dense layer for classification
    dense = Dense(1, activation='sigmoid')(concatenated)

    # Create the ensemble model
    model = Model(inputs=[model.input for model in models[:-1]], outputs=dense)

    # Compile the ensemble model if needed
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Combine the models into an ensemble
ensemble_models = [bi_lstm_model, cnn_model]
ensemble_model = create_ensemble(ensemble_models)

y_pred_bi_lstm_cnn = ensemble_model.predict([padded_test_sequences_bi_lstm, padded_test_sequences_cnn])

# Combine predictions for CRF
y_pred_crf = crf_model.predict(X_test_crf)

# Combine predictions from all models
ensemble_predictions = np.concatenate([y_pred_bi_lstm_cnn, y_pred_crf], axis=1)

# Flatten the nested lists
y_test_flat = [label for sublist in y_pred_crf for label in sublist]
y_pred_flat = [label for sublist in ensemble_predictions.tolist() for label in sublist]

mlb = MultiLabelBinarizer()
y_test_binary = mlb.fit_transform(y_test_flat)
y_pred_binary = mlb.transform(y_pred_flat)

# Compute the micro-averaged F1 score
micro_f1 = f1_score(y_test_binary, y_pred_binary, average='micro')

# Compute Hamming Loss
hamming_loss_value = hamming_loss(y_test_binary, y_pred_binary)

# Compute Jaccard Similarity Score
jaccard_score_value = jaccard_score(y_test_binary, y_pred_binary, average='samples')

print("Micro-Averaged F1 Score:", micro_f1)
print("Hamming Loss:", hamming_loss_value)
print("Jaccard Similarity Score:", jaccard_score_value)
