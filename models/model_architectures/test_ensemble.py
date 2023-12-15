import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from keras.models import load_model

from bi_lstm import padded_test_sequences as test_data_bi_lstm, test_labels as test_labels_bi_lstm
from cnn import padded_test_sequences as test_data_cnn
from crf import X_test as X_test_crf, y_test_flat

ensemble_model = load_model("saved_models/ensemble_model.keras")
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

