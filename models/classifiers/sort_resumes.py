import os
from statistics import mode

import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import MultiLabelBinarizer

from models.classifiers.prepare_model_input import prepare_crf_input, prepare_bilstm_cnn_input
from models.extract_text_from_files_to_csv import extract_text_from_files_to_csv

prepare_bilstm_cnn_input('./resume_data.csv')

mlb = MultiLabelBinarizer()

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 30)  # Set maximum column width
# Set display width to fit the console
pd.set_option('display.width', pd.get_option('display.width'))
# pd.set_option('display.max_colwidth', None)

# Load models
bi_model = load_model("saved_models/bi_lstm_model.keras")
cnn_model = load_model("saved_models/cnn_model.keras")
crf_model = joblib.load("saved_models/crf_model.joblib")
ensemble_model = load_model("saved_models/ensemble_model.keras")

# Load tokenizer
with open('tokenizer_config.json', 'r') as json_file:
    loaded_tokenizer = tokenizer_from_json(json_file.read())

# List to store data for DataFrame
data = {'text': []}
current_dir = os.path.dirname(__file__)
current_dir = current_dir if current_dir != '' else '.'
# directory to scan for any pdf and docx files
data_dir_path = current_dir + '/resume_files_to_test'
df = extract_text_from_files_to_csv(data_dir_path, False)
# Save DataFrame to CSV
df.to_csv('./test_data.csv', index=False, escapechar='\\')
data = pd.read_csv('./test_data.csv')
texts = data['text']

# Process data for Bi-LSTM
bi_max_len = bi_model.layers[0].input_shape[1]
bi_test_sequences = loaded_tokenizer.texts_to_sequences(texts)
bi_padded_test_sequences = pad_sequences(bi_test_sequences, maxlen=bi_max_len, padding='post')
bi_predictions = bi_model.predict(bi_padded_test_sequences)
bi_sigmoid_predictions = 1 / (1 + np.exp(-bi_predictions.squeeze()))

# Process data for CNN
cnn_max_len = cnn_model.layers[0].input_shape[1]
cnn_test_sequences = loaded_tokenizer.texts_to_sequences(texts)
cnn_padded_test_sequences = pad_sequences(cnn_test_sequences, maxlen=cnn_max_len, padding='post')
cnn_predictions = cnn_model.predict(cnn_padded_test_sequences)
cnn_sigmoid_predictions = 1 / (1 + np.exp(-cnn_predictions.squeeze()))

# Process data for Ensemble
ensemble_max_len = ensemble_model.layers[0].input_shape[0][1]
ensemble_test_sequences = loaded_tokenizer.texts_to_sequences(texts)
ensemble_padded_test_sequences = pad_sequences(ensemble_test_sequences, maxlen=cnn_max_len, padding='post')
ensemble_predictions = cnn_model.predict(ensemble_padded_test_sequences)
ensemble_sigmoid_predictions = 1 / (1 + np.exp(-ensemble_predictions.squeeze()))

# Process data for CRF
X_test_crf = prepare_crf_input('./test_data.csv')
crf_predictions = crf_model.predict(X_test_crf)  # Flatten the CRF predictions
crf_sequence_labeling_predictions = [label for sublist in crf_predictions for label in sublist]
# Convert labels to binary array for each sublist
binary_predictions = [[1 if label.startswith('B') or label.startswith('I') else 0 for label in sublist] for sublist in
                      crf_predictions]
crf_results = [mode(sublist) for sublist in binary_predictions]

bi_results = []
cnn_results = []
ensemble_results = []

for i, j in zip(bi_sigmoid_predictions, cnn_sigmoid_predictions):
    bi_val = 1 if round(np.mean(i), 2) >= 0.7 else 0
    cnn_val = 1 if round(np.mean(j), 2) >= 0.7 else 0

    bi_results.append(bi_val)
    cnn_results.append(cnn_val)

for idx, k in enumerate(crf_results):
    ensemble_val = 1 if round(((ensemble_sigmoid_predictions[idx] * 2) + k) / 3, 2) >= 0.7 else 0
    ensemble_results.append(ensemble_val)

# # Print results in a tabular form
results_df = pd.DataFrame({
    'File': data['file'],
    'Bi-LSTM': bi_results,
    'CNN': cnn_results,
    'CRF': crf_results,
    'Ensemble': ensemble_results
})

# Add a column for numbering starting from 1
results_df.index = results_df.index + 1

# Align 'File' column to the left
results_df.style.set_properties(**{'File': 'text-align: left'})

# Write DataFrame to a CSV file
results_df.to_csv('test_predictions.csv', index=False)

print(results_df)
