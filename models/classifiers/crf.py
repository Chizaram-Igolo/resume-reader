import joblib
import numpy as np
from sklearn_crfsuite import CRF
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score, log_loss
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from prepare_model_input import prepare_crf_input

# Retrieve training data and labels and testing data and labels
X_train, y_train, X_test, y_test = prepare_crf_input('./resume_data.csv')

print(len(X_train), len(y_train), len(X_test), len(y_test))

print(y_train)

# , y_train), len(X_test), len(y_test))

# Flatten the nested lists
y_test_flat = [label for sublist in y_test for label in sublist]

mlb = MultiLabelBinarizer()
y_test_binary = mlb.fit_transform(y_test_flat)


def build_crf_model():
    crf_model = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    return crf_model


if __name__ == "__main__":
    model = build_crf_model()

    # Fit the model with tqdm progress bar
    with tqdm(total=len(X_train), desc="Training CRF Model") as pbar:
        for x, y in zip(X_train, y_train):
            model.fit([x], [y])
            pbar.update(1)

    joblib.dump(model, "saved_models/crf_model.joblib")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    print(y_pred)

    y_pred_flat = [label for sublist in y_pred for label in sublist]
    y_pred_binary = mlb.transform(y_pred_flat)

    print(len(y_pred_binary[0]))
    print(len(y_pred_binary[6]))
    print(np.mean(y_pred_binary[0]))
    print(np.mean(y_pred_binary[1]))
    print(np.mean(y_pred_binary[2]))
    print(np.mean(y_pred_binary[3]))

    # Convert predictions and true labels to binary format
    accuracy_crf = accuracy_score(y_test_binary, y_pred_binary)

    loss_crf = log_loss(y_test_binary, y_pred_binary)

    print(f'CRF Accuracy: {accuracy_crf:.4f}')
    print(f'CRF Loss: {loss_crf:.4f}')

    # Compute the micro-averaged F1 score
    micro_f1 = f1_score(y_test_binary, y_pred_binary, average='micro')

    # Compute Hamming Loss
    hamming_loss_value = hamming_loss(y_test_binary, y_pred_binary)

    # Compute Jaccard Similarity Score
    jaccard_score_value = jaccard_score(y_test_binary, y_pred_binary, average='samples')

    print("Micro-Averaged F1 Score:", micro_f1)
    print("Hamming Loss:", hamming_loss_value)
    print("Jaccard Similarity Score:", jaccard_score_value)
