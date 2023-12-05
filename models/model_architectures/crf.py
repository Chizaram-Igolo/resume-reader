import joblib
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

from prepare_model_input import prepare_crf_input

# Retrieve training data and labels and testing data and labels
X_train, y_train, X_test, y_test = prepare_crf_input()

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

    model.fit(X_train, y_train)
    joblib.dump(model, "saved_models/crf_model.joblib")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    y_pred_flat = [label for sublist in y_pred for label in sublist]
    y_pred_binary = mlb.transform(y_pred_flat)

    # print(classification_report(y_test_binary, y_pred_binary, target_names=mlb.classes_))

    # Print classification report
    # print(classification_report(y_test, y_pred))

    # Convert predictions and true labels to binary format
    # y_test_binary = mlb.transform(y_test)
    # y_pred_binary = mlb.transform(y_pred)

    # Compute the micro-averaged F1 score
    micro_f1 = f1_score(y_test_binary, y_pred_binary, average='micro')

    # Compute Hamming Loss
    hamming_loss_value = hamming_loss(y_test_binary, y_pred_binary)

    # Compute Jaccard Similarity Score
    jaccard_score_value = jaccard_score(y_test_binary, y_pred_binary, average='samples')

    print("Micro-Averaged F1 Score:", micro_f1)
    print("Hamming Loss:", hamming_loss_value)
    print("Jaccard Similarity Score:", jaccard_score_value)
