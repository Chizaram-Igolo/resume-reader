import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load data from CSV into a DataFrame, discarding rows with NaN values
csv_file_path = './training_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path).dropna(subset=['train_label'])

# Convert 'train_label' to numeric, discard rows with non-numeric values
df['train_label'] = pd.to_numeric(df['train_label'], errors='coerce')
df = df.dropna(subset=['train_label'])

# Filter rows where 'train_label' contains only '0' or '1'
df = df[df['train_label'].isin([0, 1])]

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['resume_text'],
    df['train_label'].astype(int),  # Ensure labels are of integer type
    test_size=0.2,
    random_state=42
)


# Convert data to BIO format (beginning, inside, outside)
def bio_tagging(text, labels):
    words = text.split()

    if not isinstance(labels, (list, tuple)):
        # If labels is a single integer, convert it to a list
        labels = [labels] * len(words)

    tags = []
    for word, label in zip(words, labels):
        if len(word) == 1:
            tags.append((word, 'S-' + str(label)))
        else:
            tags.append((word[0], 'B-' + str(label)))
            for char in word[1:-1]:
                tags.append((char, 'I-' + str(label)))
            tags.append((word[-1], 'E-' + str(label)))
    return tags


# Convert training data to BIO format
train_bio_data = [bio_tagging(text, labels) for text, labels in zip(train_data, train_labels)]

# Convert test data to BIO format
test_bio_data = [bio_tagging(text, labels) for text, labels in zip(test_data, test_labels)]


# Feature extraction functions
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'suffix_3': word[-3:],
        'suffix_2': word[-2:],
        'prefix_3': word[:3],
        'prefix_2': word[:2],
        'length': len(word),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for word, label in sent]


def sent2tokens(sent):
    return [word for word, label in sent]


# Extract features from training and test data
X_train = [sent2features(sent) for sent in train_bio_data]
y_train = [sent2labels(sent) for sent in train_bio_data]

X_test = [sent2features(sent) for sent in test_bio_data]
y_test = [sent2labels(sent) for sent in test_bio_data]

# Train the CRF model
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
)

crf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = crf.predict(X_test)

# Flatten the nested lists
y_test_flat = [label for sublist in y_test for label in sublist]
y_pred_flat = [label for sublist in y_pred for label in sublist]

mlb = MultiLabelBinarizer()
# y_test_binary = mlb.fit_transform(y_test_flat)
# y_pred_binary = mlb.transform(y_pred_flat)

# print(classification_report(y_test_binary, y_pred_binary, target_names=mlb.classes_))

# Print classification report
# print(classification_report(y_test, y_pred))

# Convert predictions and true labels to binary format
y_test_binary = mlb.transform(y_test)
y_pred_binary = mlb.transform(y_pred)

# Compute the micro-averaged F1 score
micro_f1 = f1_score(y_test_binary, y_pred_binary, average='micro')

# Compute Hamming Loss
hamming_loss_value = hamming_loss(y_test_binary, y_pred_binary)

# Compute Jaccard Similarity Score
jaccard_score_value = jaccard_score(y_test_binary, y_pred_binary, average='samples')

print("Micro-Averaged F1 Score:", micro_f1)
print("Hamming Loss:", hamming_loss_value)
print("Jaccard Similarity Score:", jaccard_score_value)
