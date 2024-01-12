import json
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Check if en_core_web_md is installed
if not spacy.util.is_package("en_core_web_md"):
    # Install the package if it's not installed
    spacy.cli.download("en_core_web_md")

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_md")


def prepare_bilstm_cnn_input(csv_file_path):
    # Load data from CSV into a DataFrame, discarding rows with NaN values
    df = pd.read_csv(csv_file_path).dropna(subset=['label'])

    # Convert 'label' to numeric, discard rows with non-numeric values
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])

    # Filter rows where 'label' contains only '0' or '1'
    df = df[df['label'].isin([0, 1])]

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        df['text'],
        df['label'].astype(int),  # Ensure labels are of integer type
        test_size=0.2,
        random_state=42
    )

    # Tokenize the text data and create a vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)

    # Convert the tokenizer to a JSON-formatted string
    tokenizer_json = tokenizer.to_json()

    # Save the JSON string to a file (optional)
    with open('tokenizer_config.json', 'w') as json_file:
        json_file.write(tokenizer_json)

    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token

    # Convert text to sequences and pad them to a fixed length
    train_sequences = tokenizer.texts_to_sequences(train_data)
    max_sequence_length = max([len(seq) for seq in train_sequences])
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

    return [train_data, test_data, train_labels, test_labels, tokenizer, vocab_size, train_sequences,
            max_sequence_length, padded_train_sequences]


# def prepare_crf_input(csv_file_path):
#     # Feature extraction functions
#     def word2features(sent, i):
#         word = sent[i][0]
#         features = {
#             'bias': 1.0,
#             'word.lower()': word.lower(),
#             'word[-3:]': word[-3:],
#             'word[-2:]': word[-2:],
#             'word.isupper()': word.isupper(),
#             'word.istitle()': word.istitle(),
#             'word.isdigit()': word.isdigit(),
#             'suffix_3': word[-3:],
#             'suffix_2': word[-2:],
#             'prefix_3': word[:3],
#             'prefix_2': word[:2],
#             'length': len(word),
#         }
#         if i > 0:
#             word1 = sent[i - 1][0]
#             features.update({
#                 '-1:word.lower()': word1.lower(),
#                 '-1:word.istitle()': word1.istitle(),
#                 '-1:word.isupper()': word1.isupper(),
#             })
#         else:
#             features['BOS'] = True  # Beginning of sentence
#
#         if i < len(sent) - 1:
#             word1 = sent[i + 1][0]
#             features.update({
#                 '+1:word.lower()': word1.lower(),
#                 '+1:word.istitle()': word1.istitle(),
#                 '+1:word.isupper()': word1.isupper(),
#             })
#         else:
#             features['EOS'] = True  # End of sentence
#
#         return features
#
#     def sent2features(sent):
#         return [word2features(sent, i) for i in range(len(sent))]
#
#     def sent2labels(sent):
#         return [label for word, label in sent]
#
#     def sent2tokens(sent):
#         return [word for word, label in sent]
#
#     # Load data from CSV into a DataFrame, discarding rows with NaN values
#     df = pd.read_csv(csv_file_path)
#
#     if 'label' in df.columns:
#
#         # Convert 'label' to numeric, discard rows with non-numeric values
#         df['label'] = pd.to_numeric(df['label'], errors='coerce')
#         df = df.dropna(subset=['label'])
#
#         # Filter rows where 'label' contains only '0' or '1'
#         df = df[df['label'].isin([0, 1])]
#
#         # Shuffle the DataFrame
#         df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
#         # Split the data into training and test sets
#         train_data, test_data, train_labels, test_labels = train_test_split(
#             df['text'],
#             df['label'].astype(int),  # Ensure labels are of integer type
#             test_size=0.2,
#             random_state=42
#         )
#
#         # Convert data to BIO format (beginning, inside, outside)
#         def bio_tagging(text, labels):
#             words = text.split()
#
#             if not isinstance(labels, (list, tuple)):
#                 # If labels is a single integer, convert it to a list
#                 labels = [labels] * len(words)
#
#             tags = []
#             for word, label in zip(words, labels):
#                 if len(word) == 1:
#                     tags.append((word, 'S-' + str(label)))
#                 else:
#                     tags.append((word[0], 'B-' + str(label)))
#                     for char in word[1:-1]:
#                         tags.append((char, 'I-' + str(label)))
#                     tags.append((word[-1], 'E-' + str(label)))
#             return tags
#
#         # print(train_data)
#
#         # Convert training data to BIO format
#         train_bio_data = [bio_tagging(text, labels) for text, labels in zip(train_data, train_labels)]
#
#         # Convert test data to BIO format
#         test_bio_data = [bio_tagging(text, labels) for text, labels in zip(test_data, test_labels)]
#
#         # Extract features from training and test data
#         X_train = [sent2features(sent) for sent in train_bio_data]
#         y_train = [sent2labels(sent) for sent in train_bio_data]
#
#         X_test = [sent2features(sent) for sent in test_bio_data]
#         y_test = [sent2labels(sent) for sent in test_bio_data]
#
#         return [X_train, y_train, X_test, y_test]
#     else:
#         def bio_tagging(text):
#             words = text.split()
#
#             tags = []
#             for word in words:
#                 if len(word) == 1:
#                     tags.append((word, 'S-O'))
#                 else:
#                     tags.append((word[0], 'B-O'))
#                     for char in word[1:-1]:
#                         tags.append((char, 'I-O'))
#                     tags.append((word[-1], 'E-O'))
#             return tags
#
#         # Convert data to BIO format
#         bio_data = [bio_tagging(text) for text in df['text']]
#
#         # Extract features from the entire dataset
#         features = [sent2features(sent) for sent in bio_data]
#
#         return features  # No test set when 'label' column is not present


def prepare_crf_input(csv_file_path):
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

    # Convert training data to BIO format using spaCy
    def bio_tagging_spacy(text, labels=None):
        doc = nlp(text)
        tags = []
        if labels is not None:
            words = text.split()

            for word, label in zip(words, labels):
                if len(word) == 1:
                    tags.append((word, 'S-' + str(label)))
                else:
                    tags.append((word[0], 'B-' + str(label)))
                    for char in word[1:-1]:
                        tags.append((char, 'I-' + str(label)))
                    tags.append((word[-1], 'E-' + str(label)))

            for ent in doc.ents:
                if len(ent) == 1:
                    tags.append((ent.text, 'S-' + ent.label_))
                else:
                    tags.append((ent.text[0], 'B-' + ent.label_))
                    for char in ent.text[1:-1]:
                        tags.append((char, 'I-' + ent.label_))
                    tags.append((ent.text[-1], 'E-' + ent.label_))
            for word in doc:
                if not any(w[0] == word.text for w in tags):
                    tags.append((word.text, 'O'))
        else:
            # If no labels provided, use NER tags
            for ent in doc.ents:
                if len(ent) == 1:
                    tags.append((ent.text, 'S-' + ent.label_))
                else:
                    tags.append((ent.text[0], 'B-' + ent.label_))
                    for char in ent.text[1:-1]:
                        tags.append((char, 'I-' + ent.label_))
                    tags.append((ent.text[-1], 'E-' + ent.label_))
            for word in doc:
                if not any(w[0] == word.text for w in tags):
                    tags.append((word.text, 'O'))

        return tags

    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return [label for word, label in sent]

    def sent2tokens(sent):
        return [word for word, label in sent]

    # Load data from CSV into a DataFrame, discarding rows with NaN values
    df = pd.read_csv(csv_file_path)

    if 'label' in df.columns:
        # Convert 'label' to numeric, discard rows with non-numeric values
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])

        # Filter rows where 'label' contains only '0' or '1'
        df = df[df['label'].isin([0, 1])]

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the data into training and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            df['text'],
            df['label'].astype(int),  # Ensure labels are of integer type
            test_size=0.2,
            random_state=42
        )

        # Convert training data to BIO format using spaCy
        train_bio_data_spacy = [bio_tagging_spacy(text, labels) for text, labels in zip(train_data, train_labels)]

        # Convert test data to BIO format using spaCy
        test_bio_data_spacy = [bio_tagging_spacy(text, labels) for text, labels in zip(test_data, test_labels)]

        # Extract features from training and test data
        X_train = [sent2features(sent) for sent in train_bio_data_spacy]
        y_train = [sent2labels(sent) for sent in train_bio_data_spacy]

        X_test = [sent2features(sent) for sent in test_bio_data_spacy]
        y_test = [sent2labels(sent) for sent in test_bio_data_spacy]

        return [X_train, y_train, X_test, y_test]
    else:
        # Convert data to BIO format using spaCy
        bio_data_spacy = [bio_tagging_spacy(text) for text in df['text']]

        # Extract features from the entire dataset
        features = [sent2features(sent) for sent in bio_data_spacy]

        return features  # No test set when 'label' column is not present
