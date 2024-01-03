import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def prepare_bilstm_input():
    # Load data from CSV into a DataFrame, discarding rows with NaN values
    csv_file_path = './resume_data.csv'  # Replace with the actual path to your CSV file
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
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token

    # Convert text to sequences and pad them to a fixed length
    train_sequences = tokenizer.texts_to_sequences(train_data)
    max_sequence_length = max([len(seq) for seq in train_sequences])
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

    return [train_data, test_data, train_labels, test_labels, tokenizer, vocab_size, train_sequences,
            max_sequence_length, padded_train_sequences]


def prepare_cnn_input():
    # Load data from CSV into a DataFrame, discarding rows with NaN values
    csv_file_path = './resume_data.csv'  # Replace with the actual path to your CSV file
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
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token

    # Convert text to sequences and pad them to a fixed length
    train_sequences = tokenizer.texts_to_sequences(train_data)
    max_sequence_length = max([len(seq) for seq in train_sequences])
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

    return [train_data, test_data, train_labels, test_labels, tokenizer, vocab_size, train_sequences,
            max_sequence_length, padded_train_sequences]


def prepare_crf_input():
    # Load data from CSV into a DataFrame, discarding rows with NaN values
    csv_file_path = './resume_data.csv'  # Replace with the actual path to your CSV file
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

    return [X_train, y_train, X_test, y_test]
