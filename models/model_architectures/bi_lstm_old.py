import pandas as pd
import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load data from CSV into a DataFrame
csv_file_path = './training_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path).dropna(subset=['train_label'])

df['train_label'] = pd.to_numeric(df['train_label'], errors='coerce')
df = df.dropna(subset=['train_label'])

df[df['train_label'].isin([0, 1])]

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['resume_text'],
    df['train_label'].astype(int),
    test_size=0.2,  # Adjust the test_size as needed
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

# Define the BiLSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
train_labels = np.array(train_labels)
# train_labels = to_categorical(train_labels, num_classes=2)

model.fit(padded_train_sequences, train_labels, epochs=10, batch_size=1)

# Evaluate the model on the test set
test_sequences = tokenizer.texts_to_sequences(test_data)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')

test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1, 1)
# test_labels = to_categorical(test_labels, num_classes=2)

print("Please wait...")
loss, accuracy = model.evaluate(padded_test_sequences, test_labels)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

