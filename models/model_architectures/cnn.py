import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

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

# Tokenize the text data and create a vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token

# Convert text to sequences and pad them to a fixed length
train_sequences = tokenizer.texts_to_sequences(train_data)
max_sequence_length = max([len(seq) for seq in train_sequences])
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

# Define the CNN model
embedding_dim = 50  # You can adjust this based on your specific task and vocabulary size
filter_sizes = [3, 4, 5]  # Convolutional filters of different sizes
num_filters = 128  # Number of filters in each convolutional layer

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Conv1D(filters=num_filters, kernel_size=filter_sizes[0], activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')  # Adjust the number of units based on your task
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_train_sequences, train_labels, epochs=10, batch_size=64)

# Evaluate the model on the test set
test_sequences = tokenizer.texts_to_sequences(test_data)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')

# Ensure labels are reshaped to match the output shape of the model
test_labels = np.expand_dims(test_labels, axis=-1)

loss, accuracy = model.evaluate(padded_test_sequences, test_labels)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
