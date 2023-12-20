import os
import pandas as pd

from utilities.constants import resume_folder_path
from utilities.read_files import extract_text_from_pdf, extract_text_from_doc

# Directory path where resume files are stored


# List to store data for DataFrame
data = {'text': [], 'label': []}

current_dir = os.path.dirname(__file__)
current_dir = current_dir if current_dir != '' else '.'

parent_dir = os.path.dirname(current_dir)

# directory to scan for any pdf and docx files
data_dir_path = parent_dir + resume_folder_path

# Traverse through files in the folder
for root, dirs, files in os.walk(data_dir_path):
    for file in files:
        text = ""

        file_path = os.path.join(root, file)

        # Extract file extension
        _, file_extension = os.path.splitext(file)

        # Extract text based on file extension
        if file_extension.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)[0]
        elif file_extension.lower() == '.docx':
            text = extract_text_from_doc(file_path)[0]

        text = text.strip()

        if text == "":
            continue

        # Determine label from the folder name
        label = os.path.basename(root)

        # Append data to the list
        data['text'].append(text)
        data['label'].append(label)

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('./classifiers/resume_data.csv', index=False)
