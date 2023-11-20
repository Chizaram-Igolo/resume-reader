import os
import pandas as pd
import numpy as np

from lib.utils.read_files import read_pdf_and_docx

current_dir = os.path.dirname(__file__)
current_dir = current_dir if current_dir != '' else '.'

# directory to scan for any pdf and docx files
data_dir_paths = ["../../data/training_data/0", "../../data/training_data/1"]

csv_file_path = "./training_data.csv"


# Function to read text data from resume files in a folder
def read_resume_data(folder_path):
    resume_texts = []
    labels = []

    collected = read_pdf_and_docx(folder_path, command_logging=True)
    label = folder_path.split('/')[-1]

    for key, value in collected.items():
        resume_text = ' '.join(filter(None, value[0].split()))

        if resume_text != "":
            resume_texts.append(resume_text)
            labels.append(label)

    df = pd.DataFrame({'resume_text': resume_texts, 'train_label': labels})

    if os.path.exists(csv_file_path):
        try:
            df_exists = pd.read_csv(csv_file_path)

            # Check if the DataFrame is empty
            if df_exists.empty:
                df.to_csv(csv_file_path, index=False)
            else:
                df.to_csv(csv_file_path, mode='a', header=False, index=False)

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        df.to_csv(csv_file_path, index=False)


for ddp in data_dir_paths:
    read_resume_data(ddp)
