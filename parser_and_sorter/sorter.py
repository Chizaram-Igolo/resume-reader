import sys
import os
import shutil
import threading
import time
from sklearn.model_selection import train_test_split

import spacy
from spacy.matcher import PhraseMatcher

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from parser_and_sorter.sorter_helpers import ResumeSorter
from utilities.constants import unsorted_folder_path, sorted_folder_path, resume_folder_path
from utilities.read_files import read_pdf_and_docx
from utilities.soft_skills import soft_skill_keywords

# Check if en_core_web_md is installed
if not spacy.util.is_package("en_core_web_md"):
    # Install the package if it's not installed
    spacy.cli.download("en_core_web_md")

# Load the model
nlp = spacy.load("en_core_web_md")

# Initialize skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


def sort_and_move(file_path, file_content, passed_nlp, passed_skill_extractor, passed_soft_skill_keywords,
                  destination_folder):
    try:
        sorter = ResumeSorter(file_content, passed_nlp, passed_skill_extractor, passed_soft_skill_keywords)

        # Measure the start time
        start_time = time.time()

        sorter.sort()

        # Measure the end time
        end_time = time.time()

        elapsed_time = end_time - start_time

        # If sorting took longer than 60 seconds, move the file to 'sorted_folder_0'
        if elapsed_time > 60:
            print(f"Sorting took longer than 60 seconds for file: {file_path}")
            destination_folder = os.path.join(destination_folder, 'testing_data')  # Move the file to testing data
        else:
            destination_folder = os.path.join(destination_folder, 'training_data')  # Move the file to training data

        # Construct the new file path in the destination folder
        new_file_path = os.path.join(destination_folder, os.path.basename(file_path))

        # Move the file to the destination folder
        shutil.move(file_path, new_file_path)

        print(f"Resume Score: {sorter.get_score()}%")
    except Exception as e:
        print(f"An error occurred for file: {file_path}, Error: {e}")


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    parent_dir = os.path.dirname(current_dir)

    # Directory to scan for any pdf and docx files
    data_dir_path = current_dir + unsorted_folder_path

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)

    # Create destination folder
    resume_files_folder = os.path.join(parent_dir, resume_folder_path)
    os.makedirs(resume_files_folder, exist_ok=True)

    # Create training and testing folders
    training_data_folder = os.path.join(resume_files_folder, 'training_data')
    testing_data_folder = os.path.join(resume_files_folder, 'testing_data')
    os.makedirs(training_data_folder, exist_ok=True)
    os.makedirs(testing_data_folder, exist_ok=True)

    for folder in ['0', '1']:
        folder_path = os.path.join(sorted_folder_path, folder)
        files_in_folder = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

        # Split files into training and testing sets
        train_files, test_files = train_test_split(files_in_folder, test_size=0.2, random_state=42)

        # Move training files
        for file_name in train_files:
            file_path = os.path.join(folder_path, file_name)
            sort_and_move(file_path, collected[file_name], nlp, skill_extractor, soft_skill_keywords,
                          training_data_folder)

        # Move testing files
        for file_name in test_files:
            file_path = os.path.join(folder_path, file_name)
            sort_and_move(file_path, collected[file_name], nlp, skill_extractor, soft_skill_keywords,
                          testing_data_folder)

    print('\nCount: ', len(collected))


if __name__ == '__main__':
    main()
