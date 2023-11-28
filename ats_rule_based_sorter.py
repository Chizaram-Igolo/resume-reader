import sys
import os
import shutil
import threading  # Import the threading module
import time

import spacy
from spacy.matcher import Matcher, PhraseMatcher

import IPython

# load default skills data base
from skillNer.general_params import SKILL_DB

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

from models.sorter import ResumeSorter
from models.utils.read_files import read_pdf_and_docx
from constants.soft_skills import soft_skill_keywords

nlp = spacy.load('en_core_web_md')

# Initialize skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


def sort_and_move(file_path, file_content, passed_nlp, passed_skill_extractor, passed_soft_skill_keywords,
                  destination_folder_1, destination_folder_0):
    try:
        sorter = ResumeSorter(file_content, passed_nlp, passed_skill_extractor, passed_soft_skill_keywords)

        # Measure the start time
        start_time = time.time()

        sorter.sort()

        # Measure the end time
        end_time = time.time()

        elapsed_time = end_time - start_time

        # Check the sorter score and move the file accordingly
        if sorter.get_score() >= 50:
            destination_folder = destination_folder_1
        else:
            destination_folder = destination_folder_0

        # If sorting took longer than 60 seconds, move the file to 'sorted_folder_0'
        if elapsed_time > 60:
            print(f"Sorting took longer than 60 seconds for file: {file_path}")
            destination_folder = destination_folder_0

        # Construct the new file path in the destination folder
        new_file_path = os.path.join(destination_folder, os.path.basename(file_path))

        # Move the file to the destination folder
        shutil.move(file_path, new_file_path)

        print(f"Resume Score: {sorter.get_score()}%")
    except Exception as e:
        print(f"An error occurred for file: {file_path}, Error: {e}")


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    # Directory to scan for any pdf and docx files
    data_dir_path = current_dir + '/data/resume_data'

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)

    # Define destination folders
    sorted_folder_1 = current_dir + '/data/sorted/1'
    sorted_folder_0 = current_dir + '/data/sorted/0'

    # Create destination folders if they don't exist
    os.makedirs(sorted_folder_1, exist_ok=True)
    os.makedirs(sorted_folder_0, exist_ok=True)

    for file_path, file_content in collected.items():
        print('sorting file: ', file_path)

        # Create a separate thread to run the sorting operation with a timeout
        sort_thread = threading.Thread(target=sort_and_move, args=(
            file_path, file_content, nlp, skill_extractor, soft_skill_keywords, sorted_folder_1, sorted_folder_0))
        sort_thread.start()
        sort_thread.join(timeout=60)  # Wait for the thread to finish or timeout

        # If the thread is still alive, it means sorting took longer than 60 seconds
        if sort_thread.is_alive():
            print(f"Sorting took longer than 60 seconds for file: {file_path}")
            destination_folder = sorted_folder_0  # Move the file to folder 0 in case of timeout

            # Construct the new file path in the destination folder
            new_file_path = os.path.join(destination_folder, os.path.basename(file_path))

            # Move the file to the destination folder
            shutil.move(file_path, new_file_path)

        print('++++++++++++++++++++++++++++++++++++++++++')

    print('\ncount: ', len(collected))


if __name__ == '__main__':
    main()
