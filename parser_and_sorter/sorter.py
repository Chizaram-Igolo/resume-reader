import sys
import os
import shutil
import random
import threading  # Import the threading module
import time

import spacy
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

from parser_and_sorter.sorter_helpers import ResumeSorter
from utilities.read_files import read_pdf_and_docx
from utilities.soft_skills import soft_skill_keywords
from utilities.constants import resume_folder_path, sorted_folder_path

nlp = spacy.load('en_core_web_md')

# Initialize skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


current_dir = os.path.dirname(__file__)
current_dir = current_dir if current_dir != '' else '.'

parent_dir = os.path.dirname(current_dir)
resume_files_dir_path = parent_dir + resume_folder_path


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

        # Check the resume_sorter score and move the file accordingly
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


def transfer_files():
    # Specify the source folders
    folder_0 = current_dir + sorted_folder_path + '/0'
    folder_1 = current_dir + sorted_folder_path + '/1'

    # Specify the destination folders
    training_folder_0 = resume_files_dir_path + '/training_data/0'
    testing_folder_0 = resume_files_dir_path + '/testing_data/0'
    training_folder_1 = resume_files_dir_path + '/training_data/1'
    testing_folder_1 = resume_files_dir_path + '/testing_data/1'

    # Create destination folders if they don't exist
    os.makedirs(training_folder_0, exist_ok=True)
    os.makedirs(testing_folder_0, exist_ok=True)
    os.makedirs(training_folder_1, exist_ok=True)
    os.makedirs(testing_folder_1, exist_ok=True)

    # Get the list of files in each source folder
    files_folder_0 = os.listdir(folder_0)
    files_folder_1 = os.listdir(folder_1)

    # Determine the smaller number of files
    min_files = min(len(files_folder_0), len(files_folder_1))

    files_folder_0 = files_folder_0[:min_files]
    files_folder_1 = files_folder_1[:min_files]

    # Shuffle the files in each folder
    random.shuffle(files_folder_0)
    random.shuffle(files_folder_1)

    # Split files into training and testing sets (80/20 split)
    split_0 = int(0.8 * min_files)
    split_1 = int(0.8 * min_files)

    training_files_0 = files_folder_0[:split_0]
    testing_files_0 = files_folder_0[split_0:]

    training_files_1 = files_folder_1[:split_1]
    testing_files_1 = files_folder_1[split_1:]

    # Move files to training and testing folders
    for filename in training_files_0:
        src_path = os.path.join(folder_0, filename)
        dest_path = os.path.join(training_folder_0, filename)
        shutil.move(src_path, dest_path)

    for filename in testing_files_0:
        src_path = os.path.join(folder_0, filename)
        dest_path = os.path.join(testing_folder_0, filename)
        shutil.move(src_path, dest_path)

    for filename in training_files_1:
        src_path = os.path.join(folder_1, filename)
        dest_path = os.path.join(training_folder_1, filename)
        shutil.move(src_path, dest_path)

    for filename in testing_files_1:
        src_path = os.path.join(folder_1, filename)
        dest_path = os.path.join(testing_folder_1, filename)
        shutil.move(src_path, dest_path)

    print("Files moved successfully.")


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    # Directory to scan for any pdf and docx files
    data_dir_path = current_dir + '/unsorted_resume_files'

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)

    # Define destination folders
    sorted_folder_1 = current_dir + '/temp_sorted_resume_files/1'
    sorted_folder_0 = current_dir + '/temp_sorted_resume_files/0'

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
    print('Transferring files to', resume_folder_path, 'folder, please wait.')

    transfer_files()

    print('Transfer complete')


if __name__ == '__main__':
    main()
