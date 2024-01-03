import sys
import os
from tqdm import tqdm
from parser_and_sorter.parser_helpers import ResumeParser
from utilities.constants import resume_folder_path
from utilities.read_files import read_pdf_and_docx


def get_next_output_filename(output_file_path):
    base_name, ext = os.path.splitext(output_file_path)
    count = 1
    new_name = f"{base_name}_{count}{ext}"

    while os.path.exists(new_name):
        count += 1
        new_name = f"{base_name}_{count}{ext}"

    return new_name


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    parent_dir = os.path.dirname(current_dir)
    data_dir_path = parent_dir + resume_folder_path

    output_file_path = os.path.join(current_dir, 'parser_output.txt')

    # Get the next available filename
    output_file_path = get_next_output_filename(output_file_path)

    with open(output_file_path, 'w') as output_file:
        collected = read_pdf_and_docx(data_dir_path, command_logging=True)

        for file_path, file_content in tqdm(collected.items(), desc="Parsing Files", unit="file"):
            print('parsing file: ', file_path, file=output_file)

            parser = ResumeParser(file_content)
            parser.parse()

            if parser.unknown is False:
                print(parser.summary(), file=output_file)

            print('++++++++++++++++++++++++++++++++++++++++++', file=output_file)

        print('\ncount: ', len(collected), file=output_file)


if __name__ == '__main__':
    main()
