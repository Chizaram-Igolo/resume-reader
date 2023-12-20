import sys
import os
from tqdm import tqdm  # Import tqdm for the progress bar

from parser_and_sorter.parser_helpers import ResumeParser
from utilities.constants import resume_folder_path
from utilities.read_files import read_pdf_and_docx


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    parent_dir = os.path.dirname(current_dir)

    # directory to scan for any pdf and docx files
    data_dir_path = parent_dir + resume_folder_path

    output_file_path = os.path.join(current_dir, 'parser_output.txt')

    with open(output_file_path, 'w') as output_file:

        collected = read_pdf_and_docx(data_dir_path, command_logging=True)

        # Use tqdm to display a progress bar
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
