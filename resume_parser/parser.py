import sys
import os

from constants.constants import rel_resume_file_path
from resume_parser.parser_helpers import ResumeParser
from utilities.read_files import read_pdf_and_docx


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    parent_dir = os.path.dirname(current_dir)

    # directory to scan for any pdf and docx files
    data_dir_path = parent_dir + rel_resume_file_path

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)
    for file_path, file_content in collected.items():
        print('parsing file: ', file_path)

        parser = ResumeParser(file_content)
        parser.parse()

        if parser.unknown is False:
            print(parser.summary())

        print('++++++++++++++++++++++++++++++++++++++++++')

    print('\ncount: ', len(collected))


if __name__ == '__main__':
    main()
