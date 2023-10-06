import sys
import os

from lib.parser import ResumeParser
from lib.utils.read_files import read_pdf_and_docx


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    # directory to scan for any pdf and docx files
    data_dir_path = current_dir + '/data/resume_data'

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)
    for file_path, file_content in collected.items():
        print('parsing file: ', file_path)

        parser = ResumeParser()
        parser.parse(file_content)

        if parser.unknown is False:
            print(parser.summary())

        print('++++++++++++++++++++++++++++++++++++++++++')

    print('\ncount: ', len(collected))


if __name__ == '__main__':
    main()
