import sys
import os

import spacy
from spacy.matcher import Matcher, PhraseMatcher

import IPython

# load default skills data base
from skillNer.general_params import SKILL_DB

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

from lib.sorter import ResumeSorter
from lib.utils.read_files import read_pdf_and_docx
from constants.soft_skills import soft_skill_keywords

nlp = spacy.load('en_core_web_md')

# Initialize skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir != '' else '.'

    # directory to scan for any pdf and docx files
    data_dir_path = current_dir + '/data/ab'

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)
    for file_path, file_content in collected.items():
        print('sorting file: ', file_path)

        sorter = ResumeSorter(nlp, skill_extractor, soft_skill_keywords)
        sorter.sort(file_content)
        print(f"Resume Score: {sorter.get_score()}%")

        print('++++++++++++++++++++++++++++++++++++++++++')

    print('\ncount: ', len(collected))


if __name__ == '__main__':
    main()
