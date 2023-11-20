import re

from nltk.corpus import stopwords

import spacy
from spacy.matcher import Matcher, PhraseMatcher

import IPython

# load default skills data base
from skillNer.general_params import SKILL_DB

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

from docx import Document

# load pre-trained model
nlp = spacy.load('en_core_web_md')

# Initialize skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [e.lower() for e in [
    'BE', 'B.E.', 'B.Eng', 'BEng', 'B.Sc',
    'BSc', 'BS', 'B.S', 'C.A.', 'c.a.', 'B.Com', 'B. Com', 'M. Com', 'M.Com', 'M. Com .',
    'ME', 'M.E', 'M.E.', 'MS', 'M.S',
    'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
    'PHD', 'phd', 'ph.d', 'Ph.D.', 'MBA', 'mba', 'graduate', 'post-graduate', '5 year integrated masters', 'masters',
    'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
]]


def count_docx_pages(docx_file):
    doc = Document(docx_file)
    section_count = len(doc.sections)
    return section_count


page_count = count_docx_pages("../data/resume_data/2.docx")
print(f"The DOCX has {page_count} pages.")


def extract_name(resume_text):
    # Initialize matcher with a vocabulary
    matcher = Matcher(nlp.vocab)

    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [{"POS": "PROPN"}, {"POS": "PROPN"}]

    matcher.add("NAME", patterns=[pattern])

    matches = matcher(nlp_text)

    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text


def extract_phone_number(resume_text):
    matches = re.findall(re.compile(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,7}'), resume_text)
    phone_numbers = [p.replace(' ', '') for p in matches if p]

    return phone_numbers


def extract_email(resume_text):
    # Initialize the Matcher
    matcher = Matcher(nlp.vocab)

    # Define a pattern for matching email addresses using a regular expression
    email_pattern = [
        {"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}"}},
    ]

    # Add the pattern to the Matcher
    matcher.add("EMAIL_ADDRESS", [email_pattern])

    # Process the text with SpaCy
    doc = nlp(resume_text)

    # Use the Matcher to find email addresses in the processed text
    matches = matcher(doc)

    # Extract and print the found email addresses
    for match_id, start, end in matches:
        email_address = doc[start:end].text
        return email_address


def extract_skills(resume_text):
    skills = []

    try:
        annotations = skill_extractor.annotate(resume_text)

        full_matches = annotations['results']['full_matches']
        ngram_scored = annotations['results']['ngram_scored']

        s1 = [s['doc_node_value'] for s in full_matches]
        s2 = [s['doc_node_value'] for s in ngram_scored]

        skills = s1 + s2
    except ValueError as e:
        pass

    return skills


def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    edu = {}

    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            # print(tex)
            if tex.lower() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    return list(edu.keys())
