import re
import spacy

from nltk import sent_tokenize

from constants.soft_skills import soft_skill_keywords

# Load the English language model
from models.parser_methods import extract_skills

from spellchecker import SpellChecker

import enchant

import Levenshtein as lev

from dateutil import parser

from models.utils.read_files import extract_text_from_doc, extract_text_from_pdf

nlp = spacy.load('en_core_web_md')


# print(extract_text_from_doc("../data/resume_data/CHUKWU OGBONNAYA IBIAM.docx")[0])

text = "Chizaram Igolo LinkedIn  |  +234 8083298124  |  Portfolio  |      igolo.chizaram@gmail.com  |  GitHub SKILLS •   React | Next.js | Javascript | TypeScript | jQuery | Node.js | Express | Redux | Tailwind CSS |  SASS | SCSS |  Git | •   GitHub | PHP | MySQL | PostgreSQL | NoSQL | Firebase | Heroku | Vercel | AWS | GCP  | Python  | Flask | NPM | •   Yarn  | REST APIs | Webpack  | React Query  | BootStrap  | Chrome DevTools | HTML | CSS |  React Context API | •   Frontend | Full-Stack  | Project Management | Communication | Leadership | Growth Mindset  | Scheduling WORK EXPERIENCE Scriben  (Remote , Contract )  Portugal Full-Stack Engineer  Oct 2022 – Jan 2023 ● Fixed a lock up issue  in Computer -Assisted Translation  (CAT) web application  that occurred when a task was created from a large translation sheet file by rewriting SQL inserts and transactions. ● Implemented  text replacement feature in  target  (translation ) column. ● Developed features for the quality assurance tool of CAT web applicat ion and ran DevOps for the software over an Amazon Web Service infrastructure. ● Recommended and used GitHub organization with read -write permissions  granted to members to grant access while maintaining repository privacy. ●     Drove technical projects from concept to completion as Subject Matter Expert (SME) for cloud infrastructure architecture and design. ●     Designed , tested, and implemented cloud technology, enterprise applications, and big data management soluti ons.  Dibia  Lugbe, Abuja Front -End Developer  Feb 2021 – July 2021 ● Built landing page  and features for a language learning platform  (soon -to-launch)  using Next.j s (React ), TypeScript , Tailwind CSS , Firebase  databases  (NoSQL databases) , and deploy ed on Vercel . ● Created  a fast-food web template using HTML /CSS/JavaScript . ● Improved  code coverage, quality, and stability by creating unit tests for all projects. ● Enabled insights into customer  usage, patterns, and habits by embedding Google Analytics into all projects.  Juvenix Limited  (Hybrid)   Jahi, Abuja Full-Stack  Engineer  Dec 2016 – Feb 2021 ● Boosted  a hotel ier’s business visibility and improved their operational and accounting processes by  co-developing a hotel reservation and management system  serving 1000s of guests yearly  with online payment  integration . ● Grew  the company’s tech human  resource s by training and retaining recruits  – 1 Python  developer and 2 JavaScript  developers . ● Enhanced  the company’s visibility and boosted business conversion rate by  150%  by building its landing page, and contact  emailing feature using PHP , MySQL , JavaScript , HTML , and CSS. ● Improved  company eligibility for government grants and VC funding by driving 30+ internship signups after building internship application pages and emailing feature s using  the LAMP stack. ● Upgraded  the management process of projects and staff  size of 12 from spreadsheets and online notes  by introducing Click Up and AWS  Identity Access Manag ement respectively. ● Developed  loan application web portal prototype for farmers using PHP  & MySQL  (LAMP stack) in 3 months for a demo to NIRSAL (ministry  for agricultural loans) to help DERDC secure a grant. ● Boosted  the firm's business visibility and conversion rate by 100% by building its corporate website using the LAMP stack. ● Created  a clinical record -keeping system prototype for a client on a tight schedule in less than 2 week s with Python /Flask  and PostgreSQL .  EDUCATION University  FUT Minna, Nigeria BEng in Computer Engineerin g  Oct 2016  SKILLS & INTERESTS Skills:  TensorFlow  (Text and Image Classification) | Scikit -learn  | Moqups  | Visily | Blender Interests:  Artificial Intelligence, Fintech, Neural Machine Translation, E -Health VOLUNTEER WORK ● Super -Contributor at Visily.ai (UI design & Prototyping Software) online community. ● Alpha Tester on StarBorne: Sovereign Space MMORTS (Online Multiplayer Stra tegy Game)."

# text = extract_text_from_doc("../data/resume_data/CHUKWU OGBONNAYA IBIAM.docx")[0]
# text = extract_text_from_doc("../data/resume_data/CURRICULUM VITAE Popoola.docx")[0]
# text = extract_text_from_pdf("../data/resume_data/CURRICULUM VITA1 abiola CU (2).pdf")[0]

def get_action_verbs(resume_text):
    # Remove hyphens so that hyphenated verbs won't be lost.
    resume_text = resume_text.replace('-', '')

    # Process the text using spaCy
    doc = nlp(resume_text)

    # Extract action verbs in the past tense from the processed text
    past_tense_verbs = [token.morph.get("Tense") for token in doc if token.pos_ == "VERB" and
                        len(token.morph.get("Tense")) > 0 and token.morph.get("Tense")[0].lower() == "past"]

    return past_tense_verbs


def get_bullet_points(resume_text):
    # Process the text using spaCy
    doc = nlp(resume_text)

    # Define a list of bullet point symbols to check for
    bullet_symbols = ["•", "-", "*", "●", "◦", "○", "▪", "■", "►", "▶", "➤", "⦿", "⦾"]
    bullet_points = []

    for sentence in doc.sents:
        for symbol in bullet_symbols:
            if sentence.text.strip().startswith(symbol):
                bullet_points.append(sentence)

    return bullet_points


def get_hard_skills(resume_text):
    return extract_skills(resume_text)


def is_active_voice(sentence):
    for token in sentence:
        if token.pos_ == "VERB":
            return True
    return False


# Does the resume contain Action Verbs?
# Examples of Action verbs are: 'Designed', 'Developed', 'Co-ordinated' etc.
def has_action_verbs(resume_text):
    past_tense_verbs = get_action_verbs(resume_text)

    # Returns true if there were action verbs.
    return len(past_tense_verbs) > 0


# Does the resume have at least 4 bullet points?
def has_4_or_more_bullet_points(resume_text):
    bullet_points = get_bullet_points(resume_text)
    return len(bullet_points) >= 4


def is_between_450_and_650_words(resume_text):
    word_count = len([word for word in text.split() if re.match('^[a-zA-Z0-9]*$', resume_text)])

    return 450 <= word_count <= 650


def has_at_least_10_hard_skills(resume_text):
    hks = get_hard_skills(resume_text)
    return len(hks) >= 5


def has_at_least_5_soft_skills(resume_text):
    sks = []

    for soft_skill in soft_skill_keywords:
        if soft_skill in resume_text:
            sks.append(soft_skill)

    return len(sks) >= 5


def has_no_repetition(resume_text):
    # Read the text from a file or create a list of strings representing lines of text
    text_lines = [
        "This is line 1.",
        "This is line 2.",
        "This is line 3.",
        "This is line 2.",  # This line is repeated
        "This is line 4.",
    ]

    # Create a set to store unique lines
    unique_lines = set()

    # Initialize a flag to track if any lines are repeated
    repeated_lines_found = False

    # Iterate through the lines
    for line in text_lines:
        if line in unique_lines:
            repeated_lines_found = True
            print(f"Repeated line found: {line}")
        else:
            unique_lines.add(line)

    # Check if any repeated lines were found
    if not repeated_lines_found:
        print("No repeated lines found.")


def has_no_personal_pronouns(resume_text):
    resume_text = resume_text.lower()
    personal_pronouns_keywords = ["i", "you", "he", "she", "it", "we", "they",
                                  "me", "him", "her", "us", "them",
                                  "my", "your", "his", "her", "its", "our", "their",
                                  "mine", "yours", "hers", "ours", "theirs"]
    personal_pronouns = []

    for pp in personal_pronouns_keywords:
        matches = re.findall(re.compile(r'\bpp\b'), resume_text)
        if len(matches) > 0:
            personal_pronouns += matches

    return len(personal_pronouns) == 0


def uses_active_voice(resume_text):
    bullet_points = get_bullet_points(resume_text)
    active_voice_sentences = []

    for sentence in bullet_points:
        if is_active_voice(sentence):
            active_voice_sentences.append(sentence)

    return len(active_voice_sentences) >= 4


# print(uses_active_voice(text))


def has_2_or_less_long_text_blocks(resume_text, threshold=2):
    consecutive_sentences = 0

    for sentence in get_bullet_points(resume_text):
        if len(sentence.text.split()) > 30:
            print('line', sentence)
            consecutive_sentences += 1

        if consecutive_sentences > threshold:
            print(consecutive_sentences)
            return False

    print(consecutive_sentences)
    return True


# print(has_2_or_less_long_text_blocks(text))


# [print(point) for point in has_4_or_more_bullet_points(text)]
# - no repet

# Impact
# - Repetition
# - Spelling & Consistency
# Style
# - Dates

def check_numbers_and_metrics(text):
    # Parse the text using spaCy to extract sentences
    doc = nlp(text)

    # Define a regular expression pattern to match numbers and metrics
    pattern = r'\b\d+(\.\d+)?\s?[A-Za-z]+(?:\s?[A-Za-z]+)?\b'

    results = []

    # Iterate through the sentences in the text
    for sent in doc.sents:
        sentence_text = sent.text

        # Find all matches of the pattern in the sentence
        matches = re.findall(pattern, sentence_text)

        if matches:
            results.append((sentence_text, matches))

    return results


# Check for numbers and metrics in sentences
results = check_numbers_and_metrics(text)

# print(len(results))
#
# # Print the results
# for sentence, matches in results:
#     print(f"Sentence: {sentence}")
#     print(f"Numbers and Metrics: {matches}")


def strip_special_characters(word):
    # Remove special characters, brackets, and punctuation from the word
    return re.sub(r'[^a-zA-Z]', '', word)


def exclude_emails_and_websites(text):
    # Exclude email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Exclude websites (URLs)
    text = re.sub(r'http\S+', '', text)
    return text


def check_spelling_english(text, skills):
    # Create an English dictionary object
    english_dict = enchant.Dict("en_US")

    # Use spaCy for named entity recognition (NER)
    doc = nlp(text)

    # Define a list to store misspelled English words
    misspelled_words = []

    # Iterate through the words
    for token in doc:
        word = token.text

        # Check if the word is not an entity, not an email address, and not a website
        if (
            token.ent_type_ == ""
            and not re.match(r'\S+@\S+', word)
            and not re.match(r'http\S+', word)
        ):
            # Check if the word is capitalized (treat as a noun)
            is_capitalized = word.istitle()

            # Strip special characters from the word
            cleaned_word = strip_special_characters(word)

            # Check if the cleaned word is English and misspelled
            if (
                cleaned_word
                and english_dict.check(cleaned_word) is False
                and (not is_capitalized)  # Exclude capitalized words
            ):
                # Check for similarity to words in the skills list
                is_similar = any(lev.distance(cleaned_word.lower(), skill.lower()) <= 2 for skill in skills)

                if not is_similar:
                    misspelled_words.append(cleaned_word)

    return misspelled_words


# List of skills (example skills)
skills = ["Python", "Java", "JavaScript", "Machine Learning"]

# Exclude emails, websites, capitalized words, nouns, pronouns, and check against skills before checking for spelling
# mistakes
text = exclude_emails_and_websites(text)

# Check for spelling mistakes only on cleaned English words (excluding emails, websites, company names, capitalized
# words, nouns, and pronouns)
misspelled_words = check_spelling_english(text, extract_skills(text))

# print("misspelled words:", len(misspelled_words))

# Print the misspelled English words
# for word in misspelled_words:
#     print(f"Misspelled English word: {word}")


def extract_dates_and_experiences(resume_text):
    # Define a regular expression pattern to match date ranges and single dates
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(' \
                   r'?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(' \
                   r'?:ember)?)[a-zA-Z]*[.-]?\s?\d{1,2}(?:st|nd|rd|th)?[,-]?\s?\d{2,4})\b '

    # Split the resume text into lines
    lines = resume_text.split('\n')

    # Initialize lists to store extracted date ranges and experiences
    date_ranges = []
    experiences = []

    current_experience = ""

    for line in lines:
        # Find all matches of dates (date ranges or single dates) in the current line
        dates_in_line = re.findall(date_pattern, line)

        if dates_in_line:
            # Extract and parse the found dates (single dates or date ranges)
            parsed_dates = [parser.parse(date) for date in dates_in_line]

            # Remove None values from parsed dates (dates that couldn't be parsed)
            parsed_dates = [date for date in parsed_dates if date is not None]

            # If there is more than one parsed date, assume it's a date range
            if len(parsed_dates) > 1:
                date_ranges.append(parsed_dates)
            else:
                # Add the single parsed date to the list of date ranges as a tuple
                date_ranges.append((parsed_dates[0], parsed_dates[0]))

            # Append the current experience to the list of experiences
            if current_experience:
                experiences.append(current_experience.strip())

            # Reset the current experience
            current_experience = ""
        else:
            # Append the line to the current experience (if not a date or date range)
            current_experience += line + '\n'

    # Append the last experience (if any) to the list of experiences
    if current_experience:
        experiences.append(current_experience.strip())

    return date_ranges, experiences


def check_experiences_reverse_order(date_ranges):
    # Sort the parsed date ranges in reverse chronological order based on end dates
    sorted_date_ranges = sorted(date_ranges, key=lambda x: x[1], reverse=True)

    # Check if the original list of date ranges matches the sorted list (reverse chronological order)
    is_reverse_order = date_ranges == sorted_date_ranges

    return is_reverse_order


# Extract date ranges and experiences from the resume text
date_ranges, experiences = extract_dates_and_experiences(text)

print(experiences)

# Check if experiences in the resume are in reverse chronological order
is_reverse_order = check_experiences_reverse_order(date_ranges)

# Print the result
if is_reverse_order:
    print("Experiences in the resume are in reverse chronological order based on date ranges.")
else:
    print("Experiences in the resume are not in reverse chronological order based on date ranges.")
