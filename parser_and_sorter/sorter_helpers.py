import re
import math

from dateutil import parser

import Levenshtein as lev
import enchant

# Create an English dictionary object
english_dict = enchant.Dict("en_US")


def is_active_voice(sentence):
    for token in sentence:
        if token.pos_ == "VERB":
            return True
    return False


def exclude_emails_and_websites(text):
    # Exclude email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Exclude websites (URLs)
    text = re.sub(r'http\S+', '', text)
    return text


def strip_special_characters(word):
    # Remove special characters, brackets, and punctuation from the word
    return re.sub(r'[^a-zA-Z]', '', word)


def extract_dates(resume_text):
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

    return date_ranges


class ResumeSorter:

    def __init__(self, file_content, nlp, skill_extractor, soft_skill_keywords):
        # A score of >=50 will sort this resume into the relevant class; `1`
        # while a score of <50 will sort this resume into the irrelevant class; `0`
        self.score = 0
        self.raw_text = file_content[0]
        self.number_of_pages = file_content[1]

        # Process the text using spaCy
        # Remove hyphens so that hyphenated verbs won't be lost.
        self.doc = nlp(file_content[0].replace('-', ''))

        self.skill_extractor = skill_extractor
        self.hard_skill_keywords = None
        self.soft_skill_keywords = soft_skill_keywords

    def sort(self):
        self.hard_skill_keywords = self.extract_skills(self.raw_text)

        if self.has_2_or_more_action_verbs():
            self.score += 100 / 12
        if self.has_4_or_more_bullet_points():
            self.score += 100 / 12
        if self.is_between_450_and_650_words():
            self.score += 100 / 12
        if self.has_at_least_10_hard_skills():
            self.score += 100 / 12
        if self.has_at_least_5_soft_skills():
            self.score += 100 / 12
        if self.has_no_personal_pronouns():
            self.score += 100 / 12
        if self.uses_active_voice():
            self.score += 100 / 12
        if self.has_2_or_less_long_text_blocks():
            self.score += 100 / 12
        if self.is_2_pages_or_less():
            self.score += 100 / 12
        if self.has_4_or_more_numerical_metrics():
            self.score += 100 / 12
        if self.has_less_than_20_misspellings():
            self.score += 100 / 12
        if self.has_dates_in_reverse_chronological_order():
            self.score += 100 / 12

    def extract_skills(self, text):
        skills = []

        try:
            annotations = self.skill_extractor.annotate(text)

            full_matches = annotations['results']['full_matches']
            ngram_scored = annotations['results']['ngram_scored']

            s1 = [s['doc_node_value'] for s in full_matches]
            s2 = [s['doc_node_value'] for s in ngram_scored]

            skills = s1 + s2
        except ValueError as e:
            pass

        return skills

    def get_score(self):
        return math.ceil(self.score)

    def get_action_verbs(self):
        # Extract action verbs in the past tense from the processed text
        past_tense_verbs = [token.morph.get("Tense") for token in self.doc if token.pos_ == "VERB" and
                            len(token.morph.get("Tense")) > 0 and token.morph.get("Tense")[0].lower() == "past"]

        return past_tense_verbs

        # Does the resume contain Action Verbs?

    def get_bullet_points(self):

        # Define a list of bullet point symbols to check for
        bullet_symbols = ["•", "-", "*", "●", "◦", "○", "▪", "■", "►", "▶", "➤", "⦿", "⦾"]
        bullet_points = []

        for sentence in self.doc.sents:
            for symbol in bullet_symbols:
                if sentence.text.strip().startswith(symbol):
                    bullet_points.append(sentence)

        return bullet_points

    # Criterion #1
    # Examples of Action verbs are: 'Designed', 'Developed', 'Co-ordinated' etc.
    def has_2_or_more_action_verbs(self):
        past_tense_verbs = self.get_action_verbs()

        # Returns true if there were action verbs.
        return isinstance(past_tense_verbs, list) and len(past_tense_verbs) >= 2

    # Criterion #2
    # Does the resume have at least 4 bullet points?
    def has_4_or_more_bullet_points(self):
        # Define a list of bullet point symbols to check for
        bullet_symbols = ["•", "-", "*", "●", "◦", "○", "▪", "■", "►", "▶", "➤", "⦿", "⦾"]
        bullet_points = []

        for sentence in self.doc.sents:
            for symbol in bullet_symbols:
                if sentence.text.strip().startswith(symbol):
                    bullet_points.append(sentence.text)

        return len(bullet_points) >= 4

    # Criterion #3
    def is_between_450_and_650_words(self):
        word_count = len(self.raw_text.split())
        return 450 <= word_count <= 650

    # Criterion #4
    def is_2_pages_or_less(self):
        return self.number_of_pages <= 2

    # Criterion #5
    def has_at_least_10_hard_skills(self):
        return len(self.hard_skill_keywords) >= 5

    # Criterion #6
    def has_at_least_5_soft_skills(self):
        sks = []

        for soft_skill in self.soft_skill_keywords:
            if soft_skill in self.raw_text:
                sks.append(soft_skill)

        return len(sks) >= 5

    # Criterion #7
    def has_no_personal_pronouns(self):
        personal_pronouns_keywords = ["i", "you", "he", "she", "it", "we", "they",
                                      "me", "him", "her", "us", "them",
                                      "my", "your", "his", "her", "its", "our", "their",
                                      "mine", "yours", "hers", "ours", "theirs"]
        personal_pronouns = []

        for pp in personal_pronouns_keywords:
            matches = re.findall(re.compile(r'\bpp\b'), self.raw_text)
            if len(matches) > 0:
                personal_pronouns += matches

        return len(personal_pronouns) == 0

    # Criterion #8
    def uses_active_voice(self):
        bullet_points = self.get_bullet_points()
        active_voice_sentences = []

        for sentence in bullet_points:
            if is_active_voice(sentence):
                active_voice_sentences.append(sentence)

        return len(active_voice_sentences) >= 4

    # Criterion #9
    def has_2_or_less_long_text_blocks(self, threshold=2):
        consecutive_sentences = 0

        for sentence in self.get_bullet_points():
            if len(sentence.text.split()) > 30:
                consecutive_sentences += 1

            if consecutive_sentences > threshold:
                return False

        return True

    # Criterion #10
    def has_4_or_more_numerical_metrics(self):
        # Define a regular expression pattern to match numbers and metrics
        pattern = r'\b\d+(\.\d+)?\s?[A-Za-z]+(?:\s?[A-Za-z]+)?\b'

        results = []

        # Iterate through the sentences in the text
        for sent in self.doc.sents:
            sentence_text = sent.text

            # Find all matches of the pattern in the sentence
            matches = re.findall(pattern, sentence_text)

            if matches:
                results.append((sentence_text, matches))

        return len(results) >= 4

    # Criterion #11
    def has_less_than_20_misspellings(self):

        # Define a list to store misspelled English words
        misspelled_words = []

        # Iterate through the words
        for token in self.doc:
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
                    is_similar = any(lev.distance(cleaned_word.lower(), skill.lower()) <= 2 for skill in
                                     self.hard_skill_keywords)

                    if not is_similar:
                        misspelled_words.append(cleaned_word)

        return len(misspelled_words) < 20

    # Criterion #12
    def has_dates_in_reverse_chronological_order(self):
        date_ranges = extract_dates(self.raw_text)

        # Sort the parsed date ranges in reverse chronological order based on end dates
        sorted_date_ranges = sorted(date_ranges, key=lambda x: x[1], reverse=True)

        # Check if the original list of date ranges matches the sorted list (reverse chronological order)
        is_reverse_order = date_ranges == sorted_date_ranges

        return is_reverse_order
