from models.parser_methods import extract_name, extract_email, extract_phone_number, extract_skills, extract_education


class ResumeParser(object):

    def __init__(self, file_content):
        self.name = None
        self.email = None
        self.phone = None
        self.skills = None
        self.education = None
        self.unknown = True
        self.text = file_content[0]

    def parse(self):
        unknown = True

        name = extract_name(self.text)
        email = extract_email(self.text)
        phone = extract_phone_number(self.text)
        skills = extract_skills(self.text)
        education = extract_education(self.text)

        if name is not None:
            self.name = name
            unknown = False
        if email is not None:
            self.email = email
            unknown = False
        if phone is not None:
            self.phone = phone
            unknown = False
        if skills is not None:
            self.skills = skills
            unknown = False
        if education is not None:
            self.education = education
            unknown = False

        self.unknown = unknown

    def summary(self):
        text = ''
        if self.name is not None:
            text += 'name: {}\n'.format(self.name)
        if self.email is not None:
            text += 'email: {}\n'.format(self.email)
        if self.phone is not None:
            text += 'phone: {}\n'.format(self.phone)
        if len(self.skills) > 0:
            text += 'skills: {}\n'.format(self.skills)
        if len(self.education) > 0:
            text += 'education: {}\n'.format(self.education)

        return text.strip()
