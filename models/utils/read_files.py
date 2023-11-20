import os

import PyPDF2.errors
from PyPDF2 import PdfReader
import docx2txt

import zipfile
import xml.dom.minidom


def extract_text_from_pdf(file_path):
    """Extracts .pdf file text"""

    # create a pdf reader object
    file_reader = PdfReader(file_path, True)
    number_of_pages = len(file_reader.pages)

    page_number, temp = 0, ""
    word_count = 0

    while page_number < number_of_pages:
        page_obj = file_reader.pages[page_number]
        page_number += 1
        t = page_obj.extract_text()
        word_count += len(t.split())
        temp += " " + t

    result = [line.replace('\t', ' ').strip() for line in temp.split('\n') if line]

    return [' '.join(result), number_of_pages]


def extract_text_from_doc(file_path):
    temp = docx2txt.process(file_path)
    result = [line.replace('\t', ' ') for line in temp.split('\n') if line]

    document = zipfile.ZipFile(file_path)
    dxml = document.read("docProps/app.xml")
    uglyXml = xml.dom.minidom.parseString(dxml)
    number_of_pages = int(uglyXml.getElementsByTagName("Pages")[0].childNodes[0].nodeValue)

    return [' '.join(result), number_of_pages]


def read_pdf_and_docx(dir_path, collected=None, command_logging=False, callback=None):
    if collected is None:
        collected = dict()

    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)

        if os.path.isfile(file_path):
            content = None
            if f.lower().endswith('.docx'):
                if command_logging:
                    print('extracting text from docx: ', file_path)
                try:
                    content = extract_text_from_doc(file_path)
                except Exception as e:
                    continue
            elif f.lower().endswith('.pdf'):
                if command_logging:
                    print('extracting text from pdf: ', file_path)
                try:
                    content = extract_text_from_pdf(file_path)
                except Exception as e:
                    continue
            if content is not None and len(content) > 0:
                if callback is not None:
                    callback(len(collected), file_path, content)
                collected[file_path] = content
        elif os.path.isdir(file_path):
            read_pdf_and_docx(file_path, collected, command_logging, callback)

    return collected


def read_pdf(dir_path, collected=None, command_logging=False):
    if collected is None:
        collected = dict()

    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)

        if os.path.isfile(file_path):
            content = None
            if f.lower().endswith('.pdf'):
                if command_logging:
                    print('extracting text from .pdf: ', file_path)
                try:
                    content = extract_text_from_pdf(file_path)
                except Exception as e:
                    continue
            if content is not None and len(content) > 0:
                collected[file_path] = content
        elif os.path.isdir(file_path):
            read_pdf(file_path, collected)

    return collected
