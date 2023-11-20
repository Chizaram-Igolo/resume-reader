import zipfile
import xml.dom.minidom


input_odt_file = "../data/resume_data/my_cv.odt"  # Replace with your .odt file path

# page_count = count_odt_pages(input_odt_file)

# if page_count is not None:

#     print(f"Number of pages: {page_count}")


def get_page_count_docx():
    document = zipfile.ZipFile("../data/resume_data/Harshitha Challa.docx")
    dxml = document.read("docProps/app.xml")
    uglyXml = xml.dom.minidom.parseString(dxml)
    page = uglyXml.getElementsByTagName("Pages")[0].childNodes[0].nodeValue
    print("Word Page count: " + page)


get_page_count_docx()
