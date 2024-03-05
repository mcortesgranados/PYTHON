# FileName: 38_Parsing_and_Manipulating_XML_HTML_Documents.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Parsing and Manipulating XML/HTML Documents with lxml

# Python provides powerful libraries like lxml for parsing and manipulating XML and HTML documents.

from lxml import etree

# Example: Parsing and Manipulating XML

# Sample XML data
xml_data = '''
<bookstore>
    <book>
        <title lang="en">Harry Potter</title>
        <author>J.K. Rowling</author>
        <year>2005</year>
    </book>
    <book>
        <title lang="fr">Le Petit Prince</title>
        <author>Antoine de Saint-Exupéry</author>
        <year>1943</year>
    </book>
</bookstore>
'''

# Parse the XML data
root = etree.fromstring(xml_data)

# Get all book titles
titles = root.xpath('//title/text()')
print("Book Titles:", titles)

# Get the author of the first book
author = root.xpath('/bookstore/book[1]/author/text()')[0]
print("Author of the First Book:", author)
