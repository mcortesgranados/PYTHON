# Basic web scraping involves extracting information from web pages. Python provides several libraries for web scraping, including requests for making HTTP requests and BeautifulSoup for parsing HTML and XML documents. Here's a basic example of how to use these libraries for web scraping:

# Making HTTP Requests with requests:

# Use the requests.get() function to send an HTTP GET request to a URL and retrieve the HTML content of the web page.
# Syntax: response = requests.get(url)
# Example:

import requests

url = 'https://example.com'
response = requests.get(url)
if response.status_code == 200:
    html_content = response.content
else:
    print('Failed to fetch page:', response.status_code)


# Parsing HTML with BeautifulSoup:

# Use BeautifulSoup to parse the HTML content and extract specific information from it.
# You can search for HTML elements by tag name, class, id, etc., and extract text, attributes, or nested elements.
# Install BeautifulSoup using pip: pip install beautifulsoup4
# Example:
    
from bs4 import BeautifulSoup
import requests
    

url = 'https://openai.com/blog/'
response = requests.get(url)

if response.status_code == 200:

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find elements by tag name
        title = soup.find('title').text

        # Find elements by class
        paragraphs = soup.find_all(class_='paragraph')

        # Find elements by id
        main_content = soup.find(id='main-content')

        # Extract text from elements
        for p in paragraphs:
                print(p.text)
