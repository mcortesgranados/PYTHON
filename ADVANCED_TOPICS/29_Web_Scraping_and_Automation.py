# FileName: 29_Web_Scraping_and_Automation.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Web Scraping and Automation with Python

# Python provides libraries like BeautifulSoup and Selenium for web scraping and automation tasks.

# Installation Instructions:
# You can install BeautifulSoup and Selenium using pip.
# For BeautifulSoup:
# pip install beautifulsoup4
# For Selenium:
# pip install selenium
# pip install wcwidth


from bs4 import BeautifulSoup
from selenium import webdriver

# Example: Scraping data from a website using BeautifulSoup

# Initialize a web driver (you need to install the appropriate driver for your browser)
driver = webdriver.Chrome('path_to_chromedriver')

# Open a webpage using the driver
driver.get('https://example.com')

# Extract HTML content
html_content = driver.page_source

# Parse HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find elements by tag name, class, id, etc., and extract data
# Example:
# title = soup.find('h1').text
# print(title)

# Close the driver
driver.quit()
