# FileName: 42_System_Programming.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# System Programming

# Python provides modules like os and subprocess for interacting with the operating system, allowing you to perform various system-related tasks.

import os

# Example: Checking the current working directory
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

# Example: Listing files in a directory
files = os.listdir('.')
print("Files in Current Directory:", files)

# Example: Creating a new directory
new_directory = 'new_directory'
os.makedirs(new_directory)
print("New Directory Created:", new_directory)
