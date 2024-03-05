# FileName: 40_Advanced_File_Handling.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Advanced File Handling with os and shutil

# Python provides powerful modules like os and shutil for advanced file handling tasks such as file manipulation, copying, moving, and more.

import os
import shutil

# Example: Copying Files with shutil

# Source and destination paths
source_path = 'source_folder'
destination_path = 'destination_folder'

# Create destination folder if it doesn't exist
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

# Copy all files from source to destination
for file_name in os.listdir(source_path):
    source_file = os.path.join(source_path, file_name)
    destination_file = os.path.join(destination_path, file_name)
    shutil.copy(source_file, destination_file)

print("Files copied successfully!")
