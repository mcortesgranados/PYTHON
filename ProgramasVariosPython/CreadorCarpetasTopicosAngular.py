import os

# Folder path for creating folders
folder_path = r"I:\ANGULAR_FROM_SCRATCH\ANGULAR_APLICACION_CONCEPTOS"

# File path for topics.txt
topics_file_path = os.path.join(folder_path, "topics.txt")

# Check if the file exists
if not os.path.exists(topics_file_path):
    print(f"Error: File '{topics_file_path}' not found.")
    exit()

# Read topics from file
topics = []
try:
    with open(topics_file_path, "r") as file:
        topics = file.readlines()
        topics = [topic.strip() for topic in topics]
except FileNotFoundError:
    print(f"Error: File '{topics_file_path}' not found.")
    exit()

# Create folders
for i, topic in enumerate(topics):
    folder_name = os.path.join(folder_path, f"{i:03d}_{topic}")
    try:
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")
    except OSError as e:
        print(f"Failed to create folder: {folder_name} - {e}")
