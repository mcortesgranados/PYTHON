import os
import re

def process_html_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                update_html_file(file_path)

def update_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Perform replacements
    replacements = [
        (r'K:\\AWS\\HTML\\', r'..\\'),
        (r'K:\\AWS_02\\', r'..\..\..\AWS_02\\'),
        (r'K:\\AWS_03\\', r'..\..\..\AWS_03\\'),
        (r'K:\\MS_AZURE\\', r'..\..\..\MS_AZURE\\'),
        (r'K:\\MS_AZURE_02_\\', r'..\..\..\MS_AZURE_02_\\'),
        (r'K:\\MS_AZURE_03_\\', r'..\..\..\MS_AZURE_03_\\'),
        (r'K:\\GCP\\', r'..\..\..\GCP\\'),
        (r'K:\\GCP_01\\', r'..\..\..\GCP_01\\')
    ]

    updated_content = html_content
    for old_pattern, new_pattern in replacements:
        updated_content = re.sub(old_pattern, new_pattern, updated_content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)

if __name__ == "__main__":
    folder_path = r'I:\AWS\HTML\ONE_STOP_FOR_CLOUD'
    process_html_files(folder_path)
    print("HTML files updated successfully.")
