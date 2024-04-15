import os

# Define the directory path
directory = r'K:\PYTHON\MACHINE LEARNING MODELS\V002'

# Create a list to store folder names
folder_names = []

# Iterate over the folders in the directory
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    # Check if the item in the directory is a folder
    if os.path.isdir(folder_path):
        folder_names.append(folder)

# Write the folder names into a text file
output_file = 'carpetasFinales.txt'
with open(output_file, 'w') as file:
    for folder_name in folder_names:
        file.write(folder_name + '\n')

print("Folder names have been written to", output_file)
