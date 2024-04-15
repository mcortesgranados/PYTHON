import os

def create_folders_and_files(file_path):
    try:
        with open(file_path, 'r') as file:
            print("Creating folders and files:")
            count = 1
            for line in file:
                line = line.strip()
                folder_name = f"{count:03d} - {line}"
                try:
                    os.makedirs(folder_name)
                    print(f"Created folder: {folder_name}")

                    # Create a Python file inside the folder
                    file_name = os.path.join(folder_name, f"{count:03d}_{line}.py")
                    with open(file_name, 'w') as py_file:
                        py_file.write(f"# This is a Python file created by the program.\n")
                        py_file.write(f"# File name: {count:03d}_{line}.py\n")
                        py_file.write("# @author Manuela Cortes Granados - 14 Abril 2024 1:42 PM")

                    print(f"Created file: {file_name}")
                except FileExistsError:
                    print(f"Folder '{folder_name}' already exists.")
                count += 1
    except FileNotFoundError:
        print("File not found.")

# Replace 'example.txt' with the path to your text file
file_path = 'nombresCarpetas.txt'
create_folders_and_files(file_path)
