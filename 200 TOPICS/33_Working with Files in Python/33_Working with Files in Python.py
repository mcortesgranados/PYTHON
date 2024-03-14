# Creating a file
with open('example.txt', 'w') as file:
    file.write("Hello, world!")

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print("File content:", content)

# Appending to a file
with open('example.txt', 'a') as file:
    file.write("\nAppending some more content.")

# Reading lines from a file
with open('example.txt', 'r') as file:
    lines = file.readlines()
    print("Lines in the file:")
    for line in lines:
        print(line.strip())

# Writing to a file using 'write' mode
with open('new_file.txt', 'w') as file:
    file.write("This is a new file.")

# Checking file existence
import os
if os.path.exists('example.txt'):
    print("example.txt exists.")
else:
    print("example.txt does not exist.")

# Renaming a file
os.rename('example.txt', 'renamed_example.txt')

# Deleting a file
os.remove('renamed_example.txt')
