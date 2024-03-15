# Writing to a file
with open('example.txt', 'w') as file:
    file.write("Hello, world!\n")
    file.write("This is a sample file.")

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print("Content of the file:")
    print(content)
