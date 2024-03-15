# 34_Opening and Closing Files

# Opening a file in write mode
file = open('example.txt', 'w')

# Writing content to the file
file.write("Hello, world!\n")
file.write("This is a sample file.")

# Closing the file
file.close()

# Opening a file in read mode
file = open('example.txt', 'r')

# Reading content from the file
content = file.read()
print("Content of the file:")
print(content)

# Closing the file
file.close()
