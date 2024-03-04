import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Context Managers in Python

# Context managers allow you to manage resources and perform setup and cleanup actions automatically.
# They are implemented using the 'with' statement and can be created using classes or functions.

# Define a context manager class using the 'class' syntax
class FileManager:
    """
    This context manager class allows you to open and close a file automatically.
    """
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

# Header for the output
print("Demo code of Context Managers in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Use the context manager to open a file and write data to it
with FileManager("example.txt", "w") as file:
    file.write("Hello, World!")

# Output:
# Demo code of Context Managers in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------

# The file "example.txt" will be automatically closed when exiting the 'with' block,
# ensuring that resources are properly managed and cleaned up.

# Context managers help improve code readability and maintainability by encapsulating resource management logic,
# reducing the risk of resource leaks and errors related to resource handling.
