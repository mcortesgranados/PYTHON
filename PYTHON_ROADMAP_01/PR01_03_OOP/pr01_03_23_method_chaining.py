"""

Method chaining, also known as fluent interface, is a design pattern where multiple methods can be called on 
an object in sequence, with each method returning the modified object. This allows for concise and readable code. 
In Python, method chaining is achieved by having methods return self after performing their operations. 
Below is an example illustrating method chaining in Python, along with detailed explanations and documentation:

"""

# Python Method Chaining Example

class TextFormatter:
    """
    A class demonstrating method chaining in Python for text formatting.

    Attributes:
    - text (str): The text to be formatted.

    Methods:
    - __init__(self, text): Constructor method to initialize the text.
    - uppercase(self): Convert the text to uppercase.
    - lowercase(self): Convert the text to lowercase.
    - remove_whitespace(self): Remove whitespace from the text.
    """

    def __init__(self, text):
        """
        Initialize the text.

        Args:
        - text (str): The text to be formatted.
        """
        self.text = text

    def uppercase(self):
        """
        Convert the text to uppercase.

        Returns:
        - TextFormatter: The modified TextFormatter object.
        """
        self.text = self.text.upper()
        return self

    def lowercase(self):
        """
        Convert the text to lowercase.

        Returns:
        - TextFormatter: The modified TextFormatter object.
        """
        self.text = self.text.lower()
        return self

    def remove_whitespace(self):
        """
        Remove whitespace from the text.

        Returns:
        - TextFormatter: The modified TextFormatter object.
        """
        self.text = ''.join(self.text.split())
        return self

# Example: Using Method Chaining
# Let's demonstrate the usage of method chaining for text formatting.

# Create a TextFormatter object and perform method chaining
formatted_text = TextFormatter("  Hello, World!  ").uppercase().remove_whitespace()

# Display the formatted text
print("Formatted Text:", formatted_text.text)  # Output: HELLO,WORLD!

# Documenting the TextFormatter Class:
def method_chaining_documentation():
    """
    This function demonstrates method chaining in Python.

    Classes:
    - TextFormatter: A class demonstrating method chaining in Python for text formatting.
    """
    pass

# End of example
