# In this code:

# greet() function accepts two positional arguments: name and message. Positional arguments are passed in the order they are defined in the function signature.
# greet_with_defaults() function accepts two keyword arguments: name and message. Keyword arguments are passed by explicitly mentioning the parameter names.
# Both functions print a greeting message using the provided name and message.
#greet_with_defaults() function has default values for both parameters. If no value is provided for name or message, the defaults are used.
        
# Function with positional arguments
def greet(name, message):
    """This function greets the person with the given name using the provided message."""
    print(message, name)

# Call the function with positional arguments
greet("Alice", "Hello,")  # "Alice" is assigned to name and "Hello," is assigned to message

# Function with keyword arguments
def greet_with_defaults(name="Guest", message="Hello,"):
    """This function greets the person with the given name using the provided message. 
    If no name or message is provided, defaults are used."""
    print(message, name)

# Call the function with keyword arguments
greet_with_defaults()  # Defaults are used
greet_with_defaults(name="Bob")  # Only name is provided
greet_with_defaults(message="Hi there,")  # Only message is provided
greet_with_defaults(name="Charlie", message="Hey,")  # Both name and message are provided
