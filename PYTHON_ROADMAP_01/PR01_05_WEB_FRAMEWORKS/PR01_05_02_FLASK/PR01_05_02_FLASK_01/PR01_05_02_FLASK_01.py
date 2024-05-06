"""
01. Basic Hello World: Start with a simple Flask app that returns "Hello, World!".

"""

# Import the Flask module
from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the root URL ('/')
# When someone accesses the root URL, this function will be called
@app.route('/')
def hello_world():
    """This function returns a simple 'Hello, World!' message."""
    return 'Hello, World!'

# Check if the script is being run directly (not imported as a module)
if __name__ == '__main__':
    # Run the Flask application
    # The app will be hosted on the local development server with the default port 5000
    app.run()
