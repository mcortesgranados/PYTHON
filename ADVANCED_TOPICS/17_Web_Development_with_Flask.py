# FileName: 17_Web_Development_with_Flask.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# This Python code demonstrates web development with the Flask framework, a lightweight web application framework in Python.
# It creates a simple web application with a single route that returns the text "Hello, Flask!" when accessed.
# Additionally, it includes metadata such as the proposed filename, authorship information, date and time, location,
# and a link to the author's LinkedIn profile for context.

from flask import Flask

# Create a Flask application
app = Flask(__name__)

# Define a route and view function
@app.route('/')
def hello():
    return 'Hello, Flask!'

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
