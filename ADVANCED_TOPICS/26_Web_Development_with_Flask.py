# FileName: 26_Web_Development_with_Flask.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Web Development with Flask in Python

# Flask is a lightweight web application framework for Python.
# It provides tools, libraries, and technologies to help build web applications quickly and easily.

from flask import Flask

# Create a Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return 'Hello, Flask!'

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
