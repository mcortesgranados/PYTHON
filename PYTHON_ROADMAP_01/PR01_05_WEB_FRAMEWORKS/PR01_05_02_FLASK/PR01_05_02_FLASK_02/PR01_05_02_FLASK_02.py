"""
02. Routing: Define multiple routes for different pages.

"""

from flask import Flask

app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    """This function returns the home page."""
    return 'Welcome to the Home Page!'

# Define route for the about page
@app.route('/about')
def about():
    """This function returns the about page."""
    return 'About Us: We are a team of developers.'

# Define route for the contact page
@app.route('/contact')
def contact():
    """This function returns the contact page."""
    return 'Contact Us: Email us at contact@example.com.'

if __name__ == '__main__':
    app.run()
