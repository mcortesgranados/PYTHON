"""
07. Redirects and Errors: Redirect users or handle errors gracefully.

"""

from flask import Flask, redirect, url_for, abort

app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    """This function redirects users to the welcome page."""
    return redirect(url_for('welcome'))

# Define a route for the welcome page
@app.route('/welcome')
def welcome():
    """This function displays a welcome message."""
    return 'Welcome to our website!'

# Define a route for handling a custom error page
@app.route('/error')
def error():
    """This function simulates an error and handles it gracefully."""
    # Simulate an error condition
    abort(404)

# Define a custom error handler for 404 errors
@app.errorhandler(404)
def page_not_found(error):
    """This function handles 404 errors gracefully."""
    return '404 Error: Page not found', 404

if __name__ == '__main__':
    app.run()
