"""
03. Passing URL Parameters: Capture parameters from the URL.

"""

from flask import Flask

app = Flask(__name__)

# Define a route with a parameter in the URL
@app.route('/user/<username>')
def show_user_profile(username):
    """This function displays the user profile based on the username."""
    return f'User Profile: {username}'

if __name__ == '__main__':
    app.run()
