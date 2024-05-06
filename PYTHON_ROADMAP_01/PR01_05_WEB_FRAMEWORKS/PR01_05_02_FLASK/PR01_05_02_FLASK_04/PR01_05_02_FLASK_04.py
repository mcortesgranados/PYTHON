"""
04. HTTP Methods: Handle different HTTP methods.

"""

from flask import Flask, request

app = Flask(__name__)

# Define a route that handles both GET and POST requests
@app.route('/login', methods=['GET', 'POST'])
def login():
    """This function handles login requests."""
    if request.method == 'POST':
        # Handle login logic for POST requests
        username = request.form['username']
        password = request.form['password']
        # Check username and password
        return f'Logging in as {username}'
    else:
        # Show login form for GET requests
        return '''
        <form method="post">
            <p>Username: <input type="text" name="username"></p>
            <p>Password: <input type="password" name="password"></p>
            <p><input type="submit" value="Login"></p>
        </form>
        '''

if __name__ == '__main__':
    app.run()
