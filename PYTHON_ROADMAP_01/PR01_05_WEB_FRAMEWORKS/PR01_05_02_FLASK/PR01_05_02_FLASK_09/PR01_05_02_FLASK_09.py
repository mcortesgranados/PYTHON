"""
09. Sessions: Manage user sessions.

"""

from flask import Flask, request, session, redirect, url_for, render_template

app = Flask(__name__)

# Set the secret key to enable session functionality
app.secret_key = 'your_secret_key'

# Define a route for setting session data
@app.route('/login', methods=['POST'])
def login():
    """This function sets session data and redirects to the homepage."""
    # Check if username and password are correct (dummy logic)
    if request.form['username'] == 'admin' and request.form['password'] == 'password':
        session['logged_in'] = True
    else:
        session['logged_in'] = False
    return redirect(url_for('index'))

# Define a route for logging out and clearing session data
@app.route('/logout')
def logout():
    """This function clears session data and redirects to the homepage."""
    session.pop('logged_in', None)
    return redirect(url_for('index'))

# Define a route for the homepage
@app.route('/')
def index():
    """This function displays the homepage based on session data."""
    if 'logged_in' in session and session['logged_in']:
        # If the user is logged in, display a welcome message
        return render_template('index.html', username='Admin')
    else:
        # If the user is not logged in, redirect to the login page
        return render_template('login.html')

if __name__ == '__main__':
    app.run()
