"""
14. Authorization: Restrict access to certain routes based on user roles.

"""

from flask import Flask, request, jsonify, session, redirect, url_for, render_template

app = Flask(__name__)

# Set the secret key to enable session functionality
app.secret_key = 'your_secret_key'

# Sample user data (can be replaced with a database)
users = {
    'admin': {
        'password': 'admin_pass',
        'role': 'admin'
    },
    'user': {
        'password': 'user_pass',
        'role': 'user'
    }
}

# Define a decorator for checking user roles
def require_role(role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'username' in session and users[session['username']]['role'] == role:
                return func(*args, **kwargs)
            else:
                return jsonify({'error': 'Unauthorized access'}), 401
        return wrapper
    return decorator

# Define a route for the admin page (accessible only to users with 'admin' role)
@app.route('/admin')
@require_role('admin')
def admin():
    """This function displays the admin page."""
    return 'Welcome to the admin page'

# Define a route for the user page (accessible only to users with 'user' role)
@app.route('/user')
@require_role('user')
def user():
    """This function displays the user page."""
    return 'Welcome to the user page'

# Define a route for the login page
@app.route('/login', methods=['POST'])
def login():
    """This function handles user login."""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username in users and users[username]['password'] == password:
        session['username'] = username
        return 'Login successful'
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

# Define a route for logging out
@app.route('/logout')
def logout():
    """This function handles user logout."""
    session.pop('username', None)
    return 'Logout successful'

if __name__ == '__main__':
    app.run(debug=True)
