"""
11. Database Integration: Interact with databases.

"""

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
db = SQLAlchemy(app)

# Define a model for the database table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

# Create the database tables
db.create_all()

# Define a route for displaying and submitting a form to add users
@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    """This function displays a form to add a new user."""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_user = User(username=username, email=email)
        db.session.add(new_user)
        db.session.commit()
        return 'User added successfully'
    return render_template('add_user.html')

# Define a route for displaying all users
@app.route('/users')
def users():
    """This function displays all users."""
    all_users = User.query.all()
    return render_template('users.html', users=all_users)

if __name__ == '__main__':
    app.run(debug=True)
