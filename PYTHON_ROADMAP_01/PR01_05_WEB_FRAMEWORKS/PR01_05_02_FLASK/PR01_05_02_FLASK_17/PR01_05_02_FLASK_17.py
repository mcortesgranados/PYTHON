"""
17. Caching: Cache expensive computations or database queries.

pip install Flask-Caching




"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)
db = SQLAlchemy(app)

# Define a model for the database table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

# Route to fetch users from the database and cache the results
@app.route('/users')
@cache.cached(timeout=60)  # Cache results for 60 seconds
def get_users():
    """This function fetches users from the database and caches the results."""
    users = User.query.all()
    return {'users': [{'username': user.username, 'email': user.email} for user in users]}

if __name__ == '__main__':
    app.run(debug=True)
