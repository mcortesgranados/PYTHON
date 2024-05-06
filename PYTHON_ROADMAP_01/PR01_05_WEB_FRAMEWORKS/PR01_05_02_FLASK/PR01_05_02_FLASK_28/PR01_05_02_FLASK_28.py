"""
28. Admin Panel Integration: Integrate Flask-Admin for admin panel functionalities.

pip install Flask-Admin


"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'

db = SQLAlchemy(app)

# Define a SQLAlchemy model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)

# Create the database tables
db.create_all()

# Initialize Flask-Admin
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')

# Add the Product model to Flask-Admin
admin.add_view(ModelView(Product, db.session))

if __name__ == '__main__':
    app.run(debug=True)
