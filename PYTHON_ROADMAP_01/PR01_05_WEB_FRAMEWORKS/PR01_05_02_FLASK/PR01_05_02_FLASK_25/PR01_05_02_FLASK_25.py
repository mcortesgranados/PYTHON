"""
25. RESTful Forms: Create forms using Flask-WTF.

pip install Flask-WTF




"""

from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Define a Flask-WTF form
class MyForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Route for displaying the form
@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        # Process form data (e.g., save to database)
        name = form.name.data
        email = form.email.data
        # Redirect to a success page
        return redirect(url_for('success', name=name, email=email))
    return render_template('index.html', form=form)

# Route for displaying the success page
@app.route('/success/<name>/<email>')
def success(name, email):
    return f'<h1>Success!</h1><p>Name: {name}</p><p>Email: {email}</p>'

if __name__ == '__main__':
    app.run(debug=True)
