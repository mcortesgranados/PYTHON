"""
05. Templates: Render HTML templates.

"""
from flask import Flask, render_template

app = Flask(__name__)

# Define a route to render the HTML template
@app.route('/')
def index():
    """This function renders the index.html template."""
    # Pass variables to the template
    title = 'Welcome to My Website'
    message = 'Hello, World!'
    # Render the template with the variables
    return render_template('index.html', title=title, message=message)

if __name__ == '__main__':
    app.run()
