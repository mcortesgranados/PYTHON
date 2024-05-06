"""
06. Static Files: Serve static files like CSS and JavaScript.

"""

from flask import Flask, render_template

app = Flask(__name__)

# Define a route to render an HTML template with a link to the static CSS file
@app.route('/')
def index():
    """This function renders the index.html template."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
