"""
29. Google Maps Integration: Integrate Google Maps with Flask.

"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True)
