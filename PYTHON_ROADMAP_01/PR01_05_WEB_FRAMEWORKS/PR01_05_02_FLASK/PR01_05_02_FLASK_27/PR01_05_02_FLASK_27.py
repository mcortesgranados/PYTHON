"""
27. Real-time Data Visualization: Use Flask-SocketIO for real-time data visualization.

pip install Flask-SocketIO eventlet


"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Route for rendering the index page
@app.route('/')
def index():
    return render_template('index.html')

# Function to emit random data at intervals
def emit_random_data():
    while True:
        data = {'x': random.randint(0, 100), 'y': random.randint(0, 100)}
        socketio.emit('data_update', data)
        time.sleep(1)

# Start emitting random data when the server starts
@socketio.on('connect')
def handle_connect():
    emit('data_update', {'message': 'Connected to server'})
    emit_random_data()

if __name__ == '__main__':
    socketio.run(app, debug=True)
