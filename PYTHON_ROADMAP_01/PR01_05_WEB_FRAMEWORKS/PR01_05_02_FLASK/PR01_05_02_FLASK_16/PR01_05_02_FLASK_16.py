"""
16. Websockets: Implement Websockets for real-time communication.

pip install flask flask-socketio eventlet


"""

from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Route for serving the HTML page with WebSocket client code
@app.route('/')
def index():
    return render_template('index.html')

# WebSocket event handler for receiving and broadcasting messages
@socketio.on('message')
def handle_message(message):
    print('Received message:', message)
    # Broadcast the received message to all clients
    socketio.send(message)

if __name__ == '__main__':
    socketio.run(app)
