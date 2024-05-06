"""
15. Background Tasks: Execute background tasks.

"""

from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)

# Function to execute as a background task
def background_task():
    """This function simulates a long-running task."""
    print("Background task started")
    time.sleep(5)  # Simulate a long-running task (5 seconds)
    print("Background task completed")

# Define a route to trigger the background task
@app.route('/start_task')
def start_task():
    """This function starts the background task."""
    # Create and start a new thread to run the background task
    task_thread = threading.Thread(target=background_task)
    task_thread.start()
    return 'Background task started'

if __name__ == '__main__':
    app.run(debug=True)
