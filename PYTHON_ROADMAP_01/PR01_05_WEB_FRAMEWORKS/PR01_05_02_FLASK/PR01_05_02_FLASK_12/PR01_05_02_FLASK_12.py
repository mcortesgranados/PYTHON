"""
12. API Endpoints: Build RESTful APIs.

"""

from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data (can be replaced with database queries)
tasks = [
    {
        'id': 1,
        'title': 'Task 1',
        'description': 'Description for Task 1',
        'done': False
    },
    {
        'id': 2,
        'title': 'Task 2',
        'description': 'Description for Task 2',
        'done': False
    }
]

# Define a route for getting all tasks
@app.route('/tasks', methods=['GET'])
def get_tasks():
    """This function returns all tasks."""
    return jsonify(tasks)

# Define a route for getting a specific task by ID
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """This function returns a specific task by ID."""
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

# Define a route for creating a new task
@app.route('/tasks', methods=['POST'])
def create_task():
    """This function creates a new task."""
    data = request.json
    if 'title' not in data:
        return jsonify({'error': 'Title is required'}), 400
    new_task = {
        'id': len(tasks) + 1,
        'title': data['title'],
        'description': data.get('description', ''),
        'done': False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

if __name__ == '__main__':
    app.run(debug=True)
