"""
26. Background Jobs: Run background jobs with Flask-RQ2.
"""

from flask import Flask, render_template
from flask_rq2 import RQ
from rq import get_current_job
import time

app = Flask(__name__)
app.config['RQ_REDIS_URL'] = 'redis://localhost:6379/0'
rq = RQ(app)

# Function to simulate a time-consuming task
def long_running_task():
    job = get_current_job()
    for i in range(5):
        time.sleep(1)  # Simulate work
        job.meta['progress'] = i + 1
        job.save_meta()
    return 'Task completed successfully'

# Route for starting a background job
@app.route('/start_job')
def start_job():
    """This function starts a background job."""
    job = rq.get_queue().enqueue(long_running_task)
    return f'Started background job: {job.id}'

# Route for checking job progress
@app.route('/job_progress/<job_id>')
def job_progress(job_id):
    """This function checks the progress of a background job."""
    job = rq.job_class.fetch(job_id)
    if job.is_failed:
        return 'Job failed'
    elif job.is_finished:
        return 'Job completed'
    else:
        progress = job.meta.get('progress', 0)
        return f'Job progress: {progress}'

if __name__ == '__main__':
    app.run(debug=True)
