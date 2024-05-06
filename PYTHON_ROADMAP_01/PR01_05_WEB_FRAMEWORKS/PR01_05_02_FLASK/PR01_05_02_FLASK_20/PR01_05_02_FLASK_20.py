"""
20. Logging: Log events and errors.

"""

from flask import Flask
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Route that generates a sample log message
@app.route('/')
def index():
    app.logger.info('Index route accessed')
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
