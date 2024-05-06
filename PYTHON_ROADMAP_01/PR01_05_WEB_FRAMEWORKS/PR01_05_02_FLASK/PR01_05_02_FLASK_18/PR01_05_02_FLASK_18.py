"""
18. API Rate Limiting: Limit API requests per user.

pip install Flask-Limiter


"""

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address)

# Route with rate limiting applied
@app.route('/api')
@limiter.limit("5 per minute")  # Allow 5 requests per minute per IP address
def api():
    """This function represents the API endpoint."""
    return 'API response'

if __name__ == '__main__':
    app.run(debug=True)
