"""
22. OAuth Integration: Integrate OAuth for third-party authentication.

pip install Flask-OAuthlib




"""

from flask import Flask, redirect, url_for, session
from flask_oauthlib.client import OAuth

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure OAuth for Google
oauth = OAuth(app)
google = oauth.remote_app(
    'google',
    consumer_key='your_google_client_id',
    consumer_secret='your_google_client_secret',
    request_token_params={
        'scope': 'email'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

# Route for initiating Google OAuth authentication
@app.route('/login')
def login():
    return google.authorize(callback=url_for('authorized', _external=True))

# Route for handling Google OAuth callback
@app.route('/login/authorized')
def authorized():
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        return 'Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        )
    session['google_token'] = (resp['access_token'], '')
    user_info = google.get('userinfo')
    # Process user information
    return 'Logged in as: ' + user_info.data['email']

# Callback function to load Google OAuth token from session
@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

if __name__ == '__main__':
    app.run(debug=True)
