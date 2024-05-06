from flask import Flask, request, render_template
from flask_babel import Babel, _
import os

app = Flask(__name__)
babel = Babel(app)

# Configure Babel
app.config['BABEL_TRANSLATION_DIRECTORIES'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'translations')
app.config['BABEL_DEFAULT_LOCALE'] = 'en'

# Dummy data for translations
translations = {
    'en': {
        'hello': 'Hello, World!',
        'greeting': 'Welcome to my Flask app!'
    },
    'fr': {
        'hello': 'Bonjour, le monde!',
        'greeting': 'Bienvenue sur mon application Flask !'
    }
}

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for switching languages
@app.route('/set_language/<lang>')
def set_language(lang):
    """This function sets the language preference for the session."""
    if lang in translations:
        session['language'] = lang
    return redirect(request.referrer or '/')

# Template filter for translation
@app.template_filter('translate')
def translate(key):
    """This function translates the given key."""
    lang = session.get('language', 'en')
    return translations.get(lang, {}).get(key, key)

if __name__ == '__main__':
    app.run(debug=True)
