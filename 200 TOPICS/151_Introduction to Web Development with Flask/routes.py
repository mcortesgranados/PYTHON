from flask import Blueprint, render_template, request

index = Blueprint('index', __name__)

@index.route('/')
def home():
    return 'Hello, World!'

hello = Blueprint('hello', __name__)

@hello.route('/hello')
@hello.route('/hello/<name>')
def greet(name=None):
    return render_template('hello.html', name=name)

login = Blueprint('login', __name__)

@login.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Process login data
    return render_template('login.html')
