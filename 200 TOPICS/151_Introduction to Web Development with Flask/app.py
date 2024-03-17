from flask import Flask, render_template, request
from routes import index, hello, login

app = Flask(__name__)

app.register_blueprint(index)
app.register_blueprint(hello)
app.register_blueprint(login)

if __name__ == '__main__':
    app.run(debug=True)
