"""
08. Cookies: Set and retrieve cookies.

"""

from flask import Flask, request, make_response, render_template

app = Flask(__name__)

# Define a route for setting cookies
@app.route('/setcookie')
def set_cookie():
    """This function sets a cookie and redirects to the homepage."""
    # Set a cookie named 'username' with value 'john'
    resp = make_response(redirect('/'))
    resp.set_cookie('username', 'john')
    return resp

# Define a route for getting cookies
@app.route('/')
def get_cookie():
    """This function retrieves the value of the 'username' cookie."""
    # Retrieve the value of the 'username' cookie
    username = request.cookies.get('username')
    return render_template('index.html', username=username)

if __name__ == '__main__':
    app.run()
