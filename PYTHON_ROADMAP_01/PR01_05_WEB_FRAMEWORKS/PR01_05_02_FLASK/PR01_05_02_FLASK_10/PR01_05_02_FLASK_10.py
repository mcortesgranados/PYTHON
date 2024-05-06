"""
10. File Uploads: Accept file uploads from users.

"""

from flask import Flask, request, render_template

app = Flask(__name__)

# Set the path where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for the file upload form
@app.route('/upload')
def upload_form():
    """This function renders the file upload form."""
    return render_template('upload.html')

# Define a route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    """This function handles file uploads."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file to the specified folder
    file.save(app.config['UPLOAD_FOLDER'] + '/' + file.filename)

    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()
