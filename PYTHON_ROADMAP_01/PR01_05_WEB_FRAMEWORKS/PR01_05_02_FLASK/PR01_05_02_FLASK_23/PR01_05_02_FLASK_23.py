"""
23. Email Sending: Send emails from your Flask application.

pip install Flask-Mail



"""

from flask import Flask, render_template
from flask_mail import Mail, Message

app = Flask(__name__)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@example.com'
app.config['MAIL_PASSWORD'] = 'your_email_password'

mail = Mail(app)

# Route for sending a test email
@app.route('/send_email')
def send_email():
    """This function sends a test email."""
    subject = 'Test Email'
    recipients = ['recipient1@example.com', 'recipient2@example.com']
    body = render_template('email_template.html')
    
    msg = Message(subject, recipients=recipients)
    msg.html = body
    
    try:
        mail.send(msg)
        return 'Email sent successfully!'
    except Exception as e:
        return f'Failed to send email: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
