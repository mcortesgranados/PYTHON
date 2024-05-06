"""
24. PDF Generation: Generate PDF documents.

"""

from flask import Flask, make_response
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table

app = Flask(__name__)

# Route for generating a PDF document
@app.route('/generate_pdf')
def generate_pdf():
    """This function generates a PDF document."""
    # Create a response object
    response = make_response()

    # Set the content type to PDF
    response.headers['Content-Type'] = 'application/pdf'

    # Set the filename for the PDF
    response.headers['Content-Disposition'] = 'inline; filename=my_document.pdf'

    # Create a PDF document
    doc = SimpleDocTemplate(response, pagesize=letter)
    elements = []

    # Create a data table for the PDF
    data = [
        ['Name', 'Age', 'Country'],
        ['John', '30', 'USA'],
        ['Alice', '25', 'UK'],
        ['Bob', '35', 'Canada']
    ]
    table = Table(data)
    table.setStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    elements.append(table)

    # Build the PDF document
    doc.build(elements)

    return response

if __name__ == '__main__':
    app.run(debug=True)
