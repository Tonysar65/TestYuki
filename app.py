# app.py
from flask import Flask, render_template, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download')
def download():
    try:
        # Percorso assoluto per il file PDF
        pdf_path = os.path.join(os.getcwd(), 'static', 'documents', 'documento.pdf')
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)