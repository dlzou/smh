#!/usr/bin/env python
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

abs_path = os.path.abspath(__file__)
home_dir = os.path.dirname(abs_path)

UPLOAD_PATH = os.path.join(home_dir, 'uploaded')
SANITIZED_PATH = os.path.join(home_dir, 'saflashnitized')
EXTENSIONS = ['wav', 'raw']

@app.route('/', methods=['POST'])
def process_audio():
    f = request.files.get('audio', None)
    if f and valid_file(f.filename):
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(filepath)



def valid_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)