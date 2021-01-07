from flask import (flash, Flask, g, redirect, render_template, request,
    send_from_directory, url_for)
from werkzeug.utils import secure_filename
import os
import settings
import sqlite3

# configure app
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'Projects/NLP/SMS_Annotation/Input_data')

ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS
app.secret_key = settings.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_CONTENT_LENGTH


# check filename extension on uploaded files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# For now, we're going to go ahead and run the script as soon as we get a file upload
# Ultimately we want to use a queue to make sure we're doing these one at a time.
def queue_job(type, message):
    return

def process_job(job_id):
    return

def complete_job(job_id):
    return

def delete_file(filename):
    return

# most of the business is here
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        import pdb; pdb.set_trace()
        # TODO: check to see if a file was submitted at all
        if 'vec_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # TODO: if multiple files are submitted, take the first one
        file = request.files['vec_file']
        # If selected file is empty / without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
        # TODO: check file for required headers
        # TODO: run script
        # TODO: return output file
    else: 
        return render_template('upload_form.html')

# returning uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    # TODO: documentation on setting file directory permissions
    # i.e. uploads dir can't be inspected




