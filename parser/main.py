import functools
import csv
import os
import settings
import sqlite3

from flask import (Blueprint, flash, Flask, g, redirect, render_template, request,
    send_from_directory, url_for)
from parser.db import get_db
from werkzeug.utils import secure_filename

bp = Blueprint('main', __name__, url_prefix='')

ALLOWED_EXTENSIONS = {'csv'}
REQUIRED_HEADERS = {
    'vec_file': ['names']
}

# check filename extension on uploaded files
def allowed_file(file):
    return ('.' in file.filename 
        and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        and file.content_type == 'text/csv')

def check_headers(file, upload_type = 'vec_file'):
    # check that the required headers are present in the file
    headers = file.readline().decode('utf-8').split(',')
    for header in REQUIRED_HEADERS[upload_type]:
        if header not in headers:
            return False
    return True

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
@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # make sure there's a REQUIRED_HEADERS entry for each file upload input
        if not (REQUIRED_HEADERS.keys() & request.files.keys()):
            flash('No file', 'error')
            return redirect(request.url)
        # iterate through the different file uploads
        for upload_type in request.files:
            file = request.files[upload_type]
            # If selected file is empty / without filename
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if not (file and allowed_file(file)):
                flash('Invalid file type. Select a CSV to upload', 'error')
                return render_template('upload_form.html')
            if not check_headers(file, 'vec_file'):
                flash('Invalid CSV headers. Make sure uploaded CSV has required columns.', 'error')
                return render_template('upload_form.html')
            # if it passes all the checks, save file and queue for processing
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # TODO: queue for processing instead of spitting it back
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    # GET
    return render_template('upload_form.html')

# returning uploaded files; get rid of this in prod
@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    # TODO: documentation on setting file directory permissions
    # i.e. uploads dir can't be inspected




