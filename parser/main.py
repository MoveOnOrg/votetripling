import functools
import csv
import os
import settings
import sqlite3

from flask import (Blueprint, current_app, flash, Flask, g, redirect,
    render_template, request, send_from_directory, url_for)
from . import db
from werkzeug.utils import secure_filename

bp = Blueprint('main', __name__, url_prefix='')

ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_TYPES = {
    'tblc_file': { 
        'name': 'Text Banker Log Cleaning',
        'required_headers': ['names']
    },
    'tblc_tmc_file': {
        'name': 'Text Banker Log Cleaning (utilizing text message conversation)',
        'required_headers': ['names']
    },
    'sccne_file': {
        'name': 'SMS Conversation Categorization and Name Extraction',
        'required_headers': []
    },
    'sms_agg_file': {
        'name': 'SMS Aggregation',
        'required_headers': []
    }
}


def allowed_file(file):
    return ('.' in file.filename 
        and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        and file.content_type == 'text/csv')


def check_headers(file, upload_type):
    headers = file.readline().decode('utf-8').rstrip().split(',')
    for header in UPLOAD_TYPES['required_headers']:
        if header not in headers:
            return False
    return True


def queue_job(file_path, job_type, email):
    query = 'INSERT INTO jobs (file_path, job_type, email, status) VALUES (?, ?, ?, ?);'
    db.query_db(query, (file_path, job_type, email, 'queued'), insert=True)
    return True


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
        if not (UPLOAD_TYPES.keys() & request.files.keys()):
            flash('No file', 'error')
            return redirect(request.url)
        for upload_type in request.files:
            if upload_type not in UPLOAD_TYPES:
                return redirect(request.url)
            file = request.files[upload_type]
            # If selected file is empty / without filename
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if not (file and allowed_file(file)):
                flash('Invalid file type. Select a CSV to upload', 'error')
                return redirect(request.url)
            # if not check_headers(file, upload_type):
            #     flash('Invalid CSV headers. Check CSV for required columns.', 'error')
            #     return redirect(request.url)
            # if it passes all the checks, save file and queue for processing

            # TODO: Get email address!
            # TODO: process file!
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = queue_job(file_path, upload_type, 'file_test@example.com')
            flash('Queued file {} for processing as {}. Check your email {} for results.'.format(filename, upload_type), 'info')
            return redirect(request.url)

            # return redirect(url_for('uploaded_file',
            #                         filename=filename))

    # GET
    return render_template('upload_form.html')


# returning uploaded files; get rid of this in prod
@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    # TODO: documentation on setting file directory permissions
    # i.e. uploads dir can't be inspected


