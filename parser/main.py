import functools
import csv
import os
import settings
import sqlite3
import subprocess

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
    query = 'INSERT INTO jobs (input_file, job_type, email, status) VALUES (?, ?, ?, ?);'
    db.query_db(query, (file_path, job_type, email, 'queued'), write=True)
    return True


def process_job():
    # if there are any queued jobs, process the oldest one
    query = 'SELECT id, input_file, job_type FROM jobs WHERE status = ? ORDER BY created_at ASC;'
    jobs = db.query_db(query, ('queued'), one=True)
    if not jobs:
        print("No jobs to process")
        return False
    job_id = job['id']
    input_file = job['input_file']
    job_type = job['job_type']
    print("Processing job {} file {} type {}".format(job_id, input_file, job_type))
    if job_type == 'tblc_file':
        cmd = 'python ../Projects/NLP/SMS_Annotation/Code/name_cleaning.py'
        args = '-i {}'.format(input_file)
    elif job_type == 'tblc_tmc_file':
        cmd = 'python ../Projects/NLP/SMS_Annotation/Code/name_cleaning_with_responses.py'
        args = '-d {}'.format(input_file)
    elif job_type == 'sccne_file':
        cmd = 'python ../Projects/NLP/SMS_Annotation/Code/annotate_conversations.py'
        args = '-d {}'.(input_file)
    # elif job_type == 'sms_agg_file':
        # this script is to be ported to Python
    job_run = subprocess.run([cmd, args])
    # TODO: capture any errors and return whether job was successful
    # TODO: re-name the results file to a random string
    # TODO: Update db entry with results file path and status
    status = 'success'

    return status


def email_results():
    pass


def cleanup_files(filename):
    # delete files that are more than 48 hours old
    pass


def cleanup_jobs(interval):
    # delete jobs that are more than `interval` old, to keep the db from getting too big
    pass

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
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            email_name = '{}_email'.format(upload_type.split('_')[0])
            email = request.form[email_name]
            result = queue_job(file_path, upload_type, email)
            msg = ('Queued file {} for processing as {}. Check your email {} '
                   'in a few minutes for results.').format(
                       filename,
                       UPLOAD_TYPES[upload_type]['name'],
                       email)
            flash(queue_message, 'info')

            return redirect(request.url)
    # GET
    return render_template('upload_form.html')


# returning results files
@bp.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['RESULTS_FOLDER'],
                               filename)
    # TODO: documentation on setting file directory permissions
    # i.e. results dir can't be inspected


