import csv
import datetime
import functools
import os
import random
import settings
import sqlite3
import string
import subprocess

from flask import (Blueprint, current_app, flash, Flask, g, Markup,
    redirect, render_template, request, send_from_directory, url_for)
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
    },
    'vec_file': {
        'name': 'VAN Export Cleaning',
        'required_headers': ['voter_file_vanid','ContactName','NoteText']
    }
}


def allowed_file(file):
    return ('.' in file.filename 
        and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        and file.content_type == 'text/csv')

def unique_filename():
    return secure_filename(
        '{}-{}.csv'.format(
            ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(17)),
            datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        )
    )

def check_headers(file, upload_type):
    headers = file.readline().decode('utf-8').rstrip().split(',')
    for header in UPLOAD_TYPES[upload_type]['required_headers']:
        if header not in headers:
            return False
    return True


def queue_job(file_path, job_type, email):
    query = 'INSERT INTO jobs (input_file, job_type, email, status) VALUES (?, ?, ?, ?);'
    db.query_db(query, (file_path, job_type, email, 'queued'), write=True)
    return True


def process_job():
    """
        TODO: Replace with a celery queue and workers.
    """

    # if there are any queued jobs, process the oldest one
    query = 'SELECT id, input_file, job_type FROM jobs WHERE status = ? ORDER BY created_at ASC;'
    job = db.query_db(query, ('queued',), one=True)
    if not job:
        print("No jobs to process")
        return False
    job_id = job['id']
    input_file = job['input_file']
    job_type = job['job_type']
    print("Processing job {} file {} type {}".format(job_id, input_file, job_type))
    
    status = 'success'
    processing_query = 'UPDATE jobs SET status = ? WHERE id = ?;'
    update = db.query_db(processing_query, ('processing', job_id), write=True)

    # It seems silly to run these via the command line
    # but we can easily change that later.
    scripts_folder = current_app.config['SCRIPTS_FOLDER']
    scripts_home_dir = os.path.dirname(scripts_folder)
    print('shd', scripts_home_dir)
    output_file = unique_filename()
    second_output_file = None
    if job_type == 'vec_file':
        second_output_file = unique_filename()

    if job_type == 'tblc_file':
        cmd = 'python {}/name_cleaning.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif job_type == 'vec_file':
        cmd = 'python {}/van_export_cleaning.py -i {} -f {} -o {} -m {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file, second_output_file)
    # elif job_type == 'tblc_tmc_file':
    #     cmd = 'python {}/name_cleaning_with_responses.py'.format(scripts_folder)
    #     args = ['-d {}'.format(input_file)]
    # elif job_type == 'sccne_file':
    #     cmd = 'python {}/annotate_conversations.py'.format(scripts_folder)
    #     args = ['-d {}'.format(input_file)]
    # elif job_type == 'sms_agg_file':
        # this script is to be ported to Python
    else:
        print("Unknown job type {}".format(job_type))
    print("cmd", cmd)
    job_run = subprocess.run(cmd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=True)
    if job_run.returncode: # exit code of 0 is success, 1 is generic error
        status = 'error'
        err_log = job_run.stdout.decode()
        print("ERROR: Could not process job {} file {} type {}".format(job_id, input_file, job_type))
        print(err_log)
    else:
        print("SUCCESS")
    result_file = None
    if status == 'success':
        result_file = '{}/{}'.format(current_app.config['RESULTS_FOLDER'], output_file)
        if second_output_file:
            result_file == '{0}/{1}|{0}/{2}'.format(
                current_app.config['RESULTS_FOLDER'], output_file, second_output_file)
    done_query = 'UPDATE jobs SET status = ?, result_file = ? WHERE id = ?;'
    update = db.query_db(done_query, (status, result_file, job_id), write=True)
    if status == 'success':
        return True, output_file, second_output_file
    return False, err_log, None


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
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if not (file and allowed_file(file)):
                flash('Invalid file type. Select a CSV to upload', 'error')
                return redirect(request.url)
            if not check_headers(file, upload_type):
                msg = 'Invalid CSV headers. Check CSV for required columns {}.'.format(
                    UPLOAD_TYPES[upload_type]['required_headers'])
                flash(msg, 'error')
                return redirect(request.url)

            # if it passes all the checks, save file and queue for processing
            filename = unique_filename()
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.seek(0) # return pointer to beginning of file
            file.save(file_path)
            email_name = '{}_email'.format(upload_type.split('_')[0])
            email = request.form[email_name]
            result = queue_job(file_path, upload_type, email)

            # This message is currently a lie, we're not really queueing jobs now
            # except for very briefly.
            # queue_msg = ('Queued file {} for processing as {}. Check your email {} '
            #        'in a few minutes for results.').format(
            #            filename,
            #            UPLOAD_TYPES[upload_type]['name'],
            #            email)
            # queue_msg = ('Processing file {} as {}'.format(
            #                 filename, UPLOAD_TYPES[upload_type]['name']))
            # flash(queue_msg, 'info')
            success, output, second_output = process_job()
            if success:
                outcome_msg = Markup(
                    '<a href="/results/{}">Download results</a>'.format(output))
                if second_output:
                    outcome_msg = Markup(
                        'Download <a href="/results/{}">result file 1</a>'
                        ' and <a href="/results/{}">result file 2</a>').format(
                            output, second_output)
            else:
                outcome_msg = 'Error processing file {}'.format(
                    output) if os.environ['FLASK_ENV'] == 'development' else 'Error processing file'
            flash(outcome_msg, 'info')
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


