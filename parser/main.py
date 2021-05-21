import csv
import datetime
import functools
import os
import random
import string
import subprocess

import config

from __init__ import celery
from flask import (Blueprint, current_app, flash, Flask, g, Markup,
    redirect, render_template, request, send_from_directory, url_for)
from flask_mail import Mail, Message

from werkzeug.utils import secure_filename

bp = Blueprint('main', __name__, url_prefix='')

ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_TYPES = {
    'tblc_file': { 
        'name': 'Text Banker Log Cleaning',
        'required_headers': ['names']
    },
    'tblctmc_file': {
        'name': 'Text Banker Log Cleaning (utilizing text message conversation)',
        'required_headers': ['names', 'triplemessage', 'voterresponse',
            'voterfinal', 'voterpost'
        ]
    },
    'sccne_file': {
        'name': 'SMS Conversation Categorization and Name Extraction',
        'required_headers': ['noresponse', 'negresponse', 'posresponse',
            'affirmresponse', 'finalaffirmresponse', 'triplemessage',
            'voterresponse','voterfinal', 'voterpost', 'conversationid',
            'contact_phone'
        ]
    },
    'smsagg_file': {
        'name': 'SMS Aggregation',
        'required_headers': ['MessageId', 'ConversationId', 'MessageDirection', 'MessageBody', 'ContactPhone']
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


def check_headers(file_path, upload_type):
    # open file as csv and read the first line
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print("headers", headers) # for debugging
        for header in UPLOAD_TYPES[upload_type]['required_headers']:
            if header not in headers:
                f.close()
                os.remove(file_path)
                return False
    return True


@celery.task(time_limit=1800)
def process_job(data):
    # It seems silly to run these via the command line
    # but we can easily change that later.
    scripts_folder = config.SCRIPTS_FOLDER
    scripts_home_dir = os.path.dirname(scripts_folder)
    input_file = data['input_file']
    job_type = data['upload_type']
    output_file = unique_filename()
    second_output_file = None
    third_output_file = None
    if job_type in ['vec_file','sccne_file']:
        second_output_file = unique_filename()
        if job_type == 'sccne_file':
            third_output_file = unique_filename()

    cmd = None
    if job_type == 'tblc_file':
        cmd = 'python {}/name_cleaning.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif job_type == 'vec_file':
        cmd = 'python {}/van_export_cleaning.py -i {} -f {} -o {} -m {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file,
            second_output_file)
    elif job_type == 'tblctmc_file':
        cmd = 'python {}/name_cleaning_with_responses.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif job_type == 'sccne_file':
        cmd = 'python {}/annotate_conversations.py -i {} -f {} -o {} -n {} -m {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file, second_output_file,
            third_output_file)
    elif job_type == 'smsagg_file':
        cmd = 'python {}/aggregate_text_messages.py -d {} -o {}/{} -a "{}" -af "{}" -t "{}" -in "{}" -out "{}"'.format(
            scripts_folder, input_file, config.RESULTS_FOLDER,
            output_file, data['aff_regex'], data['aff_regex_final'],
            data['init_triple_phrase'], data['inbound_text'], data['outbound_text'])
    else:
        err_log = "Unknown job type {}".format(job_type)
        return False, err_log, None, None

    status = 'success'
    print("cmd", cmd)
    job_run = subprocess.run(cmd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=True)
    if job_run.returncode: # exit code of 0 is success, 1 is generic error
        status = 'error'
        err_log = job_run.stdout.decode()
        print("ERROR: Could not process file {} as type {}".format(
            input_file, job_type))
        print(err_log)
    else:
        print("SUCCESS")

    if status == 'success':
        if config.PROCESS_ASYNC:
            result_files = list(filter(
                None, [output_file, second_output_file, third_output_file]))
            result_file = '|'.join([file for file in result_files])
            email_data = {
                'success': True,
                'result_file': result_file,
                'email': data['email'] or config.TEST_TARGET_EMAIL,
                'job_type': job_type,
                'original_file': data['original_file'],
            }
            res = email_results.apply_async(args=[email_data], countdown=0)
            cleanup = cleanup_files.apply_async(args=[data], countdown=10)
        # TODO: delete input file. If processing wasn't successful, leave the input file
        # for troubleshooting
        return True, output_file, second_output_file, third_output_file
    else: # failed to process
        if config.PROCESS_ASYNC:
            email_data = {
                'success': False,
                'email': data['email'] or config.TEST_TARGET_EMAIL,
                'job_type': job_type,
                'original_file': data['original_file'],
                'error_log': err_log
            }
            res = email_results.apply_async(args=[email_data], countdown=0)
            cleanup = cleanup_files.apply_async(args=[data], countdown=10)
        return False, err_log, None, None


@celery.task
def email_results(data):
    with current_app.app_context():
        # for some reason celery spins up a new instance without access to
        # original config file, so we re-config here
        current_app.config.from_object(config)
        mail = Mail(current_app)
        msg = Message(config.EMAIL_SUBJECT,
                sender=config.EMAIL_SENDER,
                recipients=[data['email']])
        # accommodating more than one output file
        if data['success']:
            output = data['result_file'].split('|')
            results = ''.join(['<p>{}results/{}</p>'.format(config.BASE_URL, output_file) for output_file in output])
            msg.html = (
                "<p>Thank you for using the votetripling.org SMS transcript processing tool.</p>"
                "<p>The {} script successfully processed your data file {}.</p>"
                "<p>Link(s) to download the results:</p>"
                "{}"
                "<p>Your result files will be available for download for {} hours.</p>"
            ).format(UPLOAD_TYPES[data['job_type']]['name'], data['original_file'], 
                results, config.FILE_LIFE)
        else: # send along the error log
            msg.html = (
                "<p>Thank you for using the votetripling.org SMS transcript processing tool.</p>"
                "<p>The {} script failed to process your data file {}.</p>"
                "<p>It returned the following error:</p>"
                "<pre>{}</pre>"
                "<p>Please correct the issues and try again.</p>"
                ).format(UPLOAD_TYPES[data['job_type']]['name'], data['original_file'],
                    data['error_log'])
        mail.send(msg)
    return True

@celery.task
def cleanup_files(data=None):
    # delete input file
    print('FILE CLEANUP')
    if data:
        input_file = data['input_file']
        if os.path.exists(input_file):
            print('deleting input file {}'.format(input_file))
            os.remove(input_file)
        else:
            print('{} does not exist'.format(input_file))
    # delete output files that are more than config.FILE_LIFE hours old
    print('deleting results files')
    output_folder = config.RESULTS_FOLDER
    files = os.listdir(config.RESULTS_FOLDER)
    for file in files:
        try:
            date_time_obj = datetime.datetime.strptime(
                file.split('-')[1].replace('.csv', ''), '%Y%m%d%H%M%S')
        except:
            continue
        if date_time_obj + datetime.timedelta(hours=config.FILE_LIFE) < datetime.datetime.now():
            print(file)
            os.remove('{}/{}'.format(config.RESULTS_FOLDER, file))
    return True

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

            filename = unique_filename()
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            if not check_headers(file_path, upload_type):
                msg = 'Invalid CSV headers. Check CSV for required columns {}.'.format(
                    UPLOAD_TYPES[upload_type]['required_headers'])
                flash(msg, 'error')
                return redirect(request.url)

            email_name = '{}_email'.format(upload_type.split('_')[0])
            email = request.form[email_name]
            aff_regex = None
            aff_regex_final = None
            init_triple_phrase = None
            inbound_text = None
            outbound_text = None
            if upload_type == 'smsagg_file':
                aff_regex = request.form['aff_regex']
                aff_regex_final = request.form['aff_regex_final']
                init_triple_phrase = request.form['init_triple_phrase']
                inbound_text = request.form['inbound_text']
                outbound_text = request.form['outbound_text']

            outcome_msg = ('Queued file {} for processing as {}. Check your email '
                   'in a few minutes for results.').format(
                       file.filename, UPLOAD_TYPES[upload_type]['name'])
            data = {
                'original_file': file.filename,
                'input_file': file_path,
                'upload_type': upload_type,
                'email': email,
                'aff_regex': aff_regex,
                'aff_regex_final': aff_regex_final,
                'init_triple_phrase': init_triple_phrase, 
                'inbound_text': inbound_text,
                'outbound_text': outbound_text
            }
            
            if current_app.config['PROCESS_ASYNC']:
                res = process_job.apply_async(args=[data], countdown=0)
            else:
                # make the user wait for processing
                print("Processing file {} type {}".format(file_path, upload_type))
                success, output, second_output, third_output = process_job(data)
                if success:
                    output_files = list(filter(None, [output, second_output, third_output]))
                    result_links = ['<a href="/results/{}" style="padding: 0 1em;">File {}</a>'.format(
                        x, output_files.index(x) + 1) for x in output_files]
                    outcome_msg = Markup(
                        'Download results:  {}'.format('  '.join(result_links)))
                else:
                    outcome_msg = 'Error processing file {}'.format(
                        output) if current_app.config['SHOW_SCRIPT_ERRORS'] else 'Error processing file'

            flash(outcome_msg, 'info')
            return redirect(request.url)
    # GET
    return render_template('upload_form.html')


@bp.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(current_app.config['RESULTS_FOLDER'],
                               filename)


# test emailing
@bp.route('/email-results/', methods=['GET'])
def email():
    if request.method == 'GET':
        data = {
            'job_id': 1234,
            'upload_type': 'vec_file',
            'result_file': '123.csv|456.csv',
            'email': current_app.config['TEST_TARGET_EMAIL'],
            'original_file': 'EMAIL TEST'
        }
        res = email_results.apply_async(args=[data], countdown=0)
        if res.failed():
            print('emailing results failed')
        msg = ('Email queued for send! Check your inbox (and spam) for an email '
            'from "{}" with the subject "{}".').format(
                current_app.config['EMAIL_SENDER'], current_app.config['EMAIL_SUBJECT'])
        return msg
    return 'not allowed here'


# manually kick off processing
@bp.route('/process/<upload_type>/<file_name>', methods=['GET'])
def process():
    if request.method == 'GET':
        data = {
            'input_file': '{}/{}'.format(current_app.config['UPLOAD_FOLDER'], file_name),
            'upload_type': upload_type,
            'email': None
        }
        res = process_job_async.apply_async(args=[data], countdown=0)
        msg = ('File {} queued for processing as {}. Results will be emailed to {}.').format(
            file_name, upload_type, current_app.config['TEST_TARGET_EMAIL'])
        return msg
    return 'not allowed here'


# manually kick off file cleanup
@bp.route('/cleanup/', methods=['GET'])
def cleanup():
    if request.method == 'GET':
        res = cleanup_files.apply_async(countdown=0)
        return ('deleting result files older than {} hours'.format(config.FILE_LIFE))
    return 'not allowed here'