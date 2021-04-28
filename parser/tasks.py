""" 
    TODO:
    Production implementation should process jobs and 
    send emails with a queue and workers.
"""
from __init__ import celery
from flask import Blueprint, current_app, request
from flask_mail import Mail, Message
from main import UPLOAD_TYPES, unique_filename
import config


bp = Blueprint('tasks', __name__, url_prefix='/tasks')

@celery.task
def process_job_async(data):
    # It seems silly to run these via the command line
    # but we can easily change this later. Trying to avoid
    # messing with scripts unless we absolutely have to.
    input_file = data['input_file']
    upload_type = data['upload_type']
    email = data['email']
    scripts_folder = config.SCRIPTS_FOLDER
    scripts_home_dir = os.path.dirname(scripts_folder)
    output_file = unique_filename()
    second_output_file = None
    third_output_file = None
    if job_type in ['vec_file','sccne_file']:
        second_output_file = unique_filename()
        if job_type == 'sccne_file':
            third_output_file = unique_filename()

    if upload_type == 'tblc_file':
        cmd = 'python {}/name_cleaning.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif upload_type == 'vec_file':
        cmd = 'python {}/van_export_cleaning.py -i {} -f {} -o {} -m {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file, second_output_file)
    elif upload_type == 'tblctmc_file':
        cmd = 'python {}/name_cleaning_with_responses.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif upload_type == 'sccne_file':
        cmd = 'python {}/annotate_conversations.py -i {} -f {} -o {} -n {} -m {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file,
            second_output_file, third_output_file)
    elif upload_type == 'smsagg_file':
        cmd = 'python {}/aggregate_text_messages.py -d {} -o {}'.format(
            scripts_folder, input_file, output_file)
    else:
        print("Unknown job type {}".format(upload_type))
    print("cmd", cmd)
    job_run = subprocess.run(cmd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=True)
    if job_run.returncode: # exit code of 0 is success, 1 is generic error
        status = 'error'
        err_log = job_run.stdout.decode()
        print("ERROR: Could not process job {} file {} type {}".format(job_id, input_file, upload_type))
        print(err_log)
    else:
        print("SUCCESS")
    result_file = None
    if status == 'success':
        result_file = '{}/{}'.format(current_app.config['RESULTS_FOLDER'], output_file)
        if second_output_file:
            result_file == '{0}/{1}|{0}/{2}'.format(
                current_app.config['RESULTS_FOLDER'], output_file, second_output_file)
    if status == 'success':
        data = {
            'result_file': result_file,
            'email': email or config.TEST_TARGET_EMAIL,
            'upload_type': upload_type
        }
        res = email_results.apply_async(args=[data], countdown=0)
        # also delete the input file
        return True, output_file, second_output_file
    return False, err_log, None

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
        output = data['result_file'].split('|')
        results = ''.join(['<p>{}results/{}</p>'.format(config.BASE_URL, output_file) for output_file in output])
        msg.html = (
            "<p>Thank you for using the votetripling.org SMS transcript processing tool.</p>"
            "<p>The {} script successfully processed your data file.</p>"
            "<p>Link(s) to download the results:</p>"
            "{}"
            "<p>Your result files will be available for download for {} hours.</p>"
        ).format(UPLOAD_TYPES[data['upload_type']]['name'], results, config.FILE_LIFE)
        mail.send(msg)
    return True


# for testing only
@bp.route('/email-results/', methods=['GET'])
def email():
    if request.method == 'GET':
        data = {
            'job_id': 1234,
            'upload_type': 'vec_file',
            'result_file': '123.csv|456.csv',
            'email': current_app.config['TEST_TARGET_EMAIL']
        }
        res = email_results.apply_async(args=[data], countdown=0)
        if res.failed():
            print('emailing results failed')
        msg = ('Email queued for send! Check your inbox (and spam) for an email '
            'from "{}" with the subject "{}".').format(
                current_app.config['EMAIL_SENDER'], current_app.config['EMAIL_SUBJECT'])
        return msg
    return 'not allowed here'

# to manually kick off processing - note that this doesn't take an email
@bp.route('/process/<upload_type>/<file_name>', methods=['GET'])
def process():
    if request.method == 'GET':
        data = {
            'input_file': '{}/file_name'.format(config.UPLOAD_FOLDER),
            'upload_type': upload_type,
            'email': None
        }
        res = process_job_async.apply_async(args=[data], countdown=0)
        msg = ('File {} queued for processing as {}. Results will be emailed to {}.').format(
            file_name, upload_type, config.TEST_TARGET_EMAIL)
        return msg
    return 'not allowed here'

def cleanup_files(filename):
    # TODO
    # delete files from finished jobs that are more than n hours old
    pass
