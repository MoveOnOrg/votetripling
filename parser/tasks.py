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
import redis


bp = Blueprint('tasks', __name__, url_prefix='/tasks')

@celery.task
def run_script(input_file, job_type, email):

    # It seems silly to run these via the command line
    # but we can easily change this later. Trying to avoid
    # messing with scripts unless we absolutely have to.
    scripts_folder = config.SCRIPTS_FOLDER
    scripts_home_dir = os.path.dirname(scripts_folder)
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
    elif job_type == 'tblc_tmc_file':
        cmd = 'python {}/name_cleaning_with_responses.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
    elif job_type == 'sccne_file':
        cmd = 'python {}/annotate_conversations.py -i {} -f {} -o {}'.format(
            scripts_folder, input_file, scripts_home_dir, output_file)
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
    pass


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
            "<p>Your result files will be available for download for 48 hours.</p>"
        ).format(UPLOAD_TYPES[data['upload_type']]['name'], results)
        mail.send(msg)


@bp.route('/email-results/', methods=['GET'])
def index():
    if request.method == 'GET':
        # TODO: get real job data
        data = {
            'job_id': 1234,
            'upload_type': 'vec_file',
            'result_file': '123.csv|456.csv',
            'email': current_app.config['TEST_TARGET_EMAIL']
        }
        res = email_results.apply_async(args=[data], countdown=0)
        msg = ('Email queued for send! Check your inbox (and spam) for an email '
            'from "{}" with the subject "{}".').format(
                current_app.config['EMAIL_SENDER'], current_app.config['EMAIL_SUBJECT'])
        return msg
    return 'not allowed here'

@bp.route('/script/', methods=['GET'])
def index():
    if request.method == 'GET':
        # TODO: get real job data
        msg = ('Script queued for processing')
        return msg
    return 'not allowed here'

def cleanup_files(filename):
    # TODO
    # delete files from finished jobs that are more than 48 hours old
    pass


def cleanup_jobs(interval):
    # TODO
    # delete jobs that are more than `interval` old, to keep the db from getting too big
    pass