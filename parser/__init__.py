import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings

from celery import Celery
from flask import Flask
from flask_mail import Mail

# redis_client = FlaskRedis(app)

celery = Celery(__name__, broker=settings.CELERY_BROKER_URL, result_backend=settings.CELERY_RESULT_BACKEND)
mail = Mail()

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    app.config.from_mapping(
        SECRET_KEY=settings.SECRET_KEY,
        DATABASE=os.path.join(app.instance_path, 'parser.sqlite'),
        MAX_CONTENT_LENGTH=settings.MAX_CONTENT_LENGTH,
        UPLOAD_FOLDER=os.path.join(os.path.dirname(APP_ROOT), 'Projects/NLP/SMS_Annotation/Input_Data'),
        RESULTS_FOLDER=os.path.join(os.path.dirname(APP_ROOT), 'Projects/NLP/SMS_Annotation/Output_Data'),
        SCRIPTS_FOLDER=os.path.join(os.path.dirname(APP_ROOT), 'Projects/NLP/SMS_Annotation/Code'),
        CELERY_BROKER_URL=settings.CELERY_BROKER_URL,
        CELERY_RESULT_BACKEND=settings.CELERY_RESULT_BACKEND,
        MAIL_SERVER = settings.MAIL_SERVER,
        MAIL_PORT = settings.MAIL_PORT,
        MAIL_USE_TLS = True,
        MAIL_USERNAME = settings.MAIL_USERNAME,
        MAIL_PASSWORD = settings.MAIL_PASSWORD,
        EMAIL_SENDER = settings.EMAIL_SENDER,
        EMAIL_SUBJECT = settings.EMAIL_SUBJECT,
        BASE_URL = settings.BASE_URL
    )

    # redis_client.init_app(app)
    import db
    db.init_app(app)

    import main
    app.register_blueprint(main.bp)

    import tasks
    app.register_blueprint(tasks.bp)

    # if test_config is None:
    #     # load the instance config, if it exists, when not testing
    #     app.config.from_pyfile('config.py', silent=True)
    # else:
    #     # load the test config if passed in
    #     app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    celery.conf.update(app.config)
    mail.init_app(app)

    return app