import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'instance'))

from celery import Celery
from flask import Flask
import config

celery = Celery(__name__, broker=config.CELERY_BROKER_URL, result_backend=config.CELERY_RESULT_BACKEND)

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.instance_path = (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'instance')) # maybe setting this manually fixes the config problem in blueprint celery tasks
    app.config.from_mapping(
            SECRET_KEY='dev',
            DATABASE=os.path.join(app.instance_path, 'parser.sqlite'),
        )
    test_config = None
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    import db
    db.init_app(app)

    import main
    app.register_blueprint(main.bp)

    import tasks
    app.register_blueprint(tasks.bp)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # redis_client.init_app(app)
    celery.conf.update(app.config)

    return app