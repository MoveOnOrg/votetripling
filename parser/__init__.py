import os
import settings

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    app.config.from_mapping(
        SECRET_KEY=settings.SECRET_KEY,
        DATABASE=os.path.join(app.instance_path, 'parser.sqlite'),
        MAX_CONTENT_LENGTH=settings.MAX_CONTENT_LENGTH,
        UPLOAD_FOLDER=os.path.join(os.path.dirname(APP_ROOT), settings.UPLOAD_FOLDER)
    )

    from . import db
    db.init_app(app)

    from . import main
    app.register_blueprint(main.bp)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    return app