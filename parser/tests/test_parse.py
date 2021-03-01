import os
import tempfile

import pytest

from . import parser

@pytest.fixture
def client():
    db_fd, parser.app.config['DATABASE'] = tempfile.mkstemp()
    parser.app.config['TESTING'] = True

    with parser.app.test_client() as client:
        with parser.app.app_context():
            parser.init_db()
        yield client

    os.close(db_fd)
    os.unlink(parser.app.config['DATABASE'])