import os
import tempfile

import pytest

import parser

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


# TODO

# test posting wrong file extension to /
# test posting files with bad headers to /
# test posting right file with right headers to /
