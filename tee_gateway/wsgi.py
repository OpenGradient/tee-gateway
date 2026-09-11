"""WSGI entry point for a production server.

``tee_gateway/__main__.py`` builds the Flask app at import time (``application``)
and, when run as a script, serves it on Werkzeug's development server. Inside
the enclave the app is served by gunicorn instead (see ``gunicorn_conf.py`` and
``scripts/start.sh``); gunicorn imports the app from here so it never has to
import a module named ``__main__``:

    gunicorn -c python:tee_gateway.gunicorn_conf tee_gateway.wsgi:application
"""

from tee_gateway.__main__ import application

__all__ = ["application"]
