import logging
import unittest

try:
    import connexion
    from flask_testing import TestCase
    from tee_gateway.encoder import JSONEncoder

    class BaseTestCase(TestCase):
        def create_app(self):
            logging.getLogger("connexion.operation").setLevel("ERROR")
            app = connexion.App(__name__, specification_dir="../openapi/")
            app.app.json_encoder = JSONEncoder
            app.add_api("openapi.yaml", pythonic_params=True)
            return app.app

except ImportError:
    # flask_testing is an optional integration-test dependency.
    # When it is absent (e.g. in CI using only the `test` dep group),
    # expose a plain unittest.TestCase stub so that files which import
    # BaseTestCase can still be collected without error.  Any test methods
    # that actually need a live Flask app are already marked @unittest.skip.
    class BaseTestCase(unittest.TestCase):  # type: ignore[no-redef]
        pass
